# ==============================================================================
# Yojana Mitra — Web Chat Surface (FastAPI + SSE)
# File Path: product_inference/web/server.py
# ==============================================================================
"""
The browser surface. A renderer over `core_inference.session`, and nothing more.

Run with:
    python -m product_inference.web.server

**Single worker only.** `_active_tasks` is an in-process dict; a second uvicorn worker
would silently break `/stop` because the cancelling request could land on the worker that
does not hold the task.

Security posture, in order of how badly each would hurt:

1. `user_id` is resolved *only* from the signed session cookie. No route accepts it from
   a query string, body, or header. It is the partition key for the PII vault.
2. Screenshots are served through two independent checks — the requested name must begin
   with the caller's own `auto_<user_id>_` prefix, and the resolved real path must sit
   inside `screenshots/`. Portal screenshots show filled-in application forms.
3. Uploads go through `session.ingest_document`, which runs the same
   `document_handler` bouncer (magic bytes, 10MB cap, EXIF strip, Pillow re-encode) the
   bots use. There is no direct path from an upload to `document_extractor`.
4. The confirm button is not a privileged route. It posts the literal text `CONFIRM`
   through the ordinary chat endpoint. The zero-auto-submit guarantee stays where it
   lives — the planner prompt and the graph's interrupt/resume gate.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import Cookie, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse,
)

import product_inference.db as db
from core_inference import profile_form, session as core_session
from core_inference.events import Event
from product_inference.web import auth

STATIC_DIR = Path(__file__).parent / "static"
SCREENSHOT_ROOT = Path(core_session.SCREENSHOT_DIR).resolve()

# Read cap for uploads. The bouncer enforces the real 10MB policy; this exists so a
# multi-gigabyte POST cannot exhaust memory before the bouncer ever sees it.
_MAX_UPLOAD_BYTES = 12 * 1024 * 1024

app = FastAPI(title="Yojana Mitra — Web Chat", docs_url=None, redoc_url=None)

# user_id -> in-flight turn, for /stop
_active_tasks: dict[str, asyncio.Task] = {}


# ==============================================================================
# 1. IDENTITY
# ==============================================================================

def _require_user(ym_session: Optional[str]) -> str:
    """Resolve the caller's user_id from the session cookie, or 401.

    This is the only function in this module that produces a user_id.
    """
    user_id = auth.verify_token(ym_session)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated. Send /web to the Telegram bot to get a link.")
    return user_id


@app.get("/auth")
async def auth_handoff(t: str = ""):
    """Consume a Telegram handoff token and issue a session cookie."""
    user_id = auth.verify_token(t)
    if not user_id:
        return HTMLResponse(
            "<h1>Link expired</h1><p>Send <code>/web</code> to the Yojana Mitra "
            "Telegram bot again to get a fresh link.</p>",
            status_code=401,
        )

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        key=auth.COOKIE_NAME,
        value=auth.mint_token(user_id, ttl_seconds=auth.SESSION_TTL_SECONDS),
        max_age=auth.SESSION_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=os.getenv("WEB_BASE_URL", "").startswith("https://"),
    )
    return response


@app.post("/api/logout")
async def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.COOKIE_NAME)
    return response


@app.get("/api/me")
async def me(ym_session: Optional[str] = Cookie(default=None)):
    user_id = _require_user(ym_session)
    return {"user_id": user_id, "platform": user_id.split("_")[0]}


# ==============================================================================
# 2. SERVER-SENT EVENTS
# ==============================================================================

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _public_event(event: Event) -> dict:
    """Convert an Event to the wire form the browser sees.

    `Image.path` is a server-side filesystem path and never crosses the wire. It is
    replaced with the basename, which the guarded screenshot route re-resolves against
    the caller's own identity.
    """
    payload = event.to_dict()
    if payload.get("kind") == "image":
        payload["path"] = os.path.basename(payload.get("path", ""))
    return payload


async def _event_stream(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield _sse(_public_event(event))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"[Web Stream Error] {type(exc).__name__}: {exc}")
        yield _sse({"kind": "error", "text": "⚠️ Something went wrong handling that turn.", "detail": ""})
    finally:
        yield _sse({"kind": "end"})


def _stream_response(events: AsyncIterator[Event]) -> StreamingResponse:
    return StreamingResponse(
        _event_stream(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # nginx would otherwise buffer the whole stream
            "Connection": "keep-alive",
        },
    )


# ==============================================================================
# 3. CHAT
# ==============================================================================

@app.post("/api/chat")
async def chat(request: Request, ym_session: Optional[str] = Cookie(default=None)):
    """Run one turn and stream its events.

    `CONFIRM` arrives here like any other text and is passed through untouched — the
    browser's confirm button is a convenience for typing, not a separate code path.
    """
    user_id = _require_user(ym_session)

    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="empty message")

    # Match bot behaviour: a new message supersedes an in-flight one.
    previous = _active_tasks.pop(user_id, None)
    if previous and not previous.done():
        previous.cancel()

    return _stream_response(core_session.run_turn(user_id, text))


@app.post("/api/stop")
async def stop(ym_session: Optional[str] = Cookie(default=None)):
    user_id = _require_user(ym_session)
    task = _active_tasks.pop(user_id, None)
    if task and not task.done():
        task.cancel()
        return {"stopped": True}
    return {"stopped": False}


# ==============================================================================
# 4. DOCUMENT UPLOAD
# ==============================================================================

@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    ym_session: Optional[str] = Cookie(default=None),
):
    """Accept a document and stream the extraction pipeline's events.

    The bytes are written to a temp file and handed to `session.ingest_document`, which
    runs the bouncer before anything else touches them — the same order the bots use.
    """
    user_id = _require_user(ym_session)

    filename = os.path.basename(file.filename or "upload")
    if not filename.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Only PDF, PNG and JPG files are accepted.")

    # Stream to disk with a hard read cap, so an oversized POST cannot be buffered whole.
    suffix = os.path.splitext(filename)[1]
    fd, temp_path = tempfile.mkstemp(prefix=f"web_{int(time.time())}_", suffix=suffix)
    total = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File is too large (10MB maximum).")
                handle.write(chunk)
    except HTTPException:
        os.unlink(temp_path)
        raise
    except Exception:
        os.unlink(temp_path)
        raise

    async def events() -> AsyncIterator[Event]:
        try:
            async for event in core_session.ingest_document(user_id, temp_path, filename, total):
                yield event
        finally:
            # The bouncer copies what it keeps; this temp copy is always ours to remove.
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    return _stream_response(events())


# ==============================================================================
# 5. SCREENSHOTS
# ==============================================================================

@app.get("/api/screenshot/{name}")
async def screenshot(name: str, ym_session: Optional[str] = Cookie(default=None)):
    """Serve an automation screenshot belonging to the calling citizen.

    Two independent checks, because these images show a government form filled with
    someone's PII:

      1. **Ownership.** Session ids are always `auto_<user_id>` (`graph.py:735`), so a
         legitimate screenshot for this caller always begins with `auto_<user_id>_`.
         Without this, an authenticated citizen could read another's screenshots by
         guessing a user id.
      2. **Containment.** The resolved real path must be inside `screenshots/`, which
         stops `../` and encoded traversal regardless of what the name looks like.
    """
    if not name or "/" in name or "\\" in name or name != os.path.basename(name):
        raise HTTPException(status_code=400, detail="bad screenshot name")

    user_id = _require_user(ym_session)

    if not name.startswith(f"auto_{user_id}_"):
        raise HTTPException(status_code=403, detail="not your screenshot")

    candidate = (SCREENSHOT_ROOT / name).resolve()
    if not str(candidate).startswith(str(SCREENSHOT_ROOT) + os.sep):
        raise HTTPException(status_code=400, detail="bad screenshot path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="screenshot not found")

    return FileResponse(candidate, media_type="image/png")


# ==============================================================================
# 6. PROFILE
# ==============================================================================

@app.get("/api/profile")
async def get_profile(ym_session: Optional[str] = Cookie(default=None)):
    """The form definition plus whatever the citizen has already saved."""
    user_id = _require_user(ym_session)

    try:
        record = await asyncio.to_thread(db.get_or_create_user, user_id, "WebUser")
        saved = (record or {}).get("profile_data") or {}
        state = (record or {}).get("current_state", "START")
    except Exception as exc:  # noqa: BLE001
        print(f"[Web Profile Error] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Profile store is unavailable.")

    return {
        "schema": profile_form.as_json_schema(),
        "profile": saved,
        "state": state,
        "complete": not profile_form.validate_profile(saved),
    }


@app.post("/api/profile")
async def save_profile(request: Request, ym_session: Optional[str] = Cookie(default=None)):
    """Coerce and save a submitted profile.

    Coercion goes through `profile_form.apply_field`, the same function the Telegram
    keyboard uses, so a profile saved from the browser is byte-identical in shape to one
    saved from a chat form — including the derived `disability_percentage`.
    """
    user_id = _require_user(ym_session)
    body = await request.json()
    submitted = body.get("profile")
    if not isinstance(submitted, dict):
        raise HTTPException(status_code=400, detail="profile must be an object")

    data = profile_form.defaults()
    errors: dict[str, str] = {}

    for field in profile_form.FORM_FIELDS:
        if field not in submitted:
            continue
        ok, message = profile_form.apply_field(data, field, submitted[field])
        if not ok:
            errors[field] = message

    if errors:
        return JSONResponse({"ok": False, "errors": errors}, status_code=400)

    problems = profile_form.validate_profile(data)
    if problems:
        return JSONResponse({"ok": False, "problems": problems}, status_code=400)

    try:
        await asyncio.to_thread(db.update_user_state, user_id, "PROFILE_COMPLETE", data)
    except Exception as exc:  # noqa: BLE001
        print(f"[Web Profile Save Error] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Could not save profile.")

    return {"ok": True, "summary": profile_form.format_profile_summary(data, bold="**")}


@app.post("/api/reset")
async def reset(ym_session: Optional[str] = Cookie(default=None)):
    user_id = _require_user(ym_session)
    try:
        await asyncio.to_thread(core_session.do_reset, user_id)
    except Exception as exc:  # noqa: BLE001
        print(f"[Web Reset Error] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=503, detail="Reset failed.")
    return {"ok": True}


# ==============================================================================
# 7. STATIC
# ==============================================================================

@app.get("/")
async def index(ym_session: Optional[str] = Cookie(default=None)):
    if not auth.verify_token(ym_session):
        return FileResponse(STATIC_DIR / "login.html")
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/app.js")
async def app_js():
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css")
async def styles_css():
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


def main():
    import uvicorn

    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8000"))

    print(f"[Yojana Mitra Web] http://{host}:{port}")
    print("[Yojana Mitra Web] Send /web to the Telegram bot to get a sign-in link.")
    # workers=1 is not a default to rely on — it is required. See module docstring.
    uvicorn.run(app, host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
