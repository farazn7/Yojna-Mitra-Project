# ==============================================================================
# Yojana Mitra — Surface-Agnostic Turn Runner
# File Path: core_inference/session.py
# ==============================================================================
"""
One implementation of "handle a turn", rendered by every surface.

Before this module, a turn existed twice: `bot_telegram.py:292-397` and `bot.py:311-397`
held the same document pipeline near-verbatim, and the automation screenshot dispatch map
was duplicated at `bot.py:461-464` and `bot_telegram.py:489-493`. `bot_audio.py` had
neither — it has no document upload path and never sends a screenshot, because both
features were added to two of the three copies. That is the cost this module exists to
stop paying.

Everything here is an async generator of `core_inference.events`. Surfaces subscribe and
draw; they do not decide anything about ordering, validation, or vault writes.

Threading: the graph, the vision passes, and Playwright are all synchronous and slow, so
they run in worker threads and push events back through an `asyncio.Queue`. Nothing here
may be called from a thread that is also running Playwright's sync API.
"""

import asyncio
import json
import os
import shutil
import stat
import time
from typing import AsyncIterator, Optional

import product_inference.db as db
from core_inference.events import (
    AwaitConfirm, Cancelled, Error, Event, Image, Message, NodeEnter, Status,
)

# Screenshot filename conventions, keyed by the automation status that produced them.
# Previously duplicated verbatim in two bots and absent from the third.
_SHOT_BY_STATUS = {
    "awaiting_confirm": "{sid}_final_review.png",
    "complete":         "{sid}_submitted.png",
    "hitl_paused":      "{sid}_otp_intercept.png",
}

SCREENSHOT_DIR = "screenshots"

# Document types that are never worth a vault cache lookup.
_SKIP_CACHE_TYPES = {"back_of_card", "unknown"}


def _graph():
    """Import `graph_app` lazily.

    `core_inference.graph` opens a psycopg ConnectionPool at module scope. With Postgres
    down that import blocks for ~30s before its try/except swallows the timeout, which
    would make importing this module (and testing anything in it) unreasonably slow.
    """
    from core_inference.graph import graph_app
    return graph_app


def _pretty(doc_type: str) -> str:
    return doc_type.replace("_", " ").title()


# ==============================================================================
# 1. TEXT TURN
# ==============================================================================

async def _stream_graph(payload: dict, config: dict) -> AsyncIterator[tuple[str, object]]:
    """Run the sync graph stream on a worker thread, yielding `(mode, chunk)` as it goes.

    `stream_mode=["updates", "values"]` gives both: `updates` names the node that just
    ran (so a surface can show the state machine advancing), `values` carries the full
    state after it (so the last one is the final result).

    Cancellation note: cancelling the consumer stops delivery but does **not** kill the
    worker thread — an in-flight Ollama or Playwright call runs to completion. That is
    exactly how `/stop` behaves today via `asyncio.to_thread`; this preserves it rather
    than pretending otherwise.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def worker():
        try:
            for mode, chunk in _graph().stream(
                payload, config=config, stream_mode=["updates", "values"]
            ):
                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", (mode, chunk)))
        except BaseException as exc:  # noqa: BLE001 — re-raised on the consumer side
            loop.call_soon_threadsafe(queue.put_nowait, ("raise", exc))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    loop.run_in_executor(None, worker)

    while True:
        tag, item = await queue.get()
        if tag == "done":
            return
        if tag == "raise":
            raise item
        yield item


def _screenshot_for(state: dict, user_id: str) -> Optional[str]:
    """Path to the screenshot this automation status should surface, if it exists."""
    status = state.get("automation_status", "idle")
    template = _SHOT_BY_STATUS.get(status)
    if not template:
        return None

    session_id = state.get("automation_session_id") or f"auto_{user_id}"
    path = os.path.join(SCREENSHOT_DIR, template.format(sid=session_id))
    return path if os.path.exists(path) else None


async def run_turn(
    user_id: str,
    text: str,
    status_text: str = "🧠 *Yojana Mitra is processing...*",
) -> AsyncIterator[Event]:
    """Run one conversational turn, emitting events as they happen.

    `text` goes through unmodified — including a literal `CONFIRM`, which gets no special
    handling here on purpose. The human-in-the-loop gate lives in the graph's
    interrupt/resume round-trip and in the planner prompt; this layer must not become a
    third place that decides whether a submit happens.
    """
    yield Status(status_text)

    config = {"configurable": {"thread_id": user_id}}
    payload = {"messages": [{"role": "user", "content": text}], "user_id": user_id}

    final_state: dict = {}

    try:
        async for mode, chunk in _stream_graph(payload, config):
            if mode == "updates" and isinstance(chunk, dict):
                for node_name in chunk:
                    yield NodeEnter(node=node_name)
            elif mode == "values" and isinstance(chunk, dict):
                final_state = chunk

    except asyncio.CancelledError:
        yield Cancelled()
        raise
    except Exception as exc:  # noqa: BLE001
        yield Error(
            text="⚠️ Operational database exception encountered handling text generation pipelines.",
            detail=f"{type(exc).__name__}: {exc}",
        )
        return

    response = final_state.get("response") or ""
    if not response.strip():
        response = (
            "I processed your request, but the generated response was empty. "
            "Please try asking your question again!"
        )
    yield Message(text=response)

    shot = _screenshot_for(final_state, user_id)
    if shot:
        yield Image(path=shot, caption=final_state.get("automation_status", ""))

    if final_state.get("automation_status") == "awaiting_confirm":
        yield AwaitConfirm(
            prompt="Reply CONFIRM to submit, or tell me what to change.",
            session_id=final_state.get("automation_session_id", ""),
        )


# ==============================================================================
# 2. DOCUMENT INGESTION
# ==============================================================================

async def ingest_document(
    user_id: str,
    temp_path: str,
    filename: str,
    file_size: int,
) -> AsyncIterator[Event]:
    """Run an uploaded document through the full pipeline.

    The surface is responsible only for getting the bytes onto disk at `temp_path`;
    everything after that is identical on every platform.

    Order is load-bearing and matches the existing bots exactly:
      bouncer → (verify-if-expected | classify + cache check) → analyze → vault upsert
      → resume the graph if the document was one we asked for.

    The bouncer (`document_handler.download_and_process_file`) does the magic-byte check,
    the 10MB cap, EXIF stripping, and the Pillow decode/re-encode. Nothing may reach
    `document_extractor` without passing through it first.
    """
    yield Status("📄 *Document detected! Securing file...*")

    config = {"configurable": {"thread_id": user_id}}

    # Is the graph currently waiting for a specific document?
    try:
        graph_state = await asyncio.to_thread(_graph().get_state, config)
        expected_doc = (
            graph_state.values.get("awaiting_document", "")
            if graph_state and hasattr(graph_state, "values") else ""
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[Session State Check Error] Could not fetch graph state: {exc}")
        expected_doc = ""

    from product_inference.document_handler import download_and_process_file

    success, result = await download_and_process_file(
        temp_path, filename, file_size, user_id,
        sanitize_for_portal=False,
        doc_type_label=expected_doc if expected_doc else "unsolicited",
    )
    if not success:
        yield Error(text=result)
        return

    from product_inference.document_extractor import (
        analyze_document, classify_document, verify_document_type,
    )

    # ── PASS 0: verify against what we asked for, or classify from scratch ──
    if expected_doc:
        label = _pretty(expected_doc)
        yield Status(f"🔍 *Verifying if document matches requested **{label}**...*")
        matches, reason = await asyncio.to_thread(verify_document_type, result, expected_doc)
        if not matches:
            yield Error(
                text=(
                    f"⚠️ I requested your **{label}**, but this looks like something else.\n"
                    f"*(AI verification: {reason})*\n\n"
                    f"Please upload a clear photo of your **{label}** to continue."
                )
            )
            return
        doc_type = expected_doc
    else:
        yield Status("📄 *File secured! Classifying document type...*")
        doc_type = await asyncio.to_thread(classify_document, result)
        label = _pretty(doc_type)

        if doc_type not in _SKIP_CACHE_TYPES:
            cached = await asyncio.to_thread(db.check_vault_cache, user_id, doc_type)
            if cached:
                yield Message(
                    text=(
                        f"✅ **Secure Cache Hit!** I already have your verified **{label}** "
                        f"details stored in the PII Vault (expires in 24 hours). "
                        f"No need to re-scan!"
                    )
                )
                return

    label = _pretty(doc_type)
    yield Status(f"🔍 *Running AI Vision extraction & validation on **{label}**...*")

    # ── PASS 1 & 2: extract, then normalise/validate ──
    analysis = await asyncio.to_thread(analyze_document, result, doc_type)

    if not analysis["success"]:
        yield Error(
            text=f"⚠️ Extraction failed: {analysis.get('raw_text', 'Could not read document.')}"
        )
        return

    extracted = analysis["extracted_data"]

    if not extracted:
        yield Error(
            text="⚠️ The document was processed but no text could be extracted. "
                 "Try a clearer, better-lit photo."
        )
        return

    if not analysis["is_valid"]:
        errors = "\n".join(f"- {err}" for err in analysis["validation_errors"])
        yield Error(
            text=f"⚠️ **Extracted {label}, but found an issue:**\n{errors}\n\n"
                 f"*Please upload a clearer/complete photo.*"
        )
        return

    # ── Vault write (deep-merge UPSERT, 24h TTL) ──
    await asyncio.to_thread(
        db.upsert_vault,
        user_id=user_id,
        document_type=doc_type,
        extracted_fields=extracted,
        source_meta={"filename": filename, "file_path": result},
        ttl_hours=24,
    )

    if not expected_doc:
        preview = json.dumps(extracted, indent=2)
        if len(preview) > 500:
            preview = preview[:497] + "..."
        yield Message(
            text=(
                f"✅ **{label} Saved to Vault!**\n\n```json\n{preview}\n```\n"
                f"*This PII is isolated from your main profile and will auto-expire in 24 hours.*"
            )
        )
        return

    # The document was one the graph asked for — resume it.
    yield Status(f"✅ **{label} Verified & Saved!** Updating application progress...")

    async for event in run_turn(
        user_id,
        f"[DOC_RECEIVED: {doc_type}]",
        status_text="📋 *Updating application progress...*",
    ):
        # run_turn opens with its own Status; the one above already covers it.
        if isinstance(event, Status):
            continue
        yield event


# ==============================================================================
# 3. RESET
# ==============================================================================

def _force_remove(func, path, exc_info):
    """rmtree error hook: clear the read-only bit and retry once.

    Chrome leaves read-only files inside a profile directory, and on Windows `os.unlink`
    refuses those outright. Without this, rmtree aborts partway and leaves a half-deleted
    profile that still holds portal cookies.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def browser_session_ids(user_id: str) -> list[str]:
    """Every browser-session id a citizen's automation can have run under.

    The on-disk profile name is built in two layers that live in different files:
    `graph.py:714` launches automation as `f"auto_{user_id}"`, and `browser_manager`
    then prefixes `user_`. So a citizen's real directory is `user_auto_telegram_123`,
    not `user_telegram_123` — which is exactly what is on disk.

    Both ids are returned because `launch_isolated_profile` is also called with a bare
    id directly (`test_integration.py:91`, the `check_*.py` diagnostics), and a reset
    that only guessed one of the two conventions would report success having deleted
    nothing.
    """
    return [f"auto_{user_id}", user_id]


def wipe_browser_profile(user_id: str) -> bool:
    """Close any live browser session and delete this citizen's persistent Chrome profiles.

    Returns True if at least one profile directory was removed.

    This is the part of `/reset` that was missing. The DB wipe erased the profile, the
    vault and the conversation, but left the cookies that keep a citizen *logged into a
    government portal* sitting on disk — on a shared or family machine that is the single
    artefact a reset most needs to take with it.

    Closing before deleting is not optional on Windows: Chrome holds an exclusive lock on
    its user-data directory, so rmtree against a live profile fails mid-tree.
    """
    try:
        from product_inference import browser_manager
    except Exception as e:
        # Playwright missing (voice-only or web-only deployments never import it). The DB
        # wipe has already happened; a browser profile cannot exist without the launcher.
        print(f"[reset] Browser manager unavailable, skipping profile wipe: {e}")
        return False

    if not user_id:
        print("[reset] Refusing to wipe browser profiles for a blank user_id")
        return False

    root = os.path.normcase(os.path.abspath(browser_manager.BROWSER_PROFILES_ROOT))
    removed = False

    for session_id in browser_session_ids(user_id):
        try:
            browser_manager.close_session(session_id)
        except Exception as e:
            print(f"[reset] Could not close browser session {session_id}: {e}")

        path = browser_manager.profile_dir_for(session_id)

        # Guard against an odd id resolving to the shared root and taking every citizen's
        # profile with it.
        if os.path.normcase(path) == root:
            print(f"[reset] Refusing to delete profile root for session_id={session_id!r}")
            continue

        if not os.path.isdir(path):
            continue

        try:
            shutil.rmtree(path, onexc=_force_remove)
            print(f"[reset] Deleted browser profile {os.path.basename(path)}")
            removed = True
        except Exception as e:
            print(f"[reset] Could not delete browser profile {os.path.basename(path)}: {e}")

    return removed


def do_reset(user_id: str) -> None:
    """Wipe a citizen's profile, vault, checkpointer state, and browser profile.

    Moved from `bot.py:48-60`, where it was a module-local function no other surface could
    import — which is why four separate copies existed (Discord, Telegram, and both halves
    of `bot_audio.py`) and why none of them deleted the browser profile.

    The database wipe runs first and unguarded: if erasing the browser profile fails, the
    citizen's PII is still gone. Reversing that order would let a locked Chrome directory
    keep their Aadhaar extract in the vault.
    """
    import psycopg2
    from product_inference.db import DB_PARAMS

    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("DELETE FROM user_profiles WHERE platform_id = %s;", (user_id,))
    cur.execute("DELETE FROM pii_vault WHERE user_id = %s;", (user_id,))
    cur.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (user_id,))
    cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s;", (user_id,))
    conn.commit()
    cur.close()
    conn.close()

    wipe_browser_profile(user_id)
