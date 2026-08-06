# ==============================================================================
# Yojana Mitra — Web Chat Interface Tests
# File Path: testing_n_diagnostics/test_web_chat_interface.py
# ==============================================================================
"""
Covers the shared core layer and the web surface.

**Runs with Postgres and Ollama down.** Everything slow or stateful is mocked: the graph,
the vision passes, the bouncer, and `db`. That is the point — the logic extracted from the
bot handlers is now testable without standing up a container and four local models, which
it never was while it lived inside `handle_message`.

    pytest testing_n_diagnostics/test_web_chat_interface.py -v

No `pytest-asyncio`. Async generators are driven with `asyncio.run` from sync tests, and
FastAPI's `TestClient` is synchronous, so the suite adds no test-only dependency.
"""

import asyncio
import inspect
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core_inference import profile_form as pf                      # noqa: E402
from core_inference import session as core_session                 # noqa: E402
from core_inference.events import (                                # noqa: E402
    AwaitConfirm, Cancelled, Error, Image, Message, NodeEnter, Status, chunk_text,
)
from product_inference.web import auth, server                     # noqa: E402

USER = "telegram_4242"
OTHER = "telegram_9999"


# ==============================================================================
# Helpers
# ==============================================================================

def drain(agen):
    """Collect an async generator into a list from a synchronous test."""
    async def _run():
        return [event async for event in agen]
    return asyncio.run(_run())


def kinds(events):
    return [e.kind for e in events]


class FakeGraph:
    """Stands in for `graph_app`.

    `stream` yields `(mode, chunk)` pairs exactly as langgraph 1.2.9 does when given
    `stream_mode=["updates", "values"]`.
    """

    def __init__(self, nodes=(), final_state=None, awaiting_document="", raises=None):
        self.nodes = list(nodes)
        self.final_state = final_state or {}
        self.awaiting_document = awaiting_document
        self.raises = raises
        self.invocations = []

    def stream(self, payload, config=None, stream_mode=None):
        self.invocations.append(payload)
        if self.raises:
            raise self.raises
        for node in self.nodes:
            yield "updates", {node: {}}
        yield "values", self.final_state

    def get_state(self, config):
        class _Snapshot:
            values = {"awaiting_document": self.awaiting_document}
        return _Snapshot()


def use_graph(fake):
    return patch.object(core_session, "_graph", lambda: fake)


# ==============================================================================
# 1. events.chunk_text
# ==============================================================================

class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert chunk_text("hello", 100) == ["hello"]

    def test_splits_on_newline_boundaries(self):
        text = "\n".join(["x" * 30] * 4)
        chunks = chunk_text(text, 70)
        assert all(len(c) <= 70 for c in chunks)
        # No line was cut in half.
        assert "".join(chunks).replace("\n", "") == text.replace("\n", "")

    def test_single_overlong_line_is_hard_split(self):
        """The previous inline implementations emitted an over-limit chunk here.

        A 5,000-character line with no newline would be passed to Telegram whole and
        rejected by the API.
        """
        chunks = chunk_text("y" * 250, 100)
        assert all(len(c) <= 100 for c in chunks)
        assert "".join(chunks) == "y" * 250

    def test_rejects_nonpositive_limit(self):
        with pytest.raises(ValueError):
            chunk_text("abc", 0)


# ==============================================================================
# 2. profile_form
# ==============================================================================

class TestProfileForm:
    def test_thirteen_fields_in_order(self):
        assert len(pf.FORM_FLOW) == 13
        assert pf.FORM_FIELDS[0] == "name"
        assert pf.FORM_FIELDS[-1] == "income"

    @pytest.mark.parametrize("raw,expected", [
        ("45000", 45000), ("45,000", 45000), ("₹45000", 45000),
        (" 45 000 ", 45000), (45000, 45000),
    ])
    def test_income_parsing(self, raw, expected):
        ok, value = pf.coerce_field("income", raw)
        assert ok and value == expected

    @pytest.mark.parametrize("raw", ["abc", "", "12.5", "-40"])
    def test_income_rejects_non_numeric(self, raw):
        ok, message = pf.coerce_field("income", raw)
        assert not ok and isinstance(message, str)

    def test_age_bounds(self):
        assert pf.coerce_field("age", "22") == (True, 22)
        assert pf.coerce_field("age", "200")[0] is False
        assert pf.coerce_field("age", "0")[0] is False

    def test_booleans_coerced_from_strings(self):
        assert pf.coerce_field("minority", "true") == (True, True)
        assert pf.coerce_field("minority", "false") == (True, False)
        assert pf.coerce_field("minority", True) == (True, True)

    def test_disability_percentage_is_derived(self):
        """Matches bot.py:189 and bot_telegram.py:236 — asked once, stored as two fields."""
        data = {}
        assert pf.apply_field(data, "differently_abled", "true") == (True, None)
        assert data == {"differently_abled": True, "disability_percentage": 40}

        data = {}
        pf.apply_field(data, "differently_abled", "false")
        assert data["disability_percentage"] == 0

    def test_disability_percentage_is_not_a_form_question(self):
        assert "disability_percentage" not in pf.FORM_FIELDS

    def test_name_and_occupation_title_cased(self):
        assert pf.coerce_field("name", "  faraz nezam ") == (True, "Faraz Nezam")

    def test_validate_profile_gates_on_five_fields(self):
        """graph.py:is_profile_complete checks these five, not all thirteen."""
        assert pf.validate_profile({}) != []
        complete = {"gender": "Male", "age": 22, "income": 45000,
                    "caste": "OBC", "occupation": "Student"}
        assert pf.validate_profile(complete) == []

    def test_validate_rejects_stringly_typed_numbers(self):
        bad = {"gender": "Male", "age": "22", "income": 45000,
               "caste": "OBC", "occupation": "Student"}
        assert "age must be an integer" in pf.validate_profile(bad)

    def test_summary_survives_non_numeric_income(self):
        """Both bots called int(g('income', 0)) directly, which raises on a bad value."""
        out = pf.format_profile_summary({"income": "not-a-number"})
        assert "not-a-number" in out

    def test_json_schema_covers_every_field(self):
        schema = pf.as_json_schema()
        assert [s["field"] for s in schema] == pf.FORM_FIELDS
        occupation = next(s for s in schema if s["field"] == "occupation")
        assert occupation["allows_free_text"] is True
        # The sentinel must never be offered as a real option.
        assert all(o["value"] != pf.FREE_TEXT_SENTINEL for o in occupation["options"])


# ==============================================================================
# 3. session.run_turn
# ==============================================================================

class TestRunTurn:
    def test_status_precedes_message(self):
        fake = FakeGraph(nodes=["classify_intent"], final_state={"response": "Hi!"})
        with use_graph(fake):
            events = drain(core_session.run_turn(USER, "hello"))
        assert kinds(events)[0] == "status"
        assert kinds(events).index("status") < kinds(events).index("message")
        assert events[-1].text == "Hi!"

    def test_node_updates_become_nodeenter_events(self):
        fake = FakeGraph(
            nodes=["load_user_profile", "classify_intent", "handle_scheme_query"],
            final_state={"response": "ok"},
        )
        with use_graph(fake):
            events = drain(core_session.run_turn(USER, "schemes"))
        assert [e.node for e in events if isinstance(e, NodeEnter)] == [
            "load_user_profile", "classify_intent", "handle_scheme_query",
        ]

    def test_empty_response_gets_a_fallback(self):
        with use_graph(FakeGraph(final_state={"response": "   "})):
            events = drain(core_session.run_turn(USER, "hi"))
        message = [e for e in events if isinstance(e, Message)][0]
        assert "empty" in message.text.lower()

    def test_text_passes_through_unmodified(self):
        """CONFIRM must reach the graph verbatim — this layer must not interpret it."""
        fake = FakeGraph(final_state={"response": "done"})
        with use_graph(fake):
            drain(core_session.run_turn(USER, "CONFIRM"))
        assert fake.invocations[0]["messages"][0]["content"] == "CONFIRM"

    def test_graph_exception_becomes_error_event(self):
        with use_graph(FakeGraph(raises=RuntimeError("pool exhausted"))):
            events = drain(core_session.run_turn(USER, "hi"))
        error = [e for e in events if isinstance(e, Error)][0]
        assert "pool exhausted" in error.detail
        # The citizen-facing text must not leak the internal detail.
        assert "pool exhausted" not in error.text

    @pytest.mark.parametrize("status,suffix", [
        ("awaiting_confirm", "_final_review.png"),
        ("complete", "_submitted.png"),
        ("hitl_paused", "_otp_intercept.png"),
    ])
    def test_screenshot_emitted_for_each_automation_status(self, tmp_path, status, suffix):
        """`complete` is included deliberately — it was the status whose screenshot
        (the acknowledgement number the citizen most needs) went unsent for a while."""
        shot_dir = tmp_path / "screenshots"
        shot_dir.mkdir()
        (shot_dir / f"auto_{USER}{suffix}").write_bytes(b"png")

        fake = FakeGraph(final_state={
            "response": "r", "automation_status": status,
            "automation_session_id": f"auto_{USER}",
        })
        with use_graph(fake), patch.object(core_session, "SCREENSHOT_DIR", str(shot_dir)):
            events = drain(core_session.run_turn(USER, "apply"))

        images = [e for e in events if isinstance(e, Image)]
        assert len(images) == 1 and images[0].path.endswith(suffix)

    def test_no_image_when_file_missing(self, tmp_path):
        fake = FakeGraph(final_state={
            "response": "r", "automation_status": "complete",
            "automation_session_id": f"auto_{USER}",
        })
        with use_graph(fake), patch.object(core_session, "SCREENSHOT_DIR", str(tmp_path)):
            events = drain(core_session.run_turn(USER, "apply"))
        assert not [e for e in events if isinstance(e, Image)]

    def test_no_image_for_idle_status(self, tmp_path):
        (tmp_path / f"auto_{USER}_final_review.png").write_bytes(b"png")
        fake = FakeGraph(final_state={"response": "r", "automation_status": "idle"})
        with use_graph(fake), patch.object(core_session, "SCREENSHOT_DIR", str(tmp_path)):
            events = drain(core_session.run_turn(USER, "hi"))
        assert not [e for e in events if isinstance(e, Image)]

    def test_awaitconfirm_only_when_graph_says_so(self):
        with use_graph(FakeGraph(final_state={"response": "r", "automation_status": "running"})):
            events = drain(core_session.run_turn(USER, "apply"))
        assert not [e for e in events if isinstance(e, AwaitConfirm)]

        with use_graph(FakeGraph(final_state={"response": "r", "automation_status": "awaiting_confirm"})):
            events = drain(core_session.run_turn(USER, "apply"))
        assert [e for e in events if isinstance(e, AwaitConfirm)]


# ==============================================================================
# 4. session.ingest_document
# ==============================================================================

def _patch_pipeline(*, bouncer=(True, "/tmp/clean.jpg"), classify="aadhaar",
                    verify=(True, "aadhaar"), analysis=None, cache=None):
    """Patch every external call ingest_document makes."""
    analysis = analysis or {
        "success": True, "extracted_data": {"aadhaar_number": "1234"},
        "is_valid": True, "validation_errors": [],
    }

    async def fake_bouncer(*a, **k):
        return bouncer

    return [
        patch("product_inference.document_handler.download_and_process_file", fake_bouncer),
        patch("product_inference.document_extractor.classify_document", lambda *a: classify),
        patch("product_inference.document_extractor.verify_document_type", lambda *a: verify),
        patch("product_inference.document_extractor.analyze_document", lambda *a: analysis),
        patch("product_inference.db.check_vault_cache", lambda *a: cache),
        patch("product_inference.db.upsert_vault", lambda **k: None),
    ]


class TestIngestDocument:
    def _run(self, fake_graph, patches):
        stack = [use_graph(fake_graph), *patches]
        for p in stack:
            p.__enter__()
        try:
            return drain(core_session.ingest_document(USER, "/tmp/x.jpg", "x.jpg", 1000))
        finally:
            for p in reversed(stack):
                p.__exit__(None, None, None)

    def test_bouncer_rejection_stops_the_pipeline(self):
        """Nothing may reach the VLM if document_handler refused the file."""
        called = {"analyze": False}

        def spy(*a, **k):
            called["analyze"] = True
            return {}

        patches = _patch_pipeline(bouncer=(False, "❌ File too large"))
        patches[3] = patch("product_inference.document_extractor.analyze_document", spy)

        events = self._run(FakeGraph(), patches)
        assert isinstance(events[-1], Error)
        assert called["analyze"] is False

    def test_unsolicited_document_is_classified_and_saved(self):
        events = self._run(FakeGraph(awaiting_document=""), _patch_pipeline())
        assert "message" in kinds(events)
        assert "Aadhaar" in events[-1].text

    def test_cache_hit_short_circuits_before_extraction(self):
        called = {"analyze": False}

        def spy(*a, **k):
            called["analyze"] = True
            return {}

        patches = _patch_pipeline(cache={"aadhaar_number": "1234"})
        patches[3] = patch("product_inference.document_extractor.analyze_document", spy)

        events = self._run(FakeGraph(awaiting_document=""), patches)
        assert "Secure Cache Hit" in events[-1].text
        assert called["analyze"] is False

    def test_expected_document_mismatch_is_rejected(self):
        events = self._run(
            FakeGraph(awaiting_document="income_certificate"),
            _patch_pipeline(verify=(False, "looks like a PAN card")),
        )
        assert isinstance(events[-1], Error)
        assert "Income Certificate" in events[-1].text

    def test_expected_document_resumes_the_graph(self):
        fake = FakeGraph(
            awaiting_document="aadhaar",
            nodes=["request_next_document"],
            final_state={"response": "Got it — next, your income certificate."},
        )
        events = self._run(fake, _patch_pipeline())
        assert fake.invocations, "graph was never resumed"
        assert fake.invocations[0]["messages"][0]["content"] == "[DOC_RECEIVED: aadhaar]"
        assert events[-1].text.startswith("Got it")

    def test_failed_validation_warns_without_saving(self):
        saved = {"called": False}

        def spy(**k):
            saved["called"] = True

        patches = _patch_pipeline(analysis={
            "success": True, "extracted_data": {"aadhaar_number": "12"},
            "is_valid": False, "validation_errors": ["Aadhaar must be 12 digits"],
        })
        patches[5] = patch("product_inference.db.upsert_vault", spy)

        events = self._run(FakeGraph(), patches)
        assert isinstance(events[-1], Error)
        assert "12 digits" in events[-1].text
        assert saved["called"] is False

    def test_empty_extraction_is_reported(self):
        events = self._run(FakeGraph(), _patch_pipeline(analysis={
            "success": True, "extracted_data": {}, "is_valid": True, "validation_errors": [],
        }))
        assert isinstance(events[-1], Error)
        assert "no text" in events[-1].text.lower()


# ==============================================================================
# 5. Web auth
# ==============================================================================

class TestAuth:
    def test_round_trip(self):
        assert auth.verify_token(auth.mint_token(USER)) == USER

    @pytest.mark.parametrize("token", [
        None, "", "garbage", "a.b.c", "onlyonepart",
        "eyJ1aWQiOiJ4In0",                    # payload, no signature
        "eyJ1aWQiOiJ4In0.aW52YWxpZHNpZw",     # wrong signature
    ])
    def test_malformed_tokens_rejected(self, token):
        assert auth.verify_token(token) is None

    def test_tampered_payload_rejected(self):
        """Swapping the uid without re-signing must not be accepted."""
        token = auth.mint_token(USER)
        payload_b64, signature = token.split(".")
        forged = auth._b64e(b'{"uid":"telegram_9999","exp":9999999999}')
        assert auth.verify_token(f"{forged}.{signature}") is None

    def test_expired_token_rejected(self):
        assert auth.verify_token(auth.mint_token(USER, ttl_seconds=-1)) is None

    def test_signature_depends_on_secret(self):
        with patch.dict(os.environ, {"WEB_SESSION_SECRET": "secret-one"}):
            token = auth.mint_token(USER)
        with patch.dict(os.environ, {"WEB_SESSION_SECRET": "secret-two"}):
            assert auth.verify_token(token) is None

    def test_handoff_ttl_is_short(self):
        assert auth.HANDOFF_TTL_SECONDS <= 600

    def test_handoff_url_shape(self):
        url = auth.handoff_url(USER, base_url="https://example.test/")
        assert url.startswith("https://example.test/auth?t=")
        assert auth.verify_token(url.split("t=", 1)[1]) == USER


# ==============================================================================
# 6. Web routes
# ==============================================================================

@pytest.fixture
def client():
    with patch.dict(os.environ, {"WEB_SESSION_SECRET": "test-secret-for-suite"}):
        yield TestClient(server.app)


@pytest.fixture
def signed_in(client):
    client.cookies.set(auth.COOKIE_NAME, auth.mint_token(USER))
    return client


class TestRoutesRequireAuth:
    @pytest.mark.parametrize("method,path", [
        ("get", "/api/me"), ("get", "/api/profile"),
        ("post", "/api/reset"), ("post", "/api/stop"),
        ("get", "/api/screenshot/auto_telegram_4242_submitted.png"),
    ])
    def test_no_cookie_is_401(self, client, method, path):
        assert getattr(client, method)(path).status_code == 401

    def test_forged_cookie_is_401(self, client):
        client.cookies.set(auth.COOKIE_NAME, "forged.token")
        assert client.get("/api/me").status_code == 401

    def test_expired_cookie_is_401(self, client):
        client.cookies.set(auth.COOKIE_NAME, auth.mint_token(USER, ttl_seconds=-1))
        assert client.get("/api/me").status_code == 401

    def test_valid_cookie_resolves_identity(self, signed_in):
        body = signed_in.get("/api/me").json()
        assert body["user_id"] == USER and body["platform"] == "telegram"


class TestAuthHandoff:
    def test_valid_handoff_sets_cookie_and_redirects(self, client):
        token = auth.mint_handoff(USER)
        res = client.get(f"/auth?t={token}", follow_redirects=False)
        assert res.status_code == 303
        assert auth.COOKIE_NAME in res.cookies
        assert auth.verify_token(res.cookies[auth.COOKIE_NAME]) == USER

    def test_expired_handoff_sets_no_cookie(self, client):
        token = auth.mint_token(USER, ttl_seconds=-1)
        res = client.get(f"/auth?t={token}", follow_redirects=False)
        assert res.status_code == 401
        assert auth.COOKIE_NAME not in res.cookies


class TestScreenshotGuard:
    """These images show a government form filled with someone's PII."""

    @pytest.mark.parametrize("name", [
        "../.env", "..%2f..%2f.env", "....//.env",
        "/etc/passwd", "..\\..\\.env", "sub/dir.png",
    ])
    def test_traversal_attempts_never_return_a_file(self, signed_in, name):
        res = signed_in.get(f"/api/screenshot/{name}")
        assert res.status_code in (307, 400, 403, 404)
        assert b"SECRET" not in res.content and b"TOKEN" not in res.content

    def test_cannot_read_another_citizens_screenshot(self, signed_in, tmp_path):
        shot_dir = tmp_path / "screenshots"
        shot_dir.mkdir()
        victim = shot_dir / f"auto_{OTHER}_final_review.png"
        victim.write_bytes(b"\x89PNG-someone-elses-aadhaar-form")

        with patch.object(server, "SCREENSHOT_ROOT", shot_dir.resolve()):
            res = signed_in.get(f"/api/screenshot/auto_{OTHER}_final_review.png")

        assert res.status_code == 403
        assert b"aadhaar" not in res.content

    def test_own_screenshot_is_served(self, signed_in, tmp_path):
        shot_dir = tmp_path / "screenshots"
        shot_dir.mkdir()
        (shot_dir / f"auto_{USER}_submitted.png").write_bytes(b"\x89PNG-mine")

        with patch.object(server, "SCREENSHOT_ROOT", shot_dir.resolve()):
            res = signed_in.get(f"/api/screenshot/auto_{USER}_submitted.png")

        assert res.status_code == 200
        assert res.content == b"\x89PNG-mine"


class TestNoClientSuppliedIdentity:
    def test_no_route_accepts_user_id_as_a_parameter(self):
        """`user_id` is the PII vault partition key. It may only come from the cookie.

        A route that accepted `?user_id=` would let any signed-in citizen read any
        other citizen's vault, so this is asserted against FastAPI's real route table
        rather than trusted to code review. Introspecting the bound endpoints (instead
        of grepping the source) means a route added via a router, a decorator, or a
        dependency is still covered.
        """
        for route in server.app.routes:
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None:
                continue
            params = inspect.signature(endpoint).parameters
            assert "user_id" not in params, (
                f"route {getattr(route, 'path', '?')} accepts user_id as a parameter"
            )

    def test_identity_has_exactly_one_producer(self):
        """Every authenticated route must funnel through `_require_user`."""
        source = Path(server.__file__).read_text(encoding="utf-8")
        assert source.count("def _require_user") == 1
        assert "auth.verify_token" in source

    def test_every_sensitive_route_calls_require_user(self):
        """A new route that forgets the auth check should fail here, not in production."""
        source = Path(server.__file__).read_text(encoding="utf-8")
        public = {"/", "/auth", "/healthz", "/app.js", "/styles.css", "/api/logout"}

        for route in server.app.routes:
            path = getattr(route, "path", "")
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None or path in public or not path.startswith("/api"):
                continue
            body = inspect.getsource(endpoint)
            assert "_require_user" in body, f"route {path} does not authenticate"


class TestChatRoute:
    def test_empty_message_rejected(self, signed_in):
        assert signed_in.post("/api/chat", json={"text": "   "}).status_code == 400

    def test_turn_streams_sse_events(self, signed_in):
        fake = FakeGraph(nodes=["classify_intent"], final_state={"response": "Namaste!"})
        with use_graph(fake):
            res = signed_in.post("/api/chat", json={"text": "hello"})

        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        body = res.text
        assert '"kind": "status"' in body
        assert '"kind": "nodeenter"' in body
        assert "Namaste!" in body
        assert body.rstrip().endswith('data: {"kind": "end"}')

    def test_confirm_reaches_the_graph_verbatim(self, signed_in):
        """The browser's Confirm button is not a privileged route."""
        fake = FakeGraph(final_state={"response": "Submitted."})
        with use_graph(fake):
            signed_in.post("/api/chat", json={"text": "CONFIRM"})
        assert fake.invocations[0]["messages"][0]["content"] == "CONFIRM"

    def test_image_event_carries_only_a_basename(self, signed_in, tmp_path):
        """A server filesystem path must never cross the wire."""
        shot_dir = tmp_path / "screenshots"
        shot_dir.mkdir()
        (shot_dir / f"auto_{USER}_submitted.png").write_bytes(b"png")

        fake = FakeGraph(final_state={
            "response": "done", "automation_status": "complete",
            "automation_session_id": f"auto_{USER}",
        })
        with use_graph(fake), patch.object(core_session, "SCREENSHOT_DIR", str(shot_dir)):
            body = signed_in.post("/api/chat", json={"text": "apply"}).text

        assert f"auto_{USER}_submitted.png" in body
        assert str(tmp_path) not in body


class TestUploadRoute:
    def test_rejects_unsupported_extension(self, signed_in):
        res = signed_in.post("/api/upload", files={"file": ("virus.exe", b"MZ", "application/octet-stream")})
        assert res.status_code == 400

    def test_accepted_upload_runs_the_pipeline(self, signed_in):
        patches = _patch_pipeline()
        stack = [use_graph(FakeGraph(awaiting_document="")), *patches]
        for p in stack:
            p.__enter__()
        try:
            res = signed_in.post("/api/upload", files={"file": ("aadhaar.jpg", b"\xff\xd8\xff", "image/jpeg")})
        finally:
            for p in reversed(stack):
                p.__exit__(None, None, None)

        assert res.status_code == 200
        assert "Aadhaar" in res.text


def stored_profile(data: dict, state: str = "PROFILE_COMPLETE"):
    """Patch what the DB already holds for this citizen.

    Both /api/profile verbs read the stored row now — GET to prefill, POST to merge onto
    rather than clobber. Patching it keeps the suite runnable with Postgres down, which is
    how it is usually run.
    """
    return patch("product_inference.db.get_or_create_user",
                 lambda *a, **k: {"profile_data": dict(data), "current_state": state})


class save_capture:
    """Records the (user_id, state, data) triple the endpoint would have written."""

    def __init__(self):
        self.user_id = self.state = self.data = None

    def save(self, user_id, state, data=None):
        self.user_id, self.state, self.data = user_id, state, data


class TestProfileRoute:
    def test_get_returns_schema_and_saved_profile(self, signed_in):
        saved = {"gender": "Male", "age": 22, "income": 45000,
                 "caste": "OBC", "occupation": "Student"}
        with patch("product_inference.db.get_or_create_user",
                   lambda *a: {"profile_data": saved, "current_state": "PROFILE_COMPLETE"}):
            body = signed_in.get("/api/profile").json()

        assert len(body["schema"]) == 13
        assert body["profile"]["age"] == 22
        assert body["complete"] is True

    def test_post_coerces_like_the_telegram_form(self, signed_in):
        captured = save_capture()
        payload = {
            "name": "faraz nezam", "gender": "Male", "age": "22",
            "income": "45,000", "caste": "OBC", "occupation": "student",
            "differently_abled": "true",
        }
        with stored_profile({}), patch("product_inference.db.update_user_state", captured.save):
            res = signed_in.post("/api/profile", json={"profile": payload})

        assert res.status_code == 200
        data = captured.data
        assert data["age"] == 22 and isinstance(data["age"], int)
        assert data["income"] == 45000
        assert data["name"] == "Faraz Nezam"
        assert data["differently_abled"] is True
        assert data["disability_percentage"] == 40
        assert captured.user_id == USER
        assert captured.state == "PROFILE_COMPLETE"

    def test_post_reports_field_errors(self, signed_in):
        with stored_profile({}), patch("product_inference.db.update_user_state", lambda *a: None):
            res = signed_in.post("/api/profile", json={"profile": {
                "gender": "Male", "age": "not-a-number", "income": "45000",
                "caste": "OBC", "occupation": "Student",
            }})
        assert res.status_code == 400
        assert "age" in res.json()["errors"]

    def test_post_rejects_non_object(self, signed_in):
        assert signed_in.post("/api/profile", json={"profile": "nope"}).status_code == 400


class TestPartialProfileIsAllowed:
    """The profile is a nudge, not a gate.

    Scheme search runs on five fields; all thirteen only sharpen personalisation. So a
    citizen answering the side panel one field at a time must have that work persisted,
    and must never be refused a turn for it.
    """

    def test_a_single_field_saves(self, signed_in):
        captured = save_capture()
        with stored_profile({}), patch("product_inference.db.update_user_state", captured.save):
            res = signed_in.post("/api/profile", json={"profile": {"gender": "Male"}})

        assert res.status_code == 200
        assert res.json()["ok"] is True
        assert captured.data["gender"] == "Male"

    def test_an_incomplete_save_does_not_claim_completion(self, signed_in):
        captured = save_capture()
        with stored_profile({}, state="START"), \
             patch("product_inference.db.update_user_state", captured.save):
            body = signed_in.post("/api/profile", json={"profile": {"gender": "Male"}}).json()

        # Advancing the FSM here would tell the graph a profile exists that does not.
        assert captured.state == "START"
        assert body["progress"]["usable"] is False
        assert body["summary"] is None

    def test_the_five_gate_fields_flip_it_to_usable(self, signed_in):
        captured = save_capture()
        with stored_profile({}, state="START"), \
             patch("product_inference.db.update_user_state", captured.save):
            body = signed_in.post("/api/profile", json={"profile": {
                "gender": "Male", "age": "22", "income": "45000",
                "caste": "OBC", "occupation": "Student",
            }}).json()

        assert captured.state == "PROFILE_COMPLETE"
        assert body["progress"]["usable"] is True
        # Usable is not finished — 5 of 13 answered still leaves the nudge on screen.
        assert body["progress"]["answered"] == 5
        assert body["summary"] is None

    def test_a_partial_save_never_overwrites_earlier_answers(self, signed_in):
        """The regression this endpoint was built to avoid.

        `db.update_user_state` merges with `profile_data || %s::jsonb`, so anything the
        endpoint puts in the payload wins. Building the payload from `defaults()` meant a
        one-field save silently rewrote residence to Urban and marital_status to Single.
        """
        already = {"name": "Faraz Nezam", "residence": "Rural",
                   "marital_status": "Married", "caste": "OBC"}
        captured = save_capture()
        with stored_profile(already), patch("product_inference.db.update_user_state", captured.save):
            signed_in.post("/api/profile", json={"profile": {"age": "22"}})

        assert captured.data["residence"] == "Rural"
        assert captured.data["marital_status"] == "Married"
        assert captured.data["name"] == "Faraz Nezam"
        assert captured.data["age"] == 22

    def test_unanswered_fields_are_not_invented(self, signed_in):
        """hybrid_rag turns every present field into a line of LLM context.

        A defaulted `residence` is not a harmless placeholder — it is the model being told
        this citizen lives in a city when nobody asked.
        """
        captured = save_capture()
        with stored_profile({}), patch("product_inference.db.update_user_state", captured.save):
            signed_in.post("/api/profile", json={"profile": {"name": "Faraz"}})

        for never_asked in ("residence", "marital_status", "gender", "age", "income"):
            assert never_asked not in captured.data, f"invented {never_asked}"

    def test_blank_values_are_not_errors(self, signed_in):
        """An empty box in the panel is an unanswered question, not a bad answer."""
        captured = save_capture()
        with stored_profile({}), patch("product_inference.db.update_user_state", captured.save):
            res = signed_in.post("/api/profile", json={"profile": {
                "name": "Faraz", "age": "", "income": None,
            }})

        assert res.status_code == 200
        assert "age" not in captured.data and "income" not in captured.data

    def test_get_reports_progress_for_the_nudge(self, signed_in):
        saved = {"gender": "Male", "age": 22, "income": 45000,
                 "caste": "OBC", "occupation": "Student"}
        with stored_profile(saved):
            body = signed_in.get("/api/profile").json()

        assert body["progress"]["answered"] == 5
        assert body["progress"]["total"] == 13
        assert "name" in body["progress"]["missing"]

    def test_a_full_profile_gets_a_summary_and_no_nudge(self, signed_in):
        full = {f: "Yes" for f in pf.FORM_FIELDS}
        full.update({"age": "22", "income": "45000", "gender": "Male",
                     "caste": "OBC", "occupation": "Student", "name": "Faraz"})
        captured = save_capture()
        with stored_profile({}), patch("product_inference.db.update_user_state", captured.save):
            body = signed_in.post("/api/profile", json={"profile": full}).json()

        assert body["progress"]["answered"] == 13
        assert body["summary"] is not None
        assert "Profile Saved Successfully" in body["summary"]


class TestStaticSurface:
    def test_signed_out_root_shows_the_telegram_instruction(self, client):
        body = client.get("/").text
        assert "/web" in body and "Telegram" in body

    def test_signed_in_root_shows_the_chat(self, signed_in):
        assert "composer" in signed_in.get("/").text

    def test_healthz(self, client):
        assert client.get("/healthz").json() == {"ok": True}

    def test_the_profile_is_a_panel_not_a_blocking_overlay(self, signed_in):
        page = signed_in.get("/").text
        assert 'id="profile-section"' in page
        assert 'id="nudge"' in page
        # The overlay was modal: it covered the conversation and took the composer with
        # it. Filling in a profile must never cost the citizen their place in the chat.
        assert 'class="overlay' not in page
        assert 'id="profile-overlay"' not in page

    def test_the_panel_carries_its_own_way_back_on_mobile(self, signed_in):
        # .shell.show-side hides .main, and the header lives inside .main — so the toggle
        # that opened the panel is itself hidden while the panel is open.
        page = signed_in.get("/").text
        assert 'id="btn-back"' in page
        css = signed_in.get("/styles.css").text
        assert ".shell.show-side .side .back" in css

    def test_the_confirm_button_is_still_not_a_privileged_route(self, signed_in):
        js = signed_in.get("/app.js").text
        # It must send the literal string through the ordinary chat path, so the graph's
        # interrupt/resume gate stays the only thing deciding a government form submit.
        assert 'send("CONFIRM")' in js


# ==============================================================================
# 7. Telegram renderer
# ==============================================================================
# `bot_telegram._render_events` is the Telegram half of the split: core_session decides
# *what* happened, this decides what a chat window does about it. It is where the old
# handler's real bugs lived, so it gets tested directly.

@pytest.fixture(scope="module")
def bt():
    """The Telegram bot module, imported with its module-scope `db.init_db()` stubbed.

    The bot calls `init_db()` at import time and that needs a live Postgres. Nothing in
    the renderer touches the database, so skipping the call is enough to make it testable
    on a machine with no container running.
    """
    import product_inference.db as _db
    with patch.object(_db, "init_db", lambda *a, **k: None):
        import product_inference.bot_telegram as module
    return module


class Recorder:
    """Everything the renderer did, in order."""

    def __init__(self):
        self.edits = []    # (text, kwargs) applied to the status slot
        self.replies = []  # text of every new message
        self.photos = []


class FakeSlot:
    """The placeholder message the renderer edits in place.

    `markdown_ok=False` reproduces Telegram rejecting unbalanced entities; `editable=False`
    reproduces a message too old (or already deleted) to edit at all.
    """

    def __init__(self, rec, markdown_ok=True, editable=True):
        self._rec, self._markdown_ok, self._editable = rec, markdown_ok, editable

    async def edit_text(self, text, **kwargs):
        if not self._editable:
            raise RuntimeError("Bad Request: message can't be edited")
        if kwargs.get("parse_mode") and not self._markdown_ok:
            raise RuntimeError("Bad Request: can't parse entities")
        self._rec.edits.append((text, kwargs))


class FakeUserMessage:
    def __init__(self, rec):
        self._rec = rec

    async def reply_text(self, text, **kwargs):
        self._rec.replies.append(text)
        return FakeSlot(self._rec)


class FakeUpdate:
    def __init__(self, rec):
        self.message = FakeUserMessage(rec)
        self.effective_chat = type("Chat", (), {"id": 1, "type": "private"})()
        self.effective_user = type("User", (), {"id": 4242})()


class FakeContext:
    def __init__(self, rec):
        async def send_photo(chat_id, photo):
            rec.photos.append(getattr(photo, "name", str(photo)))
        self.bot = type("Bot", (), {"send_photo": staticmethod(send_photo)})()


def render(bt_module, events, markdown_ok=True, editable=True, shown=""):
    """Drive `_render_events` over a fixed event list and report what landed on screen."""
    rec = Recorder()
    update, context = FakeUpdate(rec), FakeContext(rec)
    slot = FakeSlot(rec, markdown_ok=markdown_ok, editable=editable)

    async def _stream():
        for event in events:
            yield event

    asyncio.run(bt_module._render_events(_stream(), update, context, slot, shown=shown))
    return rec


def delivered(rec):
    """Every piece of text the citizen actually saw."""
    return [text for text, _ in rec.edits] + rec.replies


class TestTelegramMarkdown:
    def test_double_asterisk_becomes_single(self, bt):
        # Core events are written in Discord's dialect; Telegram's legacy parser needs
        # single asterisks and shows the doubled form literally.
        assert bt._tg_markdown("your **Aadhaar Card** is saved") == "your *Aadhaar Card* is saved"

    def test_spans_newlines(self, bt):
        assert bt._tg_markdown("**two\nlines**") == "*two\nlines*"

    def test_plain_text_untouched(self, bt):
        assert bt._tg_markdown("no emphasis here") == "no emphasis here"

    def test_attempts_degrade_to_plain(self, bt):
        attempts = list(bt._render_attempts("a **b**", markdown=True))
        assert attempts[0][1]["parse_mode"] == "Markdown"
        assert attempts[-1] == ("a **b**", {})

    def test_non_markdown_has_one_attempt(self, bt):
        assert list(bt._render_attempts("x", markdown=False)) == [("x", {})]


class TestRenderEvents:
    def test_status_then_message(self, bt):
        rec = render(bt, [Status("working"), Message(text="done")])
        assert [text for text, _ in rec.edits] == ["working", "done"]

    def test_node_enter_is_not_drawn(self, bt):
        # The state-machine trace is a web side-panel affordance. Echoing every node into
        # a phone chat would bury the answer.
        rec = render(bt, [NodeEnter(node="intent_router"), NodeEnter(node="rag")])
        assert rec.edits == [] and rec.replies == []

    def test_await_confirm_adds_no_button(self, bt):
        # A one-tap Confirm here would make this surface a second decision point on
        # whether a government form gets submitted. CONFIRM stays a typed human act.
        rec = render(bt, [AwaitConfirm(prompt="Reply CONFIRM", session_id="s1")])
        assert rec.edits == [] and rec.replies == []

    def test_placeholder_is_not_re_sent(self, bt):
        # Telegram 400s on an edit that changes nothing, and the bot's own placeholder is
        # the same string run_turn opens with.
        rec = render(bt, [Status("busy"), Message(text="answer")], shown="busy")
        assert [text for text, _ in rec.edits] == ["answer"]

    def test_error_is_rendered_like_a_message(self, bt):
        rec = render(bt, [Error(text="failed", detail="ValueError: x")])
        assert rec.edits[0][0] == "failed"

    def test_detail_never_reaches_the_chat(self, bt):
        rec = render(bt, [Error(text="failed", detail="psycopg2 OperationalError at host")])
        assert "psycopg2" not in " ".join(t for t, _ in rec.edits)

    def test_cancelled_is_rendered(self, bt):
        rec = render(bt, [Cancelled()])
        assert rec.edits and "cancel" in rec.edits[0][0].lower()


class TestRenderChunking:
    def test_long_message_keeps_every_chunk(self, bt):
        # The bug this replaces: the old loop edited the *same* message once per chunk and
        # then edited it again with the last one, so a long answer arrived as its final
        # 4,000 characters and everything before it was silently lost.
        body = "\n".join(f"line {i} " + "x" * 90 for i in range(300))
        rec = render(bt, [Message(text=body)])

        seen = delivered(rec)
        assert len(seen) > 1
        assert "line 0" in seen[0]
        assert "line 299" in seen[-1]

    def test_no_chunk_exceeds_the_api_limit(self, bt):
        body = "\n".join("y" * 500 for _ in range(80))
        rec = render(bt, [Message(text=body)])
        for text in delivered(rec):
            assert len(text) <= bt.TELEGRAM_CHUNK_LIMIT

    def test_overflow_chunks_are_new_messages(self, bt):
        body = "\n".join("z" * 200 for _ in range(60))
        rec = render(bt, [Message(text=body)])
        assert len(rec.edits) == 1        # first chunk replaces the placeholder
        assert len(rec.replies) >= 1      # the rest arrive as fresh messages


class TestRenderFallbacks:
    def test_bad_markdown_falls_back_to_plain(self, bt):
        # Model output routinely has an odd number of asterisks, which Telegram rejects
        # outright rather than rendering as-is. The answer must still arrive.
        rec = render(bt, [Message(text="unbalanced * asterisk")], markdown_ok=False)
        assert len(rec.edits) == 1
        assert rec.edits[0][1] == {}                       # retried without parse_mode
        assert rec.edits[0][0] == "unbalanced * asterisk"  # and unmangled

    def test_dead_slot_falls_back_to_a_new_message(self, bt):
        rec = render(bt, [Message(text="the answer")], editable=False)
        assert rec.edits == []
        assert rec.replies == ["the answer"]

    def test_a_dead_slot_does_not_lose_later_events(self, bt):
        # Once the placeholder is replaced, the fresh message becomes the slot and the
        # rest of the turn edits that — nothing after the failure is dropped.
        rec = render(bt, [Status("working"), Message(text="the answer")], editable=False)
        assert "the answer" in delivered(rec)


class TestRenderImages:
    def test_screenshot_is_sent(self, bt, tmp_path):
        shot = tmp_path / "auto_telegram_4242_final_review.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\n")
        rec = render(bt, [Image(path=str(shot), caption="awaiting_confirm")])
        assert len(rec.photos) == 1

    def test_missing_screenshot_does_not_break_the_turn(self, bt, tmp_path):
        # A screenshot the automation never wrote must not swallow the text answer.
        rec = render(bt, [
            Image(path=str(tmp_path / "nope.png")),
            Message(text="here is your status"),
        ])
        assert rec.photos == []
        assert rec.edits[0][0] == "here is your status"


class TestTelegramWiring:
    def test_web_command_sends_a_handoff_link(self, bt):
        rec = Recorder()
        asyncio.run(bt.web_command(FakeUpdate(rec), FakeContext(rec)))
        assert len(rec.replies) == 1
        assert "/auth?t=" in rec.replies[0]

    def test_web_command_warns_the_link_is_an_identity(self, bt):
        rec = Recorder()
        asyncio.run(bt.web_command(FakeUpdate(rec), FakeContext(rec)))
        assert "forward" in rec.replies[0].lower()

    def test_web_handler_is_registered(self, bt):
        assert 'CommandHandler("web", web_command)' in inspect.getsource(bt.main)

    def test_the_duplicated_pipeline_is_actually_gone(self, bt):
        # This is the point of the refactor, not a style preference: the document
        # pipeline and the screenshot map used to exist once per bot, which is how
        # bot_audio.py ended up with neither. The bot must no longer reach for the
        # graph or the extractor itself.
        source = inspect.getsource(bt)
        assert "graph_app" not in source
        assert "analyze_document" not in source
        assert "_final_review.png" not in source

    def test_form_flow_is_imported_not_redefined(self, bt):
        source = inspect.getsource(bt)
        assert "FORM_FLOW = [" not in source
        assert bt.FORM_FLOW is pf.FORM_FLOW


# ==============================================================================
# 8. Profile progress — the 13 vs the 5
# ==============================================================================

class TestProfileProgress:
    def test_the_form_is_thirteen_fields_starting_with_name(self):
        assert len(pf.FORM_FIELDS) == 13
        assert pf.FORM_FIELDS[0] == "name"
        assert pf.FORM_FIELDS[-1] == "income"

    def test_routing_gates_on_five_but_progress_counts_thirteen(self):
        five = {"gender": "Male", "age": 22, "income": 45000,
                "caste": "OBC", "occupation": "Student"}
        p = pf.progress(five)
        assert p["usable"] is True      # graph.is_profile_complete would route
        assert p["answered"] == 5       # but the citizen is far from done
        assert p["total"] == 13

    def test_false_counts_as_an_answer(self):
        """"No, I don't have a BPL card" is an answer, and hybrid_rag reads it as one."""
        p = pf.progress({"below_poverty_line": False, "minority": False})
        assert p["answered"] == 2
        assert "below_poverty_line" not in p["missing"]

    def test_blank_and_none_do_not_count(self):
        p = pf.progress({"name": "", "gender": None, "caste": "OBC"})
        assert p["answered"] == 1
        assert "name" in p["missing"] and "gender" in p["missing"]

    def test_missing_is_in_form_order(self):
        missing = pf.progress({})["missing"]
        assert missing == pf.FORM_FIELDS


# ==============================================================================
# 9. /reset erases the browser profile
# ==============================================================================

class TestBrowserProfileWipe:
    """A reset that leaves portal cookies on disk is the one failure a citizen on a
    shared machine cannot see and cannot undo."""

    def test_both_naming_conventions_are_targeted(self):
        # graph.py launches automation as auto_<user_id>; browser_manager then prefixes
        # user_. Guessing only one convention deletes nothing and reports success.
        assert core_session.browser_session_ids("telegram_9") == ["auto_telegram_9", "telegram_9"]

    def test_it_deletes_the_directory_automation_actually_creates(self, tmp_path, monkeypatch):
        bm = pytest.importorskip("product_inference.browser_manager")
        monkeypatch.setattr(bm, "BROWSER_PROFILES_ROOT", str(tmp_path))
        monkeypatch.setattr(bm, "close_session", lambda _sid: None)

        live = tmp_path / "user_auto_telegram_9"
        live.mkdir()
        (live / "Cookies").write_text("portal-session")

        assert core_session.wipe_browser_profile("telegram_9") is True
        assert not live.exists()

    def test_it_leaves_other_citizens_alone(self, tmp_path, monkeypatch):
        bm = pytest.importorskip("product_inference.browser_manager")
        monkeypatch.setattr(bm, "BROWSER_PROFILES_ROOT", str(tmp_path))
        monkeypatch.setattr(bm, "close_session", lambda _sid: None)

        mine = tmp_path / "user_auto_telegram_9"
        theirs = tmp_path / "user_auto_telegram_77"
        mine.mkdir(); theirs.mkdir()

        core_session.wipe_browser_profile("telegram_9")
        assert not mine.exists()
        assert theirs.exists()

    def test_a_blank_user_id_cannot_wipe_the_root(self, tmp_path, monkeypatch):
        bm = pytest.importorskip("product_inference.browser_manager")
        monkeypatch.setattr(bm, "BROWSER_PROFILES_ROOT", str(tmp_path))
        monkeypatch.setattr(bm, "close_session", lambda _sid: None)

        (tmp_path / "user_auto_telegram_9").mkdir()
        assert core_session.wipe_browser_profile("") is False
        assert (tmp_path / "user_auto_telegram_9").exists()

    def test_the_session_is_closed_before_the_directory_goes(self, tmp_path, monkeypatch):
        """Chrome holds an exclusive lock on its user-data dir; deleting first fails
        mid-tree on Windows and leaves a corrupted profile behind."""
        bm = pytest.importorskip("product_inference.browser_manager")
        order = []
        monkeypatch.setattr(bm, "BROWSER_PROFILES_ROOT", str(tmp_path))
        monkeypatch.setattr(bm, "close_session", lambda sid: order.append(f"close:{sid}"))

        target = tmp_path / "user_auto_telegram_9"
        target.mkdir()

        real_rmtree = core_session.shutil.rmtree

        def spy(path, **kw):
            order.append("rmtree")
            return real_rmtree(path, **kw)

        monkeypatch.setattr(core_session.shutil, "rmtree", spy)
        core_session.wipe_browser_profile("telegram_9")

        assert order.index("close:auto_telegram_9") < order.index("rmtree")

    def test_a_missing_browser_manager_does_not_break_reset(self, monkeypatch, capsys):
        """Web-only and voice-only deployments never import Playwright. The DB wipe has
        already happened by this point and must not be undone by an ImportError."""
        import builtins
        real_import = builtins.__import__

        def no_playwright(name, globals=None, locals=None, fromlist=(), level=0):
            # `from product_inference import browser_manager` arrives as name=
            # "product_inference" with fromlist=("browser_manager",), so checking the
            # module name alone would let the import straight through.
            if "browser_manager" in name or any("browser_manager" in f for f in (fromlist or ())):
                raise ImportError("no playwright here")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", no_playwright)
        assert core_session.wipe_browser_profile("telegram_9") is False
        # Asserting the return value alone would pass even with the guard deleted, since
        # a user with no profile directory also returns False. Pin it to the branch.
        monkeypatch.undo()
        assert "Browser manager unavailable" in capsys.readouterr().out

    def test_every_surface_shares_one_reset(self):
        """There were four hand-maintained copies, and all four had forgotten the
        browser profile. Re-adding a local copy is how that happens again.

        Read as text, never imported: `product_inference/bot.py` calls `bot.run(...)` at
        module scope with no `__main__` guard, so importing it logs into Discord and
        blocks forever.
        """
        root = Path(__file__).resolve().parents[1] / "product_inference"
        for name in ("bot.py", "bot_audio.py"):
            source = (root / name).read_text(encoding="utf-8")
            assert "DELETE FROM user_profiles" not in source, f"{name} grew its own reset again"
            assert "core_session.do_reset" in source, f"{name} is not using the shared reset"
