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
        captured = {}

        def fake_save(user_id, state, data):
            captured.update({"user_id": user_id, "state": state, "data": data})

        payload = {
            "name": "faraz nezam", "gender": "Male", "age": "22",
            "income": "45,000", "caste": "OBC", "occupation": "student",
            "differently_abled": "true",
        }
        with patch("product_inference.db.update_user_state", fake_save):
            res = signed_in.post("/api/profile", json={"profile": payload})

        assert res.status_code == 200
        data = captured["data"]
        assert data["age"] == 22 and isinstance(data["age"], int)
        assert data["income"] == 45000
        assert data["name"] == "Faraz Nezam"
        assert data["differently_abled"] is True
        assert data["disability_percentage"] == 40
        assert captured["user_id"] == USER
        assert captured["state"] == "PROFILE_COMPLETE"

    def test_post_reports_field_errors(self, signed_in):
        res = signed_in.post("/api/profile", json={"profile": {
            "gender": "Male", "age": "not-a-number", "income": "45000",
            "caste": "OBC", "occupation": "Student",
        }})
        assert res.status_code == 400
        assert "age" in res.json()["errors"]

    def test_post_rejects_incomplete_profile(self, signed_in):
        res = signed_in.post("/api/profile", json={"profile": {"gender": "Male"}})
        assert res.status_code == 400
        assert res.json()["problems"]

    def test_post_rejects_non_object(self, signed_in):
        assert signed_in.post("/api/profile", json={"profile": "nope"}).status_code == 400


class TestStaticSurface:
    def test_signed_out_root_shows_the_telegram_instruction(self, client):
        body = client.get("/").text
        assert "/web" in body and "Telegram" in body

    def test_signed_in_root_shows_the_chat(self, signed_in):
        assert "composer" in signed_in.get("/").text

    def test_healthz(self, client):
        assert client.get("/healthz").json() == {"ok": True}
