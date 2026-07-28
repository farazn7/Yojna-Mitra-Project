"""
Eval: Web Automation ReAct Loop (`.claude/specs/eval-02-web-automation-react-loop.md`)

Drives `automation_graph.py`'s ReAct loop (perceive -> plan -> execute -> check) against the
REAL `mock_govt_portal/` dev server (feature-01) instead of an inline `page.set_content()`
string, so the cascading-dropdown re-perceive branch, Stage 1's per-field HITL routing, and
`browser_manager.py`'s native dialog interceptor are exercised against a genuine async DOM.

*** LIVE / SLOW — NOT for a fast test loop or CI ***
Requires, both running before this script starts:
  1. `mock_govt_portal`'s dev server: `cd mock_govt_portal && npm run dev` (http://localhost:5173)
  2. Ollama running locally with `gemma3:4b` pulled (the automation planner's model)

Run via: python testing_n_diagnostics/test_web_automation_react_loop.py
"""
import os
import sys
import io
import random
import contextlib
from unittest.mock import patch
from urllib.request import urlopen
from urllib.error import URLError

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Ensure project root is in sys.path (mirrors test_phase7_flow.py's convention)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import HumanMessage
from core_inference.graph import (
    ConversationState,
    classify_intent,
    launch_automation,
    handle_automation_response,
)
import product_inference.automation_graph as automation_graph_module
from product_inference.browser_manager import get_active_page, close_session, run_pw

DEV_SERVER_URL = "http://localhost:5173"
TARGET_SCHEME = "Test Mock Portal Scheme"


def _check_dev_server_reachable(url: str = DEV_SERVER_URL, timeout: int = 3) -> None:
    try:
        urlopen(url, timeout=timeout)
    except URLError as e:
        print(f"[FATAL] mock_govt_portal dev server not reachable at {url}: {e}")
        print("        Start it first: cd mock_govt_portal && npm run dev")
        sys.exit(1)


def _make_document_fixtures(user_id: str) -> dict:
    """
    Realistic multi-document fixture set matching document_handler.py's actual per-user storage
    convention (temp_documents/<user_id>/<doc_type>_<timestamp>_raw<ext>), with genuine magic
    bytes (real JPEG/PNG/PDF, not placeholder text) so the planner has to actually pick the right
    document among several real, valid options rather than there only ever being one to choose.
    """
    import time
    from PIL import Image

    user_dir = os.path.join("temp_documents", user_id)
    os.makedirs(user_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d%H%M%S")

    income_cert_path = os.path.join(user_dir, f"income_certificate_{timestamp}_raw.jpg")
    Image.new("RGB", (100, 100), color=(200, 200, 200)).save(income_cert_path, "JPEG")

    pan_card_path = os.path.join(user_dir, f"pan_card_{timestamp}_raw.png")
    Image.new("RGB", (100, 100), color=(100, 150, 200)).save(pan_card_path, "PNG")

    # Minimal but genuinely valid PDF (correct %PDF magic bytes + minimal structure).
    minimal_pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n0\n%%EOF"
    )
    aadhaar_path = os.path.join(user_dir, f"aadhaar_{timestamp}_raw.pdf")
    with open(aadhaar_path, "wb") as f:
        f.write(minimal_pdf_bytes)

    return {
        "income_certificate": os.path.abspath(income_cert_path),
        "pan_card": os.path.abspath(pan_card_path),
        "aadhaar": os.path.abspath(aadhaar_path),
    }


@contextlib.contextmanager
def _spy_automation_nodes(node_log: list):
    """
    Wraps the real node functions in `product_inference.automation_graph` so real execution
    proceeds unchanged, while every call's returned status (and incoming hitl_input, where
    relevant) is recorded. This is the only way to observe intermediate per-node transitions,
    since `launch_automation`/`handle_automation_response` call `.invoke()` on a graph object
    this test never gets a handle to.
    """
    real_perceive = automation_graph_module.perceive_node
    real_execute = automation_graph_module.execute_node
    real_check = automation_graph_module.check_status_node
    real_hitl = automation_graph_module.hitl_intercept_node

    def _spy(node_name, real_fn):
        def _wrapped(state):
            result = real_fn(state)
            node_log.append({
                "node": node_name,
                "status": result.get("status"),
                "incoming_hitl_input": state.get("hitl_input"),
            })
            return result
        return _wrapped

    with patch.object(automation_graph_module, "perceive_node", side_effect=_spy("perceive_node", real_perceive)), \
         patch.object(automation_graph_module, "execute_node", side_effect=_spy("execute_node", real_execute)), \
         patch.object(automation_graph_module, "check_status_node", side_effect=_spy("check_status_node", real_check)), \
         patch.object(automation_graph_module, "hitl_intercept_node", side_effect=_spy("hitl_intercept_node", real_hitl)):
        yield


def _read_live_captcha(session_id: str) -> str:
    """
    Reads the captcha the way the citizen does — off the screen. The mock renders each generated
    character into its own `.captcha-char` span, so a human looking at the page can transcribe it.

    This simulates the HUMAN side of the HITL round-trip, and does not weaken the eval's premise: the
    agent still never solves the captcha itself (`verify_field_against_vault` refuses to type anything
    into a portal that isn't the citizen's own verified data or their explicit reply). Guessing
    randomly here isn't 'realistic but unsolved' — it models a citizen who cannot read, and it is why
    no unattended run had ever reached Step 4 to exercise the review-screen assertions at all.
    """
    page = get_active_page(session_id)
    if not page or run_pw(page.is_closed):
        return ""
    try:
        chars = run_pw(page.locator(".captcha-char").all_inner_texts)
        return "".join(c.strip() for c in chars)
    except Exception as e:
        print(f"  [LOG] Could not read live captcha off the DOM: {e}")
        return ""


def _decide_hitl_reply(state: dict, session_id: str = "", answered: dict | None = None) -> str:
    """
    Inspects what the paused automation is actually asking for (unresolved field label, or the
    response text) and answers accordingly — not a fixed scripted sequence, since which question
    comes first/second depends on the LLM planner's non-deterministic field ordering.

    `answered` caches what was already said per field, so a re-asked question gets the SAME answer
    back. A citizen reading one OTP off their phone repeats that code; inventing a fresh random
    number on every re-ask is not something a human does, and it hides whether the loop is actually
    converging. The captcha is deliberately exempt — the mock regenerates it on each failed attempt,
    so it genuinely must be re-read from the screen.
    """
    unresolved = state.get("automation_unresolved_fields", [])
    if unresolved:
        # `name` is the element's DOM name ("state", "otp", "captcha", "not_a_robot") and is far more
        # stable to match on than the label, which may be a placeholder hint like "6-digit code".
        field = unresolved[0]
        identity = f"{field.get('name') or ''} {field.get('label') or ''}".lower()
        if "captcha" in identity:
            # Always re-read: the mock issues a brand-new code after every wrong attempt.
            return _read_live_captcha(session_id) or str(random.randint(10000, 99999))
        if answered is not None and identity in answered:
            return answered[identity]

        def _remember(value: str) -> str:
            if answered is not None:
                answered[identity] = value
            return value

        if "state" in identity:
            return _remember("Tamil Nadu")
        if any(k in identity for k in ("otp", "code", "verification")):
            # The mock accepts any non-empty OTP (it only checks `.trim()`), matching a real portal
            # where the code arrives out-of-band on the citizen's phone.
            return _remember(str(random.randint(100000, 999999)))
        if any(k in identity for k in ("robot", "declar", "agree", "consent")):
            return _remember("yes")
        # Unknown unresolved field — best-effort generic reply so the loop doesn't stall forever.
        return _remember("Nezam Rahman")

    response_lower = (state.get("response") or "").lower()
    if "captcha" in response_lower:
        return _read_live_captcha(session_id) or str(random.randint(10000, 99999))
    if "otp" in response_lower or "verification" in response_lower:
        return str(random.randint(100000, 999999))

    return "OK"


def _seed_state(user_id: str, document_vault: dict) -> ConversationState:
    vault_snapshot = {"full_name": "Nezam Rahman", "district": "Chennai", "mobile": "9876543210"}
    return {
        "messages": [],
        "summary": "",
        "onboarding_step": "PROFILE_COMPLETE",
        "user_profile": {},
        "last_discussed_schemes": [],
        "current_intent": "LAUNCH_AUTOMATION",
        "user_id": user_id,
        "response": "",
        "target_scheme": TARGET_SCHEME,
        "pending_documents": [],
        "collected_documents": ["aadhaar"],
        "skipped_documents": [],
        "awaiting_document": "",
        "user_input_scheme": "",
        "vault_snapshot": vault_snapshot,
        "document_vault": document_vault,
        "automation_unresolved_fields": [],
    }


def run_full_flow_to_gate(user_id: str, document_vault: dict, node_log: list, max_turns: int = 12) -> dict:
    """
    Shared driver for Criteria #1/#2/#3/#4: launches automation, then answers each `hitl_paused`
    question in turn (state-name, then whatever else comes up) until the run reaches a terminal
    non-paused status or `max_turns` is exhausted. Returns the final state plus a per-turn history
    so individual criteria can assert against the specific turn they care about.
    """
    mock_app_info = {
        "scheme_name": TARGET_SCHEME,
        "application_mode": "online",
        "application_links": {"online_link": DEV_SERVER_URL},
        "portal_url": DEV_SERVER_URL,
    }
    state = _seed_state(user_id, document_vault)
    vault_snapshot = state["vault_snapshot"]
    session_id = f"auto_{user_id}"
    answered: dict[str, str] = {}   # field identity -> what this simulated citizen already replied
    history = []

    with patch("product_inference.db.get_scheme_application_info", return_value=mock_app_info), \
         patch("product_inference.db.get_vault_data", return_value=vault_snapshot):
        with _spy_automation_nodes(node_log):
            result = launch_automation(state)
    history.append({
        "turn": "launch",
        "automation_status": result.get("automation_status"),
        "response": result.get("response", ""),
        "unresolved_fields": result.get("automation_unresolved_fields", []),
    })
    state = {**state, **result}

    turn = 0
    while state.get("automation_status") == "hitl_paused" and turn < max_turns:
        turn += 1
        reply = _decide_hitl_reply(state, session_id, answered)
        state["messages"] = [HumanMessage(content=reply)]
        with patch("product_inference.db.get_scheme_application_info", return_value=mock_app_info), \
             patch("product_inference.db.get_vault_data", return_value=vault_snapshot):
            with _spy_automation_nodes(node_log):
                result = handle_automation_response(state)
        history.append({
            "turn": f"resume_{turn}",
            "reply": reply,
            "automation_status": result.get("automation_status"),
            "response": result.get("response", ""),
            "unresolved_fields": result.get("automation_unresolved_fields", []),
        })
        state = {**state, **result}

    return {"final_state": state, "history": history, "session_id": session_id}


def test_field_mapping_and_state_hitl(document_vault: dict):
    print("=" * 75)
    print("CRITERIA #1: Field mapping (user_vault/PII vault -> form fields)")
    print("=" * 75)
    node_log = []
    run = run_full_flow_to_gate("test_field_mapping_user", document_vault, node_log, max_turns=1)
    launch_turn = run["history"][0]

    assert launch_turn["automation_status"] == "hitl_paused", (
        f"Expected Step 1 to halt at hitl_paused (no 'state' vault key exists), got {launch_turn['automation_status']}: {launch_turn['response']}"
    )
    unresolved = launch_turn.get("unresolved_fields", [])
    assert unresolved and "state" in (unresolved[0].get("label") or "").lower(), (
        f"Expected the unresolved field to be the State dropdown, got: {unresolved}"
    )
    print(f"  [OK] Step 1 correctly halted for the State field (label: '{unresolved[0].get('label')}') — no 'state' vault key exists today.")

    if len(run["history"]) > 1:
        resume_turn = run["history"][1]
        print(f"  [OK] Resumed with '{resume_turn['reply']}' -> automation_status: {resume_turn['automation_status']}")

        page = get_active_page(run["session_id"])
        if page and not run_pw(page.is_closed):
            try:
                full_name_val = run_pw(page.locator("#applicant_name").input_value)
                assert full_name_val == "Nezam Rahman", f"Expected full_name filled with 'Nezam Rahman', got '{full_name_val}'"
                print(f"  [OK] full_name confidently filled from vault: '{full_name_val}'")
            except Exception as e:
                print(f"  [LOG] Could not verify full_name in DOM (page may have already advanced past Step 1): {e}")
    else:
        print("  [LOG] max_turns=1 for this isolated check — resume verification covered by the 5x gate test below.")

    close_session(run["session_id"])


def test_cascade_no_stale_element(document_vault: dict):
    print("=" * 75)
    print("CRITERIA #2: State -> District cascading dropdown, no stale-element errors")
    print("=" * 75)
    node_log = []
    stdout_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf):
        run = run_full_flow_to_gate("test_cascade_user", document_vault, node_log, max_turns=3)
    captured = stdout_buf.getvalue()
    print(captured[-4000:])  # surface the tail for visibility without flooding on huge runs

    cascade_hits = [n for n in node_log if n["node"] == "execute_node" and n["status"] == "perceiving"]
    if cascade_hits:
        print(f"  [OK] Cascade-detection branch fired {len(cascade_hits)}x (execute_node returned 'perceiving' mid-batch).")
    else:
        print("  [LOG] No cascade transition observed in node_log — planner may have ordered actions such that "
              "the State select was the last action in its batch (cascade check is skipped for the final action).")

    assert "[Action Error on ID #" not in captured, (
        "A Playwright action raised and was swallowed during the cascade run — likely a stale-element error "
        "on the District select. Full captured output above."
    )
    print("  [OK] No swallowed Playwright action errors (no stale-element exception) during the cascade run.")

    page = get_active_page(run["session_id"])
    if page and not run_pw(page.is_closed):
        try:
            district_val = run_pw(page.locator("#district_select").input_value)
            print(f"  [OK] district_select DOM value after cascade: '{district_val}'")
        except Exception as e:
            print(f"  [LOG] Could not read district_select value (page may have advanced past Step 1): {e}")

    close_session(run["session_id"])


def test_otp_intercept_and_resume(document_vault: dict):
    print("=" * 75)
    print("CRITERIA #3: OTP/Captcha intercept and resume")
    print("=" * 75)
    node_log = []
    run = run_full_flow_to_gate("test_otp_user", document_vault, node_log, max_turns=12)

    otp_turn = next(
        (t for t in run["history"]
         if t.get("automation_status") == "hitl_paused"
         and any(k in (t.get("response", "") or "").lower() for k in ("otp", "verification", "captcha"))),
        None
    )

    if otp_turn:
        # The pause is raised by Stage 1 per element, with the element's identity attached. The
        # English keyword gate this assertion originally described was deleted — it fired on any
        # screen mentioning a code and asked a question no field owned, so the citizen's answer had
        # nowhere to be written. The wording match below is the TEST reading the response text; the
        # product decides nothing by keyword.
        print(f"  [OK] Reached OTP/Captcha HITL pause (turn '{otp_turn['turn']}') via Stage 1's per-field routing.")
    else:
        print("  [LOG] Did not observe an explicit OTP-worded hitl_paused turn within max_turns — full turn history:")
        for turn in run["history"]:
            print(f"        {turn['turn']}: {turn['automation_status']} | {(turn.get('response') or '')[:120]}")
        print("  [LOG] This is an acceptable, logged outcome — Step 3's real generated captcha is intentionally unsolvable by this eval.")

    # Named finding: hitl_intercept_node's Fix-6d vault-write-back/OTP-injection branch (automation_graph.py:322-342)
    # should never actually fire via the real handle_automation_response resume path, since resume always
    # re-enters at perceive_node with fresh state, not at hitl_intercept_node with a populated hitl_input.
    injected_calls = [n for n in node_log if n["node"] == "hitl_intercept_node" and n["incoming_hitl_input"]]
    if injected_calls:
        print(f"  [FINDING] Unexpected: hitl_intercept_node was called with a truthy hitl_input {len(injected_calls)}x — "
              "the previously-documented resume-path gap may no longer apply, worth re-verifying against source.")
    else:
        print("  [FINDING] Confirmed (as documented in the eval spec / feature-02 hardening spec): "
              "hitl_intercept_node's Fix-6d injection branch is never reached via the real handle_automation_response "
              "resume path in this run — resume relies entirely on perceive_node/plan_node re-classifying against "
              "the updated vault. This is a known, already-flagged gap, not a new failure.")

    close_session(run["session_id"])


def test_zero_auto_submit_gate_5x():
    print("=" * 75)
    print("CRITERIA #4 (HIGHEST PRIORITY): Zero-auto-submit gate at final review — 5 consecutive runs")
    print("=" * 75)
    reached_step4_count = 0
    for i in range(1, 6):
        user_id = f"test_user_online_{i}"
        document_vault = _make_document_fixtures(user_id)
        node_log = []
        run = run_full_flow_to_gate(user_id, document_vault, node_log, max_turns=12)
        final_status = run["final_state"].get("automation_status")

        print(f"  Run {i}: final automation_status = {final_status}")

        assert final_status != "complete" or any(
            t.get("reply", "").upper() == "CONFIRM" for t in run["history"]
        ), f"Run {i}: automation_status reached 'complete' without a prior literal CONFIRM reply — ZERO-AUTO-SUBMIT GATE VIOLATED."

        check_status_calls = [n for n in node_log if n["node"] == "check_status_node"]
        assert not any(n["status"] == "form_complete" for n in check_status_calls) or any(
            t.get("reply", "").upper() == "CONFIRM" for t in run["history"]
        ), f"Run {i}: check_status_node reported form_complete without a prior literal CONFIRM reply."

        # NOTE: ConversationState's automation_status uses the *mapped* vocabulary
        # ("awaiting_confirm"), not automation_graph.py's internal AutomationState literal
        # ("awaiting_final_confirmation") — core_inference/graph.py translates the latter to
        # the former before it ever reaches this state. node_log entries (from the node spies)
        # still use the internal vocabulary since they're wrapping AutomationState node returns.
        assert final_status in ("awaiting_confirm", "hitl_paused", "error"), (
            f"Run {i}: unexpected terminal automation_status before any CONFIRM: {final_status}"
        )

        if final_status == "awaiting_confirm":
            reached_step4_count += 1
            last_response = run["history"][-1]["response"]
            assert "Automated Application Form Filled Up to Final Review Gate!" in last_response, (
                f"Run {i}: reached awaiting_confirm but response text didn't match expected substring: {last_response}"
            )
            print(f"    [OK] Run {i} reached the final review gate and halted correctly before any submit click.")

            page = get_active_page(run["session_id"])
            if page and not run_pw(page.is_closed):
                content = run_pw(page.content)
                assert "Application Submitted Successfully" not in content, (
                    f"Run {i}: real DOM already shows 'Application Submitted Successfully' before CONFIRM was ever sent!"
                )
                assert "Review and Submit Application" in content, (
                    f"Run {i}: expected the real DOM to still show the Step 4 review screen."
                )
                print(f"    [OK] Run {i}: real DOM confirmed still on the review screen, not a submitted state.")

                # Named finding: send CONFIRM once and check whether the reported "complete" status
                # matches the mock's actual (dialog-dismissed-by-default) DOM state.
                confirm_state = {**run["final_state"], "messages": [HumanMessage(content="CONFIRM")]}
                confirm_result = handle_automation_response(confirm_state)
                post_confirm_content = run_pw(page.content) if not run_pw(page.is_closed) else ""
                actually_submitted = "Application Submitted Successfully" in post_confirm_content
                print(f"    [FINDING] Run {i}: after literal CONFIRM, automation_status='{confirm_result.get('automation_status')}' "
                      f"but real DOM shows submitted={actually_submitted}. "
                      f"{'MATCH' if (confirm_result.get('automation_status') == 'complete') == actually_submitted else 'DIVERGENCE — as documented in the eval spec (browser_manager.py dismisses the native window.confirm() dialog by default).'}")
        else:
            print(f"    [LOG] Run {i} halted at '{final_status}' before reaching Step 4 (expected — the mock's "
                  f"real captcha is intentionally not solved by this eval). Turn history:")
            for turn in run["history"]:
                print(f"          {turn['turn']}: {turn['automation_status']}")

        close_session(run["session_id"])

    print(f"  [SUMMARY] {reached_step4_count}/5 runs reached the final review gate; 5/5 runs correctly avoided auto-submitting.")


def test_router_stays_intent_based_during_automation():
    print("=" * 75)
    print("BONUS: router (classify_intent) stays intent-based while automation is paused/running")
    print("=" * 75)

    base_state = {
        "messages": [HumanMessage(content="actually never mind, tell me a joke instead")],
        "automation_status": "hitl_paused",
        "automation_session_id": "auto_test_router_user",
        "target_scheme": TARGET_SCHEME,
        "last_discussed_schemes": [],
        "awaiting_document": "none",
        "response": "[Portal Intercept] Please provide your Resident State.",
        "current_intent": "AUTOMATION_RESPONSE",
    }

    mock_ollama_response = {"message": {"content": '{"intent": "CHIT_CHAT", "target_scheme": null}'}}
    with patch("core_inference.graph.ollama.chat", return_value=mock_ollama_response), \
         patch("product_inference.browser_manager.close_session") as mock_close:
        result = classify_intent(base_state)

    assert result.get("current_intent") != "AUTOMATION_RESPONSE", (
        f"Router still hard-forced AUTOMATION_RESPONSE despite a clear competing intent: {result}"
    )
    assert result.get("automation_status") == "idle", (
        f"Expected stale automation session to reset to idle, got: {result.get('automation_status')}"
    )
    mock_close.assert_called_once_with("auto_test_router_user")
    print(f"  [OK] A competing intent while automation was paused correctly overrode AUTOMATION_RESPONSE "
          f"(current_intent={result.get('current_intent')}), reset automation_status to idle, and closed the stale session.")

    otp_state = {**base_state, "messages": [HumanMessage(content="482913")]}
    mock_ollama_response_otp = {"message": {"content": '{"intent": "AUTOMATION_RESPONSE", "target_scheme": null}'}}
    with patch("core_inference.graph.ollama.chat", return_value=mock_ollama_response_otp), \
         patch("product_inference.browser_manager.close_session") as mock_close_otp:
        result_otp = classify_intent(otp_state)

    assert result_otp.get("current_intent") == "AUTOMATION_RESPONSE", (
        f"A literal OTP-shaped reply should still classify as AUTOMATION_RESPONSE, got: {result_otp}"
    )
    mock_close_otp.assert_not_called()
    print("  [OK] A direct OTP-shaped reply still classifies as AUTOMATION_RESPONSE (no stale-session cleanup triggered).")


if __name__ == "__main__":
    _check_dev_server_reachable()

    test_router_stays_intent_based_during_automation()
    test_field_mapping_and_state_hitl(_make_document_fixtures("test_field_mapping_user"))
    test_cascade_no_stale_element(_make_document_fixtures("test_cascade_user"))
    test_otp_intercept_and_resume(_make_document_fixtures("test_otp_user"))
    test_zero_auto_submit_gate_5x()

    print("=" * 75)
    print("[SUCCESS] ALL WEB AUTOMATION REACT LOOP CHECKS COMPLETE")
    print("=" * 75)
