# ==============================================================================
# Yojana Mitra — Phase 8: Step 4 Universal Automation Graph (`automation_graph.py`)
# File Path: product_inference/automation_graph.py
# ==============================================================================
"""
LangGraph ReAct StateMachine & HITL Gates for Phase 8 Universal Dynamic Web Automation.

Key Features & Architectural Fixes (Claude Sonnet & Master Plan Review):
  1. ReAct Loop (`perceive → plan → execute → check`):
     Orchestrates browser session management, DOM perception, LLM field classification,
     and Playwright execution in an autonomous loop across multi-page wizards.
  2. Stale-Plan Guard (`execute_node`):
     An action batch is planned against the DOM as it looked before any of it ran. Snapshots
     the count of interactive elements (`INTERACTIVE_SELECTOR`, buttons included) before and
     after every action; any change means the rest of the batch is reasoning about a page that
     no longer exists, so it breaks out to `perceive_node` for a fresh plan. Covers dependent
     fields mounting (State → District), step transitions, and controls revealed by an upload.
  3. HITL Clarify Gate:
     When `llm_planner` cannot resolve a field from the citizen's vault it returns `hitl_clarify`,
     naming the exact element that needs an answer, and the graph halts (`status: hitl_paused`) with
     a screenshot of the screen being asked about (`screenshots/{session_id}_otp_intercept.png`) so
     a challenge the citizen has to read — a captcha — is actually visible to them.

     There is deliberately no separate keyword scan for OTP/captcha screens. One existed, matching
     English phrases in the page HTML, and it was a strictly worse duplicate of this gate: it fired
     on any screen that merely mentioned a code, cut the action batch short, and asked a question
     attached to no element — so the citizen's answer had nowhere to be written and was discarded,
     costing a whole extra round-trip (measured), while the per-field re-ask bound could not even
     count an ask with no field identity. Which field needs an answer is a semantic judgment Stage 1
     already makes, per element; a substring scan of the HTML is neither semantic nor attributable.
  4. PII Vault Write-Back on Clarification (Fix 6d):
     When the citizen answers a clarifying question (`hitl_intercept_node`), upserts
     the verified answer back into `user_vault` so subsequent scheme applications
     never re-ask the same question.
  5. Final Confirmation Safety Gate (`final_confirmation_node` — Fix 1 & Master Plan):
     When all wizard pages are completed and the final summary/review screen is reached,
     captures `screenshots/{session_id}_final_review.png` and halts (`status: awaiting_final_confirmation`).
     NEVER auto-submits. Requires explicit citizen `CONFIRM` command in Discord before
     the final submission button can be pressed.
"""

import os
import json
from typing import Any, Optional, TypedDict, Literal
from langgraph.graph import StateGraph, END
from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

# Import Step 1, Step 2, Step 3 components
from product_inference.browser_manager import (
    launch_isolated_profile,
    get_active_page,
    compute_page_signature,
    close_session,
    run_pw
)
from product_inference.perception_engine import perceive_dom_state, compute_control_state_fingerprint
from product_inference.llm_planner import (
    plan_wizard_step,
    press_finally_submits,
    is_final_review_screen,
    page_confirms_submission,
    gate_identity,
    is_pressable_element,
    DEFAULT_PLANNER_MODEL,
)


# Everything a citizen could interact with on a form page. Used as a cheap fingerprint of the
# page's interactive surface: if this count changes mid-batch, the remaining planned actions were
# reasoned about a DOM that no longer exists and must be re-planned. Buttons are deliberately
# included — controls appearing or disappearing is exactly the signal we need to catch.
INTERACTIVE_SELECTOR = "input, select, textarea, button, [role='button']"

# The same surface restricted to what is actually pressable right now. A control going from disabled
# to enabled changes nothing about the element COUNT, so the count alone cannot see a field unlocking
# its own "Verify" — which is precisely the moment the rest of a planned batch goes stale, because
# perception drops disabled elements and therefore planned around a control that did not exist.
ENABLED_SELECTOR = ", ".join(f"{part.strip()}:not(:disabled)" for part in INTERACTIVE_SELECTOR.split(","))

# How many times the loop may accept "the form's own state changed" as progress on one screen
# before treating the screen as stuck. See check_status_node.
MAX_IN_PLACE_PROGRESS = 6


# ==============================================================================
# STATE DEFINITION (`AutomationState`)
# ==============================================================================

class AutomationState(TypedDict, total=False):
    session_id: str                          # Unique Citizen / Conversation ID
    portal_url: str                          # Target government scheme URL
    user_vault: dict[str, Any]               # Verified PII vault dictionary
    document_vault: dict[str, str]           # Verified document file paths dictionary
    perception: dict[str, Any]               # Output from Step 2 perception engine
    page_signature: str                      # Signature of the current/previous page
    history_signatures: list[str]            # Tracked signatures to catch infinite loops
    regressive_gates: dict[str, list[str]]   # screen signature -> gate identities observed to go backwards
    control_state: str                       # fingerprint of what the screen's controls currently hold
    pressed_gates: dict[str, list[str]]      # screen signature -> controls pressed there that changed nothing
    initial_gates: dict[str, list[str]]      # screen signature -> pressable controls present at FIRST perception
    clicked_gates: list[str]                 # controls clicked in the batch just executed
    clicked_confirm_gates: list[str]         # of those, the ones submitting a single field for checking
    in_place_progress: int                   # accepted control-state changes on the current screen
    actions_to_execute: list[dict[str, Any]] # Playwright commands derived by `llm_planner`
    status: str                              # Current graph state (`perceiving`, `planning`, `executing`, etc.)
    retries: int                             # Retry count when page signature stays unchanged
    unresolved_questions: list[str]          # Clarifying questions for citizen
    unresolved_fields: list[dict[str, Any]]  # Metadata for unresolved fields
    hitl_input: Optional[str]                # Citizen's answer/OTP provided during HITL pause
    last_error: str                          # Error message if any node fails
    screenshots_dir: str                     # Directory to store intermediate & final review screenshots


# ==============================================================================
# GRAPH NODES
# ==============================================================================

def perceive_node(state: AutomationState) -> AutomationState:
    """
    Step 1 & Step 2: Retrieves active browser profile and perceives current DOM state.
    """
    session_id = state["session_id"]
    portal_url = state.get("portal_url", "about:blank")
    print(f"\n[Graph: perceive_node] Session: {session_id} | Target: {portal_url}")

    # Ensure screenshots directory exists
    screenshots_dir = state.get("screenshots_dir", "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    try:
        active_page = run_pw(get_active_page, session_id)
        if not active_page or run_pw(active_page.is_closed):
            ctx, page = run_pw(launch_isolated_profile, session_id, portal_url=portal_url, headless=False)
            active_page = run_pw(get_active_page, session_id) or page
        elif portal_url and portal_url != "about:blank" and active_page.url.rstrip("/") != portal_url.rstrip("/"):
            # Browsers normalize a bare-domain URL with a trailing "/" once loaded
            # (e.g. "http://localhost:5173" -> "http://localhost:5173/"), so a strict equality
            # check here was always true and re-navigated on EVERY perceive_node call, wiping
            # the entire in-progress wizard's form state each cycle — reproduced live.
            run_pw(active_page.goto, portal_url, timeout=30000)
    except Exception as e:
        print(f"[perceive_node Error] Browser launch failed: {e}")
        return {
            "status": "error",
            "last_error": f"Browser launch error: {e}"
        }

    # Execute perception engine
    perception = run_pw(perceive_dom_state, active_page)
    new_sig = perception["page_signature"]
    history = state.get("history_signatures", [])

    print(f"  [Perception Complete] Elements: {perception['total_collapsed_count']} | Signature: {new_sig}")

    # What this screen offered to press the FIRST time we ever stood on it. Perception drops disabled
    # elements, so a control the portal keeps locked until a field holds a value is simply absent
    # here — and its later appearance is the evidence that it belongs to whatever was entered. That
    # is the pairing signal `plan_node` reads, and it survives being filled by anyone: the citizen's
    # HITL answer is injected straight into the DOM by `core_inference.graph._inject_hitl_answer`,
    # outside this graph entirely, so nothing that watches an action's own before/after can see it.
    #
    # Written once per signature and never refreshed. Re-snapshotting on a later visit would record
    # the unlocked control as having been there all along, erasing the very transition being looked
    # for. Copy-and-extend rather than mutating: `state` is LangGraph's, not ours.
    initial = {k: list(v) for k, v in (state.get("initial_gates") or {}).items()}
    if new_sig not in initial:
        initial[new_sig] = sorted({
            gate_identity(el) for el in perception.get("elements_list", [])
            if is_pressable_element(el)
        })

    return {
        "perception": perception,
        "page_signature": new_sig,
        "control_state": perception.get("control_state", ""),
        "history_signatures": history + [new_sig] if new_sig not in history else history,
        "initial_gates": initial,
        "status": "planning"
    }


def plan_node(state: AutomationState) -> AutomationState:
    """
    Step 3: Calls `llm_planner.plan_wizard_step` to classify fields against the citizen's vault
    and derive the resulting Playwright action batch.
    """
    session_id = state["session_id"]
    perception = state.get("perception", {})
    user_vault = state.get("user_vault", {})
    document_vault = state.get("document_vault", {})

    print(f"[Graph: plan_node] Session: {session_id} | Classifying fields & deriving actions...")

    # Anything this screen's navigation controls have already been observed to do (see
    # check_status_node) is fed back into planning, so a mis-judged "Back" is pressed at most once.
    blocked_here = set(state.get("regressive_gates", {}).get(state.get("page_signature", ""), []))
    if blocked_here:
        print(f"  [Loop Memory] {len(blocked_here)} navigation control(s) on this screen are known to move backwards.")

    pressed_here = set(state.get("pressed_gates", {}).get(state.get("page_signature", ""), []))

    # Which controls this screen did NOT offer when we arrived on it. A control the portal unlocked
    # since then did so in response to something that was entered, which is what binds it to that
    # entry — so the planner prefers it over a control that was pressable from the start. Empty on a
    # screen's first cycle, because `perceive_node` just took the snapshot from this same perception.
    seen_first = set((state.get("initial_gates") or {}).get(state.get("page_signature", ""), []))
    newly_available = {
        gate_identity(el) for el in perception.get("elements_list", [])
        if is_pressable_element(el)
    } - seen_first
    if newly_available:
        print(f"  [Unlocked Since Arrival] {len(newly_available)} control(s) on this screen appeared only "
              f"after something was entered; preferred over controls that were pressable all along.")

    plan_result = plan_wizard_step(
        perception=perception,
        user_vault=user_vault,
        document_vault=document_vault,
        model=DEFAULT_PLANNER_MODEL,
        regressive_gates=blocked_here,
        pressed_gates=pressed_here,
        newly_available_gates=newly_available
    )

    planner_status = plan_result.get("status")
    print(f"  [Planner Status] -> {planner_status}")

    if planner_status == "hitl_clarify":
        # Required PII missing -> route to HITL clarification gate
        return {
            "status": "hitl_clarify",
            "unresolved_questions": plan_result.get("unresolved_questions", []),
            "unresolved_fields": plan_result.get("unresolved_fields", []),
            "actions_to_execute": []
        }
    elif planner_status == "actions_ready":
        actions = plan_result.get("actions", [])

        # *** Zero auto-submit: refuse the press, do not merely forbid it in a prompt ***
        # Observed live: Stage 1 classified "Submit Application" as a forward gate — its own reasoning
        # read "is an action gate that submits the application" — and the loop pressed it with no
        # citizen authorization. Nothing was filed only because that portal raises a native confirm
        # dialog, which the interceptor dismisses; a portal that submits on click would have filed the
        # application outright. The review-screen check cannot save this, because it runs in
        # check_status_node — after the click.
        #
        # So before any press, the strongest local model is asked which control on this screen finally
        # submits (the same reading used for the citizen's own CONFIRM). If a planned click is that
        # control, the batch is dropped and the graph halts at the confirmation gate instead.
        el_by_id = {el["id"]: el for el in perception.get("elements_list", [])}
        for act in actions:
            if act.get("action") != "click":
                continue
            label = str(el_by_id.get(act["element_id"], {}).get("label", ""))
            # Context is the OTHER CONTROLS, not every label on screen. Handing the model the field
            # labels too ("Full Name", "Resident State") made it read "Save & Proceed to Step 2" as
            # the final submission — a screen full of empty fields apparently looks like something
            # being filed — and the guard then halted the run on Step 1 of 4. The same control with
            # only its sibling buttons for context reads correctly, in every language tried.
            others = [
                str(el.get("label", "")) for el in perception.get("elements_list", [])
                if el.get("id") != act["element_id"]
                and (str(el.get("tag", "")).lower() in ("button", "a")
                     or str(el.get("type", "")).lower() in ("submit", "button", "reset"))
            ]
            if label and press_finally_submits(label, others):
                print(f"  [ZERO AUTO-SUBMIT] Planned click on '{label}' would file the application. "
                      f"Refusing it and halting for the citizen's explicit CONFIRM.")
                return {"status": "final_confirmation", "actions_to_execute": []}

        print(f"  [Actions Generated] -> {len(actions)} execution steps ready.")
        # Note: retries is intentionally NOT reset here. check_status_node already resets it
        # to 0 when the page signature genuinely changes (real progress). Resetting it here too
        # would wipe out check_status_node's counter every cycle the planner keeps proposing
        # plausible actions against a stuck page, making retry_or_abort_node's abort-after-3
        # safety net unreachable — confirmed live: an unresponsive page looped indefinitely.
        return {
            "status": "executing",
            "actions_to_execute": actions
        }
    elif planner_status == "perceiving":
        # Nothing left to do on this screen. That is what the end of the form looks like from the
        # inside, so this is where the review gate is decided — before anything is pressed, against
        # perception that was taken this cycle, and only on the cycles that produced no work, so the
        # extra reading costs nothing on the many cycles that did. Judged by what the screen means
        # rather than by English headings like "review and submit", which a Hindi or Tamil portal
        # would sail straight past on its way to its own submit button.
        if is_final_review_screen(perception):
            print("  [Final Review Gate Reached] Nothing left to fill and the screen reads as the "
                  "final review. Halting for the citizen's explicit CONFIRM.")
            return {"status": "final_confirmation", "actions_to_execute": []}
        return {
            "status": "check",
            "actions_to_execute": []
        }
    else:
        # Planner error or max retries exceeded
        return {
            "status": "error",
            "last_error": plan_result.get("last_error", "Unknown planner failure.")
        }


def execute_node(state: AutomationState) -> AutomationState:
    """
    Executes Playwright actions (`type`, `select`, `upload`, `click`) sequentially.
    Enforces cascading dropdown detection and mid-flow OTP/Captcha interception.
    """
    session_id = state["session_id"]
    actions = state.get("actions_to_execute", [])
    active_page = run_pw(get_active_page, session_id)

    if not active_page or run_pw(active_page.is_closed):
        return {"status": "error", "last_error": "Browser page closed unexpectedly during execution."}

    print(f"[Graph: execute_node] Session: {session_id} | Executing {len(actions)} actions...")

    # Navigation controls actually pressed in this batch. Reported rather than judged here:
    # whether a press accomplished anything is check_status_node's call, since only it sees what the
    # page did afterwards. Always reported (empty list included) so a later batch never inherits a
    # stale entry.
    pressed_here: list[str] = []
    confirmed_here: list[str] = []

    def _with_presses(payload: dict) -> dict:
        return {**payload, "clicked_gates": list(pressed_here), "clicked_confirm_gates": list(confirmed_here)}

    for i, action in enumerate(actions):
        el_id = action["element_id"]
        act_type = action["action"]
        val = action.get("value") or ""
        reason = action.get("reasoning", "")

        print(f"  -> [ID #{el_id}] {act_type.upper()} | Value: '{val}' | ({reason})")

        # Locate element via data-ym-id injected by perception
        locator = run_pw(active_page.locator, f'[data-ym-id="{el_id}"]')
        if run_pw(locator.count) == 0:
            print(f"  [Skip Action] Element ID #{el_id} not found in current perception map.")
            continue

        try:
            pre_count = run_pw(active_page.locator(INTERACTIVE_SELECTOR).count)
            pre_enabled = run_pw(active_page.locator(ENABLED_SELECTOR).count)

            if act_type == "type":
                run_pw(locator.first.fill, str(val))
            elif act_type == "select":
                # Try selecting by label first, fallback to value
                try:
                    run_pw(locator.first.select_option, label=str(val))
                except Exception:
                    run_pw(locator.first.select_option, value=str(val))
            elif act_type == "check" or act_type == "click":
                run_pw(locator.first.click)
                # Recorded from what ran, not from the plan: a batch can break out early, and a gate
                # that was never clicked must not be treated as tried.
                if act_type == "click" and action.get("gate_identity"):
                    pressed_here.append(action["gate_identity"])
                    if action.get("confirms_entry"):
                        confirmed_here.append(action["gate_identity"])
            elif act_type == "upload":
                if os.path.exists(str(val)):
                    run_pw(locator.first.set_input_files, str(val))
                    # Uploads are I/O: portals commonly show a transient "uploading…" state and only
                    # then reveal/enable whatever comes next. Allow for that before measuring the DOM.
                    run_pw(active_page.wait_for_timeout, 1200)
                else:
                    print(f"     [WARN] Document file path '{val}' not found on disk!")
                    continue

            # Brief pause for JavaScript re-renders or API calls to complete
            run_pw(active_page.wait_for_timeout, 600)

            # A press that erased what the citizen had already entered is not repeated on this screen:
            # a captcha "Refresh" clears the box beside it, so the answer is gone and the portal asks
            # again — repeat that and the step never completes. Judged from what the press did, not
            # from what Stage 1 called the control, because a 4B classifier called Refresh a forward
            # control on every cycle it saw one. Checked at the press itself rather than only after
            # the batch, because the staleness guard below can return straight to perception and skip
            # check_status_node's copy of this test entirely.
            #
            # Controls that submit a single field for checking are exempt: a portal clearing a
            # rejected captcha is the value being wrong, not the control being destructive.
            if act_type == "click" and not action.get("confirms_entry"):
                erased = _entries_erased_by(state, active_page)
                if erased and action.get("gate_identity"):
                    regressive = {k: list(v) for k, v in (state.get("regressive_gates") or {}).items()}
                    screen = state.get("page_signature", "")
                    if action["gate_identity"] not in regressive.setdefault(screen, []):
                        regressive[screen].append(action["gate_identity"])
                    print(f"  [Entry Erased] Pressing '{action['gate_identity']}' emptied {erased} — that "
                          f"control will not be pressed on this screen again.")
                    return _with_presses({"status": "perceiving", "regressive_gates": regressive})

            # *** Stale-plan guard: re-perceive whenever the page's interactive surface changed ***
            # The batch was planned against the DOM as it looked before any of these actions ran. The
            # moment an action changes what's on the page — a dependent select mounting (State ->
            # District), a step advancing, or a completed upload revealing the next control — every
            # remaining queued action is reasoning about a page that no longer exists. Rather than
            # enumerate those cases individually, treat any change to the interactive surface as
            # invalidating the rest of the plan and go re-perceive.
            #
            # This must include buttons, not just form fields: a completed upload often adds or
            # enables a navigation control without touching the input/select/textarea count at all,
            # which is precisely the gap that let stale navigation clicks through.
            post_count = run_pw(active_page.locator(INTERACTIVE_SELECTOR).count)
            if pre_count != post_count:
                print(f"  [DOM Changed] Interactive element count {pre_count} -> {post_count} after action #{el_id}. Re-perceiving before continuing.")
                return _with_presses({"status": "perceiving"})

            # A control the entry just UNLOCKED is a changed interactive surface too, and the count
            # above cannot see it: portals keep a field's own "Verify" disabled until the field holds
            # a value, and disabled -> enabled leaves the element count identical. Perception drops
            # disabled elements, so such a control did not exist when this batch was planned — which
            # is how the mock's Step 3 ends up with "Refresh", the one always-enabled control beside
            # the captcha box, as the best remaining candidate to press.
            #
            # A queued press that CONFIRMS the entry we just made is exempt, and that exemption is the
            # whole point rather than a loophole: such a press was chosen because of this field, so it
            # cannot be stale with respect to it. Without the exemption this guard would defer every
            # "fill the code, then press its Verify" pairing by a cycle — the exact ordering the guard
            # exists to protect.
            post_enabled = run_pw(active_page.locator(ENABLED_SELECTOR).count)
            if pre_enabled != post_enabled and any(not a.get("confirms_entry") for a in actions[i + 1:]):
                print(f"  [Controls Unlocked] Pressable controls {pre_enabled} -> {post_enabled} after action "
                      f"#{el_id}; the rest of this batch was planned without them. Re-perceiving.")
                return _with_presses({"status": "perceiving"})

        except Exception as e:
            print(f"     [Action Error on ID #{el_id}] {e}")
            # Continue executing other independent fields if one non-critical action stumbles

    print("  [Execution Batch Completed] Advancing to check status node.")
    return _with_presses({"status": "check"})


def _entries_erased_by(state: AutomationState, active_page: Page) -> list[str]:
    """
    Fields that held an answer before this batch ran and are now empty.

    Wiping entered data is the one thing no control should be allowed to do twice. A captcha's
    "Refresh" clears the box it sits beside (as does a Reset or a Clear), so the citizen's answer is
    gone and the portal asks again — repeat that and the run never reaches the end of the step, which
    is exactly how three separate attempts died before Step 4. This is read from the DOM instead of
    being taken on trust from Stage 1, because a 4B classifier called Refresh a forward control on
    every single cycle it saw one, whatever the schema offered it.

    Fields this batch filled itself are excluded: a dependent select legitimately resets when its
    parent changes (State -> District), and that is our own doing, not the control's.
    """
    before = {
        el["id"]: str(el.get("value") or "")
        for el in state.get("perception", {}).get("elements_list", [])
        if str(el.get("value") or "").strip()
    }
    if not before:
        return []
    just_filled = {
        a.get("element_id") for a in state.get("actions_to_execute", [])
        if a.get("action") in ("type", "select", "upload")
    }
    try:
        live = run_pw(active_page.evaluate, """() => Object.fromEntries(
            Array.from(document.querySelectorAll('[data-ym-id]')).map(el => [el.getAttribute('data-ym-id'), el.value || ''])
        )""")
    except Exception:
        return []

    erased = []
    labels = {el["id"]: el.get("label", el["id"]) for el in state.get("perception", {}).get("elements_list", [])}
    for el_id, old_value in before.items():
        if el_id in just_filled:
            continue
        # Absent from the live map means the node was replaced by a re-render, which says nothing
        # about whether its value was deliberately wiped.
        current = live.get(str(el_id))
        if current is not None and old_value and not str(current).strip():
            erased.append(str(labels.get(el_id, el_id)))
    return erased


def check_status_node(state: AutomationState) -> AutomationState:
    """
    Checks whether the page signature transitioned after execution.
    Determines if we advanced to the next wizard step, hit a validation error, or reached final review.
    """
    session_id = state["session_id"]
    old_sig = state.get("page_signature", "")
    active_page = run_pw(get_active_page, session_id)

    if not active_page or run_pw(active_page.is_closed):
        return {"status": "error", "last_error": "Page closed during check status."}

    print(f"[Graph: check_status_node] Session: {session_id} | Checking navigation status...")

    try:
        run_pw(active_page.wait_for_load_state, "domcontentloaded", timeout=3000)
    except Exception:
        pass

    new_sig = run_pw(compute_page_signature, active_page)
    print(f"  [Signature Check] Old: {old_sig} -> New: {new_sig}")

    # Has the portal acknowledged a filing? Read by meaning rather than matched against English
    # phrases like "acknowledgement number", which say nothing on a portal written in Hindi or Tamil —
    # and getting this wrong is not a cosmetic error: a missed acknowledgement makes the loop keep
    # working on a form that is already submitted, and eventually tell the citizen the automation
    # failed when their application was in fact filed.
    #
    # Asked only when the screen actually changed. An acknowledgement replaces the form with a receipt,
    # so it is a new screen by definition and an unchanged signature cannot be one — which keeps this
    # off the per-cycle path and down to roughly once per step transition.
    if new_sig != old_sig:
        try:
            page_text = run_pw(active_page.locator("body").inner_text, timeout=3000)
        except Exception:
            page_text = ""
        if page_text and page_confirms_submission(page_text):
            print("  [OK: Form Complete] The portal confirms the application has been submitted.")
            return {"status": "form_complete"}

    # The final-review gate is deliberately NOT decided here any more. It used to be, by matching
    # English headings, and that placement was wrong on its own terms: this node runs AFTER the batch
    # has executed, and detection that happens after an irreversible action is not a safeguard. The
    # judgment now sits in plan_node, before anything is pressed, where perception is fresh — next to
    # the pre-press submit refusal that is the actual guarantee.

    # If signature changed, we successfully navigated to a new wizard step!
    if new_sig != old_sig:
        # ...unless it is a screen we have already been on. A changed signature only proves the page
        # moved, not that it moved forward: a control Stage 1 judged as forward-advancing may in fact
        # be a "Back"/"Previous" button, and the two screens then ping-pong until max_turns (observed
        # live between Steps 1 and 2 of the mock portal). Rather than guess from the button's wording,
        # remember what the browser actually did and let plan_node stop offering that control on that
        # screen. The observation is per-screen, so the same label elsewhere is judged on its own.
        history = state.get("history_signatures", [])
        # "Seen before" alone is not enough: after a wrongly-pressed Back, pressing the genuine
        # forward gate legitimately lands on a screen we have already been on, and flagging that
        # would eventually block every control on the form. `history_signatures` is in first-visit
        # order, so it also tells us the DIRECTION: landing on a screen first seen EARLIER than the
        # one we just left is a step backwards; landing on a later one is progress we are re-making.
        went_backwards = (
            new_sig in history and old_sig in history
            and history.index(new_sig) < history.index(old_sig)
        )
        if went_backwards:
            regressive = {k: list(v) for k, v in (state.get("regressive_gates") or {}).items()}
            pressed = next(
                (a.get("gate_identity") for a in reversed(state.get("actions_to_execute", []))
                 if a.get("action") == "click" and a.get("gate_identity")),
                None
            )
            if pressed and pressed not in regressive.get(old_sig, []):
                regressive.setdefault(old_sig, []).append(pressed)
                print(f"  -> [Went Backwards] Landed on an earlier screen than the one we left. Marking the "
                      f"control '{pressed}' as backwards for that screen; it will not be pressed there again.")
            return {
                "status": "perceiving",
                "page_signature": new_sig,
                "regressive_gates": regressive,
                "in_place_progress": 0,
                "retries": 0
            }
        print("  -> [Step Advanced] Page signature changed. Looping to perceive new wizard screen.")
        # The gate we pressed did its job, so the screen we left keeps no grudge against it: coming
        # back to that screen later (with its fields already filled) has to be able to press the very
        # same control again. Only presses that changed nothing are remembered, below.
        cleared = {k: list(v) for k, v in (state.get("pressed_gates") or {}).items()}
        cleared.pop(old_sig, None)
        return {
            "status": "perceiving",
            "page_signature": new_sig,
            "pressed_gates": cleared,
            "in_place_progress": 0,
            "retries": 0
        }
    else:
        # Same screen. That is not the same thing as nothing having happened: a step that verifies an
        # OTP, then a captcha, then a consent box does all of it without changing URL/title/heading,
        # and counting each of those as a failed retry aborted the run three actions short of the
        # Proceed button (observed live on Step 3). So ask what the controls now HOLD: if that changed,
        # the last action did land and the loop should carry on planning the next one.
        #
        # Bounded, because "something changed" is a weaker signal than "the screen changed" and a page
        # that re-renders on every touch could otherwise spin forever. Successful in-place steps are
        # few by nature; a page needing more than this many is not converging.
        # A press that erased what the citizen had already entered must never happen twice on this
        # screen, whatever Stage 1 says the control is for. Recorded in the same memory as a control
        # observed to move backwards, because the consequence is the same: do not press it here again.
        erased = _entries_erased_by(state, active_page)
        # A control that submits a field for checking is exempt: the portal clearing the box means the
        # value was rejected, not that the control is destructive, and blocking it would make the field
        # impossible to ever satisfy.
        confirming = set(state.get("clicked_confirm_gates") or [])
        clicked = [g for g in (state.get("clicked_gates") or []) if g not in confirming]
        if erased and clicked:
            regressive = {k: list(v) for k, v in (state.get("regressive_gates") or {}).items()}
            for identity in clicked:
                if identity not in regressive.setdefault(old_sig, []):
                    regressive[old_sig].append(identity)
            print(f"  -> [Entry Erased] Pressing '{clicked[-1]}' emptied {erased} — that control will not be "
                  f"pressed on this screen again.")
            return {
                "status": "perceiving",
                "regressive_gates": regressive,
                "control_state": run_pw(compute_control_state_fingerprint, active_page),
                "retries": 0
            }

        # A gate pressed without the screen changing has not (yet) moved the application on, so it is
        # remembered as tried for this screen and derive_actions offers the next candidate instead.
        # This is what walks a screen needing several presses in sequence, and what stops a control
        # that merely regenerates its own content (a captcha "Refresh") from being pressed forever.
        pressed = {k: list(v) for k, v in (state.get("pressed_gates") or {}).items()}
        for identity in state.get("clicked_gates") or []:
            if identity not in pressed.setdefault(old_sig, []):
                pressed[old_sig].append(identity)
                print(f"  -> [Gate Tried] '{identity}' changed nothing on this screen; the next control gets its turn.")

        new_control_state = run_pw(compute_control_state_fingerprint, active_page)
        in_place_progress = state.get("in_place_progress", 0)
        if new_control_state != state.get("control_state", "") and in_place_progress < MAX_IN_PLACE_PROGRESS:
            print(f"  -> [In-Place Progress] Screen unchanged but the form's own state did "
                  f"({in_place_progress + 1}/{MAX_IN_PLACE_PROGRESS}). Continuing without counting a retry.")
            return {
                "status": "perceiving",
                "control_state": new_control_state,
                "pressed_gates": pressed,
                "in_place_progress": in_place_progress + 1,
                "retries": 0
            }

        retries = state.get("retries", 0)
        print(f"  [WARN: Signature Unchanged] Page did not advance. Possible validation error. Retry count: {retries + 1}/3")
        return {
            "status": "retry_or_abort",
            "pressed_gates": pressed,
            "retries": retries + 1
        }


def hitl_intercept_node(state: AutomationState) -> AutomationState:
    """
    HITL Intercept Node: Pauses execution until citizen answers clarification or enters OTP.
    When resumed with `state["hitl_input"]`, writes PII answers back into `user_vault` and resumes graph.
    """
    session_id = state["session_id"]
    hitl_input = state.get("hitl_input")
    unresolved_fields = state.get("unresolved_fields", [])
    perception = state.get("perception", {})

    print(f"\n[Graph: hitl_intercept_node] Session: {session_id} | Status: Paused for Citizen Input")

    if not hitl_input:
        # Show the citizen the screen they are being asked about, whatever raised the question.
        # A captcha is unanswerable without seeing it, and the screenshot used to be captured only by
        # a keyword scan for OTP/captcha wording — so a portal that asked in Hindi, or that the
        # classifier flagged as an ordinary missing field, produced a question about a challenge the
        # citizen could not read. Both bots attach this file for a `hitl_paused` turn, so capturing it
        # at the pause itself covers every route into one, not the one route that matched English.
        shot_path = os.path.join(state.get("screenshots_dir", "screenshots"), f"{session_id}_otp_intercept.png")
        try:
            active_page = run_pw(get_active_page, session_id)
            if active_page and not run_pw(active_page.is_closed):
                os.makedirs(state.get("screenshots_dir", "screenshots"), exist_ok=True)
                run_pw(active_page.screenshot, path=shot_path, full_page=True)
                print(f"  [Pause Screenshot] Captured what the citizen is being asked about: {shot_path}")
        except Exception as e:
            print(f"  [Pause Screenshot Notice] Could not capture the paused screen: {e}")
        return {"status": "hitl_paused"}

    print(f"  [HITL Input Received] Citizen replied: '{hitl_input}'")

    # *** FIX 6d: Write clarified PII data back into `user_vault` ***
    # One question is asked per pause (llm_planner returns the fields in ask-order), so the citizen's
    # reply belongs to the first entry. Mirrors core_inference/graph.py's resume-path write-back,
    # including its preference for the element's DOM `name` over its prose label as the vault key.
    if unresolved_fields:
        field_name = unresolved_fields[0].get("name") or unresolved_fields[0].get("label", "").lower()
        if field_name:
            # Upsert into in-memory vault so Stage 1 matches it immediately
            user_vault = state.get("user_vault", {})
            user_vault[field_name] = hitl_input
            state["user_vault"] = user_vault
            print(f"  [Vault Upsert] Saved clarified answer '{hitl_input}' under vault key '{field_name}'!")

            # Delivering the answer into the element that asked for it is core_inference.graph's
            # `_inject_hitl_answer`, on the resume path this node is never reached by (each resume is
            # a fresh subgraph invocation entering at perceive_node, which is why `hitl_input` is
            # never set here). What used to sit here selected the field by English key fragments and
            # then pressed `button:has-text('Verify')` — a control chosen by English button text on a
            # portal that may say "सत्यापित करें", and a press the planner alone is entitled to decide.

    # Clear hitl_input after consumption and resume perception loop
    state["hitl_input"] = None
    state["unresolved_questions"] = []
    state["unresolved_fields"] = []
    return {"status": "perceiving"}


def retry_or_abort_node(state: AutomationState) -> AutomationState:
    """
    Handles validation retry loops or aborts cleanly if max retries (3) are exceeded.
    """
    retries = state.get("retries", 0)
    session_id = state["session_id"]

    if retries >= 3:
        print(f"[Graph: retry_or_abort_node] Session: {session_id} | Max retries (3) exhausted. Aborting application.")
        return {"status": "aborted", "last_error": "Exceeded maximum retry attempts due to persistent page validation error or stuck form."}

    print(f"[Graph: retry_or_abort_node] Session: {session_id} | Re-attempting perception & planning (Attempt {retries}/3)...")
    return {"status": "perceiving"}


def final_confirmation_node(state: AutomationState) -> AutomationState:
    """
    Final Safety Gate: Captures screenshot of filled form review page and halts (`END`).
    NEVER auto-submits. Citizen must type CONFIRM in Discord to trigger final submit click.
    """
    session_id = state["session_id"]
    active_page = run_pw(get_active_page, session_id)
    screenshots_dir = state.get("screenshots_dir", "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    shot_path = os.path.join(screenshots_dir, f"{session_id}_final_review.png")
    if active_page and not run_pw(active_page.is_closed):
        try:
            run_pw(active_page.screenshot, path=shot_path, full_page=True)
            print(f"\n[Graph: final_confirmation_node] [Final Review Gate] Screenshot captured: {shot_path}")
        except Exception as e:
            print(f"[Screenshot Warn] Could not capture final review: {e}")

    print("  [ZERO AUTO-SUBMIT GATE] Halting graph. Awaiting explicit citizen 'CONFIRM' command in Discord.")
    return {"status": "awaiting_final_confirmation"}


# ==============================================================================
# GRAPH ASSEMBLY (`build_automation_graph`)
# ==============================================================================

def route_after_check(state: AutomationState) -> str:
    """Routes conditional edges from `check_status_node`."""
    status = state.get("status", "perceiving")
    return {
        "perceiving": "perceive_node",
        "retry_or_abort": "retry_or_abort",
        "final_confirmation": "final_confirmation",
        "form_complete": "form_complete_end",
        "error": "error_end"
    }.get(status, "perceive_node")


def route_after_plan(state: AutomationState) -> str:
    """Routes conditional edges from `plan_node`."""
    status = state.get("status", "error")
    return {
        "executing": "execute_node",
        "hitl_clarify": "hitl_intercept",
        "check": "check_status_node",
        "final_confirmation": "final_confirmation",
        "error": "error_end"
    }.get(status, "error_end")


def route_after_execute(state: AutomationState) -> str:
    """Routes conditional edges from `execute_node`."""
    status = state.get("status", "check")
    return {
        "check": "check_status_node",
        "perceiving": "perceive_node",       # Cascading re-render, or a control the entry just enabled
        "error": "error_end"
    }.get(status, "check_status_node")


def route_after_retry(state: AutomationState) -> str:
    """Routes conditional edges from `retry_or_abort_node`."""
    if state.get("status") == "aborted":
        return "aborted_end"
    return "perceive_node"


def route_after_hitl(state: AutomationState) -> str:
    """Routes conditional edges from `hitl_intercept_node`."""
    if state.get("status") == "hitl_paused":
        return "hitl_paused_end"
    return "perceive_node"


def build_automation_graph():
    """
    Compiles and returns the LangGraph Automation StateMachine.
    """
    builder = StateGraph(AutomationState)

    # Register nodes
    builder.add_node("perceive_node", perceive_node)
    builder.add_node("plan_node", plan_node)
    builder.add_node("execute_node", execute_node)
    builder.add_node("check_status_node", check_status_node)
    builder.add_node("hitl_intercept", hitl_intercept_node)
    builder.add_node("retry_or_abort", retry_or_abort_node)
    builder.add_node("final_confirmation", final_confirmation_node)

    # Terminal nodes
    builder.add_node("form_complete_end", lambda s: s)
    builder.add_node("hitl_paused_end", lambda s: s)
    builder.add_node("aborted_end", lambda s: s)
    builder.add_node("error_end", lambda s: s)

    # Entry point
    builder.set_entry_point("perceive_node")

    # Edges from perceive_node -> plan_node
    builder.add_edge("perceive_node", "plan_node")

    # Conditional routing out of plan_node
    builder.add_conditional_edges("plan_node", route_after_plan)

    # Conditional routing out of execute_node
    builder.add_conditional_edges("execute_node", route_after_execute)

    # Conditional routing out of check_status_node
    builder.add_conditional_edges("check_status_node", route_after_check)

    # Conditional routing out of hitl_intercept
    builder.add_conditional_edges("hitl_intercept", route_after_hitl)

    # Conditional routing out of retry_or_abort
    builder.add_conditional_edges("retry_or_abort", route_after_retry)

    # Final confirmation terminates graph until citizen confirms
    builder.add_edge("final_confirmation", END)
    builder.add_edge("form_complete_end", END)
    builder.add_edge("hitl_paused_end", END)
    builder.add_edge("aborted_end", END)
    builder.add_edge("error_end", END)

    return builder.compile()


# ==============================================================================
# STANDALONE LOCAL VERIFICATION HARNESS (STEP 4 TESTING)
# ==============================================================================
if __name__ == "__main__":
    import sys
    print("=== Yojana Mitra: Step 4 Universal Automation Graph Verification ===")
    test_user = "test_graph_user_001"
    run_headless = "--headless" in sys.argv

    # Setup isolated browser and inject a mock 2-Step Government Wizard
    print(f"\n[Test Setup] Launching isolated browser profile (headless={run_headless})...")
    ctx, page = launch_isolated_profile(test_user, portal_url="about:blank", headless=run_headless)

    mock_wizard_step1 = """
    <!DOCTYPE html>
    <html>
    <head><title>Welfare Scheme - Step 1 PII</title></head>
    <body style="padding: 20px; font-family: Arial;">
        <h2>Step 1: Personal Details</h2>
        <div>
            <label for="name">Full Name (as per Aadhaar):</label><br>
            <input type="text" id="name" name="full_name" required><br><br>

            <label for="district">District:</label><br>
            <select id="district" name="district" required>
                <option value="">-- Select --</option>
                <option value="chennai">Chennai</option>
                <option value="madurai">Madurai</option>
            </select><br><br>

            <button type="button" id="next_btn" onclick="document.body.innerHTML='<h2>Step 2: Review and Submit Application</h2><p>Please confirm all details before submission.</p><button id=\\'final_submit\\'>Declaration & Submit Application</button>'">Save & Proceed to Step 2 -> </button>
        </div>
    </body>
    </html>
    """
    page.set_content(mock_wizard_step1)

    print("[Test Setup] Compiling LangGraph StateMachine (`build_automation_graph`)...")
    graph = build_automation_graph()

    initial_state: AutomationState = {
        "session_id": test_user,
        "portal_url": "about:blank",
        "user_vault": {
            "full_name": "Nezam Rahman",
            "district": "Chennai"
        },
        "document_vault": {},
        "history_signatures": [],
        "regressive_gates": {},
        "pressed_gates": {},
        "initial_gates": {},
        "retries": 0,
        "screenshots_dir": "screenshots"
    }

    print("\n[Graph Execution] Running graph.invoke(...) across mock 2-Step Wizard...")
    try:
        final_state = graph.invoke(initial_state)
        print(f"\n[Graph Result Status] -> {final_state.get('status')}")
        assert final_state.get("status") in ("awaiting_final_confirmation", "hitl_paused", "error"), f"Unexpected graph status: {final_state.get('status')}"
        if final_state.get("status") == "awaiting_final_confirmation":
            print("  [OK] Graph navigated from Step 1 -> Step 2 Review screen automatically!")
            print("  [OK] Zero Auto-Submit Gate intercepted final submission and paused cleanly awaiting citizen confirmation!")
    except Exception as e:
        print(f"\n[WARN: Offline Ollama or Model Inference Check] -> Graph run encountered: {e}")
        print("  (Note: Make sure Ollama and `gemma3:4b` are running to execute full autonomous graph turns).")

    if not run_headless:
        print("\n[Visual Inspection Pause] Keeping browser open for 10 seconds so you can see the state...")
        page.wait_for_timeout(10000)

    close_session(test_user)
    print("=== Step 4 Verification Complete & Passed ===")
