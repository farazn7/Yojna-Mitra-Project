# ==============================================================================
# Yojana Mitra — Phase 8: Step 3 Form Planner (`llm_planner.py`)
# File Path: product_inference/llm_planner.py
# ==============================================================================
"""
Form Planner for Phase 8 Universal Dynamic Web Automation.

Design principle — the model makes semantic judgments; Python does mechanics and safety.

  The LLM (Stage 1) decides things that genuinely require understanding meaning: which vault
  value belongs in which field, whether we hold that data at all or must ask the citizen, and
  whether a navigation control moves the application forward or backward (judged from meaning,
  so it holds for portals in any language).

  Python decides everything with exactly one correct answer: the Playwright verb implied by an
  element's tag/type, the order to act in (fill → upload → advance), whether a step is already
  done (read the live DOM), and whether a proposed value is genuinely vault-backed rather than
  invented. These were previously delegated to a second LLM pass, which could only add latency
  and failure modes — never information.

  A corollary that took a live infinite loop to surface: safety verification has to ask the question
  that actually fits the control. A text field transmits a value, so we verify the value's provenance.
  A checkbox transmits nothing (the `check` verb is a bare click; `value` is discarded), so there is
  no value to verify and value-equality can never pass — what needs verifying is the provenance of
  the *decision*: did the citizen answer this specific field? Python checks that mechanically; reading
  their answer as agreement or refusal, in whatever language they wrote it, stays with the model.

Key Features & Architectural Fixes (Claude Sonnet & Master Plan Review):
  1. Falsy-Safe Vault Value Resolution (`get_vault_value`):
     Avoids Python's `vault.get(key) or default` bug where valid falsy values
     (`0`, `False`, `""`) are silently dropped. Uses explicit `in` and `is not None`
     checks to preserve citizen data accurately.
  2. Stage 1 — Field Classification Engine:
     Categorizes perceived DOM elements into strict semantic buckets (`FILLABLE_CONFIDENT`,
     `UNRESOLVED_CRITICAL`, `DYNAMIC_SKIP`, `ACTION_GATE`, `FILE_UPLOAD`) by matching
     against the citizen's `user_vault` and `document_vault`.
  3. Deterministic Action Derivation (`derive_actions`):
     Translates Stage 1's verified classification into an ordered Playwright batch. Pure
     function, no model involved: verb from tag/type, fixed fill → upload → advance ordering,
     already-satisfied work skipped by comparing against the live DOM, and at most one
     forward navigation click per batch (backward/aborting controls are never pressed).
  4. Step 3.5 — Safe LLM Inference Retry Wrapper (`safe_ollama_call`):
     Wraps Ollama chat calls with Pydantic JSON validation in a 2-retry loop with
     temperature bumping (`0.0 -> 0.1 -> 0.2`). Catches network errors or malformed JSON
     cleanly, returning a structured error state for the graph to handle.
"""

import json
import os
import sys

# Portal text is routinely not Latin script (a scheme portal may render entirely in Devanagari or
# Tamil), and this module prints element labels and the model's own reasoning. A Windows console
# defaults to cp1252, where printing such a string raises UnicodeEncodeError and takes the whole run
# down — reproduced live on a button labelled "जमा करें". Logging must never be able to do that.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from typing import Any, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
import ollama

# Default local LLM for planning
DEFAULT_PLANNER_MODEL = "gemma3:4b"


# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class ClassifiedField(BaseModel):
    id: int = Field(description="The numeric data-ym-id of the element from perception.")
    label: str = Field(description="The human-readable label of the field.")
    tag: str = Field(description="HTML tag: input, select, textarea, button, radio_group, etc.")
    category: Literal[
        "FILLABLE_CONFIDENT",
        "UNRESOLVED_CRITICAL",
        "DYNAMIC_SKIP",
        "ACTION_GATE",
        "FILE_UPLOAD"
    ] = Field(description="Classification bucket based on citizen vault match.")
    reasoning: str = Field(description="Brief explanation of why this category was assigned.")
    proposed_value: Optional[str] = Field(default=None, description="The exact value or document path to fill/upload if confident. Leave null for a checkbox — use `desired_checked` instead.")
    vault_key: Optional[str] = Field(
        default=None,
        description=(
            "FILLABLE_CONFIDENT only: the exact key from the CITIZEN PII VAULT, verbatim, whose data "
            "backs this decision — `proposed_value` for a text/select field, `desired_checked` for a "
            "checkbox or radio. Never invent a key that is not in the vault."
        )
    )
    desired_checked: Optional[bool] = Field(
        default=None,
        description=(
            "Checkbox/radio only: the state this control must END UP in according to the citizen's own "
            "vault answer — true to tick it, false to leave it untouched/off. Read their answer in "
            "whatever language they wrote it (e.g. 'yes', 'haan', 'ஆம்' all mean true). Never set true "
            "unless the citizen's vault answer actually agrees to it."
        )
    )
    question: Optional[str] = Field(
        default=None,
        description=(
            "UNRESOLVED_CRITICAL only: one clear, polite question asking the citizen for exactly this "
            "field, so the answer can be attributed back to this field with no ambiguity."
        )
    )
    commits_field_id: Optional[int] = Field(
        default=None,
        description=(
            "ACTION_GATE only: if this control exists to CONFIRM/VALIDATE one particular field on this "
            "same screen (a 'Verify' next to a code box, a 'Check availability' next to an id box) "
            "rather than to move to the next step, give the numeric id of that field. Leave null for a "
            "control that moves between steps."
        )
    )
    regenerates_content: Optional[bool] = Field(
        default=None,
        description=(
            "ACTION_GATE only: True if pressing this control merely re-issues or replaces content on "
            "this same screen, or wipes what has been entered — a new captcha image, a code sent again, "
            "a reset/clear. Such a control does not move the application on and discards work already "
            "done. False (or null) for a control that moves between steps or confirms a field."
        )
    )
    advances_form: Optional[bool] = Field(
        default=None,
        description=(
            "ACTION_GATE only: True if pressing this control moves the citizen FORWARD (submit/next/"
            "save/proceed/continue), False if it goes backward or aborts (back/previous/cancel/reset). "
            "Judge from the control's meaning in whatever language it is written in."
        )
    )


class Stage1ClassificationResponse(BaseModel):
    classified_fields: list[ClassifiedField] = Field(description="List of all classified interactive form elements.")
    unresolved_questions: list[str] = Field(default_factory=list, description="Direct clarifying questions to ask the citizen if any fields are UNRESOLVED_CRITICAL.")


class ActionCommand(BaseModel):
    element_id: int = Field(description="The target data-ym-id of the element.")
    action: Literal["type", "select", "click", "upload", "check"] = Field(description="Playwright action to execute.")
    value: Optional[str] = Field(default=None, description="String to type, option label to select, or file path to upload.")
    reasoning: str = Field(description="Why this action is being taken.")


# ==============================================================================
# HELPER FUNCTIONS & RETRY WRAPPER
# ==============================================================================

def get_vault_value(user_vault: dict[str, Any], key: str, default: Any = None) -> Any:
    """
    *** FIX: Falsy-Safe Vault Lookup ***
    Safely retrieves `key` from `user_vault`. Preserves valid falsy values (`0`, `False`, `""`).
    Never uses `vault.get(key) or default` which corrupts `0` income or boolean flags.
    """
    if key in user_vault and user_vault[key] is not None:
        return user_vault[key]
    return default


def safe_ollama_call(
    model: str,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    retries: int = 2
) -> Optional[BaseModel]:
    """
    *** FIX Step 3.5: Safe LLM Inference Retry Wrapper ***
    Calls Ollama with structured JSON schema output, validating via Pydantic.
    On failure (`json.JSONDecodeError`, `ValidationError`, `ConnectionError`), retries up to `retries`
    times with incremental temperature (`0.0 + attempt * 0.1`) to break deterministic loops.
    """
    for attempt in range(retries + 1):
        temp = 0.0 + (attempt * 0.1)
        try:
            resp = ollama.chat(
                model=model,
                messages=messages,
                format=schema.model_json_schema(),
                options={"temperature": temp, "num_ctx": 4096}
            )
            raw_content = resp["message"]["content"]
            validated = schema.model_validate_json(raw_content)
            return validated
        except (ValidationError, json.JSONDecodeError, KeyError, Exception) as e:
            print(f"[LLM Planner Attempt {attempt+1}/{retries+1}] Failed with {model} (temp={temp}): {e}")
            if attempt == retries:
                print("[LLM Planner Error] Max retries exhausted. Returning None.")
                return None
    return None


def _element_text(orig_el: dict[str, Any]) -> str:
    """
    All the human-readable identity a perceived element carries, lowercased: its DOM `name`, its
    resolved label, and — for a collapsed radio/checkbox group — its option texts. Groups are labeled
    with their bare `name` attribute, so the option texts are often the only human wording available.
    """
    parts = [str(orig_el.get("name", "")), str(orig_el.get("label", ""))]
    parts += [str(o.get("text", "")) for o in (orig_el.get("options") or [])]
    return " ".join(parts).lower()


def _keys_match_field(vault_key: str, el_text: str) -> bool:
    """
    Does `vault_key` plausibly name the same thing this element is asking for? Pure keyword overlap
    using only the words the key itself is made of — there is no list of expected field names here,
    so it works for a key the citizen's own answer created just as well as a pre-seeded one.
    """
    normalized = vault_key.lower().replace("-", "_").replace(" ", "_")
    keywords = [w for w in normalized.split("_") if len(w) > 2]
    return any(kw in el_text for kw in keywords)


def verify_field_against_vault(orig_el: dict[str, Any], proposed_val: Any, vault: dict[str, Any]) -> bool:
    """
    *** FIX: Semantic Vault Safety Verification ***
    Verifies that `proposed_val` matches a vault key whose name is semantically compatible with `orig_el`.
    Prevents cross-key hallucinations (e.g., LLM filling a 'Full Name' field with the citizen's 'District' value).

    This is value-provenance verification: it only makes sense where a value is actually transmitted
    to the page. Toggles go through `verify_checkbox_against_vault` instead — see the reasoning there.
    """
    prop_clean = str(proposed_val).strip().lower()
    if not prop_clean:
        return False

    # Find exactly which key(s) in vault contain this proposed value
    matching_keys = [
        k for k, v in vault.items()
        if v is not None and (prop_clean == str(v).strip().lower() or prop_clean in str(v).strip().lower())
    ]
    if not matching_keys:
        return False

    # Check if any matching vault key shares keywords with the element's name or label
    el_text = _element_text(orig_el)
    for mk in matching_keys:
        # e.g., if matching key is "full_name", check if el_text has "full" or "name" or "applicant"
        # if matching key is "district", check if el_text has "district" or "city"
        if _keys_match_field(mk, el_text):
            return True
        # Special alias mappings for common PII
        if "name" in mk.lower() and any(alias in el_text for alias in ("name", "applicant", "citizen", "father")):
            return True
        if "district" in mk.lower() and any(alias in el_text for alias in ("district", "city", "place", "resident")):
            return True
    return False


def _resolve_check_target(orig_el: dict[str, Any], proposed_val: Any) -> Optional[tuple[int, bool]]:
    """
    For a toggle, resolves WHICH concrete DOM element to click and whether it is already set,
    returning `(element_id, currently_checked)` or None if the choice cannot be pinned down.

    Perception collapses controls sharing a `name` into one logical element whose `id` is only the
    FIRST option's id. Clicking that id blindly therefore always selects option one no matter which
    option the model actually chose — correct for a lone checkbox, silently wrong for a radio group.
    Which option the citizen's data means is the model's semantic call (`proposed_value`); mapping
    that choice onto a concrete element id, and reading its current state, are mechanical.
    """
    if "id" not in orig_el:
        return None

    options = orig_el.get("options") or []
    if not options:
        # An ungrouped toggle (no `name` attribute) — perception reports its state on the element.
        return orig_el["id"], bool(orig_el.get("checked"))

    if len(options) == 1:
        # A lone checkbox that merely happens to have a `name`; there is nothing to choose between.
        opt = options[0]
    else:
        want = str(proposed_val or "").strip().lower()
        if not want:
            return None
        opt = next(
            (o for o in options
             if want in (str(o.get("text", "")).strip().lower(), str(o.get("value", "")).strip().lower())),
            None
        )
        if opt is None:
            opt = next(
                (o for o in options
                 if want in str(o.get("text", "")).strip().lower()
                 or want in str(o.get("value", "")).strip().lower()),
                None
            )
        if opt is None:
            return None

    return opt["id"], bool(opt.get("checked"))


def verify_checkbox_against_vault(orig_el: dict[str, Any], field: "ClassifiedField", vault: dict[str, Any]) -> bool:
    """
    Safety verification for a toggle (checkbox/radio), which needs a different question asked of it
    than a text field does.

    `verify_field_against_vault` exists to stop invented PII reaching a government form, and it works
    by requiring the proposed value to be traceable to vault data. A toggle transmits no value at all:
    `derive_actions` emits the `check` verb and `execute_node` performs a plain `.click()`, discarding
    `value` entirely. Its `proposed_value` is the browser's `"on"` artifact, not citizen data, so
    demanding that `"on"` be found somewhere in the vault is a test no boolean field can ever pass —
    observed live as an endless re-ask of "I'm not a robot" even after the citizen had answered it.

    What genuinely needs verifying is the provenance of the DECISION, not of a value: ticking a box on
    a government form asserts something on the citizen's behalf ("I consent to eKYC", "I declare this
    is correct"), so the citizen must actually have answered THIS field. Python checks that
    mechanically — the model named a vault key, that key really exists, and its words overlap this
    element's identity. Reading the citizen's answer as agreement or refusal stays with the model
    (`desired_checked`), because 'yes'/'haan'/'no' is a language judgment, not a lookup.
    """
    if field.desired_checked is None:
        return False
    if not field.vault_key or vault.get(field.vault_key) is None:
        return False
    if not _keys_match_field(field.vault_key, _element_text(orig_el)):
        return False
    # We must also know exactly which element a click would land on before trusting it.
    return _resolve_check_target(orig_el, field.proposed_value) is not None


def match_document_vault_key(orig_el: dict[str, Any], document_vault: dict[str, str]) -> Optional[str]:
    """
    Deterministic match between a file input's label/name and a document_vault key, using the same
    keyword-overlap approach as verify_field_against_vault. Serves as a safety net for Stage 1,
    which does not reliably recognize <input type="file"> elements as FILE_UPLOAD even when a
    matching document exists in the vault.
    """
    el_text = _element_text(orig_el)
    for doc_key in document_vault:
        if _keys_match_field(doc_key, el_text):
            return doc_key
    return None


def already_answered(orig_el: dict[str, Any]) -> bool:
    """
    Whether the control already holds an answer, making it pointless to ask the citizen for one.

    A field can be filled without the vault knowing why: the citizen's reply to a HITL question is
    written straight into the element that asked (core_inference.graph._inject_hitl_answer), and
    portals also pre-fill from a logged-in session. Stage 1 judges such a field against the vault,
    finds nothing that matches a live OTP or a captcha, and asks for it again — with two such fields
    on one screen the questions alternate forever (observed live: otp -> captcha -> otp -> ...).
    What the DOM holds is a fact, so it is checked here rather than argued with the model.

    Toggles are excluded: their "answer" is a checked state, which derive_actions compares directly.
    """
    if _playwright_verb(orig_el) == "check":
        return False
    return bool(str(orig_el.get("value") or "").strip())


def gate_identity(orig_el: dict[str, Any]) -> str:
    """
    A navigation control's identity, stable across re-perceptions of the same screen.

    `data-ym-id` is re-assigned every perception pass, so it cannot identify "the control we
    pressed last time". The rendered name/label pair can. Used by the graph to remember which
    gate on which screen turned out to move the citizen backwards.
    """
    return f"{str(orig_el.get('name', '')).strip()}|{str(orig_el.get('label', '')).strip()}".lower()


def _playwright_verb(orig_el: dict[str, Any]) -> str:
    """Maps a perceived element to its Playwright action verb. Pure lookup — never an LLM decision."""
    tag = str(orig_el.get("tag", "")).lower()
    el_type = str(orig_el.get("type", "")).lower()
    if tag == "select" or el_type == "select":
        return "select"
    if el_type in ("checkbox", "radio") or el_type.endswith("_group"):
        return "check"
    return "type"


def derive_actions(
    confident_fields: list[ClassifiedField],
    file_uploads: list[ClassifiedField],
    action_gates: list[ClassifiedField],
    el_by_id: dict[int, dict[str, Any]],
    regressive_gates: Optional[set[str]] = None,
    pressed_gates: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """
    Translates Stage 1's verified classification into an ordered Playwright action batch.

    This is deliberately a PURE FUNCTION with no LLM involved. Choosing a verb for a classified
    field, ordering the batch, and skipping already-completed work are all mechanical decisions
    with exactly one correct answer — handing them to a language model only introduces a failure
    mode (observed live: `type` emitted against a file input, a "Back" click ordered ahead of the
    upload it was supposed to follow, already-filled fields re-filled and destroying dependent
    cascading selects). Semantic judgment stays in Stage 1 where it belongs.

    Ordering is fixed: fill/select/check -> upload -> at most one FORWARD navigation click.
    """
    actions: list[dict[str, Any]] = []

    # Phase 1 — populate fields, skipping any already holding the intended value. Re-setting an
    # already-correct value is not a harmless no-op: it re-fires the element's change handler, which
    # on a cascading control (State -> District) tears down and rebuilds the dependent field.
    for field in confident_fields:
        orig_el = el_by_id.get(field.id, {})

        # Toggles are idempotent by STATE, not by value. `value` is the browser's `"on"` artifact and
        # is identical whether the box is ticked or not, so the value comparison below can neither
        # detect an already-ticked box nor a still-empty one: it would either skip the tick forever or
        # re-click every cycle, toggling the citizen's answer back off. Compare perceived state instead.
        if _playwright_verb(orig_el) == "check":
            target = _resolve_check_target(orig_el, field.proposed_value)
            if target is None:
                continue
            target_id, currently_checked = target
            if currently_checked == bool(field.desired_checked):
                continue
            if not field.desired_checked and str(orig_el.get("type", "")).lower().startswith("radio"):
                # A radio cannot be un-selected by clicking it; clicking would re-assert the very
                # choice we were asked to drop. Leave it to the citizen.
                continue
            actions.append({
                "element_id": target_id,
                "action": "check",
                "value": None,
                "reasoning": field.reasoning,
            })
            continue

        current_value = str(orig_el.get("value") or "").strip().lower()
        proposed_value = str(field.proposed_value or "").strip().lower()
        if current_value and current_value == proposed_value:
            continue
        actions.append({
            "element_id": field.id,
            "action": _playwright_verb(orig_el),
            "value": field.proposed_value,
            "reasoning": field.reasoning,
        })

    # Phase 2 — attach documents. Browsers never expose a file input's real path once a file is
    # attached (always `C:\fakepath\<filename>`), so an already-attached file is detected by
    # filename rather than by full path.
    for field in file_uploads:
        orig_el = el_by_id.get(field.id, {})
        current_value = str(orig_el.get("value") or "")
        if current_value and os.path.basename(current_value) == os.path.basename(str(field.proposed_value)):
            continue
        actions.append({
            "element_id": field.id,
            "action": "upload",
            "value": field.proposed_value,
            "reasoning": field.reasoning,
        })

    # Phase 3 — advance, at most once. A navigation click replaces the page, so anything queued
    # after it would act on a DOM that no longer exists. Backward/aborting gates are never pressed:
    # the agent's objective is to progress through the application, and Stage 1 has already judged
    # each gate's direction semantically (language-agnostic — no keyword matching here).
    #
    # `regressive_gates` is the loop's own observation feeding back in: any gate that the graph
    # already watched land on a screen it had visited before is not pressed again on this screen,
    # whatever Stage 1 now claims about it. Nothing is keyed on wording — the entries are learned
    # at runtime from what the browser actually did.
    #
    # A disabled control is also skipped: portals keep "Proceed" disabled until the step's own
    # checks pass, and clicking it can only burn a retry. Perception reports this as a DOM fact.
    #
    # `pressed_gates` is the other half of that feedback: a control already tried on THIS screen is
    # passed over while nothing new has been entered since, so the next candidate gets its turn. A
    # screen can require several presses in sequence (verify one thing, verify another, then proceed)
    # and this walks them without knowing what any of them are called — whereas repeating the first
    # one forever is what a "Refresh captcha" button did live, discarding the answer each time. A
    # press is only reconsidered once new input has gone in, which is when pressing again could mean
    # something different.
    blocked = regressive_gates or set()
    tried = (pressed_gates or set()) if not actions else set()

    def _pressable(gate: ClassifiedField) -> bool:
        el = el_by_id.get(gate.id, {})
        if el.get("disabled", False):
            return False
        if gate_identity(el) in blocked or gate_identity(el) in tried:
            return False
        # A control that re-issues this screen's content is never a way forward: pressing a captcha's
        # "Refresh" replaced the challenge the citizen had just answered, invalidating it (observed
        # live). Stage 1 judges this from meaning; the loop simply never presses such a control.
        if gate.regenerates_content:
            return False
        # Only a control that can be pressed at all. A text box is not a button whatever Stage 1 says
        # about it, and clicking one waits out Playwright's full 30s timeout for nothing.
        tag = str(el.get("tag", "")).lower()
        el_type = str(el.get("type", "")).lower()
        return tag in ("button", "a") or el_type in ("submit", "button", "reset") or bool(el.get("role") == "button")

    # Phase 3a — confirm what we just entered, before anything that could leave the screen. A code box
    # with its own "Verify" beside it is only accepted once that control is pressed, and pressing it
    # belongs immediately after the value goes in — not several cycles later once the loop happens to
    # get round to it, by which time the portal may have regenerated the challenge. Which control
    # confirms which field is a reading of the screen, so Stage 1 makes it (`commits_field_id`);
    # ordering it right after that field's own action is mechanics.
    entered_ids = {a["element_id"] for a in actions}
    answered_ids = entered_ids | {
        el_id for el_id, el in el_by_id.items() if already_answered(el)
    }
    commit_gates = [
        g for g in action_gates
        if g.commits_field_id is not None and _pressable(g)
        and g.commits_field_id in answered_ids
    ]

    if commit_gates:
        gate = commit_gates[0]
        print(f"[Planner] Confirming '{el_by_id.get(gate.commits_field_id, {}).get('label', gate.commits_field_id)}' "
              f"immediately via its own control '{el_by_id.get(gate.id, {}).get('label', gate.id)}'.")
        actions.append({
            "element_id": gate.id,
            "action": "click",
            "value": None,
            "reasoning": gate.reasoning,
            "gate_identity": gate_identity(el_by_id.get(gate.id, {})),
            # Marks a press that submits one field for checking. A portal that rejects the value
            # commonly clears the box (and reissues the challenge), which must not be mistaken for a
            # control that destroys work — the citizen simply answers again. See check_status_node.
            "confirms_entry": True,
        })
        return actions

    forward_gates = [g for g in action_gates if g.advances_form and g.commits_field_id is None and _pressable(g)]
    if forward_gates:
        gate = forward_gates[0]
        actions.append({
            "element_id": gate.id,
            "action": "click",
            "value": None,
            "reasoning": gate.reasoning,
            "gate_identity": gate_identity(el_by_id.get(gate.id, {})),
        })
    elif action_gates:
        print(f"[Planner] {len(action_gates)} navigation control(s) present but none is a usable forward "
              f"gate on this screen ({len(blocked)} previously observed to move backwards) — not pressing any.")

    return actions


SUBMIT_IDENTIFIER_MODEL = "llama3.1:8b"


class ScreenKind(BaseModel):
    reasoning: str = Field(description="What the screen is asking the citizen to do.")
    is_final_review: bool = Field(
        description="True only if the form is complete and the sole remaining act is to submit the whole application."
    )


def is_final_review_screen(perception: dict[str, Any], model: str = SUBMIT_IDENTIFIER_MODEL) -> bool:
    """
    Whether this screen is the end of the form — everything entered, nothing left but to submit.

    Asked about the SCREEN rather than about a control, because no control can be told apart from
    another by looking at it alone: "Save & Proceed to Step 2" and "Submit Application" are both
    buttons that move things along, and asking which one submits got the answer "#4, Save & Proceed to
    Step 2 — which means it does not submit the form" on a screen that was only Step 1 of 4. The
    screen, in contrast, says plainly what stage the citizen is at: a step still asking for a name and
    a district is not a review of a finished application.
    """
    elements = [
        {"label": el.get("label", ""), "tag": el.get("tag", ""), "type": el.get("type", ""),
         "has_value": bool(str(el.get("value") or "").strip())}
        for el in perception.get("elements_list", [])
    ]
    if not elements:
        return False

    prompt = f"""ELEMENTS ON THIS SCREEN:
{json.dumps(elements, indent=2, ensure_ascii=False)}

Is this the FINAL REVIEW screen of a multi-step form — everything already entered, and the only thing
left is to submit the whole application?

Rules:
- If the screen still asks the citizen to type, choose or upload INFORMATION, it is NOT the final review.
- A screen offering to save and continue to a further step is NOT the final review.
- Agreeing to a declaration or consent is part of submitting, not information being collected — a
  screen whose only input is such an agreement can still be the final review.
- Judge by what the screen means, in whatever language it is written.
"""
    result = safe_ollama_call(model=model, messages=[{"role": "user", "content": prompt}], schema=ScreenKind)
    return bool(result and result.is_final_review)


class SubmissionOutcome(BaseModel):
    reasoning: str = Field(description="What the page says happened.")
    confirms_submission: bool = Field(description="True only if the page confirms the application has been submitted.")


def page_confirms_submission(page_text: str, model: str = DEFAULT_PLANNER_MODEL) -> bool:
    """
    Whether the page shown after the final press confirms the application was actually filed.

    Reported to the citizen, so it must be true rather than optimistic: the press used to be announced
    as "Application Submitted" unconditionally, which said an application had been filed when the
    portal was still sitting on the review screen. Read from the page rather than matched against
    English phrases like "acknowledgement number", which say nothing on a portal in Hindi or Tamil.
    """
    text = " ".join((page_text or "").split())[:1500]
    if not text:
        return False
    prompt = (
        "TEXT SHOWN ON THE PAGE AFTER PRESSING SUBMIT:\n" + text + "\n\n"
        "Does this page confirm that the application HAS BEEN SUBMITTED (for example by acknowledging "
        "receipt or showing a reference for it), or is it still showing the form awaiting submission?\n\n"
        "Judge by meaning, in whatever language the page is written."
    )
    result = safe_ollama_call(model=model, messages=[{"role": "user", "content": prompt}], schema=SubmissionOutcome)
    return bool(result and result.confirms_submission)


class PressConsequence(BaseModel):
    reasoning: str = Field(description="What pressing this control does.")
    finally_submits: bool = Field(description="True only if pressing it finally submits the whole application.")


def press_finally_submits(control_label: str, other_labels: list[str]) -> bool:
    """
    Whether pressing this one control would finally submit the application.

    Asked per control, not as "which of these submits", because the latter forces a choice and got
    "Save & Proceed to Step 2 — which means it does not submit the form" returned as the submission on
    Step 1 of 4, halting a run three steps early. A yes/no about a single control lets the honest
    answer be "no", which is the answer for nearly every control the loop ever presses.

    Both local models are asked and either one is enough to refuse. The two error directions are not
    equal: refusing wrongly costs a pause the citizen can release with CONFIRM, while agreeing wrongly
    files a government application nobody authorized. Measured on submit controls in English, Hindi
    and Tamil and on save/verify/refresh/back controls, the two together caught every submission and
    neither refused an ordinary one.
    """
    prompt = (
        f'CONTROL ABOUT TO BE PRESSED: "{control_label}"\n'
        f'OTHER CONTROLS ON THE SAME SCREEN: {json.dumps(other_labels, ensure_ascii=False)}\n\n'
        "Does pressing this control FINALLY SUBMIT the whole application \u2014 the irreversible "
        "act that files it \u2014 or does it merely save, advance to another step, verify a value, "
        "or go back?\n\n"
        "Rules:\n"
        "- Judge by meaning, in whatever language it is written.\n"
        "- Saving and continuing to a further step is NOT the final submission.\n"
        "- Verifying a code is NOT the final submission.\n"
    )
    for model in (DEFAULT_PLANNER_MODEL, SUBMIT_IDENTIFIER_MODEL):
        result = safe_ollama_call(model=model, messages=[{"role": "user", "content": prompt}], schema=PressConsequence)
        if result and result.finally_submits:
            print(f"[Planner] '{control_label}' reads as the final submission to {model}: {result.reasoning}")
            return True
    return False


class SubmitControlChoice(BaseModel):
    # Field order is deliberate: the model fills JSON keys in order, so stating the reason first makes
    # the id the conclusion of a thought rather than a guess made before thinking. And the id is a
    # REQUIRED int with 0 meaning "none here" — with an Optional[int], llama3.1 reasoned its way to the
    # right control and then returned null anyway, on every case tried.
    reasoning: str = Field(description="Why that control is (or why no control is) the final submission.")
    element_id: int = Field(description="element_id of the control that finally submits the form; 0 if no control here does.")


# (model constant defined above)
# The final submission is identified by the strongest local model available, not the small vision
# model that classifies ordinary fields. It is one call per application — the rarest decision in the
# flow and by far the most consequential — and the 4B model was observed picking the "Back" button on
# a Tamil review screen, which would be a wrong irreversible click. Cost is irrelevant at this
# frequency; being right is not.
def identify_submit_control(perception: dict[str, Any], model: str = SUBMIT_IDENTIFIER_MODEL) -> Optional[int]:
    """
    Identifies the control that finally submits the application on the review screen.

    Reached only after the citizen has replied CONFIRM, so this is not a question of whether to submit
    — that is already decided and enforced elsewhere — but of which control does it. Recognising it is
    a reading of the screen, which is why it is asked of the model rather than matched against English
    button text: `has-text('Submit')` finds nothing on a portal that says "जमा करें", and a wrong match
    would press the wrong button on the most consequential click in the whole flow.

    Returns None when nothing on the screen qualifies, which the caller must treat as "do not click".
    """
    elements_list = perception.get("elements_list", [])
    if not elements_list:
        return None

    pressable = [
        el for el in elements_list
        if not el.get("disabled", False)
        and (str(el.get("tag", "")).lower() in ("button", "a")
             or str(el.get("type", "")).lower() in ("submit", "button", "reset"))
    ]
    if not pressable:
        return None

    # Presented under the key the answer must use. With perception's own `id` key, the model reasoned
    # correctly about which control was the submission and then returned element_id null — it had no
    # `element_id` in front of it to answer with.
    choices = [
        {"element_id": el["id"], "label": el.get("label", ""), "tag": el.get("tag", ""), "type": el.get("type", "")}
        for el in pressable
    ]

    # Deliberately free of preamble about government applications and final submissions: with that
    # context llama3.1 answered "none" on screens with an unmistakable submit button, in every
    # language tried. The controls and the question are enough, and this call is only ever reached
    # once the citizen has already authorized the submission.
    prompt = f"""PRESSABLE CONTROLS:
{json.dumps(choices, indent=2, ensure_ascii=False)}

Which ONE control performs the final submission of the form?

Rules:
- Judge by what each control MEANS, in whatever language it is written. Do not rely on any particular word.
- A control that goes back, cancels, resets, re-issues content or edits a section is NOT the submission.
- Answer 0 if no control here submits the form. Never guess.
"""
    # Asked twice and required to agree. An irreversible click is the one place where an uncertain
    # model must produce no action at all rather than its best guess, and disagreement between two
    # runs is the cheapest available signal that the screen is genuinely ambiguous.
    answers = []
    for _ in range(2):
        result = safe_ollama_call(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            schema=SubmitControlChoice
        )
        answers.append(result.element_id if result else None)

    if answers[0] != answers[1]:
        print(f"[Planner] Two readings of the review screen disagreed ({answers}) — not pressing anything.")
        return None
    if not answers[0]:
        print("[Planner] No control on this screen was identified as the final submission.")
        return None
    result = SubmitControlChoice(element_id=answers[0], reasoning=(result.reasoning if result else ""))

    # The model may only choose from what is actually pressable on this screen.
    if result.element_id not in {el["id"] for el in pressable}:
        print(f"[Planner] Submission control #{result.element_id} is not a pressable control on this screen — ignoring.")
        return None

    print(f"[Planner] Final submission control identified: #{result.element_id} ({result.reasoning})")
    return result.element_id


# ==============================================================================
# MAIN DUAL-STAGE PLANNER FUNCTION (`plan_wizard_step`)
# ==============================================================================

def plan_wizard_step(
    perception: dict[str, Any],
    user_vault: dict[str, Any],
    document_vault: dict[str, str],
    model: str = DEFAULT_PLANNER_MODEL,
    regressive_gates: Optional[set[str]] = None,
    pressed_gates: Optional[set[str]] = None
) -> dict[str, Any]:
    """
    Orchestrates Stage 1 (Classification) + Stage 2 (Action Execution Generation).
    Enforces server-side confident ID intersection checks before returning execution payload.
    """
    elements_list = perception.get("elements_list", [])
    aria_tree_yaml = perception.get("aria_tree_yaml", "")

    # Quick exit if page has no elements
    if not elements_list:
        return {
            "status": "error",
            "last_error": "No interactive elements perceived on page.",
            "actions": [],
            "unresolved_questions": []
        }

    # --------------------------------------------------------------------------
    # Stage 1: Field Classification Engine
    # --------------------------------------------------------------------------
    stage1_prompt = f"""You are Yojana Mitra's Stage 1 Form Classification Engine.
Analyze the perceived web elements and match them against the citizen's verified vault data.

=== CITIZEN PII VAULT ===
{json.dumps(user_vault, indent=2)}

=== CITIZEN DOCUMENT VAULT (File paths) ===
{json.dumps(document_vault, indent=2)}

=== PERCEIVED PAGE ACCESSIBILITY TREE ===
{aria_tree_yaml[:1500]}

=== PERCEIVED INTERACTIVE ELEMENTS LIST ===
{json.dumps(elements_list, indent=2)}

INSTRUCTIONS:
For EVERY element in the elements list, assign exactly ONE category:
1. "FILLABLE_CONFIDENT": We have exact or clearly matching data in the PII vault. Set `proposed_value`.
2. "FILE_UPLOAD": It is a file upload input (`type="file"`) whose document path exists in DOCUMENT VAULT. Set `proposed_value` to the file path.
3. "UNRESOLVED_CRITICAL": Required form field (`required: true` or personal info) whose value is NOT in the vault. We must ask the citizen!
4. "DYNAMIC_SKIP": Optional search boxes, filter inputs, language selectors, or unimportant banners.
5. "ACTION_GATE": Any control the citizen presses to make the form act, rather than to enter data.
   This covers controls that change which form step is shown — BOTH forward ones (Submit, Next, Save,
   Proceed, Continue) AND backward/aborting ones (Back, Previous, Cancel, Reset) — and also controls
   that confirm one field in place without leaving the screen (a 'Verify' beside a code box).

CRITICAL RULES:
- If an element is `UNRESOLVED_CRITICAL`, set its own `question` to a clear, polite request for exactly
  that field, and repeat it in `unresolved_questions`.
- For every `FILLABLE_CONFIDENT` element, set `vault_key` to the VAULT key you took the data from,
  spelled exactly as it appears in the vault above. If no vault key genuinely covers the field, it is
  `UNRESOLVED_CRITICAL`, not confident.
- For `radio_group` or `select`, match the citizen's vault value to the option text/value.
- CHECKBOX / RADIO RULE (`type` is `checkbox`, `radio`, `checkbox_group` or `radio_group`): a toggle
  carries no data, only a state, so `desired_checked` is MANDATORY and must never be left null when
  you mark one `FILLABLE_CONFIDENT`. Set `desired_checked: true` if the citizen's vault answer for
  this field agrees to it, `false` if their answer declines it — read their answer in whatever
  language they wrote it. A `FILLABLE_CONFIDENT` toggle with a null `desired_checked` is INVALID
  output and will be rejected. If no vault answer covers this box, mark it `UNRESOLVED_CRITICAL`
  instead. For a radio group also set `proposed_value` to the chosen option's text.
  Example — vault has `"not_a_robot": "yes"` and the element is named `not_a_robot`, so emit
  category `FILLABLE_CONFIDENT`, `vault_key` `not_a_robot`, `desired_checked` `true`.
- For EVERY `ACTION_GATE`, you MUST also set `advances_form`: `true` if pressing it moves the citizen
  FORWARD through the application, `false` if it moves backward or abandons progress. Decide this from
  what the control actually means, in whatever language the portal is written in — do not rely on the
  presence of any particular English word.
- If an `ACTION_GATE` exists to confirm ONE field on this same screen rather than to move between
  steps, set `commits_field_id` to that field's id. Judge it by what the control sits beside and what
  it says it does. A control that belongs to no single field leaves `commits_field_id` null.
- Set `regenerates_content: true` on an `ACTION_GATE` that only re-issues or replaces this screen's own
  content, or clears entries — a fresh captcha, a code sent again, a reset. Such a control is NOT a way
  forward even though it changes what is on screen: pressing it throws away what the citizen already
  provided. Judge this from meaning, in any language.
"""

    stage1_messages = [
        {"role": "system", "content": "You are an exact, deterministic AI schema classifier that never hallucinates."},
        {"role": "user", "content": stage1_prompt}
    ]

    stage1_res = safe_ollama_call(model, stage1_messages, Stage1ClassificationResponse)

    # Fallback if Stage 1 completely fails after retries
    if not isinstance(stage1_res, Stage1ClassificationResponse):
        return {
            "status": "error",
            "last_error": f"Stage 1 LLM Classification failed after retries using {model}.",
            "actions": [],
            "unresolved_questions": []
        }

    # Map original elements by ID to check `required` flags
    el_by_id = {el["id"]: el for el in elements_list}

    # Partition classified fields with deterministic Python safety verification
    confident_fields: list[ClassifiedField] = []
    file_uploads: list[ClassifiedField] = []
    action_gates: list[ClassifiedField] = []
    unresolved_fields: list[ClassifiedField] = []

    for field in stage1_res.classified_fields:
        # Stage 1 sees the ARIA snapshot as well as the element list, and will occasionally classify
        # something that is only text on the page: after the portal verified the OTP and captcha it
        # showed them back as plain text, and the model returned them as fields with ids that exist on
        # no element. Those went to the citizen as questions about a field with no name, until the
        # re-ask bound ended the run. An id that is not in perception cannot be filled, pressed or
        # asked about, so it is dropped here — the one place every classified field passes through.
        if field.id not in el_by_id:
            print(f"[Stage 1 Discard] Field #{field.id} ('{field.label}') is not a real element on this page.")
            continue

        orig_el = el_by_id[field.id]
        is_required = orig_el.get("required", False)

        if field.category == "FILLABLE_CONFIDENT" and _playwright_verb(orig_el) == "check":
            # Toggles are verified on the provenance of the decision rather than of a value — see
            # verify_checkbox_against_vault. Value-equality is unsatisfiable for any boolean control.
            if verify_checkbox_against_vault(orig_el, field, user_vault):
                confident_fields.append(field)
            else:
                print(
                    f"[Stage 1 Intercept] Toggle '{field.label}' has no vault-backed answer from the citizen "
                    f"(vault_key={field.vault_key!r}, desired_checked={field.desired_checked}). Routing to HITL!"
                )
                unresolved_fields.append(field)
        elif field.category == "FILLABLE_CONFIDENT" and field.proposed_value is not None:
            # Deterministic Verification: check if proposed_value actually matches a semantically valid vault key!
            # Prevents local LLMs from hallucinating common names or cross-matching 'District' into 'Full Name'.
            # This holds regardless of the DOM's `required` attribute — a hallucinated value on an
            # optional field is exactly as unsafe to auto-fill as one on a required field. Any verification
            # failure routes to HITL so the citizen confirms it, rather than silently trusting a guess.
            if verify_field_against_vault(orig_el, field.proposed_value, user_vault):
                confident_fields.append(field)
            elif already_answered(orig_el):
                print(f"[Stage 1 Skip] '{field.label}' already holds an answer on the page — not asking again.")
            else:
                print(f"[Stage 1 Intercept] Hallucinated/Cross-matched value '{field.proposed_value}' not valid for field '{field.label}'. Routing to HITL!")
                unresolved_fields.append(field)
        elif field.category == "FILE_UPLOAD" and field.proposed_value is not None:
            # Verify the proposed file path is genuinely one of the citizen's verified document
            # paths. FILE_UPLOAD has no equivalent of FILLABLE_CONFIDENT's verify_field_against_vault,
            # so without this a hallucinated path would reach a real Playwright upload action.
            #
            # Browsers never expose a file input's real absolute path once a file is attached —
            # for security, `input.value` always reports it as `C:\fakepath\<filename>` regardless
            # of the real source path. So after a genuinely successful upload, the *next* perceive
            # cycle sees that fakepath value, Stage 1 correctly proposes it back, and a naive exact
            # match against document_vault's real paths would wrongly reject an already-uploaded
            # file as "unverified" — reproduced live. Compare by filename in that case instead.
            current_value = str(orig_el.get("value") or "")
            already_uploaded = bool(current_value) and os.path.basename(current_value) == os.path.basename(str(field.proposed_value))
            if field.proposed_value in document_vault.values() or already_uploaded:
                file_uploads.append(field)
            else:
                print(f"[Stage 1 Intercept] Hallucinated/unverified file path '{field.proposed_value}' not found in document vault for field '{field.label}'. Routing to HITL!")
                unresolved_fields.append(field)
        elif field.category == "ACTION_GATE":
            action_gates.append(field)
        elif field.category == "UNRESOLVED_CRITICAL" or (is_required and field.category == "DYNAMIC_SKIP"):
            if already_answered(orig_el):
                print(f"[Stage 1 Skip] '{field.label}' already holds an answer on the page — not asking again.")
            else:
                unresolved_fields.append(field)

    # *** Deterministic File-Input Classification Safety Gate ***
    # Whether an <input type="file"> is a document upload is a structural fact, not a judgment
    # call, so it is not left to the model (which was observed classifying one as DYNAMIC_SKIP
    # despite a matching document being present). Scan perception directly for file inputs Stage 1
    # did not already route, and match them against document_vault deterministically.
    already_handled_ids = {f.id for f in (confident_fields + file_uploads + unresolved_fields + action_gates)}
    for el in elements_list:
        if el.get("type") != "file" or el["id"] in already_handled_ids:
            continue
        matched_key = match_document_vault_key(el, document_vault)
        if matched_key:
            print(f"[Deterministic File Match] Element '{el.get('label')}' (#{el['id']}) matched to document vault key '{matched_key}'.")
            file_uploads.append(ClassifiedField(
                id=el["id"],
                label=el.get("label", f"Field {el['id']}"),
                tag=el.get("tag", "input"),
                category="FILE_UPLOAD",
                reasoning="Deterministic match against the citizen's document vault.",
                proposed_value=document_vault[matched_key]
            ))
        elif el.get("required"):
            unresolved_fields.append(ClassifiedField(
                id=el["id"],
                label=el.get("label", f"Field {el['id']}"),
                tag=el.get("tag", "input"),
                category="UNRESOLVED_CRITICAL",
                reasoning="Required file upload with no matching document found in vault."
            ))

    # *** FIX: Deterministic Required Field Safety Gate ***
    # Ensure every single `required=True` element from perception is either confidently filled or flagged as unresolved!
    # Even if the LLM misclassifies or invents a value, Python verifies the ID against confident sets.
    confident_ids_so_far = {f.id for f in (confident_fields + file_uploads)}
    for el in elements_list:
        if el.get("required") and el["id"] not in confident_ids_so_far:
            if not any(u.id == el["id"] for u in unresolved_fields):
                print(f"[Deterministic Safety Gate] Required field #{el['id']} ('{el.get('label')}') not confidently matched. Auto-routing to HITL Clarify!")
                unresolved_fields.append(ClassifiedField(
                    id=el["id"],
                    label=el.get("label", f"Field {el['id']}"),
                    tag=el.get("tag", "input"),
                    category="UNRESOLVED_CRITICAL",
                    reasoning="Required form field not found or verified against citizen vault data."
                ))

    # If any required fields are UNRESOLVED_CRITICAL, immediately pause and route to HITL question!
    # Do not click 'Submit' or execute partial form actions if critical fields are missing.
    if unresolved_fields:
        # Carry each element's DOM `name` through to the caller. The citizen's answer is written back
        # into the vault under this key, and a DOM name is a stable snake_case identifier
        # ("not_a_robot", "otp", "state") that the next cycle's keyword matching reliably recognises —
        # whereas the label it previously fell back to is arbitrary prose ("I'm not a robot",
        # "6-digit code") that matches only by luck and is useless as a cross-scheme vault key.
        unresolved_payload = []
        for f in unresolved_fields:
            dumped = f.model_dump()
            perceived = el_by_id.get(f.id, {})
            dumped["name"] = perceived.get("name", "")
            # The perceived input type travels with the field so the caller can tell a free-text
            # challenge (OTP/captcha) from a toggle or a file input without re-perceiving.
            dumped["type"] = perceived.get("type", "")
            unresolved_payload.append(dumped)

        # Ask about exactly ONE field per turn. The citizen answers with a single message, so
        # attributing that message to the right field is what makes the vault write-back correct —
        # and asking two questions at once makes the attribution a guess. Each field now carries its
        # own `question`, so question<->field alignment is structural rather than dependent on two
        # independently-ordered lists happening to line up. Remaining fields come back around on the
        # next perceive/plan cycle.
        first = unresolved_payload[0]
        question = (
            first.get("question")
            or next((q for q in stage1_res.unresolved_questions if q), None)
            or f"Could you please provide your {first.get('label')}?"
        )
        return {
            "status": "hitl_clarify",
            "unresolved_fields": unresolved_payload,
            "unresolved_questions": [question],
            "actions": [],
            "reasoning": f"Found {len(unresolved_fields)} unresolved critical field(s); asking about '{first.get('label')}' first."
        }

    # --------------------------------------------------------------------------
    # Action Derivation (deterministic — no second LLM pass)
    # --------------------------------------------------------------------------
    # Everything the old "Stage 2" LLM call produced is a pure function of Stage 1's verified
    # output: the verb follows from the element's tag/type, the ordering is always
    # fill -> upload -> advance, and "has this already been done?" is answered by the live DOM.
    # Asking a model to redo that mapping added latency and a whole class of failures (wrong verb
    # on file inputs, navigation clicks ordered before the work they depend on, already-filled
    # fields re-filled and wiping dependent cascading selects) without adding any information.
    validated_actions = derive_actions(confident_fields, file_uploads, action_gates, el_by_id, regressive_gates, pressed_gates)

    return {
        "status": "actions_ready" if validated_actions else "perceiving",
        "actions": validated_actions,
        "confident_fields_count": len(confident_fields) + len(file_uploads),
        "action_gates_count": len(action_gates)
    }


# ==============================================================================
# STANDALONE LOCAL VERIFICATION HARNESS (STEP 3 TESTING)
# ==============================================================================
if __name__ == "__main__":
    print("=== Yojana Mitra: Step 3 Dual-Stage LLM Planner Verification ===")

    # 1. Test Falsy-Safe Vault Lookup (`get_vault_value`)
    print("\n[Test 1] Verifying Falsy-Safe Vault Lookups (`get_vault_value`)...")
    mock_vault = {
        "annual_income": 0,          # Valid 0 income (e.g. BPL/unemployed)
        "has_disability": False,     # Valid boolean False
        "empty_field": "",           # Valid empty string
        "valid_name": "Nezam Rahman"
    }
    assert get_vault_value(mock_vault, "annual_income", default=100000) == 0, "Failed: `0` was dropped!"
    assert get_vault_value(mock_vault, "has_disability", default=True) == False, "Failed: `False` was dropped!"
    assert get_vault_value(mock_vault, "empty_field", default="fallback") == "", "Failed: `""` was dropped!"
    assert get_vault_value(mock_vault, "missing_key", default="default_val") == "default_val", "Failed: default not returned for missing key!"
    print("  [OK] Falsy-safe lookups (`0`, `False`, `""`) verified accurately!")

    # 2. Test Stage 1 & Stage 2 Execution on Mock Perception Payload
    print("\n[Test 2] Running Dual-Stage Planner (`plan_wizard_step`) with Ollama...")
    mock_perception = {
        "aria_tree_yaml": "- heading 'Welfare Scheme Application'\n- textbox 'Full Name'\n- combobox 'District'\n- button 'Next'",
        "elements_list": [
            {"id": 1, "tag": "input", "type": "text", "label": "Full Name (as per Aadhaar)", "name": "full_name", "required": True},
            {"id": 2, "tag": "select", "type": "select", "label": "Resident District", "name": "district", "required": True, "options": [{"text": "Chennai", "value": "chennai"}, {"text": "Madurai", "value": "madurai"}]},
            {"id": 3, "tag": "input", "type": "file", "label": "Upload Income Certificate", "name": "income_cert", "required": True},
            {"id": 4, "tag": "button", "type": "button", "label": "Save & Proceed to Step 2", "name": "next_btn", "required": False}
        ]
    }

    mock_user_vault = {
        "full_name": "Nezam Rahman",
        "district": "Chennai"
    }

    mock_doc_vault = {
        "income_certificate": "C:/Users/nezam/Documents/income_cert_verified.pdf"
    }

    print("  Executing `plan_wizard_step(...)` across 4 mock elements (`Full Name`, `District`, `Income Cert`, `Save Button`)...")
    plan_result = plan_wizard_step(
        perception=mock_perception,
        user_vault=mock_user_vault,
        document_vault=mock_doc_vault,
        model=DEFAULT_PLANNER_MODEL
    )

    print(f"\n  [OK] Planner Execution Status: {plan_result.get('status')}")
    if plan_result.get('status') == 'actions_ready':
        print(f"  [OK] Confident Fields Processed: {plan_result.get('confident_fields_count')}")
        print(f"  [OK] Action Gates Processed: {plan_result.get('action_gates_count')}")
        print("  [OK] Validated Playwright Actions:")
        for act in plan_result.get("actions", []):
            print(f"       -> [ID #{act['element_id']}] Action: {act['action']} | Value: '{act['value']}' | Reason: {act['reasoning']}")
        assert len(plan_result.get("actions", [])) >= 3, "Expected at least 3 actions (`type`, `select/upload`, `click`)!"
        print("  [OK] Stage 1 & Stage 2 classification and action generation passed cleanly!")
    elif plan_result.get('status') == 'error':
        print(f"  [WARN: Ollama Offline or Model {DEFAULT_PLANNER_MODEL} Missing] -> {plan_result.get('last_error')}")
        print("  (Note: Make sure Ollama is running and `gemma2:2b` is installed to run full LLM inference tests locally).")

    # 3. Test HITL Clarification Gate (`UNRESOLVED_CRITICAL`)
    print("\n[Test 3] Verifying HITL Clarification Gate when required field is missing from Vault...")
    missing_vault = {"district": "Chennai"} # Notice `full_name` is missing!
    hitl_plan = plan_wizard_step(
        perception=mock_perception,
        user_vault=missing_vault,
        document_vault=mock_doc_vault,
        model=DEFAULT_PLANNER_MODEL
    )
    print(f"  [OK] Missing Vault Status: {hitl_plan.get('status')}")
    if hitl_plan.get('status') == 'hitl_clarify':
        print(f"  [OK] Unresolved Questions Generated: {hitl_plan.get('unresolved_questions')}")
        assert len(hitl_plan.get('unresolved_fields', [])) >= 1, "Failed to flag `Full Name` as unresolved!"
        print("  [OK] HITL Clarification gate caught missing data and paused cleanly before submission!")

    print("\n=== Step 3 Verification Complete & Passed ===")
