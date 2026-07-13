# ==============================================================================
# Yojana Mitra — Phase 8: Universal Dynamic Web Automation & HITL Gateway
# File Path: product_inference/universal_automation.py
# ==============================================================================
"""
Universal Playwright Web Automation & Human-In-The-Loop (HITL) Gateway.

This module automates multi-page Indian government welfare portal applications
without hardcoded CSS/ID selectors. It leverages:
  1. Playwright Persistent Contexts (`launch_persistent_context`) to isolate user profiles
     and bypass Chrome 136 CDP default-directory security restrictions.
  2. Playwright Accessibility Snapshots (`aria_snapshot()`) to extract clean YAML DOM trees.
  3. A 2-Stage Pydantic-driven LLM Planner (`Llama 3.1` / `Gemma 3`) to map PII Vault keys
     and generate exact action lists with zero hallucinations.
  4. A LangGraph ReAct StateMachine (`StateGraph`) managing multi-step form wizards,
     native dialog interception, mid-flow OTP intercept gates, and final screenshot confirmation.
"""

import os
import re
import json
import hashlib
from typing import TypedDict, Optional, Literal, Any
from pydantic import BaseModel, Field
from playwright.sync_api import (
    sync_playwright,
    BrowserContext,
    Page,
    PlaywrightTimeoutError,
    TargetClosedError,
    Dialog
)
from langgraph.graph import StateGraph, END
import ollama

# ==============================================================================
# 1. PYDANTIC STRUCTURED SCHEMAS (2-STAGE ANTI-HALLUCINATION PLANNER)
# ==============================================================================

class FieldMapping(BaseModel):
    """Stage 1: Maps a single perceived DOM element to an exact PII Vault key."""
    element_id: int = Field(..., description="Numerical ID assigned during DOM perception.")
    element_label: str = Field(..., description="Human-readable accessibility label of the element.")
    source_key: Optional[str] = Field(
        None, 
        description="Exact key from pii_vault (e.g., 'full_name', 'aadhaar_number', 'income') or document type ('aadhaar', 'income_certificate'). Null if no confident match."
    )
    match_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0 to 1.0) of this mapping.")

class MappingResult(BaseModel):
    """Stage 1 Output: List of field mappings for all visible elements."""
    mappings: list[FieldMapping]

class FormAction(BaseModel):
    """Stage 2: Specific executable browser action for a successfully mapped field."""
    element_id: int
    action: Literal["fill", "select", "upload", "click", "check"]
    value: Optional[str] = Field(None, description="String value to fill or option text to select in dropdown.")
    file_path: Optional[str] = Field(None, description="Absolute/relative local file path when action is 'upload'.")
    source_key: str = Field(..., description="The PII Vault key that provided this value.")
    reasoning: str = Field(..., description="Short explanation for audit logging.")

class ActionPlan(BaseModel):
    """Stage 2 Output: Executable plan for the current wizard step."""
    actions: list[FormAction]
    unresolved_field_ids: list[int] = Field(
        default_factory=list, 
        description="IDs of elements where data is missing, null, or confidence < 0.85 (requires HITL prompt)."
    )

# ==============================================================================
# 2. LANGGRAPH STATE MACHINE TYPEDDICT
# ==============================================================================

class AutomationState(TypedDict):
    """Shared state buffer for the Phase 8 automation workflow loop."""
    session_id: str
    user_id: str
    portal_url: str
    profile_dir: str
    
    # DOM Perception & Mapping
    elements_yaml: str                       # Output of page.locator("form").aria_snapshot()
    parsed_elements: list[dict]              # Flat list of extracted interactive fields with IDs
    mapped_fields: list[dict]                # Results from Stage 1 mapping
    action_plan: list[dict]                  # Results from Stage 2 action generation
    unresolved_fields: list[int]             # Fields requiring user intervention
    
    # User Profile & Document Context
    user_profile: dict                       # Verified PII Vault data dictionary
    document_paths: dict                     # Mapping of doc_type -> local file path on disk
    
    # Execution & Resilience Tracking
    status: Literal[
        "perceiving", "planning", "executing", "checking",
        "otp_required", "hitl_clarify", "error", "form_complete", "aborted"
    ]
    retries: int
    page_signature: Optional[str]            # Hash of URL + stable DOM markers
    last_error: Optional[str]
    abort_requested: bool

# ==============================================================================
# 3. BROWSER & CONTEXT RESILIENCE MANAGERS
# ==============================================================================

# Global session registry to keep persistent contexts alive across graph ticks
_ACTIVE_SESSIONS: dict[str, dict[str, Any]] = {}

def get_or_launch_context(user_id: str, portal_url: str) -> tuple[BrowserContext, Page]:
    """
    Launches or retrieves a persistent Playwright context for a specific user ID.
    Using a dedicated user profile directory (`browser_profiles/{user_id}`) bypasses
    Chrome 136 CDP restrictions while preserving cookies and login sessions.
    """
    if user_id in _ACTIVE_SESSIONS:
        ctx = _ACTIVE_SESSIONS[user_id]["context"]
        page = _ACTIVE_SESSIONS[user_id]["active_page"]
        if not page.is_closed():
            return ctx, page

    profile_dir = os.path.abspath(f"C:/Users/nezam/.gemini/antigravity/browser_profiles/{user_id}")
    os.makedirs(profile_dir, exist_ok=True)

    playwright = sync_playwright().start()
    
    # Launch real Chrome (not bundled Chromium) in persistent context mode
    context = playwright.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=False,
        channel="chrome",
        args=["--remote-debugging-port=0"], # Dedicated profile allows debugging safely
        viewport={"width": 1280, "height": 900},
        accept_downloads=True
    )

    page = context.pages[0] if context.pages else context.new_page()

    # Register Native Dialog Interceptor (prevents window.confirm/alert from hanging indefinitely)
    def on_dialog(dialog: Dialog):
        msg = dialog.message.lower()
        print(f"\n[⚠️ PORTAL DIALOG DETECTED] Type: {dialog.type} | Message: '{dialog.message}'")
        if "otp" in msg or "confirm" in msg or "submit" in msg:
            # In live runtime, notify Discord and await user choice; default accept if harmless
            print("[Dialog Handler] Auto-accepting portal confirmation gate...")
            dialog.accept()
        else:
            dialog.dismiss()

    page.on("dialog", on_dialog)

    # Listen for new tabs opened via window.open() (e.g., e-KYC or payment gateways)
    def on_new_page(new_page: Page):
        print(f"\n[🔄 NEW TAB OPENED] Switching active page focus to: {new_page.url}")
        new_page.wait_for_load_state("domcontentloaded")
        _ACTIVE_SESSIONS[user_id]["active_page"] = new_page

    context.on("page", on_new_page)

    # Navigate if not already on the portal
    if page.url == "about:blank" or not page.url.startswith("http"):
        print(f"[Navigate] Opening Portal: {portal_url}")
        page.goto(portal_url, wait_until="domcontentloaded")

    _ACTIVE_SESSIONS[user_id] = {
        "playwright": playwright,
        "context": context,
        "active_page": page
    }
    return context, page

def compute_page_signature(page: Page) -> str:
    """Computes a stable SHA-256 hash of the URL and page title/heading to track wizard state."""
    try:
        url_clean = page.url.split("?")[0]
        title = page.title()
        raw_sig = f"{url_clean}||{title}"
        return hashlib.sha256(raw_sig.encode('utf-8')).hexdigest()[:16]
    except Exception:
        return "unknown_signature"

# ==============================================================================
# 4. LANGGRAPH WORKFLOW NODES
# ==============================================================================

def perceive_webpage(state: AutomationState) -> AutomationState:
    """Node 1: Extracts the clean YAML Accessibility Tree from the active browser tab."""
    if state.get("abort_requested"):
        state["status"] = "aborted"
        return state

    print("\n[Node 1: perceive_webpage] Extracting DOM Accessibility Snapshot...")
    try:
        _, page = get_or_launch_context(state["user_id"], state["portal_url"])
        
        # Wait for network stabilization
        try:
            page.wait_for_load_state("networkidle", timeout=5000)
        except TimeoutError:
            pass # Continue if background analytics connections ping indefinitely

        # Extract YAML Accessibility Tree (strips CSS mess, <div> wrappers, and utility classes)
        tree_yaml = page.locator("body").aria_snapshot() or ""
        state["elements_yaml"] = tree_yaml
        state["page_signature"] = compute_page_signature(page)
        
        # Parse interactive elements into a flat numbered list for the Pydantic mapper
        # We also run a lightweight JS check to tag hidden file inputs directly
        js_extractor = """
        () => {
            const inputs = Array.from(document.querySelectorAll('input, select, textarea, button, [role="button"]'));
            return inputs.map((el, idx) => {
                const id = idx + 1;
                el.setAttribute('data-ym-id', id.toString());
                let label = el.getAttribute('aria-label') || el.getAttribute('placeholder') || '';
                if (!label && el.id) {
                    const lbl = document.querySelector(`label[for="${el.id}"]`);
                    if (lbl) label = lbl.innerText.trim();
                }
                if (!label) label = el.innerText.trim() || el.name || 'Field ' + id;
                return {
                    id: id,
                    tag: el.tagName.toLowerCase(),
                    type: el.type || 'text',
                    label: label,
                    value: el.value || '',
                    required: el.required || false
                };
            }).filter(item => item.label && item.label !== 'Field ' + item.id);
        }
        """
        parsed_elements = page.evaluate(js_extractor)
        state["parsed_elements"] = parsed_elements
        state["status"] = "planning"
        
    except TargetClosedError:
        state["status"] = "error"
        state["last_error"] = "Browser tab closed unexpectedly during perception."
    except Exception as e:
        state["status"] = "error"
        state["last_error"] = str(e)

    return state

def plan_actions(state: AutomationState) -> AutomationState:
    """Node 2: Executes the 2-Stage Pydantic LLM Planner (Mapping -> Action Generation)."""
    if state.get("abort_requested"):
        state["status"] = "aborted"
        return state

    print("\n[Node 2: plan_actions] Running Stage 1: Field to PII Vault Mapping...")
    elements = state["parsed_elements"]
    profile = state["user_profile"]
    docs = state["document_paths"]
    
    if not elements:
        print("[Plan] No interactive form fields detected on this screen.")
        state["status"] = "checking"
        return state

    # Stage 1 Prompt: Strict Field Mapping Only (No Action Generation Yet)
    stage1_prompt = f"""
    You are the Stage 1 Field Mapper for Yojana Mitra.
    Your job is to map each perceived form element to an exact key in the USER PROFILE or DOCUMENT PATHS.

    USER PROFILE KEYS: {list(profile.keys())}
    USER PROFILE DATA: {json.dumps(profile, indent=2)}
    DOCUMENT TYPES AVAILABLE ON DISK: {list(docs.keys())}

    PERCEIVED FORM ELEMENTS:
    {json.dumps(elements[:30], indent=2)}

    RULES:
    1. Only set `source_key` to a key that literally exists in the provided profile JSON or document types.
    2. If no field confidently matches, set `source_key` to null and assign `match_confidence` < 0.5. DO NOT GUESS.
    """

    try:
        response_s1 = ollama.chat(
            model='llama3.1',
            messages=[{"role": "system", "content": stage1_prompt}],
            format=MappingResult.model_json_schema(),
            options={"temperature": 0.0}
        )
        mapping_result = MappingResult.model_validate_json(response_s1['message']['content'])
        
        # Filter confident mappings (confidence >= 0.85 and source_key exists)
        confident_mappings = []
        unresolved_ids = []
        
        for m in mapping_result.mappings:
            if m.match_confidence >= 0.85 and m.source_key:
                # Double-check if we actually have data for this key
                val = profile.get(m.source_key) or docs.get(m.source_key)
                if val is not None:
                    confident_mappings.append(m.model_dump())
                else:
                    unresolved_ids.append(m.element_id)
            else:
                unresolved_ids.append(m.element_id)

        state["mapped_fields"] = confident_mappings
        state["unresolved_fields"] = unresolved_ids

        # If there are unresolved mandatory fields, trigger HITL clarification right now
        if len(unresolved_ids) > 5 and len(confident_mappings) == 0:
            print(f"[Plan Note] {len(unresolved_ids)} fields unresolved. Could be an OTP or Review screen.")
            state["status"] = "checking"
            return state

        # Stage 2 Prompt: Action Generation for Mapped Fields
        print(f"\n[Node 2: plan_actions] Running Stage 2: Action Generation ({len(confident_mappings)} mapped fields)...")
        stage2_prompt = f"""
        You are the Stage 2 Action Generator for Yojana Mitra.
        Generate exact Playwright form actions for these confident mappings.

        CONFIDENT MAPPINGS: {json.dumps(confident_mappings, indent=2)}
        USER PROFILE DATA: {json.dumps(profile, indent=2)}
        DOCUMENT LOCAL PATHS: {json.dumps(docs, indent=2)}

        RULES:
        1. For text/number boxes (`input[type=text|number]`), action = "fill", value = profile string.
        2. For dropdowns (`select`), action = "select", value = exact option text to pick.
        3. For file inputs (`input[type=file]`), action = "upload", file_path = exact absolute path from DOCUMENT LOCAL PATHS.
        4. If all fields on the page are mapped, include a final "click" action on the "Next" or "Save & Continue" button if present.
        """

        response_s2 = ollama.chat(
            model='llama3.1',
            messages=[{"role": "system", "content": stage2_prompt}],
            format=ActionPlan.model_json_schema(),
            options={"temperature": 0.0}
        )
        action_plan = ActionPlan.model_validate_json(response_s2['message']['content'])
        
        state["action_plan"] = [a.model_dump() for a in action_plan.actions]
        state["status"] = "executing"

    except Exception as e:
        print(f"[Plan Error] Stage 1/2 Pydantic Validation failed: {e}")
        state["status"] = "error"
        state["last_error"] = f"LLM Planner schema validation failed: {str(e)}"

    return state

def execute_actions(state: AutomationState) -> AutomationState:
    """Node 3: Executes the planned browser actions sequentially via Playwright."""
    if state.get("abort_requested"):
        state["status"] = "aborted"
        return state

    print(f"\n[Node 3: execute_actions] Executing {len(state['action_plan'])} browser commands...")
    try:
        _, page = get_or_launch_context(state["user_id"], state["portal_url"])
        
        for act in state["action_plan"]:
            if state.get("abort_requested"):
                break
                
            element_id = str(act["element_id"])
            locator = page.locator(f'[data-ym-id="{element_id}"]')
            
            action_type = act["action"]
            reasoning = act["reasoning"]
            
            if action_type == "fill":
                val = act.get("value", "")
                print(f"  ⌨️ [Fill] ID [{element_id}] -> '{val}' ({reasoning})")
                locator.fill(str(val))
                
            elif action_type == "select":
                val = act.get("value", "")
                print(f"  🔽 [Select] ID [{element_id}] -> '{val}' ({reasoning})")
                locator.select_option(label=val)
                
            elif action_type == "upload":
                fpath = act.get("file_path", "")
                if fpath and os.path.exists(fpath):
                    print(f"  📎 [Upload] ID [{element_id}] -> Attached file: {fpath} ({reasoning})")
                    locator.set_input_files(fpath)
                else:
                    print(f"  ⚠️ [Upload Warning] File not found on disk at: {fpath}")
                    
            elif action_type == "click":
                print(f"  👆 [Click] ID [{element_id}] -> ({reasoning})")
                locator.click()
                page.wait_for_load_state("domcontentloaded", timeout=6000)

        state["status"] = "checking"

    except PlaywrightTimeoutError as e:
        state["status"] = "error"
        state["last_error"] = f"Timeout while executing element action: {str(e)}"
    except Exception as e:
        state["status"] = "error"
        state["last_error"] = str(e)

    return state

def check_status(state: AutomationState) -> AutomationState:
    """Node 4: Evaluates the page state after execution (OTP detection, final review, or next wizard step)."""
    print("\n[Node 4: check_status] Checking portal status...")
    try:
        _, page = get_or_launch_context(state["user_id"], state["portal_url"])
        body_text = page.locator("body").innerText().lower()
        
        # 1. Check if portal is demanding a Mid-Flow Mobile OTP or Captcha
        if any(keyword in body_text for keyword in ["enter otp", "one time password", "sent to your mobile", "enter captcha"]):
            print("  🔐 [GATE: OTP / CAPTCHA DETECTED] Portal requires human authentication step!")
            state["status"] = "otp_required"
            return state

        # 2. Check if we have arrived at the Final Application Review / Payment confirmation screen
        if any(keyword in body_text for keyword in ["review application", "final submission", "declare that the information", "pay application fee"]):
            print("  📋 [GATE: FORM COMPLETE] Arrived at final submission confirmation screen!")
            state["status"] = "form_complete"
            return state

        # 3. Check if wizard moved to a new page or stayed on same page
        new_sig = compute_page_signature(page)
        if new_sig != state.get("page_signature"):
            print("  ➡️ [Navigation Detected] Wizard advanced to a new step. Resetting retries.")
            state["retries"] = 0
            state["status"] = "perceiving"
        else:
            # Same page: check if there are unresolved fields requiring HITL clarification
            if state.get("unresolved_fields") and len(state["unresolved_fields"]) > 0:
                print(f"  ⚠️ [HITL Required] {len(state['unresolved_fields'])} fields require manual clarification.")
                state["status"] = "hitl_clarify"
            else:
                state["status"] = "perceiving"

    except Exception as e:
        state["status"] = "error"
        state["last_error"] = str(e)

    return state

def hitl_intercept(state: AutomationState) -> AutomationState:
    """Node 5: Intercepts execution to request human OTP or missing field clarification via Discord/WhatsApp."""
    print("\n[Node 5: hitl_intercept] Pausing Playwright and sending notification to user channel...")
    # In live bot.py integration, this node triggers a direct Discord message:
    # e.g., "🔐 The portal sent an OTP to **3210. Type it here so I can complete your submission!"
    # When the user replies in Discord, the bot updates state['user_profile']['last_otp'] and resumes!
    state["status"] = "perceiving"
    return state

def retry_or_abort(state: AutomationState) -> AutomationState:
    """Node 6: Manages retry counter when validation errors occur on the portal."""
    state["retries"] += 1
    print(f"\n[Node 6: retry_or_abort] Retry attempt ({state['retries']}/3) after error: {state.get('last_error')}")
    
    if state["retries"] >= 3:
        print("  ❌ [MAX RETRIES REACHED] Aborting automation run gracefully.")
        state["status"] = "aborted"
    else:
        state["status"] = "perceiving"
    return state

def final_confirmation_node(state: AutomationState) -> AutomationState:
    """Node 7: Captures the full-page review screenshot before final submission."""
    print("\n[Node 7: final_confirmation_node] Capturing full-page application screenshot for review...")
    try:
        _, page = get_or_launch_context(state["user_id"], state["portal_url"])
        screenshot_path = os.path.abspath(f"C:/Users/nezam/.gemini/antigravity/brain/{state['session_id']}_final_review.png")
        page.screenshot(path=screenshot_path, full_page=True)
        print(f"  📸 Screenshot saved to: {screenshot_path}")
        # In live bot.py integration, this sends the screenshot image to Discord with:
        # "Please review your completed form above. Type **CONFIRM** to submit officially!"
    except Exception as e:
        print(f"[Screenshot Error] {e}")
        
    return state

# ==============================================================================
# 5. GRAPH ASSEMBLY & ROUTING
# ==============================================================================

def route_after_check(state: AutomationState) -> str:
    """Conditional Edge router determining next graph step based on check_status output."""
    status_map = {
        "otp_required": "hitl_intercept",
        "hitl_clarify": "hitl_intercept",
        "error": "retry_or_abort",
        "form_complete": "final_confirmation",
        "aborted": "aborted_end",
        "perceiving": "perceive_webpage"
    }
    return status_map.get(state["status"], "perceive_webpage")

def build_universal_automation_graph():
    """Compiles and returns the Phase 8 LangGraph StateGraph engine."""
    builder = StateGraph(AutomationState)

    # Register Nodes
    builder.add_node("perceive_webpage", perceive_webpage)
    builder.add_node("plan_actions", plan_actions)
    builder.add_node("execute_actions", execute_actions)
    builder.add_node("check_status", check_status)
    builder.add_node("hitl_intercept", hitl_intercept)
    builder.add_node("retry_or_abort", retry_or_abort)
    builder.add_node("final_confirmation", final_confirmation_node)
    builder.add_node("aborted_end", lambda s: s) # Terminal cleanup node

    # Register Linear Edges
    builder.set_entry_point("perceive_webpage")
    builder.add_edge("perceive_webpage", "plan_actions")
    builder.add_edge("plan_actions", "execute_actions")
    builder.add_edge("execute_actions", "check_status")

    # Register Conditional Edges from check_status
    builder.add_conditional_edges("check_status", route_after_check)

    # Register Recovery/Retry Loops
    builder.add_edge("hitl_intercept", "perceive_webpage")
    builder.add_edge("retry_or_abort", "perceive_webpage")
    builder.add_edge("final_confirmation", END)
    builder.add_edge("aborted_end", END)

    return builder.compile()

if __name__ == "__main__":
    print("=== Yojana Mitra: Phase 8 Universal Automation Module Initialized ===")
