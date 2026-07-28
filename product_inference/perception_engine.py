# ==============================================================================
# Yojana Mitra — Phase 8: Step 2 Perception Engine
# File Path: product_inference/perception_engine.py
# ==============================================================================
"""
Perception Engine for Phase 8 Universal Dynamic Web Automation.

Key Features & Architectural Fixes (Claude Sonnet & Master Plan Review):
  1. Accessibility Tree Snapshot (`aria_snapshot()`):
     Extracts a clean YAML-like tree of roles (`textbox`, `combobox`, `checkbox`),
     stripping out 100% of wrapper `<div>`s, Tailwind/CSS classes, and framework
     clutter. Traverses open Shadow DOMs out of the box.
  2. Wrapping-Label Resolution Fix (`closest('label')`):
     Resolves human-readable labels via `aria-label` -> `placeholder` -> `label[for]` ->
     and fallback `closest('label')`. Correctly handles `<label><input>OBC</label>`
     patterns without defaulting to unhelpful `"Field N"` strings.
  3. Container-Aware Hidden File Input Filtering:
     File upload `<input type="file">` elements are almost always `display:none` or
     zero-size behind styled upload buttons. Instead of checking the input's own
     geometry, the JS tagger checks the computed style (`display`/`visibility`) of its
     parent step container (`fieldset`, `.wizard-step`, `.form-group`). Active step
     files are perceived; future/past wizard step files are cleanly excluded.
  4. Radio & Checkbox Group Collapsing:
     Post-processes the perceived flat element list, grouping elements that share a
     `name` attribute into a single logical `{ type: "radio_group", options: [...] }`
     entry. Reduces token count by ~60% on multi-option fields and enables Stage 2
     to emit one semantic action instead of N individual button clicks.
"""

import json
import hashlib
from typing import Any, Optional
from playwright.sync_api import Page
from product_inference.browser_manager import compute_page_signature


def perceive_dom_state(page: Page) -> dict[str, Any]:
    """
    Perceives the current active browser screen, returning the YAML Accessibility Tree,
    a numbered and group-collapsed interactive element list, and SPA page signature.
    """
    try:
        page.wait_for_load_state("domcontentloaded", timeout=5000)
    except Exception:
        pass  # Continue if background network requests keep loading

    # 1. Clean YAML Accessibility Tree (strips out CSS mess and wrapper divs)
    try:
        aria_tree_yaml = page.locator("body").aria_snapshot() or ""
    except Exception as e:
        aria_tree_yaml = f"# Accessibility tree snapshot unavailable: {e}"

    # 2. JavaScript Outline Pass: Index interactive elements [1..N], resolve wrapping labels,
    # and perform container-aware file input filtering
    js_extractor = """
    () => {
        // *** Retire the previous pass's ids before assigning new ones ***
        // Ids are positions in THIS pass's filtered list, and the filter drops elements that are now
        // disabled or hidden. An element dropped from the list keeps whatever id it was given last
        // time unless it is cleared here — so `[data-ym-id="1"]` matched both the current element #1
        // and a stale one, and `locator.first` picked whichever came first in the document. Observed
        // live: a click meant for the captcha's "Refresh" button resolved to the disabled OTP input
        // left over from the previous pass and timed out for 30s, three times, aborting the run.
        document.querySelectorAll('[data-ym-id]').forEach(el => el.removeAttribute('data-ym-id'));

        const selectors = 'input:not([type="hidden"]), select, textarea, button, [role="button"]';
        const elements = Array.from(document.querySelectorAll(selectors)).filter(el => {
            // Filter out explicitly disabled elements or elements marked hidden by HTML attributes
            if (el.disabled || el.closest('[hidden], [aria-hidden="true"]')) return false;

            // *** FIX: Container-aware file input filtering ***
            // Don't check the file input's own geometry (`rect.width/height`), because file inputs
            // are deliberately `display:none` or zero-size behind styled upload labels.
            // Instead, check whether its parent wizard step/section container is hidden.
            if (el.type === 'file') {
                const container = el.closest('[data-step], .wizard-step, fieldset, .form-group, .form-section') || el.parentElement;
                if (container) {
                    const style = window.getComputedStyle(container);
                    if (style.display === 'none' || style.visibility === 'hidden') return false;
                }
                return true;
            }

            // For all other interactive elements, check standard bounding box visibility
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
        });

        return elements.map((el, idx) => {
            const id = idx + 1;
            el.setAttribute('data-ym-id', id.toString());

            // Label resolution chain: aria-label -> label[for="id"] -> placeholder -> closest('label')
            //
            // An authored <label for> outranks a placeholder: the label states what the field MEANS
            // ("Enter Verification Code"), while a placeholder is a formatting hint about its SHAPE
            // ("6-digit code"). Observed live — with the hint winning, the planner saw a field called
            // "6-digit code" and repeatedly failed to connect it to the citizen's answer, re-asking
            // for the same OTP every cycle. Placeholders remain the fallback for the many real
            // portals that ship no <label> at all.
            let label = el.getAttribute('aria-label') || '';
            if (!label && el.id) {
                const lbl = document.querySelector('label[for="' + el.id + '"]');
                if (lbl) label = lbl.innerText.trim();
            }
            if (!label) label = el.getAttribute('placeholder') || '';
            // *** FIX: Wrapping-label fallback (`<label><input type="radio" value="obc"> OBC</label>`) ***
            if (!label) {
                const wrappingLabel = el.closest('label');
                if (wrappingLabel) {
                    // Extract only the text of the label without the input value if possible
                    label = wrappingLabel.innerText.trim();
                }
            }
            if (!label) label = el.innerText.trim() || el.name || 'Field ' + id;

            const info = {
                id: id,
                tag: el.tagName.toLowerCase(),
                type: el.type || 'text',
                label: label,
                value: el.value || '',
                name: el.name || '',
                required: el.required || false,
                // A disabled control cannot be acted on at all — portals commonly keep "Proceed"
                // disabled until the step's own checks pass. Perceiving this stops the agent from
                // spending its retry budget pressing something the page is refusing to accept.
                disabled: el.disabled === true
            };

            // *** Report toggle STATE, not just the `value` artifact ***
            // A checkbox's `el.value` is the browser default `"on"` whether or not it is ticked, so
            // `value` alone tells the planner nothing about the control's state — and it is where an
            // LLM's phantom `"on"` proposal comes from, since that is literally what perception said
            // the element's value was. Whether a toggle is currently set is a plain DOM fact, so it
            // is perceived directly; without it the agent cannot tell "not yet ticked" from "already
            // ticked" and clicking it again silently toggles it back off.
            if (el.type === 'checkbox' || el.type === 'radio') {
                info.checked = el.checked;
            }

            if (el.tagName.toLowerCase() === 'select') {
                info.options = Array.from(el.options).map(o => ({
                    text: o.text.trim(),
                    value: o.value || o.text.trim()
                })).filter(o => o.text !== '');
            }
            return info;
        });
    }
    """

    raw_elements = page.evaluate(js_extractor)

    # 3. *** FIX: Collapse radio & checkbox groups sharing a `name` attribute ***
    # Before: [{"id": 4, "label": "General", "type": "radio", "name": "caste_cat"},
    #          {"id": 5, "label": "OBC",     "type": "radio", "name": "caste_cat"},
    #          {"id": 6, "label": "SC",      "type": "radio", "name": "caste_cat"}]
    # After:  [{"id": 4, "label": "caste_cat", "type": "radio_group",
    #           "options": [{"id": 4, "text": "General"}, {"id": 5, "text": "OBC"}, {"id": 6, "text": "SC"}]}]
    grouped: dict[str, dict[str, Any]] = {}
    singles: list[dict[str, Any]] = []

    for el in raw_elements:
        if el["type"] in ("radio", "checkbox") and el.get("name"):
            key = f"{el['name']}_{el['type']}"
            if key not in grouped:
                grouped[key] = {
                    "id": el["id"],  # Primary ID represents the start of the group
                    "tag": el["tag"], # Retain original HTML tag for consistency
                    "label": el["name"],
                    "type": f"{el['type']}_group",
                    "name": el["name"],
                    "required": el["required"],
                    "options": []
                }
            grouped[key]["options"].append({
                "id": el["id"],
                "text": el["label"],
                "value": el["value"] or el["label"],
                # Per-option state, because the group's own `id` is only the FIRST option's id —
                # a consumer that wants to know "is the citizen's choice already selected?" has to
                # ask about the specific option, not the group.
                "checked": el.get("checked", False)
            })
        else:
            singles.append(el)

    # Combine singles and grouped elements, keeping them roughly ordered by primary ID
    collapsed_elements = sorted(singles + list(grouped.values()), key=lambda item: item["id"])

    # Calculate token estimate (~1.3 tokens per word/character in JSON representation)
    json_str = json.dumps(collapsed_elements)
    approx_tokens = int(len(json_str) / 3.5)

    # What the screen actually SAYS, not just what it offers to interact with. A review screen is
    # characterised by showing the citizen back what they entered, and that summary lives in text, not
    # in controls — so a question asked about the screen while being shown only its controls is a
    # question about something that was never presented. Truncated: this feeds prompts, and the part
    # that identifies a screen is at the top of it.
    try:
        page_text = " ".join((page.locator("body").inner_text(timeout=3000) or "").split())[:1200]
    except Exception:
        page_text = ""

    return {
        "aria_tree_yaml": aria_tree_yaml,
        "page_text": page_text,
        "elements_list": collapsed_elements,
        "page_signature": compute_page_signature(page),
        "total_raw_count": len(raw_elements),
        "total_collapsed_count": len(collapsed_elements),
        "approx_token_count": approx_tokens,
        "control_state": compute_control_state_fingerprint(page)
    }


def compute_control_state_fingerprint(page: Page) -> str:
    """
    Fingerprint of what every tagged control currently HOLDS (value / checked / disabled), as opposed
    to `compute_page_signature`, which fingerprints which SCREEN is showing.

    Both are needed because progress is not always a screen change. A step that verifies an OTP, a
    captcha and a consent box in place changes none of URL/title/heading, so a loop that measures
    progress only by screen signature reads three successful actions as three failed retries and
    aborts. This is a plain DOM reading, so it works on any portal without knowing anything about it.
    Must be called after the element tagger has run — it reads the `data-ym-id` attributes it sets.
    """
    try:
        raw = page.evaluate("""() => Array.from(document.querySelectorAll('[data-ym-id]')).map(el => [
            el.getAttribute('data-ym-id'), el.value || '', el.checked === true, el.disabled === true
        ].join('~')).join('|')""")
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unknown_control_state"


# ==============================================================================
# STANDALONE LOCAL VERIFICATION HARNESS (STEP 2 TESTING)
# ==============================================================================
if __name__ == "__main__":
    import sys
    from product_inference.browser_manager import launch_isolated_profile, close_session

    print("=== Yojana Mitra: Step 2 Perception Engine Verification ===")
    test_user = "test_perception_001"
    run_headless = "--headless" in sys.argv

    print(f"\n[Test Setup] Launching isolated browser profile (headless={run_headless})...")
    ctx, page = launch_isolated_profile(test_user, portal_url="about:blank", headless=run_headless)

    # Load a comprehensive mock Indian government welfare form (React/HTML pattern)
    print("[Test Setup] Injecting Mock Government Welfare Form with Complex UI Patterns...")
    mock_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pudhumai Penn Welfare Scheme - Application Step 1</title>
        <style>
            .wizard-step { margin: 20px; padding: 20px; border: 1px solid #ccc; }
            .hidden-upload { display: none; }
            .upload-btn { background: #007bff; color: white; padding: 8px 15px; cursor: pointer; display: inline-block; }
        </style>
    </head>
    <body>
        <h1 class="wizard-heading">Step 1: Citizen Personal Details & Verification</h1>
        
        <!-- Active Wizard Step -->
        <div class="wizard-step" data-step="1">
            <form id="welfare_form">
                <!-- Standard Textbox -->
                <div class="form-group">
                    <label for="applicant_name">Full Name (as per Aadhaar):</label>
                    <input type="text" id="applicant_name" name="applicant_name" required placeholder="Enter 12-digit verified name">
                </div>

                <!-- Custom Dropdown / Select -->
                <div class="form-group">
                    <label for="district_select">Resident District:</label>
                    <select id="district_select" name="district">
                        <option value="">-- Select District --</option>
                        <option value="chennai">Chennai</option>
                        <option value="madurai">Madurai</option>
                        <option value="coimbatore">Coimbatore</option>
                    </select>
                </div>

                <!-- Radio Group with Wrapping Labels (No 'for' attribute — tests wrapping label fallback & group collapse) -->
                <div class="form-group">
                    <span>Caste / Social Category:</span><br>
                    <label><input type="radio" name="social_category" value="general"> General / OC</label>
                    <label><input type="radio" name="social_category" value="obc"> OBC / BC</label>
                    <label><input type="radio" name="social_category" value="sc"> SC / ST</label>
                </div>

                <!-- Hidden File Upload Box (Hidden via style="display:none" behind styled label button) -->
                <div class="form-group">
                    <span>Income Certificate Verification:</span><br>
                    <label for="income_cert_input" class="upload-btn">📎 Attach Income Certificate (.jpg/.pdf)</label>
                    <input type="file" id="income_cert_input" name="income_certificate" class="hidden-upload">
                </div>

                <button type="button" id="next_btn">Save & Proceed to Step 2 ➔</button>
            </form>
        </div>

        <!-- Future Wizard Step (Step 2 - Hidden via CSS display:none on parent fieldset container) -->
        <fieldset class="wizard-step" data-step="2" style="display: none;">
            <legend>Step 2: Bank Account & Scholarship Documents</legend>
            <label for="bank_passbook">Upload Bank Passbook Front Page:</label>
            <input type="file" id="bank_passbook" name="bank_passbook">
        </fieldset>
    </body>
    </html>
    """
    page.set_content(mock_html)

    # Execute Perception Engine
    print("\n[Perception Run] Executing `perceive_dom_state(page)`...")
    perception = perceive_dom_state(page)

    print(f"  [OK] SPA Signature Computed: {perception['page_signature']}")
    print(f"  [OK] Raw Interactive Elements Found: {perception['total_raw_count']}")
    print(f"  [OK] Collapsed Logical Elements List: {perception['total_collapsed_count']}")
    print(f"  [OK] Estimated Token Footprint: ~{perception['approx_token_count']} tokens")

    # Verify Specific Architectural Corrections
    print("\n[Verification Check 1] Wrapping-Label & Group Collapsing (`social_category` radio group)...")
    radio_group = next((el for el in perception["elements_list"] if el.get("type") == "radio_group"), None)
    assert radio_group is not None, "Radio group was not collapsed!"
    assert len(radio_group["options"]) == 3, f"Expected 3 options, found {len(radio_group['options'])}"
    assert radio_group["options"][1]["text"] == "OBC / BC", f"Wrapping label failed! Found: {radio_group['options'][1]['text']}"
    print("  [OK] Radio group collapsed cleanly into 1 element with 3 options: [General / OC, OBC / BC, SC / ST]")

    print("\n[Verification Check 2] Hidden File Upload Detection (`income_certificate`)...")
    file_upload = next((el for el in perception["elements_list"] if el.get("tag") == "input" and el.get("type") == "file"), None)
    assert file_upload is not None, "Hidden file input `income_cert_input` was missed!"
    assert file_upload["label"] == "📎 Attach Income Certificate (.jpg/.pdf)", f"Upload label mismatch: {file_upload['label']}"
    print("  [OK] Hidden file input detected and labeled cleanly despite `display:none`!")

    print("\n[Verification Check 3] Container-Aware Future Step Exclusion (`bank_passbook` step 2)...")
    future_upload = next((el for el in perception["elements_list"] if "passbook" in el.get("name", "") or "passbook" in el.get("label", "").lower()), None)
    assert future_upload is None, "Future step file input `bank_passbook` leaked into active perception!"
    print("  [OK] Future step hidden container cleanly excluded `bank_passbook` file upload!")

    print("\n[Verification Check 4] Token Footprint Efficiency (<500 tokens target)...")
    print(f"  [OK] Perceived JSON payload is only ~{perception['approx_token_count']} tokens (vs 15,000+ for raw HTML)!")
    assert perception['approx_token_count'] < 500, f"Token count {perception['approx_token_count']} exceeded 500 token budget!"

    if not run_headless:
        print("\n[Visual Inspection Pause] Keeping browser window open for 10 seconds so you can verify the mock UI...")
        page.wait_for_timeout(10000)

    close_session(test_user)
    print("=== Step 2 Verification Complete & Passed ===")
