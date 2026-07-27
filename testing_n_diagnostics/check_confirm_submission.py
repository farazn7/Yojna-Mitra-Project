"""Exercises ONLY the post-CONFIRM submission path: control identification, native dialog
acceptance, result screenshot. Steps 1-3 are driven directly with Playwright (standing in for the
human), because the point here is the final click, not the ReAct loop.

Run: python testing_n_diagnostics/check_confirm_submission.py (needs the mock portal on :5173).
Drives Steps 1-3 directly with Playwright, then checks that only a literal CONFIRM submits.
"""
import os, sys, time
sys.path.insert(0, r"C:\Users\nezam\OneDrive\Yojna-Mitra-Project")
os.chdir(r"C:\Users\nezam\OneDrive\Yojna-Mitra-Project")

from langchain_core.messages import HumanMessage
from product_inference.browser_manager import launch_isolated_profile, get_active_page, run_pw, close_session
from core_inference.graph import handle_automation_response

SID = "confirm_t4"
URL = "http://localhost:5173"
shot = os.path.join("screenshots", f"{SID}_submitted.png")
if os.path.exists(shot):
    os.remove(shot)

run_pw(launch_isolated_profile, SID, portal_url=URL, headless=False)
page = run_pw(get_active_page, SID)

def fill(sel, val):
    run_pw(page.locator(sel).first.fill, val)

def click(text):
    run_pw(page.get_by_role("button", name=text).first.click)
    run_pw(page.wait_for_timeout, 500)

# Step 1
fill("#applicant_name", "Nezam Rahman")
run_pw(page.locator("#state_select").first.select_option, label="Tamil Nadu")
run_pw(page.wait_for_timeout, 400)
run_pw(page.locator("#district_select").first.select_option, label="Chennai")
click("Save & Proceed to Step 2")

# Step 2 — any real file will do; the portal only records its name
os.makedirs("temp_documents", exist_ok=True)
doc = os.path.abspath(os.path.join("temp_documents", "t4_dummy.jpg"))
if not os.path.exists(doc):
    from PIL import Image
    Image.new("RGB", (60, 60), (200, 200, 200)).save(doc)
run_pw(page.locator("#income_cert_input").first.set_input_files, doc)
run_pw(page.wait_for_timeout, 800)
click("Save & Proceed to Step 3")

# Step 3
fill("#otp_input", "123456")
click("Verify")
captcha = "".join(run_pw(page.locator(".captcha-char").all_inner_texts))
fill("#captcha_input", captcha)
click("Verify Captcha")
run_pw(page.locator('input[name="not_a_robot"]').first.check)
click("Save & Proceed to Step 4")
run_pw(page.locator('input[name="declaration"]').first.check)
print("On review screen:", "Review and Submit" in run_pw(page.content))

base = {
    "user_id": SID, "automation_session_id": SID, "automation_status": "awaiting_confirm",
    "application_form_url": URL, "user_profile": {}, "vault_snapshot": {},
}

# 1. A reply that is not CONFIRM must not submit anything.
held = handle_automation_response({**base, "messages": [HumanMessage(content="yes please go ahead")]})
print("\n[non-CONFIRM]", held.get("automation_status"), "|", held["response"].splitlines()[0])
assert held.get("automation_status") in (None, "awaiting_confirm")
assert "Submitted" not in held["response"]
assert not os.path.exists(shot)
assert "Application Submitted Successfully" not in run_pw(page.content)

# 2. CONFIRM submits, accepts the portal's native dialog, and captures the outcome.
done = handle_automation_response({**base, "messages": [HumanMessage(content="CONFIRM")]})
print("[CONFIRM]", done.get("automation_status"), "|", done["response"].splitlines()[0])
time.sleep(1)
content = run_pw(page.content)
print("screenshot written:", os.path.exists(shot))
print("portal shows success:", "Application Submitted Successfully" in content)
assert done.get("automation_status") == "complete", done
assert os.path.exists(shot), "no result screenshot captured"
assert "Application Submitted Successfully" in content, "portal never registered the submission"
print("\nPASS — CONFIRM path submits; a non-CONFIRM reply does not.")
close_session(SID)
