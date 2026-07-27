"""Full journey: launch -> Steps 1-4 by the agent -> citizen replies CONFIRM -> submission.

Run: python testing_n_diagnostics/check_full_journey.py (needs the mock portal on :5173).
Whole journey: agent does Steps 1-4, citizen replies CONFIRM, portal acknowledgement asserted.
"""
import sys, os, importlib.util as ilu
sys.path.insert(0, r"C:\Users\nezam\OneDrive\Yojna-Mitra-Project"); os.chdir(r"C:\Users\nezam\OneDrive\Yojna-Mitra-Project")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
spec = ilu.spec_from_file_location("_ev", r"testing_n_diagnostics\test_web_automation_react_loop.py")
ev = ilu.module_from_spec(spec); spec.loader.exec_module(ev)
from langchain_core.messages import HumanMessage
from core_inference.graph import handle_automation_response
from product_inference.browser_manager import close_session, get_active_page, run_pw

uid = "full1"
run = ev.run_full_flow_to_gate(uid, ev._make_document_fixtures(uid), [], max_turns=14)
sid = run["session_id"]
print("\n===== TURN HISTORY =====")
for t in run["history"]:
    print(f"  {t['turn']:10} reply={t.get('reply','-'):14} status={t['automation_status']:18} asked={[f.get('name') for f in t.get('unresolved_fields',[])]}")
final = run["final_state"]
print("gate status:", final.get("automation_status"))

shot = os.path.join("screenshots", f"{sid}_submitted.png")
if os.path.exists(shot): os.remove(shot)

if final.get("automation_status") == "awaiting_confirm":
    state = {"user_id": uid, "automation_session_id": sid, "automation_status": "awaiting_confirm",
             "application_form_url": ev.DEV_SERVER_URL, "user_profile": {}, "vault_snapshot": {},
             "messages": [HumanMessage(content="CONFIRM")]}
    out = handle_automation_response(state)
    print("\nafter CONFIRM ->", out.get("automation_status"))
    print(out["response"].splitlines()[0])
    page = get_active_page(sid)
    content = run_pw(page.content) if page else ""
    print("portal acknowledgement shown:", "Application Submitted Successfully" in content)
    print("result screenshot saved:", os.path.exists(shot))
    ok = out.get("automation_status") == "complete" and "Application Submitted Successfully" in content
    print("\n" + ("PASS — full journey Steps 1-4 + CONFIRM submission." if ok else "INCOMPLETE — see above."))
else:
    print("\nDid not reach the confirmation gate; nothing to CONFIRM.")
close_session(sid)
