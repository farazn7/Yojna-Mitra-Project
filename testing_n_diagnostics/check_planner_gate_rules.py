"""Pure-function checks on derive_actions' gate rules — no browser, no LLM.

Run: python testing_n_diagnostics/check_planner_gate_rules.py
"""
import sys
sys.path.insert(0, r"C:\Users\nezam\OneDrive\Yojna-Mitra-Project")

from product_inference.llm_planner import ClassifiedField, derive_actions, gate_identity

el = {
    2: {"tag": "button", "name": "", "label": "Back", "type": "button"},
    3: {"tag": "button", "name": "", "label": "Save & Proceed to Step 2", "type": "submit"},
}
gates = [
    ClassifiedField(id=2, tag="button", label="Back", category="ACTION_GATE", advances_form=True, reasoning="r"),
    ClassifiedField(id=3, tag="button", label="Save & Proceed to Step 2", category="ACTION_GATE", advances_form=True, reasoning="r"),
]
a = derive_actions([], [], gates, el)
print("no memory ->", [x["element_id"] for x in a], a[0]["gate_identity"])
assert a[0]["element_id"] == 2

blocked = {gate_identity(el[2])}
b = derive_actions([], [], gates, el, blocked)
print("with memory ->", [x["element_id"] for x in b])
assert [x["element_id"] for x in b] == [3]

# all gates blocked -> presses nothing, never guesses
c = derive_actions([], [], gates, el, {gate_identity(el[2]), gate_identity(el[3])})
print("all blocked ->", c)
assert c == []
print("PASS")

# --- press memory: same screen, nothing new entered -> next candidate gate gets its turn
el2 = {
    1: {"tag": "button", "name": "", "label": "Refresh", "type": "button"},
    2: {"tag": "button", "name": "", "label": "Verify", "type": "button"},
    5: {"tag": "input", "name": "captcha", "label": "Type the code", "type": "text"},
}
g2 = [
    ClassifiedField(id=1, tag="button", label="Refresh", category="ACTION_GATE", advances_form=True, reasoning="r"),
    ClassifiedField(id=2, tag="button", label="Verify", category="ACTION_GATE", advances_form=True, reasoning="r"),
]
first = derive_actions([], [], g2, el2)
assert [a["element_id"] for a in first] == [1]
tried = {gate_identity(el2[1])}
second = derive_actions([], [], g2, el2, None, tried)
print("after one press ->", [a["element_id"] for a in second])
assert [a["element_id"] for a in second] == [2]

# new input this cycle -> the tried gate is fair game again
fill = ClassifiedField(id=5, tag="input", label="Type the code", category="FILLABLE_CONFIDENT",
                       proposed_value="ABC12", vault_key="captcha", reasoning="r")
third = derive_actions([fill], [], g2, el2, None, tried)
print("with new input ->", [a["element_id"] for a in third])
assert [a["element_id"] for a in third] == [5, 1]

# a disabled gate is never pressed
el3 = {2: {"tag": "button", "name": "", "label": "Proceed", "type": "submit", "disabled": True}}
g3 = [ClassifiedField(id=2, tag="button", label="Proceed", category="ACTION_GATE", advances_form=True, reasoning="r")]
assert derive_actions([], [], g3, el3) == []
print("PASS 2")

# --- a field's own Verify is pressed in the same batch as the fill, ahead of the step gate
el4 = {
    1: {"tag": "input", "name": "otp", "label": "Enter Verification Code", "type": "text", "value": ""},
    2: {"tag": "button", "name": "", "label": "Verify OTP", "type": "button"},
    3: {"tag": "button", "name": "", "label": "Save & Proceed to Step 4", "type": "submit"},
}
f = ClassifiedField(id=1, tag="input", label="Enter Verification Code", category="FILLABLE_CONFIDENT",
                    proposed_value="123456", vault_key="otp", reasoning="r")
gv = ClassifiedField(id=2, tag="button", label="Verify OTP", category="ACTION_GATE",
                     advances_form=True, commits_field_id=1, reasoning="r")
gs = ClassifiedField(id=3, tag="button", label="Save & Proceed to Step 4", category="ACTION_GATE",
                     advances_form=True, reasoning="r")
out = derive_actions([f], [], [gv, gs], el4)
print("fill+verify ->", [(a["element_id"], a["action"]) for a in out])
assert [a["element_id"] for a in out] == [1, 2]

# nothing left to confirm -> the step gate is pressed, and the verify gate is never mistaken for it
el4[1]["value"] = "123456"
out2 = derive_actions([], [], [gv, gs], el4, None, {gate_identity(el4[2])})
print("after verify ->", [a["element_id"] for a in out2])
assert [a["element_id"] for a in out2] == [3]

# a text box Stage 1 wrongly called a gate is never clicked
bad = ClassifiedField(id=1, tag="input", label="Enter Verification Code", category="ACTION_GATE",
                      advances_form=True, reasoning="r")
assert derive_actions([], [], [bad], el4) == []
print("PASS 3")

# --- a screen with NO per-field confirm control: fill, then straight to the step gate
el5 = {
    1: {"tag": "input", "name": "captcha", "label": "Type the code", "type": "text", "value": ""},
    2: {"tag": "button", "name": "", "label": "Continue", "type": "submit"},
}
f5 = ClassifiedField(id=1, tag="input", label="Type the code", category="FILLABLE_CONFIDENT",
                     proposed_value="8RA7H", vault_key="captcha", reasoning="r")
g5 = ClassifiedField(id=2, tag="button", label="Continue", category="ACTION_GATE",
                     advances_form=True, reasoning="r")
out5 = derive_actions([f5], [], [g5], el5)
print("no verify button ->", [(a["element_id"], a["action"]) for a in out5])
assert [a["element_id"] for a in out5] == [1, 2]
print("PASS 4")

# --- a control that only re-issues the screen's own content is never pressed
el6 = {
    1: {"tag": "button", "name": "", "label": "Refresh", "type": "button"},
    2: {"tag": "button", "name": "", "label": "Save & Proceed", "type": "submit"},
}
g6r = ClassifiedField(id=1, tag="button", label="Refresh", category="ACTION_GATE",
                      advances_form=True, regenerates_content=True, reasoning="r")
g6f = ClassifiedField(id=2, tag="button", label="Save & Proceed", category="ACTION_GATE",
                      advances_form=True, reasoning="r")
out6 = derive_actions([], [], [g6r, g6f], el6)
print("refresh present ->", [a["element_id"] for a in out6])
assert [a["element_id"] for a in out6] == [2]
assert derive_actions([], [], [g6r], el6) == []
print("PASS 5")

