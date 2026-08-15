# ==============================================================================
# Yojana Mitra — Declarative Profile Form
# File Path: core_inference/profile_form.py
# ==============================================================================
"""
The 13 profile fields, their options, and their coercion rules — in one place.

There are no profile-collecting graph nodes. CLAUDE.md describes "13 sequential nodes
(`ask_gender`, `ask_age`, ...)"; they do not exist. Onboarding is done entirely by
platform form UI which writes the whole profile in a single
`db.update_user_state(user_id, "PROFILE_COMPLETE", data)` call. That means every new
surface has to ship its own form, and every surface was therefore re-deciding what
"income" parses to.

The widgets genuinely cannot be shared — Telegram uses inline keyboards, Discord uses
`discord.ui.Select` and a `Modal`, the browser uses HTML. The *field list, the options,
and the coercion* can be, and are, here.

Field names and coercion behaviour are lifted verbatim from the existing forms so that
profiles written before this module still read back identically:
  - booleans arrive as the strings "true"/"false" and are coerced to real bools
    (`bot.py:97`, `bot_telegram.py:233-234`)
  - `disability_percentage` is derived, not asked (`bot.py:189`, `bot_telegram.py:236`)
  - income strips `,` `₹` and spaces before parsing (`bot.py:226`, `bot_telegram.py:432`)
"""

from typing import Any, Optional

# ==============================================================================
# 1. THE FORM
# ==============================================================================
# (field, step_label, question_text, keyboard_rows or None for free text)
# Kept in this exact shape because bot_telegram.py indexes it positionally and builds
# InlineKeyboardMarkup straight from `keyboard_rows`.

FORM_FLOW: list[tuple[str, str, str, Optional[list]]] = [
    ("name",                "1/13",
     "👤 *Step 1/13* — What is your *full name*?",
     None),  # text input
    ("gender",              "2/13",
     "👤 *Step 2/13* — What is your *gender*?",
     [[("👨 Male", "Male"), ("👩 Female", "Female"), ("🌈 Other", "Other")]]),
    ("residence",           "3/13",
     "🏘 *Step 3/13* — Where do you *live*?",
     [[("🌾 Rural", "Rural"), ("🏙 Urban", "Urban")]]),
    ("caste",               "4/13",
     "📋 *Step 4/13* — What is your *caste category*?",
     [[("General", "General"), ("OBC", "OBC")], [("SC", "SC"), ("ST", "ST")]]),
    ("marital_status",      "5/13",
     "💍 *Step 5/13* — What is your *marital status*?",
     [[("Single", "Single"), ("Married", "Married")], [("Widowed", "Widowed"), ("Divorced", "Divorced")]]),
    ("occupation",          "6/13",
     "💼 *Step 6/13* — What is your *occupation*?",
     [[("🌾 Farmer", "Farmer"), ("📚 Student", "Student")],
      [("💼 Salaried", "Salaried"), ("🏪 Business Owner", "Business Owner")],
      [("🧵 Artisan", "Artisan"), ("❌ Unemployed", "Unemployed")],
      [("✏️ Type your own...", "__text__")]]),
    ("differently_abled",   "7/13",
     "♿ *Step 7/13* — Are you *differently abled*?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("minority",            "8/13",
     "🕌 *Step 8/13* — Do you belong to a *minority community*?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("below_poverty_line",  "9/13",
     "🏠 *Step 9/13* — Do you have a *BPL card* (Below Poverty Line)?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("economic_distress",   "10/13",
     "📉 *Step 10/13* — Are you facing *economic distress*?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("government_employee", "11/13",
     "🏗 *Step 11/13* — Are you a *government employee*?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("age",                 "12/13",
     "🔢 *Step 12/13* — Please *type your age* in years:",
     None),  # text input
    ("income",              "13/13",
     "💰 *Step 13/13* — Please type your annual family *income in ₹* (e.g. 45000):",
     None),  # text input
]

FORM_FIELDS: list[str] = [f[0] for f in FORM_FLOW]

# `is_profile_complete` in graph.py:109-112 gates on these five only — not all 13.
# Duplicated here as a constant rather than imported, to avoid core_inference.profile_form
# importing graph.py (which opens a Postgres pool at module scope).
REQUIRED_FOR_COMPLETE = {"gender", "age", "income", "caste", "occupation"}

# Sentinel emitted by the occupation keyboard's "Type your own..." button.
FREE_TEXT_SENTINEL = "__text__"

_BOOLEAN_FIELDS = {
    "differently_abled", "minority", "below_poverty_line",
    "economic_distress", "government_employee",
}
_INT_FIELDS = {"age", "income"}
_TITLE_CASE_FIELDS = {"name", "occupation"}


# ==============================================================================
# 2. INTROSPECTION (for surfaces that build their own widgets)
# ==============================================================================

def field_kind(field: str) -> str:
    """'choice' if the field has preset options, else 'text'."""
    for name, _label, _q, rows in FORM_FLOW:
        if name == field:
            return "choice" if rows else "text"
    raise KeyError(f"unknown profile field: {field}")


def options_for(field: str) -> list[tuple[str, str]]:
    """Flatten a field's keyboard rows to [(label, value)]. Empty for text fields.

    The row grouping only matters to Telegram's keyboard layout; a web <select> or a
    Discord dropdown wants the flat list.
    """
    for name, _label, _q, rows in FORM_FLOW:
        if name == field:
            if not rows:
                return []
            return [pair for row in rows for pair in row]
    raise KeyError(f"unknown profile field: {field}")


def question_for(field: str) -> str:
    for name, _label, question, _rows in FORM_FLOW:
        if name == field:
            return question
    raise KeyError(f"unknown profile field: {field}")


def plain_question(field: str) -> str:
    """`question_for` with Markdown emphasis stripped, for surfaces that render plain text."""
    return question_for(field).replace("*", "").replace("\\", "")


def as_json_schema() -> list[dict]:
    """The whole form as JSON-serialisable dicts, for the browser to render."""
    out = []
    for name, label, question, rows in FORM_FLOW:
        out.append({
            "field": name,
            "step": label,
            "question": plain_question(name),
            "kind": "choice" if rows else "text",
            "numeric": name in _INT_FIELDS,
            "options": [
                {"label": lbl, "value": val}
                for lbl, val in options_for(name)
                if val != FREE_TEXT_SENTINEL
            ],
            "allows_free_text": any(
                val == FREE_TEXT_SENTINEL for _lbl, val in options_for(name)
            ),
        })
    return out


# ==============================================================================
# 3. COERCION & VALIDATION
# ==============================================================================

def coerce_field(field: str, raw: Any) -> tuple[bool, Any]:
    """Coerce one raw form value to its stored type.

    Returns `(True, value)` or `(False, citizen_facing_error_message)`.

    This is the single place age/income parsing lives. Previously it existed inline in
    `bot_telegram.py:425-442` and again in `ProfileModal.on_submit` (`bot.py:225-233`),
    which is how "income" ended up accepting slightly different input on each platform.
    """
    if field in _BOOLEAN_FIELDS:
        if isinstance(raw, bool):
            return True, raw
        return True, str(raw).strip().lower() == "true"

    if field in _INT_FIELDS:
        if isinstance(raw, int) and not isinstance(raw, bool):
            cleaned = str(raw)
        else:
            cleaned = (
                str(raw)
                .replace(",", "")
                .replace("₹", "")   # ₹
                .replace(" ", "")
                .strip()
            )
        if not cleaned.isdigit():
            if field == "age":
                return False, "❌ Please enter a valid number for age (e.g. 22):"
            return False, "❌ Please enter a valid number for income (e.g. 45000):"
        value = int(cleaned)
        if field == "age" and not (0 < value < 120):
            return False, "❌ Please enter a realistic age between 1 and 119:"
        return True, value

    text = str(raw).strip()
    if not text:
        return False, "❌ This can't be empty. Please type an answer:"
    if field in _TITLE_CASE_FIELDS:
        return True, text.title()
    return True, text


def apply_field(data: dict, field: str, raw: Any) -> tuple[bool, Optional[str]]:
    """Coerce and write one field into `data` in place, including derived fields.

    Returns `(True, None)` on success or `(False, error_message)`.

    `disability_percentage` is derived here rather than asked, matching `bot.py:189` and
    `bot_telegram.py:236`. It is not in FORM_FLOW and must not be prompted for.
    """
    ok, value = coerce_field(field, raw)
    if not ok:
        return False, value

    data[field] = value
    if field == "differently_abled":
        data["disability_percentage"] = 40 if value else 0
    return True, None


def answered_fields(data: dict) -> list[str]:
    """The FORM_FLOW fields this citizen has actually answered, in form order.

    `False` counts as answered — "No, I don't have a BPL card" is an answer, and the
    scheme personalization prompt reads it as one. Only `None` and `""` are unanswered.
    """
    return [f for f in FORM_FIELDS if data.get(f) not in (None, "")]


def missing_fields(data: dict) -> list[str]:
    """The complement of `answered_fields`, in form order."""
    return [f for f in FORM_FIELDS if data.get(f) in (None, "")]


def progress(data: dict) -> dict:
    """Answered/total counts for the surfaces' "complete your profile" nudge.

    `answered` is deliberately measured against all 13 fields, not the 5 that
    `graph.py:is_profile_complete` gates routing on. Those five decide whether the graph
    will *run* a scheme search; all thirteen decide how well it is personalised
    (`hybrid_rag.py:148-158` adds one line to the LLM's context per answered field). A
    citizen sitting at 5/13 is served, but served worse, and the nudge is what tells them.
    """
    answered = answered_fields(data)
    missing = missing_fields(data)
    return {
        "answered": len(answered),
        "total": len(FORM_FIELDS),
        "missing": missing,
        "usable": not validate_profile(data),
    }


def validate_profile(data: dict) -> list[str]:
    """Return a list of problems with a whole profile dict. Empty means usable.

    Checks the five keys `graph.py:is_profile_complete` actually gates on, not all 13 —
    a profile missing `marital_status` still routes correctly.
    """
    problems = []
    for field in sorted(REQUIRED_FOR_COMPLETE):
        if data.get(field) in (None, ""):
            problems.append(f"missing required field: {field}")
    if "age" in data and not isinstance(data["age"], int):
        problems.append("age must be an integer")
    if "income" in data and not isinstance(data["income"], int):
        problems.append("income must be an integer")
    return problems


def defaults() -> dict:
    """The defaults Discord's Modal pre-fills before steps 2 and 3 (`bot.py:242-246`).

    Kept so a partially-filled web form saves the same shape a partially-filled Discord
    form does.
    """
    return {
        "gender": "Unknown",
        "differently_abled": False, "disability_percentage": 0,
        "minority": False, "below_poverty_line": False,
        "economic_distress": False, "government_employee": False,
        "residence": "Urban", "marital_status": "Single",
    }


# ==============================================================================
# 4. SUMMARY RENDERING
# ==============================================================================

def format_profile_summary(data: dict, bold: str = "**") -> str:
    """The saved-profile confirmation, identical across surfaces apart from emphasis.

    Telegram's Markdown uses `*bold*`, Discord's uses `**bold**`. Everything else in
    `bot.py:128-144` and `bot_telegram.py:125-141` was already character-for-character
    the same.

    `income` is rendered defensively: the previous implementations both called
    `int(g('income', 0))`, which raises if income was ever stored as a non-numeric
    string, taking down the whole save confirmation.
    """
    g = data.get
    b = bold

    try:
        income_str = f"₹{int(g('income', 0)):,}"
    except (TypeError, ValueError):
        income_str = f"₹{g('income', '—')}"

    def yn(field: str) -> str:
        return "Yes" if g(field) else "No"

    return (
        f"🎉 {b}Profile Saved Successfully!{b}\n\n"
        f"• {b}Name:{b} {g('name', '—')}\n"
        f"• {b}Gender:{b} {g('gender', '—')}\n"
        f"• {b}Age:{b} {g('age', '—')} yrs\n"
        f"• {b}Income:{b} {income_str}\n"
        f"• {b}Caste:{b} {g('caste', '—')}\n"
        f"• {b}Occupation:{b} {g('occupation', '—')}\n"
        f"• {b}Residence:{b} {g('residence', '—')}\n"
        f"• {b}Marital Status:{b} {g('marital_status', '—')}\n"
        f"• {b}Differently Abled:{b} {yn('differently_abled')}\n"
        f"• {b}Minority:{b} {yn('minority')}\n"
        f"• {b}BPL Card:{b} {yn('below_poverty_line')}\n"
        f"• {b}Economic Distress:{b} {yn('economic_distress')}\n"
        f"• {b}Govt Employee:{b} {yn('government_employee')}\n\n"
        f"You're all set! Ask me to {b}find schemes{b} for you now! 🎯"
    )
