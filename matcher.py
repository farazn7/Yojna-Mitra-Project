"""
matcher.py — Persona-to-Scheme eligibility matcher
Strategy: structured rule extraction + weighted scoring + hard disqualifiers
"""

import json
import re

# ── Helpers ──────────────────────────────────────────────────────────────────

def _text(scheme, *keys):
    return " ".join(str(scheme.get(k, "") or "") for k in keys).lower()

def _parse_age_range(text):
    """Extract (min_age, max_age) from eligibility text."""
    # "18 to 40 years" / "18 – 40 years"
    rng = re.search(r'(\d+)\s*(?:to|–|-)\s*(\d+)\s*year', text)
    if rng:
        return int(rng.group(1)), int(rng.group(2))
    # "18 years and below" / "upto 40 years" / "not exceed 40 years"
    mx = re.search(r'(?:upto|up to|not exceed|below|and below|maximum of?)\s*(\d+)\s*year', text)
    # "minimum of 18 years" / "at least 18 years"
    mn = re.search(r'(?:minimum of?|at least|minimum age of?)\s*(\d+)\s*year', text)
    # "should be 18 years and above"
    mn2 = re.search(r'(\d+)\s*years?\s*(?:and above|or above|and older)', text)
    min_age = int(mn.group(1)) if mn else (int(mn2.group(1)) if mn2 else None)
    max_age = int(mx.group(1)) if mx else None
    return min_age, max_age

def _parse_income_limit(text):
    """Return highest income limit in rupees found in text, or None."""
    amounts = re.findall(r'₹\s*([\d,]+)', text)
    if not amounts:
        return None
    vals = [int(a.replace(",", "")) for a in amounts if len(a.replace(",","")) >= 4]
    return max(vals) if vals else None

def deduplicate_schemes(schemes):
    seen, out = set(), []
    for s in schemes:
        name = s.get("scheme_name","")
        if name not in seen:
            seen.add(name)
            out.append(s)
    return out

# ── Scoring ──────────────────────────────────────────────────────────────────

FEMALE_EXCL_PHRASES = [
    'must be a woman', 'only women', 'only female', 'should be a woman',
    'working woman', 'exclusively for women', 'girl child',
    'only for women', 'for women only',
]
GENDER_FEMALE_WORDS = [
    'girl', 'woman', 'women', 'female', 'widow', 'bride',
    'maternity', 'pregnant', 'lactating',
]
GENDER_MALE_WORDS = ['only men', 'only male', 'men only']

CASTE_MAP = {
    "Scheduled Caste (SC)":       ['scheduled caste', 'sc/', '/sc', 'adi dravidar', 'sc/st', 'scst', 'sc '],
    "Scheduled Tribe (ST)":       ['scheduled tribe', 'st/', '/st', 'tribal', 'sc/st', 'scst', ' st '],
    "Other Backward Class (OBC)": ['other backward', 'obc', 'mbc', 'bc/', 'most backward', 'backward class'],
    "General": [],
}

OCCUPATION_MAP = {
    "farmer":    ['farmer', 'agriculture', 'cultivator', 'landless', 'crop', 'kissan', 'uzhavar'],
    "fishermen": ['fisherm', 'fisher', 'fishing', 'boat owner', 'marine'],
    "artists":   ['artist', 'kalaimamani', 'folk artist', 'music', 'dance', 'bharatanatyam', 'kalai'],
    "student":   ['student', 'scholar', 'studying', 'scholarship', 'education', 'degree'],
    "weavers":   ['weaver', 'handloom', 'powerloom', 'silk'],
}


def score_scheme(persona, scheme):
    """Returns (score, reasons). score=0 means disqualified."""
    raw   = _text(scheme, 'eligibility_rules')
    score = 0
    reasons = []

    gender    = persona['demographics']['gender']
    age       = persona['demographics']['age']
    is_student = persona['professional_and_educational']['student']
    emp_status = (persona['professional_and_educational'].get('employment_status') or '').lower()
    occupation = (persona['professional_and_educational'].get('occupation') or '').lower().strip()
    is_govt    = persona['professional_and_educational']['government_employee']
    is_bpl     = persona['financial_status']['below_poverty_line']
    in_distress= persona['financial_status']['economic_distress']
    is_disabled= persona['health_and_disability']['differently_abled']
    dis_pct    = persona['health_and_disability']['disability_percentage'] or 0
    caste      = persona['demographics']['caste']
    marital    = persona['demographics']['marital_status']
    residence  = persona['location']['residence']
    is_minority= persona['demographics']['minority']

    # ── HARD DISQUALIFIERS ────────────────────────────────────────────────────

    if gender not in ('Female',) and any(p in raw for p in FEMALE_EXCL_PHRASES):
        return 0, ["disqualified: female-only scheme"]

    if gender not in ('Male',) and any(p in raw for p in GENDER_MALE_WORDS):
        return 0, ["disqualified: male-only scheme"]

    min_age, max_age = _parse_age_range(raw)
    if min_age and age < min_age:
        return 0, [f"disqualified: age {age} < min {min_age}"]
    if max_age and age > max_age:
        return 0, [f"disqualified: age {age} > max {max_age}"]

    student_only = re.search(r'(must be|should be|only for)\s+(a\s+)?student', raw)
    if student_only and not is_student:
        return 0, ["disqualified: students-only scheme"]

    # Occupation-exclusive schemes: if scheme is strictly for an occupation the persona doesn't have
    occ_excl_map = {
        'fisherm': 'fishermen',
        'weaver':  'weavers',
    }
    for kw, occ_name in occ_excl_map.items():
        excl = re.search(rf'(must be|should be|only for|applicable to)\s+(a\s+)?{kw}', raw)
        if excl and occupation != occ_name and occ_name not in occupation:
            return 0, [f"disqualified: {occ_name}-only scheme"]

    if 'should not be a government' in raw and is_govt:
        return 0, ["disqualified: excludes govt employees"]

    if is_disabled:
        pct_req = re.search(r'(\d+)\s*%\s*(?:or more|and above|disability)', raw)
        if pct_req and dis_pct < int(pct_req.group(1)):
            return 0, [f"disqualified: disability {dis_pct}% < required {pct_req.group(1)}%"]

    # ── POSITIVE SCORING ─────────────────────────────────────────────────────

    # 1. Gender
    if gender == 'Female' and any(w in raw for w in GENDER_FEMALE_WORDS):
        score += 3; reasons.append("gender: female-oriented scheme")

    # 2. Age
    if min_age and max_age and min_age <= age <= max_age:
        score += 2; reasons.append(f"age in range {min_age}–{max_age}")
    elif min_age and not max_age and age >= min_age:
        score += 1

    # 3. Occupation (check persona occupation against scheme keywords)
    occ_matched = False
    for occ_key, keywords in OCCUPATION_MAP.items():
        if occ_key in occupation or occ_key in emp_status:
            if any(kw in raw for kw in keywords):
                score += 4; reasons.append(f"occupation: {occ_key}"); occ_matched = True; break

    # Student gets its own boost even if not caught above
    if is_student and not occ_matched and any(w in raw for w in OCCUPATION_MAP['student']):
        score += 3; reasons.append("student scheme match")

    # 4. BPL / income
    income_limit = _parse_income_limit(raw)
    if is_bpl and any(w in raw for w in ['below poverty', 'bpl', 'destitute', 'indigent', 'poorest']):
        score += 3; reasons.append("BPL match")
    elif income_limit and income_limit <= 120000 and (is_bpl or in_distress):
        score += 1; reasons.append(f"low income limit ₹{income_limit:,}")

    # 5. Caste
    for caste_key, keywords in CASTE_MAP.items():
        if caste == caste_key and keywords:
            if any(kw in raw for kw in keywords):
                score += 3; reasons.append(f"caste: {caste_key}")
            break

    # 6. Disability
    if is_disabled and any(w in raw for w in [
        'differently abled', 'disabled', 'disability', 'visually impaired',
        'deaf', 'locomotor', 'divyangjan', 'handicap'
    ]):
        score += 4; reasons.append("disability match")

    # 7. Minority
    if is_minority and any(w in raw for w in [
        'minority', 'minorities', 'muslim', 'christian', 'sikh', 'jain', 'buddhist'
    ]):
        score += 2; reasons.append("minority match")

    # 8. Transgender
    if gender == 'Transgender' and any(w in raw for w in [
        'transgender', 'trans', 'thirunangai', 'aravani'
    ]):
        score += 4; reasons.append("transgender match")

    # 9. Marital status
    if marital == 'Widowed' and any(w in raw for w in ['widow', 'widower', 'destitute women']):
        score += 3; reasons.append("widowed match")

    # 10. Residence
    if residence == 'Rural' and any(w in raw for w in ['rural', 'village', 'panchayat']):
        score += 1; reasons.append("rural match")
    elif residence == 'Urban' and any(w in raw for w in ['urban', 'city', 'corporation', 'municipality']):
        score += 1; reasons.append("urban match")

    # 11. Unemployment
    if 'unemployed' in emp_status and any(w in raw for w in ['unemployed', 'unemployment', 'job seeker']):
        score += 2; reasons.append("unemployment match")

    return score, reasons


def match_persona_to_schemes(persona, schemes, top_n=5):
    results = []
    for scheme in schemes:
        score, reasons = score_scheme(persona, scheme)
        if score > 0:
            results.append({
                "scheme_name": scheme["scheme_name"],
                "portal_url":  scheme.get("portal_url", ""),
                "score":       score,
                "reasons":     reasons,
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open("schemes.json", "r", encoding="utf-8") as f:
        schemes = deduplicate_schemes(json.load(f))
    with open("personas/personas.json", "r", encoding="utf-8") as f:
        personas = json.load(f)

    print(f"Loaded {len(schemes)} unique schemes, {len(personas)} personas\n")
    all_results = {}

    for persona in personas:
        pid = persona["id"]
        top = match_persona_to_schemes(persona, schemes, top_n=5)
        all_results[pid] = top

        print(f"{'='*60}")
        print(f"Persona: {pid}")
        if not top:
            print("  No matching schemes found.")
        for i, r in enumerate(top, 1):
            print(f"  {i}. [{r['score']}pts] {r['scheme_name']}")
            print(f"       → {', '.join(r['reasons'])}")
        print()

    with open("match_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("Saved to match_results.json")