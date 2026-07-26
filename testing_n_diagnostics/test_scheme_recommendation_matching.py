"""
SLOW / LIVE-INFRA EVAL. Do not fold into a fast test loop.

Ground-truth retrieval-correctness check for core_inference/hybrid_rag.py's
run_yojana_pipeline(). Unlike test_hybrid_rag.py (which only asserts a
non-empty response), this asserts the *correct* scheme(s) actually land in
run_yojana_pipeline()'s returned schemes_fetched list, using a labeled
dataset in scheme_eval_labels.json built from live-verified DB values.

Requires:
  - Postgres+pgvector running locally with `government_schemes` populated
    (see core_inference/hybrid_rag.py:setup_db_connection for connection
    params).
  - Ollama running locally with `nomic-embed-text` and
    `hf.co/qwen/qwen3-8b-gguf:q4_k_m` pulled.
  - ~25 live pipeline calls total (10 personas x 2 top_k values, plus 5
    hard-negative cases) -- each involves a live embedding call and an LLM
    generation, so a full run is slow.

Run with:
    pytest testing_n_diagnostics/test_scheme_recommendation_matching.py -v -s
"""
import json
import time
from collections import Counter
from pathlib import Path

import pytest

import sys
import os

# core_inference/hybrid_rag.py prints emoji diagnostics (e.g. "🔍 Hybrid
# Database Filtering Matches:") on every call. Windows' default console
# codepage (cp1252) can't encode them, which otherwise crashes every test
# here before assertions even run. product_inference/test_integration.py
# works around the same issue at its own module top; mirrored here.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_inference.hybrid_rag import run_yojana_pipeline

THIS_DIR = Path(__file__).resolve().parent
LOG_PATH = THIS_DIR / "scheme_eval_log.md"

# Copied verbatim from test_hybrid_rag.py's TEST_QUERIES. Not imported: that
# module has cwd-dependent import-time side effects (deletes
# rag_evaluation_log.md, and its @pytest.mark.parametrize decorator calls
# get_personas() at decoration time using a bare relative path that only
# resolves when cwd == testing_n_diagnostics/).
TEST_QUERIES = {
    "p1_widowed_farmer": "Vanakkam, my husband passed away recently and I am struggling to manage our small family farm in the village alone. Is there any financial help or seed subsidy available for a widow like me?",
    "p2_rural_student": "I just finished school and got admission into a college engineering course, but my family's income is very low. Can I get a government scholarship or education loan to pay my fees?",
    "p3_differently_abled_artist": "Hello, I am a traditional folk artist living in Chennai. I have a 40% physical disability and my art income has completely stopped this month. Are there any government welfare grants or instrument awards for disabled artists?",
    "p4_minority_female_student": "Hi, I am a girl student currently studying in my final year of B.Sc. I completed all my schooling in a government high school. Am I eligible for the monthly stipend scheme or any higher education allowance for girls?",
    "p5_transgender_skills": "I am a transgender individual currently looking for a job. I want to set up my own small business or tailoring shop so I can earn a living. Are there any free skill training programs or entrepreneurship loans for me?",
    "p6_fisherman_distress": "During the annual fishing ban period, we cannot go out into the sea and my family has zero earnings. Does the fisheries department provide any monthly financial relief or lean period cash assistance for fishermen?",
    "p7_govt_employee_health": "I am working as a government school teacher and I need to check the details regarding the New Health Insurance Scheme coverage for my family's medical treatment at an empanelled private hospital.",
    "p8_young_mother_rural": "I am pregnant with my first child and my husband works for daily wages. Are there any nutrition schemes or cash benefits given by the government to poor pregnant mothers during delivery?",
    "p9_minority_business_owner": "I run a small mobile repair shop and belong to a minority community. I want to expand my shop and buy new tools. Can I get a low-interest business loan from the minority development corporation without collateral?",
    "p10_senior_citizen_bpl": "I am an old man over 70 years old. I cannot work anymore due to my age and severe leg problems. I have a BPL card. Is there any monthly old age pension or free walking stick scheme I can apply for?"
}


def load_personas():
    with open(THIS_DIR / "personas" / "personas.json", "r", encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)}


def load_labels():
    with open(THIS_DIR / "scheme_eval_labels.json", "r", encoding="utf-8") as f:
        return json.load(f)


PERSONAS = load_personas()
LABELS = load_labels()


def normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())


def has_expected_hit(expected_names, fetched_names) -> bool:
    normalized_fetched = {normalize(n) for n in fetched_names}
    return any(normalize(e) in normalized_fetched for e in expected_names)


def rank_of(name, fetched_names):
    target = normalize(name)
    for i, n in enumerate(fetched_names, start=1):
        if normalize(n) == target:
            return i
    return None


_POSITIVE_RESULTS = []


@pytest.fixture(scope="session", autouse=True)
def _summary_report():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    yield
    if not _POSITIVE_RESULTS:
        return

    lines = ["# Scheme Recommendation Matching — Hit-Rate Summary\n"]
    for top_k in sorted({r["top_k"] for r in _POSITIVE_RESULTS}):
        rows = [r for r in _POSITIVE_RESULTS if r["top_k"] == top_k]
        scored = [r for r in rows if r["expected"]]
        hits = sum(1 for r in scored if r["hit"])
        lines.append(f"\n## top_k={top_k}\n")
        lines.append(f"Hit rate: {hits}/{len(scored)} ({hits / len(scored) * 100:.0f}%)\n" if scored else "No scored cases.\n")
        for r in rows:
            status = "HIT" if r["hit"] else ("MISS" if r["expected"] else "OBSERVE")
            lines.append(
                f"- `{r['persona_id']}` [{status}] confidence={r['confidence']} "
                f"rank={r['rank']} expected={r['expected']} fetched={r['fetched']}\n"
            )

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.writelines(lines)
    print("\n" + "".join(lines))


POSITIVE_PARAMS = [
    (case, top_k)
    for case in LABELS["positive_cases"]
    for top_k in (3, 10)
]
POSITIVE_IDS = [f"{case['persona_id']}-top{top_k}" for case, top_k in POSITIVE_PARAMS]


@pytest.mark.parametrize("case,top_k", POSITIVE_PARAMS, ids=POSITIVE_IDS)
def test_positive_case(case, top_k):
    persona_id = case["persona_id"]
    persona = PERSONAS[persona_id]
    query = TEST_QUERIES[persona_id]

    _response, fetched = run_yojana_pipeline(persona, query, top_k=top_k)

    dup_counts = Counter(normalize(n) for n in fetched)
    dups = {name: count for name, count in dup_counts.items() if count > 1}
    if dups:
        print(f"[{persona_id}@top_k={top_k}] duplicate scheme_name(s) in schemes_fetched: {dups}")

    try:
        if case["expected_scheme_names"]:
            hit = has_expected_hit(case["expected_scheme_names"], fetched)
            rank = next(
                (r for name in case["expected_scheme_names"] if (r := rank_of(name, fetched)) is not None),
                None,
            )
            _POSITIVE_RESULTS.append({
                "persona_id": persona_id, "top_k": top_k, "hit": hit, "rank": rank,
                "expected": case["expected_scheme_names"], "fetched": fetched,
                "confidence": case["confidence"],
            })
            assert hit, (
                f"[{persona_id}@top_k={top_k}] expected one of {case['expected_scheme_names']} "
                f"in schemes_fetched, got: {fetched}"
            )
        else:
            _POSITIVE_RESULTS.append({
                "persona_id": persona_id, "top_k": top_k, "hit": None, "rank": None,
                "expected": [], "fetched": fetched, "confidence": case["confidence"],
            })
            normalized_fetched = {normalize(n) for n in fetched}
            blocklist = case.get("near_miss_blocklist", [])
            for blocked in blocklist:
                assert normalize(blocked) not in normalized_fetched, (
                    f"[{persona_id}@top_k={top_k}] near-miss scheme '{blocked}' "
                    f"unexpectedly surfaced: {fetched}"
                )
            if not blocklist:
                print(f"[{persona_id}@top_k={top_k}] no confident match, no blocklist — observation only. Retrieved: {fetched}")
    finally:
        time.sleep(3)


@pytest.mark.parametrize("hn", LABELS["hard_negatives"], ids=[h["id"] for h in LABELS["hard_negatives"]])
def test_hard_negative_exclusion(hn):
    base = PERSONAS[hn["base_persona_id"]]
    mutated = {**base, **hn["profile_overrides"]}
    query = TEST_QUERIES[hn["base_persona_id"]]

    try:
        _response, fetched = run_yojana_pipeline(mutated, query, top_k=10)

        normalized_fetched = {normalize(n) for n in fetched}
        assert normalize(hn["excluded_scheme_name"]) not in normalized_fetched, (
            f"[{hn['id']}] expected '{hn['excluded_scheme_name']}' excluded after "
            f"{hn['profile_overrides']} ({hn['boundary_field']} boundary), but found in: {fetched}"
        )
    finally:
        time.sleep(3)
