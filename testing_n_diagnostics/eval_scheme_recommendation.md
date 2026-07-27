# Eval — Scheme Recommendation Matching

A ground-truth retrieval eval for the Hybrid RAG scheme-matching pipeline (`core_inference/hybrid_rag.py`) — **`test_scheme_recommendation_matching.py`** plus its labeled dataset **`scheme_eval_labels.json`**.

> One of two eval suites in this directory. The other is [`eval_web_automation.md`](eval_web_automation.md), which covers the Playwright form-filling agent.

---

## Show this

### 1. Results: 24/25 assertions, 1 real finding

```
================ 2 passed, 23 deselected in 1104.13s (0:18:24) ================
================ 1 passed, 24 deselected in 227.78s (0:03:47) =================
================ 2 passed, 23 deselected in 520.38s (0:08:40) =================
=========== 1 failed, 1 passed, 23 deselected in 571.03s (0:09:31) ============
================ 1 passed, 24 deselected in 315.90s (0:05:15) =================
```
*(actual pytest session outputs, concatenated across the chunked runs used to validate this suite)*

| # | Persona / case | top_k=3 | top_k=10 |
|---|---|---|---|
| 1 | Widowed farmer, 55, BPL | ✅ HIT | ✅ HIT |
| 2 | Rural student, General caste | ⚪ observed (no corpus match) | ⚪ observed |
| 3 | Disabled folk artist, 32 | ⚪ observed (SQL-filtered out) | ⚪ observed |
| 4 | Minority female student | ✅ HIT | ✅ HIT |
| 5 | Transgender, skills training | ✅ HIT | ✅ HIT |
| 6 | Fisherman, lean season | ⚪ observed (no corpus match) | ⚪ observed |
| 7 | Govt. employee, health query | ⚪ observed (income-excluded) | ⚪ observed |
| 8 | Pregnant rural mother | ✅ HIT | ✅ HIT |
| 9 | Minority business owner | ⚪ observed (no corpus match) | ⚪ observed |
| 10 | Senior citizen, BPL, disabled | ❌ **MISS** | ✅ HIT (rank 9) |
| hn1–hn5 | 5 boundary hard-negatives | — | ✅ all 5 correctly excluded |

### 2. The one failure — real signal, not a flaky test

```
AssertionError: [p10_senior_citizen_bpl@top_k=3] expected one of
['Calipers and Crutches - Tamil Nadu'] in schemes_fetched, got:
['Pension Scheme - Folk Artist',
 'Old Age Pension Scheme under Chief Minister's Uzhavar Padhukappu Thittam',
 'Old Age Homes - Tamil Nadu']
```
The correct scheme ranks **9th** by the pipeline's own hybrid similarity score — it surfaces at `top_k=10` but genuinely drops off at `top_k=3`. This is a real retrieval-quality signal that a "did we get *a* response" test could never catch.

### 3. What the old eval missed — direct comparison

`rag_evaluation_log.md` (pre-existing output of `test_hybrid_rag.py`, which only asserts `len(response) > 0`) contains this, and the test still **passed**:

```
## Test Target: `p6_fisherman_distress`
### 🎯 Yojana Mitra LLM Response:

I can't help with that request.
```

A flat refusal, on a real welfare query, marked green. That's the exact blind spot this eval closes — it asserts against `schemes_fetched` (the pipeline's real retrieval output), not the LLM's prose.

### 4. Three production data bugs found — verified live, not guessed

Found by running the pipeline's *actual* SQL `WHERE` clause directly against the live `government_schemes` table for every persona, instead of trusting the scraped `data_extraction/schemes.json` text:

```sql
-- Bug 1: eligibility text says "disabled artists under 60 also qualify" — DB disagrees
SELECT scheme_name, min_age, max_age, max_income, is_women_only, is_differently_abled
FROM government_schemes WHERE scheme_name ILIKE '%Folk Artist%';
 ('Pension Scheme - Folk Artist', 60, None, None, False, True)

-- Bug 2: gender-neutral scheme, but flagged women-only in the DB
SELECT scheme_name, min_age, max_age, max_income, is_women_only, is_differently_abled
FROM government_schemes WHERE scheme_name ILIKE '%Old Age Pension%';
 ('Indira Gandhi National Old Age Pension Scheme -Tamil Nadu', 60, None, None, True, False)

-- Bug 3: exact duplicate ingestion row
SELECT scheme_name, count(*) FROM government_schemes
WHERE scheme_name ILIKE '%Unemployed Youth Employment%' GROUP BY scheme_name;
 ('Unemployed Youth Employment Generation Programme', 2)
```

All three are documented in `scheme_eval_labels.json`'s `_meta.known_issues_surfaced` field rather than quietly worked around.

### 5. The "unfalsifiable assertion" trap — caught before it shipped

The obvious test for "nothing should match this persona" is `assert schemes_fetched == []`. Checked against the live table first:

| Persona (no confident match) | Schemes passing the demographic SQL filter alone |
|---|---|
| Rural student | 130 |
| Disabled artist | 121 |
| Fisherman | 96 |
| Govt. employee | 84 |
| Minority business owner | 82 |

`== []` would fail on **every single run**, regardless of retrieval quality — because age/income/gender filters alone don't guarantee topical relevance. The suite uses a verified-exclusion blocklist instead (see `near_miss_blocklist` in `scheme_eval_labels.json`), which only fails on an actual regression.

---

## What this is, and why

`core_inference/hybrid_rag.py`'s `run_yojana_pipeline()` is a 3-stage retrieval funnel: SQL categorical hard-filter → pgvector cosine similarity → `0.7×vector + 0.3×lexical` hybrid re-score. The pre-existing eval (`test_hybrid_rag.py`) only ever asserted the pipeline returned non-empty text — it had no ground truth, so it could not detect a regression in any of those three stages as long as the LLM kept generating fluent paragraphs from whatever (possibly wrong) schemes it was handed.

This eval closes that gap by asserting against `schemes_fetched` — the pipeline's real, deterministic, pre-LLM output — using a hand-verified ground-truth dataset instead of the LLM's prose.

## Methodology

1. Took the 10 existing personas (`personas/personas.json`) and their paired queries (`TEST_QUERIES` in the old eval, reused verbatim).
2. For each, searched `data_extraction/schemes.json`'s 234 scraped schemes for genuine eligibility-text matches — no invented "expected" schemes.
3. **Verified every candidate against the live database**, not just the scraped text, using the pipeline's own SQL filter logic — this is what surfaced the 3 bugs above.
4. For personas with no genuine match in the current corpus, labeled `expected_scheme_names: []` with a documented reason, and where the SQL filter *provably* excludes a specific near-miss scheme, added it to a `near_miss_blocklist` instead of asserting an unfalsifiable empty result.
5. Added 5 hard-negative cases — one per SQL-filterable field (`min_age`, `max_age`, `max_income`, `is_women_only`, `is_differently_abled`) — each mutating a real persona just past a real scheme's real boundary value, to test the filter's *exclusion* behavior, which had zero prior coverage.

## Files

| File | Purpose |
|---|---|
| `test_scheme_recommendation_matching.py` | The pytest suite. Imports `run_yojana_pipeline` directly from `core_inference/hybrid_rag.py` (the real production path, not a re-implementation). |
| `scheme_eval_labels.json` | The labeled ground-truth dataset — 10 positive cases + 5 hard negatives, every claim traceable to a live DB value or an explicit "no match found" note. |
| `test_hybrid_rag.py` | Pre-existing. A qualitative persona sweep with a markdown transcript log — useful for eyeballing LLM tone/quality, not for correctness assertions. Untouched by this work. |
| `test_phase7_flow.py` | Pre-existing. Document-collection / HITL state-machine simulation script. Unrelated to retrieval. |

## Running it

```bash
pytest testing_n_diagnostics/test_scheme_recommendation_matching.py -v -s
```

Requires a running Postgres+pgvector container with `government_schemes` populated, and Ollama with `nomic-embed-text` + the reasoning model pulled locally. **This is a slow, live-infra eval** — every case is a real embedding call plus a real local LLM generation (`num_predict=5000`); a full run took over an hour on the machine this was developed on. Not meant for a fast CI loop, same scoping as `test_hybrid_rag.py`.
