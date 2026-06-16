import json
import pytest
from matcher import match_persona_to_schemes, deduplicate_schemes

# ── Setup: Load data once for the whole test suite ──────────────────────────

@pytest.fixture(scope="session")
def loaded_schemes():
    with open('schemes.json', 'r', encoding='utf-8') as f:
        # Re-use the deduplicate function so tests match reality
        return deduplicate_schemes(json.load(f))

@pytest.fixture(scope="session")
def loaded_personas():
    with open('personas/personas.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Helper function to generate clean names for the pytest output
def get_persona_ids():
    with open('personas/personas.json', 'r', encoding='utf-8') as f:
        personas = json.load(f)
    return [p['id'] for p in personas]

# ── The Test ─────────────────────────────────────────────────────────────────

# This decorator tells pytest to run this function 10 times, 
# passing in a different persona dictionary each time.
@pytest.mark.parametrize("persona_index", range(10), ids=get_persona_ids())
def test_persona_gets_valid_results(persona_index, loaded_personas, loaded_schemes):
    # Grab the specific persona for this test run
    persona = loaded_personas[persona_index]
    
    # Run our heuristic matcher
    top_schemes = match_persona_to_schemes(persona, loaded_schemes, top_n=5)
    
    # Assert 1: We actually got results back. 
    # (A completely empty return means our logic is too strict or broken).
    assert len(top_schemes) > 0, f"No schemes found for persona: {persona['id']}"
    
    # Assert 2: We didn't get more than the requested top_n limit.
    assert len(top_schemes) <= 5, f"Returned {len(top_schemes)} schemes, expected max 5"
    
    # Assert 3: The results have the correct structure and valid scores.
    for result in top_schemes:
        assert 'scheme_name' in result, "Result is missing scheme_name"
        assert 'score' in result, "Result is missing score"
        assert result['score'] > 0, "Matched scheme has a score of 0 (should be disqualified)"