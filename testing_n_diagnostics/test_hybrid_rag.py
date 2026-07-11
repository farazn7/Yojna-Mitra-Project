import pytest
import json
import psycopg2
import ollama
import os
import re
import time

# ── Dynamic Query Mapping ──────────────────────────────────────────────────
# Realistic colloquial queries paired strictly to each human persona ID
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

def get_personas():
    with open('personas/personas.json', 'r', encoding='utf-8') as f:
        return json.load(f)

LOG_FILE = "rag_evaluation_log.md"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def log_to_markdown(persona_id, query, retrieved_schemes, response):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"## Test Target: `{persona_id}`\n")
        f.write(f"**🗣️ Incoming WhatsApp Message:** _{query}_\n\n")
        f.write("### 🔍 Hybrid Database Filtering Matches:\n")
        if not retrieved_schemes:
            f.write("_No schemes cleared the hard demographic criteria filters._\n\n")
        for idx, row in enumerate(retrieved_schemes, 1):
            f.write(f"{idx}. **{row[0]}** (Distance: {row[4]:.4f}) | URL: <{row[1]}>\n")
        f.write(f"\n### 🎯 Yojana Mitra LLM Response:\n\n{response}\n")
        f.write("\n" + "="*80 + "\n\n")

# ── Safe Database Search ───────────────────────────────────────────────────
def perform_hybrid_search(persona, user_query, top_k=3):
    conn = psycopg2.connect(
        dbname="postgres", user="postgres", password="mysecretpassword",
        host="localhost", port="5432"
    )
    cur = conn.cursor()

    # Vector Embedding generation for search vector
    vector_response = ollama.embeddings(model='nomic-embed-text', prompt=user_query)
    query_vector = vector_response['embedding']

    # CRUCIAL: Safe income mapping for the SQL evaluation block
    # If the user has a null income value (like our students), we pass 0 
    # so they safely pass the SQL requirement: (max_income >= user_income)
    safe_income = persona['income'] if persona['income'] is not None else 0

    sql_query = """
        SELECT scheme_name, portal_url, details, eligibility_rules,
               (embedding <=> %s::vector) as distance
        FROM government_schemes
        WHERE 
            (min_age IS NULL OR min_age <= %s)
            AND (max_age IS NULL OR max_age >= %s)
            AND (max_income IS NULL OR max_income >= %s)
            AND (is_women_only = FALSE OR %s = 'Female')
            AND (is_differently_abled = FALSE OR %s = TRUE)
        ORDER BY distance ASC
        LIMIT %s;
    """
    
    cur.execute(sql_query, (
        query_vector,
        persona['age'],
        persona['age'],
        safe_income,
        persona['gender'],
        persona['differently_abled'],
        top_k
    ))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

# ── Thermal-Optimized Inference ────────────────────────────────────────────
def sanitize_response(response_text, context_urls):
    """Remove URLs that were not provided in the retrieval context."""
    found_urls = re.findall(r'https?://[^\s)\]]+', response_text)
    for url in found_urls:
        clean_url = url.rstrip('.,;:')
        if clean_url not in context_urls:
            response_text = response_text.replace(url, "[verified URL not available in provided context]")
    return response_text


def generate_rag_response(user_query, retrieved_schemes):
    if not retrieved_schemes:
        return "Namaste. Based on your current profile details, I could not find an exact match among the active government schemes. Could you please share more information about your situation?"

    context_blocks = []
    context_urls = set()
    
    for idx, row in enumerate(retrieved_schemes, 1):
        scheme_name, url, details, eligibility = row[0], row[1], row[2], row[3]
        if url:
            context_urls.add(url)
        context_blocks.append(
            f"--- SCHEME {idx} ---\nName: {scheme_name}\nURL: {url}\nDetails: {details}\nEligibility: {eligibility}\n"
        )
    context_string = "\n".join(context_blocks)

    system_prompt = f"""
    You are Yojana Mitra, a helpful WhatsApp AI assistant for Indian citizens.
    
    TASK: Read the user's message, review ALL provided schemes, and recommend the matches.

    STRICT RULES:
    1. SYNTHESIZE ALL DATA: You MUST evaluate and discuss ALL the schemes provided in the CONTEXT DATA below. Do not ignore a scheme just because it is at the bottom of the list. Explain how each one can be utilized by the user.
    2. THE INSENSITIVITY FILTER: You must silently hide schemes that are actively morbid or conflicting with a positive life event. 
       - Example: If a user is happily pregnant and asking for nutrition/delivery support, DO NOT mention miscarriage or abortion assistance schemes.
       - Do not apologize for excluding a scheme; just silently drop it.
    3. NO HALLUCINATIONS: Do not invent URLs, facts, or external schemes. Base your response purely on the provided context.

    CONTEXT DATA:
    {context_string}
    """

    response = ollama.chat(
        model='llama3.1',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ],
        options={
            'temperature': 0.1,
            'num_predict': 600,   
            'num_thread': 4
        }
    )
    final_response = response['message']['content']
    return sanitize_response(final_response, context_urls)

# ── The Automated Test Framework Loop ──────────────────────────────────────
@pytest.mark.parametrize("persona", get_personas(), ids=[p['id'] for p in get_personas()])
def test_hybrid_rag_pipeline(persona):
    # 1. Fetch targeted custom message string
    user_query = TEST_QUERIES[persona['id']]
    
    # 2. Run Database Query (Very fast, zero thermal footprint)
    retrieved_schemes = perform_hybrid_search(persona, user_query)
    
    # 3. Generate Answer via local Llama (Heavy resource task)
    final_response = generate_rag_response(user_query, retrieved_schemes)
    
    # 4. Save results dynamically to the Markdown audit log
    log_to_markdown(persona['id'], user_query, retrieved_schemes, final_response)
    
    # 5. Core Assertions
    assert final_response is not None
    assert len(final_response) > 0

    # --- THE ANTI-OVERHEAT MEASURE ---
    # We enforce a deliberate 4-second rest sleep block at the tail end of each 
    # persona generation execution cycle. This breaks continuous resource pooling 
    # and allows hardware fan states to clear the built-up processor junction heat.
    print(f"\n Cool-down interval active for target thread: {persona['id']}...")
    time.sleep(4)