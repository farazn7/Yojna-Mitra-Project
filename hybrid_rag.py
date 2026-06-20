import psycopg2
import ollama
import re

def setup_db_connection():
    return psycopg2.connect(
        dbname="postgres", 
        user="postgres",
        password="mysecretpassword",
        host="localhost",
        port="5432"
    )

def rerank_schemes(persona, user_query, rows):
    """Boost schemes that match the user's exact need and penalize unrelated ones."""
    def normalize(text):
        return re.findall(r"[a-zA-Z]+", text.lower())

    query_terms = set(normalize(user_query))
    query_lower = user_query.lower()
    scored = []

    # Persona-specific intent hints
    persona_hint_terms = set()
    if persona.get('marital_status') == 'Widowed':
        persona_hint_terms.update({'widow', 'widowed', 'pension', 'financial', 'assistance'})
    if persona.get('student'):
        persona_hint_terms.update({'student', 'scholarship', 'education', 'college', 'fee'})
    if persona.get('differently_abled') or persona.get('disability_percentage'):
        persona_hint_terms.update({'disability', 'disabled', 'artist', 'grant', 'instrument'})
    if persona.get('government_employee'):
        persona_hint_terms.update({'government', 'employee', 'health', 'insurance', 'hospital'})

    for row in rows:
        scheme_name, url, details, eligibility, distance = row
        text = f"{scheme_name} {details} {eligibility}".lower()
        scheme_terms = set(normalize(text))

        overlap = len((query_terms | persona_hint_terms) & scheme_terms)

        # Strong boosts for exact need phrases
        phrase_boost = 0
        if any(term in text for term in ['widow', 'widowed']) and any(term in query_lower for term in ['widow', 'widowed']):
            phrase_boost += 6
        if any(term in text for term in ['pension', 'financial assistance', 'cash']) and any(term in query_lower for term in ['financial', 'pension', 'cash', 'loan', 'assistance']):
            phrase_boost += 5
        if any(term in text for term in ['scholarship', 'education', 'college', 'fee']) and any(term in query_lower for term in ['scholarship', 'education', 'college', 'fee']):
            phrase_boost += 6
        if any(term in text for term in ['disability', 'disabled', 'artist', 'instrument', 'grant']) and any(term in query_lower for term in ['disabled', 'disability', 'artist', 'instrument', 'grant']):
            phrase_boost += 6
        if any(term in text for term in ['health', 'insurance', 'hospital']) and any(term in query_lower for term in ['health', 'insurance', 'hospital']):
            phrase_boost += 6

        # Penalize obviously unrelated death/funeral schemes
        if any(term in text for term in ['funeral', 'death']) and not any(term in query_lower for term in ['death', 'funeral', 'died', 'dead']):
            phrase_boost -= 10

        score = overlap * 12 + phrase_boost - (distance * 250)
        scored.append((score, row))

    return [row for _, row in sorted(scored, key=lambda item: item[0], reverse=True)]


def perform_hybrid_search(persona, user_query, top_k=7):
    """
    Combines the user's hard demographic profile with their semantic chat query.
    """
    conn = setup_db_connection()
    cur = conn.cursor()

    print(f"🧠 1. Embedding live user query: '{user_query}'...")
    # Convert the WhatsApp message into a math vector
    vector_response = ollama.embeddings(
        model='nomic-embed-text', 
        prompt=user_query
    )
    query_vector = vector_response['embedding']

    # Handle null income safely for the SQL math
    # If income is null (like for a student), treat it as 0 so they pass the max_income check
    safe_income = persona['income'] if persona['income'] is not None else 0

    print("🔎 2. Running Hybrid SQL Query (Hard Filters + Vector Math)...")
    # THE HYBRID QUERY
    # We use the boolean flags currently in your DB (is_women_only, is_differently_abled)
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
        persona['age'],             # For min_age check
        persona['age'],             # For max_age check
        safe_income,                # For max_income check
        persona['gender'],          # Only pass if scheme is not women-only OR user is female
        persona['differently_abled'],# Only pass if scheme is not disabled-only OR user is disabled
        top_k
    ))
    
    results = cur.fetchall()
    cur.close()
    conn.close()

    # Re-rank the retrieved candidates using explicit relevance cues so the model
    # sees the most likely scheme first, instead of relying only on embedding distance.
    return rerank_schemes(persona, user_query, results)

def generate_rag_response(user_query, retrieved_schemes):
    """
    Forces Llama 3.1 to answer the live chat query using ONLY the filtered government data.
    """
    print("🤖 3. Generating RAG response with Llama 3.1...")
    
    if not retrieved_schemes:
        return "Namaste. Based on your profile and age/income details, I couldn't find an exact match right now. Could you tell me more about what you are looking for?"

    context_blocks = []
    for rank, row in enumerate(retrieved_schemes, 1):
        scheme_name = row[0]
        portal_url = row[1]
        details = row[2]
        rules = row[3]
        
        block = f"--- Scheme {rank}: {scheme_name} ---\n" \
                f"URL: {portal_url}\n" \
                f"Details: {details}\n" \
                f"Rules: {rules}\n"
        context_blocks.append(block)
        
    context_string = "\n".join(context_blocks)

    system_prompt = f"""
    You are Yojana Mitra, a helpful WhatsApp assistant for Indian citizens.
    Answer the user's query using ONLY the government schemes provided in the context below.
    If the context does not contain relevant schemes, say you cannot find a match right now.
    Keep your answer conversational, easy to understand, and always provide the portal URL.
    
    CONTEXT:
    {context_string}
    """

    response = ollama.chat(
        model='llama3.1',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ],
        options={'temperature': 0.2} # Low temperature for factual accuracy
    )
    
    return response['message']['content']

def main():
    # 1. The Database Profile (Persona without the query)
    test_persona = {
        "id": "p2_rural_student",
        "gender": "Male",
        "age": 18,
        "income": None,
        "caste": "General",
        "residence": "Rural",
        "marital_status": "Never Married",
        "differently_abled": False,
        "disability_percentage": None,
        "student": True
    }
    
    # 2. The Live Chat Input (From WhatsApp)
    live_user_query = "I just finished school and need help paying for college fees. Are there any scholarships?"
    
    print("="*60)
    print(f"PROFILE LOADED: {test_persona['id']} | Age: {test_persona['age']} | Income: {test_persona['income']}")
    print(f"LIVE MESSAGE: {live_user_query}")
    print("="*60)
    
    # Execute Hybrid Retrieval
    top_schemes = perform_hybrid_search(test_persona, live_user_query)
    
    # Execute Generation
    final_answer = generate_rag_response(live_user_query, top_schemes)
    
    print("\n" + "="*60)
    print("🎯 FINAL WHATSAPP OUTPUT:")
    print("="*60)
    print(final_answer)

if __name__ == "__main__":
    main()