import psycopg2
import ollama
import re
import difflib

def extract_and_print_thoughts(node_name: str, raw_response: str) -> str:
    """Extracts <think> tags, prints them to the terminal immediately, and returns clean text."""
    match = re.search(r'<think>(.*?)</think>', raw_response, flags=re.DOTALL | re.IGNORECASE)
    
    if match:
        thoughts = match.group(1).strip()
        clean_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
        
        print(f"\n [AGENT THOUGHTS: {node_name}]", flush=True)
        print(f"\033[90m{thoughts}\033[0m", flush=True) 
        print("-" * 50 + "\n", flush=True)
        
        # SAFE GUARDRAIL: If the model only thought and didn't answer, return a fallback
        if not clean_text:
            return "I've analyzed the schemes matching your profile, but I ran out of room to format the response. Could you please try asking your question again?"
            
        return clean_text
    
    # Fallback if no think tags exist but text is empty
    if not raw_response.strip():
        return "Namaste. I processed your request but encountered an empty response generation. Let's try that again."
        
    return raw_response.strip()

def setup_db_connection():
    return psycopg2.connect(
        dbname="postgres", 
        user="postgres",
        password="mysecretpassword",
        host="localhost",
        port="5432"
    )

def sanitize_response(response_text, context_urls):
    """Remove URLs that were not provided in the retrieval context."""
    found_urls = re.findall(r'https?://[^\s)\]]+', response_text)
    for url in found_urls:
        clean_url = url.rstrip('.,;:')
        if clean_url not in context_urls:
            response_text = response_text.replace(url, "[verified URL not available in provided context]")
    return response_text

def run_yojana_pipeline(profile_data, text, conversation_history=None, summary="", top_k=10):
    """
    THE LIVE RUNTIME GATEWAY:
    Processes the user's chat query and database profile metadata 
    through a strict SQL filter, pgvector matching, and structured LLM generation loop.
    """
    user_query = text
    conn = setup_db_connection()
    cur = conn.cursor()

    # 1. Generate live query embedding
    vector_response = ollama.embeddings(model='nomic-embed-text', prompt=user_query)
    query_vector = vector_response['embedding']

    # 2. Extract profile fields safely from the dynamic database data dictionary
    user_age = int(profile_data.get('age', 0))
    user_gender = profile_data.get('gender', 'Other')
    user_disabled = bool(profile_data.get('differently_abled', False))
    
    # Safe income parsing matching our test script defaults
    raw_income = profile_data.get('income')
    safe_income = int(raw_income) if raw_income is not None else 0

    # 3. Hybrid Database Execution Matching
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
        user_age,
        user_age,
        safe_income,
        user_gender,
        user_disabled,
        top_k
    ))
    
    retrieved_schemes = cur.fetchall()
    cur.close()
    conn.close()

    # Sort retrieved schemes by hybrid score (vector distance + string overlap)
    scored_schemes = []
    for row in retrieved_schemes:
        dist = row[4] if row[4] is not None else 1.0
        vec_sim = max(0.0, 1.0 - dist)
        str_sim = difflib.SequenceMatcher(None, user_query.lower(), row[0].lower()).ratio()
        hybrid_score = (vec_sim * 0.7) + (str_sim * 0.3)
        scored_schemes.append((hybrid_score, row))
        
    scored_schemes.sort(key=lambda x: x[0], reverse=True)
    retrieved_schemes = [x[1] for x in scored_schemes]

    # Terminal Pipeline Diagnostics Log
    print("\n" + "="*70)
    print("🔍 Hybrid Database Filtering Matches:")
    if not retrieved_schemes:
        print("  _No schemes cleared the hard demographic criteria filters._")
    for idx, row in enumerate(retrieved_schemes, 1):
        print(f"  {idx}. **{row[0]}** (Distance: {row[4]:.4f}) | URL: <{row[1]}>")
    print("="*70 + "\n")

    # 4. Guard check if no schemes pass
    if not retrieved_schemes:
        return "Namaste. Based on your current profile details, I could not find an exact match among the active government schemes. Could you please share more information about your situation?", []

    # 5. Build dynamic context blocks and extract verified target URLs
    context_blocks = []
    context_urls = set()
    schemes_fetched = []
    
    for idx, row in enumerate(retrieved_schemes, 1):
        scheme_name, url, details, eligibility = row[0], row[1], row[2], row[3]
        schemes_fetched.append(scheme_name)
        if url:
            context_urls.add(url)
        context_blocks.append(
            f"--- CANDIDATE SCHEME {idx} ---\nName: {scheme_name}\nURL: {url}\nDetails: {details}\nEligibility: {eligibility}\n"
        )
    context_string = "\n".join(context_blocks)

    # 6. Build the dynamic User Profile context explicitly for the LLM
    verified_profile = f"""
    - Age: {user_age} years old
    - Gender: {user_gender}
    - Occupation: {profile_data.get('occupation', 'None')}
    - Family Income: ₹{safe_income:,} per annum
    - Residence: {profile_data.get('residence', 'Unknown')}
    - Caste Category: {profile_data.get('caste', 'General')}
    - Minority Status: {profile_data.get('minority', False)}
    - Marital Status: {profile_data.get('marital_status', 'Single')}
    - BPL Card Holder: {profile_data.get('below_poverty_line', False)}
    """

    # 7. Format the Short-Term Memory for Context Continuity
    history_context = ""
    if summary:
        history_context += f"\n--- LONG-TERM SUMMARY ---\n{summary}\n"
        
    if conversation_history:
        history_context += "\n--- RECENT CONVERSATION HISTORY ---\n"
        for msg in conversation_history:
            role = "User" if msg["role"] == "user" else "Yojana Mitra"
            history_context += f"{role}: {msg['content']}\n"

    # 8. Strictly defined System Prompt forcing profile evaluation constraints
    system_prompt = f"""
    You are Yojana Mitra, an intelligent welfare scheme advisor for Indian citizens.

    CRITICAL USER PROFILE FACTS:
    {verified_profile}
    {history_context}
    
    TASK: Carefully evaluate the CANDIDATE SCHEMES against the user's query and verified profile.
    Categorize your response strictly using the following 3-TIER RECOMMENDATION FORMAT:

    1. EXACT MATCHES (Tier 1):
       List only the candidate schemes where the user meets ALL eligibility rules based on their profile.
       Explain precisely why they qualify and how to apply.

    2. NEAR-MISSES / ACTIONABLE GAPS (Tier 2):
       List candidate schemes where the user almost qualifies (e.g., family income slightly exceeds limit, or they need a specific certificate like caste/income certificate).
       Clearly explain the exact gap and what step they can take (e.g., updating income certificate if deductions apply).

    3. NO MATCHES FOUND (Tier 3):
       If none of the candidates in the list match or come close to what the user needs, clearly state that no exact matches were found right now in the active index.
       DO NOT hallucinate or invent scheme rules. DO NOT invent facts about the user's demographic profile.

    STRICT RULES:
    - Only mention candidate schemes from the provided CANDIDATE SCHEMES CONTEXT below.
    - Do not force-fit or recommend schemes that contradict the user's gender, caste, or age.
    - Keep your tone supportive, concise, and structured with clear markdown bullet points.

    CANDIDATE SCHEMES CONTEXT:
    {context_string}
    """

    # 9. Run inference using the exact model
    response = ollama.chat(
        model='hf.co/qwen/qwen3-8b-gguf:q4_k_m', # Fixed to lowercase
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_query}
        ],
        options={
            'temperature': 0.1,
            'num_predict': 1800,   
            'num_thread': 4
        }
    )
    
    # 10. Extract thoughts, clean, and return the exact tuple
    raw_response = response['message']['content']
    final_response = extract_and_print_thoughts("RAG PIPELINE", raw_response)
    
    cleaned_response = sanitize_response(final_response, context_urls)
    return cleaned_response, schemes_fetched