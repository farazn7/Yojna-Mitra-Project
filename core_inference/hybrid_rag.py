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

def sanitize_response(response_text, context_urls):
    """Remove URLs that were not provided in the retrieval context."""
    found_urls = re.findall(r'https?://[^\s)\]]+', response_text)
    for url in found_urls:
        clean_url = url.rstrip('.,;:')
        if clean_url not in context_urls:
            response_text = response_text.replace(url, "[verified URL not available in provided context]")
    return response_text

def run_yojana_pipeline(profile_data, user_query, top_k=3):
    """
    THE LIVE RUNTIME GATEWAY:
    Processes the user's chat query and database profile metadata 
    through a strict SQL filter, pgvector matching, and structured LLM generation loop.
    """
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
        return "Namaste. Based on your current profile details, I could not find an exact match among the active government schemes. Could you please share more information about your situation?"

    # 5. Build dynamic context blocks and extract verified target URLs
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

    # 7. Strictly defined System Prompt forcing profile evaluation constraints
    system_prompt = f"""
    You are Yojana Mitra, a helpful WhatsApp AI assistant for Indian citizens.

    CRITICAL USER PROFILE FACTS:
    {verified_profile}
    
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

    # 8. Run inference matching the optimized test framework options
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