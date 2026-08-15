import json
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama
from pydantic import BaseModel
from typing import Optional

# 1. Exact structure for LLM metadata extraction
class SchemeMetadata(BaseModel):
    min_age: Optional[int]
    max_age: Optional[int]
    max_income: Optional[int]
    target_professions: list[str]  # Populated dynamically by Llama 3.1
    is_differently_abled: bool
    is_women_only: bool

def setup_database():
    conn = psycopg2.connect(
        dbname="postgres", 
        user="postgres",
        password="mysecretpassword",
        host="localhost",
        port="5432"
    )
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)

    cur.execute("DROP TABLE IF EXISTS government_schemes;")

    # ADDED: target_professions TEXT[] to store the array of eligible occupations
    cur.execute("""
        CREATE TABLE government_schemes (
            id SERIAL PRIMARY KEY,
            -- UNIQUE because schemes.json has shipped the same scheme twice before, and a
            -- blind INSERT let both rows through. Two identical rows both clear the SQL
            -- filter in hybrid_rag.py and then occupy two of the citizen's top_k slots with
            -- the same scheme, which costs a real recommendation. Worse, the duplicate rows
            -- were not actually identical: metadata comes from an LLM extraction pass, so
            -- the same source text produced max_age=40 on one copy and NULL on the other,
            -- meaning which row won the ranking decided whether a citizen was eligible.
            scheme_name TEXT UNIQUE,
            portal_url TEXT,
            details TEXT,
            eligibility_rules TEXT,
            documents_needed TEXT,
            min_age INT,
            max_age INT,
            max_income INT,
            target_professions TEXT[], 
            is_differently_abled BOOLEAN,
            is_women_only BOOLEAN,
            embedding VECTOR(768)
        );
    """)
    return conn, cur

def extract_metadata(text: str) -> SchemeMetadata:
    """Force Llama 3.1 to extract structured JSON based on our Pydantic model."""
    # UPDATED: Added precise prompt directions for occupation array synthesis
    prompt = f"""
    Analyze the following government scheme text and extract the eligibility criteria.
    
    1. If a maximum income bound is mentioned, return it as an integer. 
    2. If age limits are mentioned, return them as integers.
    3. Identify any specific target occupations or professions mentioned (e.g., "Farmer", "Student", "Weaver", "Artisan", "Unemployed"). 
       Return them as a clean list of strings in 'target_professions'. If it applies to any occupation generally, return an empty list [].
    4. Determine if this scheme is specifically targeted at differently abled individuals, or women only.
    
    If numerical attributes are not mentioned, return null. For booleans, default to false.
    
    Text:
    {text}
    """
    
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        format=SchemeMetadata.model_json_schema(),
        options={'temperature': 0}
    )
    
    return SchemeMetadata.model_validate_json(response['message']['content'])

def get_embedding(text):
    """Generate the semantic vector using nomic-embed-text."""
    if not text.strip():
        text = "No additional details provided."
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return response['embedding']

def main():
    print("Setting up database with new hybrid schema (with occupation filtering)...")
    conn, cur = setup_database()

    with open('schemes.json', 'r', encoding='utf-8') as f:
        schemes = json.load(f)

    print(f"Starting Smart Ingestion for {len(schemes)} schemes...")
    
    # For initial safety and speed, slice this to `schemes[:10]` to verify parsing outputs!
    for i, scheme in enumerate(schemes, 1):
        scheme_name = scheme.get('scheme_name', '')
        details = scheme.get('details', '')
        eligibility = scheme.get('eligibility_rules', '')
        
        text_to_embed = f"Scheme Details: {details}\nEligibility Rules: {eligibility}"
        
        # Step A: LLM Extraction
        metadata = extract_metadata(text_to_embed)
        
        # Step B: Vector Generation
        vector = get_embedding(text_to_embed)
        
        # Step C: Hybrid Database Insert (Including the target_professions text array)
        cur.execute("""
            INSERT INTO government_schemes
            (scheme_name, portal_url, details, eligibility_rules, documents_needed,
             min_age, max_age, max_income, target_professions, is_differently_abled, is_women_only, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (scheme_name) DO NOTHING
        """, (
            scheme_name,
            scheme.get('portal_url', ''),
            details,
            eligibility,
            scheme.get('documents_needed', ''),
            metadata.min_age,
            metadata.max_age,
            metadata.max_income,
            metadata.target_professions,  # Psycopg2 automatically translates Python lists to SQL arrays
            metadata.is_differently_abled,
            metadata.is_women_only,
            vector
        ))
        
        # rowcount is 0 when ON CONFLICT skipped the row. Say so rather than printing
        # "Ingested" for a scheme that was not written -- a silent skip here is how the
        # duplicate pairs went unnoticed in the first place.
        if cur.rowcount:
            print(f"[{i}/{len(schemes)}] Ingested: {scheme_name}")
            print(f"    -> LLM Found: Professions: {metadata.target_professions} | Income: <{metadata.max_income}")
        else:
            print(f"[{i}/{len(schemes)}] SKIPPED (duplicate scheme_name already ingested): {scheme_name}")

    print("✅ All schemes ingested successfully with persistent metadata arrays and embeddings!")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()