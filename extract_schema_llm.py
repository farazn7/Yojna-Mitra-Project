import json
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama
from pydantic import BaseModel
from typing import Optional

# 1. Define the exact structure we want the LLM to extract
class SchemeMetadata(BaseModel):
    min_age: Optional[int]
    max_age: Optional[int]
    max_income: Optional[int]
    target_professions: list[str] # Let the LLM populate this dynamically!
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

    # We now have dedicated columns for our structured LLM outputs
    # portal_url is important and must be kept explicitly to avoid losing scheme URLs
    cur.execute("""
        CREATE TABLE government_schemes (
            id SERIAL PRIMARY KEY,
            scheme_name TEXT,
            portal_url TEXT,
            details TEXT,
            eligibility_rules TEXT,
            documents_needed TEXT,
            min_age INT,
            max_age INT,
            max_income INT,
            is_differently_abled BOOLEAN,
            is_women_only BOOLEAN,
            embedding VECTOR(768)
        );
    """)
    return conn, cur

def extract_metadata(text: str) -> SchemeMetadata:
    """Force Llama 3.1 to extract structured JSON based on our Pydantic model."""
    prompt = f"""
    Analyze the following government scheme text. Extract the eligibility criteria. 
    If a maximum income is mentioned, return it as an integer. If an age limit is mentioned, return it.
    Determine if this scheme is specifically targeted at differently abled individuals, or women only.
    If a value is not mentioned, return null for integers, or false for booleans.
    
    Text:
    {text}
    """
    
    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'user', 'content': prompt}],
        format=SchemeMetadata.model_json_schema(),
        options={'temperature': 0}
    )
    
    # Parse the strictly formatted JSON back into our Pydantic object
    return SchemeMetadata.model_validate_json(response['message']['content'])

def get_embedding(text):
    """Generate the semantic vector using nomic-embed-text."""
    if not text.strip():
        text = "No additional details provided."
    response = ollama.embeddings(model='nomic-embed-text', prompt=text)
    return response['embedding']

def main():
    print("Setting up database with new hybrid schema...")
    conn, cur = setup_database()

    with open('schemes.json', 'r', encoding='utf-8') as f:
        schemes = json.load(f)

    print(f"Starting Smart Ingestion for {len(schemes)} schemes...")
    
    # For testing, you might want to slice this to `schemes[:5]` so you don't wait an hour!
    for i, scheme in enumerate(schemes, 1):
        scheme_name = scheme.get('scheme_name', '')
        details = scheme.get('details', '')
        eligibility = scheme.get('eligibility_rules', '')
        
        text_to_embed = f"Scheme Details: {details}\nEligibility Rules: {eligibility}"
        
        # Step A: LLM Extraction
        metadata = extract_metadata(text_to_embed)
        
        # Step B: Vector Generation
        vector = get_embedding(text_to_embed)
        
        # Step C: Hybrid Database Insert
        cur.execute("""
            INSERT INTO government_schemes
            (scheme_name, portal_url, details, eligibility_rules, documents_needed,
             min_age, max_age, max_income, is_differently_abled, is_women_only, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            scheme_name,
            scheme.get('portal_url', ''),
            details,
            eligibility,
            scheme.get('documents_needed', ''),
            metadata.min_age,
            metadata.max_age,
            metadata.max_income,
            metadata.is_differently_abled,
            metadata.is_women_only,
            vector
        ))
        
        print(f"[{i}/{len(schemes)}] Ingested: {scheme_name}")
        print(f"    -> LLM Found: Age:{metadata.min_age}-{metadata.max_age}, Income:<{metadata.max_income}")

    print("✅ All schemes ingested successfully with metadata and embeddings!")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()