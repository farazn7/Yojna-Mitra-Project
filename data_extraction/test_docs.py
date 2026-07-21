import psycopg2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from product_inference.db import DB_PARAMS
from core_inference.doc_requirements import parse_required_documents

def test_single_scheme():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    # Get 1 random scheme
    cur.execute("""
        SELECT scheme_name, documents_needed 
        FROM government_schemes 
        WHERE documents_needed IS NOT NULL AND LENGTH(documents_needed) > 10
        ORDER BY RANDOM() 
        LIMIT 1;
    """)
    row = cur.fetchone()
    
    if not row:
        print("No schemes found with documents.")
        return
        
    name, docs = row
    print(f"\n{'='*60}")
    print(f"SCHEME: {name}")
    print(f"RAW TEXT:\n{docs}")
    print("-" * 60)
    
    print("\nSending to LLM... Please wait (should be ~2-5 seconds).")
    parsed = parse_required_documents(docs, scheme_name=name)
    
    print("\nPARSED RESULT:")
    print(f"Scannable: {parsed.get('scannable')}")
    print(f"Manual:    {parsed.get('manual')}")
    print(f"{'='*60}\n")
        
    cur.close()
    conn.close()

if __name__ == "__main__":
    test_single_scheme()
