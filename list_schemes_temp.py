import psycopg2
from product_inference.db import DB_PARAMS

def list_schemes():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT scheme_name, benefits, eligibility_rules FROM government_schemes LIMIT 5;")
        rows = cur.fetchall()
        for row in rows:
            print(f"Scheme: {row[0]}")
            print(f"Benefits Snippet: {row[1][:150] if row[1] else ''}...")
            print(f"Eligibility Snippet: {row[2][:150] if row[2] else ''}...\n")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    list_schemes()
