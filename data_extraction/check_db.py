import psycopg2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from product_inference.db import DB_PARAMS

def check_db():
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    cur.execute("SELECT scheme_name, application_mode FROM government_schemes WHERE scheme_name ILIKE '%Book Bank Scheme%';")
    rows = cur.fetchall()
    print(rows)

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_db()
