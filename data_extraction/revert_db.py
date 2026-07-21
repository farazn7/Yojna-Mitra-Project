"""
Yojana Mitra — Revert DB Script
Drops the 4 columns added by the previous batch update script so we can start fresh.
Run: .\venv\Scripts\python.exe data_extraction\revert_db.py
"""
import psycopg2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from product_inference.db import DB_PARAMS

def revert():
    print("[Revert] Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()

    columns_to_drop = [
        "application_mode",
        "application_form_url",
        "application_process",
        "sources_references"
    ]

    for col in columns_to_drop:
        try:
            cur.execute(f"ALTER TABLE government_schemes DROP COLUMN IF EXISTS {col};")
            print(f"  [OK] Dropped column: {col}")
        except Exception as e:
            print(f"  [Error] Could not drop {col}: {e}")

    cur.close()
    conn.close()
    print("[Revert Complete] All 4 columns removed. DB is clean for a fresh run.")

if __name__ == "__main__":
    revert()
