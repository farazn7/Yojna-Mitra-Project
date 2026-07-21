"""
Yojana Mitra — Push Classification to DB
=========================================
Reads application_mode and application_links from schemes.json
and pushes them into the government_schemes PostgreSQL table.

Run this AFTER scrape_and_classify.py has finished.
Run: .\\venv\\Scripts\\python.exe data_extraction\\push_to_db.py
"""

import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from product_inference.db import DB_PARAMS
import psycopg2

SCHEMES_FILE = os.path.join(os.path.dirname(__file__), "schemes.json")


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def ensure_columns(cur):
    """Add the two classification columns if they don't exist yet."""
    cur.execute("ALTER TABLE government_schemes ADD COLUMN IF NOT EXISTS application_mode VARCHAR(30) DEFAULT 'unknown';")
    cur.execute("ALTER TABLE government_schemes ADD COLUMN IF NOT EXISTS application_links JSONB;")
    safe_print("[OK] Columns ready: application_mode, application_links")


def push():
    with open(SCHEMES_FILE, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()

    ensure_columns(cur)

    updated = 0
    skipped = 0

    safe_print(f"\n[Push to DB] Processing {len(schemes)} schemes from schemes.json...\n")

    for scheme in schemes:
        name = scheme.get("scheme_name", "")
        portal_url = scheme.get("portal_url", "")
        mode = scheme.get("application_mode")
        links = scheme.get("application_links")

        # Skip schemes that haven't been classified yet
        if not mode or not links:
            skipped += 1
            continue

        try:
            cur.execute("""
                UPDATE government_schemes
                SET application_mode = %s,
                    application_links = %s
                WHERE portal_url = %s OR LOWER(scheme_name) = LOWER(%s);
            """, (
                mode,
                json.dumps(links, ensure_ascii=False),
                portal_url,
                name
            ))
            updated += 1
        except Exception as e:
            safe_print(f"  [Error] {name}: {e}")

    cur.close()
    conn.close()

    safe_print(f"[Done] Updated: {updated} | Skipped (not yet classified): {skipped}")


if __name__ == "__main__":
    push()
