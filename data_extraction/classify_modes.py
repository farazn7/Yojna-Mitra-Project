"""
Yojana Mitra — Classify Application Modes
==========================================
Reads the ALREADY POPULATED sources_references from the DB,
runs the LLM evaluator on each, and saves application_mode + application_links.

No scraping. No external requests. Just DB read → LLM → DB write.

Run: .\venv\Scripts\python.exe data_extraction\classify_modes.py
"""
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from product_inference.db import DB_PARAMS
import psycopg2


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))


def ensure_columns():
    """Add application_mode and application_links columns if missing."""
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("ALTER TABLE government_schemes ADD COLUMN IF NOT EXISTS application_mode VARCHAR(30) DEFAULT 'unknown';")
    cur.execute("ALTER TABLE government_schemes ADD COLUMN IF NOT EXISTS application_links JSONB;")
    cur.close()
    conn.close()
    safe_print("[OK] Columns ready: application_mode, application_links")


def classify_all():
    # Import the LLM evaluator
    from core_inference.doc_requirements import evaluate_application_mode

    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()

    # Fetch all schemes with their sources_references
    cur.execute("SELECT id, scheme_name, portal_url, sources_references FROM government_schemes ORDER BY id;")
    rows = cur.fetchall()

    safe_print(f"\n=== Classifying {len(rows)} schemes ===\n")

    for i, (sid, name, portal_url, sources) in enumerate(rows, 1):
        safe_print(f"[{i}/{len(rows)}] {name}")

        # Parse sources — could be None, a JSON string, or already a list
        if sources is None:
            sources_list = []
        elif isinstance(sources, str):
            try:
                sources_list = json.loads(sources)
            except json.JSONDecodeError:
                sources_list = []
        elif isinstance(sources, list):
            sources_list = sources
        else:
            sources_list = []

        # Run LLM evaluation
        eval_result = evaluate_application_mode(sources_list, scheme_name=name)

        mode = eval_result.get("mode", "unknown")
        safe_print(f"  → {mode.upper()} | pdfs={len(eval_result.get('pdf_links', []))} | online={eval_result.get('online_link')}")

        # Save to DB
        cur.execute("""
            UPDATE government_schemes
            SET application_mode = %s,
                application_links = %s
            WHERE id = %s;
        """, (
            mode,
            json.dumps(eval_result, ensure_ascii=False),
            sid
        ))

    cur.close()
    conn.close()
    safe_print(f"\n[DONE] Classified all {len(rows)} schemes.")


if __name__ == "__main__":
    ensure_columns()
    classify_all()
