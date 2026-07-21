"""
Yojana Mitra — Populate Sources & References
============================================
Iterates through all scheme URLs in configuration/progress.json,
scrapes ONLY the Sources & References section from each myscheme.gov.in page,
and saves the raw {label, url} array into a new `sources_references` JSONB column.

NO classification. NO link pinging. NO external server requests.
Only touches myscheme.gov.in (the aggregator site we already scraped).

Features:
  - Polite delays (3-5s) between each page to avoid WAF bans
  - Progress tracking (resumes from where it left off if interrupted)
  - Browser restart every 40 pages to prevent memory bloat
  - Unicode-safe terminal output

Run: .\\venv\\Scripts\\python.exe data_extraction\\populate_sources.py
"""
import json
import time
import random
import os
import sys
import psycopg2
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from product_inference.db import DB_PARAMS

# ── Config ──────────────────────────────────────────────────────────────────
MIN_DELAY     = 3.0     # minimum seconds between requests
MAX_DELAY     = 5.0     # maximum seconds between requests
BATCH_SIZE    = 40      # restart browser every N pages (memory management)
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "sources_progress.json")
# ────────────────────────────────────────────────────────────────────────────


def safe_print(text):
    """Print with unicode safety for Windows terminals."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode('ascii'))


def polite_wait():
    """Random delay to be polite to government servers."""
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))


def load_progress() -> set:
    """Load set of already-processed portal_urls."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data.get("completed", []))
    return set()


def save_progress(completed: set):
    """Save progress to disk."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed": list(completed)}, f, ensure_ascii=False)


def ensure_column():
    """Add `sources_references` JSONB column if it doesn't exist."""
    safe_print("[Setup] Ensuring sources_references column exists...")
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("ALTER TABLE government_schemes ADD COLUMN IF NOT EXISTS sources_references JSONB;")
    cur.close()
    conn.close()
    safe_print("  [OK] Column ready.")


def get_all_scheme_urls() -> list[dict]:
    """
    Pull scheme_name + portal_url from the DB itself.
    This way we scrape exactly the schemes that are in our database.
    """
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("SELECT scheme_name, portal_url FROM government_schemes ORDER BY id;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"scheme_name": r[0], "portal_url": r[1]} for r in rows]


def scrape_sources_from_page(page, url: str) -> list[dict]:
    """
    Navigate to a myscheme.gov.in URL and extract ONLY the
    Sources & References links from the #sources section.
    
    Returns a list of {"label": "...", "url": "..."} dicts.
    """
    sources = []
    
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=25000)
        # Wait a moment for any lazy-loaded content
        page.wait_for_timeout(2000)
        
        # Try primary selector: #sources a (matches the DOM screenshot the user showed)
        links = page.locator("#sources a")
        
        # Fallback: find the heading and navigate up to the container
        if links.count() == 0:
            heading = page.locator("text=Sources And References")
            if heading.count() > 0:
                # Go up to the parent container and find all links
                links = heading.first.locator("..").locator("..").locator("a")
        
        for i in range(links.count()):
            link_el = links.nth(i)
            href = link_el.get_attribute("href")
            label = link_el.inner_text().strip()
            if href and label:
                sources.append({"label": label, "url": href})
                
    except PWTimeout:
        safe_print(f"    [Timeout] Page took too long: {url}")
    except Exception as e:
        safe_print(f"    [Error] {e}")
    
    return sources


def save_sources_to_db(portal_url: str, scheme_name: str, sources: list[dict]):
    """Save the raw sources_references array to the DB."""
    conn = psycopg2.connect(**DB_PARAMS)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "UPDATE government_schemes SET sources_references = %s WHERE portal_url = %s OR scheme_name = %s;",
        (json.dumps(sources, ensure_ascii=False), portal_url, scheme_name)
    )
    cur.close()
    conn.close()


def main():
    ensure_column()
    
    schemes = get_all_scheme_urls()
    completed = load_progress()
    
    todo = [s for s in schemes if s["portal_url"] not in completed]
    
    safe_print(f"\n=== Yojana Mitra: Populate Sources & References ===")
    safe_print(f"  Total schemes in DB: {len(schemes)}")
    safe_print(f"  Already completed:   {len(completed)}")
    safe_print(f"  Remaining:           {len(todo)}")
    safe_print(f"  Delay per page:      {MIN_DELAY}-{MAX_DELAY}s")
    safe_print(f"  Browser restart:     every {BATCH_SIZE} pages")
    safe_print(f"{'='*50}\n")
    
    if not todo:
        safe_print("[Done] All schemes already have sources_references populated!")
        return
    
    with sync_playwright() as p:
        
        def new_browser():
            return p.chromium.launch(headless=True)
        
        browser = new_browser()
        page = browser.new_page()
        
        for i, scheme in enumerate(todo, 1):
            name = scheme["scheme_name"]
            url = scheme["portal_url"]
            abs_index = len(completed) + i
            
            safe_print(f"[{abs_index}/{len(schemes)}] {name}")
            
            # Restart browser periodically to prevent memory bloat
            if i > 1 and (i - 1) % BATCH_SIZE == 0:
                safe_print("  >> Restarting browser (memory management)...")
                browser.close()
                polite_wait()
                browser = new_browser()
                page = browser.new_page()
            
            # Scrape sources
            sources = scrape_sources_from_page(page, url)
            safe_print(f"  Found {len(sources)} source links")
            
            if sources:
                for s in sources:
                    safe_print(f"    - [{s['label']}] -> {s['url'][:80]}")
            
            # Save to DB (even if empty — this marks it as "scraped, has no sources")
            save_sources_to_db(url, name, sources)
            
            # Track progress
            completed.add(url)
            
            # Save progress every 10 items
            if i % 10 == 0:
                save_progress(completed)
                safe_print(f"  >> Checkpoint saved ({abs_index}/{len(schemes)})")
            
            # Polite delay before next request
            polite_wait()
        
        browser.close()
    
    # Final save
    save_progress(completed)
    safe_print(f"\n{'='*50}")
    safe_print(f"[DONE] Populated sources_references for {len(todo)} schemes.")
    safe_print(f"  Progress file: {PROGRESS_FILE}")


if __name__ == "__main__":
    main()
