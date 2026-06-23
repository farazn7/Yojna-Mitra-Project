import json
import time
import random
import os
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL      = "https://www.myscheme.gov.in"
SEARCH_URL    = f"{BASE_URL}/search/state/Tamil%20Nadu"
OUTPUT_FILE   = "schemes.json"
PROGRESS_FILE = "progress.json"   # tracks completed URLs for resume
BATCH_SIZE    = 50                # restart browser every N pages (memory)
MIN_DELAY     = 2.5               # seconds between requests
MAX_DELAY     = 5.0
MAX_RETRIES   = 3
# ────────────────────────────────────────────────────────────────────────────


def polite_wait(multiplier: float = 1.0):
    time.sleep(random.uniform(MIN_DELAY * multiplier, MAX_DELAY * multiplier))


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed_urls": [], "all_urls": []}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def load_existing_results() -> list:
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_results(data: list):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def extract_scheme_details(page, url: str) -> dict | None:
    scheme_data = {
        "scheme_name": "",
        "portal_url": url,
        "eligibility_rules": "",
        "documents_needed": ""
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_selector("#eligibility", state="attached", timeout=12000)

            raw_title = page.title()
            scheme_data["scheme_name"] = raw_title.replace(" - myScheme", "").strip()

            elig = page.locator("#eligibility .markdown-options")
            if elig.count() > 0:
                scheme_data["eligibility_rules"] = elig.first.inner_text().strip()

            docs = page.locator("#documents-required .markdown-options")
            if docs.count() > 0:
                scheme_data["documents_needed"] = docs.first.inner_text().strip()

            return scheme_data

        except PWTimeout:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"        [Timeout] attempt {attempt}/{MAX_RETRIES}. Waiting {wait:.1f}s…")
            time.sleep(wait)
        except Exception as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"        [Error] {e} — attempt {attempt}/{MAX_RETRIES}. Waiting {wait:.1f}s…")
            time.sleep(wait)

    print(f"        [SKIP] Giving up on {url}")
    return None


def crawl_all_urls(page) -> list[str]:
    print("\n=== PHASE 1: CRAWLING ALL PAGES ===")
    page.goto(SEARCH_URL, wait_until="domcontentloaded", timeout=30000)

    raw_urls = []
    page_number = 1

    while True:
        print(f"  Scanning page {page_number}…")
        try:
            page.wait_for_selector('div[role="article"] h2 a', state="attached", timeout=15000)
        except PWTimeout:
            print("  [Warning] Timed out waiting for cards — retrying once.")
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector('div[role="article"] h2 a', state="attached", timeout=20000)

        elements = page.locator('div[role="article"] h2 a').element_handles()
        for el in elements:
            href = el.get_attribute("href")
            if href:
                raw_urls.append(f"{BASE_URL}{href}")

        # Pagination
        try:
            next_num = page_number + 1
            btn = page.locator("ul.list-none li").get_by_text(str(next_num), exact=True)

            if btn.count() > 0 and btn.is_visible():
                btn.scroll_into_view_if_needed()
                btn.click()
                page.wait_for_selector(
                    f"ul.list-none li.bg-green-700:has-text('{next_num}')",
                    state="attached", timeout=12000
                )
                # Wait for new cards to replace old ones
                page.wait_for_load_state("networkidle", timeout=8000)
                polite_wait(0.5)
                page_number += 1
            else:
                print(f"  End of pagination at page {page_number}.")
                break
        except Exception as e:
            print(f"  Pagination stopped: {e}")
            break

    unique = list(dict.fromkeys(raw_urls))
    print(f"\nTotal unique schemes found: {len(unique)}")
    return unique


def scrape_schemes():
    progress = load_progress()
    all_results = load_existing_results()
    completed_set = set(progress.get("completed_urls", []))

    with sync_playwright() as p:

        def new_browser():
            return p.chromium.launch(headless=True)  # headless saves RAM

        browser = new_browser()
        page = browser.new_page()

        # ── Phase 1: collect URLs (skip if already have them) ───────────────
        if progress.get("all_urls"):
            all_urls = progress["all_urls"]
            print(f"Resuming — {len(all_urls)} URLs loaded from progress file.")
        else:
            all_urls = crawl_all_urls(page)
            progress["all_urls"] = all_urls
            save_progress(progress)

        # ── Phase 2: extract details ─────────────────────────────────────────
        todo = [u for u in all_urls if u not in completed_set]
        print(f"\n=== PHASE 2: EXTRACTING SCHEME DETAILS ===")
        print(f"  To do: {len(todo)}  |  Already done: {len(completed_set)}")

        for i, url in enumerate(todo, 1):
            abs_index = len(completed_set) + i
            print(f"\n[{abs_index}/{len(all_urls)}] {url}")

            # Restart browser every BATCH_SIZE to avoid memory bloat
            if i > 1 and (i - 1) % BATCH_SIZE == 0:
                print("  ↻ Restarting browser to reclaim memory…")
                browser.close()
                browser = new_browser()
                page = browser.new_page()
                polite_wait(1.5)

            data = extract_scheme_details(page, url)

            if data:
                all_results.append(data)
                completed_set.add(url)
                progress["completed_urls"] = list(completed_set)

            # Partial saves every 10 items
            if i % 10 == 0:
                save_results(all_results)
                save_progress(progress)
                print(f"  💾 Checkpoint saved ({len(all_results)} schemes so far).")

            polite_wait()  # politeness delay between every request

        # Final save
        save_results(all_results)
        save_progress(progress)
        browser.close()

    print(f"\n✅ Done! {len(all_results)} schemes saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    scrape_schemes()