"""
Yojana Mitra — Scrape & Classify Application Mode
===================================================
For every scheme URL in configuration/progress.json:
  1. Visit myscheme.gov.in page via Playwright → scrape Sources & References links
  2. For each source link → lightweight requests check (status, PDF or HTML, page snippet)
  3. Pass full enriched context to Llama 3.1 → intuitive classification
  4. Write results back into data_extraction/schemes.json

Final keys added per scheme:
  - sources_references : [{label, url}, ...]
  - application_mode   : "online" | "pdf_form" | "physical_only" | "unknown"
  - application_links  : {online_link, pdf_links, fallback_link}

Run: .\\venv\\Scripts\\python.exe data_extraction\\scrape_and_classify.py
Resumes safely from where it left off if interrupted.
"""

import json
import os
import sys
import re
import time
import random
import requests
import ollama
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROGRESS_FILE    = os.path.join(ROOT_DIR, "configuration", "progress.json")
SCHEMES_FILE     = os.path.join(ROOT_DIR, "data_extraction", "schemes.json")
CLASSIFY_PROG    = os.path.join(ROOT_DIR, "configuration", "classify_progress.json")

# ── Config ───────────────────────────────────────────────────────────────────
SCRAPE_MIN_DELAY = 3.0   # min seconds between myscheme.gov.in page visits
SCRAPE_MAX_DELAY = 5.0
HTTP_TIMEOUT     = 6     # seconds for external link HEAD/GET checks
SAVE_EVERY       = 10    # save schemes.json checkpoint every N schemes
# ─────────────────────────────────────────────────────────────────────────────


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def polite_wait():
    time.sleep(random.uniform(SCRAPE_MIN_DELAY, SCRAPE_MAX_DELAY))


# ── File I/O ─────────────────────────────────────────────────────────────────

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_classify_progress() -> set:
    if os.path.exists(CLASSIFY_PROG):
        return set(load_json(CLASSIFY_PROG).get("completed", []))
    return set()


def save_classify_progress(completed: set):
    save_json(CLASSIFY_PROG, {"completed": list(completed)})


# ── Step 1: Scrape Sources from myscheme.gov.in ──────────────────────────────

def scrape_sources(page, portal_url: str) -> list:
    """Visit the myscheme.gov.in page and extract all #sources a links."""
    sources = []
    try:
        page.goto(portal_url, wait_until="domcontentloaded", timeout=25000)
        page.wait_for_timeout(1500)

        links = page.locator("#sources a")

        # Fallback: navigate from the heading text
        if links.count() == 0:
            heading = page.locator("text=Sources And References")
            if heading.count() > 0:
                links = heading.first.locator("..").locator("..").locator("a")

        for i in range(links.count()):
            el = links.nth(i)
            href = el.get_attribute("href") or ""
            label = el.inner_text().strip()
            if href and label:
                sources.append({"label": label, "url": href})

    except PWTimeout:
        safe_print(f"    [Timeout] Could not load {portal_url}")
    except Exception as e:
        safe_print(f"    [Scrape Error] {e}")

    return sources


# ── Step 2: Lightweight HTTP check on external links ─────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def check_link(url: str) -> dict:
    """
    Quick requests check on an external link.
    Returns a dict with: type ("pdf"|"html"|"dead"), status_code, snippet.
    """
    if not url or not url.startswith("http"):
        return {"type": "dead", "status_code": None, "snippet": ""}

    # PDF by URL alone — no need to request
    if url.lower().endswith(".pdf"):
        return {"type": "pdf", "status_code": 200, "snippet": ""}

    try:
        resp = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT, stream=True)
        content_type = resp.headers.get("Content-Type", "").lower()

        # PDF by content type
        if "pdf" in content_type:
            return {"type": "pdf", "status_code": resp.status_code, "snippet": ""}

        # Dead / error
        if resp.status_code >= 400:
            return {"type": "dead", "status_code": resp.status_code, "snippet": ""}

        # HTML — grab a small snippet of visible text
        snippet = ""
        if "html" in content_type:
            try:
                # Read up to 8KB to extract a text snippet
                raw = b""
                for chunk in resp.iter_content(8192):
                    raw += chunk
                    break
                text = raw.decode("utf-8", errors="ignore")
                # Strip tags roughly
                text = re.sub(r"<[^>]+>", " ", text)
                text = re.sub(r"\s+", " ", text).strip()
                snippet = text[:300]
            except Exception:
                pass

        return {"type": "html", "status_code": resp.status_code, "snippet": snippet}

    except requests.exceptions.Timeout:
        return {"type": "dead", "status_code": None, "snippet": "Timed out"}
    except requests.exceptions.ConnectionError:
        return {"type": "dead", "status_code": None, "snippet": "Connection refused"}
    except Exception as e:
        return {"type": "dead", "status_code": None, "snippet": str(e)[:80]}


# ── Step 3: LLM Classification ────────────────────────────────────────────────

def build_llm_context(scheme_name: str, enriched_links: list) -> str:
    """Build the full context string for the LLM prompt."""
    lines = []
    for i, item in enumerate(enriched_links, 1):
        label = item["label"]
        url = item["url"]
        check = item["check"]
        link_type = check["type"]
        snippet = check.get("snippet", "")

        if link_type == "pdf":
            status_str = "PDF Document"
        elif link_type == "html":
            status_str = f"Loaded OK (HTTP {check['status_code']})"
        else:
            status_str = f"Dead/Unreachable ({check.get('snippet', '')})"

        lines.append(f"Link {i}:")
        lines.append(f"  Label  : \"{label}\"")
        lines.append(f"  URL    : {url}")
        lines.append(f"  Status : {status_str}")
        if snippet:
            lines.append(f"  Content: {snippet[:200]}")
        lines.append("")

    return "\n".join(lines)


LLM_SYSTEM = """You are an expert classifier for Indian government scheme application links.
You will be given a list of links from the "Sources And References" section of a scheme page.
Your job is to classify the application mode and extract the relevant links.

Key rules:
- Use BOTH the hypertext label AND the URL path to make decisions. Do NOT rely on just one.
- If a link is dead/timed-out but its label AND URL clearly imply an online application portal (e.g. label "Apply Online", URL has "apply", "register", "login" in path), classify it as the online_link anyway. Government sites often block automated checks but work for real users.
- For pdf_links: Only include PDFs that are actual APPLICATION FORMS citizens need to fill and submit. EXCLUDE PDFs that are guidelines, government orders, notifications, or scheme info documents.
- For fallback_link: Pick the most useful informational link (guidelines page, scheme details page) for citizen reference.
- physical_only means NO online portal and NO application form PDF exists at all.
- unknown means you genuinely cannot determine from the evidence given.

Respond ONLY with a valid JSON object. No explanation. No markdown. Just JSON:
{
  "mode": "online" | "pdf_form" | "physical_only" | "unknown",
  "online_link": "URL or null",
  "pdf_links": ["URL1", "URL2"],
  "fallback_link": "URL or null"
}"""


def classify_with_llm(scheme_name: str, enriched_links: list) -> dict:
    """Call Llama 3.1 to classify the scheme's application mode."""
    context = build_llm_context(scheme_name, enriched_links)

    prompt = f"""Scheme Name: "{scheme_name}"

Here are the Sources & References links found on this scheme's page:

{context}

Classify the application mode and extract relevant links. Return JSON only."""

    try:
        response = ollama.chat(
            model="llama3.1",
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.0}
        )
        raw = response["message"]["content"].strip()

        # Strip markdown code blocks if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # Extract JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

        result = json.loads(raw)

        # Normalize
        valid_modes = {"online", "pdf_form", "physical_only", "unknown"}
        if result.get("mode") not in valid_modes:
            result["mode"] = "unknown"
        result.setdefault("online_link", None)
        result.setdefault("pdf_links", [])
        result.setdefault("fallback_link", None)
        if not isinstance(result["pdf_links"], list):
            result["pdf_links"] = [result["pdf_links"]] if result["pdf_links"] else []

        return result

    except Exception as e:
        safe_print(f"    [LLM Error] {e}")
        return {
            "mode": "unknown",
            "online_link": None,
            "pdf_links": [],
            "fallback_link": None
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load all scheme URLs from progress.json
    progress_data = load_json(PROGRESS_FILE)
    all_urls = progress_data.get("completed_urls", [])
    if not all_urls:
        # Fallback: it might use "all_urls" key
        all_urls = progress_data.get("all_urls", [])

    # Load schemes.json as a list and build a lookup by portal_url
    schemes_list = load_json(SCHEMES_FILE)
    schemes_by_url = {s["portal_url"]: s for s in schemes_list}

    # Load classification progress
    classified = load_classify_progress()

    # Filter: only URLs that are in schemes.json and not yet classified
    todo = [u for u in all_urls if u in schemes_by_url and u not in classified]

    safe_print(f"\n{'='*55}")
    safe_print(f" Yojana Mitra — Scrape & Classify")
    safe_print(f"{'='*55}")
    safe_print(f"  Total URLs in progress.json : {len(all_urls)}")
    safe_print(f"  Matched in schemes.json     : {len(schemes_by_url)}")
    safe_print(f"  Already classified          : {len(classified)}")
    safe_print(f"  To process now              : {len(todo)}")
    safe_print(f"  Delay between pages         : {SCRAPE_MIN_DELAY}-{SCRAPE_MAX_DELAY}s")
    safe_print(f"{'='*55}\n")

    if not todo:
        safe_print("[Done] All schemes already classified!")
        return

    done_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, url in enumerate(todo, 1):
            scheme = schemes_by_url[url]
            name = scheme.get("scheme_name", url)
            abs_idx = len(classified) + i

            safe_print(f"[{abs_idx}/{len(all_urls)}] {name}")

            # ── Step 1: Scrape Sources & References ──
            sources = scrape_sources(page, url)
            safe_print(f"  Sources found: {len(sources)}")

            # If no sources at all → physical_only immediately, skip LLM
            if not sources:
                safe_print(f"  → No source links found. Marking as physical_only.")
                scheme["sources_references"] = []
                scheme["application_mode"] = "physical_only"
                scheme["application_links"] = {
                    "online_link": None,
                    "pdf_links": [],
                    "fallback_link": None
                }
            else:
                # ── Step 2: Check each external link ──
                enriched = []
                for link in sources:
                    check_result = check_link(link["url"])
                    enriched.append({**link, "check": check_result})
                    type_str = check_result["type"].upper()
                    safe_print(f"    [{type_str}] \"{link['label']}\" → {link['url'][:70]}")

                # ── Step 3: LLM Classification ──
                llm_result = classify_with_llm(name, enriched)
                mode = llm_result["mode"]
                safe_print(
                    f"  → LLM: {mode.upper()} | "
                    f"online={llm_result['online_link']} | "
                    f"pdfs={len(llm_result['pdf_links'])}"
                )

                # ── Step 4: Write back to scheme dict ──
                scheme["sources_references"] = sources  # raw, without check metadata
                scheme["application_mode"] = mode
                scheme["application_links"] = {
                    "online_link": llm_result["online_link"],
                    "pdf_links": llm_result["pdf_links"],
                    "fallback_link": llm_result["fallback_link"]
                }

            # Track progress
            classified.add(url)
            done_count += 1

            # Checkpoint save every SAVE_EVERY schemes
            if done_count % SAVE_EVERY == 0:
                save_json(SCHEMES_FILE, schemes_list)
                save_classify_progress(classified)
                safe_print(f"  >> Checkpoint saved ({abs_idx}/{len(all_urls)})")

            # Polite delay before next myscheme.gov.in page
            polite_wait()

        browser.close()

    # Final save
    save_json(SCHEMES_FILE, schemes_list)
    save_classify_progress(classified)

    safe_print(f"\n{'='*55}")
    safe_print(f"[DONE] Processed {done_count} schemes.")
    safe_print(f"  schemes.json updated: {SCHEMES_FILE}")
    safe_print(f"  Progress saved: {CLASSIFY_PROG}")


if __name__ == "__main__":
    main()
