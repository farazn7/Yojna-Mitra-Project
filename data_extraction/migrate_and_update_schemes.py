# ==============================================================================
# Yojana Mitra — Phase 8: Step 5.1 & 5.2 Scheme Metadata Migration & Updater
# File Path: data_extraction/migrate_and_update_schemes.py
# ==============================================================================
"""
Adds `application_process`, `application_form_url`, and `application_mode` columns
to the `government_schemes` PostgreSQL table without data loss.

Also provides targeted scraping (`update_scheme_application_metadata`) to extract
Sources & References and classify the application mode (`online`, `pdf_form`, `physical_only`)
on demand when a citizen requests to apply.
"""

import os
import re
import time
import json
from typing import Any
import psycopg2
from psycopg2.extras import RealDictCursor
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# DB connection parameters
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "mysecretpassword",
    "host": "localhost",
    "port": "5432"
}


def run_migration():
    """Executes ALTER TABLE to safely add application mode columns."""
    print("[Migration] Connecting to PostgreSQL to update `government_schemes` schema...")
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("""
            ALTER TABLE government_schemes
            ADD COLUMN IF NOT EXISTS application_process TEXT,
            ADD COLUMN IF NOT EXISTS application_form_url TEXT,
            ADD COLUMN IF NOT EXISTS application_mode VARCHAR(30) DEFAULT 'unknown';
        """)

        cur.close()
        conn.close()
        print("[OK] [Migration Complete] Added application_process, application_form_url, application_mode columns.")
    except Exception as e:
        print(f"[WARN] [Migration Warning] Could not alter schema (`government_schemes` table may not exist yet or locked): {e}")


def classify_application_mode(sources_list: list[dict[str, str]]) -> tuple[str, str]:
    """
    Analyzes extracted Sources & References links to determine application mode.
    Returns `(application_mode, application_form_url)`.
    
    Modes:
      - `online`: Has an Application Form or Portal link pointing to an online web form (.gov.in/portal, not .pdf)
      - `pdf_form`: Has an Application Form link pointing to a downloadable .pdf file
      - `physical_only`: Only has Guidelines / Revised Rates or no links at all
      - `unknown`: Fallback if unclassified
    """
    if not sources_list:
        return ("physical_only", "")

    online_url = ""
    pdf_url = ""
    general_portal_url = ""

    for item in sources_list:
        label = item.get("label", "").lower()
        url = item.get("url", "").strip()

        if not url.startswith("http"):
            continue

        # Check if URL or label indicates a PDF document
        if url.lower().endswith(".pdf") or "pdf" in label:
            if "application" in label or "form" in label or not pdf_url:
                pdf_url = url
            continue

        # Check for explicit online application / portal keywords in label or URL
        if any(k in label for k in ["apply", "application", "portal", "registration", "online", "login", "register", "website", "scheme link"]):
            online_url = url
        elif not general_portal_url and not any(k in label for k in ["guideline", "notification", "go_", "announcement"]):
            general_portal_url = url

    # Prioritize explicit online form URL, then general scheme portal URL, then PDF form, then physical
    if online_url:
        return ("online", online_url)
    elif general_portal_url:
        return ("online", general_portal_url)
    elif pdf_url:
        return ("pdf_form", pdf_url)
    elif sources_list:
        return ("physical_only", sources_list[0].get("url", ""))
    
    return ("unknown", "")


def extract_application_details_from_url(portal_url: str) -> dict[str, Any]:
    """
    Scrapes myScheme URL using Playwright to extract:
      1. Application Process tab text
      2. Sources And References links
      3. Derived application_mode & application_form_url
    """
    result = {
        "application_process": "",
        "application_form_url": "",
        "application_mode": "unknown",
        "sources_references": []
    }

    if not portal_url or not portal_url.startswith("http"):
        return result

    print(f"[On-Demand Scrape] Inspecting myScheme URL: {portal_url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(portal_url, wait_until="domcontentloaded", timeout=25000)

            # 1. Extract Application Process tab text if present
            try:
                proc_tab = page.locator("#application-process .markdown-options")
                if proc_tab.count() > 0:
                    result["application_process"] = proc_tab.first.inner_text().strip()
            except Exception:
                pass

            # 2. Extract Sources & References links (#sources a per DOM container image)
            try:
                links = page.locator("#sources a")
                if links.count() == 0:
                    sources_heading = page.locator("text=Sources And References")
                    if sources_heading.count() > 0:
                        links = sources_heading.first.locator("..").locator("..").locator("a")
                        
                for i in range(links.count()):
                    link_el = links.nth(i)
                    href = link_el.get_attribute("href")
                    text = link_el.inner_text().strip()
                    if href and text:
                        result["sources_references"].append({"label": text, "url": href})
            except Exception as e:
                print(f"  [Scrape Note] Could not parse Sources & References section: {e}")

            browser.close()

            # Derive application_mode
            mode, form_url = classify_application_mode(result["sources_references"])
            result["application_mode"] = mode
            result["application_form_url"] = form_url
            print(f"  [OK] [Scrape Success] Mode: {mode.upper()} | Form URL: {form_url[:60]}")

    except Exception as e:
        print(f"[WARN] [On-Demand Scrape Error] Failed to scrape {portal_url}: {e}")

    return result


def update_scheme_application_metadata(scheme_name: str, portal_url: str) -> dict[str, Any]:
    """
    Scrapes application metadata for a specific scheme and updates its PostgreSQL record.
    Returns the metadata dictionary (`application_mode`, `application_form_url`, etc.).
    """
    details = extract_application_details_from_url(portal_url)

    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("""
            UPDATE government_schemes
            SET application_process = %s,
                application_form_url = %s,
                application_mode = %s
            WHERE scheme_name = %s OR portal_url = %s;
        """, (
            details["application_process"],
            details["application_form_url"],
            details["application_mode"],
            scheme_name,
            portal_url
        ))

        cur.close()
        conn.close()
    except Exception as e:
        print(f"[WARN] [DB Update Error] Could not save application metadata for {scheme_name}: {e}")

    return details


if __name__ == "__main__":
    print("=== Yojana Mitra: Running Scheme Application Mode Migration ===")
    run_migration()
