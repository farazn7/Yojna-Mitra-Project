from playwright.sync_api import sync_playwright
import time

def scrape_schemes():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page()

        print("Loading the Tamil Nadu state schemes page...")
        page.goto("https://www.myscheme.gov.in/search/state/Tamil%20Nadu")

        # FIX 1: Wait for the main container to load, not the whole network
        # We wait for the basic DOM to load, then wait a few seconds for React to paint
        page.wait_for_load_state('domcontentloaded')
        time.sleep(5) # Hard pause to let the scheme cards render

        print("Page loaded. Taking a screenshot and saving HTML...")
        
        # FIX 2: Save exactly what the bot sees
        page.screenshot(path="debug_screenshot.png", full_page=True)
        with open("debug_page.html", "w", encoding="utf-8") as f:
            f.write(page.content())

        # FIX 3: Open the Visual Debugger
        print("Opening Playwright Inspector. The script is paused.")
        print("Click 'Resume' in the Inspector window to finish the script.")
        page.pause() 

        # Grab the links (you can test this selector inside the Inspector!)
        scheme_links = page.locator("a").element_handles()
        print(f"Found {len(scheme_links)} links on the page.")

        browser.close()

if __name__ == "__main__":
    scrape_schemes()