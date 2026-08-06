# ==============================================================================
# Yojana Mitra — Phase 8: Step 1 Browser Session Manager
# File Path: product_inference/browser_manager.py
# ==============================================================================
"""
Browser Session Manager for Phase 8 Universal Dynamic Web Automation.

Key Features & Architectural Fixes:
  1. Dedicated Persistent Context Isolation (`launch_isolated_profile`):
     Launches a persistent browser directory per user (`browser_profiles/{user_id}`).
     Bypasses Chrome 136 CDP default-directory security restrictions while preserving
     cookies and login sessions.
  2. Race-Condition Proof Initialization:
     Initializes `_SESSION_REGISTRY[user_id]` BEFORE attaching `page.on("dialog")` and
     `context.on("page")` event listeners. Prevents KeyError if interstitial popups
     fire during the initial `page.goto()`.
  3. HITL-Routed Dialog Interceptor (`handle_dialog`):
     Never auto-accepts native dialogs on keyword match (`submit`/`confirm`). Only
     `beforeunload` is auto-accepted. All `confirm`, `alert`, and `prompt` dialogs
     route to a human decision callback, protecting the Step 4 final screenshot gate.
  4. Active Page Tracking (`handle_new_page`):
     Listens at `BrowserContext` level for new tabs (`window.open()`), shifting the
     active page pointer when e-KYC or payment gateway tabs open.
  5. Strong SPA Page Fingerprinting (`compute_page_signature`):
     Computes SHA-256 of `URL + Title + First Heading Text` to accurately detect
     Step 1 → Step 2 transitions on Single-Page Application (SPA) wizards where the
     URL does not change.
"""

import os
import hashlib
import time
import concurrent.futures
from typing import Optional, Callable, Any
from playwright.sync_api import sync_playwright, BrowserContext, Page, Dialog, Error as PlaywrightError

# Root directory holding one persistent Chrome profile per citizen. Overridable via env so
# the path is not pinned to one developer's machine; the default is deliberately left as it
# was, so profiles a citizen has already logged a portal into keep working across this change.
BROWSER_PROFILES_ROOT = os.getenv(
    "YM_BROWSER_PROFILES_DIR",
    "C:/Users/nezam/.gemini/antigravity/browser_profiles",
)


def profile_dir_for(user_id: str, profile_base_dir: Optional[str] = None) -> str:
    """Absolute path of `user_id`'s persistent Chrome profile directory.

    Single source of truth for the `user_` prefix. `/reset` has to delete exactly the
    directory `launch_isolated_profile` creates, and re-deriving that name at the deletion
    site is precisely how the two drift apart and a reset silently leaves portal cookies
    on disk.
    """
    return os.path.abspath(
        os.path.join(profile_base_dir or BROWSER_PROFILES_ROOT, f"user_{user_id}")
    )


# Global executor for running synchronous Playwright calls in a clean thread outside any asyncio event loop
_PW_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="PW_Sync_Worker")

def run_pw(func, *args, **kwargs):
    """Executes a synchronous Playwright function on a dedicated worker thread where no asyncio event loop is running."""
    future = _PW_EXECUTOR.submit(func, *args, **kwargs)
    return future.result()

# Global registry keeping active persistent sessions alive across graph turns
_SESSION_REGISTRY: dict[str, dict[str, Any]] = {}

def authorize_one_dialog(user_id: str, ttl_seconds: int = 20) -> bool:
    """
    Authorizes the NEXT native dialog this user's page raises to be accepted, once, within
    `ttl_seconds`.

    The dialog interceptor dismisses `confirm`/`alert`/`prompt` by default, which is right: a portal
    must never be able to talk the agent into accepting something no human agreed to. But a portal
    that asks "are you sure you want to submit?" after the citizen has already replied CONFIRM is
    asking a question that has been answered — and dismissing it meant the submission the citizen
    explicitly authorized silently never happened.

    So authorization is narrow by construction: armed only by the code path that runs after a literal
    CONFIRM, good for one dialog, expiring on a timer, and consumed the moment it is used. It cannot
    be armed by anything the page does.
    """
    entry = _SESSION_REGISTRY.get(user_id)
    if not entry:
        return False
    entry["dialog_authorized_until"] = time.time() + ttl_seconds
    return True


def _consume_dialog_authorization(user_id: str) -> bool:
    """True if this user has an unexpired one-shot dialog authorization, which is spent by asking."""
    entry = _SESSION_REGISTRY.get(user_id)
    if not entry:
        return False
    until = entry.pop("dialog_authorized_until", 0)
    return bool(until) and time.time() < until


# Default callback for dialog notifications (can be overridden by bot.py for Discord relay)
def _default_dialog_callback(user_id: str, dialog_type: str, message: str) -> tuple[str, Optional[str]]:
    """
    Default console dialog handler when running standalone or testing.
    Returns (decision: "accept" | "dismiss", prompt_text: Optional[str]).
    In bot.py live runtime, this is replaced with a Discord DM question + ack wait.
    """
    print(f"\n[WARN: NATIVE DIALOG DETECTED — User: {user_id}]")
    print(f"  Type: {dialog_type} | Message: \"{message}\"")
    # For automated local testing or safe non-submit alerts, we log and dismiss by default unless overridden
    # Never auto-accept submit/confirm gates without explicit human acknowledgement
    return ("dismiss", None)

_dialog_handler_callback: Callable[[str, str, str], tuple[str, Optional[str]]] = _default_dialog_callback

def set_dialog_callback(callback: Callable[[str, str, str], tuple[str, Optional[str]]]):
    """Registers the live Discord/WhatsApp HITL callback from bot.py."""
    global _dialog_handler_callback
    _dialog_handler_callback = callback


def launch_isolated_profile(
    user_id: str,
    portal_url: Optional[str] = None,
    headless: bool = False,
    profile_base_dir: Optional[str] = None
) -> tuple[BrowserContext, Page]:
    """
    Launches or retrieves an isolated Playwright persistent browser profile for `user_id`.
    Ensures safe initialization, native dialog interception, and new-tab tracking.
    """
    # Check if session already active and valid
    if user_id in _SESSION_REGISTRY:
        ctx = _SESSION_REGISTRY[user_id]["context"]
        active_page = _SESSION_REGISTRY[user_id]["active_page"]
        try:
            if not active_page.is_closed():
                if portal_url and active_page.url == "about:blank":
                    active_page.goto(portal_url, wait_until="domcontentloaded")
                return ctx, active_page
        except Exception:
            pass # Context or page crashed; re-launch

    profile_dir = profile_dir_for(user_id, profile_base_dir)
    os.makedirs(profile_dir, exist_ok=True)

    playwright = sync_playwright().start()

    # Attempt launch with real Chrome first for best e-Governance portal compatibility, fallback to Chromium
    try:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=headless,
            channel="chrome",
            viewport={"width": 1280, "height": 900},
            accept_downloads=True
        )
    except PlaywrightError:
        print(f"[Browser Manager] Google Chrome channel not found/failed for {user_id}. Launching bundled Chromium...")
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=headless,
            viewport={"width": 1280, "height": 900},
            accept_downloads=True
        )

    page = context.pages[0] if context.pages else context.new_page()

    # *** FIX: Initialize session in registry BEFORE attaching listeners (`handle_new_page`, `handle_dialog`) ***
    _SESSION_REGISTRY[user_id] = {
        "playwright": playwright,
        "context": context,
        "active_page": page,
        "profile_dir": profile_dir
    }

    # *** FIX: HITL-routed dialog handler ***
    def handle_dialog(dialog: Dialog):
        log_msg = f"Dialog ({dialog.type}): {dialog.message}"
        print(f"[Dialog Interceptor] {log_msg}")

        # Safe to auto-handle beforeunload without human decision
        if dialog.type == "beforeunload":
            dialog.accept()
            return

        # A dialog the citizen has already authorized (see authorize_one_dialog) — spent on use.
        if _consume_dialog_authorization(user_id):
            print("  [Dialog] Accepting: the citizen authorized this submission with an explicit CONFIRM.")
            dialog.accept()
            return

        # Route all confirm, alert, and prompt dialogs through our registered HITL callback
        try:
            decision, prompt_text = _dialog_handler_callback(user_id, dialog.type, dialog.message)
            if decision == "accept":
                if dialog.type == "prompt" and prompt_text is not None:
                    dialog.accept(prompt_text)
                else:
                    dialog.accept()
            else:
                dialog.dismiss()
        except Exception as e:
            print(f"[Dialog Error] Callback failed ({e}), dismissing safety default.")
            dialog.dismiss()

    page.on("dialog", handle_dialog)

    # *** FIX: New tab listener at BrowserContext level ***
    def handle_new_page(new_page: Page):
        print(f"[New Tab Opened — User: {user_id}] Shifting active focus to new tab...")
        try:
            new_page.wait_for_load_state("domcontentloaded", timeout=10000)
        except Exception:
            pass
        if user_id in _SESSION_REGISTRY:
            _SESSION_REGISTRY[user_id]["active_page"] = new_page

    context.on("page", handle_new_page)

    # Navigate to portal URL if requested
    if portal_url and (page.url == "about:blank" or not page.url.startswith("http")):
        print(f"[Navigate] Opening Portal: {portal_url}")
        page.goto(portal_url, wait_until="domcontentloaded")

    return context, page


def get_active_page(user_id: str) -> Optional[Page]:
    """Retrieves the currently active page (including shifted popup tabs) for `user_id`."""
    if user_id in _SESSION_REGISTRY:
        page = _SESSION_REGISTRY[user_id]["active_page"]
        if not page.is_closed():
            return page
    return None


def compute_page_signature(page: Page) -> str:
    """
    Computes a stable SHA-256 fingerprint of `URL + Title + First Visible Heading`.
    Catches SPA wizard transitions (`Step 1: Personal` → `Step 2: Address`) where URL stays fixed.
    """
    try:
        url_clean = page.url.split("?")[0]
        title = page.title()
        try:
            heading = page.locator("h1, h2, h3, .step-title, .wizard-heading").first.inner_text(timeout=1000)
        except Exception:
            heading = ""
        raw_sig = f"{url_clean}||{title}||{heading}"
        return hashlib.sha256(raw_sig.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unknown_signature"


def close_session(user_id: str):
    """Safely closes and removes the persistent context for `user_id`."""
    global _PW_EXECUTOR
    if user_id in _SESSION_REGISTRY:
        entry = _SESSION_REGISTRY.pop(user_id)
        # These are the same sync-API objects launch_isolated_profile created via run_pw on a
        # worker thread — closing/stopping them directly from whatever thread called close_session()
        # violates Playwright's thread-affinity requirement just like every other Playwright call in
        # this file is already careful to avoid (see the module docstring / run_pw itself).
        try:
            run_pw(entry["context"].close)
        except Exception:
            pass
        try:
            run_pw(entry["playwright"].stop)
        except Exception:
            pass
        print(f"[Browser Manager] Session closed for user: {user_id}")

        # Once no sessions remain, recycle the shared worker pool so the next session's
        # sync_playwright() always starts on a brand-new OS thread. Playwright's sync API expects a
        # thread to host at most one Playwright instance for its whole life; ThreadPoolExecutor
        # reusing a worker thread across separate launch/close cycles (this pool is shared process-
        # wide, not per-session) left threads in a state that broke a later, unrelated session with
        # "using Playwright Sync API inside the asyncio loop" — reproduced live after 2-3 sequential
        # session cycles in one process. A long-running bot handling many users over time is exactly
        # this pattern, just spread over longer, so this isn't just a test-script artifact.
        if not _SESSION_REGISTRY:
            old_executor = _PW_EXECUTOR
            _PW_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="PW_Sync_Worker")
            old_executor.shutdown(wait=False)


def get_session_info(user_id: str) -> Optional[dict[str, Any]]:
    """Returns metadata about an active user session."""
    if user_id in _SESSION_REGISTRY:
        page = _SESSION_REGISTRY[user_id]["active_page"]
        return {
            "user_id": user_id,
            "profile_dir": _SESSION_REGISTRY[user_id]["profile_dir"],
            "current_url": page.url if not page.is_closed() else "CLOSED",
            "page_signature": compute_page_signature(page) if not page.is_closed() else "CLOSED"
        }
    return None


# ==============================================================================
# STANDALONE LOCAL VERIFICATION HARNESS (STEP 1 TESTING)
# ==============================================================================
if __name__ == "__main__":
    import sys
    print("=== Yojana Mitra: Step 1 Browser Session Manager Verification ===")
    test_user = "test_user_001"

    # Check if user passed --headless CLI flag; otherwise default to VISIBLE (headless=False) so you can watch it open!
    run_headless = "--headless" in sys.argv

    # Test 1: Launch persistent profile & verify no race conditions
    print(f"\n[Test 1] Launching isolated persistent profile (headless={run_headless})...")
    ctx, page = launch_isolated_profile(test_user, portal_url="https://example.com", headless=run_headless)
    info = get_session_info(test_user)
    print(f"  [OK] Profile initialized: {info['profile_dir']}")
    print(f"  [OK] Current URL: {info['current_url']}")
    print(f"  [OK] SPA Signature: {info['page_signature']}")

    # Test 2: Native Dialog Interception (HITL verify)
    print("\n[Test 2] Testing native dialog interceptor (`window.confirm` / `window.alert`)...")
    dialog_triggered = {"caught": False, "type": ""}
    def mock_dialog_cb(u_id, d_type, msg):
        dialog_triggered["caught"] = True
        dialog_triggered["type"] = d_type
        print(f"  [HITL Intercepted] -> Type: {d_type} | Message: '{msg}'")
        return ("dismiss", None)

    set_dialog_callback(mock_dialog_cb)
    page.evaluate("() => setTimeout(() => window.confirm('Are you sure you want to submit application?'), 100)")
    page.wait_for_timeout(500)
    assert dialog_triggered["caught"] == True, "Dialog was not caught!"
    print("  [OK] Dialog intercepted without auto-accepting or crashing!")

    # Test 3: New Tab Shift (`window.open`)
    print("\n[Test 3] Testing new tab detection & active page shifting...")
    initial_url = page.url
    page.evaluate("() => window.open('https://www.w3schools.com', '_blank')")
    page.wait_for_timeout(3000)
    new_active = get_active_page(test_user)
    print(f"  [OK] Active page pointer shifted to: {new_active.url}")
    assert new_active.url != initial_url, "Active page pointer did not shift!"

    # Visual pause so you can inspect the open browser window before it closes!
    if not run_headless:
        print("\n[Visual Inspection Pause] Browser window is visibly open on your desktop right now!")
        print("  Keeping window open for 10 seconds so you can see the new tab and verification state...")
        new_active.wait_for_timeout(10000)

    # Clean up
    print("\n[Test 4] Closing session cleanly...")
    close_session(test_user)
    print("=== Step 1 Verification Complete & Passed ===")
