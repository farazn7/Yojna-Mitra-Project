# ==============================================================================
# Yojana Mitra — Web Session Authentication
# File Path: product_inference/web/auth.py
# ==============================================================================
"""
Stateless signed tokens binding a browser session to an already-verified platform identity.

This is the one genuinely new problem the web surface introduces. Telegram and Discord
hand over a trusted user id; a browser does not. And `user_id` is the partition key for
`user_profiles`, `pii_vault`, the LangGraph checkpointer, and `browser_profiles/<user_id>/`.
If a browser were allowed to assert its own `user_id`, anyone could type `telegram_12345`
and read another citizen's extracted Aadhaar fields.

Rather than build a login next to a PII vault, identity is *delegated*: the citizen sends
`/web` to the Telegram bot, which mints a short-lived signed handoff token over the
`telegram_<id>` they are already authenticated as. The web server verifies it and issues a
longer-lived session cookie over the same id. Their profile, vault and checkpoint history
carry straight over with no migration, because it is literally the same `user_id`.

No database table. A signed token needs no storage, and a sessions table next to
`pii_vault` would mean a migration plus a second thing with a TTL to get wrong.

Stdlib `hmac`/`hashlib` only — no `itsdangerous`, no JWT library.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

# Handoff links are single-use in practice and short-lived by design: long enough to tap
# a link from a phone, short enough that a leaked link in a chat log is worthless.
HANDOFF_TTL_SECONDS = 300          # 5 minutes
SESSION_TTL_SECONDS = 7 * 24 * 3600  # 7 days

COOKIE_NAME = "ym_session"

_RUNTIME_SECRET: Optional[bytes] = None


def _secret() -> bytes:
    """The HMAC key, from `WEB_SESSION_SECRET` in the environment.

    If unset, a random key is generated once per process and a warning is printed:
    sessions then do not survive a restart, which is an acceptable dev-mode annoyance.
    It must never silently fall back to a constant — a hardcoded default in a repo is
    the same as no signature at all.
    """
    global _RUNTIME_SECRET

    env = os.getenv("WEB_SESSION_SECRET", "").strip()
    if env:
        return env.encode("utf-8")

    if _RUNTIME_SECRET is None:
        _RUNTIME_SECRET = secrets.token_bytes(32)
        print(
            "[Auth WARNING] WEB_SESSION_SECRET is not set. Generated an ephemeral key; "
            "all web sessions will be invalidated when this process restarts. "
            "Set WEB_SESSION_SECRET in .env for a persistent deployment."
        )
    return _RUNTIME_SECRET


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + padding)


def _sign(payload_b64: str) -> str:
    digest = hmac.new(_secret(), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return _b64e(digest)


def mint_token(user_id: str, ttl_seconds: int = SESSION_TTL_SECONDS) -> str:
    """Create a signed token asserting `user_id`, valid for `ttl_seconds`."""
    if not user_id or not isinstance(user_id, str):
        raise ValueError("user_id must be a non-empty string")

    payload = {"uid": user_id, "exp": int(time.time()) + int(ttl_seconds)}
    payload_b64 = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{payload_b64}.{_sign(payload_b64)}"


def verify_token(token: Optional[str]) -> Optional[str]:
    """Return the `user_id` a token attests to, or None if it is not trustworthy.

    None is returned for every failure mode — malformed, tampered payload, wrong
    signature, expired — with no distinction. A caller cannot learn *why* a token was
    rejected, and there is nothing useful it could do differently if it could.
    """
    if not token or not isinstance(token, str):
        return None

    parts = token.split(".")
    if len(parts) != 2:
        return None

    payload_b64, signature = parts

    # Constant-time comparison: a byte-by-byte early exit would leak the expected
    # signature one character at a time.
    if not hmac.compare_digest(signature, _sign(payload_b64)):
        return None

    try:
        payload = json.loads(_b64d(payload_b64))
    except Exception:  # noqa: BLE001 — any decode failure is a rejection
        return None

    if not isinstance(payload, dict):
        return None

    expires_at = payload.get("exp")
    user_id = payload.get("uid")

    if not isinstance(expires_at, int) or not isinstance(user_id, str) or not user_id:
        return None

    if time.time() >= expires_at:
        return None

    return user_id


def mint_handoff(user_id: str) -> str:
    """A short-lived token for the Telegram `/web` deep link."""
    return mint_token(user_id, ttl_seconds=HANDOFF_TTL_SECONDS)


def handoff_url(user_id: str, base_url: Optional[str] = None) -> str:
    """The full link a bot sends to move a citizen from chat to browser."""
    base = (base_url or os.getenv("WEB_BASE_URL", "http://localhost:8000")).rstrip("/")
    return f"{base}/auth?t={mint_handoff(user_id)}"
