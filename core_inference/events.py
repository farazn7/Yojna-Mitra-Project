# ==============================================================================
# Yojana Mitra — Surface Event Vocabulary
# File Path: core_inference/events.py
# ==============================================================================
"""
The event vocabulary every surface renders.

A turn is not "call the graph, get a string back". It is a sequence of things worth
telling the citizen about: a status while a 4B vision model reads their Aadhaar, the
answer itself, a portal screenshot, the confirm gate. Telegram renders these by editing
one message in place, Discord by editing its own, the browser by pushing SSE frames, and
voice mode by speaking `Message` and ignoring the rest.

That last one is the point. `bot_audio.py` today has no document pipeline and never sends
an automation screenshot, because both were added to `bot.py` and `bot_telegram.py` and not
to the third copy. With one event stream there is one place to add a feature and every
surface decides only how to draw it.

Events are plain dataclasses. `to_dict()` exists so the web layer can serialise them
without importing anything else.
"""

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


@dataclass
class Event:
    """Base class. `kind` is derived from the class name for serialisation."""

    @property
    def kind(self) -> str:
        return type(self).__name__.lower()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["kind"] = self.kind
        return payload


@dataclass
class Status(Event):
    """Transient progress line. Replaces the previous status, never accumulates.

    Surfaces that cannot show transient state (voice) drop these entirely.
    """
    text: str


@dataclass
class NodeEnter(Event):
    """A LangGraph node was entered.

    Only the browser surface renders these today — it is the one place the state machine
    can be shown actually running rather than described in a README.
    """
    node: str


@dataclass
class Message(Event):
    """The actual answer. Every surface renders this one."""
    text: str
    markdown: bool = True


@dataclass
class Image(Event):
    """A file on disk to show — currently always an automation screenshot.

    `path` is server-side and is never sent verbatim to a browser; the web layer maps it
    to a basename and serves it through a guarded route.
    """
    path: str
    caption: str = ""


@dataclass
class AwaitConfirm(Event):
    """The human-in-the-loop gate is open and waiting on an explicit CONFIRM.

    This is a *rendering hint only*. Emitting it grants nothing. The citizen still has to
    send the literal text `CONFIRM` back through the ordinary turn path, and the graph's
    interrupt/resume gate is what actually decides whether the click happens. A surface
    that ignores this event entirely still behaves correctly — it just looks worse.
    """
    prompt: str = ""
    session_id: str = ""


@dataclass
class ProfileRequired(Event):
    """The citizen has to fill the profile form before this turn can be useful."""
    text: str = ""


@dataclass
class Error(Event):
    """Something failed. `text` is citizen-facing; `detail` is for logs only.

    Keep PII out of `detail` — it is written to stdout.
    """
    text: str
    detail: str = ""


@dataclass
class Cancelled(Event):
    """The turn was cancelled by /stop."""
    text: str = "🛑 Request cancelled."


# ==============================================================================
# Rendering helpers shared by every surface
# ==============================================================================

def chunk_text(text: str, limit: int) -> list[str]:
    """Split `text` into chunks under `limit`, breaking only at newlines.

    Discord caps a message at 2,000 characters and Telegram at 4,096, and a scheme
    comparison can run past both. Breaking mid-sentence is not acceptable, so splits
    happen at line boundaries.

    A single line longer than `limit` is hard-split as a last resort — the previous
    inline implementations would emit an over-limit chunk in that case and the send
    would be rejected by the platform.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""

    for line in text.split("\n"):
        # A line that cannot fit on its own has to be broken somewhere.
        while len(line) > limit:
            if current:
                chunks.append(current.rstrip("\n"))
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]

        if len(current) + len(line) + 1 > limit:
            chunks.append(current.rstrip("\n"))
            current = line + "\n"
        else:
            current += line + "\n"

    if current.strip():
        chunks.append(current.rstrip("\n"))

    return chunks or [""]
