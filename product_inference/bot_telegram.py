import os
import asyncio
import re
import tempfile
import time
from dotenv import load_dotenv
import threading
import urllib.request

def start_heartbeat():
    def pinger():
        while True:
            try:
                urllib.request.urlopen("https://hc-ping.com/6f985ce4-0f9d-40ab-9b0f-d3dd68cd2957", timeout=10)
            except Exception:
                pass
            time.sleep(60)
    threading.Thread(target=pinger, daemon=True).start()

start_heartbeat()
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

import product_inference.db as db
from core_inference import session as core_session
from core_inference.events import (
    AwaitConfirm, Cancelled, Error, Image, Message, NodeEnter, Status, chunk_text,
)
from core_inference.profile_form import (
    FORM_FLOW, FREE_TEXT_SENTINEL, apply_field, format_profile_summary,
)
from product_inference.web import auth as web_auth

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Telegram rejects a message over 4,096 characters; 4,000 leaves room for entities.
TELEGRAM_CHUNK_LIMIT = 4000

# Placeholder text for the message that later becomes the answer. Kept as constants so
# the renderer knows what is already on screen and skips a no-op edit (Telegram 400s on
# "message is not modified").
PROCESSING_PLACEHOLDER = "🧠 *Yojana Mitra is processing...*"
DOCUMENT_PLACEHOLDER = "📄 *Document detected! Securing file...*"

# Initialize the profile schema parameters if absent
db.init_db()

# Per-user task tracker for /stop cancellation
_active_tasks: dict[str, asyncio.Task] = {}

# ─────────────────────────────────────────────────────────────
# INLINE KEYBOARD FORM — covers all 13 profile fields
# ─────────────────────────────────────────────────────────────
# The field list, options and coercion now live in core_inference.profile_form so the
# Discord widgets, these keyboards and the web form cannot drift apart. Only the
# InlineKeyboardMarkup construction is Telegram-specific and stays here.
#
# In-memory session store: { user_id: { "current_step": int, "awaiting_text": str|None, "data": dict } }
_profile_sessions: dict[str, dict] = {}


def _build_keyboard(rows: list) -> InlineKeyboardMarkup:
    """Build InlineKeyboardMarkup from list of [(label, value)] rows."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(label, callback_data=f"pf|{value}") for label, value in row]
        for row in rows
    ])


async def _send_form_step(user_id: str, step_idx: int, reply_fn):
    """Send the question for step_idx. reply_fn is an async callable (reply_text or answer)."""
    if step_idx >= len(FORM_FLOW):
        await _finish_profile(user_id, reply_fn)
        return

    field, _label, question, keyboard_rows = FORM_FLOW[step_idx]
    _profile_sessions[user_id]["current_step"] = step_idx

    if keyboard_rows:
        kb = _build_keyboard(keyboard_rows)
        await reply_fn(question, reply_markup=kb, parse_mode="Markdown")
    else:
        _profile_sessions[user_id]["awaiting_text"] = field
        await reply_fn(question, parse_mode="Markdown")


async def _finish_profile(user_id: str, reply_fn):
    """Save profile to DB and send confirmation."""
    session = _profile_sessions.pop(user_id, {})
    data = session.get("data", {})

    try:
        db.update_user_state(user_id, "PROFILE_COMPLETE", data)
    except Exception as e:
        print(f"[Profile Save Error] {e}")

    # Telegram's Markdown uses single asterisks for bold; Discord uses double.
    await reply_fn(format_profile_summary(data, bold="*"), parse_mode="Markdown")
# ─────────────────────────────────────────────────────────────

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full nuclear reset: wipes profile, vault, and LangGraph checkpointer state."""
    raw_id = update.effective_user.id
    user_id = f"telegram_{raw_id}"

    # Cancel any running task first
    task = _active_tasks.pop(user_id, None)
    if task and not task.done():
        task.cancel()

    try:
        await asyncio.to_thread(core_session.do_reset, user_id)
        await update.message.reply_text(
            "🔄 **Profile Reset Complete.**\n\n"
            "All your data has been wiped. Send me any message to start fresh!",
            parse_mode="Markdown"
        )
    except Exception as e:
        print(f"[Reset Error] {e}")
        await update.message.reply_text("⚠️ Reset encountered an error. Please try again.")

async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the step-by-step inline keyboard profile form."""
    raw_id = update.effective_user.id
    user_id = f"telegram_{raw_id}"

    # Clear any existing session and start fresh
    _profile_sessions[user_id] = {
        "current_step": 0,
        "awaiting_text": None,
        "data": {
            # Pre-fill boolean defaults; they'll be overwritten by user answers
            "differently_abled": False, "disability_percentage": 0,
            "minority": False, "below_poverty_line": False,
            "economic_distress": False, "government_employee": False,
        }
    }

    await update.message.reply_text(
        "📋 Setting up your Yojana Mitra Profile\n\n"
        "Please answer the questions below by tapping the buttons. "
        "This takes about 2 minutes and helps me find the best schemes for you!"
    )
    await _send_form_step(user_id, 0, update.message.reply_text)


async def profile_form_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button taps during the inline keyboard profile form."""
    query = update.callback_query
    await query.answer()

    raw_id = query.from_user.id
    user_id = f"telegram_{raw_id}"
    data = query.data  # format: "pf|value"

    if not data.startswith("pf|"):
        return

    value = data[3:]  # strip "pf|" prefix

    if user_id not in _profile_sessions:
        await query.edit_message_text("⚠️ Your profile session expired. Please run /profile again.")
        return

    session = _profile_sessions[user_id]
    step_idx = session.get("current_step", 0)
    field, _label, question, _kb = FORM_FLOW[step_idx]

    # User chose "type your own" for occupation
    if value == FREE_TEXT_SENTINEL:
        session["awaiting_text"] = field
        await query.edit_message_text("✏️ Please *type your occupation* and send it as a message:", parse_mode="Markdown")
        return

    # Coercion (including the derived disability_percentage) lives in profile_form.
    ok, error = apply_field(session["data"], field, value)
    if not ok:
        await query.edit_message_text(error)
        return

    stored = session["data"][field]
    display_val = ("Yes" if stored else "No") if isinstance(stored, bool) else stored

    # Show confirmed selection in the message
    clean_q = question.replace("*", "").replace("\\", "")
    await query.edit_message_text(f"✅ {clean_q}\n*Selected:* {display_val}", parse_mode="Markdown")

    # Move to next step
    next_idx = step_idx + 1
    await _send_form_step(user_id, next_idx, query.message.reply_text)

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancels the current running automation task. Silently ignores if nothing is running."""
    raw_id = update.effective_user.id
    user_id = f"telegram_{raw_id}"
    task = _active_tasks.get(user_id)
    if task and not task.done():
        _active_tasks.pop(user_id, None)
        task.cancel()
        await update.message.reply_text("🛑 **Stopped.** Send a new message to continue.", parse_mode="Markdown")
    # If nothing is running, say nothing — don't interrupt the user's flow


async def web_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Hand this Telegram identity over to the browser surface via a signed deep link."""
    raw_id = update.effective_user.id
    user_id = f"telegram_{raw_id}"

    await update.message.reply_text(
        "🌐 *Open Yojana Mitra in your browser*\n\n"
        f"{web_auth.handoff_url(user_id)}\n\n"
        "Same profile, same conversation — just a bigger screen.\n\n"
        "⚠️ This link signs in *as you* and expires in 5 minutes. "
        "Don't forward it: anyone who opens it gets your account.",
        parse_mode="Markdown",
        disable_web_page_preview=True,
    )


# ─────────────────────────────────────────────────────────────
# EVENT RENDERER
# ─────────────────────────────────────────────────────────────
# core_session emits platform-neutral events; everything below decides what a Telegram
# chat does with them. The Discord and web surfaces subscribe to the same stream and
# draw it differently — that is the whole point of the split.

def _tg_markdown(text: str) -> str:
    """Core events are written with Discord's `**bold**`; Telegram's legacy Markdown
    parser wants `*bold*` and renders the doubled form as literal asterisks."""
    return re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.S)


def _render_attempts(text: str, markdown: bool):
    """Progressively less ambitious ways to put `text` on screen.

    LLM output routinely contains an odd number of asterisks or an unclosed backtick,
    which Telegram rejects outright with a 400 rather than rendering as-is. Falling back
    to plain text means a malformed response still reaches the citizen.
    """
    if markdown:
        yield _tg_markdown(text), {"parse_mode": "Markdown"}
    yield text, {}


async def _render_events(events, update, context, status_msg, shown: str = ""):
    """Draw a core_session event stream into a Telegram chat.

    `status_msg` is the placeholder already on screen: progress edits it in place and the
    final answer replaces it, which is how the bot behaved before events existed. `shown`
    is that placeholder's text, so an identical first Status is a no-op instead of a 400.

    Overflow chunks become *new* messages. The previous implementation edited the same
    message once per chunk and then edited it again with the last chunk, so a long answer
    arrived as its final 4,000 characters with everything before it silently lost.
    """
    chat_id = update.effective_chat.id
    slot = status_msg
    last_text = shown

    async def _emit(text: str, markdown: bool):
        nonlocal slot, last_text
        if text == last_text:
            return
        last_text = text
        for body, kwargs in _render_attempts(text, markdown):
            try:
                await slot.edit_text(body, **kwargs)
                return
            except Exception:
                continue
        # The slot itself is unusable — too old to edit, or deleted. Start a new message.
        try:
            slot = await update.message.reply_text(text)
        except Exception as e:
            print(f"[Telegram] Dropped message for chat {chat_id}: {e}")

    async for ev in events:
        if isinstance(ev, Status):
            await _emit(ev.text, markdown=True)

        elif isinstance(ev, NodeEnter):
            # The state-machine trace is a web side-panel affordance; chat stays quiet.
            continue

        elif isinstance(ev, (Message, Error, Cancelled)):
            chunks = chunk_text(ev.text, TELEGRAM_CHUNK_LIMIT)
            await _emit(chunks[0], markdown=True)
            for extra in chunks[1:]:
                try:
                    slot = await update.message.reply_text(extra)
                    last_text = extra
                except Exception as e:
                    print(f"[Telegram] Dropped overflow chunk for chat {chat_id}: {e}")

        elif isinstance(ev, Image):
            try:
                with open(ev.path, 'rb') as photo_file:
                    await context.bot.send_photo(chat_id=chat_id, photo=photo_file)
            except Exception as e:
                print(f"[Telegram] Could not send screenshot {ev.path}: {e}")

        elif isinstance(ev, AwaitConfirm):
            # Deliberately not a button. The graph's own reply spells out the CONFIRM
            # instruction, and the zero-auto-submit gate must stay a typed human act —
            # adding a one-tap shortcut here would make this surface a third place that
            # decides whether a government form gets submitted.
            continue


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
        
    # Enforce DM-only (private chats)
    if update.effective_chat.type != "private":
        await update.message.reply_text("Please DM me directly to securely check eligibility!")
        return

    # Platform-prefixed ID
    raw_id = update.effective_user.id
    user_id = f"telegram_{raw_id}"

    # ---------------------------------------------------------
    # ATTACHMENTS → PII VAULT PIPELINE
    # ---------------------------------------------------------
    # Getting the bytes onto disk is the only Telegram-specific part. The bouncer, the
    # three vision passes, the cache check, the vault UPSERT and the graph resume all
    # live in core_session.ingest_document, shared with Discord and the web surface.

    # In Telegram, attachments come as Photo (compressed) or Document (file)
    file_obj = None
    filename = "unknown_file"

    if update.message.photo:
        file_obj = update.message.photo[-1]  # Highest resolution photo
        filename = f"photo_{int(time.time())}.jpg"
    elif update.message.document:
        doc = update.message.document
        if doc.file_name and doc.file_name.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            file_obj = doc
            filename = doc.file_name

    if file_obj:
        status_msg = await update.message.reply_text(DOCUMENT_PLACEHOLDER, parse_mode="Markdown")

        try:
            tg_file = await context.bot.get_file(file_obj.file_id)
            temp_path = os.path.join(
                tempfile.gettempdir(), f"tg_{int(time.time())}_{filename}"
            )
            await tg_file.download_to_drive(temp_path)
        except Exception as e:
            await status_msg.edit_text(f"❌ Failed to download file from Telegram: {e}")
            return

        # Telegram omits file_size on some photo sizes; fall back to what landed on disk.
        file_size = getattr(file_obj, 'file_size', None) or os.path.getsize(temp_path)

        await _render_events(
            core_session.ingest_document(user_id, temp_path, filename, file_size),
            update, context, status_msg, shown=DOCUMENT_PLACEHOLDER,
        )
        return

    # ---------------------------------------------------------
    # TEXT PROCESSING
    # ---------------------------------------------------------

    text_input = update.message.text
    if not text_input:
        return

    # /stop via text (in addition to /stop command)
    if text_input.strip().lower() in ("/stop", "stop"):
        task = _active_tasks.pop(user_id, None)
        if task and not task.done():
            task.cancel()
            await update.message.reply_text("🛑 **Stopped!** Your previous request has been cancelled. You can send a new message anytime.", parse_mode="Markdown")
        else:
            await update.message.reply_text("Nothing is running right now. Send me a message to get started!")
        return

    # ─── FORM SESSION: intercept text when in an inline-keyboard form ───
    if user_id in _profile_sessions:
        session = _profile_sessions[user_id]
        awaiting = session.get("awaiting_text")

        if awaiting:
            # Parsing and title-casing live in profile_form, so a typed answer here and a
            # typed answer in the Discord modal or the web form are coerced identically.
            ok, error = apply_field(session["data"], awaiting, text_input.strip())
            if not ok:
                await update.message.reply_text(error)
                return

            session["awaiting_text"] = None
            step_idx = session.get("current_step", 0)
            await _send_form_step(user_id, step_idx + 1, update.message.reply_text)
            return  # Do NOT pass to LangGraph
        else:
            # Mid-form but not awaiting text — remind user to use buttons
            await update.message.reply_text(
                "📋 Profile form in progress!\n\n"
                "Please use the buttons to answer the current question.\n"
                "Run /profile to restart if you lost track."
            )
            return

    # If this user already has a task running, cancel it before starting a new one
    old_task = _active_tasks.pop(user_id, None)
    if old_task and not old_task.done():
        old_task.cancel()

    status_msg = await update.message.reply_text(PROCESSING_PLACEHOLDER, parse_mode="Markdown")

    async def _run_graph():
        try:
            # text_input goes through untouched — including a literal CONFIRM. The
            # human-in-the-loop gate belongs to the graph's interrupt/resume round-trip;
            # this handler must not pre-empt it.
            await _render_events(
                core_session.run_turn(user_id, text_input, status_text=PROCESSING_PLACEHOLDER),
                update, context, status_msg, shown=PROCESSING_PLACEHOLDER,
            )
        except asyncio.CancelledError:
            # run_turn emits Cancelled before re-raising, but /stop can land between two
            # events, before the renderer has anything to draw.
            try:
                await status_msg.edit_text("🛑 Request cancelled.")
            except Exception:
                pass
            print(f"[/stop] Task cancelled for user {user_id}")
        except Exception as e:
            try:
                await status_msg.edit_text(
                    "⚠️ Operational database exception encountered handling text generation pipelines."
                )
            except Exception:
                pass
            print(f"Runtime Exception Event: {e}")
        finally:
            _active_tasks.pop(user_id, None)

    # Launch as a tracked asyncio Task
    _active_tasks[user_id] = asyncio.create_task(_run_graph())

def main():
    if not TOKEN:
        print("[Error] TELEGRAM_TOKEN environment variable not found. Please add it to your .env file.")
        return
        
    print(f'==========================================')
    print(f' Yojana Mitra Telegram Bot Active')
    print(f'==========================================')
    
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("profile", profile_command))
    # Signed deep link that carries this Telegram identity into the browser surface
    app.add_handler(CommandHandler("web", web_command))
    # Inline keyboard button taps for the profile form
    app.add_handler(CallbackQueryHandler(profile_form_callback, pattern=r"^pf\|"))
    # Handle text and photos/documents (profile form intercepts text when mid-form)
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.Document.ALL, handle_message))

    app.run_polling()

if __name__ == '__main__':
    main()
