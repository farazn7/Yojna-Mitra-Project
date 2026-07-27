import os
import asyncio
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
from core_inference.graph import graph_app

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Initialize the profile schema parameters if absent
db.init_db()

# Per-user task tracker for /stop cancellation
_active_tasks: dict[str, asyncio.Task] = {}

# ─────────────────────────────────────────────────────────────
# INLINE KEYBOARD FORM — covers all 13 profile fields
# ─────────────────────────────────────────────────────────────
# In-memory session store: { user_id: { "current_step": int, "awaiting_text": str|None, "data": dict } }
_profile_sessions: dict[str, dict] = {}

FORM_FLOW = [
    # (field, step_label, question_text, keyboard_rows or None for text input)
    ("name",                "1/13",
     "👤 *Step 1/13* — What is your *full name*?",
     None),  # text input
    ("gender",              "2/13",
     "👤 *Step 2/13* — What is your *gender*?",
     [[("👨 Male", "Male"), ("👩 Female", "Female"), ("🌈 Other", "Other")]]),
    ("residence",           "3/13",
     "🏘 *Step 3/13* — Where do you *live*?",
     [[("🌾 Rural", "Rural"), ("🏙 Urban", "Urban")]]),
    ("caste",               "4/13",
     "📋 *Step 4/13* — What is your *caste category*?",
     [[("General", "General"), ("OBC", "OBC")], [("SC", "SC"), ("ST", "ST")]]),
    ("marital_status",      "5/13",
     "💍 *Step 5/13* — What is your *marital status*?",
     [[("Single", "Single"), ("Married", "Married")], [("Widowed", "Widowed"), ("Divorced", "Divorced")]]),
    ("occupation",          "6/13",
     "💼 *Step 6/13* — What is your *occupation*?",
     [[("🌾 Farmer", "Farmer"), ("📚 Student", "Student")],
      [("💼 Salaried", "Salaried"), ("🏪 Business Owner", "Business Owner")],
      [("🧵 Artisan", "Artisan"), ("❌ Unemployed", "Unemployed")],
      [("✏️ Type your own...", "__text__")]]),
    ("differently_abled",   "7/13",
     "♿ *Step 7/13* — Are you *differently abled*?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("minority",            "8/13",
     "🕌 *Step 8/13* — Do you belong to a *minority community*?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("below_poverty_line",  "9/13",
     "🏠 *Step 9/13* — Do you have a *BPL card* (Below Poverty Line)?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("economic_distress",   "10/13",
     "📉 *Step 10/13* — Are you facing *economic distress*?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("government_employee", "11/13",
     "🏗 *Step 11/13* — Are you a *government employee*?",
     [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("age",                 "12/13",
     "🔢 *Step 12/13* — Please *type your age* in years:",
     None),  # text input
    ("income",              "13/13",
     "💰 *Step 13/13* — Please type your annual family *income in ₹* (e.g. 45000):",
     None),  # text input
]

FORM_FIELDS = [f[0] for f in FORM_FLOW]


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

    g = data.get
    summary = (
        "🎉 *Profile Saved Successfully!*\n\n"
        f"• *Name:* {g('name', '—')}\n"
        f"• *Gender:* {g('gender', '—')}\n"
        f"• *Age:* {g('age', '—')} yrs\n"
        f"• *Income:* ₹{int(g('income', 0)):,}\n"
        f"• *Caste:* {g('caste', '—')}\n"
        f"• *Occupation:* {g('occupation', '—')}\n"
        f"• *Residence:* {g('residence', '—')}\n"
        f"• *Marital Status:* {g('marital_status', '—')}\n"
        f"• *Differently Abled:* {'Yes' if g('differently_abled') else 'No'}\n"
        f"• *Minority:* {'Yes' if g('minority') else 'No'}\n"
        f"• *BPL Card:* {'Yes' if g('below_poverty_line') else 'No'}\n"
        f"• *Economic Distress:* {'Yes' if g('economic_distress') else 'No'}\n"
        f"• *Govt Employee:* {'Yes' if g('government_employee') else 'No'}\n\n"
        "You're all set! Ask me to *find schemes* for you now! 🎯"
    )
    await reply_fn(summary, parse_mode="Markdown")
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
        import psycopg2
        from product_inference.db import DB_PARAMS
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        # Wipe user profile and conversation state
        cur.execute("DELETE FROM user_profiles WHERE platform_id = %s;", (user_id,))
        # Wipe PII vault
        cur.execute("DELETE FROM pii_vault WHERE user_id = %s;", (user_id,))
        # Wipe LangGraph checkpointer state (thread_id = user_id)
        cur.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (user_id,))
        cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s;", (user_id,))
        conn.commit()
        cur.close()
        conn.close()
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
    if value == "__text__":
        session["awaiting_text"] = field
        await query.edit_message_text("✏️ Please *type your occupation* and send it as a message:", parse_mode="Markdown")
        return

    # Save the value
    if value in ("true", "false"):
        session["data"][field] = (value == "true")
        if field == "differently_abled":
            session["data"]["disability_percentage"] = 40 if session["data"][field] else 0
        display_val = "Yes" if value == "true" else "No"
    else:
        session["data"][field] = value
        display_val = value

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
    # 3-PASS DOCUMENT INTERCEPTOR & PII VAULT SAVING
    # ---------------------------------------------------------
    
    # In Telegram, attachments come as Photo (compressed) or Document (file)
    file_obj = None
    filename = "unknown_file"
    
    if update.message.photo:
        file_obj = update.message.photo[-1] # Highest resolution photo
        filename = f"photo_{int(time.time())}.jpg"
    elif update.message.document:
        doc = update.message.document
        if doc.file_name.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            file_obj = doc
            filename = doc.file_name
            
    if file_obj:
        processing_msg = await update.message.reply_text("📄 *Document detected! Securing file...*", parse_mode="Markdown")
        
        # Check if LangGraph is waiting for a specific document right now
        config = {"configurable": {"thread_id": user_id}}
        try:
            graph_state = await asyncio.to_thread(graph_app.get_state, config)
            expected_doc = graph_state.values.get("awaiting_document", "") if graph_state and hasattr(graph_state, "values") else ""
        except Exception as e:
            print(f"[Bot State Check Error] Could not fetch graph state: {e}")
            expected_doc = ""
            
        doc_label_for_file = expected_doc if expected_doc else "unsolicited"
        
        # Download the file using python-telegram-bot
        try:
            tg_file = await context.bot.get_file(file_obj.file_id)
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"tg_{int(time.time())}_{filename}")
            await tg_file.download_to_drive(temp_path)
        except Exception as e:
            await processing_msg.edit_text(f"❌ Failed to download file from Telegram: {e}")
            return
            
        file_size = getattr(file_obj, 'file_size', os.path.getsize(temp_path))

        # Hand it to the Bouncer
        from product_inference.document_handler import download_and_process_file
        success, result = await download_and_process_file(temp_path, filename, file_size, user_id, sanitize_for_portal=False, doc_type_label=doc_label_for_file)
        
        if not success:
            await processing_msg.edit_text(result)
            return

        from product_inference.document_extractor import classify_document, verify_document_type, analyze_document
        import json
        
        # PASS 0: Verify if expected, otherwise classify
        if expected_doc:
            await processing_msg.edit_text(f"🔍 *Verifying if document matches requested **{expected_doc.replace('_', ' ').title()}**...*", parse_mode="Markdown")
            matches, reason_or_type = await asyncio.to_thread(verify_document_type, result, expected_doc)
            if not matches:
                await processing_msg.edit_text(f"⚠️ I requested your **{expected_doc.replace('_', ' ').title()}**, but this looks like something else.\n*(AI verification: {reason_or_type})*\n\nPlease upload a clear photo of your **{expected_doc.replace('_', ' ').title()}** to continue.", parse_mode="Markdown")
                return
            doc_type = expected_doc
            display_label = doc_type.replace("_", " ").title()
        else:
            await processing_msg.edit_text("📄 *File secured! Classifying document type...*", parse_mode="Markdown")
            doc_type = await asyncio.to_thread(classify_document, result)
            display_label = doc_type.replace("_", " ").title()
            
            # CACHE CHECK
            skip_cache_types = {"back_of_card", "unknown"}
            if doc_type not in skip_cache_types:
                cached_data = await asyncio.to_thread(db.check_vault_cache, user_id, doc_type)
                if cached_data:
                    await processing_msg.edit_text(f"✅ **Secure Cache Hit!** I already have your verified **{display_label}** details stored in the PII Vault (expires in 24 hours). No need to re-scan!", parse_mode="Markdown")
                    return

        await processing_msg.edit_text(f"🔍 *Running AI Vision extraction & validation on **{display_label}**...*", parse_mode="Markdown")
        
        # PASS 1 & 2
        analysis = await asyncio.to_thread(analyze_document, result, doc_type)
        
        if analysis["success"]:
            extracted_fields = analysis["extracted_data"]
            
            if not extracted_fields:
                await processing_msg.edit_text("⚠️ The document was processed but no text could be extracted. Try a clearer, better-lit photo.")
            elif analysis["is_valid"]:
                # SAVE TO VAULT
                await asyncio.to_thread(
                    db.upsert_vault,
                    user_id=user_id,
                    document_type=doc_type,
                    extracted_fields=extracted_fields,
                    source_meta={"filename": filename, "file_path": result},
                    ttl_hours=24
                )
                
                if expected_doc:
                    # RESUME LANGGRAPH
                    await processing_msg.edit_text(f"✅ **{display_label} Verified & Saved!** Updating application progress...", parse_mode="Markdown")
                    graph_output = await asyncio.to_thread(
                        graph_app.invoke,
                        {
                            "messages": [{"role": "user", "content": f"[DOC_RECEIVED: {doc_type}]"}],
                            "user_id": user_id
                        },
                        config=config
                    )
                    ai_response = graph_output.get("response", "✅ Document received!")
                    await processing_msg.edit_text(ai_response)
                else:
                    preview = json.dumps(extracted_fields, indent=2)
                    if len(preview) > 500:
                        preview = preview[:497] + "..."
                    await processing_msg.edit_text(f"✅ **{display_label} Saved to Vault!**\n\n```json\n{preview}\n```\n*This PII is isolated from your main profile and will auto-expire in 24 hours.*", parse_mode="Markdown")
            else:
                errors_str = "\n".join([f"- {err}" for err in analysis["validation_errors"]])
                await processing_msg.edit_text(f"⚠️ **Extracted {display_label}, but found an issue:**\n{errors_str}\n\n*Please upload a clearer/complete photo.*", parse_mode="Markdown")
        else:
            await processing_msg.edit_text(f"⚠️ Extraction failed: {analysis.get('raw_text', 'Could not read document.')}")
            
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
            val = text_input.strip()

            if awaiting == "age":
                clean = val.replace(" ", "")
                if not clean.isdigit():
                    await update.message.reply_text("❌ Please enter a valid number for age (e.g. 22):")
                    return
                session["data"]["age"] = int(clean)
            elif awaiting == "income":
                clean = val.replace(",", "").replace("₹", "").replace(" ", "")
                if not clean.isdigit():
                    await update.message.reply_text("❌ Please enter a valid number for income (e.g. 45000):")
                    return
                session["data"]["income"] = int(clean)
            elif awaiting == "occupation":
                session["data"]["occupation"] = val.title()
            elif awaiting == "name":
                session["data"]["name"] = val.title()
            else:
                session["data"][awaiting] = val

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

    status_msg = await update.message.reply_text(" *Yojana Mitra is processing...*", parse_mode="Markdown")

    async def _run_graph():
        try:
            config = {"configurable": {"thread_id": user_id}}
            
            graph_output = await asyncio.to_thread(
                graph_app.invoke,
                {
                    "messages": [{"role": "user", "content": text_input}],
                    "user_id": user_id
                },
                config=config
            )

            ai_response = graph_output.get("response", "I encountered a processing anomaly. Please retry.")
            if not ai_response or not ai_response.strip():
                ai_response = "I processed your request, but the generated response was empty. Please try asking your question again!"

            # Handle screenshots
            automation_status = graph_output.get("automation_status", "idle")
            auto_session_id = graph_output.get("automation_session_id", f"auto_{user_id}")
            
            photo_path = None
            # A completed submission has its own screenshot — what the portal showed after the final
            # press, including any acknowledgement number. It is the one record the citizen most needs
            # to keep, and it was previously never sent: "complete" was not among the statuses checked.
            shot_by_status = {
                "awaiting_confirm": f"{auto_session_id}_final_review.png",
                "complete": f"{auto_session_id}_submitted.png",
                "hitl_paused": f"{auto_session_id}_otp_intercept.png",
            }
            if automation_status in shot_by_status:
                shot_name = shot_by_status[automation_status]
                shot_path = os.path.join("screenshots", shot_name)
                if os.path.exists(shot_path):
                    photo_path = shot_path

            # Safe Chunking for Telegram (4096 char limit)
            if len(ai_response) > 4000:
                lines = ai_response.split('\n')
                current_chunk = ""
                for line in lines:
                    if len(current_chunk) + len(line) + 1 > 4000:
                        await status_msg.edit_text(current_chunk)
                        status_msg_new = await update.message.reply_text("...") 
                        current_chunk = line + '\n'
                    else:
                        current_chunk += line + '\n'
                if current_chunk.strip():
                    await status_msg.edit_text(current_chunk)
            else:
                await status_msg.edit_text(ai_response)
                
            # Send screenshot if exists
            if photo_path:
                with open(photo_path, 'rb') as photo_file:
                    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=photo_file)

        except asyncio.CancelledError:
            try:
                await status_msg.edit_text("🛑 Request cancelled.")
            except:
                pass
            print(f"[/stop] Task cancelled for user {user_id}")
        except Exception as e:
            try:
                await status_msg.edit_text(" Operational database exception encountered handling text generation pipelines.")
            except:
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
    # Inline keyboard button taps for the profile form
    app.add_handler(CallbackQueryHandler(profile_form_callback, pattern=r"^pf\|"))
    # Handle text and photos/documents (profile form intercepts text when mid-form)
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.Document.ALL, handle_message))

    app.run_polling()

if __name__ == '__main__':
    main()
