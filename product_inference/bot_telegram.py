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
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import product_inference.db as db
from core_inference.graph import graph_app

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Initialize the profile schema parameters if absent
db.init_db()

# Per-user task tracker for /stop cancellation
_active_tasks: dict[str, asyncio.Task] = {}

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
            if automation_status in ("hitl_paused", "awaiting_confirm"):
                shot_name = f"{auto_session_id}_final_review.png" if automation_status == "awaiting_confirm" else f"{auto_session_id}_otp_intercept.png"
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
    # Handle text and photos/documents
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.Document.ALL, handle_message))
    
    app.run_polling()

if __name__ == '__main__':
    main()
