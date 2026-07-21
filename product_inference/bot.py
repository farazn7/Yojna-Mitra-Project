import os
import asyncio
import signal
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Import database initializer and compiled LangGraph runtime
import product_inference.db as db
from core_inference.graph import graph_app

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Initialize the profile schema parameters if absent
db.init_db()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Per-user task tracker for /stop cancellation
_active_tasks: dict[str, asyncio.Task] = {}

@bot.event
async def on_ready():
    print(f'==========================================')
    print(f' Yojana Mitra Powered by LangGraph Active')
    print(f'==========================================')

async def _do_reset(user_id: str) -> bool:
    """Shared reset logic: wipes profile, vault, and checkpointer state from DB."""
    import psycopg2
    from product_inference.db import DB_PARAMS
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    cur.execute("DELETE FROM user_profiles WHERE platform_id = %s;", (user_id,))
    cur.execute("DELETE FROM pii_vault WHERE user_id = %s;", (user_id,))
    cur.execute("DELETE FROM checkpoints WHERE thread_id = %s;", (user_id,))
    cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s;", (user_id,))
    conn.commit()
    cur.close()
    conn.close()

@bot.slash_command(name="reset", description="Wipe your profile and start fresh.")
async def reset_slash(ctx):
    user_id = f"discord_{ctx.author.id}"
    task = _active_tasks.pop(user_id, None)
    if task and not task.done():
        task.cancel()
    try:
        await asyncio.to_thread(_do_reset, user_id)
        await ctx.respond("🔄 **Profile Reset Complete.**\n\nAll your data has been wiped. Send me any message to start fresh!", ephemeral=True)
    except Exception as e:
        print(f"[Reset Error] {e}")
        await ctx.respond("⚠️ Reset encountered an error. Please try again.", ephemeral=True)

@bot.slash_command(name="stop", description="Stop the currently running task.")
async def stop_slash(ctx):
    user_id = f"discord_{ctx.author.id}"
    task = _active_tasks.get(user_id)
    if task and not task.done():
        _active_tasks.pop(user_id, None)
        task.cancel()
        await ctx.respond("🛑 **Stopped.** Send a new message to continue.", ephemeral=True)
    # Silently ignore if nothing is running


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Enforce safe direct messaging parameters
    if not isinstance(message.channel, discord.DMChannel):
        await message.channel.send(f"Hi {message.author.mention}, please DM me directly to securely check eligibility!")
        return

    user_id = f"discord_{message.author.id}"

    # ---------------------------------------------------------
    # NEW CODE: 3-PASS DOCUMENT INTERCEPTOR & PII VAULT SAVING
    # ---------------------------------------------------------
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                processing_msg = await message.channel.send("📄 *Document detected! Securing file...*")
                
                # Check if LangGraph is waiting for a specific document right now
                config = {"configurable": {"thread_id": user_id}}
                try:
                    graph_state = await asyncio.to_thread(graph_app.get_state, config)
                    expected_doc = graph_state.values.get("awaiting_document", "") if graph_state and hasattr(graph_state, "values") else ""
                except Exception as e:
                    print(f"[Bot State Check Error] Could not fetch graph state: {e}")
                    expected_doc = ""
                
                # Hand it to the Bouncer (keep high-res by default for AI Vision, use smart naming)
                from product_inference.document_handler import download_and_process
                doc_label_for_file = expected_doc if expected_doc else "unsolicited"
                success, result = await download_and_process(attachment, user_id, sanitize_for_portal=False, doc_type_label=doc_label_for_file)
                
                if not success:
                    await processing_msg.edit(content=result)
                    return

                from product_inference.document_extractor import classify_document, verify_document_type, analyze_document
                import json
                
                # PASS 0: Verify if expected, otherwise classify
                if expected_doc:
                    await processing_msg.edit(content=f"🔍 *Verifying if document matches requested **{expected_doc.replace('_', ' ').title()}**...*")
                    matches, reason_or_type = await asyncio.to_thread(verify_document_type, result, expected_doc)
                    if not matches:
                        await processing_msg.edit(content=f"⚠️ I requested your **{expected_doc.replace('_', ' ').title()}**, but this looks like something else.\n*(AI verification: {reason_or_type})*\n\nPlease upload a clear photo of your **{expected_doc.replace('_', ' ').title()}** to continue.")
                        return
                    doc_type = expected_doc
                    display_label = doc_type.replace("_", " ").title()
                else:
                    await processing_msg.edit(content="📄 *File secured! Classifying document type...*")
                    doc_type = await asyncio.to_thread(classify_document, result)
                    display_label = doc_type.replace("_", " ").title()
                    
                    # CACHE CHECK: Only for known, specific doc types (not back_of_card or unknown) during unsolicited upload
                    skip_cache_types = {"back_of_card", "unknown"}
                    if doc_type not in skip_cache_types:
                        cached_data = await asyncio.to_thread(db.check_vault_cache, user_id, doc_type)
                        if cached_data:
                            await processing_msg.edit(content=f"✅ **Secure Cache Hit!** I already have your verified **{display_label}** details stored in the PII Vault (expires in 24 hours). No need to re-scan!")
                            return

                await processing_msg.edit(content=f"🔍 *Running AI Vision extraction & validation on **{display_label}**...*")
                
                # PASS 1 & 2: Targeted extraction, JSON structuring, and regex validation
                analysis = await asyncio.to_thread(analyze_document, result, doc_type)
                
                if analysis["success"]:
                    extracted_fields = analysis["extracted_data"]
                    
                    if not extracted_fields:
                        await processing_msg.edit(content="⚠️ The document was processed but no text could be extracted. Try a clearer, better-lit photo.")
                    elif analysis["is_valid"]:
                        # SAVE TO VAULT: Store validated data, merging into existing row if present
                        await asyncio.to_thread(
                            db.upsert_vault,
                            user_id=user_id,
                            document_type=doc_type,
                            extracted_fields=extracted_fields,
                            source_meta={"filename": attachment.filename, "file_path": result},
                            ttl_hours=24
                        )
                        
                        if expected_doc:
                            # RESUME LANGGRAPH: The graph was paused waiting for this document!
                            await processing_msg.edit(content=f"✅ **{display_label} Verified & Saved!** Updating application progress...")
                            graph_output = await asyncio.to_thread(
                                graph_app.invoke,
                                {
                                    "messages": [{"role": "user", "content": f"[DOC_RECEIVED: {doc_type}]"}],
                                    "user_id": user_id
                                },
                                config=config
                            )
                            ai_response = graph_output.get("response", "✅ Document received!")
                            await processing_msg.edit(content=ai_response)
                        else:
                            # Unsolicited upload in manual testing mode
                            preview = json.dumps(extracted_fields, indent=2)
                            if len(preview) > 500:
                                preview = preview[:497] + "..."
                                
                            await processing_msg.edit(content=f"✅ **{display_label} Saved to Vault!**\n\n```json\n{preview}\n```\n*This PII is isolated from your main profile and will auto-expire in 24 hours.*")
                    else:
                        # Validation failed on a critical field — tell the user exactly what's wrong
                        errors_str = "\n".join([f"- {err}" for err in analysis["validation_errors"]])
                        await processing_msg.edit(content=f"⚠️ **Extracted {display_label}, but found an issue:**\n{errors_str}\n\n*Please upload a clearer/complete photo.*")
                else:
                    await processing_msg.edit(content=f"⚠️ Extraction failed: {analysis.get('raw_text', 'Could not read document.')}")
                
                # CRITICAL: Stop here so LangGraph doesn't process a blank text message
                return
    # ---------------------------------------------------------

    # If it wasn't an attachment, proceed to LangGraph text processing
    text_input = message.content.strip()

    # /reset command — full nuclear wipe
    if text_input.lower() in ("/reset", "!reset", "reset"):
        task = _active_tasks.pop(user_id, None)
        if task and not task.done():
            task.cancel()
        try:
            await asyncio.to_thread(_do_reset, user_id)
            await message.channel.send("🔄 **Profile Reset Complete.**\n\nAll your data has been wiped. Send me any message to start fresh!")
        except Exception as e:
            print(f"[Reset Error] {e}")
            await message.channel.send("⚠️ Reset encountered an error. Please try again.")
        return

    # /stop command — cancel in-progress task only if one is running (silent otherwise)
    if text_input.lower() in ("/stop", "!stop", "stop"):
        task = _active_tasks.get(user_id)
        if task and not task.done():
            _active_tasks.pop(user_id, None)
            task.cancel()
            await message.channel.send("🛑 **Stopped.** Send a new message to continue.")
        return

    # If this user already has a task running, cancel it before starting a new one
    old_task = _active_tasks.pop(user_id, None)
    if old_task and not old_task.done():
        old_task.cancel()

    status_msg = await message.channel.send(" *Yojana Mitra is processing...*")

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
            await status_msg.delete()

            # Check if the automation engine reached a HITL checkpoint or Final Confirmation gate with a screenshot
            automation_status = graph_output.get("automation_status", "idle")
            auto_session_id = graph_output.get("automation_session_id", f"auto_{user_id}")
            
            discord_file = None
            if automation_status in ("hitl_paused", "awaiting_confirm"):
                shot_name = f"{auto_session_id}_final_review.png" if automation_status == "awaiting_confirm" else f"{auto_session_id}_otp_intercept.png"
                shot_path = os.path.join("screenshots", shot_name)
                if os.path.exists(shot_path):
                    try:
                        discord_file = discord.File(shot_path, filename=shot_name)
                    except Exception as e:
                        print(f"[Screenshot Attachment Note] Could not load image {shot_path}: {e}")

            # SAFE CHUNKING: Break up responses longer than 2000 characters
            if len(ai_response) > 2000:
                lines = ai_response.split('\n')
                current_chunk = ""
                
                for line in lines:
                    if len(current_chunk) + len(line) + 1 > 1900:
                        await message.channel.send(current_chunk)
                        current_chunk = line + '\n'
                    else:
                        current_chunk += line + '\n'
                
                if current_chunk.strip():
                    if discord_file:
                        await message.channel.send(current_chunk, file=discord_file)
                    else:
                        await message.channel.send(current_chunk)
            else:
                if discord_file:
                    await message.channel.send(ai_response, file=discord_file)
                else:
                    await message.channel.send(ai_response)

        except asyncio.CancelledError:
            try:
                await status_msg.delete()
            except:
                pass
            print(f"[/stop] Task cancelled for user {user_id}")
        except Exception as e:
            try:
                await status_msg.delete()
            except:
                pass
            await message.channel.send(" Operational database exception encountered handling text generation pipelines.")
            print(f"Runtime Exception Event: {e}")
        finally:
            _active_tasks.pop(user_id, None)

    # Launch as a tracked asyncio Task
    _active_tasks[user_id] = asyncio.create_task(_run_graph())

# Graceful shutdown on Ctrl+C
def _handle_shutdown(sig, frame):
    print("\n[Shutdown] Gracefully stopping Yojana Mitra Discord bot...")
    for uid, task in _active_tasks.items():
        task.cancel()
    _active_tasks.clear()
    raise SystemExit(0)

signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)

bot.run(TOKEN)