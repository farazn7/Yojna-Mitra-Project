import os
import asyncio
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

@bot.event
async def on_ready():
    print(f'==========================================')
    print(f' Yojana Mitra Powered by LangGraph Active')
    print(f'==========================================')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Enforce safe direct messaging parameters
    if not isinstance(message.channel, discord.DMChannel):
        await message.channel.send(f"Hi {message.author.mention}, please DM me directly to securely check eligibility!")
        return

    user_id = str(message.author.id)

    # ---------------------------------------------------------
    # NEW CODE: 3-PASS DOCUMENT INTERCEPTOR & PII VAULT SAVING
    # ---------------------------------------------------------
    if message.attachments:
        for attachment in message.attachments:
            if attachment.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
                processing_msg = await message.channel.send("📄 *Document detected! Securing file...*")
                
                # Hand it to the Bouncer (keep high-res by default for AI Vision)
                from product_inference.document_handler import download_and_process
                success, result = await download_and_process(attachment, user_id, sanitize_for_portal=False)
                
                if not success:
                    await processing_msg.edit(content=result)
                    return

                await processing_msg.edit(content="📄 *File secured! Classifying document type...*")
                
                from product_inference.document_extractor import classify_document, analyze_document
                import json
                
                # PASS 0: Identify what kind of document this is
                doc_type = await asyncio.to_thread(classify_document, result)
                
                # If the classifier returns 'back_of_card', we don't cache-check (it's not
                # a standalone doc type), we just extract whatever data is on it and merge
                # it into the vault under its actual doc type later when LangGraph has context.
                # For now in manual testing mode, we save it as 'back_of_card'.
                display_label = doc_type.replace("_", " ").title()
                
                # CACHE CHECK: Only for known, specific doc types (not back_of_card or unknown)
                skip_cache_types = {"back_of_card", "unknown"}
                if doc_type not in skip_cache_types:
                    cached_data = await asyncio.to_thread(db.check_vault_cache, user_id, doc_type)
                    if cached_data:
                        await processing_msg.edit(content=f"✅ **Secure Cache Hit!** I already have your verified **{display_label}** details stored in the PII Vault (expires in 24 hours). No need to re-scan!")
                        return

                await processing_msg.edit(content=f"🔍 *Identified as **{display_label}**. Running AI Vision extraction & validation...*")
                
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
    status_msg = await message.channel.send(" *Yojana Mitra is processing...*")

    try:
        # Wrap the state evaluation inside an isolated background thread context to prevent thread locking
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
                await message.channel.send(current_chunk)
        else:
            await message.channel.send(ai_response)

    except Exception as e:
        try:
            await status_msg.delete()
        except:
            pass
        await message.channel.send(" Operational database exception encountered handling text generation pipelines.")
        print(f"Runtime Exception Event: {e}")

bot.run(TOKEN)