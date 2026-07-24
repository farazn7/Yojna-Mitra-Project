import os
import asyncio
import signal
import discord
from discord.ext import commands
from dotenv import load_dotenv
import threading
import urllib.request
import time

def start_heartbeat():
    def pinger():
        while True:
            try:
                urllib.request.urlopen("https://hc-ping.com/c38c6455-a777-4a12-b524-8b8dec9ef57d", timeout=10)
            except Exception:
                pass
            time.sleep(60)
    threading.Thread(target=pinger, daemon=True).start()

# Start pinging in the background immediately
start_heartbeat()

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
    await bot.tree.sync()
    print(f'==========================================')
    print(f' Yojana Mitra Powered by LangGraph Active')
    print(f'==========================================')

def _do_reset(user_id: str):
    """Shared reset logic (sync): wipes profile, vault, and checkpointer state from DB."""
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

# ---------------------------------------------------------
# FORM-BASED PROFILE UI  (all 13 fields across 3 steps)
# Step 1: Modal popup   — name, age, income, caste, occupation
# Step 2: ProfileView1  — residence, marital status, disability, minority
# Step 3: ProfileView2  — gender, bpl, economic distress, govt employee → Save
# ---------------------------------------------------------

class ProfileView2(discord.ui.View):
    """Step 3/3: BPL, economic distress, government employee → Save"""

    def __init__(self, user_id: str, profile: dict):
        super().__init__(timeout=300)
        self.user_id = user_id
        self.profile = profile

    @discord.ui.select(
        placeholder="👤 Gender",
        options=[
            discord.SelectOption(label="👨 Male", value="Male"),
            discord.SelectOption(label="👩 Female", value="Female"),
            discord.SelectOption(label="🌈 Other", value="Other"),
        ]
    )
    async def sel_gender(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["gender"] = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="🏠 Below Poverty Line (BPL)?",
        options=[
            discord.SelectOption(label="✅ Yes — I have a BPL card", value="true"),
            discord.SelectOption(label="❌ No", value="false"),
        ]
    )
    async def sel_bpl(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["below_poverty_line"] = (select.values[0] == "true")
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="📉 Facing Economic Distress?",
        options=[
            discord.SelectOption(label="✅ Yes", value="true"),
            discord.SelectOption(label="❌ No", value="false"),
        ]
    )
    async def sel_distress(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["economic_distress"] = (select.values[0] == "true")
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="🏗 Government Employee?",
        options=[
            discord.SelectOption(label="✅ Yes", value="true"),
            discord.SelectOption(label="❌ No", value="false"),
        ]
    )
    async def sel_govt(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["government_employee"] = (select.values[0] == "true")
        await interaction.response.defer()

    @discord.ui.button(label="✅ Save Profile", style=discord.ButtonStyle.success, row=3)
    async def save_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        db.update_user_state(self.user_id, "PROFILE_COMPLETE", self.profile)
        for child in self.children:
            child.disabled = True
        g = self.profile.get
        summary = (
            "🎉 **Profile Saved Successfully!**\n\n"
            f"• **Name:** {g('name', '—')}\n"
            f"• **Gender:** {g('gender', '—')}\n"
            f"• **Age:** {g('age', '—')} yrs\n"
            f"• **Income:** ₹{int(g('income', 0)):,}\n"
            f"• **Caste:** {g('caste', '—')}\n"
            f"• **Occupation:** {g('occupation', '—')}\n"
            f"• **Residence:** {g('residence', '—')}\n"
            f"• **Marital Status:** {g('marital_status', '—')}\n"
            f"• **Differently Abled:** {'Yes' if g('differently_abled') else 'No'}\n"
            f"• **Minority:** {'Yes' if g('minority') else 'No'}\n"
            f"• **BPL Card:** {'Yes' if g('below_poverty_line') else 'No'}\n"
            f"• **Economic Distress:** {'Yes' if g('economic_distress') else 'No'}\n"
            f"• **Govt Employee:** {'Yes' if g('government_employee') else 'No'}\n\n"
            "You're all set! Ask me to **find schemes** for you now! 🎯"
        )
        await interaction.response.edit_message(content=summary, view=self)


class ProfileView1(discord.ui.View):
    """Step 2/3: Residence, marital status, disability, minority → Next"""

    def __init__(self, user_id: str, profile: dict):
        super().__init__(timeout=300)
        self.user_id = user_id
        self.profile = profile

    @discord.ui.select(
        placeholder="🏘 Residence Area",
        options=[
            discord.SelectOption(label="🌾 Rural", value="Rural"),
            discord.SelectOption(label="🏙 Urban", value="Urban"),
        ]
    )
    async def sel_residence(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["residence"] = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="💍 Marital Status",
        options=[
            discord.SelectOption(label="Single", value="Single"),
            discord.SelectOption(label="Married", value="Married"),
            discord.SelectOption(label="Widowed", value="Widowed"),
            discord.SelectOption(label="Divorced", value="Divorced"),
        ]
    )
    async def sel_marital(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["marital_status"] = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="♿ Differently Abled?",
        options=[
            discord.SelectOption(label="✅ Yes", value="true"),
            discord.SelectOption(label="❌ No", value="false"),
        ]
    )
    async def sel_abled(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["differently_abled"] = (select.values[0] == "true")
        self.profile["disability_percentage"] = 40 if self.profile["differently_abled"] else 0
        await interaction.response.defer()

    @discord.ui.select(
        placeholder="🕌 Minority Community?",
        options=[
            discord.SelectOption(label="✅ Yes", value="true"),
            discord.SelectOption(label="❌ No", value="false"),
        ]
    )
    async def sel_minority(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["minority"] = (select.values[0] == "true")
        await interaction.response.defer()

    @discord.ui.button(label="Next →", style=discord.ButtonStyle.primary, row=4)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        view2 = ProfileView2(self.user_id, self.profile)
        await interaction.response.edit_message(
            content="**Step 3/3** — Almost done! Answer these last questions:",
            view=view2
        )


class ProfileModal(discord.ui.Modal):
    def __init__(self, user_id: str, **kwargs):
        super().__init__(title="Yojana Mitra Profile (1/3)", **kwargs)
        self.user_id = user_id
        self.add_item(discord.ui.TextInput(label="Full Name", style=discord.TextStyle.short))
        self.add_item(discord.ui.TextInput(label="Age (years, numbers only)", style=discord.TextStyle.short))
        self.add_item(discord.ui.TextInput(label="Annual Family Income in ₹ (numbers only)", style=discord.TextStyle.short))
        self.add_item(discord.ui.TextInput(label="Caste (General / OBC / SC / ST)", style=discord.TextStyle.short))
        self.add_item(discord.ui.TextInput(label="Occupation (e.g. Farmer, Student)", style=discord.TextStyle.short))

    async def on_submit(self, interaction: discord.Interaction):
        age_str = self.children[1].value.strip()
        income_str = self.children[2].value.strip().replace(",", "").replace("\u20b9", "")

        if not age_str.isdigit():
            await interaction.response.send_message("⚠️ Age must be a number (e.g. 25). Please run /profile again.", ephemeral=True)
            return
        if not income_str.isdigit():
            await interaction.response.send_message("⚠️ Income must be a number (e.g. 45000). Please run /profile again.", ephemeral=True)
            return

        profile = {
            "name": self.children[0].value.strip(),
            "age": int(age_str),
            "income": int(income_str),
            "caste": self.children[3].value.strip(),
            "occupation": self.children[4].value.strip(),
            # Defaults filled in step 2 & 3:
            "gender": "Unknown",
            "differently_abled": False, "disability_percentage": 0,
            "minority": False, "below_poverty_line": False,
            "economic_distress": False, "government_employee": False,
            "residence": "Urban", "marital_status": "Single",
        }

        view1 = ProfileView1(self.user_id, profile)
        await interaction.response.send_message(
            "**Step 2/3** — Select your personal details below:",
            view=view1,
            ephemeral=True
        )

@bot.tree.command(name="profile", description="Fill out your Yojana Mitra profile to get personalized schemes.")
async def profile_slash(interaction: discord.Interaction):
    user_id = f"discord_{interaction.user.id}"
    modal = ProfileModal(user_id=user_id)
    await interaction.response.send_modal(modal)
# ---------------------------------------------------------

@bot.tree.command(name="reset", description="Wipe your profile and start fresh.")
async def reset_slash(interaction: discord.Interaction):
    user_id = f"discord_{interaction.user.id}"
    task = _active_tasks.pop(user_id, None)
    if task and not task.done():
        task.cancel()
    try:
        await asyncio.to_thread(_do_reset, user_id)
        await interaction.response.send_message("🔄 **Profile Reset Complete.**\n\nAll your data has been wiped. Send me any message to start fresh!", ephemeral=True)
    except Exception as e:
        print(f"[Reset Error] {e}")
        await interaction.response.send_message("⚠️ Reset encountered an error. Please try again.", ephemeral=True)

@bot.tree.command(name="stop", description="Stop the currently running task.")
async def stop_slash(interaction: discord.Interaction):
    user_id = f"discord_{interaction.user.id}"
    task = _active_tasks.get(user_id)
    if task and not task.done():
        _active_tasks.pop(user_id, None)
        task.cancel()
        await interaction.response.send_message("🛑 **Stopped.** Send a new message to continue.", ephemeral=True)
    else:
        await interaction.response.send_message("Nothing is currently running.", ephemeral=True)


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