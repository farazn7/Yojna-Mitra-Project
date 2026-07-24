"""
bot_audio.py — Optional audio-enabled runner for BOTH Discord and Telegram.

Run this INSTEAD of bot.py + bot_telegram.py when you want Sarvam voice support.
When you don't want to spend Sarvam credits, just run the regular bots.

Usage:
    python -m product_inference.bot_audio
"""

import os
import asyncio
import threading
import time
import urllib.request
import shutil
import json
import re
import requests

import discord
from discord.ext import commands

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

from dotenv import load_dotenv
from sarvamai import SarvamAI

import product_inference.db as db
from core_inference.graph import graph_app

# ── ENV SETUP ─────────────────────────────────────────────────────────────────
load_dotenv()
DISCORD_TOKEN     = os.getenv('DISCORD_TOKEN')
TELEGRAM_TOKEN    = os.getenv('TELEGRAM_TOKEN')
SARVAM_API_KEY    = os.getenv('SARVAM_API_KEY')

db.init_db()
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
os.makedirs("./temp_voice", exist_ok=True)

# Per-user task trackers for /stop
_discord_tasks:  dict[str, asyncio.Task] = {}
_telegram_tasks: dict[str, asyncio.Task] = {}

# ── HEARTBEATS ────────────────────────────────────────────────────────────────
def _start_heartbeats():
    """Ping Healthchecks.io every 60 s for both bots."""
    URLS = [
        "https://hc-ping.com/6f985ce4-0f9d-40ab-9b0f-d3dd68cd2957",   # Telegram
        "https://hc-ping.com/c38c6455-a777-4a12-b524-8b8dec9ef57d",   # Discord
    ]
    def _ping():
        while True:
            for url in URLS:
                try:
                    urllib.request.urlopen(url, timeout=10)
                except Exception:
                    pass
            time.sleep(60)
    threading.Thread(target=_ping, daemon=True).start()

# ── SARVAM HELPERS ────────────────────────────────────────────────────────────
def get_tts_language(text: str) -> str:
    if re.search(r'[\u0900-\u097F]', text):
        return "hi-IN"
    return "en-IN"


def execute_sarvam_stt(local_file_path: str, user_id: str) -> str:
    """Transcribe audio to text using Sarvam Saaras STT."""
    user_dir = f"./temp_voice/{user_id}"
    out_dir  = os.path.join(user_dir, "output")

    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    os.makedirs(out_dir, exist_ok=True)

    isolated = os.path.join(user_dir, os.path.basename(local_file_path))
    shutil.move(local_file_path, isolated)

    job = sarvam_client.speech_to_text_job.create_job(
        model="saaras:v3", mode="transcribe",
        language_code="unknown", with_diarization=False
    )
    job.upload_files(file_paths=[isolated])
    job.start()
    job.wait_until_complete()
    job.download_outputs(output_dir=out_dir)

    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith('.json'):
                with open(os.path.join(root, f), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                shutil.rmtree(user_dir, ignore_errors=True)
                if isinstance(data, dict):
                    for key in ("transcript", "text"):
                        if key in data:
                            return data[key].strip()
                    if "model_output" in data and "transcript" in data["model_output"]:
                        return data["model_output"]["transcript"].strip()
                    if "segments" in data:
                        return " ".join(s.get("transcript", s.get("text", "")) for s in data["segments"]).strip()
                raise KeyError("Could not find transcript key in Sarvam response.")

    shutil.rmtree(user_dir, ignore_errors=True)
    raise FileNotFoundError("Sarvam STT completed but produced no JSON output.")


def execute_sarvam_tts(text: str, output_path: str, lang_code: str):
    """Stream TTS audio from Sarvam Bulbul to a local file."""
    resp = requests.post(
        "https://api.sarvam.ai/text-to-speech/stream",
        headers={"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"},
        json={
            "text": text, "target_language_code": lang_code,
            "speaker": "shubh", "model": "bulbul:v3",
            "pace": 1, "speech_sample_rate": 22050,
            "output_audio_codec": "mp3", "enable_preprocessing": True
        },
        stream=True
    )
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


# ── SHARED: INVOKE GRAPH AND OPTIONALLY REPLY WITH VOICE ──────────────────────
async def _run_and_reply_discord(text_input: str, user_id: str, message: discord.Message,
                                  status_msg: discord.Message, is_voice: bool):
    try:
        config = {"configurable": {"thread_id": user_id}}
        graph_output = await asyncio.to_thread(
            graph_app.invoke,
            {"messages": [{"role": "user", "content": text_input}], "user_id": user_id},
            config=config
        )
        ai_response = graph_output.get("response", "I encountered a processing error. Please retry.")
        if not ai_response or not ai_response.strip():
            ai_response = "I processed your request but the response was empty. Please try again."
        await status_msg.delete()

        if is_voice:
            voice_msg = await message.channel.send("Synthesizing voice response...")
            try:
                out_path = f"./temp_voice/response_{user_id}.mp3"
                lang = get_tts_language(ai_response)
                await asyncio.to_thread(execute_sarvam_tts, ai_response, out_path, lang)
                await voice_msg.delete()
                await message.channel.send(file=discord.File(out_path))
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception as tts_err:
                await voice_msg.edit(content="Voice synthesis failed, sending text instead.")
                print(f"[TTS Error] {tts_err}")
                await message.channel.send(ai_response)
        else:
            # Safe chunking for Discord 2000-char limit
            if len(ai_response) > 2000:
                for line in ai_response.split('\n'):
                    if line:
                        await message.channel.send(line[:2000])
            else:
                await message.channel.send(ai_response)

    except asyncio.CancelledError:
        try:
            await status_msg.delete()
        except Exception:
            pass
    except Exception as e:
        try:
            await status_msg.delete()
        except Exception:
            pass
        await message.channel.send("An error occurred. Please try again.")
        print(f"[Discord Audio Bot Error] {e}")
    finally:
        _discord_tasks.pop(user_id, None)


async def _run_and_reply_telegram(text_input: str, user_id: str, update: Update,
                                   context: ContextTypes.DEFAULT_TYPE,
                                   status_msg, is_voice: bool):
    try:
        config = {"configurable": {"thread_id": user_id}}
        graph_output = await asyncio.to_thread(
            graph_app.invoke,
            {"messages": [{"role": "user", "content": text_input}], "user_id": user_id},
            config=config
        )
        ai_response = graph_output.get("response", "I encountered a processing error. Please retry.")
        if not ai_response or not ai_response.strip():
            ai_response = "I processed your request but the response was empty. Please try again."

        if is_voice:
            await status_msg.edit_text("Synthesizing voice response...")
            try:
                out_path = f"./temp_voice/response_{user_id}.mp3"
                lang = get_tts_language(ai_response)
                await asyncio.to_thread(execute_sarvam_tts, ai_response, out_path, lang)
                await status_msg.delete()
                with open(out_path, "rb") as af:
                    await context.bot.send_audio(chat_id=update.effective_chat.id, audio=af)
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception as tts_err:
                await status_msg.edit_text("Voice synthesis failed, sending text instead.")
                print(f"[TTS Error] {tts_err}")
                await update.message.reply_text(ai_response)
        else:
            # Safe chunking for Telegram 4096-char limit
            if len(ai_response) > 4000:
                lines = ai_response.split('\n')
                chunk = ""
                for line in lines:
                    if len(chunk) + len(line) + 1 > 4000:
                        await status_msg.edit_text(chunk)
                        status_msg = await update.message.reply_text("...")
                        chunk = line + '\n'
                    else:
                        chunk += line + '\n'
                if chunk.strip():
                    await status_msg.edit_text(chunk)
            else:
                await status_msg.edit_text(ai_response)

    except asyncio.CancelledError:
        try:
            await status_msg.edit_text("Request cancelled.")
        except Exception:
            pass
    except Exception as e:
        try:
            await status_msg.edit_text("An error occurred. Please try again.")
        except Exception:
            pass
        print(f"[Telegram Audio Bot Error] {e}")
    finally:
        _telegram_tasks.pop(user_id, None)


# ── DISCORD BOT ───────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True
discord_bot = commands.Bot(command_prefix='!', intents=intents)

class ProfileView2(discord.ui.View):
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

    @discord.ui.select(placeholder="🏠 Below Poverty Line (BPL)?", options=[discord.SelectOption(label="✅ Yes — I have a BPL card", value="true"), discord.SelectOption(label="❌ No", value="false")])
    async def sel_bpl(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["below_poverty_line"] = (select.values[0] == "true")
        await interaction.response.defer()

    @discord.ui.select(placeholder="📉 Facing Economic Distress?", options=[discord.SelectOption(label="✅ Yes", value="true"), discord.SelectOption(label="❌ No", value="false")])
    async def sel_distress(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["economic_distress"] = (select.values[0] == "true")
        await interaction.response.defer()

    @discord.ui.select(placeholder="🏗 Government Employee?", options=[discord.SelectOption(label="✅ Yes", value="true"), discord.SelectOption(label="❌ No", value="false")])
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
            f"• **Name:** {g('name', '—')}\n• **Gender:** {g('gender', '—')}\n• **Age:** {g('age', '—')} yrs\n"
            f"• **Income:** ₹{int(g('income', 0)):,}\n• **Caste:** {g('caste', '—')}\n"
            f"• **Occupation:** {g('occupation', '—')}\n• **Residence:** {g('residence', '—')}\n"
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
    def __init__(self, user_id: str, profile: dict):
        super().__init__(timeout=300)
        self.user_id = user_id
        self.profile = profile

    @discord.ui.select(placeholder="🏘 Residence Area", options=[discord.SelectOption(label="🌾 Rural", value="Rural"), discord.SelectOption(label="🏙 Urban", value="Urban")])
    async def sel_residence(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["residence"] = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(placeholder="💍 Marital Status", options=[discord.SelectOption(label="Single", value="Single"), discord.SelectOption(label="Married", value="Married"), discord.SelectOption(label="Widowed", value="Widowed"), discord.SelectOption(label="Divorced", value="Divorced")])
    async def sel_marital(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["marital_status"] = select.values[0]
        await interaction.response.defer()

    @discord.ui.select(placeholder="♿ Differently Abled?", options=[discord.SelectOption(label="✅ Yes", value="true"), discord.SelectOption(label="❌ No", value="false")])
    async def sel_abled(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["differently_abled"] = (select.values[0] == "true")
        self.profile["disability_percentage"] = 40 if self.profile["differently_abled"] else 0
        await interaction.response.defer()

    @discord.ui.select(placeholder="🕌 Minority Community?", options=[discord.SelectOption(label="✅ Yes", value="true"), discord.SelectOption(label="❌ No", value="false")])
    async def sel_minority(self, interaction: discord.Interaction, select: discord.ui.Select):
        self.profile["minority"] = (select.values[0] == "true")
        await interaction.response.defer()

    @discord.ui.button(label="Next →", style=discord.ButtonStyle.primary, row=4)
    async def next_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        for child in self.children:
            child.disabled = True
        view2 = ProfileView2(self.user_id, self.profile)
        await interaction.response.edit_message(content="**Step 3/3** — Almost done! Answer these last questions:", view=view2)


class ProfileModal(discord.ui.Modal):
    def __init__(self, user_id: str, **kwargs):
        super().__init__(title="Yojana Mitra Profile (1/3)", **kwargs)
        self.user_id = user_id
        self.add_item(discord.ui.TextInput(label="Full Name", style=discord.TextStyle.short))
        self.add_item(discord.ui.TextInput(label="Age (years, numbers only)", style=discord.TextStyle.short))
        self.add_item(discord.ui.TextInput(label="Annual Family Income in ₹ (numbers only)", style=discord.TextStyle.short))
        self.add_item(discord.ui.TextInput(label="Caste (General/OBC/SC/ST)", style=discord.TextStyle.short))
        self.add_item(discord.ui.TextInput(label="Occupation (e.g. Farmer, Student)", style=discord.TextStyle.short))

    async def on_submit(self, interaction: discord.Interaction):
        age_str = self.children[1].value.strip()
        income_str = self.children[2].value.strip().replace(",", "").replace("\u20b9", "")

        if not age_str.isdigit() or not income_str.isdigit():
            await interaction.response.send_message("⚠️ Age and Income must be numbers. Please run /profile again.", ephemeral=True)
            return

        profile = {
            "name": self.children[0].value.strip(),
            "age": int(age_str),
            "income": int(income_str),
            "caste": self.children[3].value.strip(),
            "occupation": self.children[4].value.strip(),
            "gender": "Unknown",
            "differently_abled": False, "disability_percentage": 0,
            "minority": False, "below_poverty_line": False,
            "economic_distress": False, "government_employee": False,
            "residence": "Urban", "marital_status": "Single",
        }
        view1 = ProfileView1(self.user_id, profile)
        await interaction.response.send_message("**Step 2/3** — Select your personal details below:", view=view1, ephemeral=True)


@discord_bot.tree.command(name="profile", description="Fill out your Yojana Mitra profile to get personalized schemes.")
async def profile_slash(interaction: discord.Interaction):
    user_id = f"discord_{interaction.user.id}"
    modal = ProfileModal(user_id=user_id)
    await interaction.response.send_modal(modal)


@discord_bot.event
async def on_ready():
    await discord_bot.tree.sync()
    print("==========================================")
    print(" Yojana Mitra Discord (Audio Mode) Active")
    print("==========================================")

@discord_bot.event
async def on_message(message: discord.Message):
    if message.author == discord_bot.user:
        return
    if not isinstance(message.channel, discord.DMChannel):
        await message.channel.send(f"Hi {message.author.mention}, please DM me to get started!")
        return

    user_id = f"discord_{message.author.id}"
    text_input = message.content.strip()
    is_voice = False

    # ── Audio attachment → STT ─────────────────────────────────────────────
    if message.attachments:
        for att in message.attachments:
            if "audio" in str(att.content_type) or att.filename.endswith(('.ogg', '.mp3', '.wav', '.m4a')):
                is_voice = True
                processing_msg = await message.channel.send("Voice message received, transcribing...")
                try:
                    tmp_path = f"./temp_voice/{user_id}_{att.filename}"
                    await att.save(tmp_path)
                    text_input = await asyncio.to_thread(execute_sarvam_stt, tmp_path, user_id)
                    await processing_msg.edit(content=f'Transcribed: "{text_input}"')
                except Exception as e:
                    await processing_msg.edit(content="Failed to transcribe audio. Please try again.")
                    print(f"[Discord STT Error] {e}")
                    return
                break

    if not text_input:
        return

    # /reset and /stop
    if text_input.lower() in ("/reset", "!reset", "reset"):
        old = _discord_tasks.pop(user_id, None)
        if old and not old.done():
            old.cancel()
        try:
            import psycopg2
            from product_inference.db import DB_PARAMS
            conn = psycopg2.connect(**DB_PARAMS)
            cur = conn.cursor()
            for tbl, col in [("user_profiles","platform_id"),("pii_vault","user_id"),
                              ("checkpoints","thread_id"),("checkpoint_writes","thread_id")]:
                cur.execute(f"DELETE FROM {tbl} WHERE {col} = %s;", (user_id,))
            conn.commit(); cur.close(); conn.close()
            await message.channel.send("Profile reset complete. Send a message to start fresh!")
        except Exception as e:
            print(f"[Reset Error] {e}")
            await message.channel.send("Reset encountered an error. Please try again.")
        return

    if text_input.lower() in ("/stop", "!stop", "stop"):
        task = _discord_tasks.get(user_id)
        if task and not task.done():
            _discord_tasks.pop(user_id, None)
            task.cancel()
            await message.channel.send("Stopped. Send a new message to continue.")
        return

    # Cancel any old task, start new one
    old = _discord_tasks.pop(user_id, None)
    if old and not old.done():
        old.cancel()

    status_msg = await message.channel.send("Yojana Mitra is processing...")
    _discord_tasks[user_id] = asyncio.create_task(
        _run_and_reply_discord(text_input, user_id, message, status_msg, is_voice)
    )


# ── TELEGRAM BOT ──────────────────────────────────────────────────────────────

_profile_sessions: dict[str, dict] = {}

FORM_FLOW = [
    ("name",                "1/13", "👤 *Step 1/13* — What is your *full name*?", None),
    ("gender",              "2/13", "👤 *Step 2/13* — What is your *gender*?", [[("👨 Male", "Male"), ("👩 Female", "Female"), ("🌈 Other", "Other")]]),
    ("residence",           "3/13", "🏘 *Step 3/13* — Where do you *live*?", [[("🌾 Rural", "Rural"), ("🏙 Urban", "Urban")]]),
    ("caste",               "4/13", "📋 *Step 4/13* — What is your *caste category*?", [[("General", "General"), ("OBC", "OBC")], [("SC", "SC"), ("ST", "ST")]]),
    ("marital_status",      "5/13", "💍 *Step 5/13* — What is your *marital status*?", [[("Single", "Single"), ("Married", "Married")], [("Widowed", "Widowed"), ("Divorced", "Divorced")]]),
    ("occupation",          "6/13", "💼 *Step 6/13* — What is your *occupation*?", [[("🌾 Farmer", "Farmer"), ("📚 Student", "Student")], [("💼 Salaried", "Salaried"), ("🏪 Business Owner", "Business Owner")], [("🧵 Artisan", "Artisan"), ("❌ Unemployed", "Unemployed")], [("✏️ Type your own...", "__text__")]]),
    ("differently_abled",   "7/13", "♿ *Step 7/13* — Are you *differently abled*?", [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("minority",            "8/13", "🕌 *Step 8/13* — Do you belong to a *minority community*?", [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("below_poverty_line",  "9/13", "🏠 *Step 9/13* — Do you have a *BPL card* (Below Poverty Line)?", [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("economic_distress",   "10/13", "📉 *Step 10/13* — Are you facing *economic distress*?", [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("government_employee", "11/13", "🏗 *Step 11/13* — Are you a *government employee*?", [[("✅ Yes", "true"), ("❌ No", "false")]]),
    ("age",                 "12/13", "🔢 *Step 12/13* — Please *type your age* in years:", None),
    ("income",              "13/13", "💰 *Step 13/13* — Please type your annual family *income in ₹* (e.g. 45000):", None),
]

def _build_keyboard(rows: list) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=f"pf|{value}") for label, value in row] for row in rows])

async def _send_form_step(user_id: str, step_idx: int, reply_fn):
    if step_idx >= len(FORM_FLOW):
        session = _profile_sessions.pop(user_id, {})
        data = session.get("data", {})
        try: db.update_user_state(user_id, "PROFILE_COMPLETE", data)
        except Exception as e: print(f"[Profile Save Error] {e}")
        g = data.get
        summary = (
            "🎉 *Profile Saved Successfully!*\n\n"
            f"• *Name:* {g('name', '—')}\n• *Gender:* {g('gender', '—')}\n• *Age:* {g('age', '—')} yrs\n"
            f"• *Income:* ₹{int(g('income', 0)):,}\n• *Caste:* {g('caste', '—')}\n"
            f"• *Occupation:* {g('occupation', '—')}\n• *Residence:* {g('residence', '—')}\n"
            f"• *Marital Status:* {g('marital_status', '—')}\n"
            f"• *Differently Abled:* {'Yes' if g('differently_abled') else 'No'}\n"
            f"• *Minority:* {'Yes' if g('minority') else 'No'}\n"
            f"• *BPL Card:* {'Yes' if g('below_poverty_line') else 'No'}\n"
            f"• *Economic Distress:* {'Yes' if g('economic_distress') else 'No'}\n"
            f"• *Govt Employee:* {'Yes' if g('government_employee') else 'No'}\n\n"
            "You're all set! Ask me to *find schemes* for you now! 🎯"
        )
        await reply_fn(summary, parse_mode="Markdown")
        return

    field, _label, question, keyboard_rows = FORM_FLOW[step_idx]
    _profile_sessions[user_id]["current_step"] = step_idx
    if keyboard_rows:
        await reply_fn(question, reply_markup=_build_keyboard(keyboard_rows), parse_mode="Markdown")
    else:
        _profile_sessions[user_id]["awaiting_text"] = field
        await reply_fn(question, parse_mode="Markdown")

async def tg_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = f"telegram_{update.effective_user.id}"
    text_input = update.message.text or ""
    is_voice = False

    # ── Voice / Audio attachment → STT ─────────────────────────────────────
    voice_obj = update.message.voice or update.message.audio
    if voice_obj:
        is_voice = True
        processing_msg = await update.message.reply_text("Voice message received, transcribing...")
        try:
            tg_file = await context.bot.get_file(voice_obj.file_id)
            ext = ".ogg" if update.message.voice else ".mp3"
            tmp_path = f"./temp_voice/{user_id}_voice{ext}"
            await tg_file.download_to_drive(tmp_path)
            text_input = await asyncio.to_thread(execute_sarvam_stt, tmp_path, user_id)
            await processing_msg.edit_text(f'Transcribed: "{text_input}"')
        except Exception as e:
            await processing_msg.edit_text("Failed to transcribe audio. Please try again.")
            print(f"[Telegram STT Error] {e}")
            return

    if not text_input:
        return

    # /stop via text
    if text_input.strip().lower() in ("/stop", "stop"):
        task = _telegram_tasks.get(user_id)
        if task and not task.done():
            _telegram_tasks.pop(user_id, None)
            task.cancel()
            await update.message.reply_text("Stopped. Send a new message to continue.")
        return

    if user_id in _profile_sessions:
        session = _profile_sessions[user_id]
        awaiting = session.get("awaiting_text")
        if awaiting:
            val = text_input.strip()
            if awaiting == "age":
                val = val.replace(" ", "")
                if not val.isdigit():
                    await update.message.reply_text("❌ Please enter a valid number for age:")
                    return
                session["data"]["age"] = int(val)
            elif awaiting == "income":
                val = val.replace(",", "").replace("₹", "").replace(" ", "")
                if not val.isdigit():
                    await update.message.reply_text("❌ Please enter a valid number for income:")
                    return
                session["data"]["income"] = int(val)
            elif awaiting == "occupation":
                session["data"]["occupation"] = val.title()
            elif awaiting == "name":
                session["data"]["name"] = val.title()
            else:
                session["data"][awaiting] = val
            session["awaiting_text"] = None
            await _send_form_step(user_id, session["current_step"] + 1, update.message.reply_text)
            return
        else:
            await update.message.reply_text("📋 Profile form in progress! Please use the buttons.")
            return

    old = _telegram_tasks.pop(user_id, None)
    if old and not old.done():
        old.cancel()

    status_msg = await update.message.reply_text("Yojana Mitra is processing...")
    _telegram_tasks[user_id] = asyncio.create_task(
        _run_and_reply_telegram(text_input, user_id, update, context, status_msg, is_voice)
    )


async def tg_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = f"telegram_{update.effective_user.id}"
    old = _telegram_tasks.pop(user_id, None)
    if old and not old.done():
        old.cancel()
    try:
        import psycopg2
        from product_inference.db import DB_PARAMS
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        for tbl, col in [("user_profiles","platform_id"),("pii_vault","user_id"),
                          ("checkpoints","thread_id"),("checkpoint_writes","thread_id")]:
            cur.execute(f"DELETE FROM {tbl} WHERE {col} = %s;", (user_id,))
        conn.commit(); cur.close(); conn.close()
        await update.message.reply_text("Profile reset complete. Send a message to start fresh!")
    except Exception as e:
        print(f"[Reset Error] {e}")
        await update.message.reply_text("Reset encountered an error. Please try again.")

async def tg_profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = f"telegram_{update.effective_user.id}"
    _profile_sessions[user_id] = {
        "current_step": 0, "awaiting_text": None,
        "data": {"differently_abled": False, "disability_percentage": 0, "minority": False,
                 "below_poverty_line": False, "economic_distress": False, "government_employee": False}
    }
    await update.message.reply_text("📋 Setting up your Yojana Mitra Profile\n\nPlease answer the questions below by tapping the buttons.")
    await _send_form_step(user_id, 0, update.message.reply_text)

async def tg_profile_form_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = f"telegram_{query.from_user.id}"
    data = query.data
    if not data.startswith("pf|"): return
    val = data[3:]
    if user_id not in _profile_sessions:
        await query.edit_message_text("⚠️ Your profile session expired. Please run /profile again.")
        return
    session = _profile_sessions[user_id]
    step_idx = session.get("current_step", 0)
    field, _label, question, _kb = FORM_FLOW[step_idx]

    if val == "__text__":
        session["awaiting_text"] = field
        await query.edit_message_text("✏️ Please *type your occupation* and send it as a message:", parse_mode="Markdown")
        return

    if val in ("true", "false"):
        session["data"][field] = (val == "true")
        if field == "differently_abled":
            session["data"]["disability_percentage"] = 40 if session["data"][field] else 0
        disp = "Yes" if val == "true" else "No"
    else:
        session["data"][field] = val
        disp = val

    clean_q = question.replace("*", "").replace("\\", "")
    await query.edit_message_text(f"✅ {clean_q}\n*Selected:* {disp}", parse_mode="Markdown")
    await _send_form_step(user_id, step_idx + 1, query.message.reply_text)


async def tg_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = f"telegram_{update.effective_user.id}"
    task = _telegram_tasks.get(user_id)
    if task and not task.done():
        _telegram_tasks.pop(user_id, None)
        task.cancel()
        await update.message.reply_text("Stopped. Send a new message to continue.")


# ── MAIN: RUN BOTH BOTS CONCURRENTLY ─────────────────────────────────────────
async def run_telegram():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("reset", tg_reset_command))
    app.add_handler(CommandHandler("stop",  tg_stop_command))
    app.add_handler(CommandHandler("profile", tg_profile_command))
    app.add_handler(CallbackQueryHandler(tg_profile_form_callback, pattern=r"^pf\|"))
    app.add_handler(MessageHandler(
        filters.TEXT | filters.VOICE | filters.AUDIO | filters.PHOTO | filters.Document.ALL,
        tg_handle_message
    ))
    async with app:
        await app.start()
        print("==========================================")
        print(" Yojana Mitra Telegram (Audio Mode) Active")
        print("==========================================")
        await app.updater.start_polling()
        # Keep running until cancelled
        await asyncio.Event().wait()
        await app.updater.stop()
        await app.stop()


async def main():
    _start_heartbeats()
    await asyncio.gather(
        discord_bot.start(DISCORD_TOKEN),
        run_telegram()
    )


if __name__ == '__main__':
    if not DISCORD_TOKEN:
        print("[Error] DISCORD_TOKEN not found in .env")
    elif not TELEGRAM_TOKEN:
        print("[Error] TELEGRAM_TOKEN not found in .env")
    elif not SARVAM_API_KEY:
        print("[Error] SARVAM_API_KEY not found in .env")
    else:
        asyncio.run(main())