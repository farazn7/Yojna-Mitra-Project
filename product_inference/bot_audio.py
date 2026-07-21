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

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

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

@discord_bot.event
async def on_ready():
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