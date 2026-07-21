import os
import sys
import requests
import discord
from discord.ext import commands
from dotenv import load_dotenv
from product_inference import db
import ollama
from sarvamai import SarvamAI
import json
import shutil
import asyncio
import re

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

# Initialize the database layout
db.init_db()

# Initialize the Sarvam AI SDK Client
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Ensure temporary media directory paths exist safely
os.makedirs("./temp_voice", exist_ok=True)

# ── HELPER: DYNAMIC LANGUAGE DETECTION ────────────────────────────────────────
def get_tts_language(text: str) -> str:
    """
    Scans the text for Devanagari (Hindi) characters. 
    Routes to Hindi TTS if found, otherwise defaults to Indian English.
    """
    if re.search(r'[\u0900-\u097F]', text):
        return "hi-IN"
    return "en-IN"

# ── WEEK 4: PRODUCTION SARVAM AI SPEECH-TO-TEXT ENGINE ────────────────────────
def execute_sarvam_stt(local_file_path: str, user_id: str) -> str:
    """
    Concurrency-safe transcription pipeline isolated by user_id.
    Parses the target JSON asset downloaded from Sarvam.
    """
    user_dir = f"./temp_voice/{user_id}"
    out_dir = os.path.join(user_dir, "output")
    
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    os.makedirs(out_dir, exist_ok=True)

    isolated_audio_filename = os.path.basename(local_file_path)
    isolated_audio_path = os.path.join(user_dir, isolated_audio_filename)
    shutil.move(local_file_path, isolated_audio_path)

    job = sarvam_client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="transcribe",
        language_code="unknown",
        with_diarization=False
    )
    
    job.upload_files(file_paths=[isolated_audio_path])
    job.start()
    job.wait_until_complete()
    job.download_outputs(output_dir=out_dir)
    
    for root, dirs, files in os.walk(out_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                print(f"📁 Found Sarvam JSON transcription asset at: {json_path}")
                
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                try:
                    shutil.rmtree(user_dir)
                except:
                    pass
                
                if isinstance(data, dict):
                    if "transcript" in data:
                        return data["transcript"].strip()
                    if "text" in data:
                        return data["text"].strip()
                    if "model_output" in data and "transcript" in data["model_output"]:
                        return data["model_output"]["transcript"].strip()
                    if "segments" in data:
                        return " ".join([seg.get("transcript", seg.get("text", "")) for seg in data["segments"]]).strip()
                
                raise KeyError(f"Could not find transcript text key inside Sarvam JSON schema.")
                
    try:
        shutil.rmtree(user_dir)
    except:
        pass
        
    raise FileNotFoundError("Sarvam STT data pipeline completed, but no json results were found.")

# ── WEEK 4: PRODUCTION SARVAM AI TEXT-TO-SPEECH STREAMER ──────────────────────
def execute_sarvam_tts(text_payload: str, output_path: str, lang_code: str):
    """
    Queries Bulbul V3 streaming transcription matrices over raw HTTP connection pools.
    Dynamically sets target language based on input text structure.
    """
    api_url = "https://api.sarvam.ai/text-to-speech/stream"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text_payload,
        "target_language_code": lang_code, 
        "speaker": "shubh",
        "model": "bulbul:v3",
        "pace": 1,
        "speech_sample_rate": 22050,
        "output_audio_codec": "mp3",
        "enable_preprocessing": True
    }
    
    with requests.post(api_url, headers=headers, json=payload, stream=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# ── WEEK 4: LLM LIGHTWEIGHT INTENT ROUTER ─────────────────────────────────────
def classify_user_intent(user_input: str) -> str:
    system_instruction = """
    You are a fast semantic classifier system router.
    Analyze the user input text and respond with EXACTLY one word:
    - 'CHIT_CHAT' if the user is saying hello, goodbye, thanking you, or asking casual conversational questions.
    - 'SCHEME_QUERY' if the user is asking about welfare schemes, scholarships, monetary problems, jobs, or financial assistance.
    Do not output any punctuation or other text.
    """
    try:
        response = ollama.chat(
            model='llama3.1',
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': user_input}
            ],
            options={'temperature': 0.0}
        )
        classification = response['message']['content'].strip()
        return "CHIT_CHAT" if "CHIT_CHAT" in classification else "SCHEME_QUERY"
    except:
        return "SCHEME_QUERY" 

@bot.event
async def on_ready():
    print(f'==========================================')
    print(f'🤖 Yojana Mitra Full Profiler Online')
    print(f'==========================================')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if not isinstance(message.channel, discord.DMChannel):
        await message.channel.send(f"Hi {message.author.mention}, please DM me directly to securely check eligibility!")
        return

    user_id = f"discord_{message.author.id}"
    username = message.author.name
    
    user_record = db.get_or_create_user(user_id, username)
    current_state = user_record['current_state']
    
    text = message.content.strip()
    is_voice_mode = False

    # ── AUDIO ATTACHMENT INTERCEPTOR (THREAD SAFE) ────────────────────────────
    if message.attachments:
        for attachment in message.attachments:
            if "audio" in str(attachment.content_type) or attachment.filename.endswith(('.ogg', '.mp3', '.wav', '.m4a')):
                is_voice_mode = True
                processing_msg = await message.channel.send("🎙️ *Voice message detected! Transcribing via Sarvam STT...*")
                try:
                    unique_filename = f"{user_id}_{attachment.filename}"
                    temp_audio_path = os.path.join("./temp_voice", unique_filename)
                    await attachment.save(temp_audio_path)
                    
                    # RUN IN BACKGROUND THREAD TO PREVENT DISCORD CRASH
                    transcribed_text = await asyncio.to_thread(execute_sarvam_stt, temp_audio_path, user_id)
                    
                    text = transcribed_text.strip()
                    await processing_msg.edit(content=f"📝 *Transcribed Content:* \"{text}\"")
                    
                except Exception as audio_err:
                    await processing_msg.edit(content="❌ Failed to process or transcribe the incoming audio stream.")
                    print(f"Audio Handshake Exception: {audio_err}")
                    return

    # --- FULL 13-STEP PROFILE STATE MACHINE ---
    if current_state == 'START':
        await message.channel.send(
            f"Hello {message.author.name}! Let's build your profile for Yojana Mitra. 🇮🇳\n\n"
            "**Step 1:** What is your **Gender**? (e.g., Male, Female, Other)"
        )
        db.update_user_state(user_id, 'AWAITING_GENDER')
        return

    elif current_state == 'AWAITING_GENDER':
        db.update_user_state(user_id, 'AWAITING_AGE', {'gender': text})
        await message.channel.send("**Step 2:** What is your **Age**? (Numbers only)")
        return

    elif current_state == 'AWAITING_AGE':
        if not text.isdigit():
            await message.channel.send("❌ Please enter a valid number for age:")
            return
        db.update_user_state(user_id, 'AWAITING_INCOME', {'age': int(text)})
        await message.channel.send("**Step 3:** What is your annual family **Income** in INR? (Numbers only, e.g., 45000)")
        return

    elif current_state == 'AWAITING_INCOME':
        if not text.isdigit():
            await message.channel.send("❌ Please enter a valid number for income:")
            return
        db.update_user_state(user_id, 'AWAITING_CASTE', {'income': int(text)})
        await message.channel.send("**Step 4:** What is your **Caste** category? (e.g., General, OBC, SC, ST)")
        return

    elif current_state == 'AWAITING_CASTE':
        db.update_user_state(user_id, 'AWAITING_RESIDENCE', {'caste': text})
        await message.channel.send("**Step 5:** What is your area of **Residence**? (Rural / Urban)")
        return

    elif current_state == 'AWAITING_RESIDENCE':
        db.update_user_state(user_id, 'AWAITING_MARITAL', {'residence': text})
        await message.channel.send("**Step 6:** What is your **Marital Status**? (e.g., Single, Married, Widowed, Divorced)")
        return

    elif current_state == 'AWAITING_MARITAL':
        db.update_user_state(user_id, 'AWAITING_DISABLED', {'marital_status': text})
        await message.channel.send("**Step 7:** Are you **Differently Abled**? (Yes / No)")
        return

    elif current_state == 'AWAITING_DISABLED':
        is_disabled = text.lower() in ['yes', 'y', 'true']
        if is_disabled:
            db.update_user_state(user_id, 'AWAITING_DISABILITY_PERC', {'differently_abled': True})
            await message.channel.send("**Step 7b:** What is your **Disability Percentage**? (Enter number, or type 'None')")
        else:
            db.update_user_state(user_id, 'AWAITING_MINORITY', {'differently_abled': False, 'disability_percentage': None})
            await message.channel.send("**Step 8:** Do you belong to a **Minority** community? (Yes / No)")
        return

    elif current_state == 'AWAITING_DISABILITY_PERC':
        perc = int(text) if text.isdigit() else None
        db.update_user_state(user_id, 'AWAITING_MINORITY', {'disability_percentage': perc})
        await message.channel.send("**Step 8:** Do you belong to a **Minority** community? (Yes / No)")
        return

    elif current_state == 'AWAITING_MINORITY':
        is_minority = text.lower() in ['yes', 'y', 'true']
        db.update_user_state(user_id, 'AWAITING_BPL', {'minority': is_minority})
        await message.channel.send("**Step 9:** Do you possess a **Below Poverty Line (BPL)** card? (Yes / No)")
        return

    elif current_state == 'AWAITING_BPL':
        is_bpl = text.lower() in ['yes', 'y', 'true']
        db.update_user_state(user_id, 'AWAITING_DISTRESS', {'below_poverty_line': is_bpl})
        await message.channel.send("**Step 10:** Are you facing **Economic Distress**? (Yes / No)")
        return

    elif current_state == 'AWAITING_DISTRESS':
        is_distress = text.lower() in ['yes', 'y', 'true']
        db.update_user_state(user_id, 'AWAITING_GOVT_EMP', {'economic_distress': is_distress})
        await message.channel.send("**Step 11:** Are you a **Government Employee**? (Yes / No)")
        return

    elif current_state == 'AWAITING_GOVT_EMP':
        is_govt = text.lower() in ['yes', 'y', 'true']
        db.update_user_state(user_id, 'AWAITING_OCCUPATION', {'government_employee': is_govt})
        await message.channel.send("**Step 12:** What is your primary **Occupation**? (e.g., Farmer, Student, Artisan, Unemployed)")
        return

    elif current_state == 'AWAITING_OCCUPATION':
        db.update_user_state(user_id, 'PROFILE_COMPLETE', {'occupation': text})
        
        updated_user = db.get_or_create_user(user_id, username)
        d = updated_user['profile_data']
        
        summary = (
            "🎉 **Yojana Mitra Profile Created Successfully!**\n"
            "The following structure is safely synced to PostgreSQL:\n\n"
            f"• **Gender:** {d.get('gender')}\n"
            f"• **Age:** {d.get('age')} years\n"
            f"• **Income:** ₹{d.get('income'):,}\n"
            f"• **Caste:** {d.get('caste')}\n"
            f"• **Residence:** {d.get('residence')}\n"
            f"• **Marital Status:** {d.get('marital_status')}\n"
            f"• **Differently Abled:** {d.get('differently_abled')} (Perc: {d.get('disability_percentage')}%)\n"
            f"• **Minority:** {d.get('minority')}\n"
            f"• **BPL Status:** {d.get('below_poverty_line')}\n"
            f"• **Economic Distress:** {d.get('economic_distress')}\n"
            f"• **Govt Employee:** {d.get('government_employee')}\n"
            f"• **Occupation:** {d.get('occupation')}\n\n"
            "You are all set. Type any question now to search matching schemes!"
        )
        await message.channel.send(summary)
        return

    # --- ACTIVE CONVERSATION RUNTIME ---
    elif current_state == 'PROFILE_COMPLETE':
        status_msg = await message.channel.send("🤖 *Yojana Mitra is analyzing the input stream...*")
        profile_data = user_record['profile_data']
        
        try:
            # RUN CLASSIFIER IN THREAD
            intent = await asyncio.to_thread(classify_user_intent, text)
            
            if intent == 'CHIT_CHAT':
                await status_msg.edit(content="💬 *Thinking...*")
                
                # Dynamic Prompt: Reply in the same language the user spoke
                chat_prompt = "You are Yojana Mitra, a helpful AI assistant. Reply briefly, politely, and naturally. You MUST reply in the exact same language the user used (e.g., if they spoke English, use English. If they spoke Hindi/Hinglish, use Hindi/Hinglish)."
                response = await asyncio.to_thread(
                    ollama.chat,
                    model='llama3.1',
                    messages=[
                        {'role': 'system', 'content': chat_prompt},
                        {'role': 'user', 'content': text}
                    ]
                )
                ai_response = response['message']['content']
            else:
                await status_msg.edit(content="🔍 *Searching government scheme matrices...*")
                from core_inference import hybrid_rag
                
                # RUN RAG PIPELINE IN BACKGROUND THREAD
                ai_response = await asyncio.to_thread(hybrid_rag.run_yojana_pipeline, profile_data, text)
            
            await status_msg.delete()
            
            if is_voice_mode:
                voice_status = await message.channel.send("🗣️ *Synthesizing voice response audio streams via Sarvam Bulbul...*")
                try:
                    out_audio_path = f"./temp_voice/response_{user_id}.mp3"
                    
                    # DYNAMICALLY DETECT LANGUAGE FOR TTS
                    target_lang = get_tts_language(ai_response)
                    
                    # RUN TTS STREAM IN BACKGROUND THREAD
                    await asyncio.to_thread(execute_sarvam_tts, ai_response, out_audio_path, target_lang)
                    
                    await voice_status.delete()
                    await message.channel.send(file=discord.File(out_audio_path))
                    
                    if os.path.exists(out_audio_path):
                        os.remove(out_audio_path)
                except Exception as tts_err:
                    await voice_status.edit(content="⚠️ Voice output synthesis faulted. Falling back to clean text delivery alternative.")
                    print(f"TTS Engine Fault Event: {tts_err}")
                    await message.channel.send(ai_response)
            else:
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
            await message.channel.send("❌ Operational exception encountered handling production runtime inference layers.")
            print(f"Runtime Exception Event: {e}")
            
        return

bot.run(TOKEN)