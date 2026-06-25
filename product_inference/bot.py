import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
import db
from core_inference import hybrid_rag

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

db.init_db()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ── WEEK 4: SARVAM AI SPEECH-TO-TEXT GATEWAY ──────────────────────────────────
def transcrib_voice_payload(audio_bytes) -> str:
    """
    Placeholder for your Week 4 Sarvam Saaras STT implementation.
    Passes raw audio bytes directly into Sarvam endpoints.
    """
    # TODO: Connect your Sarvam AI Client initialization here
    # response = sarvam_client.speech_to_text(audio_bytes, language_code="hi-IN")
    # return response.text
    
    # Mocking returning text string for test query verification:
    return "I just finished school and need help paying for college fees. Are there any scholarships?"

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

    user_id = str(message.author.id)
    username = message.author.name
    
    user_record = db.get_or_create_user(user_id, username)
    current_state = user_record['current_state']
    
    # Base input initialization
    text = message.content.strip()

    # ── AUDIO ATTACHMENT INTERCEPTOR ──────────────────────────────────────────
    if message.attachments:
        for attachment in message.attachments:
            # Detect native Discord voice notes or explicit audio uploads
            if "audio" in str(attachment.content_type) or attachment.filename.endswith(('.ogg', '.mp3', '.wav', '.m4a')):
                processing_msg = await message.channel.send("🎙️ *Voice message detected! Transcribing via Sarvam STT...*")
                try:
                    # 1. Download raw audio bytes straight from Discord CDN
                    audio_bytes = await attachment.read()
                    
                    # 2. Pass bytes through Sarvam Pipeline
                    transcribed_text = transcrib_voice_payload(audio_bytes)
                    
                    # 3. Seamlessly mutate input parameters to flow directly down into runtime states
                    text = transcribed_text.strip()
                    await processing_msg.edit(content=f"📝 *Transcribed:* \"{text}\"")
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
        status_msg = await message.channel.send("🤖 *Yojana Mitra is analyzing your profile against the active scheme matrix...*")
        
        profile_data = user_record['profile_data']
        
        try:
            from core_inference import hybrid_rag
            ai_response = hybrid_rag.run_yojana_pipeline(profile_data, text)
            
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
            await message.channel.send("❌ Operational database exception encountered handling RAG generation pipelines.")
            print(f"Runtime Exception Event: {e}")
            
        return

bot.run(TOKEN)