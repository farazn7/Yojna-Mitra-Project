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