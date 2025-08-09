import discord
from discord import app_commands
import os
import re
import httpx
import asyncio
import json
import time
import tempfile
from dotenv import load_dotenv
from xai_sdk import Client
from typing import AsyncGenerator, Optional

try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None  # OpenAI is optional depending on provider

# load env variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

# Provider/config
AI_PROVIDER = os.getenv("AI_PROVIDER", "xai").lower()  # "xai" or "openai"

# xAI
GROK_API_KEY = os.getenv("GROK_API_KEY")
MODEL = os.getenv("MODEL", "grok-3-mini")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")

# Shared behavior
PROMPT = os.getenv("PROMPT")
SEARCH_ENABLED = os.getenv("SEARCH_ENABLED", "false").lower() == "true"
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "true").lower() == "true"
WORD_CHUNK_SIZE = int(os.getenv("WORD_CHUNK_SIZE", "12"))
EDIT_COOLDOWN_SECONDS = float(os.getenv("EDIT_COOLDOWN_SECONDS", "1.5"))

# Image generation settings (OpenAI preferred)
IMAGE_GEN_ENABLED = os.getenv("IMAGE_GEN_ENABLED", "true").lower() == "true"
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "high")  # low | medium | high | auto
IMAGE_BACKGROUND = os.getenv("IMAGE_BACKGROUND", "auto")  # transparent | auto
IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "png")  # png | jpeg | webp

# Voice / TTS settings (OpenAI TTS)
VOICE_ENABLED = os.getenv("VOICE_ENABLED", "true").lower() == "true"
VOICE_ALLOWED_USER_ID = os.getenv("VOICE_ALLOWED_USER_ID", "374703513315442691")
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "alloy")

# Initialize xAI client for image generation
xai_client = Client(api_key=GROK_API_KEY) if GROK_API_KEY else None

# Initialize OpenAI client if available
openai_client = OpenAIClient(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAIClient) else None

def is_user_allowed_for_images(user_id: str) -> bool:
    """All users are allowed to use image generation now."""
    return True

def generate_image(prompt: str) -> dict:
    """Generate an image using OpenAI's GPT Image API (gpt-image-1)."""
    if not openai_client:
        return {"error": "OpenAI client not initialized - check your OPENAI_API_KEY"}

    try:
        result = openai_client.images.generate(
            model=OPENAI_IMAGE_MODEL,
            prompt=prompt,
            size=IMAGE_SIZE,
            quality=IMAGE_QUALITY,
            background=IMAGE_BACKGROUND,
        )

        if not result or not getattr(result, "data", None):
            return {"error": "No image returned by OpenAI"}

        image_base64 = result.data[0].b64_json
        file_suffix = ".png" if IMAGE_FORMAT.lower() == "png" else ".jpg" if IMAGE_FORMAT.lower() == "jpeg" else ".webp"
        with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
            tmp_path = tmp.name
        import base64
        with open(tmp_path, "wb") as f:
            f.write(base64.b64decode(image_base64))

        return {"file_path": tmp_path, "filename": os.path.basename(tmp_path), "revised_prompt": result.data[0].get("revised_prompt") if hasattr(result.data[0], "get") else None}
    except Exception as e:
        return {"error": f"Error generating image: {str(e)}"}

# --- AI Querying ---
async def query_xai_api_stream(context_messages: str, question: str) -> AsyncGenerator[str, None]:
    """
    Asynchronously queries the Grok API with streaming enabled and yields content chunks.
    """
    if not GROK_API_KEY:
        yield "error: grok_api_key is not configured."
        return

    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK_API_KEY}"
    }

    # define the messages sent to the api
    messages = [
        {
            "role": "system",
            "content": PROMPT
        },
        {
            "role": "user",
            "content": f"previous messages:\n\"\"{context_messages}\"\"\n\nuser query: \"{question}\""
        }
    ]

    # Prepare the payload
    payload = {
        "messages": messages,
        "model": MODEL,
        "stream": True,
    }

    # Add search parameters if enabled and "web" is mentioned in the question
    if SEARCH_ENABLED and "web" in question.lower():
        payload["search_parameters"] = {
            "mode": "on",
            "max_search_results": MAX_SEARCH_RESULTS,
            "sources": [
                {"type": "web", "safe_search": True},
                {"type": "news", "safe_search": True},
                {"type": "x"}
            ],
            "return_citations": True
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if chunk.get("choices") and chunk["choices"][0].get("delta"):
                                content = chunk["choices"][0]["delta"].get("content")
                                if content:
                                    yield content
                        except (json.JSONDecodeError, IndexError, KeyError):
                            print(f"Could not parse chunk from stream: {data_str}")
                            continue
    except httpx.RequestError as e:
        yield f"error communicating with grok api: {e}"
    except Exception as e:
        yield f"error communicating with grok api: {e}"


async def query_openai_api_stream(context_messages: str, question: str) -> AsyncGenerator[str, None]:
    """Asynchronously queries OpenAI Responses API with optional streaming."""
    if not openai_client:
        yield "error: openai client not initialized - check your OPENAI_API_KEY"
        return

    # Build input using Responses API format
    user_input = [
        {
            "role": "user",
            "content": f"previous messages:\n\"\"{context_messages}\"\"\n\nuser query: \"{question}\"",
        }
    ]

    # We will use streaming events
    try:
        stream = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=user_input,
            instructions=PROMPT or "You are a helpful assistant.",
            stream=True,
        )
        for event in stream:  # note: OpenAI SDK yields events synchronously
            # Bridge synchronous iterator into async generator
            await asyncio.sleep(0)  # yield to event loop
            try:
                if hasattr(event, "type") and event.type == "response.output_text.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        yield str(delta)
                elif hasattr(event, "type") and event.type == "response.completed":
                    break
            except Exception:
                continue
    except Exception as e:
        yield f"error communicating with openai api: {e}"


async def query_ai_api_stream(context_messages: str, question: str) -> AsyncGenerator[str, None]:
    """Dispatch to the configured provider."""
    provider = AI_PROVIDER
    if provider == "openai":
        async for chunk in query_openai_api_stream(context_messages, question):
            yield chunk
    else:
        async for chunk in query_xai_api_stream(context_messages, question):
            yield chunk

# setup the discord client
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True
intents.voice_states = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# Voice connection handle
voice_lock = asyncio.Lock()
voice_client: Optional[discord.VoiceClient] = None

# Ensure opus is loaded for voice
try:
    if not discord.opus.is_loaded():
        discord.opus.load_opus('libopus.so.0')
        print('Loaded Opus library libopus.so.0')
except Exception as e:
    print(f'Warning: could not load Opus library automatically: {e}')


async def synthesize_and_play(text: str):
    global voice_client
    if not VOICE_ENABLED:
        return
    if not (voice_client and voice_client.is_connected()):
        return
    if not openai_client:
        print("VOICE_ENABLED but OpenAI client not initialized; skipping TTS")
        return

    async with voice_lock:
        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        # Generate TTS via OpenAI (stream to file)
        try:
            with openai_client.audio.speech.with_streaming_response.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=text[:4000],  # basic limit for TTS prompt length
            ) as response:
                response.stream_to_file(tmp_path)
        except Exception as e:
            print(f"TTS error: {e}")
            return

        # Play in discord via ffmpeg
        if voice_client and voice_client.is_connected():
            if voice_client.is_playing():
                # wait for current to finish
                while voice_client.is_playing():
                    await asyncio.sleep(0.25)
            source = discord.FFmpegPCMAudio(tmp_path)
            voice_client.play(source)
            while voice_client.is_playing():
                await asyncio.sleep(0.25)

# on ready event
@client.event
async def on_ready():
    print(f'we have logged in as {client.user} (id: {client.user.id})')
    print(f"bot is ready and listening for mentions.")
    try:
        await tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print(f"Failed to sync commands: {e}")


@tree.command(name="join", description="Have the bot join your current voice channel (restricted)")
async def join(interaction: discord.Interaction):
    global voice_client
    if str(interaction.user.id) != VOICE_ALLOWED_USER_ID:
        await interaction.response.send_message("You are not allowed to use this command.", ephemeral=True)
        return

    if not isinstance(interaction.user, discord.Member):
        await interaction.response.send_message("Cannot resolve your voice state.", ephemeral=True)
        return

    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.response.send_message("You must be connected to a voice channel.", ephemeral=True)
        return

    channel = interaction.user.voice.channel
    try:
        if voice_client and voice_client.is_connected():
            if voice_client.channel.id == channel.id:
                await interaction.response.send_message(f"Already connected to {channel.name}.", ephemeral=True)
                return
            await voice_client.move_to(channel)
        else:
            voice_client = await channel.connect()
        await interaction.response.send_message(f"Joined {channel.name}.", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"Failed to join: {e}", ephemeral=True)


@tree.command(name="leave", description="Have the bot leave the current voice channel")
async def leave(interaction: discord.Interaction):
    global voice_client
    if not voice_client or not voice_client.is_connected():
        await interaction.response.send_message("I'm not connected to a voice channel.", ephemeral=True)
        return
    try:
        await voice_client.disconnect(force=True)
        voice_client = None
        await interaction.response.send_message("Left the voice channel.", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f"Failed to leave: {e}", ephemeral=True)

# on message event
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if not (client.user and client.user.mentioned_in(message)):
        return

    # bot is mentioned, now determine context
    context_for_grok = ""
    question_text = message.content

    if client.user:
        question_text = re.sub(rf'<@!?{client.user.id}>', '', question_text).strip()

    if not question_text:
        await message.reply(f"it looks like you mentioned me but didn't ask a question after it!")
        return

    # Check if this is an image generation request
    if IMAGE_GEN_ENABLED and "image" in question_text.lower():
        # Extract the image prompt (remove "image" keyword)
        image_prompt = re.sub(r'\bimage\b', '', question_text, flags=re.IGNORECASE).strip()
        
        if not image_prompt:
            await message.reply("Please provide a description for the image you want to generate. Example: `@bot image a cat sitting on a rainbow`")
            return
        
        await message.channel.typing()
        
        print(f"Generating image for user {message.author.name} (ID: {message.author.id}) with prompt: {image_prompt}")
        
        # Generate the image via OpenAI GPT Image
        result = generate_image(image_prompt)
        
        if "error" in result:
            await message.reply(f"❌ Error generating image: {result['error']}")
            return
        
        file_path = result.get("file_path")
        filename = result.get("filename", "image.png")
        revised_prompt = result.get("revised_prompt", "N/A") or "N/A"

        # Create an embed for the image
        embed = discord.Embed(
            title="🎨 Generated Image",
            description=f"**Original prompt:** {image_prompt}\n**Revised prompt:** {revised_prompt}",
            color=0x00ff00
        )
        embed.set_image(url=f"attachment://{filename}")
        embed.set_footer(text=f"Requested by {message.author.name}", icon_url=message.author.avatar.url if message.author.avatar else None)

        if file_path and os.path.exists(file_path):
            try:
                await message.reply(embed=embed, file=discord.File(file_path, filename=filename))
            finally:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
        else:
            await message.reply("❌ Could not access the generated image file.")
        return

    # Add the author's name to the question text
    question_text = f"{message.author.name} asks: {question_text}"

    if message.reference and message.reference.resolved:
        # if a reply, use the message that was replied to as the context
        replied_message = message.reference.resolved
        context_for_grok = f"message from {replied_message.author.name}: {replied_message.content}"
        print(f"context is a replied message: {replied_message.id}")
    else:
        # if not a reply, get the last 10 messages as context
        print(f"context is message history from channel: {message.channel.name}")
        history_messages = []
        # fetch last 10 messages
        async for historic_msg in message.channel.history(limit=10, before=message):
            history_messages.append(f"{historic_msg.author.name}: {historic_msg.content}")

        if history_messages:
            history_messages.reverse()
            context_for_grok = "\n".join(history_messages)
            print(f"fetched {len(history_messages)} messages for context.")
        else:
            context_for_grok = "no previous messages found in this channel to use as context."
            print("no message history found for context (or only bot's own message)." )

    print(f"context to be sent to grok api:\n{context_for_grok}")
    print(f"question asked: {question_text}")

    # --- Streaming Response Logic ---
    sent_message = None
    full_response = ""
    current_chunk = ""
    last_edit_time = 0

    try:
        # Send initial message
        sent_message = await message.reply("🧠 Thinking...")

        async for chunk in query_ai_api_stream(context_for_grok, question_text):
            full_response += chunk
            current_chunk += chunk

            # Edit message in word chunks to simulate streaming
            if len(current_chunk.split()) >= WORD_CHUNK_SIZE and (time.time() - last_edit_time) > EDIT_COOLDOWN_SECONDS:
                if len(full_response) > 1990: # Leave some buffer
                     await sent_message.edit(content=full_response[:1990] + "...")
                     break # Stop streaming if message is too long
                
                await sent_message.edit(content=full_response + "...")
                current_chunk = ""
                last_edit_time = time.time()
                
        # Final edit with the complete message
        if sent_message:
            final_content = full_response
            if len(final_content) > 2000:
                final_content = final_content[:2000]
            
            # If the message is empty (e.g., an error occurred early)
            if not final_content.strip():
                 final_content = "I am sorry, I encountered an error and could not provide a response."

            await sent_message.edit(content=final_content)
            # Also speak in voice if connected and OpenAI TTS available
            try:
                await synthesize_and_play(final_content)
            except Exception as e:
                print(f"Voice playback error: {e}")

    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        if sent_message:
            await sent_message.edit(content="An error occurred while generating the response.")

# main
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("error: discord_bot_token not found in .env file.")
    elif not PROMPT:
        print("error: PROMPT not found in .env file.")
    elif AI_PROVIDER == "openai" and not OPENAI_API_KEY:
        print("error: OPENAI_API_KEY not set while AI_PROVIDER=openai.")
    elif AI_PROVIDER == "xai" and not GROK_API_KEY:
        print("error: GROK_API_KEY not set while AI_PROVIDER=xai.")
    else:
        try:
            client.run(DISCORD_BOT_TOKEN)
        except discord.errors.LoginFailure:
            print("error: failed to log in. please check your discord_bot_token.")
        except Exception as e:
            print(f"an unexpected error occurred while trying to run the bot: {e}")
