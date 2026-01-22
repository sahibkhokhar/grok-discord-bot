import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
from typing import AsyncGenerator, Final

import discord
import httpx
from discord import app_commands
from dotenv import load_dotenv

try:
    from openai import OpenAI as OpenAIClient
except Exception:
    OpenAIClient = None

# Ensure stdout is line-buffered for Docker logs
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("discord-bot")

# load env variables
load_dotenv()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%s; using %s", name, value, default)
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%s; using %s", name, value, default)
        return default


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
SEARCH_ENABLED = env_bool("SEARCH_ENABLED", False)
MAX_SEARCH_RESULTS = env_int("MAX_SEARCH_RESULTS", 5)
SHOW_SOURCES = env_bool("SHOW_SOURCES", True)
WORD_CHUNK_SIZE = max(1, env_int("WORD_CHUNK_SIZE", 12))
EDIT_COOLDOWN_SECONDS = max(0.1, env_float("EDIT_COOLDOWN_SECONDS", 1.5))
MESSAGE_HISTORY_LIMIT = max(1, env_int("MESSAGE_HISTORY_LIMIT", 30))
DISCORD_MESSAGE_LIMIT: Final[int] = 2000
DISCORD_MESSAGE_EDIT_LIMIT: Final[int] = 1990

# Blocked users (comma-separated user IDs)
BLOCKED_USER_IDS = set()
blocked_users_str = os.getenv("BLOCKED_USER_IDS", "")
if blocked_users_str.strip():
    try:
        BLOCKED_USER_IDS = set(int(uid.strip()) for uid in blocked_users_str.split(",") if uid.strip())
    except ValueError:
        print("Warning: Invalid user IDs in BLOCKED_USER_IDS, ignoring.")

# Image generation settings (OpenAI preferred)
IMAGE_GEN_ENABLED = env_bool("IMAGE_GEN_ENABLED", True)
OPENAI_IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
IMAGE_SIZE = os.getenv("IMAGE_SIZE", "1024x1024")
IMAGE_QUALITY = os.getenv("IMAGE_QUALITY", "high")  # low | medium | high | auto
IMAGE_BACKGROUND = os.getenv("IMAGE_BACKGROUND", "auto")  # transparent | auto
IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "png")  # png | jpeg | webp

# Initialize OpenAI client if available
openai_client = OpenAIClient(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAIClient) else None

def build_system_prompt() -> str:
    base_prompt = PROMPT or "You are a helpful assistant."
    return (
        f"{base_prompt}\n\n"
        "You are in a Discord conversation. Respond naturally and directly to the user's question. "
        "Do not repeat the message format or echo usernames in your response."
    )

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

        image_base64 = getattr(result.data[0], "b64_json", None)
        if not image_base64:
            return {"error": "No image payload returned by OpenAI"}
        file_suffix = ".png" if IMAGE_FORMAT.lower() == "png" else ".jpg" if IMAGE_FORMAT.lower() == "jpeg" else ".webp"
        with tempfile.NamedTemporaryFile(suffix=file_suffix, delete=False) as tmp:
            tmp_path = tmp.name
        import base64
        with open(tmp_path, "wb") as f:
            f.write(base64.b64decode(image_base64))

        revised_prompt = getattr(result.data[0], "revised_prompt", None)
        return {
            "file_path": tmp_path,
            "filename": os.path.basename(tmp_path),
            "revised_prompt": revised_prompt,
        }
    except Exception as e:
        logger.exception("Error generating image")
        return {"error": f"Error generating image: {e}"}

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
        "Authorization": f"Bearer {GROK_API_KEY}",
    }

    # define the messages sent to the api
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": f"{context_messages}\n\n{question}"
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
                            logger.warning("Could not parse chunk from stream: %s", data_str)
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
            "content": f"{context_messages}\n\n{question}",
        }
    ]

    # We will use streaming events
    try:
        stream = openai_client.responses.create(
            model=OPENAI_MODEL,
            input=user_input,
            instructions=build_system_prompt(),
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

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# on ready event
@client.event
async def on_ready():
    logger.info("Logged in as %s (id: %s)", client.user, client.user.id)
    logger.info("Bot is ready and listening for mentions.")
    logger.info("AI_PROVIDER=%s, IMAGE_GEN=%s", AI_PROVIDER, IMAGE_GEN_ENABLED)
    if BLOCKED_USER_IDS:
        logger.info("Blocked users: %s user(s)", len(BLOCKED_USER_IDS))
    
    # Print all guilds the bot is in
    logger.info("%s", "=" * 60)
    logger.info("CONNECTED TO %s SERVER(S):", len(client.guilds))
    logger.info("%s", "=" * 60)
    for guild in client.guilds:
        owner_info = f"{guild.owner.name} (ID: {guild.owner_id})" if guild.owner else f"ID: {guild.owner_id}"
        logger.info("Server: %s", guild.name)
        logger.info("  ID: %s", guild.id)
        logger.info("  Owner: %s", owner_info)
        logger.info("  Members: %s", guild.member_count)
        logger.info("  Created: %s", guild.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
        if guild.me and guild.me.joined_at:
            logger.info("  Bot joined: %s", guild.me.joined_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
    logger.info("%s", "=" * 60)
    
    try:
        await tree.sync()
        logger.info("Slash commands synced.")
    except Exception as e:
        logger.exception("Failed to sync commands")

# on message event
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Check if user is blocked
    if message.author.id in BLOCKED_USER_IDS:
        logger.info("Blocked user %s (ID: %s) tried to use the bot", message.author.name, message.author.id)
        return

    if not (client.user and client.user.mentioned_in(message)):
        return

    # bot is mentioned, now determine context
    context_for_grok = ""
    question_text = message.content

    if client.user:
        question_text = re.sub(rf'<@!?{client.user.id}>', '', question_text).strip()

    if not question_text:
        await message.reply("It looks like you mentioned me but didn't ask a question after it.")
        return

    # Check if this is an image generation request
    if IMAGE_GEN_ENABLED and "image" in question_text.lower():
        # Extract the image prompt (remove "image" keyword)
        image_prompt = re.sub(r'\bimage\b', '', question_text, flags=re.IGNORECASE).strip()
        
        if not image_prompt:
            await message.reply("Please provide a description for the image you want to generate. Example: `@bot image a cat sitting on a rainbow`")
            return
        
        logger.info(
            "Generating image for user %s (ID: %s) with prompt: %s",
            message.author.name,
            message.author.id,
            image_prompt,
        )
        
        # Generate the image via OpenAI GPT Image
        async with message.channel.typing():
            result = await asyncio.to_thread(generate_image, image_prompt)
        
        if "error" in result:
            await message.reply(f"❌ Error generating image: {result['error']}")
            return
        
        file_path = result.get("file_path")
        filename = result.get("filename", "image.png")
        revised_prompt = result.get("revised_prompt") or "N/A"

        # Create an embed for the image
        embed = discord.Embed(
            title="🎨 Generated Image",
            description=f"**Original prompt:** {image_prompt}\n**Revised prompt:** {revised_prompt}",
            color=0x00ff00
        )
        embed.set_image(url=f"attachment://{filename}")
        embed.set_footer(
            text=f"Requested by {message.author.name}",
            icon_url=message.author.avatar.url if message.author.avatar else None,
        )

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
        context_for_grok = f"{replied_message.author.name}: {replied_message.content}"
        logger.info("Context is a replied message: %s", replied_message.id)
    else:
        # if not a reply, get recent messages as context
        logger.info("Context is message history from channel: %s", message.channel.name)
        history_messages = []
        # fetch recent messages
        async for historic_msg in message.channel.history(limit=MESSAGE_HISTORY_LIMIT, before=message):
            history_messages.append(f"{historic_msg.author.name}: {historic_msg.content}")

        if history_messages:
            history_messages.reverse()
            context_for_grok = "\n".join(history_messages)
            logger.info("Fetched %s messages for context.", len(history_messages))
        else:
            context_for_grok = "[This is the start of the conversation]"
            logger.info("No message history found for context (or only bot's own message).")

    logger.debug("Context to be sent to api:\n%s", context_for_grok)
    logger.info("Question asked: %s", question_text)

    # --- Streaming Response Logic ---
    sent_message = None
    full_response = ""
    current_chunk = ""
    last_edit_time = 0.0

    try:
        # Send initial message
        sent_message = await message.reply("🧠 Thinking...")

        async for chunk in query_ai_api_stream(context_for_grok, question_text):
            full_response += chunk
            current_chunk += chunk

            # Edit message in word chunks to simulate streaming
            if len(current_chunk.split()) >= WORD_CHUNK_SIZE and (time.monotonic() - last_edit_time) > EDIT_COOLDOWN_SECONDS:
                if len(full_response) > DISCORD_MESSAGE_EDIT_LIMIT:  # Leave some buffer
                    await sent_message.edit(content=full_response[:DISCORD_MESSAGE_EDIT_LIMIT] + "...")
                     break # Stop streaming if message is too long
                
                await sent_message.edit(content=full_response + "...")
                current_chunk = ""
                last_edit_time = time.monotonic()
                
        # Final edit with the complete message
        if sent_message:
            final_content = full_response
            if len(final_content) > DISCORD_MESSAGE_LIMIT:
                final_content = final_content[:DISCORD_MESSAGE_LIMIT]
            
            # If the message is empty (e.g., an error occurred early)
            if not final_content.strip():
                final_content = "I am sorry, I encountered an error and could not provide a response."

            await sent_message.edit(content=final_content)

    except Exception as e:
        logger.exception("An error occurred during streaming")
        if sent_message:
            await sent_message.edit(content="An error occurred while generating the response.")

# main
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        logger.error("DISCORD_BOT_TOKEN not found in .env file.")
    elif not PROMPT:
        logger.error("PROMPT not found in .env file.")
    elif AI_PROVIDER == "openai" and not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY not set while AI_PROVIDER=openai.")
    elif AI_PROVIDER == "xai" and not GROK_API_KEY:
        logger.error("GROK_API_KEY not set while AI_PROVIDER=xai.")
    else:
        try:
            client.run(DISCORD_BOT_TOKEN)
        except discord.errors.LoginFailure:
            logger.error("Failed to log in. Please check your DISCORD_BOT_TOKEN.")
        except Exception as e:
            logger.exception("An unexpected error occurred while trying to run the bot")