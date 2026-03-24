import asyncio
import json
import logging
import os
import random
import re
import sys
import tempfile
import time
from typing import Any, AsyncGenerator, Final

import discord
import httpx
from discord import app_commands
from dotenv import load_dotenv

from local_tools import (
    build_chat_completion_tools,
    build_responses_api_tools,
    dispatch_local_function,
)

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

# Random chat behavior
RANDOM_CHAT_ENABLED = env_bool("RANDOM_CHAT_ENABLED", False)
RANDOM_CHAT_INTERVAL_MIN_MINUTES = max(1, env_int("RANDOM_CHAT_INTERVAL_MIN_MINUTES", 20))
RANDOM_CHAT_INTERVAL_MAX_MINUTES = max(1, env_int("RANDOM_CHAT_INTERVAL_MAX_MINUTES", 40))
# Ensure min is not greater than max
if RANDOM_CHAT_INTERVAL_MIN_MINUTES > RANDOM_CHAT_INTERVAL_MAX_MINUTES:
    logger.warning(
        "RANDOM_CHAT_INTERVAL_MIN_MINUTES (%s) > RANDOM_CHAT_INTERVAL_MAX_MINUTES (%s), swapping values",
        RANDOM_CHAT_INTERVAL_MIN_MINUTES,
        RANDOM_CHAT_INTERVAL_MAX_MINUTES
    )
    RANDOM_CHAT_INTERVAL_MIN_MINUTES, RANDOM_CHAT_INTERVAL_MAX_MINUTES = RANDOM_CHAT_INTERVAL_MAX_MINUTES, RANDOM_CHAT_INTERVAL_MIN_MINUTES
RANDOM_CHAT_CHANCE = env_float("RANDOM_CHAT_CHANCE", 0.25)
RANDOM_CHAT_RECENT_SECONDS = max(30, env_int("RANDOM_CHAT_RECENT_SECONDS", 300))
RANDOM_CHAT_CHANNEL_IDS_RAW = os.getenv("RANDOM_CHAT_CHANNEL_IDS", "")
RANDOM_CHAT_CHANNEL_IDS = []
if RANDOM_CHAT_CHANNEL_IDS_RAW.strip():
    for raw_id in RANDOM_CHAT_CHANNEL_IDS_RAW.split(","):
        try:
            channel_id = int(raw_id.strip())
            RANDOM_CHAT_CHANNEL_IDS.append(channel_id)
        except ValueError:
            logger.warning("Invalid channel id in RANDOM_CHAT_CHANNEL_IDS: %s", raw_id)

if RANDOM_CHAT_CHANCE > 1:
    RANDOM_CHAT_CHANCE = min(1.0, RANDOM_CHAT_CHANCE / 100.0)
RANDOM_CHAT_CHANCE = max(0.0, min(1.0, RANDOM_CHAT_CHANCE))

# Blocked users (comma-separated user IDs)
BLOCKED_USER_IDS = set()
blocked_users_str = os.getenv("BLOCKED_USER_IDS", "")
if blocked_users_str.strip():
    try:
        BLOCKED_USER_IDS = set(int(uid.strip()) for uid in blocked_users_str.split(",") if uid.strip())
    except ValueError:
        logger.warning("Invalid user IDs in BLOCKED_USER_IDS, ignoring.")

# Image generation settings (xAI Imagine)
IMAGE_GEN_ENABLED = env_bool("IMAGE_GEN_ENABLED", True)
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "grok-imagine-image")
IMAGE_ASPECT_RATIO = os.getenv("IMAGE_ASPECT_RATIO", "auto")

# Per-user image rate limiting
IMAGE_RATE_LIMIT_PER_DAY = env_int("IMAGE_RATE_LIMIT_PER_DAY", 2)
IMAGE_UNLIMITED_USER_IDS: set[int] = set()
_img_unlimited_raw = os.getenv("IMAGE_UNLIMITED_USER_IDS", "")
if _img_unlimited_raw.strip():
    try:
        IMAGE_UNLIMITED_USER_IDS = set(
            int(uid.strip()) for uid in _img_unlimited_raw.split(",") if uid.strip()
        )
    except ValueError:
        logger.warning("Invalid user IDs in IMAGE_UNLIMITED_USER_IDS, ignoring.")

# Local tools (time, dice, calculator, text, TTS)
LOCAL_TOOLS_ENABLED = env_bool("LOCAL_TOOLS_ENABLED", True)
TTS_ENABLED = env_bool("TTS_ENABLED", True)
LOCAL_TOOLS_MAX_ROUNDS = max(1, min(32, env_int("LOCAL_TOOLS_MAX_ROUNDS", 8)))

# Initialize OpenAI client if available
openai_client = OpenAIClient(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAIClient) else None

# --- Per-user image rate limiting (in-memory) ---
from collections import defaultdict
from datetime import datetime, timedelta, timezone as tz

_image_usage: dict[int, list[datetime]] = defaultdict(list)


def check_image_rate_limit(user_id: int) -> tuple[bool, int]:
    """
    Returns (allowed, remaining).
    Prunes entries older than 24 h, then checks the count.
    """
    if user_id in IMAGE_UNLIMITED_USER_IDS:
        return True, -1
    if IMAGE_RATE_LIMIT_PER_DAY <= 0:
        return True, -1

    now = datetime.now(tz.utc)
    cutoff = now - timedelta(days=1)
    entries = [t for t in _image_usage[user_id] if t > cutoff]
    _image_usage[user_id] = entries

    remaining = max(0, IMAGE_RATE_LIMIT_PER_DAY - len(entries))
    return len(entries) < IMAGE_RATE_LIMIT_PER_DAY, remaining


def record_image_use(user_id: int) -> None:
    _image_usage[user_id].append(datetime.now(tz.utc))

def build_system_prompt() -> str:
    base_prompt = PROMPT or "You are a helpful assistant."
    prompt = (
        f"{base_prompt}\n\n"
        "You are in a Discord conversation. Respond naturally and directly to the user's question. "
        "Do not repeat the message format or echo usernames in your response."
    )
    if LOCAL_TOOLS_ENABLED:
        prompt += (
            "\n\nYou have function tools for local time/date, dice, calculator, text transforms"
            + (" and short text-to-speech." if TTS_ENABLED else ".")
            + " Call a tool when you need real values from the bot (time, dice, math, etc.). "
            "You will receive the tool results in the same turn before you finish—use them in your answer. "
            "Do not use bracket tags like [TIME]…[/TIME]; use the functions instead. "
            "Do not call tools unless they help the user's request."
        )
    return prompt


def build_random_chat_prompt() -> str:
    return (
        "You are a normal participant in a Discord chat. "
        "Write a single, natural message that blends in with the recent conversation. "
        "Match the tone, slang, and style of the participants. "
        "Do not mention being an AI, bot, or assistant. "
        "Do not explain yourself. Keep it short and conversational."
    )

def generate_image(prompt: str) -> dict:
    """Generate an image using xAI's Imagine API (grok-imagine-image)."""
    if not GROK_API_KEY:
        return {"error": "GROK_API_KEY not configured — required for image generation."}

    url = "https://api.x.ai/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK_API_KEY}",
    }
    body: dict[str, Any] = {
        "model": IMAGE_MODEL,
        "prompt": prompt,
        "response_format": "b64_json",
    }
    if IMAGE_ASPECT_RATIO and IMAGE_ASPECT_RATIO != "auto":
        body["aspect_ratio"] = IMAGE_ASPECT_RATIO

    try:
        with httpx.Client(timeout=120.0) as http:
            resp = http.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        images = data.get("data") or []
        if not images:
            return {"error": "No image returned by xAI"}

        b64 = images[0].get("b64_json")
        image_url = images[0].get("url")

        if b64:
            import base64 as b64mod
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            with open(tmp_path, "wb") as f:
                f.write(b64mod.b64decode(b64))
            return {"file_path": tmp_path, "filename": os.path.basename(tmp_path)}

        if image_url:
            with httpx.Client(timeout=60.0) as http:
                img_resp = http.get(image_url)
                img_resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_resp.content)
                tmp_path = tmp.name
            return {"file_path": tmp_path, "filename": os.path.basename(tmp_path)}

        return {"error": "No image data in xAI response."}
    except httpx.HTTPStatusError as e:
        logger.exception("xAI image API HTTP error")
        return {"error": f"xAI image API error: {e.response.status_code}"}
    except Exception as e:
        logger.exception("Error generating image")
        return {"error": f"Error generating image: {e}"}

# --- AI Querying ---
async def query_xai_api_stream(
    context_messages: str,
    question: str,
    system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
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
            "content": system_prompt or build_system_prompt(),
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


async def query_openai_api_stream(
    context_messages: str,
    question: str,
    system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
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
            instructions=system_prompt or build_system_prompt(),
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


async def query_ai_api_stream(
    context_messages: str,
    question: str,
    system_prompt: str | None = None,
) -> AsyncGenerator[str, None]:
    """Dispatch to the configured provider."""
    provider = AI_PROVIDER
    if provider == "openai":
        async for chunk in query_openai_api_stream(context_messages, question, system_prompt):
            yield chunk
    else:
        async for chunk in query_xai_api_stream(context_messages, question, system_prompt):
            yield chunk


def _openai_response_final_text(resp: Any) -> str:
    """Best-effort final user-visible text from a Responses API result."""
    t = (getattr(resp, "output_text", None) or "").strip()
    if t:
        return t
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "output_text":
                        tx = (part.get("text") or "").strip()
                        if tx:
                            return tx
                else:
                    tx = (getattr(part, "text", None) or "").strip()
                    if tx:
                        return tx
    return ""


def run_openai_agent_loop(
    context_messages: str,
    question: str,
    system_prompt: str,
) -> tuple[str, str | None]:
    """Responses API tool loop; model sees real tool outputs before the final reply."""
    if not openai_client:
        return "error: openai client not initialized - check your OPENAI_API_KEY", None

    tools = build_responses_api_tools(TTS_ENABLED)
    input_list: list[Any] = [
        {
            "role": "user",
            "content": f"{context_messages}\n\n{question}",
        }
    ]
    tts_path: str | None = None

    for round_i in range(LOCAL_TOOLS_MAX_ROUNDS):
        try:
            resp = openai_client.responses.create(
                model=OPENAI_MODEL,
                instructions=system_prompt,
                input=input_list,
                tools=tools,
                parallel_tool_calls=True,
                stream=False,
            )
        except Exception as e:
            logger.exception("OpenAI agent request failed")
            return f"error communicating with openai api: {e}", tts_path

        output = list(getattr(resp, "output", []) or [])
        input_list = list(input_list)
        input_list.extend(output)

        calls = [x for x in output if getattr(x, "type", None) == "function_call"]
        if not calls:
            text = _openai_response_final_text(resp)
            return (text if text else "(no response)"), tts_path

        logger.info("OpenAI agent tool round %s: %s call(s)", round_i + 1, len(calls))
        for item in calls:
            name = getattr(item, "name", "") or ""
            args = getattr(item, "arguments", None) or "{}"
            call_id = getattr(item, "call_id", None)
            out, tp = dispatch_local_function(name, args, tts_enabled=TTS_ENABLED)
            if tp:
                tts_path = tp
            input_list.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": out,
                }
            )

    return "error: too many tool rounds; try a simpler question.", tts_path


def run_xai_agent_loop(
    context_messages: str,
    question: str,
    system_prompt: str,
) -> tuple[str, str | None]:
    """xAI chat completions tool loop (OpenAI-compatible messages + tools)."""
    if not GROK_API_KEY:
        return "error: grok_api_key is not configured.", None

    tools = build_chat_completion_tools(TTS_ENABLED)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context_messages}\n\n{question}"},
    ]
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROK_API_KEY}",
    }
    tts_path: str | None = None

    for round_i in range(LOCAL_TOOLS_MAX_ROUNDS):
        payload: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
        }
        if SEARCH_ENABLED and "web" in question.lower():
            payload["search_parameters"] = {
                "mode": "on",
                "max_search_results": MAX_SEARCH_RESULTS,
                "sources": [
                    {"type": "web", "safe_search": True},
                    {"type": "news", "safe_search": True},
                    {"type": "x"},
                ],
                "return_citations": True,
            }

        try:
            with httpx.Client(timeout=120.0) as http:
                response = http.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            logger.exception("xAI agent HTTP error")
            return f"error communicating with grok api: {e}", tts_path
        except Exception as e:
            logger.exception("xAI agent request failed")
            return f"error communicating with grok api: {e}", tts_path

        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return "error: unexpected response from grok api.", tts_path

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            logger.info("xAI agent tool round %s: %s call(s)", round_i + 1, len(tool_calls))
            messages.append(msg)
            for tc in tool_calls:
                tid = tc.get("id") or ""
                fn = (tc.get("function") or {}).get("name") or ""
                args = (tc.get("function") or {}).get("arguments") or "{}"
                out, tp = dispatch_local_function(fn, args, tts_enabled=TTS_ENABLED)
                if tp:
                    tts_path = tp
                messages.append(
                    {"role": "tool", "tool_call_id": tid, "content": out}
                )
            continue

        content = (msg.get("content") or "").strip()
        return (content if content else "(no response)"), tts_path

    return "error: too many tool rounds; try a simpler question.", tts_path


def run_provider_agent_loop(
    provider: str,
    context_messages: str,
    question: str,
    system_prompt: str,
) -> tuple[str, str | None]:
    if provider == "openai":
        return run_openai_agent_loop(context_messages, question, system_prompt)
    return run_xai_agent_loop(context_messages, question, system_prompt)


async def generate_ai_response(context_messages: str, prompt: str, system_prompt: str | None = None) -> str:
    full_response = ""
    async for chunk in query_ai_api_stream(context_messages, prompt, system_prompt):
        full_response += chunk
    return full_response.strip()

# setup the discord client
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
random_chat_task: asyncio.Task | None = None

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

    global random_chat_task
    if random_chat_task is None:
        random_chat_task = asyncio.create_task(random_chat_loop())
        logger.info("Random chat loop started.")

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
        image_prompt = re.sub(r'\bimage\b', '', question_text, flags=re.IGNORECASE).strip()

        if not image_prompt:
            await message.reply(
                "Please provide a description for the image you want to generate. "
                "Example: `@bot image a cat sitting on a rainbow`"
            )
            return

        # Rate limit check
        allowed, remaining = check_image_rate_limit(message.author.id)
        if not allowed:
            await message.reply(
                f"You've used all **{IMAGE_RATE_LIMIT_PER_DAY}** image generations for today. "
                "Try again in 24 hours!"
            )
            return

        logger.info(
            "Generating image for user %s (ID: %s) with prompt: %s",
            message.author.name,
            message.author.id,
            image_prompt,
        )

        async with message.channel.typing():
            result = await asyncio.to_thread(generate_image, image_prompt)

        if "error" in result:
            await message.reply(f"❌ Error generating image: {result['error']}")
            return

        record_image_use(message.author.id)
        _, remaining_after = check_image_rate_limit(message.author.id)

        file_path = result.get("file_path")
        filename = result.get("filename", "image.png")

        footer_parts = [f"Requested by {message.author.name}"]
        if remaining_after >= 0:
            footer_parts.append(f"{remaining_after} generation(s) remaining today")

        embed = discord.Embed(
            title="🎨 Generated Image",
            description=f"**Prompt:** {image_prompt}",
            color=0x00ff00,
        )
        embed.set_image(url=f"attachment://{filename}")
        embed.set_footer(
            text=" · ".join(footer_parts),
            icon_url=message.author.avatar.url if message.author.avatar else None,
        )

        if file_path and os.path.exists(file_path):
            try:
                await message.reply(
                    embed=embed,
                    file=discord.File(file_path, filename=filename),
                )
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
        # if a reply, use messages before + replied message (primary) and after (secondary, lower priority)
        replied_message = message.reference.resolved
        replied_content = f"{replied_message.author.name}: {replied_message.content}"
        # messages before the replied message (primary context)
        before_messages = []
        async for historic_msg in message.channel.history(limit=MESSAGE_HISTORY_LIMIT, before=replied_message):
            before_messages.append(f"{historic_msg.author.name}: {historic_msg.content}")
        if before_messages:
            before_messages.reverse()
            primary_context = "\n".join(before_messages) + "\n" + replied_content
        else:
            primary_context = replied_content
        # messages after the replied message (secondary context, lower priority)
        after_limit = max(5, MESSAGE_HISTORY_LIMIT // 3)
        after_messages = []
        async for historic_msg in message.channel.history(limit=after_limit, after=replied_message):
            if historic_msg.id == message.id:
                continue
            after_messages.append(f"{historic_msg.author.name}: {historic_msg.content}")
        if after_messages:
            context_for_grok = (
                "[Primary context - conversation leading up to the message you're replying to:]\n"
                f"{primary_context}\n\n"
                "[Additional context - messages after the replied message (lower priority):]\n"
                + "\n".join(after_messages)
            )
        else:
            context_for_grok = primary_context
        logger.info("Context is reply mode: message %s, %s before, %s after", replied_message.id, len(before_messages), len(after_messages))
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

    # --- Response: agent loop (tools) or plain stream ---
    sent_message = None
    full_response = ""
    current_chunk = ""
    last_edit_time = 0.0
    tts_path: str | None = None
    system_prompt = build_system_prompt()

    try:
        sent_message = await message.reply("🧠 Thinking...")

        if LOCAL_TOOLS_ENABLED:
            async with message.channel.typing():
                full_response, tts_path = await asyncio.to_thread(
                    run_provider_agent_loop,
                    AI_PROVIDER,
                    context_for_grok,
                    question_text,
                    system_prompt,
                )
            # Simulate streaming by revealing word-by-word from the finished text
            words_shown = 0
            last_edit_time = time.monotonic()
            for match in re.finditer(r"\S+", full_response):
                words_shown += 1
                if words_shown % WORD_CHUNK_SIZE != 0:
                    continue
                if (time.monotonic() - last_edit_time) < EDIT_COOLDOWN_SECONDS:
                    continue
                preview = full_response[: match.end()]
                if len(preview) > DISCORD_MESSAGE_EDIT_LIMIT:
                    await sent_message.edit(
                        content=preview[:DISCORD_MESSAGE_EDIT_LIMIT] + "..."
                    )
                    break
                await sent_message.edit(content=preview + "...")
                last_edit_time = time.monotonic()
        else:
            async for chunk in query_ai_api_stream(
                context_for_grok, question_text, system_prompt
            ):
                full_response += chunk
                current_chunk += chunk
                stream_display = full_response
                if len(current_chunk.split()) >= WORD_CHUNK_SIZE and (
                    time.monotonic() - last_edit_time
                ) > EDIT_COOLDOWN_SECONDS:
                    if len(stream_display) > DISCORD_MESSAGE_EDIT_LIMIT:
                        await sent_message.edit(
                            content=stream_display[:DISCORD_MESSAGE_EDIT_LIMIT] + "..."
                        )
                        break
                    await sent_message.edit(content=stream_display + "...")
                    current_chunk = ""
                    last_edit_time = time.monotonic()

        if sent_message:
            final_content = full_response
            if len(final_content) > DISCORD_MESSAGE_LIMIT:
                final_content = final_content[:DISCORD_MESSAGE_LIMIT]
            if not final_content.strip():
                final_content = "I am sorry, I encountered an error and could not provide a response."

            if tts_path and os.path.exists(tts_path):
                try:
                    filename = os.path.basename(tts_path)
                    await sent_message.delete()
                    await message.reply(
                        final_content,
                        file=discord.File(tts_path, filename=filename),
                    )
                finally:
                    try:
                        os.remove(tts_path)
                    except Exception:
                        pass
            else:
                await sent_message.edit(content=final_content)
                if tts_path:
                    try:
                        os.remove(tts_path)
                    except Exception:
                        pass

    except Exception:
        logger.exception("An error occurred while generating the response")
        if sent_message:
            await sent_message.edit(
                content="An error occurred while generating the response."
            )


async def random_chat_loop() -> None:
    await client.wait_until_ready()
    if not RANDOM_CHAT_ENABLED:
        logger.info("Random chat disabled.")
        return
    if not RANDOM_CHAT_CHANNEL_IDS:
        logger.warning("Random chat enabled but no channel IDs configured.")
        return

    while not client.is_closed():
        # Calculate a random interval between min and max for each iteration
        interval_minutes = random.randint(RANDOM_CHAT_INTERVAL_MIN_MINUTES, RANDOM_CHAT_INTERVAL_MAX_MINUTES)
        interval_seconds = interval_minutes * 60
        logger.info("Next random chat check in %s minutes", interval_minutes)
        await asyncio.sleep(interval_seconds)

        if random.random() > RANDOM_CHAT_CHANCE:
            continue

        now = discord.utils.utcnow()
        for channel_id in RANDOM_CHAT_CHANNEL_IDS:
            channel = client.get_channel(channel_id)
            if channel is None:
                try:
                    channel = await client.fetch_channel(channel_id)
                except Exception:
                    logger.warning("Failed to fetch channel %s", channel_id)
                    continue

            if not isinstance(channel, discord.TextChannel):
                continue

            history_messages = []
            newest_message = None
            async for historic_msg in channel.history(limit=MESSAGE_HISTORY_LIMIT):
                if newest_message is None:
                    newest_message = historic_msg
                history_messages.append(f"{historic_msg.author.name}: {historic_msg.content}")

            if not newest_message or newest_message.author == client.user:
                continue

            recency_seconds = (now - newest_message.created_at).total_seconds()
            if recency_seconds > RANDOM_CHAT_RECENT_SECONDS:
                continue

            history_messages.reverse()
            context_for_ai = "\n".join(history_messages) if history_messages else "[No recent context]"
            prompt = (
                "Based on the recent messages, write one short message that fits the conversation. "
                "Do not address anyone by name unless others did. Keep it under 200 characters."
            )

            try:
                response = await generate_ai_response(
                    context_for_ai,
                    prompt,
                    system_prompt=build_random_chat_prompt(),
                )
                if response:
                    lower_response = response.strip().lower()
                    if lower_response.startswith("error:"):
                        logger.warning("Skipping random chat due to provider error: %s", response)
                        continue
                    await channel.send(response[:DISCORD_MESSAGE_LIMIT])
                    logger.info("Random chat message sent to channel %s", channel.id)
            except Exception:
                logger.exception("Failed to send random chat message")

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