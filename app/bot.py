import discord
from discord import app_commands
import os
import re
import httpx
import asyncio
import json
import time
import tempfile
import logging
from dotenv import load_dotenv
from xai_sdk import Client
from typing import AsyncGenerator, Optional, Dict, Tuple
from discord.ext import voice_recv, tasks
import wave
import threading

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
SPEAK_TEXT_REPLIES = os.getenv("SPEAK_TEXT_REPLIES", "false").lower() == "true"  # keep VC and chat separate by default

# Speech-to-text (transcription) via OpenAI - for voice messages/attachments and VC
STT_ENABLED = os.getenv("STT_ENABLED", "true").lower() == "true"
OPENAI_STT_MODEL = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
STT_INACTIVITY_SECONDS = float(os.getenv("STT_INACTIVITY_SECONDS", "0.8"))
STT_MIN_MS = int(os.getenv("STT_MIN_MS", "400"))
VOICE_CONTEXT_TURNS = int(os.getenv("VOICE_CONTEXT_TURNS", "6"))

# Auto-response settings (simulate new user)
AUTO_RESPONSE_ENABLED = os.getenv("AUTO_RESPONSE_ENABLED", "false").lower() == "true"
AUTO_RESPONSE_INTERVAL_SECONDS = int(os.getenv("AUTO_RESPONSE_INTERVAL_SECONDS", "300"))  # default 5 minutes
AUTO_RESPONSE_MESSAGE_COUNT = int(os.getenv("AUTO_RESPONSE_MESSAGE_COUNT", "10"))  # read last 10 messages
AUTO_RESPONSE_CHANNEL_IDS = os.getenv("AUTO_RESPONSE_CHANNEL_IDS", "")  # comma-separated channel IDs (empty = all channels)

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
voice_sink = None
voice_flusher_task: Optional[asyncio.Task] = None
voice_context_history: Dict[int, list] = {}

# Ensure opus is loaded for voice
try:
    if not discord.opus.is_loaded():
        discord.opus.load_opus('libopus.so.0')
        print('Loaded Opus library libopus.so.0')
except Exception as e:
    print(f'Warning: could not load Opus library automatically: {e}')


async def synthesize_and_play(text: str):
    global voice_client
    print(f"[VC TTS START] Generating TTS for: {text[:100]}...")
    
    if not VOICE_ENABLED:
        print("[VC TTS] VOICE_ENABLED=false, skipping")
        return
    if not (voice_client and voice_client.is_connected()):
        print("[VC TTS] No voice client connected, skipping")
        return
    if not openai_client:
        print("[VC TTS] OpenAI client not initialized, skipping")
        return

    async with voice_lock:
        print("[VC TTS] Acquired voice lock, creating temp file")
        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        print(f"[VC TTS] Temp file: {tmp_path}")

        # Generate TTS via OpenAI (stream to file)
        try:
            print(f"[VC TTS] Calling OpenAI TTS model={OPENAI_TTS_MODEL} voice={OPENAI_TTS_VOICE}")
            with openai_client.audio.speech.with_streaming_response.create(
                model=OPENAI_TTS_MODEL,
                voice=OPENAI_TTS_VOICE,
                input=text[:4000],  # basic limit for TTS prompt length
            ) as response:
                response.stream_to_file(tmp_path)
            print(f"[VC TTS] Audio generated, file size: {os.path.getsize(tmp_path)} bytes")
        except Exception as e:
            print(f"[VC TTS ERROR] Failed to generate audio: {e}")
            try:
                os.remove(tmp_path)
            except:
                pass
            return

        # Play in discord via ffmpeg
        if voice_client and voice_client.is_connected():
            print("[VC TTS] Starting playback in Discord")
            if voice_client.is_playing():
                print("[VC TTS] Waiting for current audio to finish...")
                while voice_client.is_playing():
                    await asyncio.sleep(0.25)
            
            try:
                source = discord.FFmpegPCMAudio(tmp_path)
                voice_client.play(source)
                print("[VC TTS] Audio playback started")
                while voice_client.is_playing():
                    await asyncio.sleep(0.25)
                print("[VC TTS] Audio playback finished")
            except Exception as e:
                print(f"[VC TTS ERROR] Playback failed: {e}")
            finally:
                try:
                    os.remove(tmp_path)
                    print("[VC TTS] Temp file cleaned up")
                except Exception as e:
                    print(f"[VC TTS] Failed to cleanup temp file: {e}")
        else:
            print("[VC TTS] Voice client disconnected during TTS generation")
            try:
                os.remove(tmp_path)
            except:
                pass

# Background task for auto-response feature
@tasks.loop(seconds=AUTO_RESPONSE_INTERVAL_SECONDS if AUTO_RESPONSE_ENABLED else 3600)
async def auto_respond_task():
    """Periodically reads past messages and responds as a new user joining the conversation."""
    if not AUTO_RESPONSE_ENABLED:
        return
    
    try:
        # Parse channel IDs if specified
        target_channel_ids = []
        if AUTO_RESPONSE_CHANNEL_IDS:
            target_channel_ids = [int(cid.strip()) for cid in AUTO_RESPONSE_CHANNEL_IDS.split(",") if cid.strip()]
        
        # Iterate through all text channels the bot has access to
        for guild in client.guilds:
            for channel in guild.text_channels:
                # Skip if we have specific channels configured and this isn't one of them
                if target_channel_ids and channel.id not in target_channel_ids:
                    continue
                
                # Check if bot has permission to read and send messages
                if not channel.permissions_for(guild.me).read_messages or not channel.permissions_for(guild.me).send_messages:
                    continue
                
                try:
                    # Fetch recent messages
                    messages = []
                    async for msg in channel.history(limit=AUTO_RESPONSE_MESSAGE_COUNT):
                        # Skip bot's own messages
                        if msg.author == client.user:
                            continue
                        messages.append(f"{msg.author.name}: {msg.content}")
                    
                    # If no messages found, skip this channel
                    if not messages:
                        continue
                    
                    # Reverse to get chronological order
                    messages.reverse()
                    context = "\n".join(messages)
                    
                    # Create a prompt as if the bot is a new user observing the conversation
                    question = "You are a new person who just joined this chat and read the recent conversation. Respond naturally as if you're contributing to the discussion. Don't introduce yourself, just jump in with a relevant comment, question, or observation about what was discussed. Keep it casual and conversational."
                    
                    print(f"[AUTO-RESPONSE] Generating response for channel: {channel.name} (guild: {guild.name})")
                    
                    # Generate response
                    full_response = ""
                    async for chunk in query_ai_api_stream(context, question):
                        full_response += chunk
                    
                    # If response is empty, skip
                    if not full_response.strip():
                        continue
                    
                    # Limit response length
                    if len(full_response) > 2000:
                        full_response = full_response[:2000]
                    
                    # Send the response
                    await channel.send(full_response)
                    print(f"[AUTO-RESPONSE] Sent response to {channel.name}: {full_response[:100]}...")
                    
                except discord.Forbidden:
                    print(f"[AUTO-RESPONSE] No permission to access channel: {channel.name}")
                except Exception as e:
                    print(f"[AUTO-RESPONSE] Error processing channel {channel.name}: {e}")
                    
    except Exception as e:
        print(f"[AUTO-RESPONSE] Error in auto_respond_task: {e}")

@auto_respond_task.before_loop
async def before_auto_respond():
    """Wait until the bot is ready before starting the auto-response task."""
    await client.wait_until_ready()

# on ready event
@client.event
async def on_ready():
    print(f'we have logged in as {client.user} (id: {client.user.id})')
    print(f"bot is ready and listening for mentions.")
    # quiet noisy gateway logs from voice-recv extension
    logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
    try:
        await tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print(f"Failed to sync commands: {e}")
    
    # Start auto-response task if enabled
    if AUTO_RESPONSE_ENABLED:
        auto_respond_task.start()
        print(f"[AUTO-RESPONSE] Task started - will run every {AUTO_RESPONSE_INTERVAL_SECONDS} seconds")
        if AUTO_RESPONSE_CHANNEL_IDS:
            print(f"[AUTO-RESPONSE] Target channels: {AUTO_RESPONSE_CHANNEL_IDS}")
        else:
            print("[AUTO-RESPONSE] Will respond in all accessible channels")


def _write_wav(pcm_bytes: bytes, path: str, sample_rate: int = 48000, channels: int = 2):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


class TranscribeSink(voice_recv.AudioSink):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        super().__init__()
        self.loop = loop
        self.user_buffers: Dict[int, bytearray] = {}
        self.user_last_active: Dict[int, float] = {}
        self.buffer_lock = threading.Lock()
        self.user_packet_counts: Dict[int, int] = {}
        self.user_last_packet_log: Dict[int, float] = {}

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData):
        if data.pcm is None or user is None:
            return
        uid = int(user.id)
        now = time.time()
        with self.buffer_lock:
            buf = self.user_buffers.get(uid)
            if buf is None:
                buf = bytearray()
                self.user_buffers[uid] = buf
            buf.extend(data.pcm)
            self.user_last_active[uid] = now
            # debug: every ~50 packets or 1s, log that we're receiving
            cnt = self.user_packet_counts.get(uid, 0) + 1
            self.user_packet_counts[uid] = cnt
            last_log = self.user_last_packet_log.get(uid, 0.0)
            if cnt % 50 == 0 or (now - last_log) > 1.0:
                print(f"[VC RX] uid={uid} packets={cnt} buffer_bytes={len(buf)}")
                self.user_last_packet_log[uid] = now

    async def flush_inactive(self, inactivity_seconds: float = STT_INACTIVITY_SECONDS, min_ms: int = STT_MIN_MS):
        """Periodically called to flush user buffers to STT when inactive."""
        now = time.time()
        to_flush: Dict[int, bytes] = {}
        with self.buffer_lock:
            for uid, last in list(self.user_last_active.items()):
                buf = self.user_buffers.get(uid)
                if not buf:
                    continue
                # Rough ms based on 16-bit stereo 48kHz => 192kB per second
                ms_len = int(len(buf) / 192_000 * 1000)
                if (now - last) >= inactivity_seconds and ms_len >= min_ms:
                    print(f"[VC FLUSH] uid={uid} bytes={len(buf)} ms~{ms_len}")
                    to_flush[uid] = bytes(buf)
                    self.user_buffers[uid] = bytearray()

        for uid, pcm in to_flush.items():
            await self._transcribe_and_respond(uid, pcm)

    async def _transcribe_and_respond(self, uid: int, pcm: bytes):
        print(f"[VC STT START] Processing {len(pcm)} bytes for uid={uid}")
        
        if not openai_client:
            print("[VC STT ERROR] OpenAI client not initialized")
            return
            
        try:
            # Create temp WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            print(f"[VC STT] Created temp WAV file: {tmp_path}")
            
            _write_wav(pcm, tmp_path)
            file_size = os.path.getsize(tmp_path)
            print(f"[VC STT] WAV file written, size: {file_size} bytes")
            
            # Transcribe with OpenAI
            text = None
            print(f"[VC STT] Calling OpenAI transcription model={OPENAI_STT_MODEL}")
            with open(tmp_path, "rb") as f:
                stt_resp = openai_client.audio.transcriptions.create(
                    model=OPENAI_STT_MODEL,
                    file=f,
                )
                text = getattr(stt_resp, "text", None) or str(stt_resp)
            
            # Cleanup temp file
            try:
                os.remove(tmp_path)
                print("[VC STT] Temp file cleaned up")
            except Exception as e:
                print(f"[VC STT] Failed to cleanup temp file: {e}")
                
            if not text or not text.strip():
                print("[VC STT] No text transcribed or empty result")
                return
                
            username = f"<@{uid}>"
            question = f"{username}: {text}"
            
            # Build per-channel voice context separate from text chat
            try:
                ch_id = int(self.voice_client.channel.id)
            except Exception:
                ch_id = 0
            history = voice_context_history.setdefault(ch_id, [])
            context = "\n".join(history[-VOICE_CONTEXT_TURNS:]) if history else "voice chat"

            print(f"[VC STT SUCCESS] {username}: {text}")
            print(f"[VC LLM START] Generating response with {len(context)} chars context")

            # Get AI response (streaming assembled)
            full = ""
            chunk_count = 0
            async for chunk in query_ai_api_stream(context, question):
                full += chunk
                chunk_count += 1
                # stream log (less verbose)
                if chunk_count % 5 == 0:  # every 5th chunk
                    snip = chunk.replace('\n', ' ')[:80]
                    if snip.strip():
                        print(f"[VC LLM Δ{chunk_count}] {snip}")
                        
            if not full.strip():
                full = "I'm sorry, I couldn't process that."
                
            print(f"[VC LLM SUCCESS] Generated {len(full)} chars in {chunk_count} chunks")
            
            # Update voice context history
            history.extend([f"User {username}: {text}", f"Assistant: {full}"])
            if len(history) > (VOICE_CONTEXT_TURNS * 2):
                del history[: len(history) - (VOICE_CONTEXT_TURNS * 2)]

            # Speak in voice
            print(f"[VC RESPONSE] Assistant: {full[:200]}{'...' if len(full) > 200 else ''}")
            await synthesize_and_play(full)
            
        except Exception as e:
            print(f"[VC ERROR] Voice transcribe/respond failed: {e}")
            import traceback
            traceback.print_exc()

    # realtime path removed


async def _start_voice_listen(vc: discord.VoiceProtocol):
    global voice_sink, voice_flusher_task
    loop = asyncio.get_running_loop()
    voice_sink = TranscribeSink(loop)
    # Directly listen with our PCM sink (wants_opus=False)
    vc.listen(voice_sink)
    print("[VC] Listening started (chained STT→LLM→TTS)")

    async def _flusher():
        while vc.is_connected():
            await asyncio.sleep(1.0)
            try:
                await voice_sink.flush_inactive()
                # Debug metric: buffer sizes
                total_bytes = sum(len(b) for b in getattr(voice_sink, 'user_buffers', {}).values())
                if total_bytes:
                    print(f"[VC] buffered bytes={total_bytes}")
            except Exception as e:
                print(f"Flusher error: {e}")
    voice_flusher_task = asyncio.create_task(_flusher())


@tree.command(name="join", description="Have the bot join your current voice channel (restricted)")
async def join(interaction: discord.Interaction):
    global voice_client
    # Defer immediately to avoid Unknown interaction (10062)
    try:
        await interaction.response.defer(ephemeral=True)
    except Exception:
        pass

    if str(interaction.user.id) != VOICE_ALLOWED_USER_ID:
        await interaction.followup.send("You are not allowed to use this command.", ephemeral=True)
        return

    if not isinstance(interaction.user, discord.Member):
        await interaction.followup.send("Cannot resolve your voice state.", ephemeral=True)
        return

    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.followup.send("You must be connected to a voice channel.", ephemeral=True)
        return

    channel = interaction.user.voice.channel
    try:
        if voice_client and voice_client.is_connected():
            if voice_client.channel.id == channel.id:
                await interaction.followup.send(f"Already connected to {channel.name}.", ephemeral=True)
                return
            await voice_client.move_to(channel)
        else:
            # Use voice receive client
            voice_client = await channel.connect(cls=voice_recv.VoiceRecvClient)
            # Start listening for inbound audio
            try:
                await _start_voice_listen(voice_client)
            except Exception as e:
                print(f"Failed to start voice receive: {e}")
        await interaction.followup.send(f"Joined {channel.name}.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"Failed to join: {e}", ephemeral=True)


@tree.command(name="leave", description="Have the bot leave the current voice channel")
async def leave(interaction: discord.Interaction):
    global voice_client
    # Defer immediately
    try:
        await interaction.response.defer(ephemeral=True)
    except Exception:
        pass
    if not voice_client or not voice_client.is_connected():
        await interaction.followup.send("I'm not connected to a voice channel.", ephemeral=True)
        return
    try:
        await voice_client.disconnect(force=True)
        voice_client = None
        try:
            global voice_flusher_task
            if voice_flusher_task:
                voice_flusher_task.cancel()
                voice_flusher_task = None
        except Exception:
            pass
        await interaction.followup.send("Left the voice channel.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"Failed to leave: {e}", ephemeral=True)

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

    # If there's an audio attachment and STT is enabled, transcribe it and use as the user's question
    if STT_ENABLED and message.attachments:
        audio_attachment = None
        for att in message.attachments:
            content_type = (att.content_type or "").lower()
            name_lower = att.filename.lower() if att.filename else ""
            if (
                content_type.startswith("audio/")
                or name_lower.endswith((".ogg", ".oga", ".mp3", ".wav", ".m4a", ".webm"))
            ):
                audio_attachment = att
                break

        if audio_attachment and openai_client:
            try:
                await message.channel.typing()
                # Download audio
                async with httpx.AsyncClient(timeout=60.0) as dl_client:
                    audio_bytes = (await dl_client.get(audio_attachment.url)).content

                # Save to temp file and send to OpenAI transcription
                with tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio_attachment.filename or "audio")[1] or ".ogg", delete=False) as tmp:
                    tmp_path = tmp.name
                    tmp.write(audio_bytes)

                with open(tmp_path, "rb") as f:
                    stt_resp = openai_client.audio.transcriptions.create(
                        model=OPENAI_STT_MODEL,
                        file=f,
                    )
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

                transcribed_text = getattr(stt_resp, "text", None) or str(stt_resp)
                if transcribed_text:
                    # Replace user's question with the transcription
                    question_text = f"{message.author.name} says (voice): {transcribed_text}"
            except Exception as e:
                print(f"Transcription error: {e}")

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

    # Add the author's name to the question text (text-only path; VC is separate)
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
            if SPEAK_TEXT_REPLIES:
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
