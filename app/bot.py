import discord
import os
import re
import httpx
import asyncio
import json
import time
from dotenv import load_dotenv
from xai_sdk import Client

# load env variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GROK_API_KEY = os.getenv("GROK_API_KEY")
PROMPT = os.getenv("PROMPT")
MODEL = os.getenv("MODEL")
SEARCH_ENABLED = os.getenv("SEARCH_ENABLED", "false").lower() == "true"
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
SHOW_SOURCES = os.getenv("SHOW_SOURCES", "true").lower() == "true"
WORD_CHUNK_SIZE = int(os.getenv("WORD_CHUNK_SIZE", "12"))
EDIT_COOLDOWN_SECONDS = float(os.getenv("EDIT_COOLDOWN_SECONDS", "1.5"))
# Image generation settings
IMAGE_GEN_ENABLED = os.getenv("IMAGE_GEN_ENABLED", "true").lower() == "true"
ALLOWED_IMAGE_USERS = os.getenv("ALLOWED_IMAGE_USERS", "").split(",") if os.getenv("ALLOWED_IMAGE_USERS") else []
# Clean up any empty strings and whitespace
ALLOWED_IMAGE_USERS = [user_id.strip() for user_id in ALLOWED_IMAGE_USERS if user_id.strip()]

# Initialize xAI client for image generation
xai_client = Client(api_key=GROK_API_KEY) if GROK_API_KEY else None

def is_user_allowed_for_images(user_id: str) -> bool:
    """Check if user is allowed to use image generation"""
    return str(user_id) in ALLOWED_IMAGE_USERS

def generate_image(prompt: str) -> dict:
    """Generate an image using xAI's image generation API"""
    if not xai_client:
        return {"error": "xAI client not initialized - check your GROK_API_KEY"}
    
    try:
        response = xai_client.image.sample(
            model="grok-2-image",
            prompt=prompt,
            image_format="url"
        )
        return {"url": response.url, "revised_prompt": response.prompt}
    except Exception as e:
        return {"error": f"Error generating image: {str(e)}"}

# query the grok api
async def query_grok_api_stream(context_messages: str, question: str):
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

# setup the discord client
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.guilds = True

client = discord.Client(intents=intents)

# on ready event
@client.event
async def on_ready():
    print(f'we have logged in as {client.user} (id: {client.user.id})')
    print(f"bot is ready and listening for mentions.")

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
        # Check if user is allowed to use image generation
        if not is_user_allowed_for_images(str(message.author.id)):
            await message.reply("❌ You are not allowed to use the image generation feature.")
            return
        
        # Extract the image prompt (remove "image" keyword)
        image_prompt = re.sub(r'\bimage\b', '', question_text, flags=re.IGNORECASE).strip()
        
        if not image_prompt:
            await message.reply("Please provide a description for the image you want to generate. Example: `@bot image a cat sitting on a rainbow`")
            return
        
        await message.channel.typing()
        
        print(f"Generating image for user {message.author.name} (ID: {message.author.id}) with prompt: {image_prompt}")
        
        # Generate the image
        result = generate_image(image_prompt)
        
        if "error" in result:
            await message.reply(f"❌ Error generating image: {result['error']}")
            return
        
        # Create an embed for the image
        embed = discord.Embed(
            title="🎨 Generated Image",
            description=f"**Original prompt:** {image_prompt}\n**Revised prompt:** {result.get('revised_prompt', 'N/A')}",
            color=0x00ff00
        )
        embed.set_image(url=result['url'])
        embed.set_footer(text=f"Requested by {message.author.name}", icon_url=message.author.avatar.url if message.author.avatar else None)
        
        await message.reply(embed=embed)
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

        async for chunk in query_grok_api_stream(context_for_grok, question_text):
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

    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        if sent_message:
            await sent_message.edit(content="An error occurred while generating the response.")

# main
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("error: discord_bot_token not found in .env file.")
    elif not GROK_API_KEY:
        print("error: grok_api_key not found in .env file.")
    elif not PROMPT:
        print("error: PROMPT not found in .env file.")
    else:
        try:
            client.run(DISCORD_BOT_TOKEN)
        except discord.errors.LoginFailure:
            print("error: failed to log in. please check your discord_bot_token.")
        except Exception as e:
            print(f"an unexpected error occurred while trying to run the bot: {e}")
