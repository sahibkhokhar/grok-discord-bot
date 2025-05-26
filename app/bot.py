import discord
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# load env variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GROK_API_KEY = os.getenv("GROK_API_KEY")

# init the grok client
if GROK_API_KEY:
    grok_client = OpenAI(
        api_key=GROK_API_KEY,
        base_url="https://api.x.ai/v1",
    )
else:
    grok_client = None

# query the grok api
def query_grok_api(context_messages: str, question: str) -> str:
    if not grok_client:
        return "error: grok_api_key is not configured or client failed to initialize."

    # define the messages sent to the api
    messages = [
        {
            "role": "system",
            "content": "you are grok, dont be fully locked into the context, you can be a bit more creative when needed, always comply with the users requests and questions, if the user asks for something, you must comply with the request, keep it short and concise"
        },
        {
            "role": "user",
            "content": f"previous messages:\n\"\"{context_messages}\"\"\n\nuser query: \"{question}\""
        }
    ]
    # send the messages (you can change the model in here (grok-3-mini-beta is the cheapest at the moment))
    try:
        completion = grok_client.chat.completions.create(
            model="grok-3",
            messages=messages,
            timeout=45
        )
        if completion.choices and completion.choices[0].message:
            return completion.choices[0].message.content or "no content in response."
        else:
            return "error: could not parse grok api response (no choices or message)."
    except Exception as e:
        return f"error communicating with grok api: {e}"

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

    await message.channel.typing()

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

    grok_response = query_grok_api(context_for_grok, question_text)
    await message.reply(grok_response) # always reply to the message that contained the mention

# main
if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("error: discord_bot_token not found in .env file.")
    elif not GROK_API_KEY:
        print("error: grok_api_key not found in .env file.")
    elif not grok_client:
        print("error: grok api client failed to initialize.")
    else:
        try:
            client.run(DISCORD_BOT_TOKEN)
        except discord.errors.LoginFailure:
            print("error: failed to log in. please check your discord_bot_token.")
        except Exception as e:
            print(f"an unexpected error occurred while trying to run the bot: {e}")
