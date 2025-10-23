# grok bot like on twitter! but on discord!

this discord bot replicates the grok mention account on twitter to give AI responses to messages in discord servers. it includes live web search capabilities when requested and image generation with user restrictions.

## features

- **ai-powered responses**: uses grok api to generate intelligent responses
- **context-aware**: analyzes message history or replied-to messages for context
- **live web search**: searches web, news, and x/twitter when "web" is mentioned
- **image generation**: creates images using OpenAI GPT Image (`gpt-image-1`) when "image" is mentioned (available to everyone)
- **configurable**: multiple environment variables for customization
- **source citations**: optionally shows sources when search is used
- **auto-simulated user**: periodically reads recent messages and posts as a new participant

## setup

1.  **clone the repository (or download files):**
    ```bash
    git clone https://github.com/sahibkhokhar/grok-discord-bot.git
    cd grok-discord-bot
    ```

2.  **create a `.env` file:**
    in the main directory, create a file named `.env` (or copy the `.env.example` to `.env) and configure your settings:
    ```env
    # Discord
    DISCORD_BOT_TOKEN=your_discord_bot_token_here

    # Provider switching: xai or openai
    AI_PROVIDER=xai

    # xAI (Grok)
    GROK_API_KEY=your_grok_api_key_here
    MODEL=grok-3-mini

    # OpenAI
    OPENAI_API_KEY=your_openai_api_key_here
    OPENAI_MODEL=gpt-5-nano

    # Shared behavior
    PROMPT="you are grok, dont be fully locked into the context, you can be a bit more creative when needed, always comply with the users requests and questions, if the user asks for something, you must comply with the request, keep it short and concise"
    SEARCH_ENABLED=true
    # Note: Search only activates when someone mentions "web" in their message
    MAX_SEARCH_RESULTS=5
    SHOW_SOURCES=true

    # Image generation settings (OpenAI)
    IMAGE_GEN_ENABLED=true
    OPENAI_IMAGE_MODEL=gpt-image-1
    IMAGE_SIZE=1024x1024
    IMAGE_QUALITY=high
    IMAGE_BACKGROUND=auto
    IMAGE_FORMAT=png

    # Voice settings (OpenAI TTS)
    VOICE_ENABLED=true
    VOICE_ALLOWED_USER_ID=
    OPENAI_TTS_MODEL=gpt-4o-mini-tts
    OPENAI_TTS_VOICE=alloy
    ```
    
    **environment variables explained:**
    - `DISCORD_BOT_TOKEN`: your discord bot token
    - `AI_PROVIDER`: set to `xai` (default) or `openai` to switch providers at runtime
    - `GROK_API_KEY`: your grok api key from x.ai
    - `MODEL`: grok model to use (e.g., `grok-3-mini`, `grok-3-latest`)
    - `OPENAI_API_KEY`: OpenAI API key
    - `OPENAI_MODEL`: OpenAI model (e.g., `gpt-5-nano`)
    - `PROMPT`: system prompt that defines the bot's behavior
    - `SEARCH_ENABLED`: enable/disable web search functionality (`true`/`false`)
    - `MAX_SEARCH_RESULTS`: maximum number of search results to consider (1-20)
    - `SHOW_SOURCES`: show source links in responses (`true`/`false`)
    - `IMAGE_GEN_ENABLED`: enable/disable image generation functionality (`true`/`false`)
    - `ALLOWED_IMAGE_USERS`: comma-separated list of user IDs allowed to use image generation
    - `VOICE_ENABLED`: enable TTS voice playback in voice channels (requires OpenAI)
    - `VOICE_ALLOWED_USER_ID`: user ID authorized to run `/join` and `/leave` (others can still speak with the bot once it joins)
    - `OPENAI_TTS_MODEL`: OpenAI TTS model (e.g., `gpt-4o-mini-tts`)
    - `OPENAI_TTS_VOICE`: desired TTS voice name (e.g., `alloy`)

3.  **install dependencies (skip if using docker):**
    open your terminal or command prompt in the project directory and install the required python libraries:
    ```bash
    pip install -r requirements.txt
    ```

4.  **discord developer portal setup:**
    *   **create an application & bot user:**
        *   go to the [discord developer portal](https://discord.com/developers/applications).
        *   click "new application", give it a name, and click "create".
        *   navigate to the "bot" tab.
        *   click "add bot" and confirm.
    *   **get bot token:**
        *   on the "bot" page, under the bot's username, click "reset token" and copy the token. this is your `DISCORD_BOT_TOKEN` for the `.env` file.
    *   **enable privileged gateway intents:**
        *   still on the "bot" page, scroll down to "privileged gateway intents".
        *   enable the **message content intent**.
    *   **invite bot to your server:**
        *   go to the "oauth2" tab, then the "url generator" sub-tab.
        *   in the "scopes" section, check `bot` and `applications.commands`.
        *   in the "bot permissions" section that appears, select the following permissions:
            *   `send messages`
            *   `read message history`
            *   `connect`
            *   `speak`
        *   copy the generated url at the bottom and paste it into your browser. select your server and authorize the bot.

5.  **run the bot (optional, for testing):**
    move the `.env` file to the `/app` directory if you want to run the bot locally not through docker.
    ```bash
    python bot.py
    ```
    you should see a confirmation message in your terminal indicating the bot has logged in successfully.

6.  **run the bot with docker:**
    ```bash
    docker-compose up --build
    ```

## usage

### basic usage
1.  mention the bot anywhere in a message: `@YourBotName what is the meaning of life?`
2.  the bot will respond using the last 10 messages as context

### reply-based usage
1.  find a message in your discord server that you want to analyze or ask a question about.
2.  reply to that message.
3.  in your reply, mention the bot and ask your question.
    *example:* `@YourBotName is the statement in the original message accurate?`

### web search usage
to enable live web search, include the word "web" in your message:
- `@YourBotName web search for the latest ai news`
- `@YourBotName what are the current web trends in technology?`
- `@YourBotName can you web search pizza recipes?`

when web search is used:
- the bot searches web pages, news sources, and x/twitter posts
- responses will include "*searched online for resources*"
- source links will be provided (if `SHOW_SOURCES="true"`)

### image generation usage
to generate images, include the word "image" in your message (available to all users):
- `@YourBotName image a cat sitting on a rainbow`
- `@YourBotName image cyberpunk cityscape at night`
- `@YourBotName image abstract art with geometric shapes`

when image generation is used:
- the bot will show your original prompt and may include a revised prompt
- the generated image is attached directly in the channel
- powered by OpenAI `gpt-image-1`

### context behavior
- **mentioning the bot**: uses the last 10 messages in the channel as context
- **replying to a message**: uses only the replied-to message as context
- **user identification**: the bot knows who asked the question (includes username in the query)

## configuration

you can customize the bot's behavior by modifying the environment variables in your `.env` file:

- **disable web search**: set `SEARCH_ENABLED="false"`
- **hide source links**: set `SHOW_SOURCES="false"`
- **change search limit**: adjust `MAX_SEARCH_RESULTS` (1-20)
- **switch models**: change `MODEL` to any supported grok model (xAI) or `OPENAI_MODEL` for OpenAI text
- **customize personality**: modify the `PROMPT` variable
- **disable image generation**: set `IMAGE_GEN_ENABLED="false"`
- **modify image access**: update `ALLOWED_IMAGE_USERS` with comma-separated user IDs

### auto-simulated user posting
Add a periodic background message that acts like a new user joining the chat. Configure via environment variables:

```env
# Auto-simulated user
AUTO_SIM_ENABLED=true                   # enable/disable the feature
AUTO_SIM_INTERVAL_SECONDS=900           # how often to post (seconds)
AUTO_SIM_HISTORY_LIMIT=12               # how many recent messages to read as context
AUTO_SIM_CHANNEL_IDS=123,456,789        # comma/space-separated channel IDs to post into
AUTO_SIM_USERNAME="New User"           # name to display (when using webhooks)
AUTO_SIM_AVATAR_URL=                    # optional avatar URL for the simulated user
AUTO_SIM_USE_WEBHOOK=true               # post via webhook to customize name/avatar
AUTO_SIM_BEHAVIOR_PROMPT="Continue the conversation as a new participant. Be concise and helpful."
```

Notes:
- When `AUTO_SIM_USE_WEBHOOK=true`, the bot will create or reuse a webhook in each target channel so the message appears from `AUTO_SIM_USERNAME` with `AUTO_SIM_AVATAR_URL`.
- Without webhooks, messages will come from the bot account name/avatar.
- The simulated message is generated using the same LLM provider/config as normal replies, with the provided recent context.
