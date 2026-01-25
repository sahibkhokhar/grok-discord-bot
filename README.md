# grok bot like on twitter! but on discord!

this discord bot gives AI responses to messages in discord servers. it includes live web search capabilities and image generation.

## features

- **ai-powered responses**: uses xAI Grok or OpenAI models to generate intelligent responses
- **context-aware**: analyzes message history or replied-to messages for context
- **live web search**: searches web, news, and x/twitter when "web" is mentioned (xAI only)
- **image generation**: creates images using OpenAI GPT Image (`gpt-image-1`) when "image" is mentioned (available to everyone)
- **provider switching**: easily switch between xAI (Grok) and OpenAI (ChatGPT)
- **configurable**: multiple environment variables for customization

## setup

1.  **clone the repository (or download files):**
    ```bash
    git clone https://github.com/sahibkhokhar/grok-discord-bot.git
    cd grok-discord-bot
    ```

2.  **create a `.env` file:**
    in the main directory, create a file named `.env` and configure your settings:
    ```env
    # Discord
    DISCORD_BOT_TOKEN=your_discord_bot_token_here

    # Provider switching: xai or openai
    AI_PROVIDER=openai

    # xAI (Grok)
    GROK_API_KEY=your_grok_api_key_here
    MODEL=grok-3-mini

    # OpenAI
    OPENAI_API_KEY=your_openai_api_key_here
    OPENAI_MODEL=gpt-5-nano

    # Shared behavior
    PROMPT="you are a helpful assistant; keep it short and concise"
    SEARCH_ENABLED=true
    MAX_SEARCH_RESULTS=5
    SHOW_SOURCES=false
    WORD_CHUNK_SIZE=12
    EDIT_COOLDOWN_SECONDS=1.5
    MESSAGE_HISTORY_LIMIT=30
    
    # User blocking (comma-separated user IDs)
    BLOCKED_USER_IDS=123456789,987654321

    # Image generation settings (OpenAI)
    IMAGE_GEN_ENABLED=true
    OPENAI_IMAGE_MODEL=gpt-image-1
    IMAGE_SIZE=1024x1024
    IMAGE_QUALITY=high
    IMAGE_BACKGROUND=auto
    IMAGE_FORMAT=png
    
    # Random chat (optional)
    RANDOM_CHAT_ENABLED=false
    RANDOM_CHAT_INTERVAL_MIN_MINUTES=20
    RANDOM_CHAT_INTERVAL_MAX_MINUTES=40
    RANDOM_CHAT_CHANCE=0.25
    RANDOM_CHAT_RECENT_SECONDS=300
    RANDOM_CHAT_CHANNEL_IDS=
    ```
    
    **environment variables explained:**
    - `DISCORD_BOT_TOKEN`: your discord bot token
    - `AI_PROVIDER`: set to `xai` (default) or `openai` to switch providers
    - `GROK_API_KEY`: your grok api key from x.ai
    - `MODEL`: grok model to use (e.g., `grok-3-mini`, `grok-3-latest`)
    - `OPENAI_API_KEY`: OpenAI API key
    - `OPENAI_MODEL`: OpenAI model (e.g., `gpt-5-nano`, `gpt-5`, `gpt-4.1`)
    - `PROMPT`: system prompt that defines the bot's behavior
    - `SEARCH_ENABLED`: enable/disable web search functionality (`true`/`false`)
    - `MAX_SEARCH_RESULTS`: maximum number of search results to consider (1-20)
    - `SHOW_SOURCES`: show source links in responses (`true`/`false`)
    - `MESSAGE_HISTORY_LIMIT`: number of previous messages to fetch for context (default: 30)
    - `BLOCKED_USER_IDS`: comma-separated list of Discord user IDs that are blocked from using the bot (e.g., `123456789,987654321`)
    - `IMAGE_GEN_ENABLED`: enable/disable image generation (`true`/`false`)
    - `OPENAI_IMAGE_MODEL`: OpenAI image model (default: `gpt-image-1`)
    - `IMAGE_SIZE`: image dimensions (`1024x1024`, `1024x1536`, `1536x1024`, `auto`)
    - `IMAGE_QUALITY`: image quality (`low`, `medium`, `high`, `auto`)
    - `IMAGE_BACKGROUND`: background type (`transparent`, `auto`)
    - `IMAGE_FORMAT`: output format (`png`, `jpeg`, `webp`)
    - `RANDOM_CHAT_ENABLED`: enable/disable random chat feature (`true`/`false`, default: `false`)
    - `RANDOM_CHAT_INTERVAL_MIN_MINUTES`: minimum minutes between random chat checks (default: 20)
    - `RANDOM_CHAT_INTERVAL_MAX_MINUTES`: maximum minutes between random chat checks (default: 40)
    - `RANDOM_CHAT_CHANCE`: probability of sending a message when conditions are met (0.0-1.0, default: 0.25)
    - `RANDOM_CHAT_RECENT_SECONDS`: only chat if there's been activity in the last N seconds (default: 300)
    - `RANDOM_CHAT_CHANNEL_IDS`: comma-separated list of channel IDs where random chat is allowed

3.  **install dependencies (skip if using docker):**
    ```bash
    pip install -r app/requirements.txt
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
        *   in the "scopes" section, check `bot`.
        *   in the "bot permissions" section that appears, select the following permissions:
            *   `send messages`
            *   `read message history`
            *   `embed links`
            *   `attach files`
        *   copy the generated url at the bottom and paste it into your browser. select your server and authorize the bot.

5.  **run the bot (optional, for testing):**
    ```bash
    cd app
    python bot.py
    ```

6.  **run the bot with docker:**
    ```bash
    docker compose up --build
    ```

## usage

### basic usage
1.  mention the bot anywhere in a message: `@YourBotName what is the meaning of life?`
2.  the bot will respond using recent message history as context (default: 30 messages)

### reply-based usage
1.  find a message in your discord server that you want to analyze or ask a question about.
2.  reply to that message.
3.  in your reply, mention the bot and ask your question.
    *example:* `@YourBotName is the statement in the original message accurate?`

### web search usage
to enable live web search (xAI only), include the word "web" in your message:
- `@YourBotName web search for the latest ai news`
- `@YourBotName what are the current web trends in technology?`

when web search is used:
- the bot searches web pages, news sources, and x/twitter posts
- responses will include context from search results

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
- **mentioning the bot**: uses recent message history in the channel as context (configurable via `MESSAGE_HISTORY_LIMIT`, default: 30 messages)
- **replying to a message**: uses only the replied-to message as context
- **user identification**: the bot knows who asked the question (includes username in the query)

## configuration

you can customize the bot's behavior by modifying the environment variables in your `.env` file:

- **switch providers**: set `AI_PROVIDER=openai` or `AI_PROVIDER=xai`
- **switch models**: change `MODEL` (for xAI) or `OPENAI_MODEL` (for OpenAI)
- **disable web search**: set `SEARCH_ENABLED=false`
- **disable image generation**: set `IMAGE_GEN_ENABLED=false`
- **customize personality**: modify the `PROMPT` variable
- **adjust image quality**: change `IMAGE_QUALITY` to `low`, `medium`, `high`, or `auto`
- **block users**: add user IDs to `BLOCKED_USER_IDS` (comma-separated) to prevent specific users from using the bot
