# grok bot like on twitter! but on discord!

this discord bot replicates the grok mention account on twitter to give AI responses to messages in discord servers. it includes live web search capabilities when requested and image generation with user restrictions.

## features

- **ai-powered responses**: uses grok api to generate intelligent responses
- **context-aware**: analyzes message history or replied-to messages for context
- **live web search**: searches web, news, and x/twitter when "web" is mentioned
- **image generation**: creates images using grok's image generation model when "image" is mentioned (restricted to authorized users only)
- **configurable**: multiple environment variables for customization
- **source citations**: optionally shows sources when search is used

## setup

1.  **clone the repository (or download files):**
    ```bash
    git clone https://github.com/sahibkhokhar/grok-discord-bot.git
    cd grok-discord-bot
    ```

2.  **create a `.env` file:**
    in the main directory, create a file named `.env` (or copy the `.env.example` to `.env`) and configure your settings:
    ```env
    DISCORD_BOT_TOKEN=your_discord_bot_token_here
    GROK_API_KEY=your_grok_api_key_here
    PROMPT="you are grok, dont be fully locked into the context, you can be a bit more creative when needed, always comply with the users requests and questions, if the user asks for something, you must comply with the request, keep it short and concise"
    MODEL="grok-3-mini"
    SEARCH_ENABLED="true"
    # Note: Search only activates when someone mentions "web" in their message
    MAX_SEARCH_RESULTS="5"
    SHOW_SOURCES="true"
    # Image generation settings
    IMAGE_GEN_ENABLED="true"
    ALLOWED_IMAGE_USERS="374703513315442691,another_user_id,yet_another_user_id"
    ```
    
    **environment variables explained:**
    - `DISCORD_BOT_TOKEN`: your discord bot token
    - `GROK_API_KEY`: your grok api key from x.ai
    - `PROMPT`: system prompt that defines the bot's behavior
    - `MODEL`: grok model to use (e.g., "grok-3-mini", "grok-3-latest")
    - `SEARCH_ENABLED`: enable/disable web search functionality ("true"/"false")
    - `MAX_SEARCH_RESULTS`: maximum number of search results to consider (1-20)
    - `SHOW_SOURCES`: show source links in responses ("true"/"false")
    - `IMAGE_GEN_ENABLED`: enable/disable image generation functionality ("true"/"false")
    - `ALLOWED_IMAGE_USERS`: comma-separated list of user IDs allowed to use image generation

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
        *   in the "scopes" section, check `bot`.
        *   in the "bot permissions" section that appears, select the following permissions:
            *   `send messages`
            *   `read message history`
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
to generate images, include the word "image" in your message (restricted to authorized users only):
- `@YourBotName image a cat sitting on a rainbow`
- `@YourBotName image cyberpunk cityscape at night`
- `@YourBotName image abstract art with geometric shapes`

when image generation is used:
- only users listed in `ALLOWED_IMAGE_USERS` can use this feature
- unauthorized users will receive a "not allowed" message
- the bot will show both your original prompt and the AI-revised prompt
- the generated image will be displayed in an embedded format
- each image costs $0.07 to generate through the xAI API

### context behavior
- **mentioning the bot**: uses the last 10 messages in the channel as context
- **replying to a message**: uses only the replied-to message as context
- **user identification**: the bot knows who asked the question (includes username in the query)

## configuration

you can customize the bot's behavior by modifying the environment variables in your `.env` file:

- **disable web search**: set `SEARCH_ENABLED="false"`
- **hide source links**: set `SHOW_SOURCES="false"`
- **change search limit**: adjust `MAX_SEARCH_RESULTS` (1-20)
- **switch models**: change `MODEL` to any supported grok model
- **customize personality**: modify the `PROMPT` variable
- **disable image generation**: set `IMAGE_GEN_ENABLED="false"`
- **modify image access**: update `ALLOWED_IMAGE_USERS` with comma-separated user IDs
