# grok bot like on twitter! but on discord!

this discord bot replicates the grok mention account on twitter to give AI responses to messages in discord servers.

## setup

1.  **clone the repository (or download files):**
    ```bash
    git clone https://github.com/sahibkhokhar/grok-discord-bot.git
    cd grok-discord-bot
    ```

2.  **create a `.env` file:**
    in the main directory, create a file named `.env` (or copy the `.env.example` to `.env`and add your discord bot token and grok api key:
    ```env
    DISCORD_BOT_TOKEN=your_discord_bot_token_here
    GROK_API_KEY=your_grok_api_key_here
    ```
    *   replace `your_discord_bot_token_here` with your actual discord bot token.
    *   replace `your_grok_api_key_here` with your actual grok api key.

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

1.  find a message in your discord server that you want to analyze or ask a question about.
2.  reply to that message.
3.  in your reply, mention the bot (e.g., `@YourBotName`) and then type your question.
    *example:* `@YourBotName is the statement in the original message accurate according to current knowledge?`

the bot will then process the original message and your question using the grok api and post a reply.
