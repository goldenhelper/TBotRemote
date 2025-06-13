# TBot Remote

Telegram bot that integrates several large language models and is designed for deployment on AWS Lambda.  The bot keeps long‑term chat history, supports roles, manages user tokens and can describe media sent in conversations.

## Features

* **Multi‑model support** – choose between Google Gemini, Anthropic Claude and OpenAI models.
* **Media handling** – images, videos, stickers and GIFs are automatically described using Gemini【F:MEDIA_HANDLING.md†L1-L25】.
* **Chat history and notes** – messages are stored in DynamoDB (or files when testing) with an optional memory updater for long‑term notes.
* **Role manager** – define global or chat‑specific personalities stored in DynamoDB.
* **Token management** – per‑chat token balance with atomic updates.
* **AWS Lambda ready** – webhook handler is provided for serverless deployment.
* **Local testing tools** – run via polling or webhook without deploying to AWS.

## Requirements

* Python 3.12
* Telegram bot token
* API keys for Gemini, Claude and OpenAI
* AWS credentials with access to DynamoDB and SSM Parameter Store

## Configuration

Create a `.env` file for local testing.  `LocalTestingConfig` in `config_local.py` lists all available variables:

```env
BOT_TOKEN=your_telegram_bot_token
CLAUDE_API_KEY=your_claude_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
ADMIN_USER_ID=123456789
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
DEFAULT_ROLE_ID=role-id-from-dynamodb
AWS_REGION=eu-north-1
TOKEN_TABLE_NAME=telegram-bot-tokens
CHAT_HISTORY_TABLE=telegram-bot-chat-history
ROLE_TABLE_NAME=telegram-bot-roles
```

You will need three DynamoDB tables (`telegram-bot-tokens`, `telegram-bot-chat-history` and `telegram-bot-roles`).  System prompts and formatting instructions are typically stored in AWS SSM Parameter Store.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running Locally

For development it is easiest to run the bot in polling mode:

```bash
python local_testing_polling.py
```

Alternatively you can use a local webhook server:

```bash
python local_testing_webhooks.py  # requires an externally reachable URL
```

Use `setup_webhook.py` to register the webhook URL with Telegram.

## Deployment to AWS Lambda

1. Package the function and dependencies (example on Windows PowerShell):

   ```powershell
   mkdir deployment
   cd deployment
   Copy-Item ..\lambda_function.py .
   Copy-Item ..\models.py .
   Copy-Item ..\exceptions.py .
   Copy-Item -Recurse ..\handlers .
   Copy-Item -Recurse ..\services .
   Copy-Item ..\requirements.txt .
   pip install -r requirements.txt --platform manylinux2014_x86_64 --only-binary=:all: --target . --python-version 3.12
   Compress-Archive -Path * -DestinationPath ..\function.zip -Force
   cd ..
   ```

2. Upload `function.zip` to AWS Lambda and set the same environment variables as in your `.env` file.
3. Configure an API Gateway or webhook URL for Telegram to call the Lambda handler.

## Bot Commands

User commands:

* `/start` – register or reset the chat
* `/help` – show available commands
* `/tokens_amount` – check remaining tokens
* `/ask_for_tokens` – request more tokens from the admin

Admin commands (restricted to the configured administrator):

* `/choose_model` – select the LLM model
* `/get_model_data` – show usage counters
* `/choose_role` – set a role for the chat
* `/add_role` – create a new role
* `/remove_role` – delete a role
* `/give_bot_tokens` – add tokens to a chat
* `/clear_history` – wipe stored conversation
* `/get_notes` – retrieve stored notes
* `/set_aliveness` – set spontaneous reply probability
* `/how_alive` – display current aliveness value
* `/reset_model_data` – reset global model usage statistics

Additional commands are registered in `lambda_function.py` and `local_testing_polling.py`.

## Architecture Overview

* **lambda_function.py** – entry point used by AWS Lambda
* **handlers/** – message and media processing logic
* **services/** – integrations for LLM providers, token management, chat storage and roles
* **models.py** – dataclasses for chat messages and token transactions

Refer to `MEDIA_HANDLING.md` for detailed information about how images and video are processed.

## Security Notes

API keys and system prompts should be stored in environment variables or AWS Parameter Store.  Avoid committing secrets to the repository.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request
