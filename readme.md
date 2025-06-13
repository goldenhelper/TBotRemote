# Telegram Bot with Claude Integration

A Telegram bot that integrates with Anthropic's Claude AI, featuring token management and chat history. Built for AWS Lambda deployment.

## Features

- Claude AI Integration
- Token Management System
- DynamoDB Backend
- Secure System Prompt Encryption
- Local Testing Support

## Prerequisites

- Python 3.12
- AWS Account with DynamoDB access
- Telegram Bot Token
- Claude API Key

## Setup

1. **Environment Variables**
   Create a `.env` file with:
   ```
   BOT_TOKEN=your_telegram_bot_token
   CLAUDE_API_KEY=your_claude_api_key
   ADMIN_USER_ID=your_telegram_user_id
   DECRYPTION_KEY=your_encryption_key
   DYNAMODB_REGION=eu-north-1
   TOKEN_TABLE_NAME=telegram-bot-tokens
   CHAT_HISTORY_TABLE=telegram-bot-chat-history
   ```

2. **DynamoDB Tables**
   Create two DynamoDB tables:
   - `telegram-bot-tokens` (partition key: `user_id`)
   - `telegram-bot-chat-history` (partition key: `user_id`)

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Local Testing

1. **Using Polling (Recommended for Development)**
   ```bash
   python local_testing_polling.py
   ```

## Deployment

1. **Create Deployment Package**
   ```powershell
   # Create deployment directory
   mkdir deployment
   cd deployment

   # Copy files
   Copy-Item ..\lambda_function.py .
   Copy-Item ..\models.py .
   Copy-Item ..\exceptions.py .
   Copy-Item ..\encrypted_prompt.bin .
   Copy-Item -Recurse ..\handlers .
   Copy-Item -Recurse ..\services .
   Copy-Item -Recurse ..\config .
   Copy-Item ..\requirements.txt .

   # Install dependencies
   pip install -r requirements.txt --platform manylinux2014_x86_64 --only-binary=:all: --target . --python-version 3.12

   # Create ZIP
   Compress-Archive -Path * -DestinationPath ..\function.zip -Force
   cd ..
   ```

2. **Upload to AWS Lambda**
   - Create new Lambda function
   - Upload `function.zip`
   - Set environment variables
   - Configure API Gateway trigger

## Bot Commands

- `/start` - Register new user
- `/help` - Show help message
- `/tokens_amount` - Check remaining tokens
- `/ask_for_tokens` - Request more tokens from admin

## Architecture

- **Lambda Function**: Main entry point
- **Message Handler**: Core bot logic
- **Services**:
  - Token Manager: DynamoDB-based token system
  - Chat History: Message history management
  - Claude Service: AI integration

## Security

- System prompt is encrypted
- Sensitive data stored in environment variables
- Token-based access control
- Admin-only functions

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
