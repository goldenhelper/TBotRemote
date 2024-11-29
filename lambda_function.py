import json
import logging
import asyncio
from typing import Dict, Any
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from telegram.request import HTTPXRequest
from config.config import LambdaConfig
from services.token_manager import TokenManager
from services.chat_history import ChatHistoryManager
from services.claude_service import ClaudeService
from handlers.message_handler import MainHandler
from cryptography.fernet import Fernet


# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

async def webhook(event: Dict[str, Any], context: Any, application: Application) -> Dict[str, Any]:
    """AWS Lambda handler for Telegram webhook"""
    try:
        if not event.get('body'):
            return {
                'statusCode': 400,
                'body': json.dumps('No body found in request')
            }

        # Initialize application
        await application.initialize()

        body = json.loads(event['body'])
        update = Update.de_json(body, application.bot)
        await application.process_update(update)

        # Shutdown application
        await application.shutdown()

        return {
            'statusCode': 200,
            'body': json.dumps('OK')
        }
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps('LOL, ur so stupid')
        }

def create_application(custom_config = None):
    """Create and configure the bot application for Lambda"""
    config = custom_config if custom_config else LambdaConfig()
    
    # Initialize bot with custom request object for Lambda
    request = HTTPXRequest(connection_pool_size=1)
    application = Application.builder() \
        .token(config.bot_token) \
        .concurrent_updates(True) \
        .http_version("1.1") \
        .get_updates_request(request) \
        .build()

    # Initialize services
    token_manager = TokenManager(
        table_name=config.token_table_name,
        region=config.dynamodb_region
    )
    
    chat_history = ChatHistoryManager(
        table_name=config.chat_history_table,
        region=config.dynamodb_region
    )
    
    claude_service = ClaudeService(api_key=config.claude_api_key)

    # Load system prompt
    with open("encrypted_prompt.bin", "rb") as file:
        encrypted_prompt = file.read()

    f = Fernet(config.decryption_key)
    system_prompt = f.decrypt(encrypted_prompt).decode()

    # Initialize message handler
    message_handler = MainHandler(
        token_manager=token_manager,
        chat_history=chat_history,
        claude_service=claude_service,
        admin_user_id=config.admin_user_id,
        system_prompt=system_prompt
    )

    # Register handlers
    application.add_handler(CommandHandler("start", message_handler.start_command))
    application.add_handler(CommandHandler("help", message_handler.help_command))
    application.add_handler(CommandHandler("tokens_amount", message_handler.get_tokens_command))
    application.add_handler(CommandHandler("ask_for_tokens", message_handler.ask_for_tokens_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler.handle_message))

    return application


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler function"""
    global application
    application = create_application()

    return asyncio.get_event_loop().run_until_complete(webhook(event, context, application))





