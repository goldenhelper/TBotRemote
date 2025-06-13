import json
import logging
import asyncio
from typing import Dict, Any
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler
from telegram.request import HTTPXRequest
from config.config import LambdaConfig
from services.model_manager import ModelManager
from services.chat_history import ChatHistoryManager
from services.chat_history import AWSStorage
from handlers.message_handler import MainHandler
import boto3

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

    bot_id = config.bot_token.split(':')[0]

    system_prompt = boto3.client('ssm').get_parameter(Name='/telegram-bot/system-prompt')['Parameter']['Value']


    # Initialize services
    model_manager = ModelManager(
        token_table_name=config.token_table_name,
        region=config.aws_region
    )
    
    chat_history_manager = ChatHistoryManager(
        storage=AWSStorage(
            table_name=config.chat_history_table,
            region=config.aws_region,
            default_system_prompt=system_prompt
        )
    )

    # Initialize message handler
    message_handler = MainHandler(
        model_manager=model_manager,
        chat_history_manager=chat_history_manager,
        api_keys={
            'gemini': config.gemini_api_key, 
            'claude': config.claude_api_key,
            'openai': config.openai_api_key
        },
        admin_user_id=config.admin_user_id,
        bot_id=bot_id,
        aws_region=config.aws_region
    )

    # Register handlers
    application.add_handler(CommandHandler("start", message_handler.start_command))
    application.add_handler(CommandHandler("help", message_handler.help_command))
    application.add_handler(CommandHandler("tokens_amount", message_handler.get_tokens_command))
    application.add_handler(CommandHandler("ask_for_tokens", message_handler.ask_for_tokens_command))
    application.add_handler(CommandHandler("set_model", message_handler.set_model_command))
    application.add_handler(CommandHandler("get_model_data", message_handler.get_model_data_command))
    application.add_handler(CommandHandler("set_system_prompt", message_handler.set_system_prompt_command))
    application.add_handler(CommandHandler("get_system_prompt", message_handler.get_system_prompt_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler.handle_message))
    
    # Add callback query handlers
    application.add_handler(CallbackQueryHandler(message_handler.choose_role_button_callback, pattern="^choose_role_"))
    application.add_handler(CallbackQueryHandler(message_handler.remove_role_button_callback, pattern="^remove_role_"))
    application.add_handler(CallbackQueryHandler(message_handler.set_model_button_callback, pattern="^set_model_"))

    return application


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler function"""
    global application
    application = create_application()

    return asyncio.get_event_loop().run_until_complete(webhook(event, context, application))
