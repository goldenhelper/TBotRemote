"""
Webhook server for Render deployment with Supabase backend.
Uses Flask + Gunicorn for production serving.
"""
import os
import json
import asyncio
import logging
from flask import Flask, request, jsonify
from telegram import Update, Bot
from telegram.ext import Application

from config_supabase import SupabaseConfig
from handlers.register_handlers import register_handlers
from handlers.message_handler import MainHandler
from services.supabase_storage import SupabaseStorage
from services.supabase_model_manager import SupabaseModelManager
from services.supabase_role_manager import SupabaseRoleManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Silence noisy loggers
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('telegram').setLevel(logging.INFO)

app = Flask(__name__)

# Global application instance
telegram_app: Application = None

# Default formatting info
DEFAULT_FORMATTING_INFO = """
Format your responses using Telegram's supported markdown:
- *bold* for emphasis
- _italic_ for subtle emphasis
- `code` for inline code
- ```language
code block
``` for code blocks
- [text](url) for links

Keep responses concise and well-structured.
"""


def create_application():
    """Create and configure the Telegram application with Supabase backend."""
    global telegram_app

    if telegram_app is not None:
        return telegram_app

    config = SupabaseConfig()

    # Initialize Supabase services (settings are loaded from Supabase config table)
    model_manager = SupabaseModelManager(
        supabase_url=config.supabase_url,
        supabase_key=config.supabase_key,
    )

    role_manager = SupabaseRoleManager(
        supabase_url=config.supabase_url,
        supabase_key=config.supabase_key,
    )

    storage = SupabaseStorage(
        supabase_url=config.supabase_url,
        supabase_key=config.supabase_key,
        role_manager=role_manager,
    )

    bot_id = config.bot_token.split(':')[0]

    # Initialize message handler (settings like max_num_roles are loaded from Supabase)
    message_handler = MainHandler(
        model_manager=model_manager,
        storage=storage,
        role_manager=role_manager,
        api_keys={
            'gemini': config.gemini_api_key,
            'claude': config.claude_api_key,
            'openai': config.openai_api_key,
        },
        admin_user_id=config.admin_user_id,
        formatting_info=DEFAULT_FORMATTING_INFO,
        bot_id=int(bot_id),
        aws_region='',  # Not used with Supabase
        alertobot_token=config.alertobot_token,
    )

    # Build application
    telegram_app = Application.builder().token(config.bot_token).build()

    # Register all handlers
    register_handlers(telegram_app, message_handler)

    logger.info("Telegram application created successfully")
    return telegram_app


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for keep-alive pings."""
    return jsonify({"status": "ok"}), 200


@app.route('/wakeup', methods=['GET', 'POST'])
def wakeup():
    """
    Wake-up endpoint that gives the bot a chance to send spontaneous messages.

    This endpoint:
    1. Gets all chats with come_to_life_chance > 0
    2. For each chat, rolls the dice based on their aliveness setting
    3. If successful, sends a spontaneous message to that chat

    Can be called by a cron job to periodically wake the bot.
    """
    try:
        application = create_application()

        async def process_wakeup():
            async with application:
                # Get the message handler from the registered handlers
                message_handler = None
                for handler in application.handlers.get(0, []):
                    if hasattr(handler, 'callback') and hasattr(handler.callback, '__self__'):
                        handler_instance = handler.callback.__self__
                        if hasattr(handler_instance, 'send_spontaneous_message'):
                            message_handler = handler_instance
                            break

                if message_handler is None:
                    logger.error("Could not find message handler for wakeup")
                    return {"status": "error", "message": "Handler not found"}, 0

                # Get chats with aliveness > 0
                alive_chats = message_handler.storage.get_chats_with_aliveness()

                if not alive_chats:
                    return {"status": "ok", "message": "No alive chats", "sent": 0}, 0

                # Check bot tokens once before processing any chats
                bot_balance = await message_handler.model_manager.get_tokens(message_handler.bot_id)
                if bot_balance is None or bot_balance <= 0:
                    await message_handler.send_alert(f"🚨 Bot tokens depleted! The bot can no longer send spontaneous messages.")
                    return {"status": "ok", "message": "Bot has no tokens", "sent": 0}, 0

                sent_count = 0
                for chat_data in alive_chats:
                    chat_id = chat_data['chat_id']
                    chance = chat_data['come_to_life_chance']
                    if await message_handler.send_spontaneous_message(chat_id, application.bot, come_to_life_chance=chance):
                        sent_count += 1

                return {"status": "ok", "checked": len(alive_chats), "sent": sent_count}, sent_count

        result, sent = asyncio.run(process_wakeup())
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in wakeup: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        "status": "Bot is running",
        "health": "/health",
        "webhook": "/webhook",
        "wakeup": "/wakeup"
    }), 200


@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming Telegram webhook updates."""
    try:
        if not request.json:
            logger.warning("Received empty webhook request")
            return jsonify({"error": "No data received"}), 400

        # Create application if not exists
        application = create_application()

        # Parse update
        update = Update.de_json(request.json, application.bot)

        # Process update asynchronously
        async def process():
            async with application:
                await application.process_update(update)

        asyncio.run(process())

        return jsonify({"status": "ok"}), 200

    except Exception as e:
        logger.error(f"Error processing webhook: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # For local testing
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
