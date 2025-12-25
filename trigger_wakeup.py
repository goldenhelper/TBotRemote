#!/usr/bin/env python3
"""
Trigger the wakeup functionality locally.
This simulates what the /wakeup endpoint does when called by a cron job.

Usage:
    python trigger_wakeup.py                  # Normal mode (dice roll applies)
    python trigger_wakeup.py --force-response # Force the bot to respond
"""
import argparse
import asyncio
import logging

from config_supabase import SupabaseConfig
from handlers.message_handler import MainHandler
from services.supabase_storage import SupabaseStorage
from services.supabase_model_manager import SupabaseModelManager
from services.supabase_role_manager import SupabaseRoleManager
from telegram import Bot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


async def trigger_wakeup(force_response: bool = False):
    """Trigger the wakeup process locally."""
    if force_response:
        logger.info("Force response mode enabled - bot will be told to respond")

    config = SupabaseConfig()

    # Initialize services
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

    bot_id = int(config.bot_token.split(':')[0])

    # Initialize message handler
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
        bot_id=bot_id,
        aws_region='',
        alertobot_token=config.alertobot_token,
    )

    # Create bot instance
    bot = Bot(token=config.bot_token)

    # Get chats with aliveness > 0
    alive_chats = storage.get_chats_with_aliveness()

    if not alive_chats:
        logger.info("No alive chats found")
        return

    logger.info(f"Found {len(alive_chats)} alive chats")

    # Check bot tokens once before processing any chats
    bot_balance = await model_manager.get_tokens(bot_id)
    if bot_balance is None or bot_balance <= 0:
        logger.warning("Bot has no tokens for spontaneous messages")
        await message_handler.send_alert(f"🚨 Bot tokens depleted! The bot can no longer send spontaneous messages.")
        return

    logger.info(f"Bot has {bot_balance} tokens")

    sent_count = 0
    for chat_data in alive_chats:
        chat_id = chat_data['chat_id']
        chance = chat_data['come_to_life_chance']
        logger.info(f"Processing chat {chat_id} (aliveness: {chance*100:.0f}%)")

        if await message_handler.send_spontaneous_message(chat_id, bot, come_to_life_chance=chance, force_response=force_response):
            sent_count += 1
            logger.info(f"✓ Sent spontaneous message to chat {chat_id}")
        else:
            logger.info(f"✗ No message sent to chat {chat_id} (dice roll or other condition)")

    logger.info(f"Done! Checked {len(alive_chats)} chats, sent {sent_count} messages")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trigger bot wakeup for spontaneous messages')
    parser.add_argument('--force-response', action='store_true',
                        help='Force the bot to respond (skip dice roll, tell model it must respond)')
    args = parser.parse_args()

    asyncio.run(trigger_wakeup(force_response=args.force_response))
