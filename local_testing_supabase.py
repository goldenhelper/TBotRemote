"""
Local testing script using Supabase instead of AWS.
Uses polling mode for development.
"""
from telegram import Update
from telegram.ext import ApplicationBuilder
from telegram.request import HTTPXRequest
from config_supabase import SupabaseConfig
from handlers.message_handler import MainHandler
from handlers.register_handlers import register_handlers
from services.supabase_storage import SupabaseStorage
from services.supabase_model_manager import SupabaseModelManager
from services.supabase_role_manager import SupabaseRoleManager
import logging


# Default formatting info (you can customize this or load from Supabase config table)
DEFAULT_FORMATTING_INFO = """
=== CRITICAL: OUTPUT FORMAT ===

❌ NEVER output: message_id: 124150, reply_to_id: 124149, assistant[Fri, 13. Jun 2025 12:18]: Hi there!
✅ ONLY output: Hi there!

The metadata you see is ONLY for understanding. NEVER include it in responses.

=== MESSAGE FORMAT (FOR UNDERSTANDING ONLY) ===

Messages appear as: message_id: 12345, reply_to_id: 12344, John[Fri, 13. Jun 09:49]: Hello!
- This shows: ID 12345 from John, replying to 12344
- YOU output ONLY the content part
- reply_to_id shows reply chains
- You reply to the LAST message (highest ID)

=== MEDIA FORMATS ===

Media appears as:
- [Image: description] 
- [GIF/Animation: Xs, WxH] description
- [Video: Xs, WxH]
- [Sticker: emoji]
- [Document: file.pdf (type, size)]

Example: John[10:30] [Image: cat on laptop]: Work from home 😅
Means: John sent cat pic with that caption

=== CONVERSATION RULES ===

NEW TO CHAT:
- If you see only 1-2 messages, you were just added
- Introduce yourself naturally per your role
- Don't pretend you know prior context
- Acknowledge being new if appropriate

GROUP CHATS:
- Track who sends each message
- You reply to whoever sent LAST message
- Use names when addressing someone specific
- Watch reply chains for context

PRIVATE CHATS:
- One consistent person
- More personal tone appropriate

MEDIA:
- GIFs = reactions/memes
- Respond to emotion, not just describe
- Consider cultural context

MEMORY:
- <notes> = long-term info
- Recent messages override old notes
- Track relationships, preferences

=== YOUR RESPONSES ===

1. OUTPUT ONLY MESSAGE TEXT - NO METADATA
2. Natural conversation, no technical references
3. Stay in character per <role>
4. Address the right person
5. Acknowledge media + text together

ROLE VARIETY:
- Don't repeat exact phrases every time
- Vary your expressions while staying in character
- Natural speech includes variety - even characters evolve
- Catchphrases are fine occasionally, not every message
- Show personality through actions, not just repeated lines

REMEMBER: Output pure message text only. No IDs, timestamps, or formatting.
"""


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger('services').setLevel(logging.DEBUG)
    logging.getLogger('handlers').setLevel(logging.DEBUG)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.INFO)

    # Load config
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

    # You can load formatting_info from Supabase config table if needed:
    # result = storage.client.table('config').select('value').eq('key', 'formatting_info').execute()
    # formatting_info = result.data[0]['value'] if result.data else DEFAULT_FORMATTING_INFO
    formatting_info = DEFAULT_FORMATTING_INFO

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
        formatting_info=formatting_info,
        bot_id=int(bot_id),
        aws_region='',  # Not used with Supabase
        alertobot_token=config.alertobot_token,
    )

    # Configure request with longer timeouts
    request = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=20.0,
        read_timeout=20.0,
        write_timeout=20.0,
        pool_timeout=10.0,
    )

    application = (
        ApplicationBuilder()
        .token(config.bot_token)
        .request(request)
        .get_updates_request(request)
        .build()
    )

    # Register all handlers
    register_handlers(application, message_handler)

    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
