"""
Local testing script using Supabase instead of AWS.
Uses polling mode for development.
"""
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)
from config_supabase import SupabaseConfig
from handlers.message_handler import MainHandler
from services.supabase_storage import SupabaseStorage
from services.supabase_model_manager import SupabaseModelManager
from services.supabase_role_manager import SupabaseRoleManager
import logging


# Default formatting info (you can customize this or load from Supabase config table)
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

    # Initialize Supabase services
    model_manager = SupabaseModelManager(
        supabase_url=config.supabase_url,
        supabase_key=config.supabase_key,
        default_model=config.default_model,
        default_role_id=config.default_role_id,
        default_memory_updater_model=config.default_memory_updater_model,
        default_tokens_for_new_chats=config.default_tokens_for_new_chats,
    )

    role_manager = SupabaseRoleManager(
        supabase_url=config.supabase_url,
        supabase_key=config.supabase_key,
        default_role_id=config.default_role_id,
    )

    storage = SupabaseStorage(
        supabase_url=config.supabase_url,
        supabase_key=config.supabase_key,
        default_model=config.default_model,
        default_role_id=config.default_role_id,
        role_manager=role_manager,
        default_memory_updater_model=config.default_memory_updater_model,
        default_come_to_life_chance=config.default_come_to_life_chance,
        default_tokens_for_new_chats=config.default_tokens_for_new_chats,
    )

    # You can load formatting_info from Supabase config table if needed:
    # result = storage.client.table('config').select('value').eq('key', 'formatting_info').execute()
    # formatting_info = result.data[0]['value'] if result.data else DEFAULT_FORMATTING_INFO
    formatting_info = DEFAULT_FORMATTING_INFO

    bot_id = config.bot_token.split(':')[0]

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
        formatting_info=formatting_info,
        bot_id=int(bot_id),
        aws_region='',  # Not used with Supabase
        max_num_roles=config.max_num_roles,
        max_role_name_length=config.max_role_name_length,
        video_analyzer_model=config.video_analyzer_model,
    )

    application = ApplicationBuilder().token(config.bot_token).build()

    # Register handlers
    application.add_handler(CommandHandler("start", message_handler.start_command))
    application.add_handler(CommandHandler("help", message_handler.help_command))
    application.add_handler(CommandHandler("tokens_amount", message_handler.get_tokens_command))
    application.add_handler(CommandHandler("ask_for_tokens", message_handler.ask_for_tokens_command))

    application.add_handler(CommandHandler("choose_model", message_handler.choose_model_command))
    application.add_handler(CommandHandler("get_model_data", message_handler.get_model_data_command))
    application.add_handler(CommandHandler("get_role", message_handler.get_role_command))
    application.add_handler(CommandHandler("choose_role", message_handler.choose_role_command))
    application.add_handler(CommandHandler("add_role", message_handler.add_role_command))
    application.add_handler(CommandHandler("remove_role", message_handler.remove_role_command))
    application.add_handler(CommandHandler("give_bot_tokens", message_handler.give_bot_tokens_command))
    application.add_handler(CommandHandler("clear_history", message_handler.clear_chat_history_command))
    application.add_handler(CommandHandler("get_notes", message_handler.get_notes_command))
    application.add_handler(CommandHandler("set_aliveness", message_handler.set_come_to_life_chance_command))
    application.add_handler(CommandHandler("how_alive", message_handler.get_come_to_life_chance_command))
    application.add_handler(CommandHandler("reset_model_data", message_handler.reset_model_data_command))

    # Callback query handlers
    application.add_handler(CallbackQueryHandler(
        message_handler.choose_role_button_callback, pattern="^choose_role_"
    ))
    application.add_handler(CallbackQueryHandler(
        message_handler.remove_role_button_callback, pattern="^remove_role_"
    ))
    application.add_handler(CallbackQueryHandler(
        message_handler.choose_model_button_callback, pattern="^choose_model_"
    ))

    application.add_handler(MessageHandler(~filters.COMMAND, message_handler.handle_message))

    print("Bot starting with Supabase backend...")
    print(f"Supabase URL: {config.supabase_url}")

    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
