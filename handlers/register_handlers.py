"""
Central place to register all bot handlers.
Both render_webhook.py and local_testing_supabase.py import this.
"""
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters


def register_handlers(application, message_handler):
    """Register all command, callback, and message handlers."""

    # Command handlers
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
    application.add_handler(CommandHandler("bot_tokens", message_handler.bot_tokens_command))
    application.add_handler(CommandHandler("clear_history", message_handler.clear_history_command))
    application.add_handler(CommandHandler("delete_chat", message_handler.delete_chat_command))
    application.add_handler(CommandHandler("get_notes", message_handler.get_notes_command))
    application.add_handler(CommandHandler("set_aliveness", message_handler.set_come_to_life_chance_command))
    application.add_handler(CommandHandler("how_alive", message_handler.get_come_to_life_chance_command))
    application.add_handler(CommandHandler("reset_model_data", message_handler.reset_model_data_command))
    application.add_handler(CommandHandler("add_admin", message_handler.add_admin_command))
    application.add_handler(CommandHandler("remove_admin", message_handler.remove_admin_command))
    application.add_handler(CommandHandler("list_admins", message_handler.list_admins_command))
    application.add_handler(CommandHandler("get_settings", message_handler.get_settings_command))
    application.add_handler(CommandHandler("set_setting", message_handler.set_setting_command))

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

    # Message handler (must be last)
    application.add_handler(MessageHandler(~filters.COMMAND, message_handler.handle_message))
