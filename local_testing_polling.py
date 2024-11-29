# local_testing_polling.py
# for testing polling


import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from config_local import LocalTestingConfig
from handlers.message_handler import MainHandler
from services.token_manager import TokenManager
from services.chat_history import ChatHistoryManager
from services.claude_service import ClaudeService
from cryptography.fernet import Fernet

def main():
    # uses run_polling

    config = LocalTestingConfig()
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

    application = Application.builder().token(config.bot_token).build()

    # Register handlers
    application.add_handler(CommandHandler("start", message_handler.start_command))
    application.add_handler(CommandHandler("help", message_handler.help_command))
    application.add_handler(CommandHandler("tokens_amount", message_handler.get_tokens_command))
    application.add_handler(CommandHandler("ask_for_tokens", message_handler.ask_for_tokens_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler.handle_message))


    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
