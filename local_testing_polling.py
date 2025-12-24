# local_testing_polling.py
# for testing polling

#TODO:
# think more deeply about the implementation of context messages, since some messages could be completely omitted
# if, for example, the chain of replies is too deep

# think of the token management for bot because tokens for his comint to life are shared between chats
# probably, logical to just use the tokens of the chat itself directly

# solve the thing with roles (there should be chat-specific set of available roles)
# when a user wants to add a role, the role should be scanned by an LLM and if flagged, sent to me for a confirmation

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from config_local import LocalTestingConfig
from handlers.message_handler import MainHandler
from services.model_manager import ModelManager
from services.role_manager import RoleManager, Role
from services.chat_history import FileStorage, AWSStorage   
import boto3
import logging

def main():
    # Set up basic logging to see INFO and DEBUG messages
    logging.basicConfig(
        level=logging.INFO, # Set a default level for all loggers
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Set the logging level for our application modules to DEBUG
    logging.getLogger('services').setLevel(logging.DEBUG)
    logging.getLogger('handlers').setLevel(logging.DEBUG)
    # Silence verbose third-party loggers
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('telegram').setLevel(logging.INFO)
    logging.getLogger('apscheduler').setLevel(logging.WARNING)

    # uses run_polling
    config = LocalTestingConfig()

    # Initialize services
    model_manager = ModelManager(
        token_table_name=config.token_table_name,
        model_usage_table_name=config.model_usage_table_name,
        allowed_models_limits_resource_name=getattr(config, 'allowed_models_limits_param', '/telegram-bot/allowed_model_limits'),
        region=config.aws_region
    )
    
    # default_system_prompt = "Ты - Омар Хайам. Отвечай на вопросы людей, только крайне заумными словами."
    formatting_info = boto3.client(
        'ssm',
        aws_access_key_id=config.aws_access_key_id,
        aws_secret_access_key=config.aws_secret_access_key,
        region_name=config.aws_region
    ).get_parameter(Name='/telegram-bot/system-prompt/formatting-info')['Parameter']['Value']

    
    role_manager = RoleManager(
        table_name=config.role_table_name, 
        aws_region=config.aws_region,
        default_role_id=config.default_role_id
    )


    # chat_history_manager = ChatHistoryManager(
    #     storage=AWSStorage(
    #         table_name=config.chat_history_table,
    #         region=config.aws_region,
    #         default_role=default_role,
    #         default_model=config.default_model
    #         default_memory_updater_model=config.default_memory_updater_model
    #     ),
    #     role_manager=role_manager
    # )

    storage = FileStorage(
        storage_dir="chat_history", 
        default_role_id=config.default_role_id,
        default_model=config.default_model, 
        default_memory_updater_model=config.default_memory_updater_model,
        role_manager=role_manager
    )
        

    bot_id = config.bot_token.split(':')[0]

    # Initialize message handler
    message_handler = MainHandler(
        model_manager=model_manager,
        storage=storage,
        role_manager=role_manager,
        api_keys={'gemini': config.gemini_api_key, 'claude': config.claude_api_key, 'openai': config.openai_api_key},
        admin_user_id=config.admin_user_id,
        formatting_info=formatting_info,
        bot_id=int(bot_id),
        aws_region=config.aws_region,
        max_num_roles=config.max_num_roles,
        max_role_name_length=config.max_role_name_length
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
    application.add_handler(CommandHandler("clear_history", message_handler.clear_history_command))
    application.add_handler(CommandHandler("delete_chat", message_handler.delete_chat_command))
    application.add_handler(CommandHandler("get_notes", message_handler.get_notes_command))
    application.add_handler(CommandHandler("set_aliveness", message_handler.set_come_to_life_chance_command))
    application.add_handler(CommandHandler("how_alive", message_handler.get_come_to_life_chance_command))
    application.add_handler(CommandHandler("reset_model_data", message_handler.reset_model_data_command))
    
    # Add callback query handler for role selection buttons
    application.add_handler(CallbackQueryHandler(message_handler.choose_role_button_callback, pattern="^choose_role_"))
    application.add_handler(CallbackQueryHandler(message_handler.remove_role_button_callback, pattern="^remove_role_"))
    application.add_handler(CallbackQueryHandler(message_handler.choose_model_button_callback, pattern="^choose_model_"))

    application.add_handler(MessageHandler(~filters.COMMAND, message_handler.handle_message))


    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
