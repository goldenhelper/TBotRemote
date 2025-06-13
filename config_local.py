from pydantic_settings import BaseSettings, SettingsConfigDict

class LocalTestingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    
    bot_token: str
    claude_api_key: str
    gemini_api_key: str
    openai_api_key: str
    admin_user_id: int 
    aws_access_key_id: str
    aws_secret_access_key: str
    default_role_id: str
    aws_region: str = 'eu-north-1'
    token_table_name: str = 'telegram-bot-tokens'
    chat_history_table: str = 'telegram-bot-chat-history'
    role_table_name: str = 'telegram-bot-roles'
    default_model: str = 'gemini-2.5-flash-preview-05-20'
    default_memory_updater_model: str = 'gemini-2.5-flash-preview-05-20'
    max_num_roles: int = 10
    max_role_name_length: int = 100

    def __init__(self):
        super().__init__()
        


