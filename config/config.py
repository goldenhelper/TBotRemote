from pydantic_settings import BaseSettings, SettingsConfigDict

class LambdaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')
    
    bot_token: str
    claude_api_key: str
    admin_user_id: str
    decryption_key: str
    dynamodb_region: str = 'eu-north-1'
    token_table_name: str = 'telegram-bot-tokens'
    chat_history_table: str = 'telegram-bot-chat-history'
