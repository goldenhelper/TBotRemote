"""
Supabase configuration for local testing.
Copy this to config_local.py or use it directly.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class SupabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Telegram
    bot_token: str
    admin_user_id: int

    # AI API Keys
    claude_api_key: str
    gemini_api_key: str
    openai_api_key: str

    # Supabase
    supabase_url: str
    supabase_key: str  # Use service_role key for full access

    # Defaults
    default_role_id: str
    default_model: str = 'gemini-3-flash-preview'
    default_memory_updater_model: str = 'gemini-3-flash-preview'
    video_analyzer_model: str = 'gemini-3-flash-preview'
    default_come_to_life_chance: float = 0.1

    # Limits
    max_num_roles: int = 10
    max_role_name_length: int = 100
    default_tokens_for_new_chats: int = 15000

    def __init__(self):
        super().__init__()
