"""
Supabase configuration.
Only contains secrets and connection info - all other settings are in Supabase.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class SupabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Telegram
    bot_token: str
    admin_user_id: int  # Primary admin, will be auto-added to Supabase admin list

    # AI API Keys
    claude_api_key: str
    gemini_api_key: str
    openai_api_key: str

    # Supabase
    supabase_url: str
    supabase_key: str  # Use service_role key for full access

    # Alert bot (optional)
    alertobot_token: str = ""  # Token for alert bot notifications
