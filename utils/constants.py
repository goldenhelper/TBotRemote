BOT_USER_DESCRIPTION = "assistant"
CHANCE_TO_COME_ALIVE = 0.1
CHAT_HISTORY_DEPTH = 2000
SHORT_TERM_MEMORY = 25
CLAUDE_MAX_TOKENS = 2048
DEFAULT_COME_TO_LIFE_CHANCE = 0.1
DEFAULT_TOKENS_FOR_NEW_USERS = 6000

thinking_models_tokens = {  
    'claude-4-sonnet-latest-extended-thinking': 2048
}

gemini_thinking_models = [
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-05-20"
]

# Default allowed models limits used as a fallback if SSM configuration is unavailable
ALLOWED_MODELS_LIMITS = {
    "gemini": {
        "gemini-2.5-flash-preview-05-20": 500,
        "gemini-2.5-pro-preview-06-05": 100,
    },
    "claude": {
        "claude-4-opus-latest": 1000,
        "claude-4-sonnet-latest": 1000,
    },
    "openai": {
        "gpt-5-2025-08-07": 125,
    },
    "openrouter": {
        "kimi-k2": 500,
    },
}