from anthropic import Anthropic
import logging
from typing import Tuple, List
import json
from utils.constants import BOT_USER_DESCRIPTION, thinking_models_tokens, CLAUDE_MAX_TOKENS

logger = logging.getLogger(__name__)

# NOTE: claude max tokens is a костыль

DEBUG = True

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class ClaudeService:
    def __init__(self, api_key: str, model_name: str) -> None:
        """Initializes the ClaudeService class.
        params:
            api_key: str: The API key for the Claude API.
            model_name: str: The name of the Claude model to use.
            thinking_tokens: int: The number of tokens to use for thinking. If None, thinking is disabled.
        """
        
        self.client = Anthropic(api_key=api_key)
        if model_name in thinking_models_tokens:
            self.thinking_tokens = thinking_models_tokens[model_name]
            if model_name == "claude-3-7-sonnet-latest-extended-thinking":
                model_name = "claude-3-7-sonnet-latest"
        else:
            self.thinking_tokens = None
        self.model_name = model_name
        self.output_tokens_multiplier = 5

    @classmethod
    def format_messages(cls, context_messages: List[dict]) -> List[dict]:
        formatted_messages = []
        for msg in context_messages:
            context_type = msg['context_type']
            
            if msg['user'] == BOT_USER_DESCRIPTION:
                role = "assistant"
                content = ('[reply_chain]: ' if context_type == 'reply_chain' else '') + msg['content']
            else:
                role = "user"
                content = (
                    f"{msg['user']} [{context_type}][{msg['timestamp']}]: {msg['content']}"
                )

            formatted_messages.append({
                "role": role,
                "content": content
            })

        return formatted_messages
    

    async def get_response(self, system_prompt: str, *, context_messages: List[dict] | None = None, query_without_context: str | None = None) -> Tuple[str, int, int, str | None]:
        assert (context_messages is None) != (query_without_context is None), "Either context_messages or query_without_context must be given"

        try:
            if context_messages is not None:   
                formatted_messages = ClaudeService.format_messages(context_messages)
            else:
                formatted_messages = [{"role": "user", "content": query_without_context}]
            
            if self.thinking_tokens is not None:
                assert self.thinking_tokens < CLAUDE_MAX_TOKENS, "Thinking tokens must be less than CLAUDE_MAX_TOKENS"
                is_thinking = True
                thinking = {
                    "type": "enabled",
                    "budget_tokens": self.thinking_tokens
                    }
            else:
                is_thinking = False
                thinking = {
                    "type": "disabled"
                    }

            input_tokens = self.client.messages.count_tokens(
                system=system_prompt,
                thinking=thinking,
                model=self.model_name,
                messages=formatted_messages
            )
            
            input_token_count = json.loads(input_tokens.model_dump_json())["input_tokens"]
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=system_prompt,
                thinking=thinking,
                messages=formatted_messages
            )
        
            log("Claude response:", response.content)


            if is_thinking:
                if len(response.content) == 1:
                    raise Exception("Claude thinks but no thinking block returned")


                text = next((msg for msg in response.content if msg.type == "text")).text
                thinking_message = next((msg for msg in response.content if msg.type == "thinking")).thinking
                redacted_thinking = next((msg for msg in response.content if msg.type == "redacted_thinking"), None)
                if redacted_thinking:
                    redacted_thinking = redacted_thinking.redacted_thinking
                    log("Some thinking was redacted")
            else:
                text = response.content[0].text
                thinking_message = None

            return (
                text,
                input_token_count,
                response.usage.output_tokens,
                thinking_message
            )
            

        except Exception as e:
            logger.error(f"Error getting Claude response: {e}")
            raise