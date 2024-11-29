from anthropic import Anthropic
import logging
from typing import Tuple
import json

logger = logging.getLogger(__name__)

class ClaudeService:
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-latest"):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    async def get_response(self, text: str, system_prompt: str) -> Tuple[str, int, int]:
        try:
            input_tokens = self.client.beta.messages.count_tokens(
                betas=["token-counting-2024-11-01"],
                system=system_prompt,
                model=self.model,
                messages=[{"role": "user", "content": text}]
            )
            
            input_token_count = json.loads(input_tokens.model_dump_json())["input_tokens"]
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": text}]
            )
            
            return (
                response.content[0].text,
                input_token_count,
                response.usage.output_tokens
            )
        except Exception as e:
            logger.error(f"Error getting Claude response: {e}")
            raise