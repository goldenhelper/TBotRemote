from google import genai
import logging
from typing import Tuple, List
import json
from utils.constants import BOT_USER_DESCRIPTION

logger = logging.getLogger(__name__)

DEBUG = True

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# TODO: add thinking to the response

class GeminiService:
    def __init__(self, api_key: str, model_name: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.output_tokens_multiplier = 4
        self.system_prompt = None

    @classmethod
    def format_messages(cls, context_messages: List[dict]) -> str:
        """
        Format a list of context messages into a single string.
        
        Args:
            context_messages: List of message dictionaries containing user, context_type, timestamp, and content
            
        Returns:
            A formatted string with all messages concatenated with newlines
        """
        formatted_messages = ""
        for msg in context_messages:
            context_type = msg['context_type']
            content = f"{msg['user']} [{context_type}][{msg['timestamp']}]: {msg['content']}"
            formatted_messages += content + "\n"
    
        return formatted_messages


    async def get_response(self, system_prompt: str, *, context_messages: List[dict] | None, query_without_context = str | None) -> Tuple[str, int, int]:
        """
        Makes a request to the Gemini model.

        Args:
            system_prompt (str): The system prompt to use. If different from the previous prompt, a new model is created.
            context_messages (List[dict]): The context messages to give to the model.
            query_without_context (str): The query to give to the model if context_messages is None.

        Returns:
            Tuple[str, int, int]: The response text, the number of tokens used for the prompt, and the number of tokens used for the response.
        """

        assert context_messages is None or query_without_context is None 

        try:
            if context_messages is not None:
                formatted_messages = GeminiService.format_messages(context_messages)
            else:
                formatted_messages = query_without_context

            if self.system_prompt is None:
                self.system_prompt = system_prompt

            response = self.client.models.generate_content(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt),
                contents=formatted_messages
            )
            
            return (
                response.text,
                response.usage_metadata.prompt_token_count,
                response.usage_metadata.candidates_token_count
            )
        except Exception as e:
            logger.error(f"Error getting Gemini response: {e}")
            raise