from typing import List, Tuple, Union, Dict, Any
import logging
from models import ChatMessage

logger = logging.getLogger(__name__)

DEBUG = True

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class BaseAIService:
    """Base class for AI service implementations (Gemini, Claude, etc.)"""
    
    def __init__(self, api_key: str, model_name: str) -> None:
        """Initialize the base AI service.
        
        Args:
            api_key: API key for the service
            model_name: Name of the model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self.output_tokens_multiplier = 1  # Default value, should be overridden by subclasses
        
    @classmethod
    def format_messages(cls, context_messages: List[ChatMessage]) -> str:
        """
        Format a list of context messages into a single string.
        
        Args:
            context_messages: List of messages (ChatMessage objects)
            
        Returns:
            A formatted string with all messages concatenated with newlines
        """
        formatted_messages = ""

        for msg in context_messages:
            # Build the base message format
            base_format = f"message_id: {msg.message_id}, reply_to_id: {msg.reply_to_id}, {msg.user}[{msg.timestamp}]"
            
            # Include media description if available
            if msg.media_type and msg.media_description:
                content = f"{base_format} {msg.media_description}:\n{msg.content}"
            else:
                content = f"{base_format}:\n{msg.content}"
            
            formatted_messages += content + "\n"
    
        return formatted_messages
    
    async def get_response(self, system_prompt: str, *, context_messages: List[dict] | None = None, query_without_context: str | None = None, flat_history: bool = False) -> Tuple[str, int, int, str | None, Dict[int, str] | None]:
        """
        Get a response from the AI model.
        
        Args:
            system_prompt: System prompt to guide the model
            context_messages: List of message dictionaries for context
            query_without_context: Query string without context
            
        Returns:
            Tuple containing:
            - response text
            - input token count
            - output token count
            - thinking text (if applicable)
            - video analysis results mapping message_id to analysis text (if any)
            
        Note: This method should be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement get_response method")

    @staticmethod
    def _encode_image_to_base64(image_data: bytes) -> Tuple[str, str]:
        """Encode raw image bytes into base64 and guess its media type.

        Args:
            image_data: Bytes of the original image file.

        Returns
        -------
        Tuple[str, str]
            1. The base-64 encoded data (utf-8 string).
            2. The best-guess MIME type, e.g. ``image/png``.

        The helper relies on Pillow to detect the image format.  When the
        format cannot be determined we fall back to ``image/jpeg`` so that
        downstream services still receive a sensible default.
        """
        # Local import to avoid the hard dependency if consumers never call
        # this helper.
        import base64
        import io
        from PIL import Image

        try:
            img = Image.open(io.BytesIO(image_data))
            if img.format:
                media_type = f"image/{img.format.lower()}"
            else:
                media_type = "image/jpeg"
        except Exception as e:
            logger.warning(f"Could not determine image format: {e}, defaulting to jpeg")
            media_type = "image/jpeg"

        base64_data = base64.standard_b64encode(image_data).decode("utf-8")
        return base64_data, media_type
