from openai import AsyncOpenAI
import logging
from typing import Tuple, List, Dict
import asyncio
from utils.constants import BOT_USER_DESCRIPTION
from services.base_service import BaseAIService, log
from models import ChatMessage

logger = logging.getLogger(__name__)

DEBUG = True

# OpenRouter base URL for API requests
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter model identifiers
OPENROUTER_MODELS = [
    "kimi-k2",  # Short name for convenience
    "moonshotai/kimi-k2-0905",  # Full OpenRouter model ID
    "grok-4.1-fast",  # Short name
    "x-ai/grok-4.1-fast",  # Full OpenRouter model ID
    "deepseek-v3.2",  # Short name
    "deepseek/deepseek-v3.2",  # Full OpenRouter model ID
]

# Models that support vision/image input (add new vision models here)
VISION_CAPABLE_MODELS = {
    "x-ai/grok-4.1-fast",
}


class OpenRouterService(BaseAIService):
    def __init__(self, api_key: str, model_name: str, video_analyzer=None) -> None:
        """Initializes the OpenRouterService class.

        Args:
            api_key: str: The API key for the OpenRouter API.
            model_name: str: The name of the OpenRouter model to use (e.g., kimi-k2).
            video_analyzer: Optional video analysis service (e.g., GeminiService instance)
                          If provided, will be used for video analysis capabilities.
        """

        super().__init__(api_key, model_name)

        # OpenRouter uses OpenAI-compatible API
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://telegram-bot.app",  # Optional for rankings
                "X-Title": "Telegram AI Bot",  # Optional for rankings
            }
        )

        # Map short names to full OpenRouter model IDs
        self.openrouter_model = self._get_openrouter_model_id(model_name)
        self.output_tokens_multiplier = 3  # Similar to OpenAI
        self.thinking_tokens: int | None = None

        # Inject video analyzer for video analysis capabilities
        self.video_analyzer = video_analyzer

        # Max tokens for response
        self.max_tokens = 4096

    def _get_openrouter_model_id(self, model_name: str) -> str:
        """Map user-friendly model names to OpenRouter model IDs."""
        model_mapping = {
            "kimi-k2": "moonshotai/kimi-k2-0905",
            "grok-4.1-fast": "x-ai/grok-4.1-fast",
            "deepseek-v3.2": "deepseek/deepseek-v3.2",
        }
        return model_mapping.get(model_name, model_name)

    def _supports_vision(self) -> bool:
        """Check if the current model supports vision/image input."""
        return self.openrouter_model in VISION_CAPABLE_MODELS

    async def analyze_video(self, video_data: bytes, mime_type: str, system_prompt: str) -> str:
        """
        Analyze video content. If a video analyzer is available, use it; otherwise provide fallback.

        Args:
            video_data: Raw video bytes
            mime_type: MIME type of the video
            system_prompt: System prompt for analysis including role context

        Returns:
            str: Video analysis description
        """
        try:
            if self.video_analyzer and hasattr(self.video_analyzer, 'analyze_video'):
                logger.info(f"Delegating video analysis to {type(self.video_analyzer).__name__}")
                return await self.video_analyzer.analyze_video(video_data, mime_type, system_prompt)
            else:
                logger.info(f"No video analyzer available. Providing fallback description.")
                return f"The video couldn't be analyzed."

        except Exception as e:
            logger.error(f"Error in video analysis: {e}", exc_info=True)
            return f"The video couldn't be analyzed."

    async def get_response(self, system_prompt: str, *, context_messages: List[ChatMessage] | None = None, query_without_context: str | None = None, flat_history: bool = False) -> Tuple[str, int, int, str | None, Dict[int, str] | None]:
        """
        Makes a request to the OpenRouter model, supporting multimodal chat history.
        """
        if not (context_messages is None or query_without_context is None):
            raise ValueError("Either context_messages or query_without_context must be given")

        try:
            if context_messages is not None:
                # Build multimodal message content
                formatted_messages = []

                # Add system message
                formatted_messages.append({
                    "role": "system",
                    "content": system_prompt
                })

                for msg in context_messages:
                    role = 'assistant' if msg.user == BOT_USER_DESCRIPTION else 'user'

                    # Debug logging for media messages
                    if msg.media_type:
                        logger.info(f"Processing {msg.media_type} msg {msg.message_id}: content='{msg.content}', media_description='{msg.media_description}'")

                    content_parts = []

                    # Handle media content first
                    if msg.media_type in ['image', 'sticker']:
                        if self._supports_vision() and msg.media_data:
                            # Model supports vision - send image data
                            try:
                                base64_data, media_type = self._encode_image_to_base64(msg.media_data)

                                if len(msg.media_data) > 20 * 1024 * 1024:  # 20MB limit
                                    logger.warning(f"Media file too large ({len(msg.media_data)} bytes), using description only for message {msg.message_id}")
                                    if msg.media_description:
                                        content_parts.append({
                                            "type": "text",
                                            "text": msg.media_description
                                        })
                                else:
                                    content_parts.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{media_type};base64,{base64_data}",
                                            "detail": "high"
                                        }
                                    })
                                    logger.info(f"Added image data for message {msg.message_id}: {len(base64_data)} base64 chars, MIME: {media_type}")
                            except Exception as e:
                                logger.error(f"Could not process media for msg {msg.message_id}: {e}", exc_info=True)
                                if msg.media_description:
                                    content_parts.append({
                                        "type": "text",
                                        "text": f"[{msg.media_type} could not be processed: {msg.media_description}]"
                                    })
                        elif msg.media_description:
                            # Model doesn't support vision - use text description
                            content_parts.append({
                                "type": "text",
                                "text": msg.media_description
                            })
                            logger.info(f"Using text description for {msg.media_type} in message {msg.message_id} (model doesn't support vision)")

                    # Handle text content (including media descriptions for non-image media)
                    text_content = ""

                    # Build the message prefix with user info (like format_messages does)
                    base_format = f"message_id: {msg.message_id}, reply_to_id: {msg.reply_to_id}, {msg.user}[{msg.timestamp}]"

                    # Build message content with user info prefix
                    if msg.media_type and msg.media_type not in ['image', 'sticker'] and msg.media_description:
                        text_content += f"{base_format} {msg.media_description}:\n{msg.content or ''}"
                    else:
                        text_content += f"{base_format}:\n{msg.content or ''}"

                    # Add text content if we have any
                    if text_content.strip():
                        content_parts.append({
                            "type": "text",
                            "text": text_content
                        })

                    # Only add the message if we have content
                    if content_parts:
                        # OpenRouter expects content to be a string if only text, or array if multimodal
                        if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                            message_content = content_parts[0]["text"]
                        else:
                            message_content = content_parts

                        formatted_messages.append({
                            "role": role,
                            "content": message_content
                        })

                        # Debug logging
                        logger.info(f"Added message {msg.message_id} with {len(content_parts)} content parts")
                        for i, part in enumerate(content_parts):
                            if part["type"] == "image_url":
                                logger.info(f"  Part {i}: Image - URL length: {len(part['image_url']['url'])}")
                            elif part["type"] == "text":
                                text_preview = part["text"][:100]
                                logger.info(f"  Part {i}: Text - '{text_preview}{'...' if len(part['text']) > 100 else ''}'")

            else:
                # Handle single, non-contextual query
                formatted_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_without_context}
                ]

            logger.info(f"Sending request to OpenRouter model '{self.openrouter_model}' with multimodal content.")

            # Enhanced debug logging for media content
            for i, message in enumerate(formatted_messages):
                content = message.get("content")
                if isinstance(content, list):
                    logger.info(f"Message {i} role: {message['role']}, parts count: {len(content)}")
                    for j, part in enumerate(content):
                        if part["type"] == "image_url":
                            logger.info(f"  Part {j}: Image - URL length: {len(part['image_url']['url'])}")
                        elif part["type"] == "text":
                            text_preview = part["text"][:100]
                            logger.info(f"  Part {j}: Text - '{text_preview}{'...' if len(part['text']) > 100 else ''}'")
                elif isinstance(content, str):
                    text_preview = content[:100]
                    logger.info(f"Message {i} role: {message['role']}, text: '{text_preview}{'...' if len(content) > 100 else ''}'")

            # Make the API call with retry logic
            response = None
            max_retries = 3
            delay = 2
            for attempt in range(max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.openrouter_model,
                        messages=formatted_messages,
                        max_tokens=self.max_tokens,
                        temperature=0.7
                    )
                    break  # success
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"OpenRouter API error, retrying in {delay}s (attempt {attempt+1}/{max_retries}): {e}")
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    else:
                        raise

            if response is None:
                raise RuntimeError("Failed to get response from OpenRouter after retries")

            logger.debug(f"OpenRouter response: {response}")

            # Parse response content
            text = response.choices[0].message.content or ""
            thinking_message = None

            # Get token usage
            input_token_count = response.usage.prompt_tokens if response.usage else 0
            output_token_count = response.usage.completion_tokens if response.usage else 0

            logger.info(f"OpenRouter response parsed. Tokens - Input: {input_token_count}, Output: {output_token_count}.")
            logger.debug(f"Returning text snippet: {text[:100]}...")

            return (
                text,
                input_token_count,
                output_token_count,
                thinking_message,
                None  # OpenRouter doesn't perform video analysis fallback
            )

        except Exception as e:
            logger.error(f"Error getting OpenRouter response: {e}", exc_info=True)
            raise
