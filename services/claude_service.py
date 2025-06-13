from anthropic import Anthropic
import logging
from typing import Tuple, List, Dict
import json
import asyncio
from utils.constants import BOT_USER_DESCRIPTION, thinking_models_tokens, CLAUDE_MAX_TOKENS
from services.base_service import BaseAIService, log
from models import ChatMessage

logger = logging.getLogger(__name__)

# NOTE: claude max tokens is a костыль

DEBUG = True

class ClaudeService(BaseAIService):
    def __init__(self, api_key: str, model_name: str, video_analyzer=None) -> None:
        """Initializes the ClaudeService class.
        
        Args:
            api_key: str: The API key for the Claude API.
            model_name: str: The name of the Claude model to use.
            video_analyzer: Optional video analysis service (e.g., GeminiService instance)
                          If provided, will be used for video analysis capabilities.
        """
        
        super().__init__(api_key, model_name)
        self.client = Anthropic(api_key=api_key)
        if model_name in thinking_models_tokens:
            self.thinking_tokens = thinking_models_tokens[model_name]
            if model_name == "claude-3-7-sonnet-latest-extended-thinking":
                model_name = "claude-3-7-sonnet-latest"
        else:
            self.thinking_tokens = None
        self.model_name = model_name
        self.output_tokens_multiplier = 5
        
        # Inject video analyzer for video analysis capabilities
        self.video_analyzer = video_analyzer

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
                return f"[Video content analysis not supported - {len(video_data)} bytes, MIME: {mime_type}]"
            
        except Exception as e:
            logger.error(f"Error in video analysis: {e}", exc_info=True)
            return f"[Video analysis failed: {str(e)}]"

    async def get_response(self, system_prompt: str, *, context_messages: List[ChatMessage] | None = None, query_without_context: str | None = None) -> Tuple[str, int, int, str | None, Dict[int, str] | None]:
        """
        Makes a request to the Claude model, supporting multimodal chat history.
        """
        if not (context_messages is None or query_without_context is None):
            raise ValueError("Either context_messages or query_without_context must be given")

        try:
            if context_messages is not None:
                # Build multimodal message content
                formatted_messages = []
                
                for msg in context_messages:
                    role = 'assistant' if msg.user == BOT_USER_DESCRIPTION else 'user'
                    
                    # Debug logging for media messages
                    if msg.media_type:
                        logger.info(f"Processing {msg.media_type} msg {msg.message_id}: content='{msg.content}', media_description='{msg.media_description}'")
                    
                    content_parts = []
                    
                    # Handle media content first
                    if msg.media_type in ['image', 'sticker'] and msg.media_data:
                        try:
                            # Encode image to base64 for Claude
                            base64_data, media_type = self._encode_image_to_base64(msg.media_data)
                            
                            # Check file size - Claude has limits on media size
                            if len(msg.media_data) > 20 * 1024 * 1024:  # 20MB limit
                                logger.warning(f"Media file too large ({len(msg.media_data)} bytes), using description only for message {msg.message_id}")
                                # Fall back to text description
                                if msg.media_description:
                                    content_parts.append({
                                        "type": "text",
                                        "text": msg.media_description
                                    })
                            else:
                                # Add image content
                                content_parts.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data,
                                    },
                                })
                                logger.info(f"Added image data for message {msg.message_id}: {len(base64_data)} base64 chars, MIME: {media_type}")
                        except Exception as e:
                            logger.error(f"Could not process media for msg {msg.message_id}: {e}", exc_info=True)
                            # Fall back to text description
                            if msg.media_description:
                                content_parts.append({
                                    "type": "text",
                                    "text": f"[{msg.media_type} could not be processed: {msg.media_description}]"
                                })
                    
                    # Handle text content (including media descriptions for non-image media)
                    text_content = ""
                    
                    # Build the message prefix with user info (like format_messages does)
                    base_format = f"message_id: {msg.message_id}, reply_to_id: {msg.reply_to_id}, {msg.user}[{msg.timestamp}]"
                    
                    # Build message content with user info prefix
                    if msg.media_type and msg.media_type not in ['image', 'sticker'] and msg.media_description:
                        text_content = f"{base_format} {msg.media_description}:\n{msg.content or ''}"
                    else:
                        text_content = f"{base_format}:\n{msg.content or ''}"
                    
                    # Add text content if we have any
                    if text_content.strip():
                        content_parts.append({
                            "type": "text",
                            "text": text_content
                        })
                    
                    # Only add the message if we have content
                    if content_parts:
                        formatted_messages.append({
                            "role": role,
                            "content": content_parts
                        })
                        
                        # Debug logging
                        logger.info(f"Added message {msg.message_id} with {len(content_parts)} content parts")
                        for i, part in enumerate(content_parts):
                            if part["type"] == "image":
                                logger.info(f"  Part {i}: Image - MIME: {part['source']['media_type']}")
                            elif part["type"] == "text":
                                text_preview = part["text"][:100]
                                logger.info(f"  Part {i}: Text - '{text_preview}{'...' if len(part['text']) > 100 else ''}'")
                
            else: 
                # Handle single, non-contextual query
                formatted_messages = [{"role": "user", "content": query_without_context}]
            
            # Set up thinking configuration
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

            # Count input tokens
            try:
                input_tokens = self.client.messages.count_tokens(
                    system=system_prompt,
                    thinking=thinking,
                    model=self.model_name,
                    messages=formatted_messages
                )
                input_token_count = json.loads(input_tokens.model_dump_json())["input_tokens"]
            except Exception as e:
                logger.warning(f"Could not count input tokens: {e}, using estimate")
                input_token_count = 0  # fallback
            
            logger.info(f"Sending request to Claude model '{self.model_name}' with multimodal content.")
            
            # Enhanced debug logging for media content
            for i, message in enumerate(formatted_messages):
                if isinstance(message.get("content"), list):
                    logger.info(f"Message {i} role: {message['role']}, parts count: {len(message['content'])}")
                    for j, part in enumerate(message["content"]):
                        if part["type"] == "image":
                            logger.info(f"  Part {j}: Image - MIME: {part['source']['media_type']}")
                        elif part["type"] == "text":
                            text_preview = part["text"][:100]
                            logger.info(f"  Part {j}: Text - '{text_preview}{'...' if len(part['text']) > 100 else ''}'")
            
            # Make the API call with retry logic
            response = None
            max_retries = 3
            delay = 2
            for attempt in range(max_retries):
                try:
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=CLAUDE_MAX_TOKENS,
                        system=system_prompt,
                        thinking=thinking,
                        messages=formatted_messages
                    )
                    break  # success
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Claude API error, retrying in {delay}s (attempt {attempt+1}/{max_retries}): {e}")
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    else:
                        raise
            
            if response is None:
                raise RuntimeError("Failed to get response from Claude after retries")

            logger.debug(f"Claude response: {response.content}")

            # Parse response content
            if is_thinking:
                text = ""
                thinking_message = None
                
                for content_block in response.content:
                    if content_block.type == "text":
                        text += content_block.text
                    elif content_block.type == "thinking":
                        thinking_message = content_block.thinking
                
                if not text and not thinking_message:
                    raise Exception("Claude thinking model returned no content")
            else:
                text = ""
                thinking_message = None
                
                for content_block in response.content:
                    if content_block.type == "text":
                        text += content_block.text

            logger.info(f"Claude response parsed. Tokens - Input: {input_token_count}, Output: {response.usage.output_tokens}.")
            logger.debug(f"Returning text snippet: {text[:100]}...")
            if thinking_message:
                logger.debug(f"Returning thinking summary snippet: {thinking_message[:100]}...")

            return (
                text,
                input_token_count,
                response.usage.output_tokens,
                thinking_message,
                None  # Claude doesn't perform video analysis fallback
            )
            
        except Exception as e:
            logger.error(f"Error getting Claude response: {e}", exc_info=True)
            raise