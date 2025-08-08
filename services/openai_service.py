from openai import AsyncOpenAI
import logging
from typing import Tuple, List, Dict
import json
import asyncio
from utils.constants import BOT_USER_DESCRIPTION, CLAUDE_MAX_TOKENS
from services.base_service import BaseAIService, log
from models import ChatMessage

logger = logging.getLogger(__name__)

DEBUG = True

# OpenAI reasoning models that have different behavior
OPENAI_REASONING_MODELS = [
    "gpt-5-2025-08-07"
]

class OpenAIService(BaseAIService):
    def __init__(self, api_key: str, model_name: str, video_analyzer=None) -> None:
        """Initializes the OpenAIService class.
        
        Args:
            api_key: str: The API key for the OpenAI API.
            model_name: str: The name of the OpenAI model to use (e.g., gpt-4o, gpt-4o-mini, o1-preview).
            video_analyzer: Optional video analysis service (e.g., GeminiService instance)
                          If provided, will be used for video analysis capabilities.
        """
        
        super().__init__(api_key, model_name)
        self.client = AsyncOpenAI(api_key=api_key)
        self.model_name = model_name
        self.output_tokens_multiplier = 3  # OpenAI typical multiplier
        # OpenAI models don't expose a separate 'thinking' budget like Claude/Gemini
        self.thinking_tokens: int | None = None
        
        # Inject video analyzer for video analysis capabilities
        self.video_analyzer = video_analyzer
        
        # Check if this is a reasoning model
        self.is_reasoning_model = model_name in OPENAI_REASONING_MODELS
        
        # Set max tokens based on model type
        if self.is_reasoning_model:
            self.max_tokens = 25000  # Reasoning models support higher token counts
        else:
            self.max_tokens = 4096   # Standard models

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

    async def get_response(self, system_prompt: str, *, context_messages: List[ChatMessage] | None = None, query_without_context: str | None = None) -> Tuple[str, int, int, str | None, Dict[int, str] | None]:
        """
        Makes a request to the OpenAI model, supporting multimodal chat history.
        """
        if not (context_messages is None or query_without_context is None):
            raise ValueError("Either context_messages or query_without_context must be given")

        try:
            if context_messages is not None:
                # Build multimodal message content
                formatted_messages = []
                
                # Add system message (except for reasoning models)
                if not self.is_reasoning_model:
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
                    if msg.media_type in ['image', 'sticker'] and msg.media_data:
                        try:
                            # Encode image to base64 for OpenAI
                            base64_data, media_type = self._encode_image_to_base64(msg.media_data)
                            
                            # Check file size - OpenAI has limits on media size
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
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{media_type};base64,{base64_data}",
                                        "detail": "high"  # Use high detail for better analysis
                                    }
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
                    
                    # For reasoning models, include system prompt in first user message
                    if self.is_reasoning_model and role == 'user' and len(formatted_messages) == 0:
                        text_content = f"{system_prompt}\n\n"
                    
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
                        # OpenAI expects content to be a string if only text, or array if multimodal
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
                if self.is_reasoning_model:
                    # Include system prompt in user message for reasoning models
                    formatted_messages = [{
                        "role": "user", 
                        "content": f"{system_prompt}\n\n{query_without_context}"
                    }]
                else:
                    formatted_messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query_without_context}
                    ]
            
            logger.info(f"Sending request to OpenAI model '{self.model_name}' with multimodal content.")
            
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
                    # Use different parameters for reasoning vs regular models
                    if self.is_reasoning_model:
                        response = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=formatted_messages,
                            max_completion_tokens=self.max_tokens
                        )
                    else:
                        response = await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=formatted_messages,
                            max_tokens=self.max_tokens,
                            temperature=0.7
                        )
                    break  # success
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"OpenAI API error, retrying in {delay}s (attempt {attempt+1}/{max_retries}): {e}")
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    else:
                        raise
            
            if response is None:
                raise RuntimeError("Failed to get response from OpenAI after retries")

            logger.debug(f"OpenAI response: {response}")

            # Parse response content
            text = response.choices[0].message.content or ""
            thinking_message = None
            
            # For reasoning models, extract reasoning if available
            if self.is_reasoning_model and hasattr(response.choices[0].message, 'reasoning'):
                thinking_message = response.choices[0].message.reasoning

            # Get token usage
            input_token_count = response.usage.prompt_tokens if response.usage else 0
            output_token_count = response.usage.completion_tokens if response.usage else 0

            logger.info(f"OpenAI response parsed. Tokens - Input: {input_token_count}, Output: {output_token_count}.")
            logger.debug(f"Returning text snippet: {text[:100]}...")
            if thinking_message:
                logger.debug(f"Returning thinking summary snippet: {thinking_message[:100]}...")

            return (
                text,
                input_token_count,
                output_token_count,
                thinking_message,
                None  # OpenAI doesn't perform video analysis fallback
            )
            
        except Exception as e:
            logger.error(f"Error getting OpenAI response: {e}", exc_info=True)
            raise 