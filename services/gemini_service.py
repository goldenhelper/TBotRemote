from google import genai
import logging
from typing import Tuple, List, Dict
import json
from utils.constants import BOT_USER_DESCRIPTION, thinking_models_tokens, CLAUDE_MAX_TOKENS
from services.base_service import BaseAIService, log
from google.genai.types import Content, Part, Blob, GenerateContentConfig, ThinkingConfig
from models import ChatMessage
import io
from PIL import Image
import base64
import asyncio
from google.genai import errors as genai_errors
import re

logger = logging.getLogger(__name__)

DEBUG = True

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# TODO: add thinking to the response

class GeminiService(BaseAIService):
    def __init__(self, api_key: str, model_name: str, thinking_model: bool = False, video_analyzer=None) -> None:
        """Initializes the GeminiService class.
        
        Args:
            api_key: str: The API key for the Gemini API.
            model_name: str: The name of the Gemini model to use.
            thinking_model: bool: Whether to use the thinking model.
            video_analyzer: Optional video analysis service (e.g., another GeminiService instance)
                          If provided, will be used for video analysis capabilities.
        """
        super().__init__(api_key, model_name)
        self.client = genai.Client(api_key=api_key)
        self.output_tokens_multiplier = 4
        self.system_prompt = None
        self.thinking_model = thinking_model
        # For compatibility with memory updater logic
        self.thinking_tokens = 1 if thinking_model else None

        # Detect if the model natively supports video content (Gemini 2.5 lineage)
        # Only enable inline video for models that can handle it in complex conversation contexts
        # gemini-2.5-flash-preview-05-20 works for isolated video analysis but fails in chat context
        self.supports_video = model_name.startswith("gemini-2.5") and not model_name.startswith("gemini-2.5-flash-preview")
        
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
                # Fallback to basic analysis using this service's own client
                logger.info(f"Using built-in Gemini video analysis")
                
                contents = [Content(
                    role='user',
                    parts=[
                        Part(text="Analyze this video/animation content."),
                        Part(inline_data=Blob(data=video_data, mime_type=mime_type))
                    ]
                )]
                
                logger.info(f"Sending request to video analysis model '{self.model_name}' with system prompt: {system_prompt}")
                
                config = GenerateContentConfig(system_instruction=system_prompt)
                
                response = None
                max_retries = 3
                delay = 2
                for attempt in range(max_retries):
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            config=config,
                            contents=contents
                        )
                        break  # success
                    except genai_errors.ServerError as se:
                        if ('503' in str(se)) and attempt < max_retries - 1:
                            logger.warning(f"Gemini 503 UNAVAILABLE on video analysis, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(delay)
                            delay *= 2
                            continue
                        else:
                            raise
                if response is None:
                    raise RuntimeError("Failed to get response from Gemini after retries")
                
                analysis_result = response.text
                logger.info(f"Video analysis completed: {analysis_result[:100]}...")
                return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing video: {e}", exc_info=True)
            return f"[Video analysis failed: {str(e)}]"

    async def get_response(self, system_prompt: str, *, context_messages: List[ChatMessage] | None = None, query_without_context: str | None = None) -> Tuple[str, int, int, str | None, Dict[int, str] | None]:
        """
        Makes a request to the Gemini model, supporting multimodal chat history.
        
        Returns:
            Tuple containing:
            - response_text: The generated response
            - input_tokens: Number of input tokens
            - output_tokens: Number of output tokens  
            - thinking: Thinking output if available
            - video_analysis_results: Dict mapping message_id to analysis text for any videos analyzed during fallback
        """
        if not (context_messages is None or query_without_context is None):
            raise ValueError("Either context_messages or query_without_context must be given")

        try:
            contents = []
            if context_messages is not None:
                for msg in context_messages:
                    role = 'model' if msg.user == BOT_USER_DESCRIPTION else 'user'
                    
                    # Debug logging for animation messages
                    if msg.media_type == 'animation':
                        logger.info(f"Processing animation msg {msg.message_id}: content='{msg.content}', media_description='{msg.media_description}'")
                    
                    final_parts = []
                    
                    # Build the message prefix with user info (like format_messages does)
                    base_format = f"message_id: {msg.message_id}, reply_to_id: {msg.reply_to_id}, {msg.user}[{msg.timestamp}]"
                    
                    # Build text content with user info prefix
                    if msg.media_type and msg.media_description:
                        text_content = f"{base_format} {msg.media_description}:\n{msg.content or ''}"
                    else:
                        text_content = f"{base_format}:\n{msg.content or ''}"

                    # Add the text part if it's not empty
                    if text_content.strip():
                        final_parts.append(Part(text=text_content))

                    # Only add binary media data for images, stickers, and—if supported—animations/videos
                    if msg.media_type in ['image', 'sticker'] and msg.media_data:
                        try:
                            # Use Pillow for images and stickers
                            img = Image.open(io.BytesIO(msg.media_data))
                            mime_type = f'image/{img.format.lower()}' if img.format else 'image/jpeg'
                            # Create an explicit Part from bytes and mime type
                            final_parts.append(Part(inline_data=Blob(data=msg.media_data, mime_type=mime_type)))
                        except Exception as e:
                            logger.error(f"Could not process media for msg {msg.message_id}: {e}", exc_info=True)
                            # Try to infer mime type
                            fallback_mime = 'image/jpeg'
                            try:
                                final_parts.append(Part(inline_data=Blob(data=msg.media_data, mime_type=fallback_mime)))
                            except Exception:
                                # If even that fails, fallback to text description so request is still valid
                                final_parts.append(Part(text=f"[{msg.media_type} could not be processed]"))
                    
                    # Inline animations/videos if the model supports it
                    if msg.media_type in ['animation', 'video'] and msg.media_data and self.supports_video:
                        try:
                            # Simple MIME detection
                            if msg.media_data[:3] == b'GIF':
                                anim_mime = 'image/gif'
                            else:
                                anim_mime = 'video/mp4'  # Telegram usually sends .mp4 animations

                            final_parts.append(Part(inline_data=Blob(data=msg.media_data, mime_type=anim_mime)))
                        except Exception as e:
                            logger.error(f"Failed to attach animation data for msg {msg.message_id}: {e}", exc_info=True)
                            # fallback to already prepared text part only
                    
                    # Filter out any empty parts (no text and no inline_data)
                    valid_parts = [p for p in final_parts if (getattr(p, 'text', None) and p.text.strip()) or (getattr(p, 'inline_data', None) is not None)]
                    if valid_parts:
                        contents.append(Content(role=role, parts=valid_parts))

            else: # Handle single, non-contextual query
                contents = [query_without_context]

            # Debug print: create a sanitized copy for logging WITHOUT mutating the original contents
            sanitized_contents_print = []
            for content_item in contents:
                if hasattr(content_item, 'parts'):
                    new_parts = []
                    for part in content_item.parts:
                        if getattr(part, 'inline_data', None):
                            size = len(getattr(part.inline_data, 'data', b'') or b'')
                            mime_type = getattr(part.inline_data, 'mime_type', 'unknown')
                            new_parts.append(Part(text=f"[Image data of size {size} bytes, mime_type={mime_type}]") )
                        else:
                            # Preserve textual part as-is
                            new_parts.append(Part(text=getattr(part, 'text', '')))
                    sanitized_contents_print.append(Content(role=content_item.role, parts=new_parts))
                else:
                    sanitized_contents_print.append(content_item)
            print("*"*100 + f"\nContents: {sanitized_contents_print}\n" + "*"*100)

            if self.system_prompt is None:
                self.system_prompt = system_prompt
            
            config = GenerateContentConfig(system_instruction=self.system_prompt)
            # Use thinking config for the main model
            if self.thinking_model:
                config.thinking_config = ThinkingConfig(include_thoughts=True)


            
            # Enhanced debug logging for media content
            for i, content_item in enumerate(contents):
                if hasattr(content_item, 'parts'):
                    logger.info(f"Content {i} role: {content_item.role}, parts count: {len(content_item.parts)}")
                    for j, part in enumerate(content_item.parts):
                        if getattr(part, 'inline_data', None):
                            size = len(getattr(part.inline_data, 'data', b'') or b'')
                            mime_type = getattr(part.inline_data, 'mime_type', 'unknown')
                            logger.info(f"  Part {j}: Binary data - {size} bytes, MIME: {mime_type}")
                        elif getattr(part, 'text', None):
                            text_preview = getattr(part, 'text', '')[:100]
                            logger.info(f"  Part {j}: Text - '{text_preview}{'...' if len(getattr(part, 'text', '')) > 100 else ''}'")
            
            if logger.isEnabledFor(logging.DEBUG):
                sanitized_contents = []
                # The contents can be a list of Content objects or a list with a single string.
                for content_item in contents:
                    if hasattr(content_item, 'parts'):
                        new_parts = []
                        for part in content_item.parts:
                            if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data') and part.inline_data.data:
                                size = len(part.inline_data.data)
                                mime_type = part.inline_data.mime_type
                                new_parts.append(Part(text=f"[Image data of size {size} bytes with mime_type {mime_type}]"))
                            else:
                                new_parts.append(part)
                        sanitized_contents.append(Content(role=content_item.role, parts=new_parts))
                    else:
                        # This handles the case where the content_item is just a string (e.g., query_without_context)
                        sanitized_contents.append(content_item)
                logger.debug(f"Request contents payload: {sanitized_contents}")

            response = None
            max_retries = 3
            delay = 2
            # Track if we've already attempted an image fallback conversion / removal
            image_fallback_attempted = False
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        config=config,
                        contents=contents
                    )
                    break  # success
                except genai_errors.ServerError as se:
                    # Temporary unavailability
                    if ('503' in str(se)) and attempt < max_retries - 1:
                        logger.warning(
                            f"Gemini 503 UNAVAILABLE on main model, retrying in {delay}s (attempt {attempt+1}/{max_retries})")
                        await asyncio.sleep(delay)
                        delay *= 2
                        continue
                    else:
                        raise
                except genai_errors.ClientError as ce:
                    # Handle invalid image arguments from Gemini
                    err_msg = str(ce)
                    if ("Unable to process input image" in err_msg) and not image_fallback_attempted:
                        logger.warning("Gemini could not process one or more images. Attempting JPEG conversion / placeholder fallback and retrying once.")

                        def _sanitize_parts(parts: List[Part]) -> List[Part]:
                            sanitized = []
                            for p in parts:
                                if getattr(p, 'inline_data', None) is not None:
                                    try:
                                        # Attempt to load image and convert to JPEG which Gemini reliably supports
                                        img_bytes = getattr(p.inline_data, 'data', None)
                                        if img_bytes:
                                            try:
                                                img = Image.open(io.BytesIO(img_bytes))
                                                jpeg_buf = io.BytesIO()
                                                img.convert('RGB').save(jpeg_buf, format='JPEG')
                                                sanitized.append(Part(inline_data=Blob(data=jpeg_buf.getvalue(), mime_type='image/jpeg')))
                                                continue
                                            except Exception as conv_exc:
                                                logger.debug(f"JPEG conversion failed: {conv_exc}")
                                        # If conversion failed, fall back to text placeholder
                                        sanitized.append(Part(text='[Image omitted due to processing error]'))
                                    except Exception as inner_exc:
                                        logger.debug(f"Image sanitization error: {inner_exc}")
                                        sanitized.append(Part(text='[Image omitted due to processing error]'))
                                else:
                                    sanitized.append(p)
                            return sanitized

                        # Rebuild contents with sanitized parts (no problematic inline images)
                        new_contents = []
                        for c in contents:
                            if hasattr(c, 'parts'):
                                new_contents.append(Content(role=c.role, parts=_sanitize_parts(c.parts)))
                            else:
                                new_contents.append(c)

                        contents = new_contents
                        image_fallback_attempted = True
                        # after adjusting, retry the same attempt index (do not increment attempt) with same delay
                        continue
                    else:
                        # Re-raise for other client errors or if fallback already attempted
                        raise
            
            if response is None:
                raise RuntimeError("Failed to get response from Gemini after retries")
            
            logger.debug(f"Full response object from Gemini: {response}")
            
            text = ""
            thinking_summary = None
            if self.thinking_model:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        thinking_summary = part.text
                    elif hasattr(part, 'text'):
                        text += part.text
            else:
                text = response.text

            prompt_tokens = response.usage_metadata.prompt_token_count
            candidates_tokens = response.usage_metadata.candidates_token_count
            
            if self.thinking_model and hasattr(response.usage_metadata, 'thoughts_token_count'):
                candidates_tokens += response.usage_metadata.thoughts_token_count

            logger.info(f"Gemini response parsed. Tokens - Prompt: {prompt_tokens}, Candidate: {candidates_tokens}.")
            logger.debug(f"Returning text snippet: {text[:100]}...")
            if thinking_summary:
                logger.debug(f"Returning thinking summary snippet: {thinking_summary[:100]}...")

            # Clean the response text before sending
            clean = re.sub(r'^\[(?:Image|GIF/Animation|Video|Sticker|Document):[^\]]*\]\s*', '', text, flags=re.I)
            return (
                clean,
                prompt_tokens,
                candidates_tokens,
                thinking_summary,
                None  # Gemini doesn't perform video analysis fallback in main conversation flow
            )
        except Exception as e:
            logger.error(f"Error getting Gemini response: {e}", exc_info=True)
            raise