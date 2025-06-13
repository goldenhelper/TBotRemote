from telegram import Update
from telegram.ext import ContextTypes
from services.image_service import ImageService
from services.gemini_service import GeminiService
from models import ChatMessage
import logging
from google import genai
from google.genai.types import Part
import base64
from utils.constants import BOT_USER_DESCRIPTION

logger = logging.getLogger(__name__)

class MediaHandler:
    """
    Handler for processing media messages in Telegram (images, videos, stickers, etc.)
    
    This class handles:
    - Image retrieval and storage
    - Video message processing
    - Sticker handling
    - Document and audio handling
    """
    
    def __init__(self, storage, *, video_analyzer=None, api_key=None):
        """
        Initialize the MediaHandler.

        Parameters
        ----------
        storage : Storage
            Storage layer for persisting chat messages.
        video_analyzer : BaseAIService | None, optional
            Pre-constructed service to analyse video/animation content.  If
            supplied, MediaHandler will reuse it instead of building its own
            GeminiService instance, ensuring a single shared analyser across
            the application.
        api_key : str | None, optional
            Gemini API key used to create a fallback analyser when
            ``video_analyzer`` is *not* provided.
        """
        self.storage = storage
        self.image_service = ImageService()

        # Prefer the injected analyser to avoid multiple GeminiService pools.
        if video_analyzer is not None:
            self.video_analyzer = video_analyzer
        elif api_key is not None:
            # Create a fresh GeminiService for video analysis.
            self.video_analyzer = GeminiService(
                api_key=api_key,
                model_name="gemini-2.5-flash-preview-05-20",  # Stable model for video analysis
                thinking_model=False,
            )
        else:
            self.video_analyzer = None
        
        # These will be set by the message handler
        self.current_role = None
        self.chat_type = None
        self.chat_notes = None
        self.main_supports_video = False  # Whether the primary chat model can take video inline

    def set_context(self, current_role, chat_type, chat_notes, main_supports_video: bool = False):
        """Set role, chat notes, chat type, and whether the main model supports inlining video."""
        self.current_role = current_role
        self.chat_notes = chat_notes
        self.chat_type = chat_type
        self.main_supports_video = main_supports_video

    def get_video_analysis_prompt(self, duration, width, height) -> str:
        """
        Generate a role-aware prompt for video analysis.
        
        Args:
            duration: Video duration
            width: Video width  
            height: Video height
            
        Returns:
            str: System prompt for video analysis
        """
        role_section = ""
        if self.current_role and hasattr(self.current_role, 'prompt'):
            role_section = f"""
            <main_bot_role>
            The following is the role/personality of the MAIN BOT that will use your description to respond to users:
            {self.current_role.prompt}
            </main_bot_role>
            """
        
        note_section = ""
        if self.chat_notes:
            note_section = f"""
            <previous_chat_notes>
            {self.chat_notes}
            </previous_chat_notes>
            """

        chat_context = ""
        if self.chat_type == 'private':
            chat_context = "This is a private chat between the user and the bot."
        elif self.chat_type == 'group':
            chat_context = "This is a group chat with multiple participants."
        
        system_prompt = f"""
            {role_section}
            {note_section}
            <task>
            You are a VIDEO ANALYZER providing factual descriptions for another AI model (the main bot). 
            Your ONLY job is to describe what you see in the video/animation.
            The main bot (described above) will use your description to respond to the user in its own character.
            You are NOT the main bot - you are just the analyzer providing visual information.
            Do NOT adopt the main bot's personality or respond to the user yourself.
            </task>

            <video_metadata>
            Duration: {duration}s
            Dimensions: {width}x{height}
            {chat_context}
            </video_metadata>

            <instructions>
            1. You are ONLY the video analyzer - not the main bot with the personality described above.
            2. Observe key visual elements: people, objects, actions, text, colours, lighting, atmosphere, movement.
            3. Mention important audio cues only if they change understanding.
            4. Avoid speculation beyond what is visible/audible.
            5. Write a neutral, factual description that helps the main bot understand the content.
            6. Do NOT address the user directly, provide commentary, reactions, or adopt the main bot's personality.
            7. Do NOT use meta-phrases such as "This video shows" or "Here is a description".
            8. Write 6–12 sentences (or more if needed) to capture all significant details; err on the side of thoroughness rather than brevity. Use the chat's language.
            </instructions>

            <output_format>
            Return ONLY the description text - no direct user address.
            </output_format>
            """
        
        return system_prompt
    
    async def handle_media_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> ChatMessage:
        """
        Process a media message and store it appropriately.
        
        Args:
            update (Update): Telegram update object
            context (ContextTypes.DEFAULT_TYPE): Telegram context
            
        Returns:
            ChatMessage: The processed message object, or None if no media was found
        """
        if not update.message:
            return None
            
        message = update.message
        chat_id = update.effective_chat.id
        
        # Create a message from the Telegram message first
        current_node = ChatMessage.from_telegram_message(message)
        
        # If there's no media, return early
        if not current_node.media_type:
            return None
        
        # Handle different media types with description
        try:
            logger.info(f"Media type processing for message {current_node.message_id}: media_type={current_node.media_type}")
            logger.info(f"Telegram message attributes: video={bool(message.video)}, animation={bool(message.animation)}, document={bool(message.document)}")
            
            if message.video:
                logger.info("Taking video branch")
                # Handle video description
                file_id = message.video.file_id
                description = await self.describe_video(context.bot, file_id, message.video)
                current_node.media_description = description
                
            elif message.animation:
                logger.info("Taking animation branch")
                # Handle GIF/animation description
                file_id = message.animation.file_id
                logger.info(f"About to call describe_animation with file_id={file_id}")
                description, anim_bytes = await self.describe_animation(context.bot, file_id, message.animation)
                logger.info(f"describe_animation returned: '{description}'")
                current_node.media_description = description
                if anim_bytes and self.main_supports_video:
                    current_node.media_data = anim_bytes
                logger.info(f"Set media_description to: '{current_node.media_description}' and media_data_present={current_node.media_data is not None}")
                
            elif message.document:
                logger.info("Taking document branch")
                # Check if document is actually a GIF/animation (matching the logic in models.py)
                is_gif = (message.document.mime_type and 'gif' in message.document.mime_type.lower()) or \
                        (message.document.file_name and message.document.file_name.lower().endswith('.gif'))
                is_video = (message.document.mime_type and message.document.mime_type.lower().startswith('video/')) or \
                          (message.document.file_name and any(message.document.file_name.lower().endswith(ext) for ext in ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv']))
                
                logger.info(f"Document processing: is_gif={is_gif}, is_video={is_video}, mime_type={message.document.mime_type}, filename={message.document.file_name}")
                
                if is_gif:
                    logger.info(f"Treating document as GIF animation for message {current_node.message_id}")

                    # Update media_type to match refined detection
                    current_node.media_type = 'animation'

                    class MockAnimation:
                        def __init__(self, doc):
                            self.duration = getattr(doc, 'duration', None) or 'unknown'
                            self.width = getattr(doc, 'width', None) or 'unknown'
                            self.height = getattr(doc, 'height', None) or 'unknown'

                    file_id = message.document.file_id
                    description, anim_bytes = await self.describe_animation(
                        context.bot, file_id, MockAnimation(message.document)
                    )
                    logger.info(f"Generated GIF animation description: '{description}'")
                    current_node.media_description = description
                    if anim_bytes and self.main_supports_video:
                        current_node.media_data = anim_bytes

                elif is_video:
                    logger.info(f"Treating document as VIDEO for message {current_node.message_id}")

                    # Update media_type to match refined detection
                    current_node.media_type = 'video'

                    file_id = message.document.file_id
                    description = await self.describe_video(context.bot, file_id, message.document)
                    current_node.media_description = description
                else:
                    # Handle as regular document
                    description = self.describe_document(message.document)
                    current_node.media_description = description
                
            elif message.audio or message.voice:
                # Handle audio/voice description
                if message.audio:
                    description = self.describe_audio(message.audio)
                else:
                    description = f"Voice message ({message.voice.duration} seconds)"
                current_node.media_description = description
                
        except Exception as e:
            logger.error(f"Error describing media: {str(e)}")
            # If description fails, still store the message with basic info
            current_node.media_description = f"[{current_node.media_type}]"
        
        # Store the message with its description
        if current_node:
            await self.storage.add_message(chat_id, current_node)
            
        return current_node
    
    async def describe_video(self, bot, file_id: str, video_info) -> str:
        """
        Describe a video based on its metadata.
        
        Args:
            bot: Telegram bot instance
            file_id (str): ID of the video file
            video_info: Telegram video object with metadata
            
        Returns:
            str: Description of the video
        """
        try:
            # For now, we'll use metadata description
            # In the future, we could download a frame and analyze it
            duration = video_info.duration if video_info.duration else "unknown"
            width = video_info.width if video_info.width else "unknown"
            height = video_info.height if video_info.height else "unknown"
            
            return f"[Video: {duration}s, {width}x{height}]"
            
        except Exception as e:
            logger.error(f"Error describing video: {str(e)}")
            return "[Video]"
    
    async def describe_sticker(self, sticker) -> str:
        """
        Describe a sticker based on its emoji and type.
        
        Args:
            sticker: Telegram sticker object
            
        Returns:
            str: Description of the sticker
        """
        try:
            sticker_type = "sticker"
            if sticker.is_animated:
                sticker_type = "animated sticker"
            elif sticker.is_video:
                sticker_type = "video sticker"
            
            emoji = sticker.emoji if sticker.emoji else "unknown"
            set_name = sticker.set_name if sticker.set_name else "unknown set"
            
            return f"[{sticker_type.capitalize()}: {emoji} from {set_name}]"
            
        except Exception as e:
            logger.error(f"Error describing sticker: {str(e)}")
            return "[Sticker]"
    
    async def describe_animation(self, bot, file_id: str, animation_info) -> tuple[str, bytes | None]:
        """
        Describe an animation/GIF or video using specialized video analysis.
        
        Args:
            bot: Telegram bot instance
            file_id (str): ID of the animation file
            animation_info: Telegram animation object or mock object
            
        Returns:
            tuple[str, bytes | None]: Description of the animation and media data if available
        """
        logger.info(f"describe_animation called with file_id={file_id}")
        video_data = None  # Initialize
        try:
            duration = animation_info.duration if animation_info.duration else "unknown"
            width = animation_info.width if animation_info.width else "unknown"
            height = animation_info.height if animation_info.height else "unknown"
            
            logger.info(f"Animation metadata: duration={duration}, width={width}, height={height}")
            
            # Try to download and analyze the video content
            try:
                # Download the video data
                video_data = await self.image_service.download_image(bot, file_id)
                if video_data and len(video_data) > 0:
                    # Determine MIME type
                    if video_data.startswith(b'GIF'):
                        mime_type = 'image/gif'
                    else:
                        mime_type = 'video/mp4'  # Default for Telegram animations
                    
                    # Create analysis prompt
                    analysis_prompt = self.get_video_analysis_prompt(duration, width, height)
                    
                    analysis_result = None
                    # If the main model lacks video support, delegate to analyzer
                    if not self.main_supports_video and self.video_analyzer:
                        logger.info(f"Using video analyzer to analyze animation {file_id}")
                        analysis_result = await self.video_analyzer.analyze_video(video_data, mime_type, analysis_prompt)
                    
                    if analysis_result:
                        logger.info(f"Video analysis completed: '{analysis_result[:100]}...'")
                        # Add an explicit tag so downstream models know whether this is a GIF or generic MP4 animation
                        tag = "GIF/Animation" if mime_type == 'image/gif' else "Animation"
                        description_with_tag = f"[{tag}: {duration}s, {width}x{height}] {analysis_result}"
                        return description_with_tag, video_data if self.main_supports_video else None
                    # If no external analysis, fall through to basic description
                else:
                    if not self.video_analyzer:
                        logger.warning(f"No video analyzer available for {file_id}")
                    else:
                        logger.warning(f"Could not download video data for {file_id}")
            
            except Exception as e:
                logger.error(f"Error downloading/analyzing video {file_id}: {e}")
            
            # Fallback to basic metadata description (also retains media_data if available)
            # If we reach this point, we could not perform full analysis – provide a clear fallback tag
            tag = "GIF/Animation" if (video_data and video_data.startswith(b'GIF')) else "Animation"
            description = f"[{tag}: {duration}s, {width}x{height} - please analyze the visual content, motion, and any actions in this animation]"
            logger.info(f"Generated fallback description: '{description}'")
            return description, video_data if self.main_supports_video else None
            
        except Exception as e:
            logger.error(f"Error describing animation: {str(e)}")
            return "[Video/Animation - please analyze the visual content]", None
    
    def describe_document(self, document) -> str:
        """
        Describe a document based on its metadata.
        
        Args:
            document: Telegram document object
            
        Returns:
            str: Description of the document
        """
        try:
            file_name = document.file_name if document.file_name else "unnamed"
            mime_type = document.mime_type if document.mime_type else "unknown type"
            file_size = document.file_size if document.file_size else 0
            
            # Convert file size to human readable format
            size_str = self._format_file_size(file_size)
            
            return f"[Document: {file_name} ({mime_type}, {size_str})]"
            
        except Exception as e:
            logger.error(f"Error describing document: {str(e)}")
            return "[Document]"
    
    def describe_audio(self, audio) -> str:
        """
        Describe an audio file based on its metadata.
        
        Args:
            audio: Telegram audio object
            
        Returns:
            str: Description of the audio
        """
        try:
            title = audio.title if audio.title else "Unknown"
            performer = audio.performer if audio.performer else "Unknown artist"
            duration = audio.duration if audio.duration else 0
            
            # Format duration as mm:ss
            minutes = duration // 60
            seconds = duration % 60
            duration_str = f"{minutes}:{seconds:02d}"
            
            return f"[Audio: {title} by {performer} ({duration_str})]"
            
        except Exception as e:
            logger.error(f"Error describing audio: {str(e)}")
            return "[Audio]"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """
        Convert file size in bytes to human readable format.
        
        Args:
            size_bytes (int): File size in bytes
            
        Returns:
            str: Human readable file size
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    async def get_media_info(self, bot, file_id, media_type="image"):
        """
        Get information about a media file.
        
        Args:
            bot: Telegram bot instance
            file_id (str): ID of the media file
            media_type (str): Type of media (image, video, etc.)
            
        Returns:
            dict: Information about the media file
        """
        info = {
            "file_id": file_id,
            "type": media_type
        }
        
        if media_type == "image":
            # Get the URL for the image
            url, success = await self.image_service.get_image_url(bot, file_id)
            if success:
                info["url"] = url
                
        return info
