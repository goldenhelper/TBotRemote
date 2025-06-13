import logging
from typing import Tuple, Optional, Dict
from telegram import Bot
import base64
import io

logger = logging.getLogger(__name__)

class ImageService:
    """
    Service for handling image retrieval from Telegram.
    
    This class provides methods to:
    - Download images from Telegram
    - Return image data in various formats for further processing
    """
    
    def __init__(self):
        """
        Initialize the ImageService.
        """
        
    async def download_image(self, bot: Bot, file_id: str) -> Optional[bytes]:
        """
        Download an image file from Telegram using its file_id.
        
        Args:
            bot (Bot): Telegram bot instance
            file_id (str): ID of the file to download
            
        Returns:
            Optional[bytes]: The image data as bytes, or None if download failed
        """
        try:
            file = await bot.get_file(file_id)
            # Use the built-in method to download directly to memory
            with io.BytesIO() as buffer:
                await file.download_to_memory(out=buffer)
                return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            return None
    
    async def get_image_data(self, bot: Bot, file_id: str, format: str = "bytes") -> Tuple[Optional[bytes], bool]:
        """
        Get image data in the requested format.
        
        Args:
            bot (Bot): Telegram bot instance
            file_id (str): ID of the image file to retrieve
            format (str): Output format - 'bytes', 'base64', or 'url'
            
        Returns:
            Tuple[Optional[bytes or str], bool]: (image_data, success flag)
        """
        image_data = await self.download_image(bot, file_id)
        if not image_data:
            return None, False
        
        if format == "bytes":
            return image_data, True
        elif format == "base64":
            base64_data = base64.b64encode(image_data).decode("utf-8")
            return base64_data, True
        elif format == "url":
            # Return the file path directly - this is a temporary URL
            file = await bot.get_file(file_id)
            return file.file_path, True
        else:
            logger.error(f"Unknown format requested: {format}")
            return None, False
            
    async def get_image_url(self, bot: Bot, file_id: str) -> Tuple[Optional[str], bool]:
        """
        Get a temporary URL for the image.
        
        Args:
            bot (Bot): Telegram bot instance
            file_id (str): ID of the image file
            
        Returns:
            Tuple[Optional[str], bool]: (image_url, success flag)
        """
        try:
            file = await bot.get_file(file_id)
            return file.file_path, True
        except Exception as e:
            logger.error(f"Error getting image URL: {str(e)}")
            return None, False
