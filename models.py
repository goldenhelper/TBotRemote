from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import asdict
from telegram import Message


class TransactionType(Enum):
    DEDUCTION = "deduction"
    ADDITION = "addition"

class TransactionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TokenTransaction:
    chat_id: int
    amount: int
    timestamp: str
    transaction_type: TransactionType
    status: TransactionStatus
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ChatMessage:
    message_id: int
    user: str
    content: str
    timestamp: str
    reply_to_id: Optional[int] = None
    media_type: Optional[str] = None
    file_id: Optional[str] = None
    media_description: Optional[str] = None
    sticker: Optional[Dict] = None
    media_data: Optional[bytes] = field(default=None, repr=False, compare=False)
    reasoning: Optional[str] = field(default=None, repr=False, compare=False)

    @classmethod
    def from_telegram_message(cls, message: Message) -> 'ChatMessage':
        content = message.text or ""
        media_type = None
        file_id = None
        media_description = None
        sticker = {"emoji": None, "is_animated": None, "is_video": None}

        # Handle image messages
        if message.photo:
            media_type = "image"
            file_id = message.photo[-1].file_id  # Get the highest resolution photo
            content = message.caption or ""
        
        # Handle video messages
        elif message.video:
            media_type = "video"
            file_id = message.video.file_id
            content = message.caption or ""
        
        # Handle sticker messages
        elif message.sticker:
            media_type = "sticker"
            file_id = message.sticker.file_id
            sticker["emoji"] = message.sticker.emoji
            sticker["is_animated"] = message.sticker.is_animated
            sticker["is_video"] = message.sticker.is_video
            content = sticker["emoji"] or ""
        
        # Handle other types of media
        elif message.document:
            # Determine if the document is a GIF animation or a regular video/file
            is_gif = (
                message.document.mime_type and 'gif' in message.document.mime_type.lower()
            ) or (
                message.document.file_name and message.document.file_name.lower().endswith('.gif')
            )
            is_video = (
                message.document.mime_type and message.document.mime_type.lower().startswith('video/')
            ) or (
                message.document.file_name and any(
                    message.document.file_name.lower().endswith(ext)
                    for ext in ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv']
                )
            )

            if is_gif:
                media_type = "animation"  # keep existing label for GIFs
            elif is_video:
                media_type = "video"
            else:
                media_type = "document"
            
            file_id = message.document.file_id
            content = message.caption or ""
        elif message.audio:
            media_type = "audio"
            file_id = message.audio.file_id
            content = message.caption or ""
        elif message.voice:
            media_type = "voice"
            file_id = message.voice.file_id
            content = message.caption or ""
        elif message.animation:
            media_type = "animation"
            file_id = message.animation.file_id
            content = message.caption or ""

        return cls(
            message_id=message.message_id,
            user=format_user_info(message.from_user),
            content=content,
            timestamp=message.date.strftime("%a, %d. %b %Y %H:%M"),
            reply_to_id=message.reply_to_message.message_id if message.reply_to_message else None,
            media_type=media_type,
            file_id=file_id,
            media_description=media_description,
            sticker=sticker
        )

    def to_dict(self):
        d = asdict(self)
        del d['media_data']
        # keep reasoning for internal inspection
        return d

def format_user_info(user):
    return f"{user.first_name} {user.last_name if user.last_name else ''}"
