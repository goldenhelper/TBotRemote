from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any
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

    @classmethod
    def from_telegram_message(cls, message: Message) -> 'ChatMessage':
        return cls(
            message_id=message.message_id,
            user=format_user_info(message.from_user),
            content=message.text,
            timestamp=message.date.isoformat(),
            reply_to_id=message.reply_to_message.message_id if message.reply_to_message else None
        )


    def to_dict(self):
        return asdict(self)

def format_user_info(user):
    return f"{user.first_name} {user.last_name if user.last_name else ''}"


