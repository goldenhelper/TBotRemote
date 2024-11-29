from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any

class TransactionType(Enum):
    DEDUCTION = "deduction"
    ADDITION = "addition"

class TransactionStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TokenTransaction:
    user_id: str
    amount: int
    timestamp: datetime
    transaction_type: TransactionType
    status: TransactionStatus
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ChatMessage:
    user_id: str
    username: str
    text: str
    timestamp: datetime
    is_bot_response: bool
    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, user_id: str, username: str, text: str, is_bot_response: bool, timestamp: datetime = None):
        self.user_id = user_id
        self.username = username
        self.text = text
        self.is_bot_response = is_bot_response
        self.timestamp = timestamp if timestamp else datetime.now() # TODO: test this

    @classmethod
    def from_telegram_message(cls, message: "TGMessage"):
        return cls(
            user_id=str(message.from_user.id),
            username=message.from_user.username,
            text=message.text,
            timestamp=message.date,
            is_bot_response=message.from_user.is_bot
        )
