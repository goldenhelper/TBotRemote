import boto3
from datetime import datetime
import logging
from models import ChatMessage

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    def __init__(self, table_name: str, region: str):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)

    async def save_message(self, message: ChatMessage) -> None:
        """Save a message to the chat history
        parameters:
            message: ChatMessage - the message to save
        """
        try:
            self.table.put_item(
                Item={
                    'user_id': str(message.user_id),
                    'timestamp': message.timestamp.isoformat(),
                    'username': message.username,
                    'message': message.text,
                    'message_type': "Assistant_response" if message.is_bot_response else "User_message",
                    'metadata': message.metadata or {}
                }
            )
        except Exception as e:
            logger.error(f"Error saving chat history: {e}")
            raise