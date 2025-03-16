from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union
import boto3
from datetime import datetime, timedelta
from models import ChatMessage
from telegram import Update
from telegram.ext import ContextTypes
import json
import os
from telegram import Message
from utils.constants import *
import re
import functools
from collections import OrderedDict
import logging
from services.role_manager import RoleManager, Role
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

DEBUG = True

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Notes (delete them later) increment_notes_counter isn't decorated because of an implementation detail, namely when the file is created the counter should be set to zero
# to check: maximal chat history length functionality

def get_role(prompt):
    return re.search(r"<role>.*?</role>", prompt, flags=re.DOTALL).group(0)

def update_role(prompt, new_role):
    return re.sub(r"<role>.*?</role>", f"<role>\n{new_role}\n</role>", prompt, flags=re.DOTALL)

# NOTE: I am not using get() anywhere and instead subscription is used, because of the @initialize_if_not_exists decorator




class Storage(ABC):
    """
    Base Storage class for handling chat history persistence.
    Both FileStorage and AWSStorage should inherit from this class.
    """
    @abstractmethod
    async def save_conversation(self, chat_id: int, tracker: "ConversationTracker"):
        """Save conversation state"""
        pass
    
    @abstractmethod
    async def load_conversation(self, chat_id: int) -> "ConversationTracker":
        """Load conversation state"""
        pass
    
    @abstractmethod
    async def add_message(self, chat_id: int, message: ChatMessage):
        """Add a single message to the conversation history"""
        pass
    
    @abstractmethod
    async def clear_chat_history(self, chat_id: int):
        """Delete the chat history for the given chat ID"""
        pass
    
    @abstractmethod
    async def initialize_chat(self, chat_id: int):
        """Initialize a new chat with default structure"""
        pass
    
    @abstractmethod
    async def get_current_role_id(self, *, chat_id: int):
        """Get current role for the chat"""
        pass

    @abstractmethod
    async def get_available_roles_ids(self, *, chat_id: int):
        """Get available roles for the chat"""
        pass

    @abstractmethod
    async def get_notes(self, *, chat_id: int):
        """Get notes about the conversation"""
        pass
    
    @abstractmethod
    async def set_notes(self, notes_text: str, *, chat_id: int):
        """Set notes about the conversation"""
        pass
    
    @abstractmethod
    async def get_model(self, *, chat_id: int) -> str:
        """Get model"""
        pass
    
    @abstractmethod
    async def set_model(self, model: str, *, chat_id: int):
        """Set model"""
        pass
    
    @abstractmethod
    async def get_memory_updater_model(self, *, chat_id: int) -> str:
        """Get memory updater model"""
        pass
    
    @abstractmethod
    async def set_memory_updater_model(self, model: str, *, chat_id: int):
        """Set memory updater model"""
        pass
    
    @abstractmethod
    async def get_come_to_life_chance(self, *, chat_id: int) -> float:
        """Get come to life chance"""
        pass
    
    @abstractmethod
    async def set_come_to_life_chance(self, chance: float, *, chat_id: int):
        """Set come to life chance"""
        pass
    
    @abstractmethod
    async def get_chat_info(self, *, chat_id: int):
        """Get chat info in a single call"""
        pass
    
    @abstractmethod
    async def increment_notes_counter(self, short_term_memory: int, *, chat_id: int):
        """Increment the counter that tracks how many messages ago the notes were last updated"""
        pass
        
    async def get_message_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE, 
                                max_context_messages: int) -> List[dict]:
        """Get conversation context prioritizing reply chains"""
        message = update.message
        chat_id = message.chat_id
        
        # Load conversation tracker from storage
        tracker = await self.load_conversation(chat_id)
        
        # Add current message to tracker
        current_node = ChatMessage.from_telegram_message(message)
        tracker.add_message(current_node)
        
        context_messages = []
        seen_messages = set()
        
        # First, get the reply chain for the current message
        reply_chain = tracker.get_reply_chain(current_node.message_id, max_depth=max_context_messages)
        for node in reply_chain:
            if node.message_id not in seen_messages:
                context_messages.append({
                    'role': 'user',
                    'content': node.content,
                    'user': node.user,
                    'message_id': node.message_id,
                    'timestamp': node.timestamp,
                    'reply_to': node.reply_to_id,
                    'context_type': 'reply_chain'
                })
                seen_messages.add(node.message_id)


        log(f"IM HERE")
        context_messages[-1]['context_type'] = 'current'
        assert context_messages[-1]['message_id'] == current_node.message_id
        
        # Then add recent messages up to max_context_messages
        
        recent_messages = [msg for msg in reversed(tracker.messages.values()) if msg.message_id not in seen_messages][:max_context_messages - len(context_messages)]
        log(f"Adding recent messages: {recent_messages}")
        log(f"Seen messages: {seen_messages}")
        log(f"reversed(tracker.messages.values()): {list(reversed(tracker.messages.values()))[:max_context_messages]}")

        context_messages.extend([{
            'role': 'user',
            'content': msg.content,
            'user': msg.user,
            'message_id': msg.message_id,
            'timestamp': msg.timestamp,
            'reply_to': msg.reply_to_id,
            'context_type': 'recent'
        } for msg in recent_messages])

        # Sort messages
        def sort_key(msg):
            # If this is a reply, use the timestamp of the message it's replying to
            # This ensures replies come right after their parent messages
            if msg['user'] == BOT_USER_DESCRIPTION and msg['reply_to']:
                parent_msg = tracker.messages[msg['reply_to']]
                if parent_msg:
                    # Add a small offset to ensure it comes right after the parent
                    return (parent_msg.message_id, msg['message_id'])
            return (msg['message_id'], msg['message_id'])
            
        context_messages.sort(key=sort_key)

        # Save updated conversation state
        await self.save_conversation(chat_id, tracker)
        
        return context_messages

class ConversationTracker:
    def __init__(self, storage: Storage):
        self.messages: OrderedDict[int, ChatMessage] = OrderedDict()  # message_id -> ChatMessage
        self.reply_graph: Dict[int, int] = {}  # message_id -> reply message_id
        self.storage = storage
        self.current_role_id = None
        self.available_roles_ids = []
        self.model = ""
        self.memory_updater_model = ""
        self.come_to_life_chance = 0.1
        self.notes = {"text": "", "last_updated_msgs_ago": 0}
        self.timestamp = datetime.now()
        
    def add_message(self, node: ChatMessage):
        """Add a message to the conversation tracker"""
        self.messages[node.message_id] = node
        if node.reply_to_id:  # Changed from reply_to_message_id to reply_to_id
            self.reply_graph[node.message_id] = node.reply_to_id
            
    def get_messages_dict(self):
        """Get messages as a regular dictionary for serialization"""
        return {k: v.to_dict() for k, v in self.messages.items()}
        
    def get_reply_chain(self, message_id: int, max_depth: int):
        """Get the chain of replies leading to this message (including the message itself)"""
        chain = []
        current_id = message_id
        depth = 0
        
        # Traverse the reply chain up to max_depth
        while current_id and current_id in self.messages and depth < max_depth:
            chain.append(self.messages[current_id])
            # Get the message this is replying to
            current_id = self.messages[current_id].reply_to_id  # Changed from reply_to_message_id to reply_to_id
            depth += 1
            
        return chain

class AWSStorage(Storage):
    def __init__(self, table_name: str, default_model: str, default_role_id: str, region: str = 'us-east-1', default_memory_updater_model: str | None = None, default_come_to_life_chance: float = DEFAULT_COME_TO_LIFE_CHANCE):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)
        self.default_role_id = default_role_id
        self.default_model = default_model
        self.default_memory_updater_model = default_memory_updater_model if default_memory_updater_model is not None else default_model
        self.default_come_to_life_chance = default_come_to_life_chance
        
    async def save_conversation(self, chat_id: int, tracker: "ConversationTracker"):
        """Save conversation state to DynamoDB"""
        self.table.put_item(Item={
            'chat_id': str(chat_id),
            'model': tracker.model,
            'memory_updater_model': tracker.memory_updater_model,
            'come_to_life_chance': tracker.come_to_life_chance,
            'timestamp': datetime.now().isoformat(),
            'messages': {str(k): v.to_dict() for k, v in tracker.messages.items()},
            'reply_graph': {str(k): v for k, v in tracker.reply_graph.items()},
            'role': tracker.role if tracker.role else self.default_role,
            'notes': tracker.notes if tracker.notes else "",
            'ttl': int((datetime.now() + timedelta(days=1)).timestamp())
        })
    
    async def load_conversation(self, chat_id: int) -> "ConversationTracker":
        """Load conversation state from DynamoDB"""
        response = self.table.get_item(Key={'chat_id': str(chat_id)})
        tracker = ConversationTracker(self)
        
        if 'Item' in response:
            messages = response['Item'].get('messages', {})
            for msg_data in messages.values():
                node = ChatMessage(**msg_data)
                tracker.add_message(node)
                
            tracker.role = response['Item'].get('role', BOT_USER_DESCRIPTION)
            tracker.model = response['Item'].get('model', self.default_model)
            tracker.memory_updater_model = response['Item'].get('memory_updater_model', self.default_memory_updater_model)
            tracker.come_to_life_chance = response['Item'].get('come_to_life_chance', self.default_come_to_life_chance)
            tracker.notes = response['Item'].get('notes', "")
                
        return tracker

    async def add_message(self, chat_id: int, message: ChatMessage):
        """Add a single message to the conversation history"""
        # Load existing data if available
        tracker = await self.load_conversation(chat_id)
        
        # Add new message to the data
        tracker.add_message(message)

        # Save updated conversation
        await self.save_conversation(chat_id, tracker)

    async def clear_chat_history(self, chat_id: int):
        """Delete the table item corresponding to the chat ID"""
        self.table.delete_item(Key={'chat_id': str(chat_id)})

    async def initialize_chat(self, chat_id: int):
        """Initialize a new DynamoDB item with default structure."""
        self.table.put_item(Item={
            'chat_id': str(chat_id),
            'model': self.default_model,
            'memory_updater_model': self.default_memory_updater_model,
            'current_role_id': self.default_role,
            'available_roles_ids': self.global_roles_ids,
            'messages': {},
            'notes': {'text': '', 'last_updated_msgs_ago': 0},
            'come_to_life_chance': self.default_come_to_life_chance,
            'timestamp': datetime.now().isoformat(),
            'reply_graph': {},
            'ttl': int((datetime.now() + timedelta(days=1)).timestamp())
        })

    @staticmethod
    def initialize_if_not_exists(func):
        """Decorator that initializes a DynamoDB item if it doesn't exist before executing the function."""
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            chat_id = kwargs.get('chat_id')
            if chat_id is None:
                raise ValueError("Function must have chat_id as a keyword argument")
                
            response = self.table.get_item(Key={'chat_id': str(chat_id)})
            if 'Item' not in response:
                await self.initialize_chat(chat_id)
                    
            return await func(self, *args, **kwargs)
            
        return wrapper

    @initialize_if_not_exists
    async def set_role_by_id(self, role_id: str, *, chat_id: int):
        """Set system prompt in DynamoDB"""
        self.table.update_item(
            Key={'chat_id': str(chat_id)},
            UpdateExpression='SET #r = :sp',
            ExpressionAttributeNames={'#r': 'role_id'},
            ExpressionAttributeValues={':sp': role_id}
        )

    @initialize_if_not_exists
    async def get_notes(self, *, chat_id: int):
        """Get notes about the conversation from DynamoDB. Returns dict with 'text' and 'last_updated_msgs_ago' keys."""
        response = self.table.get_item(Key={'chat_id': str(chat_id)})
        return response['Item']['notes']

    @initialize_if_not_exists
    async def set_notes(self, notes_text: str, *, chat_id: int):
        """Set notes about the conversation in DynamoDB"""
        self.table.update_item(
            Key={'chat_id': str(chat_id)},
            UpdateExpression='SET #n = :np',
            ExpressionAttributeNames={'#n': 'notes'},
            ExpressionAttributeValues={
                ':np': {'text': notes_text, 'last_updated_msgs_ago': 0} 
            }
        )

    @initialize_if_not_exists
    async def get_model(self, *, chat_id: int) -> str:
        """Get model from DynamoDB"""
        response = self.table.get_item(Key={'chat_id': str(chat_id)})
        return response['Item'].get('model', self.default_model)

    @initialize_if_not_exists
    async def set_model(self, model: str, *, chat_id: int):
        """Set model in DynamoDB"""
        self.table.update_item(
            Key={'chat_id': str(chat_id)},
            UpdateExpression='SET #m = :m',
            ExpressionAttributeNames={'#m': 'model'},
            ExpressionAttributeValues={':m': model}
        )

    @initialize_if_not_exists
    async def get_memory_updater_model(self, *, chat_id: int) -> str:
        """Get memory updater model from DynamoDB. If not set, return the current model."""
        response = self.table.get_item(Key={'chat_id': str(chat_id)})
        return response['Item'].get('memory_updater_model', self.get_model(chat_id=chat_id))

    @initialize_if_not_exists
    async def set_memory_updater_model(self, model: str, *, chat_id: int):
        """Set memory updater model in DynamoDB"""
        self.table.update_item(
            Key={'chat_id': str(chat_id)},
            UpdateExpression='SET #mu = :mu',
            ExpressionAttributeNames={'#mu': 'memory_updater_model'},
            ExpressionAttributeValues={':mu': model}
        )

    @initialize_if_not_exists
    async def get_come_to_life_chance(self, *, chat_id: int) -> float:
        """Get come to life chance from DynamoDB. If not set, return the default value."""
        response = self.table.get_item(Key={'chat_id': str(chat_id)})
        return response['Item'].get('come_to_life_chance', self.default_come_to_life_chance)
    
    @initialize_if_not_exists
    async def set_come_to_life_chance(self, chance: float, *, chat_id: int):
        """Set come to life chance in DynamoDB"""
        self.table.update_item(
            Key={'chat_id': str(chat_id)},
            UpdateExpression='SET #ctlc = :ctlc',
            ExpressionAttributeNames={'#ctlc': 'come_to_life_chance'},
            ExpressionAttributeValues={':ctlc': chance}
        )
    
    @initialize_if_not_exists
    async def get_chat_info(self, *, chat_id: int):
        """
        Returns a dictionary with all chat information in a single call.
        This is more efficient than making multiple separate calls.
        
        Returns:
            dict: A dictionary containing chat information such as:
                - role
                - notes
                - model
                - memory_updater_model
                - come_to_life_chance
                - and other available fields
        """
        response = self.table.get_item(Key={'chat_id': str(chat_id)})
        item = response['Item']
        # Create a dictionary with the requested information
        info = {
            'role': item['role'],
            'notes': item['notes'],
            'model': item['model'],
            'memory_updater_model': item['memory_updater_model'],
            'come_to_life_chance': item['come_to_life_chance'],
            'timestamp': item['timestamp'],
            }
            
        return info
        
        
    async def increment_notes_counter(self, short_term_memory: int, *, chat_id: int) -> bool:
        """Increment the counter that tracks how many messages ago the notes were last updated
        Returns true if the counter was reset
        """
        update_notes = True
        response = self.table.get_item(Key={'chat_id': str(chat_id)})
        
        if 'Item' not in response:
            await self.initialize_chat(chat_id)
            update_notes = False
        else:
            notes = response['Item'].get('notes', {'text': '', 'last_updated_msgs_ago': 0})
            current_count = notes.get('last_updated_msgs_ago', 0)
            
            # Reset counter if it exceeds short_term_memory
            if current_count >= short_term_memory:
                new_count = 1
            else:
                new_count = current_count + 1
                update_notes = False
            
            # Update the counter
            self.table.update_item(
                Key={'chat_id': str(chat_id)},
                UpdateExpression='SET #n.#l = :new_count',
                ExpressionAttributeNames={
                    '#n': 'notes',
                    '#l': 'last_updated_msgs_ago'
                },
                ExpressionAttributeValues={
                    ':new_count': new_count
                }
            )
        
        return update_notes
        

class FileStorage(Storage):
    def __init__(self, storage_dir: str, default_role_id: str, default_model: str, role_manager: RoleManager,
                 default_memory_updater_model: str | None = None, 
                 default_come_to_life_chance: float = DEFAULT_COME_TO_LIFE_CHANCE, 
                 ):
        """Initialize file storage with a directory to store chat files"""
        self.storage_dir = storage_dir
        self.default_role_id = default_role_id
        self.default_model = default_model
        self.default_memory_updater_model = default_memory_updater_model if default_memory_updater_model is not None else default_model
        self.default_come_to_life_chance = default_come_to_life_chance
        self.role_manager = role_manager
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_file_path(self, chat_id: int) -> str:
        return os.path.join(self.storage_dir, f"chat_{chat_id}.json")
        
    async def save_conversation(self, chat_id: int, tracker: "ConversationTracker"):
        """Save conversation state to a JSON file
        Save only last CHAT_HISTORY_DEPTH messages"""
        data = {
            'chat_id': str(chat_id),
            'model': tracker.model,
            'come_to_life_chance': tracker.come_to_life_chance,
            'memory_updater_model': tracker.memory_updater_model,
            'current_role_id': tracker.current_role_id,
            'available_roles_ids': tracker.available_roles_ids,
            'timestamp': datetime.now().isoformat(),
            'messages': tracker.get_messages_dict(),
            'reply_graph': {str(k): v for k, v in tracker.reply_graph.items()},
            'notes': tracker.notes,
            'ttl': int((datetime.now() + timedelta(days=1)).timestamp())
        }
        
        file_path = self._get_file_path(chat_id)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    async def load_conversation(self, chat_id: int) -> "ConversationTracker":
        """Load conversation state from a JSON file"""
        file_path = self._get_file_path(chat_id)
        tracker = ConversationTracker(self)
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                messages = data.get('messages', {})
                for msg_data in messages.values():
                    node = ChatMessage(**msg_data)
                    tracker.add_message(node)
                tracker.current_role_id = data['current_role_id']
                tracker.available_roles_ids = data['available_roles_ids']
                tracker.model = data['model']
                tracker.memory_updater_model = data['memory_updater_model']
                tracker.notes = data['notes']
                tracker.come_to_life_chance = data['come_to_life_chance']
        
        return tracker

    async def add_message(self, chat_id: int, message: ChatMessage):
        """Add a single message to the conversation history"""
        
        # Load existing data if available
        tracker = await self.load_conversation(chat_id)
        
        # Add new message to the data
        tracker.add_message(message)


        await self.save_conversation(chat_id, tracker)

    async def clear_chat_history(self, chat_id: int):
        """Delete the chat history file for the given chat ID"""
        file_path = self._get_file_path(chat_id)
        if os.path.exists(file_path):
            os.remove(file_path)

    async def initialize_chat(self, chat_id: int):
        """Initialize a new chat file with default structure."""
        file_path = self._get_file_path(chat_id)
        if not os.path.exists(file_path):
            log(f"Creating file {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            data = {
                'chat_id': str(chat_id),
                "model": self.default_model,
                "memory_updater_model": self.default_memory_updater_model,
                "current_role_id": self.default_role_id,
                "available_roles_ids": [],
                "messages": {},
                "notes": {"text": "", "last_updated_msgs_ago": 0},
                "come_to_life_chance": self.default_come_to_life_chance,
                "timestamp": datetime.now().isoformat(),
                "reply_graph": {},
                "ttl": int((datetime.now() + timedelta(days=1)).timestamp())
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def initialize_if_not_exists(func):
        async def wrapper(self, *args, **kwargs):
            chat_id = kwargs.get('chat_id')
            if chat_id is None:
                raise ValueError("Function must have chat_id as a keyword argument")
            await self.initialize_chat(chat_id)            
            return await func(self, *args, **kwargs)
        return wrapper

    @initialize_if_not_exists
    async def get_current_role_id(self, *, chat_id: int) -> str:
        """Return the current role for the given chat ID."""
        file_path = self._get_file_path(chat_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            current_role_id = data['current_role_id']

        return current_role_id

    @initialize_if_not_exists
    async def get_current_role(self, *, chat_id: int) -> Role:
        """Return the current role for the given chat ID."""
        current_role_id = await self.get_current_role_id(chat_id=chat_id)
        
        return await self.role_manager.get_role_by_id(role_id=current_role_id)

    @initialize_if_not_exists
    async def set_current_role_id(self, role_id: str, *, chat_id: int) -> None:
        """Set the current role for the given chat ID."""
        file_path = self._get_file_path(chat_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['current_role_id'] = role_id

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @initialize_if_not_exists
    async def get_available_roles_ids(self, *, chat_id: int) -> list[str]:
        """
        Get all roles available for a specific chat.
        
        Args:
            chat_id (str): The chat ID to get roles for
            
        Returns:
            list: Combined list of chat-specific and global roles
        """
        try:
            
            file_path = self._get_file_path(chat_id)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data['available_roles_ids']
            
        except Exception as e:
            # Log the error and return just global roles as fallback
            print(f"Error retrieving chat roles for chat {chat_id}: {str(e)}")
            return []

    @initialize_if_not_exists
    async def get_available_roles(self, *, chat_id: int) -> Dict[str, Role]:
        """Return all roles available for a specific chat."""
        ids = await self.get_available_roles_ids(chat_id=chat_id)
        return await self.role_manager.get_roles_by_ids(ids)

    @initialize_if_not_exists
    async def add_available_role_id(self, role_id: str, *, chat_id: int) -> None:
        """Add a new role to the chat."""
        
        file_path = self._get_file_path(chat_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['available_roles_ids'].append(role_id)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @initialize_if_not_exists
    async def remove_available_role_id(self, role_id: str, *, chat_id: int) -> None:
        file_path = self._get_file_path(chat_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data['available_roles_ids'].remove(role_id)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @initialize_if_not_exists
    async def get_notes(self, *, chat_id: int):
        file_path = self._get_file_path(chat_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            notes = data['notes']
            return notes
        
    @initialize_if_not_exists
    async def set_notes(self, notes_text: str, *, chat_id: int):
        file_path = self._get_file_path(chat_id)
        
        # Load existing data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Update notes
        data['notes'] = {'text': notes_text, 'last_updated_msgs_ago': 0}
        
        # Write back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @initialize_if_not_exists
    async def get_model(self, *, chat_id: int) -> str:
        file_path = self._get_file_path(chat_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['model']

    @initialize_if_not_exists
    async def set_model(self, model: str, *, chat_id: int):
        file_path = self._get_file_path(chat_id)
        
        # Load existing data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['model'] = model
        
        # Write back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @initialize_if_not_exists
    async def get_memory_updater_model(self, *, chat_id: int) -> str:
        """Get memory updater model"""
        file_path = self._get_file_path(chat_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data['memory_updater_model']

    @initialize_if_not_exists
    async def set_memory_updater_model(self, model: str, *, chat_id: int):
        """Set memory updater model"""
        file_path = self._get_file_path(chat_id)
        
        # Load existing data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['memory_updater_model'] = model
        
        # Write back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @initialize_if_not_exists
    async def get_come_to_life_chance(self, *, chat_id: int) -> float:
        """Get come to life chance from the file. If not set, return the default value."""
        file_path = self._get_file_path(chat_id)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('come_to_life_chance', self.default_come_to_life_chance)
        
    @initialize_if_not_exists
    async def set_come_to_life_chance(self, chance: float, *, chat_id: int):
        """Set come to life chance """
        file_path = self._get_file_path(chat_id)
        
        # Load existing data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['come_to_life_chance'] = chance
        
        # Write back with UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @initialize_if_not_exists
    async def get_chat_info(self, *, chat_id: int):
        """
        Returns a dictionary with all chat information in a single call.
        This is more efficient than making multiple separate calls.
        
        Returns:
            dict: A dictionary containing chat information such as:
                - role - dict with 'role_name' and 'role_prompt' and 'is_global'
                - notes - dictionary with 'text' and 'last_updated_msgs_ago'
                - model - string
                - memory_updater_model - string
                - come_to_life_chance - float
                - timestamp - string
        """
        file_path = self._get_file_path(chat_id)
    
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Create a dictionary with the requested information
            info = {
                'current_role_id': data['current_role_id'],
                'available_roles_ids': data['available_roles_ids'],
                'notes': data['notes'],
                'model': data['model'],
                'memory_updater_model': data['memory_updater_model'],
                'come_to_life_chance': data['come_to_life_chance'],
                'timestamp': data['timestamp']
            }
            
            return info
    
     
    async def increment_notes_counter(self, short_term_memory: int, *, chat_id: int):
        """Returns true if the counter was reset"""
        update_notes = True

        file_path = self._get_file_path(chat_id)
        if not os.path.exists(file_path):
            self.initialize_chat(chat_id)
            update_notes = False

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            notes = data['notes']
            current_count = notes['last_updated_msgs_ago']
            
            # Reset counter if it exceeds short_term_memory
            if current_count >= short_term_memory:
                new_count = 1
            else:
                new_count = current_count + 1
                update_notes = False
            
            # Update the counter
            data['notes']['last_updated_msgs_ago'] = new_count
            
            # Write back with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return update_notes
 
 
