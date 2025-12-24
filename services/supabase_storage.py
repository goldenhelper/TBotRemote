"""
Supabase storage implementation for the Telegram bot.
Replaces DynamoDB-based AWSStorage with Supabase (PostgreSQL).
"""
from typing import Dict, List, Optional
from datetime import datetime
from collections import OrderedDict
import functools
import logging

from supabase import create_client, Client
from models import ChatMessage
from services.chat_history import Storage, ConversationTracker
from utils.constants import DEFAULT_COME_TO_LIFE_CHANCE

logger = logging.getLogger(__name__)


class SupabaseStorage(Storage):
    """
    Supabase-based storage for chat history and settings.
    Replaces AWSStorage with a PostgreSQL backend.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        role_manager=None,
    ):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.role_manager = role_manager
        # Settings are loaded dynamically from Supabase via get_bot_setting()

    @property
    def default_model(self) -> str:
        return self.get_bot_setting('default_model')

    @property
    def default_role_id(self) -> Optional[str]:
        return self.get_bot_setting('default_role_id')

    @property
    def default_memory_updater_model(self) -> str:
        return self.get_bot_setting('default_memory_updater_model')

    @property
    def default_come_to_life_chance(self) -> float:
        return self.get_bot_setting('default_come_to_life_chance')

    @property
    def default_tokens_for_new_chats(self) -> int:
        return self.get_bot_setting('default_tokens_for_new_chats')

    @property
    def max_num_roles(self) -> int:
        return self.get_bot_setting('max_num_roles')

    @property
    def max_role_name_length(self) -> int:
        return self.get_bot_setting('max_role_name_length')

    @property
    def video_analyzer_model(self) -> str:
        return self.get_bot_setting('video_analyzer_model')

    async def initialize_chat(self, chat_id: int):
        """Initialize a new chat with default structure."""
        # Check if chat exists
        result = self.client.table('chats').select('chat_id').eq('chat_id', chat_id).execute()

        if not result.data:
            self.client.table('chats').insert({
                'chat_id': chat_id,
                'model': self.default_model,
                'memory_updater_model': self.default_memory_updater_model,
                'current_role_id': self.default_role_id,
                'come_to_life_chance': self.default_come_to_life_chance,
                'notes_text': '',
                'notes_last_updated_msgs_ago': 0,
                'tokens': self.default_tokens_for_new_chats,
            }).execute()

    @staticmethod
    def initialize_if_not_exists(func):
        """Decorator that initializes chat if it doesn't exist."""
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            chat_id = kwargs.get('chat_id')
            if chat_id is None:
                raise ValueError("Function must have chat_id as a keyword argument")
            await self.initialize_chat(chat_id)
            return await func(self, *args, **kwargs)
        return wrapper

    async def save_conversation(self, chat_id: int, tracker: "ConversationTracker"):
        """Save conversation state to Supabase."""
        # Update chat metadata
        self.client.table('chats').upsert({
            'chat_id': chat_id,
            'model': tracker.model,
            'memory_updater_model': tracker.memory_updater_model,
            'come_to_life_chance': tracker.come_to_life_chance,
            'current_role_id': tracker.current_role_id or self.default_role_id,
            'notes_text': tracker.notes.get('text', '') if isinstance(tracker.notes, dict) else '',
            'notes_last_updated_msgs_ago': tracker.notes.get('last_updated_msgs_ago', 0) if isinstance(tracker.notes, dict) else 0,
        }).execute()

        # Save messages (upsert each message)
        for msg in tracker.messages.values():
            msg_data = {
                'chat_id': chat_id,
                'message_id': msg.message_id,
                'user_name': msg.user,
                'content': msg.content,
                'timestamp': msg.timestamp,
                'reply_to_id': msg.reply_to_id,
                'media_type': msg.media_type,
                'file_id': msg.file_id,
                'media_description': msg.media_description,
                'sticker': msg.sticker,
                'reasoning': msg.reasoning,
            }
            self.client.table('messages').upsert(
                msg_data,
                on_conflict='chat_id,message_id'
            ).execute()

        # Update available roles (junction table)
        # First delete existing, then insert new
        self.client.table('chat_roles').delete().eq('chat_id', chat_id).execute()
        for role_id in tracker.available_roles_ids:
            self.client.table('chat_roles').insert({
                'chat_id': chat_id,
                'role_id': role_id,
            }).execute()

    @initialize_if_not_exists
    async def load_conversation(self, *, chat_id: int) -> "ConversationTracker":
        """Load conversation state from Supabase."""
        tracker = ConversationTracker(self)

        # Load chat metadata
        chat_result = self.client.table('chats').select('*').eq('chat_id', chat_id).execute()

        if chat_result.data:
            chat = chat_result.data[0]
            tracker.model = chat.get('model', self.default_model)
            tracker.memory_updater_model = chat.get('memory_updater_model', self.default_memory_updater_model)
            tracker.come_to_life_chance = chat.get('come_to_life_chance', self.default_come_to_life_chance)
            tracker.current_role_id = chat.get('current_role_id', self.default_role_id)
            tracker.notes = {
                'text': chat.get('notes_text', ''),
                'last_updated_msgs_ago': chat.get('notes_last_updated_msgs_ago', 0)
            }

        # Load messages (ordered by message_id)
        msg_result = self.client.table('messages')\
            .select('*')\
            .eq('chat_id', chat_id)\
            .order('message_id')\
            .execute()

        for msg_data in msg_result.data:
            # Handle sticker field - ensure it's a dict
            sticker = msg_data.get('sticker')
            if sticker is None:
                sticker = {"emoji": None, "is_animated": None, "is_video": None}

            node = ChatMessage(
                message_id=msg_data['message_id'],
                user=msg_data['user_name'],
                content=msg_data['content'] or '',
                timestamp=msg_data['timestamp'] or '',
                reply_to_id=msg_data.get('reply_to_id'),
                media_type=msg_data.get('media_type'),
                file_id=msg_data.get('file_id'),
                media_description=msg_data.get('media_description'),
                sticker=sticker,
                reasoning=msg_data.get('reasoning'),
            )
            tracker.add_message(node)

        # Load available roles
        roles_result = self.client.table('chat_roles')\
            .select('role_id')\
            .eq('chat_id', chat_id)\
            .execute()
        tracker.available_roles_ids = [r['role_id'] for r in roles_result.data]

        return tracker

    async def add_message(self, chat_id: int, message: ChatMessage):
        """Add a single message to the conversation history."""
        await self.initialize_chat(chat_id)

        msg_data = {
            'chat_id': chat_id,
            'message_id': message.message_id,
            'user_name': message.user,
            'content': message.content,
            'timestamp': message.timestamp,
            'reply_to_id': message.reply_to_id,
            'media_type': message.media_type,
            'file_id': message.file_id,
            'media_description': message.media_description,
            'sticker': message.sticker,
            'reasoning': message.reasoning,
        }
        self.client.table('messages').upsert(
            msg_data,
            on_conflict='chat_id,message_id'
        ).execute()

    async def update_message_description(self, chat_id: int, message_id: int, new_description: str):
        """Update the media description of a specific message."""
        self.client.table('messages')\
            .update({'media_description': new_description})\
            .eq('chat_id', chat_id)\
            .eq('message_id', message_id)\
            .execute()
        logger.info(f"Updated message {message_id} description in Supabase")

    async def clear_messages(self, chat_id: int):
        """Clear all messages for a chat but keep the chat entry (tokens, settings, etc.)."""
        self.client.table('messages').delete().eq('chat_id', chat_id).execute()
        # Also clear notes since they're derived from message history
        self.client.table('chats').update({'notes_text': '', 'notes_last_updated_msgs_ago': 0}).eq('chat_id', chat_id).execute()
        logger.info(f"Cleared messages and notes for chat {chat_id}")

    async def delete_chat(self, chat_id: int):
        """Delete the chat and all its messages, roles, tokens, and settings."""
        # Messages are deleted via CASCADE, but let's be explicit
        self.client.table('messages').delete().eq('chat_id', chat_id).execute()
        self.client.table('chat_roles').delete().eq('chat_id', chat_id).execute()
        self.client.table('chats').delete().eq('chat_id', chat_id).execute()
        logger.info(f"Deleted chat {chat_id} and all associated data")

    @initialize_if_not_exists
    async def get_current_role_id(self, *, chat_id: int) -> str:
        """Get current role ID for the chat."""
        result = self.client.table('chats')\
            .select('current_role_id')\
            .eq('chat_id', chat_id)\
            .execute()
        if result.data:
            role_id = result.data[0].get('current_role_id')
            if role_id is None:
                logger.warning(
                    f"Chat {chat_id} has NULL current_role_id - data integrity issue. "
                    f"Fixing by setting to default: {self.default_role_id}"
                )
                # Fix the data integrity issue
                self.client.table('chats')\
                    .update({'current_role_id': self.default_role_id})\
                    .eq('chat_id', chat_id)\
                    .execute()
                return self.default_role_id
            return role_id
        return self.default_role_id

    @initialize_if_not_exists
    async def get_current_role(self, *, chat_id: int):
        """Get current role object for the chat."""
        current_role_id = await self.get_current_role_id(chat_id=chat_id)
        return await self.role_manager.get_role_by_id(role_id=current_role_id)

    @initialize_if_not_exists
    async def get_available_roles(self, *, chat_id: int):
        """Return all roles available for a specific chat."""
        ids = await self.get_available_roles_ids(chat_id=chat_id)
        return await self.role_manager.get_roles_by_ids(ids)

    @initialize_if_not_exists
    async def set_current_role_id(self, role_id: str, *, chat_id: int):
        """Set current role ID for the chat."""
        self.client.table('chats')\
            .update({'current_role_id': role_id})\
            .eq('chat_id', chat_id)\
            .execute()

    @initialize_if_not_exists
    async def get_available_roles_ids(self, *, chat_id: int) -> List[str]:
        """Get available role IDs for the chat."""
        result = self.client.table('chat_roles')\
            .select('role_id')\
            .eq('chat_id', chat_id)\
            .execute()
        return [r['role_id'] for r in result.data]

    @initialize_if_not_exists
    async def add_available_role_id(self, role_id: str, *, chat_id: int):
        """Add a role to the chat's available roles."""
        self.client.table('chat_roles').upsert({
            'chat_id': chat_id,
            'role_id': role_id,
        }).execute()

    @initialize_if_not_exists
    async def remove_available_role_id(self, role_id: str, *, chat_id: int):
        """Remove a role from the chat's available roles."""
        self.client.table('chat_roles')\
            .delete()\
            .eq('chat_id', chat_id)\
            .eq('role_id', role_id)\
            .execute()

    @initialize_if_not_exists
    async def get_notes(self, *, chat_id: int) -> dict:
        """Get notes about the conversation."""
        result = self.client.table('chats')\
            .select('notes_text, notes_last_updated_msgs_ago')\
            .eq('chat_id', chat_id)\
            .execute()
        if result.data:
            return {
                'text': result.data[0].get('notes_text', ''),
                'last_updated_msgs_ago': result.data[0].get('notes_last_updated_msgs_ago', 0)
            }
        return {'text': '', 'last_updated_msgs_ago': 0}

    @initialize_if_not_exists
    async def set_notes(self, notes_text: str, *, chat_id: int):
        """Set notes about the conversation."""
        self.client.table('chats')\
            .update({
                'notes_text': notes_text,
                'notes_last_updated_msgs_ago': 0
            })\
            .eq('chat_id', chat_id)\
            .execute()

    @initialize_if_not_exists
    async def get_model(self, *, chat_id: int) -> str:
        """Get current model for the chat."""
        result = self.client.table('chats')\
            .select('model')\
            .eq('chat_id', chat_id)\
            .execute()
        return result.data[0]['model'] if result.data else self.default_model

    @initialize_if_not_exists
    async def set_model(self, model: str, *, chat_id: int):
        """Set current model for the chat."""
        self.client.table('chats')\
            .update({'model': model})\
            .eq('chat_id', chat_id)\
            .execute()

    @initialize_if_not_exists
    async def get_memory_updater_model(self, *, chat_id: int) -> str:
        """Get memory updater model for the chat."""
        result = self.client.table('chats')\
            .select('memory_updater_model')\
            .eq('chat_id', chat_id)\
            .execute()
        return result.data[0]['memory_updater_model'] if result.data else self.default_memory_updater_model

    @initialize_if_not_exists
    async def set_memory_updater_model(self, model: str, *, chat_id: int):
        """Set memory updater model for the chat."""
        self.client.table('chats')\
            .update({'memory_updater_model': model})\
            .eq('chat_id', chat_id)\
            .execute()

    @initialize_if_not_exists
    async def get_come_to_life_chance(self, *, chat_id: int) -> float:
        """Get come to life chance for the chat."""
        result = self.client.table('chats')\
            .select('come_to_life_chance')\
            .eq('chat_id', chat_id)\
            .execute()
        return result.data[0]['come_to_life_chance'] if result.data else self.default_come_to_life_chance

    @initialize_if_not_exists
    async def set_come_to_life_chance(self, chance: float, *, chat_id: int):
        """Set come to life chance for the chat."""
        self.client.table('chats')\
            .update({'come_to_life_chance': chance})\
            .eq('chat_id', chat_id)\
            .execute()

    @initialize_if_not_exists
    async def get_chat_info(self, *, chat_id: int) -> dict:
        """Get all chat information in a single call."""
        result = self.client.table('chats')\
            .select('*')\
            .eq('chat_id', chat_id)\
            .execute()

        if result.data:
            chat = result.data[0]
            return {
                'current_role_id': chat.get('current_role_id'),
                'available_roles_ids': await self.get_available_roles_ids(chat_id=chat_id),
                'notes': {
                    'text': chat.get('notes_text', ''),
                    'last_updated_msgs_ago': chat.get('notes_last_updated_msgs_ago', 0)
                },
                'model': chat.get('model', self.default_model),
                'memory_updater_model': chat.get('memory_updater_model', self.default_memory_updater_model),
                'come_to_life_chance': chat.get('come_to_life_chance', self.default_come_to_life_chance),
                'timestamp': chat.get('updated_at', ''),
            }
        return {}

    async def increment_notes_counter(self, short_term_memory: int, *, chat_id: int) -> bool:
        """
        Increment the counter that tracks how many messages ago the notes were last updated.
        Returns True if the counter was reset (i.e., notes should be updated).
        """
        await self.initialize_chat(chat_id)

        result = self.client.table('chats')\
            .select('notes_last_updated_msgs_ago')\
            .eq('chat_id', chat_id)\
            .execute()

        if not result.data:
            return False

        current_count = result.data[0].get('notes_last_updated_msgs_ago', 0)

        if current_count >= short_term_memory:
            new_count = 1
            update_notes = True
        else:
            new_count = current_count + 1
            update_notes = False

        self.client.table('chats')\
            .update({'notes_last_updated_msgs_ago': new_count})\
            .eq('chat_id', chat_id)\
            .execute()

        return update_notes

    # ==================== Admin Management ====================

    def get_admin_user_ids(self) -> List[int]:
        """Get list of admin user IDs from config table. Raises error if Supabase not accessible."""
        result = self.client.table('config')\
            .select('value')\
            .eq('key', 'admin_user_ids')\
            .execute()

        if result.data:
            return [int(uid) for uid in result.data[0]['value']]
        # Entry not found is OK - return empty list (primary admin will be added on startup)
        return []

    def add_admin_user(self, user_id: int) -> bool:
        """Add a user ID to the admin list."""
        try:
            current_admins = self.get_admin_user_ids()
            if user_id in current_admins:
                return False  # Already an admin

            current_admins.append(user_id)
            self.client.table('config')\
                .upsert({'key': 'admin_user_ids', 'value': current_admins})\
                .execute()
            logger.info(f"Added user {user_id} to admin list")
            return True
        except Exception as e:
            logger.error(f"Error adding admin user {user_id}: {e}")
            return False

    def remove_admin_user(self, user_id: int) -> bool:
        """Remove a user ID from the admin list."""
        try:
            current_admins = self.get_admin_user_ids()
            if user_id not in current_admins:
                return False  # Not an admin

            current_admins.remove(user_id)
            self.client.table('config')\
                .upsert({'key': 'admin_user_ids', 'value': current_admins})\
                .execute()
            logger.info(f"Removed user {user_id} from admin list")
            return True
        except Exception as e:
            logger.error(f"Error removing admin user {user_id}: {e}")
            return False

    def is_admin(self, user_id: int) -> bool:
        """Check if a user ID is in the admin list."""
        return user_id in self.get_admin_user_ids()

    # ==================== Bot Settings ====================

    def get_bot_settings(self) -> dict:
        """Get all bot settings from config table. Raises error if not accessible."""
        result = self.client.table('config')\
            .select('value')\
            .eq('key', 'bot_settings')\
            .execute()

        if not result.data:
            raise RuntimeError("Bot settings not found in Supabase. Please initialize the 'bot_settings' config entry.")

        return result.data[0]['value']

    def get_bot_setting(self, key: str):
        """Get a specific bot setting. Raises error if not accessible."""
        settings = self.get_bot_settings()
        if key not in settings:
            raise KeyError(f"Bot setting '{key}' not found in Supabase config.")
        return settings[key]

    def set_bot_setting(self, key: str, value) -> bool:
        """Set a specific bot setting."""
        try:
            settings = self.get_bot_settings()
            settings[key] = value
            self.client.table('config')\
                .upsert({'key': 'bot_settings', 'value': settings})\
                .execute()
            logger.info(f"Updated bot setting {key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Error setting bot setting {key}: {e}")
            return False

    def set_bot_settings(self, settings: dict) -> bool:
        """Set multiple bot settings at once."""
        try:
            current = self.get_bot_settings()
            current.update(settings)
            self.client.table('config')\
                .upsert({'key': 'bot_settings', 'value': current})\
                .execute()
            logger.info(f"Updated bot settings: {settings}")
            return True
        except Exception as e:
            logger.error(f"Error setting bot settings: {e}")
            return False
