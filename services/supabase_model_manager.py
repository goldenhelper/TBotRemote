"""
Supabase-based model manager for the Telegram bot.
Replaces DynamoDB-based ModelManager with Supabase (PostgreSQL).
"""
import logging
from typing import Optional
from datetime import datetime

from supabase import create_client, Client
from models import TokenTransaction, TransactionStatus, TransactionType
from exceptions import TokenError
from utils.constants import DEFAULT_TOKENS_FOR_NEW_USERS, ALLOWED_MODELS_LIMITS

logger = logging.getLogger(__name__)


class SupabaseModelManager:
    """
    Manages tokens and model usage using Supabase.
    Replaces the DynamoDB-based ModelManager.
    """

    def __init__(self, supabase_url: str, supabase_key: str,
                 default_model: str = 'gemini-2.0-flash',
                 default_role_id: str = None,
                 default_memory_updater_model: str = None,
                 default_tokens_for_new_chats: int = 6000):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.default_model = default_model
        self.default_role_id = default_role_id
        self.default_memory_updater_model = default_memory_updater_model or default_model
        self.default_tokens_for_new_chats = default_tokens_for_new_chats

    def _load_allowed_models_limits(self) -> dict:
        """Load allowed models limits from config table with fallback to local constants."""
        try:
            result = self.client.table('config')\
                .select('value')\
                .eq('key', 'allowed_models_limits')\
                .execute()

            if result.data:
                return result.data[0]['value']
            return ALLOWED_MODELS_LIMITS
        except Exception:
            return ALLOWED_MODELS_LIMITS

    def reset_model_data(self):
        """Reset query counts to 0 for all known models."""
        limits = self._load_allowed_models_limits()
        for provider_models in limits.values():
            for model_name in provider_models.keys():
                self.client.table('model_usage').upsert({
                    'model_name': model_name,
                    'query_count': 0,
                }).execute()

    def used_model(self, model_name: str):
        """Increment usage counter for a model."""
        # First, try to get existing count
        result = self.client.table('model_usage')\
            .select('query_count')\
            .eq('model_name', model_name)\
            .execute()

        if result.data:
            current_count = result.data[0]['query_count']
            self.client.table('model_usage')\
                .update({'query_count': current_count + 1})\
                .eq('model_name', model_name)\
                .execute()
        else:
            # Insert new record
            self.client.table('model_usage').insert({
                'model_name': model_name,
                'query_count': 1,
            }).execute()

    def _get_model_query_count(self, model_name: str) -> int:
        """Return current query_count for a model, defaulting to 0 if absent."""
        result = self.client.table('model_usage')\
            .select('query_count')\
            .eq('model_name', model_name)\
            .execute()

        if result.data:
            return int(result.data[0].get('query_count', 0))
        return 0

    def get_model_usage(self, model_name: str) -> int:
        """Public method to get current usage for a model."""
        return self._get_model_query_count(model_name)

    def get_all_model_usage(self) -> dict:
        """Return usage counts for all known models from limits."""
        limits = self._load_allowed_models_limits()
        usage: dict[str, int] = {}
        for provider_models in limits.values():
            for model_name in provider_models.keys():
                usage[model_name] = self._get_model_query_count(model_name)
        return usage

    def get_allowed_models_limits(self) -> dict:
        """Expose allowed models limits mapping."""
        return self._load_allowed_models_limits()

    def best_allowed_model(self, model_name: str) -> str:
        """Pick the first model under its allowed limit for the given provider family."""
        requested = model_name.lower()
        limits = self._load_allowed_models_limits()

        if requested in limits['gemini']:
            provider_key = 'gemini'
        elif requested in limits['claude']:
            provider_key = 'claude'
        elif requested in limits['openai']:
            provider_key = 'openai'
        else:
            raise ValueError(f"Unknown model family: {model_name}")

        provider_limits = limits.get(provider_key, {})
        for candidate_model, limit in provider_limits.items():
            if self._get_model_query_count(candidate_model) < int(limit):
                return candidate_model

        return ''

    async def get_tokens(self, chat_id: int) -> Optional[int]:
        """Get the number of tokens for a chat."""
        try:
            result = self.client.table('chats')\
                .select('tokens')\
                .eq('chat_id', chat_id)\
                .execute()

            if not result.data:
                return None

            return int(result.data[0].get('tokens', 0))
        except Exception as e:
            logger.error(f"Error getting tokens for chat {chat_id}: {e}")
            raise TokenError(f"Failed to get token balance: {str(e)}")

    async def use_tokens(self, chat_id: int, amount: int) -> bool:
        """Use a certain amount of tokens for a chat."""
        try:
            transaction = TokenTransaction(
                chat_id=chat_id,
                amount=amount,
                timestamp=datetime.now(),
                transaction_type=TransactionType.DEDUCTION,
                status=TransactionStatus.PENDING,
            )
            await self._atomic_token_operation(transaction)
            return True
        except TokenError:
            raise
        except Exception as e:
            logger.error(f"Error using tokens for chat {chat_id}: {e}")
            raise

    async def add_tokens(self, chat_id: int, amount: int) -> None:
        """Add a certain amount of tokens to a chat."""
        try:
            transaction = TokenTransaction(
                chat_id=chat_id,
                amount=amount,
                timestamp=datetime.now(),
                transaction_type=TransactionType.ADDITION,
                status=TransactionStatus.PENDING,
            )
            await self._atomic_token_operation(transaction)
        except TokenError:
            raise
        except Exception as e:
            logger.error(f"Error adding tokens for chat {chat_id}: {e}")
            raise TokenError(f"Failed to add tokens: {str(e)}")

    async def _atomic_token_operation(self, transaction: TokenTransaction) -> None:
        """Perform an atomic token operation."""
        try:
            balance = await self.get_tokens(transaction.chat_id)
            if balance is None:
                raise TokenError("Chat has not been added")

            if transaction.transaction_type == TransactionType.DEDUCTION:
                transaction.amount = min(transaction.amount, balance)

                if balance == 0:
                    raise TokenError("Balance is 0, cannot deduct tokens")

                new_balance = balance - transaction.amount
            else:
                new_balance = balance + transaction.amount

            # Update tokens
            self.client.table('chats')\
                .update({'tokens': new_balance})\
                .eq('chat_id', transaction.chat_id)\
                .execute()

        except TokenError:
            raise
        except Exception as e:
            logger.error(f"Operation failed with error: {str(e)}")
            raise

    async def add_chat_if_not_exists(self, chat_id: int) -> None:
        """Add a chat to the table if it doesn't exist."""
        result = self.client.table('chats')\
            .select('chat_id')\
            .eq('chat_id', chat_id)\
            .execute()

        if not result.data:
            insert_data = {
                'chat_id': chat_id,
                'tokens': self.default_tokens_for_new_chats,
                'model': self.default_model,
                'memory_updater_model': self.default_memory_updater_model,
            }
            if self.default_role_id:
                insert_data['current_role_id'] = self.default_role_id
            self.client.table('chats').insert(insert_data).execute()

    async def set_tokens(self, chat_id: int, amount: int) -> None:
        """Set tokens to a specific amount (admin function)."""
        result = self.client.table('chats')\
            .select('chat_id')\
            .eq('chat_id', chat_id)\
            .execute()

        if result.data:
            self.client.table('chats')\
                .update({'tokens': amount})\
                .eq('chat_id', chat_id)\
                .execute()
        else:
            self.client.table('chats').insert({
                'chat_id': chat_id,
                'tokens': amount,
            }).execute()
