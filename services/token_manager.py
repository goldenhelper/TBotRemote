import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
import logging
from datetime import datetime
from typing import Optional
import asyncio
from models import TokenTransaction, TransactionStatus, TransactionType
from exceptions import TokenError

logger = logging.getLogger(__name__)

class TokenManager:
    def __init__(self, table_name: str, region: str):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.table = self.dynamodb.Table(table_name)

    async def get_tokens(self, user_id: str) -> tuple[int, Optional[str]]:
        """Get the number of tokens for a user and their username."""
        try:
            # Run DynamoDB operation in a thread pool since boto3 is not async native
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                self._dynamo_get_tokens,
                user_id
            )
            item = response.get('Item', None)

            if item is None:
                return None, None

            return int(item.get('tokens', 'wth')), item.get('username')
        except Exception as e:
            logger.error(f"Error getting tokens for user {user_id}: {e}")
            raise TokenError(f"Failed to get token balance: {str(e)}")

    async def use_tokens(self, user_id: str, amount: int) -> bool:
        """Use a certain amount of tokens for a user."""
        try:
            transaction = TokenTransaction(
                user_id=user_id,
                amount=amount,
                timestamp=datetime.now(),
                transaction_type=TransactionType.DEDUCTION,
                status=TransactionStatus.PENDING,
            )
            await self._atomic_token_operation(transaction)
            return True
        except TokenError:
            return False
        except Exception as e:
            logger.error(f"Error using tokens for user {user_id}: {e}")
            return False

    async def add_tokens(self, user_id: str, amount: int) -> None:
        """Add a certain amount of tokens to a user."""
        try:
            transaction = TokenTransaction(
                user_id=user_id,
                amount=amount,
                timestamp=datetime.now(),
                transaction_type=TransactionType.ADDITION,
                status=TransactionStatus.PENDING,
            )
            await self._atomic_token_operation(transaction)
        except TokenError:
            raise
        except Exception as e:
            logger.error(f"Error adding tokens for user {user_id}: {e}")
            raise TokenError(f"Failed to add tokens: {str(e)}")

    async def _atomic_token_operation(self, transaction: TokenTransaction) -> None:
        """Perform an atomic token operation."""
        balance, _ = await self.get_tokens(transaction.user_id)
        if balance is None:
            raise TokenError("User does not exist")

        try:
            update_expr = 'ADD tokens :val'
            expr_values = {
                ':val': transaction.amount if transaction.transaction_type == TransactionType.ADDITION else -transaction.amount
            }

            if transaction.transaction_type == TransactionType.DEDUCTION:
                expr_values[':min'] = transaction.amount
                condition_expr = 'tokens >= :min'
            else:
                condition_expr = None

            # Run DynamoDB operation in a thread pool
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._dynamo_update_tokens(
                    transaction.user_id,
                    update_expr,
                    expr_values,
                    condition_expr
                )
            )

        except ClientError as e:
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise TokenError("Insufficient tokens")
            raise

    async def add_user_if_not_exists(self, user_id, username: str) -> None:
        """Add a user to the table if they don't exist."""
        loop = asyncio.get_running_loop()
        user_id = str(user_id)

        # Check if user exists
        response = await loop.run_in_executor(
            None,
            self._dynamo_get_tokens,
            user_id
        )
        
        # If user doesn't exist, add them
        if not response.get('Item'):
            await loop.run_in_executor(
                None,
                lambda: self.table.put_item(Item={'user_id': user_id, 'name': username, 'tokens': 0})
            )

    def _dynamo_get_tokens(self, user_id: str):
        """Synchronous DynamoDB get operation."""
        return self.table.get_item(Key={'user_id': str(user_id)})

    def _dynamo_update_tokens(self, user_id: str, update_expr: str, expr_values: dict, condition_expr: Optional[str] = None):
        """Synchronous DynamoDB update operation."""
        params = {
            'Key': {'user_id': str(user_id)},
            'UpdateExpression': update_expr,
            'ExpressionAttributeValues': expr_values
        }
        if condition_expr:
            params['ConditionExpression'] = condition_expr
            
        return self.table.update_item(**params)