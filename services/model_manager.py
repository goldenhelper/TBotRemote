import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
import logging
from datetime import datetime
from typing import Optional
import asyncio
from models import TokenTransaction, TransactionStatus, TransactionType
from exceptions import TokenError
import json
from collections import OrderedDict
from utils.constants import DEFAULT_TOKENS_FOR_NEW_USERS

logger = logging.getLogger(__name__)

PARAMETER_NAME = "/telegram-bot/model_data"

DEBUG = True

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class ModelManager:
    def __init__(self, token_table_name: str, model_usage_table_name: str, allowed_models_limits_resource_name: str, region: str):
        self.dynamodb = boto3.resource('dynamodb', region_name=region)
        self.token_table = self.dynamodb.Table(token_table_name)
        self.model_usage_table = self.dynamodb.Table(model_usage_table_name)    

        # Parameter Store resource name for allowed models limits
        self.ssm_client = boto3.client('ssm', region_name=region)
        self.allowed_models_limits_resource_name = allowed_models_limits_resource_name

    def get_model_data(self):
        
        response = self.model_usage_table.get_item(Key={'model_name': 'all'})
        return response['Item']
        
        
    def reset_model_data(self):
        """Reset all query counts to 0."""
        self.model_usage_table.update_item(
            Key={'model_name': 'all'},
            UpdateExpression='SET query_count = :zero',
            ExpressionAttributeValues={':zero': 0}
        )


    def used_model(self, model_name: str):
        # increment query count for the model
        self.model_usage_table.update_item(
            Key={'model_name': model_name},
            UpdateExpression='SET query_count = query_count + :one',
            ExpressionAttributeValues={':one': 1}
        )
        

    def best_allowed_model(self, model_name: str) -> str:
        """Find the best allowed model based on the query count."""
        model = None

        # Get allowed models limits from Parameter Store
        response = self.ssm_client.get_parameter(Name=self.allowed_models_limits_resource_name)
        allowed_models_limits = json.loads(response['Parameter']['Value'])

        if model_name.lower().startswith("gemini"):
            for model, limit in allowed_models_limits['gemini'].items():
                if self.get_model_data()[model] < limit:
                    return model
            
        elif model_name.lower().startswith("claude"):
            for model, limit in allowed_models_limits['claude'].items():
                if self.get_model_data()[model] < limit:
                    return model
        elif model_name.lower() in allowed_models_limits['openai']:
            for model, limit in allowed_models_limits['openai'].items():
                if self.get_model_data()[model] < limit:
                    return model
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # all model limits have been surpassed:
        if model is None:
            return ''


    async def get_tokens(self, chat_id: int) -> int:
        """Get the number of tokens for a user and their username."""

        try:
            # Run DynamoDB operation in a thread pool since boto3 is not async native
            loop = asyncio.get_running_loop()
            
            response = await loop.run_in_executor(
                None, 
                self._dynamo_get_tokens,
                chat_id
            )
           
            item = response.get('Item', None)

            if item is None:
                return None

            return int(item.get('tokens', 'wth'))
        except Exception as e:
            logger.error(f"Error getting tokens for chat {chat_id}: {e}")
            raise TokenError(f"Failed to get token balance: {str(e)}")


    async def use_tokens(self, chat_id: int, amount: int) -> bool:
        """Use a certain amount of tokens for a user."""
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
        except TokenError as e:
            raise
        except Exception as e:
            logger.error(f"Error using tokens for user {chat_id}: {e}")
            raise


    async def add_tokens(self, chat_id: int, amount: int) -> None:
        """Add a certain amount of tokens to a user."""
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
            logger.error(f"Error adding tokens for user {chat_id}: {e}")
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
                    raise TokenError(f"Something could have gone wrong, balance is 0, but transaction type is deduction")

            

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
                    transaction.chat_id,
                    update_expr,
                    expr_values,
                    condition_expr
                )
            )
            

        except ClientError as e:
            log(f"DynamoDB error: {e.response['Error']['Code']}")
            if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
                raise TokenError("Insufficient tokens")
            raise
        except Exception as e:
            log(f"Operation failed with error: {str(e)}")
            raise


    async def add_chat_if_not_exists(self, chat_id: int) -> None:
        """Add a user to the table if they don't exist."""
        loop = asyncio.get_running_loop()
        chat_id = str(chat_id)

        # Check if user exists
        response = await loop.run_in_executor(
            None,
            self._dynamo_get_tokens,
            chat_id
        )
        
        # If user doesn't exist, add them
        if not response.get('Item'):
            await loop.run_in_executor(
                None,
                lambda: self.token_table.put_item(Item={'chat_id': chat_id, 'tokens': DEFAULT_TOKENS_FOR_NEW_USERS})
            )


    def _dynamo_get_tokens(self, chat_id: int):
        """Synchronous DynamoDB get operation."""
        
        return self.token_table.get_item(Key={'chat_id': str(chat_id)})
        

    def _dynamo_update_tokens(self, chat_id: int, update_expr: str, expr_values: dict, condition_expr: Optional[str] = None):
        """Synchronous DynamoDB update operation."""
        params = {
            'Key': {'chat_id': str(chat_id)},
            'UpdateExpression': update_expr,
            'ExpressionAttributeValues': expr_values
        }
        if condition_expr:
            params['ConditionExpression'] = condition_expr
            
        return self.token_table.update_item(**params)