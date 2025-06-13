from dataclasses import dataclass
from typing import Dict, List
import boto3
import json
import uuid
import logging
import time
from boto3.dynamodb.conditions import Attr

logger = logging.getLogger()

#NOTE: think whether this behavior is okay

class Role:
    def __init__(self, name, prompt, is_global=False):
        self.name = name
        self.prompt = prompt
        self.is_global = is_global
        

    @classmethod
    def from_dict(cls, data):
        # Map DynamoDB item fields to constructor parameters
        return cls(
            name=data.get('role_name', data.get('name')),
            prompt=data.get('role_prompt', data.get('prompt')),
            is_global=data.get('is_global', False)
        )

    def to_dict(self):
        return {
            'name': self.name,
            'prompt': self.prompt,
            'is_global': self.is_global
        }

    def __str__(self):
        return self.name
    
    def make_global(self):
        """Set this role as global"""
        self.is_global = True
        
    def make_non_global(self):
        """Set this role as non-global"""
        self.is_global = False



class RoleManager:
    def __init__(self, table_name: str, aws_region: str, default_role_id: str):
        self.dynamodb = boto3.resource('dynamodb', region_name=aws_region)
        self.table = self.dynamodb.Table(table_name)
        self.default_role_id = default_role_id
        self.global_roles_cache = None
        self.global_roles_ttl = 0

    async def get_global_roles(self) -> Dict[str, Role]:
        """
        Retrieve global roles with caching for performance.
        
        Returns:
            list: List of role items with is_global=True
        """
        current_time = time.time()
        
        # Check if cache is valid
        if self.global_roles_cache is None or current_time > self.global_roles_ttl:
            try:
                # Query for global roles
                response = self.table.scan(FilterExpression=Attr('is_global').eq(True))
                self.global_roles_cache = {item['role_id']: Role.from_dict(item) for item in response['Items']}
                # Cache for 5 minutes
                self.global_roles_ttl = current_time + 300
            except Exception as e:
                # Log the error and return empty list if there's an issue
                print(f"Error retrieving global roles: {str(e)}")
                return {}
        
        print(self.global_roles_cache)
        return self.global_roles_cache

    async def get_role_by_id(self, role_id: str) -> Role:
        """Returns role based on their ID""" 
        return Role.from_dict(self.table.get_item(Key={'role_id': role_id})['Item'])

    async def get_roles_by_ids(self, role_ids: List[str]) -> Dict[str, Role]:
        """Returns dict of roles based on their IDs""" 
        roles = {}
        
        # Handle empty list case
        if not role_ids:
            return roles
            
        # If it's a single ID (string), convert to list
        if isinstance(role_ids, str):
            role_ids = [role_ids]
            
        try:
            # For small lists, we can use a scan with a filter
            if len(role_ids) <= 20:  # DynamoDB has limits on filter expressions
                filter_conditions = []
                for role_id in role_ids:
                    filter_conditions.append(boto3.dynamodb.conditions.Attr('role_id').eq(role_id))
                
                # Combine conditions with OR
                filter_expression = filter_conditions[0]
                for condition in filter_conditions[1:]:
                    filter_expression = filter_expression | condition
                
                response = self.table.scan(FilterExpression=filter_expression)
                roles = {item['role_id']: Role.from_dict(item) for item in response.get('Items', [])}
            else:
                # For larger lists, fetch items individually and combine results
                for role_id in role_ids:
                    response = self.table.scan(
                        FilterExpression=boto3.dynamodb.conditions.Attr('role_id').eq(role_id)
                    )
                    for item in response.get('Items', []):
                        roles[item['role_id']] = Role.from_dict(item)
        except Exception as e:
            logger.error(f"Error retrieving roles by IDs: {str(e)}")
            
        return roles

    async def get_defualt_role(self) -> Role:
        """Get default role"""

        return await self.get_role_by_id(self.default_role_id)

    async def get_role_by_name(self, role_name: str) -> Role | None:
        """Get role by its name, since name is not primary key, this is done via table scan"""
        filter_expression = boto3.dynamodb.conditions.Attr('role_name').eq(role_name)
        
        response = self.table.scan(
            FilterExpression=filter_expression
        )
        
        # Transform the list of items into a dictionary mapping role_id to role_prompt
        roles = [Role.from_dict(item) for item in response.get('Items', [])]
        
        return roles

    def add_new_role(self, role: Role) -> str:
        """Adds new role and returns its id."""
        # Generate a unique ID for the role
        role_id = str(uuid.uuid4())
        self.table.put_item(
            Item={  
                'role_id': role_id,
                'role_name': role.name,
                'role_prompt': role.prompt,
                'is_global': role.is_global
            }
        )
        return role_id

    def set_role_global_status(self, role_id: str, is_global: bool) -> None:
        """Set a role's global status"""
        self.table.update_item(
            Key={'role_id': role_id},
            UpdateExpression='SET is_global = :val',
            ExpressionAttributeValues={':val': is_global}
        )

    def make_role_global(self, role_id: str) -> None:
        """Make a role global"""
        self.set_role_global_status(role_id, True)
        
    def make_role_non_global(self, role_id: str) -> None:
        """Make a role non-global"""
        self.set_role_global_status(role_id, False)
        
    async def get_global_role_ids(self) -> List[str]:
        """Get IDs of all global roles"""
        return list((await self.get_global_roles()).keys())