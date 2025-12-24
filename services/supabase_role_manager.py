"""
Supabase-based role manager for the Telegram bot.
Replaces DynamoDB-based RoleManager with Supabase (PostgreSQL).
"""
from typing import Dict, List, Optional
import uuid
import time
import logging

from supabase import create_client, Client
from services.role_manager import Role

logger = logging.getLogger(__name__)


class SupabaseRoleManager:
    """
    Manages bot roles/personalities using Supabase.
    Replaces the DynamoDB-based RoleManager.
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.global_roles_cache: Optional[Dict[str, Role]] = None
        self.global_roles_ttl: float = 0

    @property
    def default_role_id(self) -> Optional[str]:
        """Get default role ID from bot settings."""
        try:
            result = self.client.table('config')\
                .select('value')\
                .eq('key', 'bot_settings')\
                .execute()

            if result.data:
                return result.data[0]['value'].get('default_role_id')
            return None
        except Exception:
            return None

    async def get_global_roles(self) -> Dict[str, Role]:
        """
        Retrieve global roles with caching for performance.
        """
        current_time = time.time()

        # Check if cache is valid
        if self.global_roles_cache is None or current_time > self.global_roles_ttl:
            try:
                result = self.client.table('roles')\
                    .select('*')\
                    .eq('is_global', True)\
                    .execute()

                self.global_roles_cache = {
                    item['role_id']: Role(
                        name=item['role_name'],
                        prompt=item['role_prompt'],
                        is_global=True
                    )
                    for item in result.data
                }
                # Cache for 5 minutes
                self.global_roles_ttl = current_time + 300
            except Exception as e:
                logger.error(f"Error retrieving global roles: {str(e)}")
                return {}

        return self.global_roles_cache

    async def get_role_by_id(self, role_id: str) -> Optional[Role]:
        """Returns role based on its ID."""
        try:
            result = self.client.table('roles')\
                .select('*')\
                .eq('role_id', role_id)\
                .execute()

            if result.data:
                item = result.data[0]
                return Role(
                    name=item['role_name'],
                    prompt=item['role_prompt'],
                    is_global=item.get('is_global', False)
                )
            return None
        except Exception as e:
            logger.error(f"Error getting role {role_id}: {str(e)}")
            return None

    async def get_roles_by_ids(self, role_ids: List[str]) -> Dict[str, Role]:
        """Returns dict of roles based on their IDs."""
        roles = {}

        if not role_ids:
            return roles

        if isinstance(role_ids, str):
            role_ids = [role_ids]

        try:
            result = self.client.table('roles')\
                .select('*')\
                .in_('role_id', role_ids)\
                .execute()

            for item in result.data:
                roles[item['role_id']] = Role(
                    name=item['role_name'],
                    prompt=item['role_prompt'],
                    is_global=item.get('is_global', False)
                )
        except Exception as e:
            logger.error(f"Error retrieving roles by IDs: {str(e)}")

        return roles

    async def get_defualt_role(self) -> Optional[Role]:
        """Get default role."""
        return await self.get_role_by_id(self.default_role_id)

    async def get_role_by_name(self, role_name: str) -> List[Role]:
        """Get roles by name."""
        try:
            result = self.client.table('roles')\
                .select('*')\
                .eq('role_name', role_name)\
                .execute()

            return [
                Role(
                    name=item['role_name'],
                    prompt=item['role_prompt'],
                    is_global=item.get('is_global', False)
                )
                for item in result.data
            ]
        except Exception as e:
            logger.error(f"Error getting role by name {role_name}: {str(e)}")
            return []

    def add_new_role(self, role: Role) -> str:
        """Adds new role and returns its id."""
        role_id = str(uuid.uuid4())

        self.client.table('roles').insert({
            'role_id': role_id,
            'role_name': role.name,
            'role_prompt': role.prompt,
            'is_global': role.is_global,
        }).execute()

        # Invalidate cache if adding a global role
        if role.is_global:
            self.global_roles_cache = None

        return role_id

    def set_role_global_status(self, role_id: str, is_global: bool) -> None:
        """Set a role's global status."""
        self.client.table('roles')\
            .update({'is_global': is_global})\
            .eq('role_id', role_id)\
            .execute()

        # Invalidate cache
        self.global_roles_cache = None

    def make_role_global(self, role_id: str) -> None:
        """Make a role global."""
        self.set_role_global_status(role_id, True)

    def make_role_non_global(self, role_id: str) -> None:
        """Make a role non-global."""
        self.set_role_global_status(role_id, False)

    async def get_global_role_ids(self) -> List[str]:
        """Get IDs of all global roles."""
        return list((await self.get_global_roles()).keys())

    def delete_role(self, role_id: str) -> None:
        """Delete a role by ID."""
        self.client.table('roles')\
            .delete()\
            .eq('role_id', role_id)\
            .execute()

        # Invalidate cache
        self.global_roles_cache = None

    def update_role(self, role_id: str, name: Optional[str] = None, prompt: Optional[str] = None) -> None:
        """Update a role's name and/or prompt."""
        updates = {}
        if name is not None:
            updates['role_name'] = name
        if prompt is not None:
            updates['role_prompt'] = prompt

        if updates:
            self.client.table('roles')\
                .update(updates)\
                .eq('role_id', role_id)\
                .execute()

            # Invalidate cache in case it was a global role
            self.global_roles_cache = None
