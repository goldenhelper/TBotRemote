"""Quick test script to check Supabase connection and data."""
from config_supabase import SupabaseConfig
from supabase import create_client

config = SupabaseConfig()
client = create_client(config.supabase_url, config.supabase_key)

print(f"Supabase URL: {config.supabase_url}")
print(f"Default Role ID from config: {config.default_role_id}")
print()

# Check roles table
print("=== Roles in database ===")
result = client.table('roles').select('*').execute()
if result.data:
    for role in result.data:
        print(f"  role_id: {role['role_id']}")
        print(f"  role_name: {role['role_name']}")
        print(f"  is_global: {role['is_global']}")
        print()
else:
    print("  No roles found!")

# Check if the specific role exists
print(f"=== Looking for role_id: {config.default_role_id} ===")
result = client.table('roles').select('*').eq('role_id', config.default_role_id).execute()
if result.data:
    print(f"  Found: {result.data[0]}")
else:
    print("  NOT FOUND!")

# Check chats table
print()
print("=== Chats in database ===")
result = client.table('chats').select('*').execute()
if result.data:
    for chat in result.data:
        print(f"  {chat}")
else:
    print("  No chats yet")

# Fix chats with NULL current_role_id
print()
print("=== Fixing chats with NULL current_role_id ===")
result = client.table('chats').select('chat_id, current_role_id').execute()
for chat in result.data:
    if chat.get('current_role_id') is None:
        print(f"  Fixing chat {chat['chat_id']}...")
        client.table('chats').update({
            'current_role_id': config.default_role_id
        }).eq('chat_id', chat['chat_id']).execute()
        print(f"  Set current_role_id to {config.default_role_id}")
print("Done!")
