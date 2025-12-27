"""Quick test to debug Telegram connection issues"""
import asyncio
import httpx
from config_supabase import SupabaseConfig

async def test_connection():
    config = SupabaseConfig()
    token = config.bot_token
    
    # Mask token for display
    masked = token[:10] + "..." + token[-5:] if len(token) > 15 else "INVALID"
    print(f"Bot token (masked): {masked}")
    print(f"Token length: {len(token)}")
    
    # Test basic internet connectivity
    print("\n1. Testing basic internet connectivity...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://httpbin.org/ip")
            print(f"   httpbin.org: OK - Your IP: {r.json().get('origin', 'unknown')}")
    except Exception as e:
        print(f"   httpbin.org: FAILED - {type(e).__name__}: {e}")
    
    # Test Telegram API connectivity
    print("\n2. Testing Telegram API connectivity...")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"https://api.telegram.org/bot{token}/getMe")
            if r.status_code == 200:
                data = r.json()
                if data.get("ok"):
                    bot = data["result"]
                    print(f"   Telegram API: OK")
                    print(f"   Bot username: @{bot.get('username')}")
                    print(f"   Bot ID: {bot.get('id')}")
                else:
                    print(f"   Telegram API: ERROR - {data}")
            else:
                print(f"   Telegram API: HTTP {r.status_code} - {r.text[:200]}")
    except httpx.ConnectTimeout:
        print("   Telegram API: TIMEOUT - Cannot connect to api.telegram.org")
        print("   Possible causes:")
        print("   - Telegram is blocked in your network/country")
        print("   - Firewall blocking the connection")
        print("   - VPN needed or VPN causing issues")
    except Exception as e:
        print(f"   Telegram API: FAILED - {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
