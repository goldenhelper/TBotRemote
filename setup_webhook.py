# setup_webhook.py - Run locally to set up Telegram webhook

import asyncio
import os
from telegram import Bot

# === CONFIGURE THIS ===
# Your Render URL (get this from the Render dashboard after deployment)
RENDER_URL = "https://your-bot-name.onrender.com"  # Change this!

# Your bot token (or set as environment variable)
BOT_TOKEN = os.environ.get("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
# ======================

webhook_url = f"{RENDER_URL}/webhook"


async def setup_webhook():
    bot = Bot(token=BOT_TOKEN)
    
    # Set webhook
    await bot.set_webhook(webhook_url)
    
    # Print webhook info
    webhook_info = await bot.get_webhook_info()
    print(f"✅ Webhook is set to: {webhook_info.url}")


async def delete_webhook():
    bot = Bot(token=BOT_TOKEN)
    await bot.delete_webhook()
    print("🗑️ Webhook deleted")


async def get_webhook_info():
    bot = Bot(token=BOT_TOKEN)
    webhook_info = await bot.get_webhook_info()
    print(f"📡 Current webhook: {webhook_info.url or 'None'}")
    print(f"   Pending updates: {webhook_info.pending_update_count}")
    if webhook_info.last_error_message:
        print(f"   Last error: {webhook_info.last_error_message}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "delete":
            asyncio.run(delete_webhook())
        elif sys.argv[1] == "info":
            asyncio.run(get_webhook_info())
        else:
            print("Usage: python setup_webhook.py [delete|info]")
    else:
        asyncio.run(setup_webhook())