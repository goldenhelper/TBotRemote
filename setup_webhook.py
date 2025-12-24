# setup_webhook.py - Run locally to set up Telegram webhook

import asyncio
import os
from telegram import Bot
from telegram.error import InvalidToken, TelegramError
from dotenv import load_dotenv

# Load local .env if present (so both RENDER_URL and BOT_TOKEN can be configured there)
load_dotenv()

# === CONFIGURE THIS ===
# Your Render URL (get this from the Render dashboard after deployment)
# You can also set env var RENDER_URL to override this without editing the file.
RENDER_URL = os.environ.get("RENDER_URL", "https://tbotremote.onrender.com")

# Your bot token (or set as environment variable)
BOT_TOKEN = (
    os.environ.get("BOT_TOKEN")
    or os.environ.get("bot_token")  # allow matching pydantic field name too
)
# ======================

webhook_url = f"{RENDER_URL}/webhook"


def _require_bot_token() -> str:
    token = (BOT_TOKEN or "").strip()
    if not token or token == "YOUR_BOT_TOKEN_HERE":
        raise SystemExit(
            "BOT_TOKEN is not set.\n"
            "Fix: set environment variable BOT_TOKEN (or add it to your .env), e.g.\n"
            "  PowerShell: $env:BOT_TOKEN='123456:ABC...'\n"
            "  .env file:  BOT_TOKEN=123456:ABC...\n"
        )
    # Very lightweight sanity check: Telegram tokens are "<digits>:<secret>"
    if ":" not in token or not token.split(":", 1)[0].isdigit():
        raise SystemExit(
            f"BOT_TOKEN doesn't look like a Telegram token: {token!r}\n"
            "Expected format: '<bot_id>:<token_secret>'"
        )
    return token


async def setup_webhook():
    bot = Bot(token=_require_bot_token())
    
    # Set webhook
    try:
        await bot.set_webhook(webhook_url)
    except InvalidToken as e:
        raise SystemExit(
            "Telegram rejected your bot token.\n"
            "Double-check BOT_TOKEN (BotFather), and make sure you restarted your shell after setting env vars."
        ) from e
    except TelegramError as e:
        raise SystemExit(f"Telegram API error while setting webhook: {e}") from e
    
    # Print webhook info
    webhook_info = await bot.get_webhook_info()
    print(f"✅ Webhook is set to: {webhook_info.url}")


async def delete_webhook():
    bot = Bot(token=_require_bot_token())
    await bot.delete_webhook()
    print("🗑️ Webhook deleted")


async def get_webhook_info():
    bot = Bot(token=_require_bot_token())
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