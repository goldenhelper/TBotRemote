# setup_webhook.py (separate script to run locally to set up webhook)

import asyncio
from config_local import LocalTestingConfig
from telegram import Bot

webhook_url = "https://229d-2-206-160-174.ngrok-free.app" + "/webhook"

async def setup_webhook():
    config = LocalTestingConfig()
    bot = Bot(token=config.bot_token)
    
    # Set webhook
    await bot.set_webhook(webhook_url)
    
    # Print webhook info
    webhook_info = await bot.get_webhook_info()
    print(f"Webhook is set to: {webhook_info.url}")

if __name__ == "__main__":
    # This is only for setting up the webhook, not for the Lambda function
    asyncio.run(setup_webhook())