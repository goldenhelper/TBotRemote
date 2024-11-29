from telegram import Update
from telegram.ext import ContextTypes
from telegram.error import BadRequest
import logging
from datetime import datetime
from typing import Optional
from models import ChatMessage
from services.token_manager import TokenManager
from services.chat_history import ChatHistoryManager
from services.claude_service import ClaudeService

logger = logging.getLogger(__name__)

OUTPUT_TOKENS_MULTIPLIER = 5

HELP_COMMAND_TEXT = "Хелпа"

class MainHandler:
    def __init__(self, 
                 token_manager: TokenManager,
                 chat_history: ChatHistoryManager,
                 claude_service: ClaudeService,
                 admin_user_id: str,
                 system_prompt: str):
        self.token_manager = token_manager
        self.chat_history = chat_history
        self.claude_service = claude_service
        self.admin_user_id = admin_user_id
        self.system_prompt = system_prompt


    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            if not await self._check_message_relevance(update.message, context.bot):
                return

            user = update.message.from_user
            user_id = str(user.id)

            balance, _ = await self.token_manager.get_tokens(user_id)
            if balance is None:
                await update.message.reply_text(f"Сори, бро. Походу, мы с тобой еще не общались. Начни с /start.")
                return
            
            chat_id = update.effective_chat.id

            # Handle admin responses
            if str(chat_id) == self.admin_user_id and update.message.reply_to_message:
                if await self._handle_admin_response(update, context):
                    return


            await self._process_user_message(update, context)

        except Exception as e:
            logger.error(f"Error in handle_message: {e}", exc_info=True)
            await self._send_error_message(update, context, str(e))
    

    async def _check_message_relevance(self, message, bot) -> bool:
        """Check if the message should be processed by the bot"""
        bot_username = bot.username  # Add this line
        is_reply_to_bot = (
            message.reply_to_message and 
            message.reply_to_message.from_user and 
            message.reply_to_message.from_user.id == bot.id
        )
        is_bot_mentioned = bot_username and f"@{bot_username}" in message.text
        is_private_chat = message.chat.type == "private"

        return is_reply_to_bot or is_bot_mentioned or is_private_chat


    async def _send_message_to_admin(self, message: str, bot) -> None:
        """Send a message to the admin"""
        await bot.send_message(chat_id=self.admin_user_id, text=message)


    async def _forward_to_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward the message to the admin"""
        await update.message.forward(chat_id=self.admin_user_id)


    async def _send_error_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, error_message: str) -> None:
        await update.message.reply_text("Произошла какая-то хрень. Мишка уведомлен. Считайте проблему уже почти решенной.")
        await self._send_message_to_admin(f"Something has gone wrong, hehehe. Error: {error_message}", context.bot)
        await self._forward_to_admin(update, context)


    async def _record_chat_history(self, user_message: ChatMessage, assistant_message: ChatMessage) -> None:
        await self.chat_history.save_message(user_message)
        await self.chat_history.save_message(assistant_message)


    async def _deduct_tokens(self, user_id: str, input_tokens: int, output_tokens: int) -> None:
        await self.token_manager.use_tokens(user_id, input_tokens + 5 * output_tokens)


    async def _process_user_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.message.from_user
        user_id = str(user.id)
        text = update.message.text
        username = user.username or user.first_name

        reply_to_message = update.message.reply_to_message
        if reply_to_message and reply_to_message.from_user.id == context.bot.id and reply_to_message.text.startswith(HELP_COMMAND_TEXT):
            await self._send_message_to_admin(
                f"HelpRequest:{update.effective_chat.id}:{update.message.message_id}:{user_id}\n"
                f"User {username} asked for HEEEEELP.", context.bot)
            await self._forward_to_admin(update, context)
            return

        # check if user has enough tokens
        balance, _ = await self.token_manager.get_tokens(user_id)
        
        if balance > 0:
            try:
                response_text, input_tokens, output_tokens = await self.claude_service.get_response(
                    text, self.system_prompt
                )

                if response_text == "DONT KNOW":
                    await self._send_message_to_admin("The model doesn't know the answer.", context.bot)
                    await self._forward_to_admin(update, context)   
                else:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=response_text,
                        reply_to_message_id=update.message.message_id
                    )

                    await self._deduct_tokens(user_id, input_tokens, output_tokens)

                    await self._record_chat_history(
                        user_message=ChatMessage.from_telegram_message(update.message),
                        assistant_message=ChatMessage(context.bot.id, context.bot.username, response_text, is_bot_response=True)
                    )

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await self._send_error_message(update, context, str(e))
        else:
            await update.message.reply_text(f"Закончились у тебя токены... Мда, не повезло. Но ходят слухи, что если использовать команду /ask_for_tokens, Мишка добавит тебе еще.")


    async def _handle_admin_response(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle responses from admin to user questions or token requests"""
        original_message_text = update.message.reply_to_message.text
        text = update.message.text

        # Handle token requests
        if original_message_text.startswith("TokenRequest:"):
            _, original_chat_id, user_message_id, user_id = original_message_text.split("\n")[0].split(":")
            await self._handle_admin_answer_to_token_request(context, update, original_chat_id, user_message_id, user_id, text)
            return True
        
        # Handle help requests
        if original_message_text.startswith("HelpRequest:"):
            _, original_chat_id, user_message_id, user_id = original_message_text.split("\n")[0].split(":")
            
            await self._handle_admin_answer_to_help_request(context, update, original_chat_id, user_message_id, user_id, text)
            return True
        
        # Handle other admin responses here...
        print("Other admin response")
        return False
    

    async def _handle_admin_answer_to_token_request(self, context, update, chat_id, message_id, user_id, text):
        """Process admin's response to token request"""
        if not text.isdigit():
            await update.message.reply_text("A number is expected!")
            return

        tokens = int(text)
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Мишка был настолько щедрым, что добавил тебе {tokens} токенов!",
            reply_to_message_id=message_id
        )
        await self.token_manager.add_tokens(user_id, tokens)
        await update.message.reply_text("Tokens have been added to the user!")


    async def _handle_admin_answer_to_help_request(self, context, update, chat_id, message_id, user_id, text):
        """Process admin's response to help request"""
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"Мишка соблаговолил потратить свое бесценное время на ответ:\n{text}",
            reply_to_message_id=message_id
        )


    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        await update.message.reply_text(text = f"Привет, я ботик-обормотик. Спроси меня что-нибудь о Мишке!")
        user = update.message.from_user

        username = user.username or user.first_name
        if username is None:
            username = "unknown"
        
        await self.token_manager.add_user_if_not_exists(user.id, username)


    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        await update.message.reply_text(
            text = (
                f"{HELP_COMMAND_TEXT}\n"
                "Список команд:\n"
                "/start - стартуем\n"
                "/help - хелпа\n"
                "/get_tokens - показывает сколько еще вы сможете узнавать прекрасное о Мишке\n"
                "/ask_for_tokens - поныть на токены и, может, Мишка скинет вам еще\n"
                "Если вы хотите задать вопрос Мишке, сформулируйте его, подумайте, подумайте, подумайте...\n"
                "Если не помогло, то можете ответить на это сообщение вопросом и, возможно, Мишка ответит."
            )
        )
        

    def command_for_identified_users(func):
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_id = str(update.message.from_user.id)
            balance, _ = await self.token_manager.get_tokens(user_id)
            if balance is None:
                await update.message.reply_text(f"Сори, бро. Походу, мы с тобой еще не общались. Начни с /start.")
                return
            return await func(self, update, context)
        return wrapper


    @command_for_identified_users
    async def get_tokens_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /get_tokens is issued."""
        user_id = str(update.message.from_user.id)
        remaining_tokens, username = await self.token_manager.get_tokens(user_id)
        await update.message.reply_text(f"У вас {remaining_tokens} токенов.")


    @command_for_identified_users
    async def ask_for_tokens_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /ask_for_tokens is issued."""
        user = update.message.from_user
        user_id = str(user.id)
        chat_id = update.effective_chat.id

        await context.bot.send_message(
            chat_id=self.admin_user_id,
            text=f"TokenRequest:{chat_id}:{update.message.message_id}:{user_id}\n"
                 f"User {user.username or user.first_name} asked for tokens. "
                 f"Their current amount is {(await self.token_manager.get_tokens(user_id))[0]}"
        )
        await update.message.reply_text(f"Мишка уведомлен о вашей просьбе. Ждите милости от него.")