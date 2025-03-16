from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.error import BadRequest
import logging
from typing import List, Callable

from urllib3 import response
from models import ChatMessage
from services.model_manager import ModelManager, allowed_models_limits
from services.claude_service import ClaudeService
from services.gemini_service import GeminiService
from services.chat_history import Storage
from services.role_manager import RoleManager, Role
import boto3
from random import random
from utils.constants import *
import re


logger = logging.getLogger(__name__)

HELP_COMMAND_TEXT = "Хелпа"

user_commands = {
    "/start": "Стартуем.", 
    "/help": "Хелпа.", 
    "/tokens_amount": "Показывает как долго ботик еще сможет терпеть этот чат.", 
    "/ask_for_tokens": "Поныть на токены и, может, Мишка зарядит ботика.",
    "/set_aliveness" : "Устанавливает активность ботика.",
    "/how_alive" : "Показывает насколько ботик живой."
}

admin_commands = {
    "/set_model" : "Меняет модельку.", 
    "/get_model_data" : "Показывает текущую модельку.", 
    "/set_role" : "Меняет роль.", 
    "/get_role" : "Показывает текущую роль.",
    "/give_bot_tokens" : "Дает боту токены.",
    "/clear_history" : "Чистка истории (но не совести))",
    "/choose_role": "Выбрать роль для бота."
}   



DEBUG = True


# NOTE: 
# for the memory updater, thinking is assumed to be true


# TODO: 
# 1. put the system prompots in a separate file
# 2. fix gemini 

def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class MainHandler:
    """
    Main handler for processing Telegram bot messages and commands.
    
    This class manages:
    - Message processing and routing
    - Model management and token usage
    - Chat history and context management
    - Role-based responses
    - Admin commands and special features
    
    Attributes:
        model_manager (ModelManager): Manages AI model selection and token usage
        chat_history_manager (ChatHistoryManager): Handles message history and context
        admin_user_id (int): Telegram ID of the bot administrator
        bot_id (int): Telegram ID of the bot itself
        formatting_info (str): Instructions for message formatting
        api_keys (dict): API keys for different AI services
        aws_region (str): AWS region for SSM client
        role_manager (RoleManager): Manages bot personality roles
        cool_user_ids (List[int]): List of privileged user IDs
    """

    def __init__(self, 
                 model_manager: ModelManager,
                 role_manager: RoleManager,
                 storage: Storage,
                 admin_user_id: int,
                 bot_id: int,
                 formatting_info: str,
                 api_keys: dict,
                 aws_region: str,
                 max_num_roles: int,
                 max_role_name_length: int
                 ):
        """Initialize the MainHandler with required services and configurations."""
        self.model_manager = model_manager
        self.storage = storage
        self.role_manager = role_manager
        self.admin_user_id = admin_user_id
        self.ssm_client = boto3.client('ssm', region_name=aws_region)
        self.api_keys = api_keys
        self.bot_id = bot_id
        self.formatting_info = formatting_info
        self.cool_user_ids = [self.admin_user_id, 476593109]
        self.max_num_roles = max_num_roles
        self.max_role_name_length = max_role_name_length


    def get_full_system_prompt(self) -> str:
        """
        Generate the complete system prompt by combining role, instructions, and notes.
        
        Returns:
            str: Formatted system prompt with role, instructions, and notes sections
        """

        chat_instruction = ''
        if self.chat_type == 'private':
            chat_instruction = 'This is a private chat (i.e you will communicate with a single individual).\n'
        elif self.chat_type == 'group':
            chat_instruction = 'You are a part of a group with several other members.\n'

        return f"{chat_instruction}<role>\n{self.current_role.prompt}\n</role>\n<instructions>\n{self.formatting_info}\n</instructions>\n<notes>\n{self.notes}\n</notes>"


    def get_memory_updater_prompt(self, is_reasoner: bool, context_messages: str) -> str:
        if is_reasoner:
            output_format =  '''- Output ONLY the final notes without tags.
                - Notes should be UNDER 250 words.
                - Notes should be in the same language as your role or the conversation.
                - Use bullet points for clarity and organization.
                - Include temporal and relationship tags where appropriate.'''
        else:
            output_format = '''- First, write your thinking inside <thinking> tags.  
                - Then, output the final notes inside <updated_notes> tags.
                - Notes should be UNDER 250 words.
                - Notes should be in the same language as your role or the conversation.
                - Use bullet points for clarity and organization.
                - Include temporal and relationship tags where appropriate.'''

        system_prompt = f'''
            <role>
            {self.current_role.prompt}
            </role>
            <task>
            SYSTEM PROMPT UPDATE: Refresh your long-term memory (notes).  
            Your existing notes are below and you will be provided the last {SHORT_TERM_MEMORY} messages. Update your notes by adding new critical information that aligns with your role. Prioritize facts that help you stay consistent, personalized, and effective in your role.
            </task>
            <current_notes>
            {self.notes['text']}
            </current_notes>
            <instructions>
            1. **Chain-of-Thought Process**:  
                - **Step 2**: Analyze recent messages. Identify key details with special attention to:
                    * Character traits, preferences, and patterns
                    * Time-sensitive information (events, deadlines, future plans)
                    * Emotional states or personal situations
                    * For group chats: Interpersonal dynamics and relationships between members
                - **Step 3**: Cross-check with existing notes. Flag duplicates, outdated info, or role-critical gaps.  
                - **Step 4**: Decide what to add/modify/retain based on:
                    * Relevance to your role
                    * Temporal importance (attach dates where applicable)
                    * Long-term value for relationship building
                    * Group dynamics (for group chats)
                
            2. **Temporal Information Handling**:
                - For time-sensitive information (e.g., "birthday next week", "job interview tomorrow"):
                    * Add a temporal tag [Until: YYYY-MM-DD] for expiration dates
                    * Update/remove expired time-sensitive entries
                - For persistent traits or preferences:
                    * Mark with [Core Trait] to indicate high retention priority
                - For temporary states (moods, situations):
                    * Use to update understanding of character but don't retain specific instances unless pattern-forming

            3. **Memory Retention Guidelines**:
                - Preserve information about core traits, preferences, and important facts
                - Be conservative with deletions - only remove notes that are:
                    * Conclusively outdated or superseded
                    * Contradicted by newer, more reliable information
                    * No longer relevant to your role or the conversation trajectory
                - If uncertain about relevance, retain the information
                - Condense similar or related information to maintain conciseness

            4. **Group Chat Relationship Tracking**:
                - Map relationships and dynamics between members:
                    * Identify close friendships, rivalries, or professional connections
                    * Note conversation patterns (who talks to whom, response tones)
                    * Record shared experiences or inside references between members
                    * Track shifting alliances or relationship changes over time
                - Use relationship tags to organize interpersonal information:
                    * [Relation: Person1-Person2] to mark relationship-specific notes
                    * [Group Dynamic] for patterns involving multiple members
                - Consider the social context when responding (e.g., avoiding topics sensitive to specific relationships)

            5. **Output Format**:  
                {output_format}
            </instructions>
            <example_response>
            <!-- Example: Life Coach Role (Private Chat) -->
            <thinking>
                1. My role is "Life Coach": I need to track goals, obstacles, personal values, and progress.
                2. In recent messages (dated 2025-02-25):
                - User mentioned starting a new job on March 15th
                - User expressed anxiety about public speaking
                - User shared they'll be visiting parents April 5-10
                - User mentioned they value work-life balance repeatedly
                3. Current notes review:
                - Already have "enjoys hiking on weekends" - still relevant
                - Have "preparing for job interview" - now outdated since they got the job
                - Have "struggles with morning routine" - no recent mention, but likely still relevant
                4. Actions needed:
                - Remove job interview note and replace with new job info
                - Add public speaking anxiety as character trait
                - Add time-sensitive parent visit
                - Strengthen note about work-life balance as a core value
                - Retain hiking preference and morning routine struggle
            </thinking>
            <updated_notes>
                - [Core Trait] Values work-life balance strongly
                - [Core Trait] Experiences anxiety about public speaking
                - [Until: 2025-03-15] Starting new job on March 15th
                - [Until: 2025-04-10] Visiting parents April 5-10
                - Enjoys hiking on weekends
                - Struggles with morning routine consistency
            </updated_notes>

            <!-- Example: Group Moderator Role (Group Chat) -->
            <thinking>
                1. My role is "Group Moderator": I need to track group dynamics, individual preferences, discussion topics, and potential sensitivities.
                2. In recent messages (dated 2025-03-01):
                - Alex and Jamie discussed their shared interest in rock climbing
                - Taylor expressed frustration when interrupted by Chris twice
                - Sam mentioned planning a group hike on March 15th
                - Alex has consistently shown expertise in finance topics
                - Jamie and Taylor appear to know each other outside the group
                3. Current notes review:
                - Already noted "Chris tends to dominate conversations" - reinforced by recent behavior
                - Have "group book discussion scheduled Feb 20th" - now outdated
                - Have "Sam is new to the group" - still relevant
                4. Actions needed:
                - Remove outdated book discussion note
                - Add Jamie-Alex rock climbing connection
                - Add Taylor-Chris tension
                - Add Sam's hiking plan
                - Note Jamie-Taylor external relationship
                - Strengthen note about Alex's finance expertise
                - Retain Chris's conversation style and Sam's newcomer status
            </thinking>
            <updated_notes>
                - [Group Dynamic] Group tends to focus on outdoor activities and finance topics
                - [Until: 2025-03-15] Sam organizing group hike on March 15th
                - [Core Trait] Alex has expertise in finance discussions
                - [Core Trait] Chris tends to interrupt others, particularly noticeable with Taylor
                - [Relation: Alex-Jamie] Share interest in rock climbing
                - [Relation: Jamie-Taylor] Appear to have relationship outside the group
                - Sam is relatively new to the group
            </updated_notes>
            </example_response>
            '''
        


        chat_instruction = ''
        if self.chat_type == 'private':
            chat_instruction = 'This is a private chat (i.e with one single individual).\n'
        elif self.chat_type == 'group':
            chat_instruction = 'This is a group chat with several members.\n'

        final_prompt = chat_instruction + system_prompt + "<context_messages>\n" + context_messages + "</context_messages>\n" 

        final_prompt += "Updated notes:\n" if is_reasoner else ""

        return final_prompt

    async def use_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE, context_messages: List[dict], is_coming_to_life = False) -> str | None:
        """
        Process a message using the AI model and handle the response.
        
        Args:
            update (Update): Telegram update object
            context (ContextTypes.DEFAULT_TYPE): Telegram context
            context_messages (List[dict]): Previous messages for context
            is_coming_to_life (bool): Whether this is a spontaneous bot response
        """
        
        # Get response from AI model
        response_text, input_tokens, output_tokens, thinking = await self.llm_service.get_response(
            system_prompt=self.get_full_system_prompt(),
            context_messages=context_messages   
        )

        chat_id = update.message.chat_id

        if is_coming_to_life:
            chat_id = self.bot_id

        # update the total query count
        self.model_manager.used_model(self.llm_service.model_name)
        
        # update the query count for the chat
        # await self.storage.used_model(model_name=self.llm_service.model_name, chat_id=chat_id)

        await self.deduct_tokens(chat_id, input_tokens, output_tokens, self.llm_service.output_tokens_multiplier)

        bot_message = await update.message.reply_text(response_text)
        
        # Store user's message
        bot_message = ChatMessage(
            message_id=bot_message.message_id,
            user=BOT_USER_DESCRIPTION,
            content=response_text,
            timestamp=bot_message.date.isoformat(),
            reply_to_id=update.message.message_id
        )

        chat_id = update.message.chat_id
        await self.storage.add_message(chat_id, bot_message)

        # Update memory if needed
        if await self.storage.increment_notes_counter(chat_id=update.message.chat_id, short_term_memory=SHORT_TERM_MEMORY):
            updated_notes, input_tokens, output_tokens, notes_thinking = await self.update_memory(context_messages, chat_id)
            await self.send_message_to_admin(f"Context messages: {GeminiService.format_messages(context_messages)}\n\nThinking: {thinking}\n\nUpdated notes: {updated_notes}", context.bot)

        if DEBUG:
            formatted_messages = self.llm_service.format_messages(context_messages)

            if type(formatted_messages) is str:
                log(formatted_messages)
            else:
                log("Formatted messages:", '\n'.join(str(msg) for msg in formatted_messages))

        return thinking


    async def set_come_to_life_chance(self, chance, chat_id):
        """Set the come to life chance"""
        await self.storage.set_come_to_life_chance(chance, chat_id=chat_id)
        self.come_to_life_chance = chance


    async def _chance_to_come_to_life(self, update: Update, context: ContextTypes.DEFAULT_TYPE, context_messages: List[dict]):
        """
        If the bot comes to life, it will answer the user's message.
        """
        log("Bot ID:", self.bot_id)
        
        try:
            balance = await self.model_manager.get_tokens(self.bot_id)
            if balance == 0:
                return

            if random() < self.come_to_life_chance:
                await self.use_bot(update, context, context_messages=context_messages, is_coming_to_life=True)


        except Exception as e:
            logger.error(f"Error in coming to life: {e}", exc_info=True)


    async def update_memory(self, context_messages: List[dict], chat_id):
        is_thinking = self.memory_updater.thinking_tokens is not None

        context_messages_str = GeminiService.format_messages(context_messages)
        prompt = self.get_memory_updater_prompt(is_thinking, context_messages_str)

        if DEBUG:
            log(f"Memory updater prompt: {prompt[-2000:]}")

        response_text, input_tokens, output_tokens, thinking = await self.memory_updater.get_response(
            system_prompt=prompt,
            query_without_context = "Updated notes:"
        )

        # for reasoners, the thinking is thinking, and the response_text is the updated_notes
        # for non-reasoners, the response_text has the following format: <thinking></thinking><updated_notes></updated_notes>
        if is_thinking:
            updated_notes = response_text
        else: 
            assert thinking == None
            thinking = response_text.split('</thinking>')[0].split('<thinking>')[1]
            updated_notes = response_text.split('</updated_notes>')[0].split('<updated_notes>')[1]
        

        await self.storage.set_notes(updated_notes, chat_id=chat_id)
        return updated_notes, input_tokens, output_tokens, thinking


    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:

            if update.message is None:
                return

            chat_id = update.effective_chat.id

            # Handle admin responses
            if (chat_id == self.admin_user_id or update.message.from_user.id in self.cool_user_ids) and update.message.reply_to_message:
                if await self.handle_admin_response(update, context):
                    return


            # Handle user answers to requests


            if await self.handle_user_request(update, context):
                return 

            self.chat_type = update.effective_chat.type

            # Replace multiple individual getter calls with a single get_chat_info call
            chat_info = await self.storage.get_chat_info(chat_id=chat_id)
            self.current_role_id = chat_info['current_role_id']
            self.available_roles_ids = chat_info['available_roles_ids']
            self.notes = chat_info['notes']
            self.come_to_life_chance = chat_info['come_to_life_chance']
            self.current_role = await self.storage.get_current_role(chat_id=chat_id)
            model_name = chat_info['model']

            memory_updater_model_name = chat_info['memory_updater_model']

            if model_name.startswith('claude'):
                self.llm_service = ClaudeService(api_key = self.api_keys['claude'], model_name=model_name)
                self.memory_updater = ClaudeService(api_key = self.api_keys['claude'], model_name=memory_updater_model_name)
            elif model_name.startswith('gemini'):
                self.llm_service = GeminiService(api_key = self.api_keys['gemini'], model_name=model_name)
                self.memory_updater = GeminiService(api_key = self.api_keys['gemini'], model_name=memory_updater_model_name)
            else:
                raise ValueError(f"Unknown model: {model_name} or Unknown memory updater model: {memory_updater_model_name}")


            context_messages = await self.storage.get_message_context(update, context, SHORT_TERM_MEMORY)
            nice_context_messages = "\n\n".join(str(msg) for msg in context_messages)
            # await self.send_message_to_admin(f"Context messages: {nice_context_messages}", context.bot)

            if await self.storage.increment_notes_counter(chat_id=update.message.chat_id, short_term_memory=SHORT_TERM_MEMORY):
                updated_notes, input_tokens, output_tokens, thinking  = await self.update_memory(context_messages, chat_id)
                log("Thinking:", thinking)
                await self.send_message_to_admin(f"Context messages: {nice_context_messages}\n\nThinking: {thinking}\n\nUpdated notes: {updated_notes}", context.bot)

            # Check if the message is relevant and have a chance to come to life
            if not await self._check_message_relevance(update.message, context.bot):
                if random() < self.come_to_life_chance:
                    await self._chance_to_come_to_life(update, context, context_messages)
                
                return
            

            user = update.message.from_user

            balance = await self.model_manager.get_tokens(update.effective_chat.id)
            if balance is None:
                await update.message.reply_text(f"Сори, бро. Базированный бот еще не был активирован. Начни с /start.")
                return

            await self.process_user_message(update, context, context_messages)

        except Exception as e:
            logger.error(f"Error in handle_message: {e}", exc_info=True)
            await self.send_message_to_admin(str(update), context.bot)
            try:
                await self.send_error_message(update, context, str(e))
            except Exception as e:
                logger.error(f"Metaerror: Error in sending error message: {e}", exc_info=True)
    

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
        

    def set_model(self, chat_id: int, model_name: str):
        """
        Set the AI model for a specific chat.
        
        Args:
            chat_id (int): Telegram chat ID
            model_name (str): Name of the AI model to use
            
        Raises:
            ValueError: If the model name is not recognized
        """
        # Get the best allowed model based on current limits
        model_name = self.model_manager.best_allowed_model(model_name)

        # Store the model selection for this chat
        self.storage.set_model(chat_id=chat_id, model_name=model_name)

        # Initialize the appropriate service based on model type
        if model_name.startswith('claude'):
            self.llm_service = ClaudeService(api_key=self.api_keys['claude'], model_name=model_name)
        elif model_name.startswith('gemini'):
            self.llm_service = GeminiService(api_key=self.api_keys['gemini'], model_name=model_name)
        else:
            raise ValueError(f"Unknown model: {model_name}")


    async def send_message_to_admin(self, message: str, bot) -> None:
        """Send a message to the admin"""
        await bot.send_message(chat_id=self.admin_user_id, text=message[-4096:])


    async def forward_to_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Forward the message to the admin"""
        await update.message.forward(chat_id=self.admin_user_id)


    async def send_error_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, error_message: str) -> None:
        if update.message:
            await update.message.reply_text("Произошла какая-то хрень. Мишка уведомлен. Считайте проблему уже почти решенной.")

        await self.send_message_to_admin(f"Something has gone wrong, hehehe. Error: {error_message}", context.bot)
        await self.forward_to_admin(update, context)


    async def deduct_tokens(self, user_id: int, input_tokens: int, output_tokens: int, output_tokens_multiplier: int) -> None:
        await self.model_manager.use_tokens(str(user_id), input_tokens + output_tokens_multiplier * output_tokens)

    #TODO: implement this
    async def role_approved_by_bot(self, role_name: str) -> bool:
        return True


    async def process_user_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, context_messages: List[dict]) -> None:
        """
        Process an incoming user message and generate a response.
        
        Args:
            update (Update): Telegram update object
            context (ContextTypes.DEFAULT_TYPE): Telegram context
            context_messages (List[dict]): Previous messages for context
            
        Handles:
        - Token balance checking
        - Model usage limits
        - Response generation
        - Error handling
        """
        
        

        # check if the chat has enough tokens
        balance = await self.model_manager.get_tokens(update.effective_chat.id)


        data = self.model_manager.get_model_data()
        model = self.llm_service.model_name
        model_lineika = 'gemini' if model.startswith('gemini') else 'claude'

        if model not in allowed_models_limits.get(model_lineika, {}):
            await self.send_error_message(update, context, f"The model is not in allowed_models_limits. Model: {model}")
            return

    
        if allowed_models_limits[model_lineika][model] is not None and data[model] >= allowed_models_limits[model_lineika][model]:
            self.set_model(self.model_manager.best_allowed_model(model))
            await update.message.reply_text(f"Сори, бро, моделька устала. Моделька сменилась на лучшую доступную модельку этой линейки.")


        if balance == 0:
            await update.message.reply_text(f"Закончились токены... Мда, не повезло. Но ходят слухи, что если использовать команду /ask_for_tokens, Мишка добавит еще.")
            return
        
        try:
            await self.use_bot(update, context, context_messages)
            

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await self.send_error_message(update, context, str(e))


    async def handle_admin_response(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle responses from admin to user questions or token requests"""
        message_text = update.message.reply_to_message.text
        admin_answer = update.message.text

        # Handle token requests
        if message_text.startswith("TokenRequest:"):
            _, original_chat_id, original_message_id = message_text.split("\n")[0].split(":")
            await self.handle_admin_answer_to_token_request(update, context, original_chat_id, original_message_id, admin_answer)
            return True
        
        # Handle help requests
        if message_text.startswith("HelpRequest:"):
            _, original_chat_id, original_message_id, user_id = message_text.split("\n")[0].split(":")
            
            await self.handle_admin_answer_to_help_request(update, context, original_chat_id, original_message_id, user_id, admin_answer)
        
            return True

        if message_text.startswith("NewRoleApprovalRequest:"):
            _, original_chat_id, original_message_id, user_id = message_text.split("\n")[0].split(":")
            role_name = message_text.split("\n")[1].split(":")[1].strip()
            role_prompt = message_text.split("\n")[2].split(":")[1].strip()
            await self.handle_admin_answer_to_new_role_approval_request(update, context, original_chat_id, original_message_id, user_id, role_name, role_prompt, admin_answer)
            return True
        # Handle other admin responses here...
        log("Other admin response")
        return False
    

    async def handle_user_request(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """Handle user requests"""
        user = update.message.from_user
        text = update.message.text
        username = user.username or user.first_name

        reply_to_message = update.message.reply_to_message
        if reply_to_message and reply_to_message.from_user.id == self.bot_id and reply_to_message.text.startswith(HELP_COMMAND_TEXT):
            await self.send_message_to_admin(
                f"HelpRequest:{update.effective_chat.id}:{update.message.message_id}:{user.id}\n"
                f"User {username} asked for HEEEEELP.", context.bot)
            await self.forward_to_admin(update, context)
            return True 


        if reply_to_message and reply_to_message.from_user.id == self.bot_id and reply_to_message.text.startswith("NewRoleAddition:"):
            #edit message after it has been replied to, and remove the NewRoleAddition prefix
            await reply_to_message.edit_text(reply_to_message.text.split('\n', maxsplit=1)[1])
            
            _, role_name = reply_to_message.text.split("\n")[0].split(":")
            # the role is checked at first, before being added
            if await self.role_approved_by_bot(text):
                await self.add_role(update, context, role_name, role_prompt=text, chat_id=update.effective_chat.id)
                await update.message.reply_text(f"Добавил {role_name}.")
            else:
                await update.message.reply_text(f"Шото подозрительное. Спрошу у Мишки можно ли мне такую личность.")
                await self.send_message_to_admin(f"NewRoleApprovalRequest:{update.effective_chat.id}:{update.message.message_id}:{user.id}\nName: {role_name}\nThe prompt:\n{text}", context.bot)
                
            return True
        
        return False


    async def handle_admin_answer_to_token_request(self, update, context, original_chat_id: int, original_message_id: int, admin_answer: str) -> None:
        """Process admin's response to token request"""
        if not admin_answer.isdigit():
            await update.message.reply_text("A number is expected!")
            return

        tokens = int(admin_answer)
        await context.bot.send_message(
            chat_id=original_chat_id,
            text=f"Мишка был настолько щедрым, что добавил тебе {tokens} токенов!",
            reply_to_message_id=original_message_id
        )
        await self.model_manager.add_tokens(original_chat_id, tokens)
        await update.message.reply_text("Tokens have been added to the user!")


    async def handle_admin_answer_to_help_request(self, update, context, original_chat_id: int, original_message_id: int, user_id: int, admin_answer: str) -> None:
        """Process admin's response to help request"""
        await context.bot.send_message(
            chat_id=original_chat_id,
            text=f"Мишка соблаговолил потратить свое бесценное время на ответ:\n{admin_answer}",
            reply_to_message_id=original_message_id
        )


    async def handle_admin_answer_to_new_role_approval_request(self, update, context, original_chat_id: int, original_message_id: int, user_id: int, role_name: str, role_prompt: str, admin_answer: str) -> None:
        """Process admin's response to new role approval request"""
        if admin_answer.lower() in ("yes", "да", "+", "1", "ok", "ок"):
            await self.add_role(update, context, role_name, role_prompt=role_prompt, chat_id=original_chat_id)
            await context.bot.send_message(
                chat_id=original_chat_id,
                text=f"О, Мишка заапрувил {role_name}.",
                reply_to_message_id=original_message_id
            )

        else:
            await context.bot.send_message(
                chat_id=original_chat_id,
                text=f"Штош, Мишка не заапрувил это.",
                reply_to_message_id=original_message_id
            )


    async def add_role(self, update, context, role_name: str, role_prompt: str, chat_id: int) -> None:
        """Process admin's response to role addition"""
        role = Role(role_name, role_prompt)
        new_role_id = self.role_manager.add_new_role(role)
        await self.storage.add_available_role_id(new_role_id, chat_id=chat_id)
        await self.send_message_to_admin(f"Role {role_name} added!", context.bot)


    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        await update.message.reply_text(text = f"Привет, я ботик-обормотик. Спроси меня что-нибудь о Мишке!")
        
        await self.model_manager.add_chat_if_not_exists(update.effective_chat.id)


    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        is_admin = update.message.from_user.id == self.admin_user_id

        help_text = [
            f"{HELP_COMMAND_TEXT}",
            "Список команд:\n",
        ]
        
        for command in user_commands:
            help_text.append(f"{command} - {user_commands[command]}\n")
        
        if is_admin:
            help_text.append("Команды для крутых:\n")
            for command in admin_commands:
                help_text.append(f"{command} - {admin_commands[command]}\n")
        
        help_text.extend([
            "Если вы хотите задать вопрос Мишке, сформулируйте его, подумайте, подумайте, подумайте...",
            "Если не помогло, то можете ответить на это сообщение вопросом и, возможно, Мишка ответит."
        ])
        
        await update.message.reply_text("\n".join(help_text))
        
    @staticmethod
    def command_in_identified_chats(func):
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            chat_id = update.effective_chat.id
            balance = await self.model_manager.get_tokens(chat_id)
            if balance is None:
                await update.message.reply_text(f"Сори, бро. Базабот еще не был активирован. Начни с /start.")
                return
            return await func(self, update, context)
        return wrapper

    @staticmethod
    def command_for_admin(func):
        async def wrapper(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
            if update.message.from_user.id == self.admin_user_id or update.message.from_user.id in self.cool_user_ids:
                return await func(self, update, context)
            else:
                await update.message.reply_text("Сори, братишка, но ты здесь не босс.")
        return wrapper


    @command_in_identified_chats
    async def get_tokens_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /get_tokens is issued."""
        chat = update.effective_chat
        remaining_tokens = await self.model_manager.get_tokens(chat.id)

        if chat.type == "group":
            message = f"В этом чате ботика хватит на {remaining_tokens} токенов."
        elif chat.type == "private":
            message = f"У вас {remaining_tokens} токенов."
        else:
            await self.send_message_to_admin(self, f"Unknown chat type: {chat.type}.", context.bot)
            raise

        await update.message.reply_text(message)


    @command_in_identified_chats
    async def ask_for_tokens_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /ask_for_tokens is issued."""
        chat_id = update.effective_chat.id

        await context.bot.send_message(
            chat_id=self.admin_user_id,
            text=f"TokenRequest:{chat_id}:{update.message.message_id}\n"
                 f"The current amount in the chat is {await self.model_manager.get_tokens(chat_id)}"
        )
        await update.message.reply_text(f"Мишка уведомлен о вашей просьбе. Ждите милости от него.")


    @command_for_admin
    async def set_model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Set the current model"""
        model_name = update.message.text.split(" ", 1)[1]
        model = self.model_manager.best_allowed_model(model_name)
        chat_id = update.message.chat_id
        await self.storage.set_model(model, chat_id=chat_id)
        
        # Get updated chat info after setting the model
        model = await self.storage.get_model(chat_id=chat_id)
        await update.message.reply_text(f"Current model set to {model}")


    @command_for_admin
    async def get_model_data_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Get the model data"""
        model_data = self.model_manager.get_model_data()
        current_model = await self.storage.get_model(update.message.chat_id)
        text = f"Current model (in this chat): {current_model}\nData across all chats:\n"
        for model in model_data:
            text += f"{model}: {model_data[model]} queries\n"

        await update.message.reply_text(text)


    @command_for_admin
    async def get_role_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Get the system prompt"""
        role = await self.storage.get_current_role(chat_id=update.message.chat_id)
        await update.message.reply_text(f"Текущая личность ботика:\n{role.name}\n{role.prompt}")


    @command_for_admin
    async def get_notes_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Get the current notes"""
        notes = (await self.storage.get_notes(chat_id=update.message.chat_id))['text']
        await update.message.reply_text(notes)


    # NOTE: there's a bug that if one responds multiple times to the message, multiple roles will be added
    @command_for_admin
    async def add_role_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Add a role to choose"""
        try: 
            
            num_current_roles = len(await self.storage.get_available_roles_ids(chat_id=update.message.chat_id))
            if num_current_roles >= self.max_num_roles:
                await update.message.reply_text(f"Сори, у ботика уже слишком много личностей. Максимум {self.max_num_roles}. Если хочешь новую, то удали роль с помощью /delete_role.")
                return

            name = update.message.text.split(" ", 1)[1]

            if len(name) > self.max_role_name_length:
                await update.message.reply_text(f"Воу, воу, воу. Что это за длинющее имя такое. Пж не больше {self.max_role_name_length} символов.")
                return

            global_roles = (await self.role_manager.get_global_roles()).values()
            chat_specific_roles = (await self.storage.get_available_roles(chat_id=update.message.chat_id)).values()
            global_roles_names = {role.name for role in global_roles}
            chat_specific_roles_names = {role.name for role in chat_specific_roles}

            if name in global_roles_names or name in chat_specific_roles_names:
                await update.message.reply_text(f"Сори, но такая личность уже есть. Пожалуйста, выберите другое имя.")
                return

            await context.bot.send_message(chat_id=update.message.chat_id, text=f"NewRoleAddition:{name}\nЧто же за личность {name} (ответь на это сообщение описанием)?\n\nP.s уже было добавлено {num_current_roles}/{self.max_num_roles} личностей.")
        except IndexError :
            await update.message.reply_text(f"Напиши пж \"/add_role имя_роли\"")
        except Exception as e:
            await self.send_error_message(update, context, f"Error: {e}")


    @command_for_admin
    async def give_bot_tokens_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Give the bot tokens"""
        tokens = int(update.message.text.split(" ", 1)[1])
        await self.model_manager.add_tokens(self.bot_id, tokens)
        await update.message.reply_text(f"Tokens have been added to the bot!")


    @command_for_admin
    async def clear_chat_history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Clear the chat history"""
        await self.storage.clear_chat_history(update.message.chat_id)
        await update.message.reply_text("Ваша история почищена (но не ваша совесть))")


    @command_in_identified_chats
    async def choose_role_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Let user choose a role for the bot"""
        chat_specific_roles = await self.storage.get_available_roles(chat_id=update.effective_chat.id)
        global_roles = await self.role_manager.get_global_roles()
        roles = chat_specific_roles | global_roles

        keyboard = []
        for role_id, role in roles.items():
            keyboard.append([InlineKeyboardButton(role.name, callback_data=f"choose_role_{role_id}")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Choose a role:", reply_markup=reply_markup)


    @command_in_identified_chats
    async def remove_role_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Delete a role"""
        chat_id = update.effective_chat.id
        chat_specific_role_ids = set(await self.storage.get_available_roles_ids(chat_id=chat_id)) - {await self.storage.get_current_role_id(chat_id=chat_id)}  # without current role
        roles_dict = await self.role_manager.get_roles_by_ids(chat_specific_role_ids)

        keyboard = []
        for role_id, role in roles_dict.items():
            keyboard.append([InlineKeyboardButton(role.name, callback_data=f"remove_role_{role_id}")])

        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Choose a role to remove:", reply_markup=reply_markup)


    @command_in_identified_chats
    async def choose_role_button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle role selection button press"""
        query = update.callback_query
        await query.answer()

        chat_id = query.message.chat_id
        role_id = query.data.replace("choose_role_", "")

        # add a separate function for this
        await self.storage.set_current_role_id(role_id, chat_id=chat_id)
        self.current_role_id = role_id

        await query.edit_message_text(f"Role set to: {(await self.storage.get_current_role(chat_id=chat_id)).name}")

    @command_in_identified_chats
    async def remove_role_button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle role selection button press"""
        query = update.callback_query
        await query.answer()

        chat_id = query.message.chat_id
        role_id = query.data.replace("remove_role_", "")

        # add a separate function for this
        await self.storage.remove_available_role_id(role_id, chat_id=chat_id)

        await query.edit_message_text(f"Role removed.")


    @command_for_admin
    async def set_role_global_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Set a role as global or non-global. If empty, assume True"""
        try:
            args = update.message.text.split(" ", 2)
            if len(args) < 2 or len(args) > 3:
                await update.message.reply_text("Usage: /set_role_global <role_name> <true|false>")
                return
                
            role_name = args[1]
            is_global = args[2].lower() in ('true', 'yes', '1') if len(args) > 2 else True
            
            # Find the role by name
            roles = self.role_manager.get_role_by_name(role_name)
            if not roles:
                await update.message.reply_text(f"Role '{role_name}' not found")
                return
                
            if len(roles) > 1:
                await update.message.reply_text(f"More than one role found: {', '.join([role.name for role in roles])}")
                return
            
            # Update the role's global status
            if is_global:
                self.role_manager.make_role_global(role.id)
                await update.message.reply_text(f"Role '{role_name}' is now global")
            else:
                self.role_manager.make_role_non_global(role.id)
                await update.message.reply_text(f"Role '{role_name}' is now non-global")
                
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")

    @command_in_identified_chats
    async def get_come_to_life_chance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Get the come to life chance"""
        chance = await self.storage.get_come_to_life_chance(chat_id=update.message.chat_id)
        await update.message.reply_text(f"Я жив на целых {round(chance*100)} процентов.")


    @command_in_identified_chats
    async def set_come_to_life_chance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set the come to life chance"""
        try:
            input_text = update.message.text.split(" ", 1)[1]
            
            # Check if input is too long before conversion
            if len(input_text) > 3:  # More than 3 digits is definitely not 0-100
                await update.message.reply_text("Я могу быть жив только от 0 до 100 процентов.")
                return
                
            percent = int(input_text)
            if percent < 0 or percent > 100:
                await update.message.reply_text("Я могу быть жив только от 0 до 100 процентов.")
                return
                
            # Convert percentage to decimal (0-1 range)
            chance = percent / 100.0
            
            await self.set_come_to_life_chance(chance, update.effective_chat.id)
            await update.message.reply_text(f"Теперь ботик жив на все {percent} процентов")
        except (ValueError, IndexError):
            await update.message.reply_text("Пожалуйста, укажите число от 0 до 100.")


    