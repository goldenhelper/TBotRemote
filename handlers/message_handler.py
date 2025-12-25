from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.error import BadRequest
import logging
from typing import List, Callable

from urllib3 import response
from models import ChatMessage
from services.model_manager import ModelManager
from services.claude_service import ClaudeService
from services.gemini_service import GeminiService
from services.openai_service import OpenAIService
from services.image_service import ImageService
from services.chat_history import Storage
from services.role_manager import RoleManager, Role
from services.base_service import BaseAIService
from handlers.media_handler import MediaHandler
import boto3
from random import random
from utils.constants import *
import re
from telegram import Bot


logger = logging.getLogger(__name__)

HELP_COMMAND_TEXT = "Хелпа"

user_commands = {
    "/start": "Стартуем.", 
    "/help": "Хелпа.", 
    "/tokens_amount": "Показывает как долго ботик еще сможет терпеть этот чат.", 
    "/ask_for_tokens": "Поныть на токены и, может, Мишка зарядит ботика.",
    "/set_aliveness" : "Устанавливает активность ботика.",
    "/how_alive" : "Показывает насколько ботик живой.",
    "/clear_history" : "Очистить историю сообщений (токены и настройки сохраняются)",
    "/delete_chat" : "Полностью удалить чат (включая токены и настройки)",
    "/get_role" : "Показывает текущую роль.",
    "/add_role" : "Добавляет новую личность боту! Просто напиши /add_role и бот всё спросит сам.",
    "/choose_role": "Выбрать роль для бота."
}

admin_commands = {
    "/get_model_data" : "Показывает текущую модельку.",
    "/give_bot_tokens" : "Дает боту токены.",
    "/add_admin" : "Добавить админа: /add_admin <user_id>",
    "/remove_admin" : "Удалить админа: /remove_admin <user_id>",
    "/list_admins" : "Показать список админов.",
    "/get_settings" : "Показать все настройки бота.",
    "/set_setting" : "Изменить настройку: /set_setting <key> <value>",
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
                 ):
        """Initialize the MainHandler with required services and configurations."""
        self.model_manager = model_manager
        self.storage = storage
        self.role_manager = role_manager
        self.admin_user_id = admin_user_id
        self.ssm_client = boto3.client('ssm', region_name=aws_region) if aws_region else None
        self.api_keys = api_keys
        self.bot_id = bot_id
        self.formatting_info = formatting_info
        # Ensure primary admin from config is in Supabase admin list
        if not self.storage.is_admin(self.admin_user_id):
            self.storage.add_admin_user(self.admin_user_id)
        # Video analyzer and media handler are created lazily
        self._video_analyzer = None
        self._video_analyzer_model = None
        self._media_handler = None

    @property
    def max_num_roles(self) -> int:
        return self.storage.max_num_roles

    @property
    def max_role_name_length(self) -> int:
        return self.storage.max_role_name_length

    @property
    def video_analyzer_model(self) -> str:
        return self.storage.video_analyzer_model

    @property
    def video_analyzer(self) -> GeminiService:
        """Lazily create video analyzer, recreating if model changed."""
        current_model = self.video_analyzer_model
        if self._video_analyzer is None or self._video_analyzer_model != current_model:
            self._video_analyzer = GeminiService(
                api_key=self.api_keys.get('gemini'),
                model_name=current_model,
                thinking_model=False,
            )
            self._video_analyzer_model = current_model
        return self._video_analyzer

    @property
    def media_handler(self) -> MediaHandler:
        """Lazily create media handler."""
        if self._media_handler is None:
            self._media_handler = MediaHandler(self.storage, video_analyzer=self.video_analyzer)
        return self._media_handler

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
            chat_instruction = 'You are a part of a group with several other members. The user you should respond to is the last one who sent a message.\n'

        

        
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


    async def use_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE, context_messages: List[ChatMessage], is_coming_to_life = False):
        """
        Process a message using the AI model and handle the response.
        
        Args:
            update (Update): Telegram update object
            context (ContextTypes.DEFAULT_TYPE): Telegram context
            context_messages (List[ChatMessage]): Previous messages for context
            is_coming_to_life (bool): Whether this is a spontaneous bot response
        """
        
        #NOTE
        # Get response from AI model
        response_text, input_tokens, output_tokens, thinking, video_analysis_results = await self.llm_service.get_response(
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

        # Sanitize model output: strip media headers like "[GIF/Animation: ...]" and any accidental
        # metadata lines such as "message_id: ..., reply_to_id: ..., assistant[...]" that the model
        # might have echoed.
        cleaned_text = re.sub(r'^\s*message_id:[^\n]*\n?', '', response_text, flags=re.IGNORECASE)

        # Remove leading bracketed media description (Image/GIF/Video/Sticker/Document) if present
        cleaned_text = re.sub(r'^\s*\[(?:Image|GIF/Animation|Video|Sticker|Document):[^\]]*\]\s*', '', cleaned_text, flags=re.IGNORECASE)

        bot_message = await update.message.reply_text(cleaned_text)
        
        # Store user's message
        bot_message = ChatMessage(
            message_id=bot_message.message_id,
            user=BOT_USER_DESCRIPTION,
            content=response_text,
            timestamp=bot_message.date.strftime("%a, %d. %b %Y %H:%M"),
            reply_to_id=update.message.message_id,
            reasoning=thinking,
        )

        chat_id = update.message.chat_id
        await self.storage.add_message(chat_id, bot_message)
        
        # Update any messages that got video analysis during fallback
        if video_analysis_results:
            for message_id, analysis_text in video_analysis_results.items():
                logger.info(f"Updating stored message {message_id} with video analysis: '{analysis_text[:100]}...'")
                await self.storage.update_message_description(chat_id, message_id, analysis_text)

        # Update memory if needed
        if await self.storage.increment_notes_counter(chat_id=update.message.chat_id, short_term_memory=SHORT_TERM_MEMORY):
            updated_notes, input_tokens, output_tokens, notes_thinking = await self.update_memory(context_messages, chat_id)
            await self.send_message_to_admin(f"Context messages: {BaseAIService.format_messages(context_messages)}\n\nThinking: {thinking}\n\nUpdated notes: {updated_notes}", context.bot)

        if DEBUG:
            formatted_messages = BaseAIService.format_messages(context_messages)
            log(formatted_messages)

        return thinking


    async def set_come_to_life_chance(self, chance, chat_id):
        """Set the come to life chance"""
        await self.storage.set_come_to_life_chance(chance, chat_id=chat_id)
        self.come_to_life_chance = chance


    async def _chance_to_come_to_life(self, update: Update, context: ContextTypes.DEFAULT_TYPE, context_messages: List[ChatMessage]):
        """
        If the bot comes to life, it will answer the user's message.
        """
        log("Bot ID:", self.bot_id)

        try:
            # Ensure bot's chat exists before attempting to use tokens
            await self.model_manager.add_chat_if_not_exists(self.bot_id)

            balance = await self.model_manager.get_tokens(self.bot_id)
            if balance is None or balance == 0:
                return

            if random() < self.come_to_life_chance:
                await self.use_bot(update, context, context_messages=context_messages, is_coming_to_life=True)


        except Exception as e:
            logger.error(f"Error in coming to life: {e}", exc_info=True)


    async def update_memory(self, context_messages: List[ChatMessage], chat_id):
        is_thinking = self.memory_updater.thinking_tokens is not None

        # Format ChatMessage objects to string format
        context_messages_str = BaseAIService.format_messages(context_messages)
        prompt = self.get_memory_updater_prompt(is_thinking, context_messages_str)

        if DEBUG:
            log(f"Memory updater prompt: {prompt[-2000:]}")

        response_text, input_tokens, output_tokens, thinking, _ = await self.memory_updater.get_response(
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

            # Explicitly add the current message to history before gathering context
            current_node = ChatMessage.from_telegram_message(update.message)
            await self.storage.add_message(chat_id, current_node)

            # Handle admin responses
            if self.storage.is_admin(update.message.from_user.id) and update.message.reply_to_message:
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

            provider_key = _provider_key(model_name)
            service_cls   = PROVIDERS[provider_key]

            video_analyzer = GeminiService(
                api_key=self.api_keys['gemini'],
                model_name=self.video_analyzer_model,
                thinking_model=False
            ) if provider_key != 'gemini' else None
            init_kwargs = {
                'api_key': self.api_keys[provider_key],
                'model_name': model_name,
            }
            if provider_key == 'gemini':
                init_kwargs['video_analyzer'] = video_analyzer
            else:
                init_kwargs['video_analyzer'] = video_analyzer
            self.llm_service = service_cls(**init_kwargs)

            # Initialise memory updater service. It can be a different provider than the main LLM,
            # so determine its provider independently.

            mu_provider_key = _provider_key(memory_updater_model_name)
            mu_service_cls  = PROVIDERS[mu_provider_key]

            mu_kwargs = {
                'api_key': self.api_keys[mu_provider_key],
                'model_name': memory_updater_model_name,
            }

            # Only pass the video analyzer if the memory updater itself is not Gemini.
            # GeminiService already supports multimodal natively and passing another Gemini
            # instance can create unnecessary recursion.
            if mu_provider_key != 'gemini':
                mu_kwargs['video_analyzer'] = video_analyzer

            self.memory_updater = mu_service_cls(**mu_kwargs)

            # Set context for media handler (role, chat type, and long-term notes)
            self.media_handler.set_context(
                self.current_role,
                self.chat_type,
                self.notes.get('text', '') if isinstance(self.notes, dict) else self.notes,
                hasattr(self.llm_service, 'supports_video') and getattr(self.llm_service, 'supports_video'),
            )
            
            # Retrieve recent context messages before handling new input
            context_messages = await self.storage.get_message_context(update, context, SHORT_TERM_MEMORY, max_reply_depth=SHORT_TERM_MEMORY//2)
            
            # Process any media in the message first
            media_message = await self.media_handler.handle_media_message(update, context)
                
            # If the incoming message was media-only (e.g., GIF) ensure it is included in the prompt so
            # Gemini receives at least one content item. We add it to the context seen by the model.
            if media_message is not None:
                logger.info(f"Created media_message: id={media_message.message_id}, type={media_message.media_type}, file_id={media_message.file_id}, has_media_data={media_message.media_data is not None}")
                # Check if this message is already in context_messages to avoid duplicates
                existing_ids = {msg.message_id for msg in context_messages}
                if media_message.message_id not in existing_ids:
                    context_messages.append(media_message)
                    logger.info(f"Added media_message to context")
                else:
                    # Find and update the existing message with the media description
                    for i, existing_msg in enumerate(context_messages):
                        if existing_msg.message_id == media_message.message_id:
                            # Update the existing message with media description and other properties
                            existing_msg.media_description = media_message.media_description
                            existing_msg.media_type = media_message.media_type
                            existing_msg.file_id = media_message.file_id
                            logger.info(f"Updated existing message {media_message.message_id} with media_description: '{media_message.media_description}'")
                            break

            # Continue with normal text message processing (context_messages already up to date)

            # Fetch media data for context before passing to the model
            logger.info(f"About to fetch media for {len(context_messages)} context messages")
            await self._fetch_media_for_context(context_messages, context.bot)

            def format_context_message(msg):
                """Format a context message with media description if available."""
                base_msg = f"{msg.user} [{msg.timestamp}]: {msg.content}"
                if msg.media_description:
                    return f"{base_msg}\n    [Media: {msg.media_description}]"
                return base_msg

            nice_context_messages = "\n\n".join(format_context_message(msg) for msg in context_messages)
            await self.send_message_to_admin(f"Context messages: {nice_context_messages}", context.bot)

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


    async def _fetch_media_for_context(self, messages: List[ChatMessage], bot: Bot):
        """Fetches media data for messages that contain images, stickers, or GIF animations."""
        for msg in messages:
            if msg.media_type in ['image', 'sticker', 'animation'] and msg.file_id:
                if msg.media_type == 'animation' and msg.media_description and msg.media_data is None and not self.media_handler.main_supports_video:
                    logger.info(f"Skipping media download for animation message {msg.message_id} (already described)")
                    continue
                logger.info(f"Fetching media for message {msg.message_id}, type: {msg.media_type}, file_id: {msg.file_id}")
                try:
                    # Stickers are often webp, animations are .gif or .mp4; image_service can handle raw bytes fetch.
                    media_data = await self.media_handler.image_service.download_image(bot, msg.file_id)
                    if media_data:
                        # Check file size - Gemini has limits on media size
                        if len(media_data) > 20 * 1024 * 1024:  # 20MB limit
                            logger.warning(f"Media file too large ({len(media_data)} bytes), skipping for message {msg.message_id}")
                            continue
                        msg.media_data = media_data
                        logger.info(f"Successfully downloaded {len(media_data)} bytes for message {msg.message_id}")
                    else:
                        logger.warning(f"No media data downloaded for message {msg.message_id}")
                except Exception as e:
                    logger.error(f"Failed to fetch media for context (file_id: {msg.file_id}): {e}")


    async def _check_message_relevance(self, message, bot) -> bool:
        """Check if the message should be processed by the bot"""
        bot_username = bot.username
        is_reply_to_bot = (
            message.reply_to_message and 
            message.reply_to_message.from_user and 
            message.reply_to_message.from_user.id == bot.id
        )
        
        is_private_chat = message.chat.type == "private"
        is_bot_mentioned = message.text is not None and f"@{bot_username}" in message.text
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
            # Claude delegates video work to the shared analyser
            self.llm_service = ClaudeService(
                api_key=self.api_keys['claude'],
                model_name=model_name,
                video_analyzer=self.video_analyzer,
            )
        elif model_name.startswith('gemini'):
            self.llm_service = GeminiService(
                api_key=self.api_keys['gemini'], 
                model_name=model_name, 
                video_analyzer=self.video_analyzer
            )
        elif model_name.startswith(('openai', 'gpt', 'o1', 'o3')):
            # OpenAI delegates video work to the shared analyser
            self.llm_service = OpenAIService(
                api_key=self.api_keys['openai'],
                model_name=model_name,
                video_analyzer=self.video_analyzer,
            )
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
        await self.model_manager.use_tokens(user_id, input_tokens + output_tokens_multiplier * output_tokens)

    #TODO: implement this
    async def role_approved_by_bot(self, role_name: str) -> bool:
        return True


    async def process_user_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE, context_messages: List[ChatMessage]) -> None:
        """
        Process an incoming user message and generate a response.
        
        Args:
            update (Update): Telegram update object
            context (ContextTypes.DEFAULT_TYPE): Telegram context
            context_messages (List[ChatMessage]): Previous messages for context
            
        Handles:
        - Token balance checking
        - Model usage limits
        - Response generation
        - Error handling
        """
        
        

        # check if the chat has enough tokens
        balance = await self.model_manager.get_tokens(update.effective_chat.id)
        usage = self.model_manager.get_all_model_usage()
        limits = self.model_manager.get_allowed_models_limits()
        model = self.llm_service.model_name
        if model.startswith('gemini'):
            model_lineika = 'gemini'
        elif model.startswith('claude'):
            model_lineika = 'claude'
        elif model.startswith(('openai', 'gpt', 'o1', 'o3')):
            model_lineika = 'openai'
        else:
            model_lineika = 'unknown'  # Should not happen, caught earlier

        if model not in limits.get(model_lineika, {}):
            await self.send_error_message(update, context, f"The model is not in allowed_models_limits. Model: {model}")
            return

    
        if limits[model_lineika][model] is not None and usage.get(model, 0) >= limits[model_lineika][model]:
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
        message_text = update.message.reply_to_message.text or ""
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


        # Step 1: User replied with role name
        if reply_to_message and reply_to_message.from_user.id == self.bot_id and reply_to_message.text.startswith("NewRoleName:"):
            role_name = text.strip()

            # Validate name length
            if len(role_name) > self.max_role_name_length:
                await update.message.reply_text(f"Воу, воу, воу. Что это за длинющее имя такое. Пж не больше {self.max_role_name_length} символов.")
                return True

            # Check for duplicates
            global_roles = (await self.role_manager.get_global_roles()).values()
            chat_specific_roles = (await self.storage.get_available_roles(chat_id=update.message.chat_id)).values()
            global_roles_names = {role.name for role in global_roles}
            chat_specific_roles_names = {role.name for role in chat_specific_roles}

            if role_name in global_roles_names or role_name in chat_specific_roles_names:
                await update.message.reply_text(f"Сори, но такая личность уже есть. Пожалуйста, выбери другое имя.")
                return True

            # Edit the prompt message to remove the prefix
            await reply_to_message.edit_text("Как назовём новую личность?")

            # Ask for description
            num_current_roles = len(await self.storage.get_available_roles_ids(chat_id=update.message.chat_id))
            await context.bot.send_message(
                chat_id=update.message.chat_id,
                text=f"NewRoleAddition:{role_name}\nОтлично! Теперь опиши личность {role_name} (ответь на это сообщение описанием).\n\nP.s уже добавлено {num_current_roles}/{self.max_num_roles} личностей."
            )
            return True

        # Step 2: User replied with role description
        if reply_to_message and reply_to_message.from_user.id == self.bot_id and reply_to_message.text.startswith("NewRoleAddition:"):
            # Edit message after it has been replied to, and remove the NewRoleAddition prefix
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
        is_admin = self.storage.is_admin(update.message.from_user.id)

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
            if self.storage.is_admin(update.message.from_user.id):
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


    @command_in_identified_chats
    async def choose_model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Set the current model using buttons"""
        # Create buttons for Gemini models
        keyboard = []
        limits = self.model_manager.get_allowed_models_limits()

        # Add Gemini models
        gemini_buttons = []
        for model in limits.get('gemini', {}).keys():
            gemini_buttons.append(InlineKeyboardButton(model, callback_data=f"choose_model_{model}"))
        
        # Add Claude models
        claude_buttons = []
        for model in limits.get('claude', {}).keys():
            claude_buttons.append(InlineKeyboardButton(model, callback_data=f"choose_model_{model}"))

        # Add OpenAI models
        openai_buttons = []
        for model in limits.get('openai', {}).keys():
            openai_buttons.append(InlineKeyboardButton(model, callback_data=f"choose_model_{model}"))
        
        # Add buttons to keyboard
        keyboard.append(gemini_buttons)
        keyboard.append(claude_buttons)
        keyboard.append(openai_buttons)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Choose a model:", reply_markup=reply_markup)


    @command_in_identified_chats
    async def choose_model_button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle model selection button press"""
        query = update.callback_query
        await query.answer()
        
        chat_id = query.message.chat_id
        model_name = query.data.replace("choose_model_", "")
        
        # Set the model
        await self.storage.set_model(model_name, chat_id=chat_id)
        
        # Get updated chat info after setting the model
        current_model = await self.storage.get_model(chat_id=chat_id)
        
        # Update the message
        await query.edit_message_text(f"Current model set to {current_model}")


    @command_for_admin
    async def get_model_data_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Get the model data"""
        model_data = self.model_manager.get_all_model_usage()
        current_model = await self.storage.get_model(chat_id=update.message.chat_id)
        text = f"Current model (in this chat): {current_model}\nData across all chats:\n"
        for model in model_data:
            text += f"{model}: {model_data[model]} queries\n"

        await update.message.reply_text(text)


    @command_for_admin
    async def reset_model_data_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reset the model data"""
        self.model_manager.reset_model_data()
        await update.message.reply_text("Model data reset.")

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


    @command_for_admin
    async def add_role_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Add a role. Supports both interactive flow (/add_role) and direct (/add_role name)."""
        try:
            num_current_roles = len(await self.storage.get_available_roles_ids(chat_id=update.message.chat_id))
            if num_current_roles >= self.max_num_roles:
                await update.message.reply_text(f"Сори, у ботика уже слишком много личностей. Максимум {self.max_num_roles}. Если хочешь новую, то удали роль с помощью /remove_role.")
                return

            parts = update.message.text.split(" ", 1)

            # Interactive flow: just /add_role
            if len(parts) == 1:
                await context.bot.send_message(
                    chat_id=update.message.chat_id,
                    text=f"NewRoleName:\nКак назовём новую личность? (ответь на это сообщение)"
                )
                return

            # Direct flow: /add_role name
            name = parts[1].strip()

            if len(name) > self.max_role_name_length:
                await update.message.reply_text(f"Воу, воу, воу. Что это за длинющее имя такое. Пж не больше {self.max_role_name_length} символов.")
                return

            global_roles = (await self.role_manager.get_global_roles()).values()
            chat_specific_roles = (await self.storage.get_available_roles(chat_id=update.message.chat_id)).values()
            global_roles_names = {role.name for role in global_roles}
            chat_specific_roles_names = {role.name for role in chat_specific_roles}

            if name in global_roles_names or name in chat_specific_roles_names:
                await update.message.reply_text(f"Сори, но такая личность уже есть. Пожалуйста, выбери другое имя.")
                return

            await context.bot.send_message(
                chat_id=update.message.chat_id,
                text=f"NewRoleAddition:{name}\nОпиши личность {name} (ответь на это сообщение описанием).\n\nP.s уже добавлено {num_current_roles}/{self.max_num_roles} личностей."
            )
        except Exception as e:
            await self.send_error_message(update, context, f"Error: {e}")


    @command_for_admin
    async def give_bot_tokens_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Give the bot tokens"""
        tokens = int(update.message.text.split(" ", 1)[1])
        await self.model_manager.add_tokens(self.bot_id, tokens)
        await update.message.reply_text(f"Tokens have been added to the bot!")


    @command_in_identified_chats
    async def clear_history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Clear message history but keep tokens and settings"""
        await self.storage.clear_messages(update.message.chat_id)
        await update.message.reply_text("История сообщений очищена. Токены и настройки сохранены.")

    @command_in_identified_chats
    async def delete_chat_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Delete the chat completely (history, tokens, settings)"""
        await self.storage.delete_chat(update.message.chat_id)
        await update.message.reply_text("Чат полностью удалён. При следующем сообщении будет создан новый чат с начальными токенами.")


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
            roles = await self.role_manager.get_role_by_name(role_name)
            if not roles:
                await update.message.reply_text(f"Role '{role_name}' not found")
                return
                
            if len(roles) > 1:
                await update.message.reply_text(f"More than one role found: {', '.join([r.name for r in roles])}")
                return
            
            # Update the role's global status
            role = roles[0]
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


    @command_for_admin
    async def add_admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Add a user as admin. Usage: /add_admin <user_id>"""
        try:
            args = update.message.text.split()
            if len(args) != 2:
                await update.message.reply_text("Usage: /add_admin <user_id>")
                return

            user_id = int(args[1])
            if self.storage.add_admin_user(user_id):
                await update.message.reply_text(f"User {user_id} added as admin.")
            else:
                await update.message.reply_text(f"User {user_id} is already an admin.")
        except ValueError:
            await update.message.reply_text("Invalid user ID. Please provide a numeric ID.")

    @command_for_admin
    async def remove_admin_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Remove a user from admins. Usage: /remove_admin <user_id>"""
        try:
            args = update.message.text.split()
            if len(args) != 2:
                await update.message.reply_text("Usage: /remove_admin <user_id>")
                return

            user_id = int(args[1])
            # Prevent removing the primary admin from config
            if user_id == self.admin_user_id:
                await update.message.reply_text("Cannot remove the primary admin.")
                return

            if self.storage.remove_admin_user(user_id):
                await update.message.reply_text(f"User {user_id} removed from admins.")
            else:
                await update.message.reply_text(f"User {user_id} is not an admin.")
        except ValueError:
            await update.message.reply_text("Invalid user ID. Please provide a numeric ID.")

    @command_for_admin
    async def list_admins_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """List all admin users."""
        admin_ids = self.storage.get_admin_user_ids()
        if admin_ids:
            admin_list = "\n".join([f"• {uid}" for uid in admin_ids])
            await update.message.reply_text(f"Admin users:\n{admin_list}")
        else:
            await update.message.reply_text("No admins configured.")

    @command_for_admin
    async def get_settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show all bot settings."""
        settings = self.storage.get_bot_settings()
        lines = ["Bot Settings:"]
        for key, value in settings.items():
            lines.append(f"• {key}: {value}")
        await update.message.reply_text("\n".join(lines))

    @command_for_admin
    async def set_setting_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Set a bot setting. Usage: /set_setting <key> <value>"""
        try:
            args = update.message.text.split(maxsplit=2)
            if len(args) < 3:
                settings = self.storage.get_bot_settings()
                keys = ", ".join(settings.keys())
                await update.message.reply_text(f"Usage: /set_setting <key> <value>\n\nAvailable keys: {keys}")
                return

            key = args[1]
            value_str = args[2]

            # Validate key exists
            settings = self.storage.get_bot_settings()
            if key not in settings:
                await update.message.reply_text(f"Unknown setting: {key}\n\nAvailable: {', '.join(settings.keys())}")
                return

            # Convert value to appropriate type
            current_value = settings[key]
            if isinstance(current_value, bool):
                value = value_str.lower() in ('true', 'yes', '1', 'on')
            elif isinstance(current_value, int):
                value = int(value_str)
            elif isinstance(current_value, float):
                value = float(value_str)
            elif current_value is None:
                # For None values (like default_role_id), keep as string or None
                value = None if value_str.lower() in ('none', 'null', '') else value_str
            else:
                value = value_str

            if self.storage.set_bot_setting(key, value):
                await update.message.reply_text(f"Setting updated: {key} = {value}")
            else:
                await update.message.reply_text(f"Failed to update setting: {key}")
        except ValueError as e:
            await update.message.reply_text(f"Invalid value: {e}")

    async def send_spontaneous_message(self, chat_id: int, bot: Bot) -> bool:
        """
        Send a spontaneous message to a chat if conditions are met.

        Args:
            chat_id: The Telegram chat ID to send to
            bot: The Telegram Bot instance

        Returns:
            bool: True if message was sent, False otherwise
        """
        try:
            # Get chat info
            chat_info = await self.storage.get_chat_info(chat_id=chat_id)
            if not chat_info:
                logger.warning(f"Chat {chat_id} not found for spontaneous message")
                return False

            # Check if bot has tokens
            balance = await self.model_manager.get_tokens(self.bot_id)
            if balance is None or balance <= 0:
                logger.info(f"Bot has no tokens for spontaneous message")
                return False

            # Roll the dice based on come_to_life_chance
            chance = chat_info.get('come_to_life_chance', 0)
            if random() >= chance:
                logger.debug(f"Spontaneous message dice roll failed for chat {chat_id}")
                return False

            # Set up context
            self.current_role_id = chat_info['current_role_id']
            self.notes = chat_info['notes']
            self.current_role = await self.storage.get_current_role(chat_id=chat_id)
            self.chat_type = 'group'  # Assume group for spontaneous messages
            model_name = chat_info['model']

            # Initialize LLM service
            provider_key = _provider_key(model_name)
            service_cls = PROVIDERS[provider_key]

            video_analyzer = GeminiService(
                api_key=self.api_keys['gemini'],
                model_name=self.video_analyzer_model,
                thinking_model=False
            ) if provider_key != 'gemini' else None

            init_kwargs = {
                'api_key': self.api_keys[provider_key],
                'model_name': model_name,
                'video_analyzer': video_analyzer,
            }
            self.llm_service = service_cls(**init_kwargs)

            # Get recent messages for context
            context_messages = await self.storage.get_recent_messages(chat_id, limit=SHORT_TERM_MEMORY)
            if not context_messages:
                logger.info(f"No messages in chat {chat_id} for spontaneous response")
                return False

            # Generate response
            response_text, input_tokens, output_tokens, thinking, _ = await self.llm_service.get_response(
                system_prompt=self.get_full_system_prompt(),
                context_messages=context_messages
            )

            # Deduct tokens from bot's balance
            await self.deduct_tokens(self.bot_id, input_tokens, output_tokens, self.llm_service.output_tokens_multiplier)

            # Update model usage
            self.model_manager.used_model(self.llm_service.model_name)

            # Sanitize and send
            cleaned_text = re.sub(r'^\s*message_id:[^\n]*\n?', '', response_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'^\s*\[(?:Image|GIF/Animation|Video|Sticker|Document):[^\]]*\]\s*', '', cleaned_text, flags=re.IGNORECASE)

            sent_message = await bot.send_message(chat_id=chat_id, text=cleaned_text)

            # Store the bot's message
            bot_message = ChatMessage(
                message_id=sent_message.message_id,
                user=BOT_USER_DESCRIPTION,
                content=response_text,
                timestamp=sent_message.date.strftime("%a, %d. %b %Y %H:%M"),
                reply_to_id=None,
                reasoning=thinking,
            )
            await self.storage.add_message(chat_id, bot_message)

            logger.info(f"Sent spontaneous message to chat {chat_id}")
            return True

        except Exception as e:
            logger.error(f"Error sending spontaneous message to chat {chat_id}: {e}", exc_info=True)
            return False


PROVIDERS = {
    'claude': ClaudeService,
    'gemini': GeminiService,
    'openai': OpenAIService,
}

def _provider_key(model_name: str) -> str:
    if model_name.startswith('gemini'):  return 'gemini'
    if model_name.startswith('claude'):  return 'claude'
    if model_name.startswith(('openai', 'gpt', 'o1', 'o3')):  return 'openai'
    raise ValueError(f'Unknown model: {model_name}')