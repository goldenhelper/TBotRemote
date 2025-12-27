-- Supabase Schema for Telegram Bot
-- Run this in the Supabase SQL Editor

-- 1. Chats table (chat settings/metadata)
CREATE TABLE chats (
    chat_id BIGINT PRIMARY KEY,
    model TEXT NOT NULL DEFAULT 'gemini-2.0-flash',
    memory_updater_model TEXT NOT NULL DEFAULT 'gemini-2.0-flash',
    current_role_id UUID,
    come_to_life_chance REAL DEFAULT 0.1,
    notes_text TEXT DEFAULT '',
    notes_last_updated_msgs_ago INTEGER DEFAULT 0,
    tokens INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Messages table (conversation history)
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    chat_id BIGINT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    message_id BIGINT NOT NULL,  -- Telegram's message ID
    user_name TEXT NOT NULL,
    content TEXT DEFAULT '',
    timestamp TEXT,  -- Keep as text to match your current format
    reply_to_id BIGINT,  -- Telegram message ID this replies to
    media_type TEXT,
    file_id TEXT,
    media_description TEXT,
    sticker JSONB DEFAULT '{"emoji": null, "is_animated": null, "is_video": null}',
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Ensure unique message per chat
    UNIQUE(chat_id, message_id)
);

-- Index for fast lookups
CREATE INDEX idx_messages_chat_id ON messages(chat_id);
CREATE INDEX idx_messages_chat_created ON messages(chat_id, created_at DESC);

-- 3. Roles table
CREATE TABLE roles (
    role_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_name TEXT NOT NULL,
    role_prompt TEXT NOT NULL,
    is_global BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for global roles lookup
CREATE INDEX idx_roles_global ON roles(is_global) WHERE is_global = TRUE;

-- 4. Chat-specific roles (junction table)
CREATE TABLE chat_roles (
    chat_id BIGINT REFERENCES chats(chat_id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(role_id) ON DELETE CASCADE,
    PRIMARY KEY (chat_id, role_id)
);

-- 5. Model usage tracking
CREATE TABLE model_usage (
    model_name TEXT PRIMARY KEY,
    query_count INTEGER DEFAULT 0,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 6. Config table (replaces SSM Parameter Store)
CREATE TABLE config (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default model limits
INSERT INTO config (key, value) VALUES (
    'allowed_models_limits',
    '{
        "gemini": {
            "gemini-2.0-flash": 1000,
            "gemini-2.5-pro-preview-05-06": 100
        },
        "claude": {
            "claude-sonnet-4-20250514": 100
        },
        "openai": {
            "gpt-4o": 100,
            "o3-2025-04-16": 50
        }
    }'::jsonb
);

-- Insert admin user IDs (replace with your actual admin user IDs)
INSERT INTO config (key, value) VALUES (
    'admin_user_ids',
    '[]'::jsonb
);

-- Insert bot settings (can be changed via Telegram commands)
-- Note: memory_updater_prompt and video_analyzer_prompt use placeholders:
--   {role_prompt} - the current role's prompt
--   {short_term_memory} - number of messages for context
--   {notes_text} - current notes content
--   {output_format} - output format instructions (auto-generated based on model type)
INSERT INTO config (key, value) VALUES (
    'bot_settings',
    '{
        "default_model": "gemini-2.0-flash",
        "default_memory_updater_model": "gemini-2.0-flash",
        "video_analyzer_model": "gemini-2.0-flash",
        "default_role_id": null,
        "default_come_to_life_chance": 0.1,
        "default_tokens_for_new_chats": 15000,
        "max_num_roles": 10,
        "max_role_name_length": 100,
        "short_term_memory": 25,
        "memory_updater_prompt": "<role>\n{role_prompt}\n</role>\n<task>\nSYSTEM PROMPT UPDATE: Refresh your long-term memory (notes).\nYour existing notes are below and you will be provided the last {short_term_memory} messages. Update your notes by adding new critical information that aligns with your role. Prioritize facts that help you stay consistent, personalized, and effective in your role.\n</task>\n<current_notes>\n{notes_text}\n</current_notes>\n<instructions>\n1. **Chain-of-Thought Process**:\n    - **Step 2**: Analyze recent messages. Identify key details with special attention to:\n        * Character traits, preferences, and patterns\n        * Time-sensitive information (events, deadlines, future plans)\n        * Emotional states or personal situations\n        * For group chats: Interpersonal dynamics and relationships between members\n    - **Step 3**: Cross-check with existing notes. Flag duplicates, outdated info, or role-critical gaps.\n    - **Step 4**: Decide what to add/modify/retain based on:\n        * Relevance to your role\n        * Temporal importance (attach dates where applicable)\n        * Long-term value for relationship building\n        * Group dynamics (for group chats)\n\n2. **Temporal Information Handling**:\n    - For time-sensitive information (e.g., \"birthday next week\", \"job interview tomorrow\"):\n        * Add a temporal tag [Until: YYYY-MM-DD] for expiration dates\n        * Update/remove expired time-sensitive entries\n    - For persistent traits or preferences:\n        * Mark with [Core Trait] to indicate high retention priority\n    - For temporary states (moods, situations):\n        * Use to update understanding of character but don''t retain specific instances unless pattern-forming\n\n3. **Memory Retention Guidelines**:\n    - Preserve information about core traits, preferences, and important facts\n    - Be conservative with deletions - only remove notes that are:\n        * Conclusively outdated or superseded\n        * Contradicted by newer, more reliable information\n        * No longer relevant to your role or the conversation trajectory\n    - If uncertain about relevance, retain the information\n    - Condense similar or related information to maintain conciseness\n\n4. **Group Chat Relationship Tracking**:\n    - Map relationships and dynamics between members:\n        * Identify close friendships, rivalries, or professional connections\n        * Note conversation patterns (who talks to whom, response tones)\n        * Record shared experiences or inside references between members\n        * Track shifting alliances or relationship changes over time\n    - Use relationship tags to organize interpersonal information:\n        * [Relation: Person1-Person2] to mark relationship-specific notes\n        * [Group Dynamic] for patterns involving multiple members\n    - Consider the social context when responding\n\n5. **Output Format**:\n    {output_format}\n</instructions>",
        "video_analyzer_prompt": "<task>\nYou are a VIDEO ANALYZER providing factual descriptions for another AI model (the main bot).\nYour ONLY job is to describe what you see in the video/animation.\nThe main bot (described above) will use your description to respond to the user in its own character.\nYou are NOT the main bot - you are just the analyzer providing visual information.\nDo NOT adopt the main bot''s personality or respond to the user yourself.\n</task>\n\n<instructions>\n1. You are ONLY the video analyzer - not the main bot with the personality described above.\n2. Observe key visual elements: people, objects, actions, text, colours, lighting, atmosphere, movement.\n3. Mention important audio cues only if they change understanding.\n4. Avoid speculation beyond what is visible/audible.\n5. Write a neutral, factual description that helps the main bot understand the content.\n6. Do NOT address the user directly, provide commentary, reactions, or adopt the main bot''s personality.\n7. Do NOT use meta-phrases such as \"This video shows\" or \"Here is a description\".\n8. Write 6-12 sentences (or more if needed) to capture all significant details; err on the side of thoroughness rather than brevity. Use the chat''s language.\n</instructions>\n\n<output_format>\nReturn ONLY the description text - no direct user address.\n</output_format>"
    }'::jsonb
);

-- Function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers
CREATE TRIGGER chats_updated_at
    BEFORE UPDATE ON chats
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER model_usage_updated_at
    BEFORE UPDATE ON model_usage
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER config_updated_at
    BEFORE UPDATE ON config
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Function to enforce message limit per chat (1000 messages)
CREATE OR REPLACE FUNCTION enforce_message_limit()
RETURNS TRIGGER AS $$
BEGIN
    -- Delete oldest messages if we exceed 1000 per chat
    DELETE FROM messages
    WHERE id IN (
        SELECT id FROM messages
        WHERE chat_id = NEW.chat_id
        ORDER BY created_at DESC
        OFFSET 1000
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER limit_messages_per_chat
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION enforce_message_limit();

-- Add foreign key for current_role_id (after roles table exists)
ALTER TABLE chats
    ADD CONSTRAINT fk_chats_current_role
    FOREIGN KEY (current_role_id) REFERENCES roles(role_id) ON DELETE SET NULL;

-- Row Level Security (optional but recommended)
-- Enable RLS on all tables
ALTER TABLE chats ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_usage ENABLE ROW LEVEL SECURITY;
ALTER TABLE config ENABLE ROW LEVEL SECURITY;

-- For now, allow all operations (your bot uses service role key)
-- You can tighten this later if needed
CREATE POLICY "Allow all for service role" ON chats FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON messages FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON roles FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON chat_roles FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON model_usage FOR ALL USING (true);
CREATE POLICY "Allow all for service role" ON config FOR ALL USING (true);
