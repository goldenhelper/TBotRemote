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
        "max_role_name_length": 100
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
