# Media Handling Features

This Telegram bot now supports comprehensive media handling with automatic descriptions using Google Gemini's multimodal capabilities.

## Supported Media Types

### 1. **Images** 📷
- Automatically describes images using Gemini's vision capabilities
- Provides concise 1-2 sentence descriptions focusing on main subjects and context
- Example: `[Image: A golden retriever playing with a tennis ball in a sunny park]`

### 2. **Videos** 🎥
- Captures video metadata (duration, resolution)
- Example: `[Video: 30s, 1920x1080]`
- Future enhancement: Frame extraction for content analysis

### 3. **Stickers** 🎨
- Identifies sticker type (regular, animated, video)
- Extracts emoji representation and sticker set name
- Example: `[Animated sticker: 😂 from funny_pack]`

### 4. **GIFs/Animations** 🎬
- Full visual content analysis using Gemini's video processing capabilities
- Analyzes motion, objects, scenes, and actions within the animation
- Includes metadata (duration, resolution) plus AI-generated content description
- Example: `[GIF/Animation: 5s, 480x360 - please analyze the visual content and motion in this animation]`
- **NEW**: Raw animation bytes are sent to Gemini for comprehensive visual understanding
- Detects GIFs that Telegram may deliver as a `document` (often converted to MP4) and relabels them internally as `animation` for consistent processing
- Falls back to MIME-type sniffing (`image/gif`, `video/mp4`) when standard image decoders cannot open the file

### 5. **Documents** 📄
- Shows filename, file type, and size
- GIFs and MP4s sent as documents are automatically reclassified as animations (see above)
- Example: `[Document: report.pdf (application/pdf, 2.3 MB)]`

### 6. **Audio** 🎵
- Displays title, artist, and duration
- Example: `[Audio: Bohemian Rhapsody by Queen (5:55)]`

### 7. **Voice Messages** 🎤
- Shows duration of voice messages
- Example: `[Voice message (15 seconds)]`

## How It Works

1. **Media Detection**: When a user sends any media, the bot automatically detects the type
2. **Description Generation**: 
   - For images: Downloads and sends to Gemini for AI-powered description
   - For GIFs/animations: Downloads raw bytes and sends to Gemini for full visual content analysis
   - For other media: Extracts and formats metadata
3. **Context Integration**: Media descriptions are included in chat history, allowing the AI to reference previously shared media
4. **Storage**: All media information is stored with messages for full conversation context

## Technical Implementation

### Media Handler (`handlers/media_handler.py`)
- Processes all incoming media messages
- Uses Google Gemini API for image descriptions
- Handles errors gracefully with fallback descriptions
- Detects GIF/MP4 documents and converts them into `animation` messages for unified downstream handling  
- Falls back to MIME sniffing when Pillow fails and applies size-limit and duplicate-prevention guards

### Message Model (`models.py`)
- Extended to support media fields:
  - `media_type`: Type of media (image, video, sticker, etc.)
  - `file_id`: Telegram file identifier
  - `media_description`: Generated description
  - `sticker`: Dictionary with emoji and animation info

### Message Formatting (`services/base_service.py`)
- Formats messages to include media descriptions in AI context
- Example format: `User [timestamp] [Image: Description]: Caption text`

## Configuration

The media handler requires a Google Gemini API key, which should be provided in your environment configuration. The bot uses the `gemini-2.0-flash` model (or your configured default) for image and animation descriptions.

## Usage Examples

1. **Sharing a Photo**
   ```
   User: [sends photo of a cat]
   Bot sees: "User [10:30] [Image: A tabby cat sleeping on a windowsill with sunlight streaming in]: Look at my cat!"
   ```

2. **Sending a Sticker**
   ```
   User: [sends laughing emoji sticker]
   Bot sees: "User [10:31] [Animated sticker: 😂 from default_pack]:"
   ```

3. **Sharing a Document**
   ```
   User: [sends PDF file]
   Bot sees: "User [10:32] [Document: meeting_notes.pdf (application/pdf, 1.2 MB)]: Here are the notes"
   ```

## Benefits

- **Enhanced Context**: The AI understands what media was shared, improving response relevance
- **Accessibility**: Media content is described in text form
- **Memory**: Media descriptions are stored in chat history for long-term context
- **Natural Integration**: Descriptions appear naturally in the conversation flow 