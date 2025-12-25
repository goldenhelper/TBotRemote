#!/usr/bin/env python3
"""
Test the response cleaning and reply_to parsing logic.
"""
import re

def parse_reply_to(response_text: str) -> tuple:
    """Parse optional reply_to from response, return (reply_to_id, remaining_text)"""
    reply_match = re.match(r'^\s*reply_to:\s*(\d+)\s*\n', response_text, flags=re.IGNORECASE)
    if reply_match:
        reply_to_id = int(reply_match.group(1))
        remaining = response_text[reply_match.end():]
        return reply_to_id, remaining
    return None, response_text

def clean_response(response_text: str) -> str:
    """Replicate the cleaning logic from message_handler.py"""
    # First regex: remove message_id metadata lines
    cleaned_text = re.sub(r'^\s*message_id:[^\n]*\n?', '', response_text, flags=re.IGNORECASE)

    # Second regex: remove media description headers
    cleaned_text = re.sub(r'^\s*\[(?:Image|GIF/Animation|Video|Sticker|Document):[^\]]*\]\s*', '', cleaned_text, flags=re.IGNORECASE)

    return cleaned_text


print("="*60)
print("Testing reply_to parsing:")
print("="*60)

reply_to_tests = [
    # (input, expected_reply_to, expected_message_start)
    ("reply_to: 15519\nHaha, that's exactly what I was thinking!", 15519, "Haha"),
    ("Reply_To: 12345\nSome response", 12345, "Some"),
    ("  reply_to: 999\nIndented", 999, "Indented"),
    ("No reply_to here, just a message", None, "No reply"),
    ("reply_to:123\nNo space after colon", None, "reply_to"),  # Requires space
    ("reply_to: abc\nNot a number", None, "reply_to"),
    ("", None, ""),
    ("reply_to: 555\n", 555, ""),  # reply_to with empty message after
]

for input_text, expected_id, expected_start in reply_to_tests:
    reply_id, remaining = parse_reply_to(input_text)

    id_ok = reply_id == expected_id
    start_ok = remaining.startswith(expected_start) if expected_start else True

    status = "✓" if id_ok and start_ok else "✗ FAIL"

    print(f"{status} Input: {repr(input_text)[:50]}")
    print(f"     Reply ID: {reply_id} (expected: {expected_id})")
    print(f"     Remaining starts with: {repr(remaining[:20]) if remaining else repr('')}")
    print()


print("="*60)
print("Testing cleaning logic:")
print("="*60)

# Test cases
test_cases = [
    # (input, expected_non_empty, description)
    ("", False, "Empty string"),
    ("   ", False, "Whitespace only"),
    ("Hello world!", True, "Normal message"),
    ("message_id: 123\nHello!", True, "Message with metadata prefix"),
    ("Привет! Как дела?", True, "Russian text"),
    ("🎄 Merry Christmas!", True, "Emoji message"),
]

for input_text, should_have_content, description in test_cases:
    cleaned = clean_response(input_text)
    has_content = bool(cleaned and cleaned.strip())

    status = "✓" if has_content == should_have_content else "✗ FAIL"

    print(f"{status} {description}")
    print(f"   Input:   {repr(input_text)[:60]}")
    print(f"   Output:  {repr(cleaned)[:60]}")
    print()
