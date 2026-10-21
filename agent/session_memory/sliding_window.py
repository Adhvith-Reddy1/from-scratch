import tiktoken

MAX_TOKENS = 3000
TRIM_TOKENS = 500

encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))

def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""

def count_conversation_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(_content_to_str(m.get("content", ""))) for m in messages)

def sliding_window(messages: list[dict]) -> list[dict]:
    if count_conversation_tokens(messages) <= MAX_TOKENS:
        return messages

    first = messages[:1]      # always keep the opening user message
    rest  = messages[1:]

    count = 0
    cutoff = 0
    for i, msg in enumerate(rest):
        count += count_tokens(_content_to_str(msg.get("content", "")))
        if count >= TRIM_TOKENS:
            cutoff = i + 1
            break

    return first + rest[cutoff:]
