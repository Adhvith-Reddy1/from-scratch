"""
LLM-powered fact extraction and conversation summarization.

extract_facts()  — pulls durable user facts from a finished conversation
summarize()      — produces a short paragraph summary for episode storage
"""

from anthropic import Anthropic

_client = Anthropic()
_MODEL  = "claude-haiku-4-5-20251001"


def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _history_text(messages: list[dict]) -> str:
    lines = []
    for m in messages:
        role    = m.get("role", "unknown")
        content = _content_to_str(m.get("content", ""))
        if content.strip():
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def extract_facts(messages: list[dict]) -> list[dict[str, str]]:
    """
    Ask the LLM to pull persistent facts from the conversation.
    Returns a list of {"key": ..., "value": ...} dicts.

    The LLM is prompted to emit one fact per line as  KEY: VALUE
    so we can parse without JSON (which Haiku sometimes mangles).
    """
    history = _history_text(messages)
    if not history.strip():
        return []

    prompt = f"""You are extracting persistent facts about the user from a conversation.

Return ONLY facts that are durable and would be useful in future conversations:
- Preferences ("prefers concise answers", "uses dark mode")
- Projects or work context ("working on Agent_Learning project", "uses Python")
- Personal context ("name is Alex", "works at Acme Corp")

Do NOT include facts that are only relevant to this single conversation.
Do NOT include anything the assistant said — only things about the USER.

Format: one fact per line as   KEY: VALUE
If there are no durable facts, output exactly: NONE

Conversation:
{history}"""

    response = _client.messages.create(
        model=_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()

    if raw.upper() == "NONE" or not raw:
        return []

    facts = []
    for line in raw.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key   = key.strip()
        value = value.strip()
        if key and value:
            facts.append({"key": key, "value": value})

    return facts


def summarize(messages: list[dict]) -> str:
    """
    Produce a short paragraph summary of the conversation for episode storage.
    """
    history = _history_text(messages)
    if not history.strip():
        return ""

    prompt = f"""Summarize this conversation in 2-4 sentences. Include:
- The main topic(s) discussed
- Any decisions or conclusions reached
- Any code, files, or systems that were worked on

Conversation:
{history}"""

    response = _client.messages.create(
        model=_MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
