from anthropic import Anthropic

_summary_client = Anthropic()

def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""

def summarize_history(messages: list[dict]) -> list[dict]:
    history_text = "\n".join(
        f"{m.get('role', 'unknown')}: {_content_to_str(m.get('content', ''))}"
        for m in messages
    )
    prompt = f"""Summarize this chat history. Retain all relevant details, key facts, and decisions made.

Chat history:
{history_text}"""

    response = _summary_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = response.content[0].text.strip()
    return [{"role": "user", "content": f"The following is a summary of our conversation so far. Use this as context to continue: {summary}"}]
