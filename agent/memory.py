from session_memory.sliding_window import (
    sliding_window, count_conversation_tokens, MAX_TOKENS
)
from session_memory.summarization import summarize_history

SUMMARIZE_TOKEN_THRESHOLD = 3000
SUMMARIZE_TURN_THRESHOLD  = 10
KEEP_RECENT_MESSAGES      = 4   # keep last 2 turns (user + assistant each) verbatim


def apply_memory(messages: list[dict], strategy: str) -> tuple[list[dict], dict]:
    """
    Apply the chosen memory strategy to chat_history.
    Returns (new_messages, stats) where stats has token counts for measurement.
    """
    before = count_conversation_tokens(messages)

    if strategy == "sliding":
        result = sliding_window(messages)
    elif strategy == "summarize":
        result = _apply_summarization(messages)
    else:
        result = messages  # "full" — no trimming

    after = count_conversation_tokens(result)
    stats = {
        "strategy": strategy,
        "messages_before": len(messages),
        "messages_after":  len(result),
        "tokens_before":   before,
        "tokens_after":    after,
        "tokens_saved":    before - after,
    }
    return result, stats


def _apply_summarization(messages: list[dict]) -> list[dict]:
    turns = sum(1 for m in messages if m.get("role") == "user")
    tokens = count_conversation_tokens(messages)

    if tokens < SUMMARIZE_TOKEN_THRESHOLD and turns < SUMMARIZE_TURN_THRESHOLD:
        return messages

    if len(messages) <= KEEP_RECENT_MESSAGES:
        return messages

    to_summarize = messages[:-KEEP_RECENT_MESSAGES]
    recent       = messages[-KEEP_RECENT_MESSAGES:]

    summary = summarize_history(to_summarize)
    return summary + recent


def print_memory_stats(stats: dict) -> None:
    if stats["tokens_saved"] <= 0:
        return
    print(
        f"\n[memory/{stats['strategy']}] "
        f"{stats['messages_before']}→{stats['messages_after']} msgs | "
        f"{stats['tokens_before']}→{stats['tokens_after']} tokens "
        f"(saved {stats['tokens_saved']})"
    )
