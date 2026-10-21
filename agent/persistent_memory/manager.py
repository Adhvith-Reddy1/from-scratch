"""
Orchestrates persistent memory across conversation boundaries.

Call on_conversation_start() before the first LLM turn to get a context block.
Call on_conversation_end() after the user exits to persist facts + episode.
"""

from .store import init_db, upsert_fact, get_all_facts, save_episode
from .extraction import extract_facts, summarize
from .retrieval import get_embedding, find_relevant_episodes


def on_conversation_start(opening_query: str) -> str:
    """
    Build the memory context block to inject into the system prompt.

    1. Load all stored facts (always injected — they're small)
    2. Find episodes relevant to the opening query
    3. Return a formatted string ready to append to the system prompt
    """
    init_db()

    sections = []

    facts = get_all_facts()
    if facts:
        fact_lines = "\n".join(f"- {f['key']}: {f['value']}" for f in facts)
        sections.append(f"## Remembered facts about the user\n{fact_lines}")

    episodes = find_relevant_episodes(opening_query, top_k=3)
    if episodes:
        ep_lines = []
        for ep in episodes:
            ep_lines.append(
                f"- [{ep['created_at'][:10]}] (similarity {ep['similarity']}) {ep['summary']}"
            )
        sections.append(f"## Relevant past conversations\n" + "\n".join(ep_lines))

    if not sections:
        return ""

    return "\n\n---\n## Persistent Memory\n" + "\n\n".join(sections)


def on_conversation_end(messages: list[dict]) -> dict:
    """
    Persist facts and episode from the completed conversation.

    1. Extract facts → upsert each into the facts table
    2. Summarize the conversation → save as episode with embedding
    Returns a stats dict for display.
    """
    init_db()

    # --- fact extraction ---
    new_facts = extract_facts(messages)
    for f in new_facts:
        upsert_fact(f["key"], f["value"])

    # --- episode storage ---
    summary = summarize(messages)
    episode_id = None
    if summary:
        embedding  = get_embedding(summary)
        turn_count = sum(1 for m in messages if m.get("role") == "user")
        episode_id = save_episode(summary, embedding, turn_count)

    return {
        "facts_extracted": len(new_facts),
        "episode_saved":   episode_id is not None,
        "episode_id":      episode_id,
        "summary_preview": summary[:120] + "..." if len(summary) > 120 else summary,
    }
