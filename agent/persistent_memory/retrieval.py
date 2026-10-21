"""
Episode retrieval via embedding similarity.

Uses OpenAI text-embedding-3-small (already a project dependency).

get_embedding()         — returns a float vector for a string
find_relevant_episodes() — top-k cosine-similar past episodes for a query
"""

import math
from openai import OpenAI

from .store import get_all_episodes

_client: OpenAI | None = None
_EMBED_MODEL = "text-embedding-3-small"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def get_embedding(text: str) -> list[float]:
    """Return a vector embedding for text via OpenAI."""
    text = text.replace("\n", " ").strip()
    resp = _get_client().embeddings.create(model=_EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _cosine(a: list[float], b: list[float]) -> float:
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_relevant_episodes(query: str, top_k: int = 3) -> list[dict]:
    """
    Return the top_k past episodes most semantically similar to query.
    Each returned dict has: summary, turn_count, created_at, similarity.
    Episodes with similarity < 0.3 are excluded as irrelevant.
    """
    episodes = get_all_episodes()
    if not episodes:
        return []

    query_vec = get_embedding(query)

    scored = []
    for ep in episodes:
        sim = _cosine(query_vec, ep["embedding"])
        if sim >= 0.3:
            scored.append({**ep, "similarity": round(sim, 3)})

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:top_k]
