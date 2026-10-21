"""
Retrieval metrics — pure functions, no LLM calls.

All functions take retrieved_ids (ordered list) and relevant_ids (set).
"""

from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    query_id: int
    query: str
    retrieved_ids: list[str]
    relevant_ids: set[str] = field(default_factory=set)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return sum(1 for rid in relevant_ids if rid in top_k) / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def aggregate(results: list[RetrievalResult], k: int) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        f"precision@{k}": round(sum(precision_at_k(r.retrieved_ids, r.relevant_ids, k) for r in results) / n, 4),
        f"recall@{k}":    round(sum(recall_at_k(r.retrieved_ids, r.relevant_ids, k)    for r in results) / n, 4),
        "mrr":            round(sum(reciprocal_rank(r.retrieved_ids, r.relevant_ids)    for r in results) / n, 4),
    }
