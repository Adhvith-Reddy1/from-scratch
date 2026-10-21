"""
Answer relevance evaluation — Claude-as-judge.

Asks: does the answer actually address what the question asked?
Returns a score in [1, 5] and a one-sentence explanation.
"""

import re
from anthropic import Anthropic

_client = Anthropic()
_MODEL  = "claude-haiku-4-5-20251001"

_PROMPT = """\
Rate how well this answer addresses the question on a scale of 1–5.

Rubric:
  1 — Off-topic, refuses to answer, or answers a completely different question
  2 — Barely touches the question; misses the main point
  3 — Partially addresses the question but leaves significant gaps
  4 — Mostly answers the question with only minor omissions
  5 — Directly and completely answers the question as asked

Question: {question}

Answer:
{answer}

Respond in exactly this format (no other text):
SCORE: <integer 1–5>
REASON: <one sentence>"""


def evaluate_relevance(question: str, answer: str) -> dict:
    """
    Returns:
        score       — float in [1, 5]
        explanation — one-sentence reason from the judge
    """
    prompt = _PROMPT.format(question=question, answer=answer)

    resp = _client.messages.create(
        model=_MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()

    score_match  = re.search(r"SCORE:\s*([1-5])", raw)
    reason_match = re.search(r"REASON:\s*(.+)", raw)

    score       = float(score_match.group(1))  if score_match  else 3.0
    explanation = reason_match.group(1).strip() if reason_match else raw[:120]

    return {"score": score, "explanation": explanation}
