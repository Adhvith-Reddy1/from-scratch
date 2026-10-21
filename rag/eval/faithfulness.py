"""
Faithfulness evaluation — Claude-as-judge.

Asks: does the answer only contain claims supported by the retrieved context?
Returns a score in [0, 1] and a list of unsupported claims.
"""

from anthropic import Anthropic

_client = Anthropic()
_MODEL  = "claude-haiku-4-5-20251001"

_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to its source context.

Question asked:
{question}

Retrieved context (the ONLY allowed source of information):
{context}

Answer to evaluate:
{answer}

Task: Identify every specific factual claim in the answer that is NOT directly supported \
by the retrieved context above. A claim is unsupported if it contains information that \
cannot be verified from the given passages — even if the claim may be true in general.

Rules:
- List each unsupported claim on its own line starting with "UNSUPPORTED: "
- If every claim in the answer is backed by the context, output exactly: FULLY_SUPPORTED
- Do NOT flag hedged statements like "I could not find..." or "The context does not mention..."
- Do NOT flag reasoning or paraphrasing that stays within what the context says"""


def evaluate_faithfulness(
    question: str,
    context_chunks: list[str],
    answer: str,
) -> dict:
    """
    Returns:
        score          — float in [0, 1]; 1 = fully supported, decreases by 0.15 per unsupported claim
        unsupported    — list of unsupported claim strings
        count          — number of unsupported claims
    """
    context = "\n\n---\n\n".join(chunk.strip() for chunk in context_chunks if chunk.strip())
    prompt  = _PROMPT.format(question=question, context=context, answer=answer)

    resp = _client.messages.create(
        model=_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()

    if raw == "FULLY_SUPPORTED":
        unsupported = []
    else:
        unsupported = [
            line.removeprefix("UNSUPPORTED:").strip()
            for line in raw.splitlines()
            if line.strip().startswith("UNSUPPORTED:")
        ]

    score = max(0.0, round(1.0 - len(unsupported) * 0.15, 3))
    return {"score": score, "unsupported": unsupported, "count": len(unsupported)}
