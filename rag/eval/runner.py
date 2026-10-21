"""
RAG eval runner — orchestrates retrieve → generate → judge for each question.

Usage:
    config = EvalConfig(name="...", retriever_fn=..., k=5, ...)
    report = run_eval(config, questions)
    print_report(report)
    save_report(report, "results.json")
"""

import time
import json
import ollama
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

from anthropic import Anthropic

from .metrics import RetrievalResult, aggregate, precision_at_k, recall_at_k, reciprocal_rank
from .faithfulness import evaluate_faithfulness
from .relevance import evaluate_relevance

_client     = Anthropic()
_GEN_MODEL  = "claude-haiku-4-5-20251001"
_JUDGE_MODEL = "llama3.1:8b"   # local judge for retrieval relevance (cheap, no API cost)

# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    name: str                      # human-readable label for the report header
    retriever_fn: Callable         # (query: str, k: int) -> list[dict] — each dict has 'id' and 'text'
    k: int = 5
    chunking_desc: str = ""        # e.g. "recursive, 500 chars"
    embed_model: str   = "nomic-embed-text"
    generator_model: str = _GEN_MODEL


@dataclass
class QuestionResult:
    question_id: int
    question: str
    retrieved_texts: list[str]
    retrieved_ids: list[str]
    relevant_ids: set[str]
    answer: str
    faithfulness: dict             # {score, unsupported, count}
    relevance: dict                # {score, explanation}
    retrieval_time_s: float
    generation_time_s: float

    @property
    def total_time_s(self) -> float:
        return self.retrieval_time_s + self.generation_time_s

    def retrieval_result(self) -> RetrievalResult:
        return RetrievalResult(
            query_id=self.question_id,
            query=self.question,
            retrieved_ids=self.retrieved_ids,
            relevant_ids=self.relevant_ids,
        )


@dataclass
class EvalReport:
    config: EvalConfig
    question_results: list[QuestionResult]
    generated_at: str = field(default_factory=lambda: date.today().isoformat())

    # ── aggregate metrics ─────────────────────────────────────────────────────

    def retrieval_metrics(self) -> dict:
        return aggregate([r.retrieval_result() for r in self.question_results], self.config.k)

    def avg_faithfulness(self) -> float:
        scores = [r.faithfulness["score"] for r in self.question_results]
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    def avg_unsupported_claims(self) -> float:
        counts = [r.faithfulness["count"] for r in self.question_results]
        return round(sum(counts) / len(counts), 2) if counts else 0.0

    def avg_relevance(self) -> float:
        scores = [r.relevance["score"] for r in self.question_results]
        return round(sum(scores) / len(scores), 2) if scores else 0.0

    def avg_latency(self) -> float:
        times = [r.total_time_s for r in self.question_results]
        return round(sum(times) / len(times), 2) if times else 0.0

    def worst_performers(self, n: int = 3) -> list[dict]:
        """
        Return the n worst results by composite badness score.
        Flags the primary failure mode for each.
        """
        k = self.config.k
        scored = []
        for r in self.question_results:
            rr = r.retrieval_result()
            p  = precision_at_k(rr.retrieved_ids, rr.relevant_ids, k)
            rc = recall_at_k(rr.retrieved_ids, rr.relevant_ids, k)
            rrank = reciprocal_rank(rr.retrieved_ids, rr.relevant_ids)
            faith = r.faithfulness["score"]
            rel   = r.relevance["score"] / 5.0

            # lower is worse
            composite = (p + rc + rrank + faith + rel) / 5.0

            if p == 0 and rc == 0:
                reason = "retrieval miss (no relevant chunks)"
            elif r.faithfulness["count"] > 0:
                reason = f"hallucinated {r.faithfulness['count']} claim(s)"
            elif r.relevance["score"] <= 2:
                reason = "answer not relevant"
            else:
                reason = f"low composite ({composite:.2f})"

            scored.append({
                "question_id": r.question_id,
                "question": r.question[:80],
                "composite": round(composite, 3),
                "reason": reason,
            })

        scored.sort(key=lambda x: x["composite"])
        return scored[:n]


# ── Helpers ───────────────────────────────────────────────────────────────────

_RELEVANCE_JUDGE_PROMPT = """\
You are a relevance judge for a retrieval system.

Question: {question}

Chunk:
\"\"\"{chunk}\"\"\"

Is this chunk relevant to answering the question?
Reply with exactly one word: YES or NO."""


def _judge_chunk_relevance(question: str, chunk_text: str) -> bool:
    """Use the local Ollama model to decide if a chunk is relevant."""
    prompt = _RELEVANCE_JUDGE_PROMPT.format(question=question, chunk=chunk_text.strip())
    try:
        resp = ollama.chat(
            model=_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return resp["message"]["content"].strip().upper().startswith("YES")
    except Exception:
        return False


_GEN_PROMPT = """\
You are a precise assistant. Answer the question using ONLY the context below.
If the context does not contain enough information, say so explicitly — do not guess.

Context:
{context}

Question: {question}

Answer:"""


def _generate_answer(question: str, chunks: list[str], model: str) -> str:
    context = "\n\n---\n\n".join(c.strip() for c in chunks if c.strip())
    prompt  = _GEN_PROMPT.format(context=context, question=question)
    resp = _client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


# ── Core runner ───────────────────────────────────────────────────────────────

def run_eval(
    config: EvalConfig,
    questions: list[dict],
    judgment_cache: Optional[dict] = None,
    verbose: bool = True,
) -> EvalReport:
    """
    Run the full eval loop for all questions.

    judgment_cache can be passed in to share relevance judgments with another
    eval run (avoids re-judging the same chunk twice).
    """
    if judgment_cache is None:
        judgment_cache = {}

    question_results: list[QuestionResult] = []

    for q in questions:
        qid      = q["id"]
        question = q["question"]

        if verbose:
            print(f"\n  Q{qid}: {question}")

        # ── 1. Retrieve ───────────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = config.retriever_fn(question, config.k)
        retrieval_time = time.perf_counter() - t0

        retrieved_ids   = [c["id"]   for c in chunks]
        retrieved_texts = [c["text"] for c in chunks]

        # ── 2. Label relevant chunks via local LLM judge ──────────────────────
        relevant_ids: set[str] = set()
        for chunk in chunks:
            key = (question, chunk["text"])
            if key not in judgment_cache:
                judgment_cache[key] = _judge_chunk_relevance(question, chunk["text"])
            if judgment_cache[key]:
                relevant_ids.add(chunk["id"])
                if verbose:
                    print(f"    ✅ [{chunk.get('source', '')}] {chunk['text'][:60]}…")
            else:
                if verbose:
                    print(f"    ❌ [{chunk.get('source', '')}] {chunk['text'][:60]}…")

        # ── 3. Generate answer ────────────────────────────────────────────────
        t1 = time.perf_counter()
        answer = _generate_answer(question, retrieved_texts, config.generator_model)
        generation_time = time.perf_counter() - t1

        if verbose:
            print(f"    Answer: {answer[:120]}…")

        # ── 4. Faithfulness ───────────────────────────────────────────────────
        faith = evaluate_faithfulness(question, retrieved_texts, answer)
        if verbose and faith["count"] > 0:
            print(f"    ⚠️  {faith['count']} unsupported claim(s): {faith['unsupported'][:1]}")

        # ── 5. Relevance ──────────────────────────────────────────────────────
        rel = evaluate_relevance(question, answer)
        if verbose:
            print(f"    Relevance: {rel['score']}/5 — {rel['explanation'][:80]}")

        question_results.append(QuestionResult(
            question_id=qid,
            question=question,
            retrieved_texts=retrieved_texts,
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            answer=answer,
            faithfulness=faith,
            relevance=rel,
            retrieval_time_s=round(retrieval_time, 3),
            generation_time_s=round(generation_time, 3),
        ))

    return EvalReport(config=config, question_results=question_results)


# ── Reporting ─────────────────────────────────────────────────────────────────

_SEP = "─" * 45

def print_report(report: EvalReport) -> None:
    rm     = report.retrieval_metrics()
    k      = report.config.k
    faith  = report.avg_faithfulness()
    uncl   = report.avg_unsupported_claims()
    rel    = report.avg_relevance()
    lat    = report.avg_latency()
    worst  = report.worst_performers(n=3)

    config_line = " | ".join(filter(None, [
        report.config.chunking_desc,
        report.config.embed_model,
        report.config.generator_model,
    ])) or report.config.name

    print(f"\nRAG Eval Report — {report.generated_at}")
    print(f"Configuration: {config_line}")
    print(_SEP)
    print(f"Retrieval Precision@{k}:  {rm.get(f'precision@{k}', 0):.2f}")
    print(f"Retrieval Recall@{k}:     {rm.get(f'recall@{k}', 0):.2f}")
    print(f"Mean Reciprocal Rank:   {rm.get('mrr', 0):.2f}")
    print(f"Faithfulness:           {faith:.2f} (avg unsupported claims: {uncl}/answer)")
    print(f"Answer Relevance:       {rel:.1f}/5.0")
    print(f"End-to-end Latency:     {lat:.1f}s avg")
    print(_SEP)

    if worst:
        parts = [f"Q{w['question_id']} ({w['reason']})" for w in worst]
        print(f"Worst performers: {', '.join(parts)}")
    print()


def save_report(report: EvalReport, path: str) -> None:
    k  = report.config.k
    rm = report.retrieval_metrics()

    output = {
        "generated_at": report.generated_at,
        "config": {
            "name":            report.config.name,
            "chunking_desc":   report.config.chunking_desc,
            "embed_model":     report.config.embed_model,
            "generator_model": report.config.generator_model,
            "k":               k,
        },
        "summary": {
            f"precision@{k}": rm.get(f"precision@{k}"),
            f"recall@{k}":    rm.get(f"recall@{k}"),
            "mrr":            rm.get("mrr"),
            "faithfulness":   report.avg_faithfulness(),
            "avg_unsupported_claims": report.avg_unsupported_claims(),
            "answer_relevance": report.avg_relevance(),
            "avg_latency_s":  report.avg_latency(),
        },
        "worst_performers": report.worst_performers(n=5),
        "per_question": [
            {
                "id":              r.question_id,
                "question":        r.question,
                f"precision@{k}":  precision_at_k(r.retrieved_ids, r.relevant_ids, k),
                f"recall@{k}":     recall_at_k(r.retrieved_ids, r.relevant_ids, k),
                "reciprocal_rank": reciprocal_rank(r.retrieved_ids, r.relevant_ids),
                "faithfulness":    r.faithfulness,
                "relevance":       r.relevance,
                "retrieval_time_s":   r.retrieval_time_s,
                "generation_time_s":  r.generation_time_s,
                "total_time_s":       r.total_time_s,
                "answer":          r.answer,
            }
            for r in report.question_results
        ],
    }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Report saved to {path}")
