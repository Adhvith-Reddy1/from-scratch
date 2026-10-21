"""
Full RAG evaluation — retrieval + faithfulness + answer relevance.

Run a single config:
    python rag_eval_full.py

Run all five retrieval strategies and compare:
    python rag_eval_full.py --compare

Save results:
    python rag_eval_full.py --output results_full.json
"""

import os
import sys
import json
import argparse
import httpx
import chromadb

sys.path.insert(0, os.path.dirname(__file__))

from retrieval.vector_store import VectorStore
from retrieval.multi_query  import multi_query_retrieve
from retrieval.reranker      import rerank
from eval.runner import EvalConfig, run_eval, print_report, save_report

CHROMA_PATH  = os.path.join(os.path.dirname(__file__), "chroma_db")
EMBED_MODEL  = "nomic-embed-text"
K            = 5

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "eval", "questions.json")
with open(QUESTIONS_PATH) as f:
    QUESTIONS = json.load(f)


# ── Embedding helper (Ollama) ─────────────────────────────────────────────────

def ollama_embed(text: str) -> list[float]:
    resp = httpx.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60.0,
    )
    return resp.json()["embedding"]


# ── Retriever factories ───────────────────────────────────────────────────────
# Each factory returns a (query: str, k: int) -> list[dict] callable.
# Each dict must have "id", "text", and optionally "source".

def _chroma_to_chunks(results: dict) -> list[dict]:
    return [
        {"id": cid, "text": doc, "source": meta.get("source", "")}
        for cid, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        )
    ]


def make_single_retriever(collection) -> callable:
    def retrieve(query: str, k: int = K) -> list[dict]:
        results = collection.query(
            query_embeddings=[ollama_embed(query)],
            n_results=k,
        )
        return _chroma_to_chunks(results)
    return retrieve


def make_multi_query_retriever(store: VectorStore) -> callable:
    def retrieve(query: str, k: int = K) -> list[dict]:
        results = multi_query_retrieve(query, store, k=k)
        return _chroma_to_chunks(results)
    return retrieve


def make_rerank_retriever(store: VectorStore) -> callable:
    def retrieve(query: str, k: int = K) -> list[dict]:
        candidates = multi_query_retrieve(query, store, k=k * 3)
        results    = rerank(query, candidates, k=k)
        return _chroma_to_chunks(results)
    return retrieve


# ── Build all configs ─────────────────────────────────────────────────────────

def build_configs(chroma_client, recursive_store: VectorStore) -> list[EvalConfig]:
    collections = {
        "contextual": "test_collection",
        "fixed":      "fixed_chunk_collection",
        "recursive":  "recursive_chunk_collection",
    }

    configs = []
    for label, coll_name in collections.items():
        coll = chroma_client.get_or_create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine"},
        )
        configs.append(EvalConfig(
            name=label,
            retriever_fn=make_single_retriever(coll),
            k=K,
            chunking_desc=label,
            embed_model=EMBED_MODEL,
        ))

    configs.append(EvalConfig(
        name="multi_query",
        retriever_fn=make_multi_query_retriever(recursive_store),
        k=K,
        chunking_desc="recursive + multi-query expansion",
        embed_model=EMBED_MODEL,
    ))
    configs.append(EvalConfig(
        name="rerank",
        retriever_fn=make_rerank_retriever(recursive_store),
        k=K,
        chunking_desc="recursive + multi-query + LLM rerank",
        embed_model=EMBED_MODEL,
    ))
    return configs


# ── Comparison table ──────────────────────────────────────────────────────────

def print_comparison(reports: list) -> None:
    col_w = 12
    names = [r.config.name for r in reports]
    header = f"{'Metric':<26}" + "".join(f"{n:>{col_w}}" for n in names)

    print("\n\n══ FULL EVAL COMPARISON ══")
    print("─" * len(header))
    print(header)
    print("─" * len(header))

    k = reports[0].config.k

    rows = [
        (f"Precision@{k}", lambda r: r.retrieval_metrics()[f"precision@{k}"]),
        (f"Recall@{k}",    lambda r: r.retrieval_metrics()[f"recall@{k}"]),
        ("MRR",            lambda r: r.retrieval_metrics()["mrr"]),
        ("Faithfulness",   lambda r: r.avg_faithfulness()),
        ("Answer Relevance /5", lambda r: r.avg_relevance()),
        ("Latency (s)",    lambda r: r.avg_latency()),
    ]

    for label, fn in rows:
        values = {r.config.name: fn(r) for r in reports}
        best   = max(values, key=values.get)
        row    = f"{label:<26}"
        for name, val in values.items():
            cell = f"{val:.2f}" + ("◀" if name == best else " ")
            row += f"{cell:>{col_w}}"
        print(row)

    print("─" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare",  action="store_true", help="Evaluate all 5 retrieval strategies")
    parser.add_argument("--config",   default="recursive",
                        choices=["contextual", "fixed", "recursive", "multi_query", "rerank"],
                        help="Single config to run (ignored if --compare)")
    parser.add_argument("--output",   default=None, help="Path to save JSON report")
    parser.add_argument("--quiet",    action="store_true", help="Suppress per-question verbose output")
    args = parser.parse_args()

    chroma_client  = chromadb.PersistentClient(path=CHROMA_PATH)
    recursive_store = VectorStore(database="recursive_chunk_collection", path=CHROMA_PATH)
    all_configs     = build_configs(chroma_client, recursive_store)
    judgment_cache  = {}

    if args.compare:
        reports = []
        for config in all_configs:
            print(f"\n{'='*55}")
            print(f"  Evaluating: {config.name}")
            print(f"{'='*55}")
            report = run_eval(config, QUESTIONS, judgment_cache=judgment_cache, verbose=not args.quiet)
            print_report(report)
            reports.append(report)

        print_comparison(reports)

        if args.output:
            combined = {r.config.name: json.loads(json.dumps({
                "summary": {
                    f"precision@{config.k}": r.retrieval_metrics()[f"precision@{config.k}"],
                    f"recall@{config.k}":    r.retrieval_metrics()[f"recall@{config.k}"],
                    "mrr":                   r.retrieval_metrics()["mrr"],
                    "faithfulness":          r.avg_faithfulness(),
                    "answer_relevance":      r.avg_relevance(),
                    "avg_latency_s":         r.avg_latency(),
                }
            })) for r in reports}
            with open(args.output, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"\nComparison saved to {args.output}")
    else:
        config = next(c for c in all_configs if c.name == args.config)
        print(f"\nRunning eval for: {config.name}")
        report = run_eval(config, QUESTIONS, judgment_cache=judgment_cache, verbose=not args.quiet)
        print_report(report)
        if args.output:
            save_report(report, args.output)


if __name__ == "__main__":
    main()
