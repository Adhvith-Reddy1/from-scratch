# from-scratch

A collection of projects built from scratch to understand how they actually work —
no orchestration frameworks (no LangChain, LlamaIndex, or agent SDKs). The chunking,
retrieval, agent loop, memory, and evaluation code here is all hand-written.

## Projects

| Folder | What it is | Built from scratch |
| --- | --- | --- |
| [`rag/`](rag/) | A retrieval-augmented generation pipeline | document loaders, three chunking strategies, a Chroma-backed vector store, multi-query + LLM reranking retrieval, and a retrieval/faithfulness/relevance eval harness |
| [`agent/`](agent/) | A tool-using research agent with a CLI and a web UI | the agent loop, pluggable model backends, session + persistent memory, and citation grounding verification |

The agent reuses the RAG pipeline directly (`agent/` imports from `../rag`), so the two
live in one repo.

## Requirements

- **Python 3.12**
- **[Ollama](https://ollama.com)** running locally for embeddings and local models:
  ```bash
  ollama pull nomic-embed-text   # embeddings
  ollama pull llama3.1:8b        # local generation / judge
  ollama pull qwen2.5:14b        # contextual chunking (optional)
  ```
- API keys (only where used): `ANTHROPIC_API_KEY` for Claude judges/backends,
  `TAVILY_API_KEY` for the agent's web search.

Each project has its own `requirements.txt`; install the one you need in a virtualenv.

---

## `rag/` — RAG pipeline

```bash
cd rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Ingest documents** into a Chroma collection (see `pipeline.py`):

```python
from retrieval.vector_store import VectorStore
from pipeline import index_file

store = VectorStore(database="recursive_chunk_collection", path="./chroma_db")
index_file("documents/clownfish.md", store, chunking_method="recursive")
```

**Run the evaluation harness** (`rag_eval_full.py`) over the indexed collections —
measures precision@k, recall@k, MRR, faithfulness, and answer relevance:

```bash
python rag_eval_full.py                 # single strategy (default: recursive)
python rag_eval_full.py --compare       # compare all retrieval strategies
python rag_eval_full.py --output results.json
```

Sample Wikipedia-derived documents (marine biology, drug discovery, finance,
science fiction) are in `documents/`, with evaluation questions in `eval/questions.json`.

---

## `agent/` — research agent (CLI + web UI)

```bash
cd agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in your API keys
```

The agent searches the local knowledge base first, falls back to web search, cites every
claim, and then verifies those citations against the retrieved sources.

**CLI:**

```bash
python cli.py "What is an alpheid shrimp?" --backend anthropic
# --backend anthropic | openai | ollama
# --memory-strategy full | sliding | summarize
```

**Web UI** (chat + drag-and-drop document ingestion):

```bash
uvicorn api:app --reload      # then open http://localhost:8000
```

**Agent evals** (`evals.json`, with an LLM-as-judge that fact-checks via web search):

```bash
python evals.py --backend anthropic
```

### How it fits together

- **Backends** (`backends/`) — one `stream()` interface over Anthropic, OpenAI, and Ollama.
- **Memory** — session strategies (`session_memory/`: sliding window, summarization) to
  keep context bounded, plus persistent memory (`persistent_memory/`: SQLite facts +
  embedding-retrieved episodes) that carries across conversations.
- **Grounding** (`grounding.py`) — extracts each `[Source: ...]` citation and uses an
  LLM judge to score whether the cited passage actually supports the claim.
