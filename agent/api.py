import asyncio
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anthropic import Anthropic
from tools import REGISTRY
from grounding import verify_grounding
from config import SYSTEM

CHROMA_PATH = Path(__file__).parent.parent / "rag" / "chroma_db"
DOCS_DIR = Path(__file__).parent / "uploaded_docs"

app = FastAPI()

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# Registry of indexed documents: {"filename": "...", "summary": "..."}
_indexed_docs: list[dict] = []
_summary_client = Anthropic()

# Conversation sessions: session_id -> chat_history
_sessions: dict[str, list] = {}


def _build_system_prompt() -> str:
    if not _indexed_docs:
        return SYSTEM
    doc_lines = "\n".join(f"- {d['filename']}: {d['summary']}" for d in _indexed_docs)
    return SYSTEM + f"\n\nThe following documents have been indexed in the local knowledge base — always search them first for relevant questions:\n{doc_lines}"


def _generate_summary(text_sample: str) -> str:
    response = _summary_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=60,
        messages=[{"role": "user", "content": f"In one sentence, describe what this document is about:\n\n{text_sample[:2000]}"}],
    )
    return response.content[0].text.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_backend(backend_name: str, model: str | None, system: str | None = None):
    if backend_name == "anthropic":
        from backends.anthropic import AnthropicBackend
        return AnthropicBackend(model or "claude-sonnet-4-6", system=system)
    elif backend_name == "openai":
        from backends.openai import OpenAIBackend
        return OpenAIBackend(model or "gpt-4o", system=system)
    else:
        from backends.openai import OpenAIBackend
        return OpenAIBackend(model or "llama3.1:8b", base_url="http://localhost:11434/v1", system=system)


def _run_agent(query: str, chat_history: list, backend, put: Callable) -> None:
    """Blocking agent loop — appends to chat_history in place, emits SSE events via put()."""
    chat_history.append({"role": "user", "content": query})

    def on_token(text: str):
        put({"type": "token", "content": text})

    try:
        result = backend.stream(chat_history, on_token=on_token)

        while result.stop_reason == "tool_use":
            put({"type": "tool", "name": result.tool_name, "input": result.tool_input})

            tool_fn = REGISTRY.get(result.tool_name)
            if tool_fn is None:
                tool_result = f"Unknown tool: {result.tool_name}"
            else:
                try:
                    tool_result = tool_fn(**result.tool_input)
                except Exception as e:
                    tool_result = f"Tool failed: {e}"

            chat_history.append(result.raw_content)
            chat_history.append(backend.tool_result_message(result.tool_id, tool_result))

            result = backend.stream(chat_history, on_token=on_token)

        report = verify_grounding(result.text, chat_history)
        put({"type": "grounding", "report": report})
        chat_history.append(result.raw_content)

    except Exception as e:
        put({"type": "error", "message": str(e)})

    put({"type": "done"})


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text()


class QueryRequest(BaseModel):
    query: str
    backend: str = "anthropic"
    model: str | None = None
    session_id: str | None = None


@app.post("/query")
async def query(req: QueryRequest):
    session_id = req.session_id or str(uuid.uuid4())
    chat_history = _sessions.setdefault(session_id, [])

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(event: dict):
        loop.call_soon_threadsafe(queue.put_nowait, event)

    backend = _make_backend(req.backend, req.model, system=_build_system_prompt())
    thread = threading.Thread(target=_run_agent, args=(req.query, chat_history, backend, put), daemon=True)
    thread.start()

    async def generate():
        yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id})}\n\n"
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] == "done":
                break

    return StreamingResponse(generate(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    DOCS_DIR.mkdir(exist_ok=True)
    dest = DOCS_DIR / file.filename
    dest.write_bytes(await file.read())

    ext = dest.suffix.lower()
    try:
        if ext == ".md":
            from rag.ingestion.markdown_loader import ingest_md
            raw = ingest_md(str(dest))
        elif ext == ".pdf":
            from rag.ingestion.pdf_loader import ingest_pdf
            raw = ingest_pdf(str(dest))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use .md or .pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from rag.chunking.recursive import recursive_chunks
    from rag.retrieval.vector_store import VectorStore

    chunks = recursive_chunks(raw)
    store = VectorStore(database="recursive_chunk_collection", path=str(CHROMA_PATH))
    store.add_documents(chunks)

    sample_text = " ".join(c["text"] for c in chunks[:3])
    summary = _generate_summary(sample_text)
    _indexed_docs.append({"filename": file.filename, "summary": summary})

    return {"filename": file.filename, "chunks_indexed": len(chunks), "summary": summary}


@app.get("/documents")
async def list_documents():
    return _indexed_docs


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    global _indexed_docs
    if not any(d["filename"] == filename for d in _indexed_docs):
        raise HTTPException(status_code=404, detail=f"{filename} not found in index")

    from rag.retrieval.vector_store import VectorStore
    store = VectorStore(database="recursive_chunk_collection", path=str(CHROMA_PATH))
    deleted = store.delete_by_source(filename)

    _indexed_docs = [d for d in _indexed_docs if d["filename"] != filename]
    return {"filename": filename, "chunks_deleted": deleted}
