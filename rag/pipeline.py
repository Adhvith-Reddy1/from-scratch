import os
from ingestion.pdf_loader import ingest_pdf
from ingestion.markdown_loader import ingest_md
from ingestion.web_loader import ingest_webpage
from chunking.fixed import fixed_chunks
from chunking.recursive import recursive_chunks
from chunking.contextual import contextual_chunks
from retrieval.vector_store import VectorStore


def index_file(filepath: str, store: VectorStore, chunking_method: str = "recursive"):
    # 1. Detect format and load
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        raw = ingest_pdf(filepath)
    elif ext == ".md":
        raw = ingest_md(filepath)
    elif filepath.startswith("http"):
        raw = ingest_webpage(filepath)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # 2. Chunk based on chosen method
    if chunking_method == "fixed":
        chunks = fixed_chunks(raw)
    elif chunking_method == "recursive":
        chunks = recursive_chunks(raw)
    elif chunking_method == "contextual":
        chunks = recursive_chunks(raw)        # chunk first
        chunks = contextual_chunks(chunks)    # then enrich
    else:
        raise ValueError(f"Unknown chunking method: {chunking_method}")

    # 3. Embed and store
    store.add_documents(chunks)
    print(f"Indexed {filepath}: {len(chunks)} chunks via {chunking_method}")
