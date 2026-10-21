def fixed_chunks(documents: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks = []
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": i // (chunk_size - overlap)
                    }
                })

    return chunks
