def recursive_chunks(documents: list[dict], chunk_size: int = 500, overlap: int = 50, separators: list[str] | None = None) -> list[dict]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    all_chunks = []
    for doc in documents:
        chunks = _split_text(doc["text"], doc["metadata"], chunk_size, overlap, separators)
        all_chunks.extend(chunks)

    # Add chunk indices per source
    for i, chunk in enumerate(all_chunks):
        chunk["metadata"]["chunk_index"] = i

    return all_chunks


def _split_text(text: str, metadata: dict, chunk_size: int, overlap: int, separators: list[str]) -> list[dict]:
    # Base case: text already fits
    if len(text) <= chunk_size:
        return [{"text": text.strip(), "metadata": {**metadata}}] if text.strip() else []

    # Try each separator in order
    for i, sep in enumerate(separators):
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        chunks = []
        current = parts[0]

        for part in parts[1:]:
            piece = sep + part if sep.strip() else part

            if len(current) + len(piece) <= chunk_size:
                current += piece
            else:
                if current.strip():
                    chunks.append(current.strip())
                # Start new chunk with overlap from end of previous
                if overlap > 0 and chunks:
                    current = chunks[-1][-overlap:] + piece
                else:
                    current = piece

        if current.strip():
            chunks.append(current.strip())

        # Check if any chunk is still too large — recurse with next separator
        final = []
        for chunk in chunks:
            if len(chunk) > chunk_size and i + 1 < len(separators):
                final.extend(_split_text(chunk, metadata, chunk_size, overlap, separators[i + 1:]))
            else:
                final.append({"text": chunk, "metadata": {**metadata}})

        return final

    # All separators exhausted — hard split by characters
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text.strip():
            chunks.append({"text": chunk_text.strip(), "metadata": {**metadata}})
    return chunks
