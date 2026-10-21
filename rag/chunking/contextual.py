from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="unused")


def gen_doc_summary(doc_title: str) -> str:
    prompt = f"""Here is the title of a document: {doc_title}
    Infer what the content could be about and generate a 1-2 sentence description."""

    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def add_context(chunk_text: str, doc_summary: str) -> str:
    prompt = f"""Document summary: {doc_summary}
    Given the above document context, write a 1-2 sentence description
    of what the following chunk is about. Be specific.
    Chunk: {chunk_text}"""

    response = client.chat.completions.create(
        model="qwen2.5:14b",
        messages=[{"role": "user", "content": prompt}],
        timeout=60.0
    )
    context = response.choices[0].message.content
    return f"{context}\n\n{chunk_text}"


def contextual_chunks(chunks: list[dict]) -> list[dict]:
    # Generate one summary per source document
    summaries = {}
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        if source not in summaries:
            summaries[source] = gen_doc_summary(source)
            print(f"Summary for {source}: {summaries[source]}")

    # Add context to each chunk
    enriched = []
    for chunk in chunks:
        source = chunk["metadata"]["source"]
        contextualized_text = add_context(chunk["text"], summaries[source])
        enriched.append({
            "text": contextualized_text,
            "metadata": {
                **chunk["metadata"],
                "has_context": True
            }
        })

    return enriched
