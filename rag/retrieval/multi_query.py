from openai import OpenAI
from retrieval.vector_store import VectorStore

_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def call_llm(prompt: str, model: str = "llama3.1:8b") -> str:
    response = _client.chat.completions.create(
        model=model,
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def multi_query_retrieve(query: str, store: VectorStore, k: int = 5) -> list[dict]:
    prompt = f"""Generate 3 different search queries for this question.
    Return only the queries, one per line.
    Question: {query}"""

    variations = [v.strip() for v in call_llm(prompt).strip().split("\n") if v.strip()]
    seen_ids = set()
    ids, documents, metadatas, distances = [], [], [], []

    for var in variations:
        results = store.query(var, k=k)
        for i, doc_id in enumerate(results["ids"][0]):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                ids.append(results["ids"][0][i])
                documents.append(results["documents"][0][i])
                metadatas.append(results["metadatas"][0][i])
                distances.append(results["distances"][0][i])

    # sort by distance ascending (most similar first) and return top k
    combined = sorted(zip(distances, ids, documents, metadatas), key=lambda x: x[0])[:k]
    distances, ids, documents, metadatas = zip(*combined) if combined else ([], [], [], [])

    return {"ids": [list(ids)], "documents": [list(documents)], "metadatas": [list(metadatas)], "distances": [list(distances)]}
