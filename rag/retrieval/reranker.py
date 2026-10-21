import re
import json
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

def rerank(query: str, chunks: dict, k: int = 5) -> dict:
    docs = chunks["documents"][0]
    unranked_chunks = [{"id": i, "text": doc} for i, doc in enumerate(docs)]

    prompt = f"""Given the query {query}, rank these passages from most
    to least relevant. Return the ids of the top {k} most relevant chunks
    as a JSON array of integers only, no other text. Here are the chunks: \n{unranked_chunks}"""

    rankings_str = call_llm(prompt)

    match = re.search(r'\[[\d,\s]+\]', rankings_str)
    if match:
        rankings = json.loads(match.group())
    else:
        rankings = list(range(min(k, len(docs))))

    rankings = [r for r in rankings if 0 <= r < len(docs)][:k]

    return {
        "ids": [[chunks["ids"][0][i] for i in rankings]],
        "documents": [[chunks["documents"][0][i] for i in rankings]],
        "metadatas": [[chunks["metadatas"][0][i] for i in rankings]],
        "distances": [[chunks["distances"][0][i] for i in rankings]],
    }
