import numpy as np
import chromadb
import httpx


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection.

    Embeddings are produced locally via Ollama (nomic-embed-text).
    """

    def __init__(self, database="documents", path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(database)

    def get_embedding(self, text: str) -> list[float]:
        response = httpx.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text},
            timeout=60.0,
        )
        return np.array(response.json()["embedding"])

    def add_documents(self, documents: list[dict]):
        for i, doc in enumerate(documents):
            embedding = self.get_embedding(doc["text"])
            self.collection.upsert(
                ids=[f"{doc['metadata']['source']}-chunk-{i}"],
                documents=[doc["text"]],
                embeddings=[embedding.tolist()],
                metadatas=[doc["metadata"]],
            )
        print(f"{len(documents)} chunks added")

    def query(self, query: str, k: int = 5) -> list[str]:
        embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
        )
        return self.create_citation(results)

    @staticmethod
    def create_citation(results: dict) -> list[str]:
        cited = []
        for chunk_id, doc, meta in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            citation = f"[Source: {meta['source']}, chunk: {chunk_id}]"
            cited.append(f"{doc}\n{citation}")
        return cited

    def delete_by_source(self, source: str) -> int:
        results = self.collection.get(where={"source": source})
        ids = results["ids"]
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def query_with_filter(self, query: str, filters: dict, k: int = 5) -> dict:
        embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            where=filters,
        )
        return results
