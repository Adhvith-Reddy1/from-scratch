import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import httpx
from trafilatura import extract
from typing import Any, Callable
from config import tavily_client

from rag.retrieval.vector_store import VectorStore

TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": "ALWAYS call this first for questions about marine biology, drug discovery, science fiction, or basic finance. Searches the local document knowledge base and returns relevant passages with source citations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for answers to a query. Use this ONLY for current events or after search_documents returns no relevant results. Returns the title, url and content for each result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"],
        }
    },
    {
        "name": "read_webpage",
        "description": "Extract text from a webpage. Returns webpage text as a string.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"}
            },
            "required": ["url"],
        }
    },
]

def _extract_search_results(response: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"title": item["title"], "url": item["url"], "content": item["content"]}
        for item in response["results"]
    ]

def web_search(query: str) -> str:
    result = tavily_client.search(query=query, search_depth="advanced")
    return json.dumps(_extract_search_results(result))

def read_webpage(url: str) -> str:
    try:
        response = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return f"Error: page returned status {response.status_code}"
        extracted = extract(response.text)
        if extracted is None:
            return "Could not extract text from page"
        return extracted[:4000]
    except httpx.RequestError as e:
        return f"Network error: {e}"

_CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "rag", "chroma_db")

def search_documents(query: str, k: int = 3) -> list[str]:
    store = VectorStore(database="recursive_chunk_collection", path=_CHROMA_PATH)
    return store.query(query, k)


REGISTRY: dict[str, Callable[..., Any]] = {
    "web_search": web_search,
    "read_webpage": read_webpage,
    "search_documents": search_documents,
}
