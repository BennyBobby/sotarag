import json
import requests
from src.config import OLLAMA_HOST, LLM_MODEL, COLLECTION_NAME
from src.engine.embedding import get_embeddings
from src.engine.vector_db import search_in_qdrant


def _build_prompt(question: str, formatted_context: str) -> str:
    return f"""You are SotaRAG, a specialized scientific assistant.
Use the following pieces of retrieved context to answer the user's question.
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

CONTEXT:
{formatted_context}

USER QUESTION:
{question}

SCIENTIFIC ANSWER:
"""


def _get_context(question: str) -> tuple[list, str]:
    query_vector = get_embeddings([question])[0]
    chunks = search_in_qdrant(COLLECTION_NAME, query_vector, limit=3)
    formatted = "\n\n".join(
        [f"Source: {c['source']} (Score: {c['score']:.2f}):\n{c['text']}" for c in chunks]
    )
    return chunks, formatted


def ask_sotarag(question: str) -> dict:
    """Non-streaming RAG: returns answer + sources."""
    context_chunks, formatted_context = _get_context(question)
    prompt = _build_prompt(question, formatted_context)

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "")
    except Exception as e:
        answer = f"Error connecting to Ollama: {str(e)}"

    sources = [
        {"title": c["source"], "url": c.get("url", ""), "score": round(c["score"], 2)}
        for c in context_chunks
    ]
    return {"answer": answer, "sources": sources}


def stream_sotarag(question: str):
    """Streaming RAG: yields text tokens, then appends formatted sources."""
    context_chunks, formatted_context = _get_context(question)
    prompt = _build_prompt(question, formatted_context)

    try:
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
    except Exception as e:
        yield f"\n\nError connecting to Ollama: {str(e)}"

    # Append sources as formatted markdown at the end of the stream
    unique_sources = {c["source"]: c.get("url", "") for c in context_chunks}
    if unique_sources:
        sources_md = "\n\n---\n**Sources:**\n" + "\n".join(
            f"- [{title}]({url})" if url else f"- {title}"
            for title, url in unique_sources.items()
        )
        yield sources_md


if __name__ == "__main__":
    result = ask_sotarag("What are the advantages of the Transformer architecture?")
    print(result["answer"])
