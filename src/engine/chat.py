import requests
from src.engine.embedding import get_embeddings
from src.engine.vector_db import search_in_qdrant


def ask_sotarag(question: str):
    """
    RAG Logic: Search context in Qdrant and generate answer with Ollama.
    """

    query_vector = get_embeddings([question])[0]
    context_chunks = search_in_qdrant("sota_papers", query_vector, limit=3)
    formatted_context = "\n\n".join(
        [
            f"Source {c['source']} (Score: {c['score']:.2f}):\n{c['text']}"
            for c in context_chunks
        ]
    )

    prompt = f"""
    You are SotaRAG, a specialized scientific assistant. 
    Use the following pieces of retrieved context to answer the user's question.
    If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

    CONTEXT:
    {formatted_context}

    USER QUESTION: 
    {question}

    SCIENTIFIC ANSWER:
    """

    try:
        response = requests.post(
            "http://ollama:11434/api/generate",
            json={"model": "llama3.1", "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json().get("response")
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"


if __name__ == "__main__":
    user_query = "What are the advantages of the Transformer architecture?"
    answer = ask_sotarag(user_query)
    print(answer)
