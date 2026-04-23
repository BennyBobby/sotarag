import json
import requests
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import redis as sync_redis

from src.config import REDIS_URL, QDRANT_URL, OLLAMA_HOST, COLLECTION_NAME
from src.tasks import broker, ingest_paper_task
from src.crawler.arxiv_client import search_arxiv
from src.engine.chat import ask_sotarag, stream_sotarag
from src.engine.vector_db import get_indexed_papers, delete_paper

HISTORY_KEY = "sotarag:chat_history"
redis_client = sync_redis.from_url(REDIS_URL, decode_responses=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await broker.startup()
    yield
    await broker.shutdown()


app = FastAPI(title="SotaRAG API", version="0.1.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    topic: str
    max_results: int = 5


class ChatRequest(BaseModel):
    question: str


def _load_history() -> list:
    raw = redis_client.get(HISTORY_KEY)
    return json.loads(raw) if raw else []


def _save_to_history(question: str, answer: str):
    history = _load_history()
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    redis_client.set(HISTORY_KEY, json.dumps(history))


@app.get("/health")
def health():
    services = {}

    try:
        r = requests.get(f"{QDRANT_URL}/healthz", timeout=3)
        services["qdrant"] = "ok" if r.ok else "error"
    except Exception:
        services["qdrant"] = "unreachable"

    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        services["ollama"] = "ok" if r.ok else "error"
    except Exception:
        services["ollama"] = "unreachable"

    try:
        redis_client.ping()
        services["redis"] = "ok"
    except Exception:
        services["redis"] = "unreachable"

    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"
    return {"status": overall, "services": services}


@app.get("/papers")
def list_papers():
    return get_indexed_papers()


@app.post("/papers/search-and-ingest")
async def search_and_ingest(req: SearchRequest):
    articles = search_arxiv(req.topic, req.max_results)
    if not articles:
        raise HTTPException(status_code=404, detail="No articles found for this topic.")

    tasks = []
    for art in articles:
        task = await ingest_paper_task.kiq(
            art["pdf_url"],
            art["title"],
            art.get("authors", []),
            art.get("published", ""),
            art.get("summary", ""),
        )
        tasks.append({"title": art["title"], "task_id": task.task_id})

    return {"tasks": tasks, "count": len(tasks)}


@app.delete("/papers")
def remove_paper(pdf_url: str):
    papers = get_indexed_papers()
    if pdf_url not in papers.values():
        raise HTTPException(status_code=404, detail="Paper not found.")
    delete_paper(COLLECTION_NAME, pdf_url)
    return {"status": "deleted", "pdf_url": pdf_url}


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    result = await broker.result_backend.get_result(task_id)
    if result is None:
        return {"status": "pending"}
    return {"status": "done", "result": result.return_value}


@app.post("/chat")
def chat(req: ChatRequest):
    result = ask_sotarag(req.question)
    _save_to_history(req.question, result["answer"])
    return result


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def generate():
        full = ""
        for token in stream_sotarag(req.question):
            full += token
            yield token
        _save_to_history(req.question, full)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/chat/history")
def get_history():
    return _load_history()


@app.delete("/chat/history")
def clear_history():
    redis_client.delete(HISTORY_KEY)
    return {"status": "cleared"}
