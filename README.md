# SotaRAG

A RAG (Retrieval-Augmented Generation) system for scientific papers. Search ArXiv papers by topic, index them automatically, and chat with an AI that answers based on their content.

## Architecture

```
ArXiv API → PDF Download → Text Chunks → Embeddings → Qdrant
                                                          ↓
                  User Question → FastAPI → Qdrant Search → Ollama (LLM) → Streamed Answer
```

**Services:**
| Service | Role | Port |
|---|---|---|
| Streamlit | Web UI | 8501 |
| FastAPI | REST API | 8000 |
| Taskiq Worker | Async ingestion worker | — |
| Qdrant | Vector database | 6333 |
| Redis | Task queue + chat history | 6379 |
| Ollama | Embeddings + LLM inference | 11434 |

**Models used:**
- Embeddings: `mxbai-embed-large` (1024 dimensions)
- LLM: `llama3.1`

---

## Quick Start (Docker)

```bash
docker compose up --build
```

Then open [http://localhost:8501](http://localhost:8501).

> **Note:** First startup pulls Ollama models (~2 GB). This takes a few minutes.

---

## Local Development (without Docker)

### 1. Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Qdrant, Redis, and Ollama running locally

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if your services run on non-default ports
```

### 4. Pull Ollama models

```bash
ollama pull mxbai-embed-large
ollama pull llama3.1
```

### 5. Run each service in a separate terminal

```bash
# FastAPI
uv run uvicorn src.api.main:app --reload --port 8000

# Taskiq worker
uv run taskiq worker src.tasks:broker --workers 2

# Streamlit UI
uv run streamlit run src/ui/app.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (Qdrant, Ollama, Redis) |
| `GET` | `/papers` | List indexed papers |
| `POST` | `/papers/search-and-ingest` | Search ArXiv and queue ingestion |
| `GET` | `/tasks/{task_id}` | Check ingestion task status |
| `POST` | `/chat` | Ask a question (non-streaming) |
| `POST` | `/chat/stream` | Ask a question (streaming) |
| `GET` | `/chat/history` | Get full conversation history |
| `DELETE` | `/chat/history` | Clear conversation history |

Interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

### Example: search and ingest

```bash
curl -X POST http://localhost:8000/papers/search-and-ingest \
  -H "Content-Type: application/json" \
  -d '{"topic": "vision transformers", "max_results": 3}'
```

### Example: chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is self-attention?"}'
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama service URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant service URL |
| `REDIS_URL` | `redis://localhost:6379` | Redis service URL |
| `API_URL` | `http://localhost:8000` | FastAPI URL (used by Streamlit) |
| `COLLECTION_NAME` | `sota_papers` | Qdrant collection name |
| `EMBEDDING_MODEL` | `mxbai-embed-large` | Ollama embedding model |
| `LLM_MODEL` | `llama3.1` | Ollama generation model |

---

## Project Structure

```
src/
├── api/
│   └── main.py          # FastAPI REST API
├── crawler/
│   └── arxiv_client.py  # ArXiv search
├── engine/
│   ├── chat.py          # RAG logic (streaming + non-streaming)
│   ├── embedding.py     # Ollama embeddings
│   ├── ingest.py        # Ingestion pipeline orchestrator
│   ├── processor.py     # PDF download + text splitting
│   └── vector_db.py     # Qdrant CRUD
├── ui/
│   └── app.py           # Streamlit interface
├── config.py            # Centralized configuration
├── logger.py            # Shared logging configuration
└── tasks.py             # Taskiq async tasks
tests/
├── test_api.py          # API endpoint tests
├── test_chat.py         # RAG logic tests
├── test_embedding.py    # Embedding tests
├── test_ingest.py       # Ingestion pipeline tests
└── test_processor.py    # PDF processing tests
```
