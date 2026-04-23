from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    """TestClient with broker and redis mocked out."""
    with (
        patch("src.api.main.broker") as mock_broker,
        patch("src.api.main.redis_client") as mock_redis,
    ):
        mock_broker.startup = AsyncMock()
        mock_broker.shutdown = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.ping.return_value = True

        from src.api.main import app
        with TestClient(app) as c:
            yield c


def test_health_all_ok(client):
    with (
        patch("src.api.main.requests.get") as mock_get,
        patch("src.api.main.redis_client") as mock_redis,
    ):
        ok_resp = MagicMock()
        ok_resp.ok = True
        mock_get.return_value = ok_resp
        mock_redis.ping.return_value = True

        r = client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "services" in body


def test_health_degraded_when_service_down(client):
    with (
        patch("src.api.main.requests.get", side_effect=Exception("unreachable")),
        patch("src.api.main.redis_client") as mock_redis,
    ):
        mock_redis.ping.side_effect = Exception("no redis")

        r = client.get("/health")

    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["services"]["qdrant"] == "unreachable"
    assert body["services"]["ollama"] == "unreachable"
    assert body["services"]["redis"] == "unreachable"


def test_list_papers(client):
    with patch("src.api.main.get_indexed_papers", return_value={"Paper A": "http://x.com/a.pdf"}):
        r = client.get("/papers")

    assert r.status_code == 200
    assert r.json() == {"Paper A": "http://x.com/a.pdf"}


def test_search_and_ingest_no_results(client):
    with patch("src.api.main.search_arxiv", return_value=[]):
        r = client.post("/papers/search-and-ingest", json={"topic": "unknown topic xyz"})

    assert r.status_code == 404


def test_chat_returns_answer(client):
    with (
        patch("src.api.main.ask_sotarag", return_value={"answer": "42", "sources": []}),
        patch("src.api.main.redis_client") as mock_redis,
    ):
        mock_redis.get.return_value = None

        r = client.post("/chat", json={"question": "What is life?"})

    assert r.status_code == 200
    assert r.json()["answer"] == "42"


def test_get_history_empty(client):
    with patch("src.api.main.redis_client") as mock_redis:
        mock_redis.get.return_value = None
        r = client.get("/chat/history")

    assert r.status_code == 200
    assert r.json() == []


def test_clear_history(client):
    with patch("src.api.main.redis_client") as mock_redis:
        r = client.delete("/chat/history")

    assert r.status_code == 200
    assert r.json()["status"] == "cleared"
