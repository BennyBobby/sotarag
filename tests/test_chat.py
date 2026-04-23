from unittest.mock import patch, MagicMock

from src.engine.chat import ask_sotarag, stream_sotarag


def _mock_context():
    return [
        {"text": "Transformers use attention.", "source": "Attention Paper", "url": "http://arxiv.org/1706.03762", "score": 0.95},
    ]


def test_ask_sotarag_returns_answer_and_sources():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"response": "Transformers are great."}

    with (
        patch("src.engine.chat.get_embeddings", return_value=[[0.1] * 1024]),
        patch("src.engine.chat.search_in_qdrant", return_value=_mock_context()),
        patch("src.engine.chat.requests.post", return_value=mock_resp),
    ):
        result = ask_sotarag("What are transformers?")

    assert "answer" in result
    assert result["answer"] == "Transformers are great."
    assert isinstance(result["sources"], list)
    assert result["sources"][0]["title"] == "Attention Paper"


def test_ask_sotarag_handles_ollama_error():
    with (
        patch("src.engine.chat.get_embeddings", return_value=[[0.1] * 1024]),
        patch("src.engine.chat.search_in_qdrant", return_value=_mock_context()),
        patch("src.engine.chat.requests.post", side_effect=Exception("connection refused")),
    ):
        result = ask_sotarag("What are transformers?")

    assert "Error connecting to Ollama" in result["answer"]


def test_stream_sotarag_yields_tokens():
    lines = [
        b'{"response": "Trans", "done": false}',
        b'{"response": "formers", "done": true}',
    ]
    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines.return_value = iter(lines)

    with (
        patch("src.engine.chat.get_embeddings", return_value=[[0.1] * 1024]),
        patch("src.engine.chat.search_in_qdrant", return_value=_mock_context()),
        patch("src.engine.chat.requests.post", return_value=mock_resp),
    ):
        tokens = list(stream_sotarag("What are transformers?"))

    text = "".join(tokens)
    assert "Trans" in text
    assert "formers" in text
    assert "Sources" in text  # sources markdown appended at end


def test_stream_sotarag_yields_error_on_exception():
    with (
        patch("src.engine.chat.get_embeddings", return_value=[[0.1] * 1024]),
        patch("src.engine.chat.search_in_qdrant", return_value=_mock_context()),
        patch("src.engine.chat.requests.post", side_effect=Exception("timeout")),
    ):
        tokens = list(stream_sotarag("What?"))

    assert any("Error" in t for t in tokens)
