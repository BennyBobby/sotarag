from unittest.mock import patch, MagicMock

from src.engine.embedding import get_embeddings


def _mock_ollama_response(vector: list):
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"embeddings": [vector]}
    return mock_resp


def test_returns_one_embedding_per_chunk():
    vector = [0.1] * 1024
    mock_resp = _mock_ollama_response(vector)

    with patch("src.engine.embedding.requests.post", return_value=mock_resp):
        result = get_embeddings(["chunk one", "chunk two"])

    assert len(result) == 2
    assert result[0] == vector


def test_skips_chunk_on_error():
    good_vector = [0.2] * 1024
    good_resp = _mock_ollama_response(good_vector)
    bad_resp = MagicMock()
    bad_resp.raise_for_status.side_effect = Exception("connection refused")

    with patch("src.engine.embedding.requests.post", side_effect=[bad_resp, good_resp]):
        result = get_embeddings(["bad chunk", "good chunk"])

    assert len(result) == 1
    assert result[0] == good_vector


def test_returns_empty_list_for_no_chunks():
    with patch("src.engine.embedding.requests.post") as mock_post:
        result = get_embeddings([])

    mock_post.assert_not_called()
    assert result == []
