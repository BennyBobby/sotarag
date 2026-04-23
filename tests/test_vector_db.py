from unittest.mock import patch, MagicMock

import pytest

import src.engine.vector_db as vdb


def _mock_client():
    return MagicMock()


def test_init_collection_creates_if_missing():
    mock = _mock_client()
    mock.get_collections.return_value.collections = []

    with patch.object(vdb, "client", mock):
        vdb.init_collection("test_col")

    mock.create_collection.assert_called_once()


def test_init_collection_skips_if_exists():
    existing = MagicMock()
    existing.name = "sota_papers"
    mock = _mock_client()
    mock.get_collections.return_value.collections = [existing]

    with patch.object(vdb, "client", mock):
        vdb.init_collection("sota_papers")

    mock.create_collection.assert_not_called()


def test_paper_exists_returns_true_when_found():
    point = MagicMock()
    mock = _mock_client()
    mock.scroll.return_value = ([point], None)

    with patch.object(vdb, "client", mock):
        result = vdb.paper_exists("sota_papers", "http://arxiv.org/pdf/1234.pdf")

    assert result is True


def test_paper_exists_returns_false_when_not_found():
    mock = _mock_client()
    mock.scroll.return_value = ([], None)

    with patch.object(vdb, "client", mock):
        result = vdb.paper_exists("sota_papers", "http://arxiv.org/pdf/9999.pdf")

    assert result is False


def test_upsert_creates_correct_number_of_points():
    mock = _mock_client()
    chunks = ["chunk A", "chunk B", "chunk C"]
    vectors = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    metadata = {"title": "Test Paper", "pdf_url": "http://x.com/p.pdf", "authors": [], "published": "", "abstract": ""}

    with patch.object(vdb, "client", mock):
        vdb.upsert_to_qdrant("sota_papers", chunks, vectors, metadata)

    call_kwargs = mock.upsert.call_args.kwargs
    assert len(call_kwargs["points"]) == 3


def test_search_returns_formatted_results():
    hit = MagicMock()
    hit.payload = {"text": "some text", "source": "Paper X", "url": "http://x.com", "authors": [], "published": "2024"}
    hit.score = 0.95
    mock = _mock_client()
    mock.query_points.return_value.points = [hit]

    with patch.object(vdb, "client", mock):
        results = vdb.search_in_qdrant("sota_papers", [0.1] * 1024, limit=1)

    assert len(results) == 1
    assert results[0]["text"] == "some text"
    assert results[0]["score"] == 0.95
