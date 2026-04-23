from unittest.mock import patch, MagicMock
import io
import pytest

from src.engine.processor import download_and_split_pdf


def _make_pdf_response(text: str):
    """Build a mock requests.Response whose content is a minimal PDF with the given text."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)

    mock_resp = MagicMock()
    mock_resp.content = buf.read()
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def test_returns_chunks_for_valid_pdf():
    text = "Transformers are a type of neural network architecture. " * 50
    mock_resp = _make_pdf_response(text)

    with patch("src.engine.processor.requests.get", return_value=mock_resp):
        chunks = download_and_split_pdf("http://fake/paper.pdf")

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)


def test_returns_empty_list_on_http_error():
    with patch("src.engine.processor.requests.get", side_effect=Exception("timeout")):
        chunks = download_and_split_pdf("http://fake/paper.pdf")

    assert chunks == []


def test_returns_empty_list_for_image_based_pdf():
    """A PDF with no extractable text should return []."""
    import fitz

    doc = fitz.open()
    doc.new_page()  # blank page — no text
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    buf.seek(0)

    mock_resp = MagicMock()
    mock_resp.content = buf.read()
    mock_resp.raise_for_status = MagicMock()

    with patch("src.engine.processor.requests.get", return_value=mock_resp):
        chunks = download_and_split_pdf("http://fake/paper.pdf")

    assert chunks == []


def test_chunk_size_is_respected():
    text = "word " * 2000
    mock_resp = _make_pdf_response(text)

    with patch("src.engine.processor.requests.get", return_value=mock_resp):
        chunks = download_and_split_pdf("http://fake/paper.pdf", chunk_size=200, chunk_overlap=0)

    assert all(len(c) <= 200 for c in chunks)
