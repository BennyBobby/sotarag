from unittest.mock import patch, call

import src.engine.ingest as ingest_module
from src.engine.ingest import ingest_paper


def test_skips_when_paper_already_indexed():
    with (
        patch("src.engine.ingest.init_collection"),
        patch("src.engine.ingest.paper_exists", return_value=True),
        patch("src.engine.ingest.download_and_split_pdf") as mock_download,
    ):
        ingest_paper("http://x.com/p.pdf", "Some Paper")

    mock_download.assert_not_called()


def test_full_pipeline_called_for_new_paper():
    chunks = ["chunk1", "chunk2"]
    vectors = [[0.1] * 1024, [0.2] * 1024]

    with (
        patch("src.engine.ingest.init_collection"),
        patch("src.engine.ingest.paper_exists", return_value=False),
        patch("src.engine.ingest.download_and_split_pdf", return_value=chunks),
        patch("src.engine.ingest.get_embeddings", return_value=vectors),
        patch("src.engine.ingest.upsert_to_qdrant") as mock_upsert,
    ):
        ingest_paper("http://x.com/p.pdf", "New Paper", authors=["Alice"], published="2024")

    mock_upsert.assert_called_once()


def test_stops_early_when_pdf_has_no_text():
    with (
        patch("src.engine.ingest.init_collection"),
        patch("src.engine.ingest.paper_exists", return_value=False),
        patch("src.engine.ingest.download_and_split_pdf", return_value=[]),
        patch("src.engine.ingest.get_embeddings") as mock_embed,
    ):
        ingest_paper("http://x.com/bad.pdf", "Bad PDF")

    mock_embed.assert_not_called()
