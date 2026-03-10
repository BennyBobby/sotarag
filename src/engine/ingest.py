from src.engine.processor import download_and_split_pdf
from src.engine.embedding import get_embeddings
from src.engine.vector_db import init_collection, upsert_to_qdrant


def ingest_paper(url: str, title: str):
    print(f"--- Ingesting: {title} ---")
    init_collection("sota_papers")
    chunks = download_and_split_pdf(url)
    if not chunks:
        return
    vectors = get_embeddings(chunks)
    metadata = {"title": title, "pdf_url": url}
    upsert_to_qdrant("sota_papers", chunks, vectors, metadata)

    print(f"--- Successfully ingested {title} ---")


if __name__ == "__main__":
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"
    ingest_paper(test_url, "Attention Is All You Need")
