from src.config import COLLECTION_NAME
from src.engine.processor import download_and_split_pdf
from src.engine.embedding import get_embeddings
from src.engine.vector_db import init_collection, upsert_to_qdrant, paper_exists


def ingest_paper(url: str, title: str, authors: list = None, published: str = None, abstract: str = None):
    print(f"--- Ingesting: {title} ---")
    init_collection(COLLECTION_NAME)

    if paper_exists(COLLECTION_NAME, url):
        print(f"--- Already indexed, skipping: {title} ---")
        return

    chunks = download_and_split_pdf(url)
    if not chunks:
        return
    vectors = get_embeddings(chunks)
    metadata = {
        "title": title,
        "pdf_url": url,
        "authors": authors or [],
        "published": published or "",
        "abstract": abstract or "",
    }
    upsert_to_qdrant(COLLECTION_NAME, chunks, vectors, metadata)
    print(f"--- Successfully ingested {title} ---")


if __name__ == "__main__":
    test_url = "https://arxiv.org/pdf/1706.03762.pdf"
    ingest_paper(test_url, "Attention Is All You Need")
