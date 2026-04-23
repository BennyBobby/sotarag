from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import uuid
from src.config import QDRANT_URL, COLLECTION_NAME
from src.logger import get_logger

logger = get_logger(__name__)
client = QdrantClient(url=QDRANT_URL)


def init_collection(collection_name: str = COLLECTION_NAME):
    """Create a collection in Qdrant if it doesn't exist."""
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        logger.info("Collection '%s' created.", collection_name)


def paper_exists(collection_name: str, pdf_url: str) -> bool:
    """Check if a paper URL is already indexed to avoid duplicates."""
    results, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="url", match=MatchValue(value=pdf_url))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(results) > 0


def upsert_to_qdrant(collection_name: str, chunks: list, vectors: list, metadata: dict):
    """Store chunks and their vectors into Qdrant."""
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk,
                    "source": metadata.get("title", "Unknown"),
                    "url": metadata.get("pdf_url", ""),
                    "authors": metadata.get("authors", []),
                    "published": metadata.get("published", ""),
                    "abstract": metadata.get("abstract", ""),
                    "chunk_index": i,
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    logger.info("Successfully indexed %d chunks in Qdrant.", len(points))


def search_in_qdrant(collection_name: str, query_vector: list, limit: int = 3):
    """Search for the most similar chunks in Qdrant."""
    hits = client.query_points(
        collection_name=collection_name, query=query_vector, limit=limit
    ).points

    results = []
    for hit in hits:
        results.append(
            {
                "text": hit.payload.get("text"),
                "source": hit.payload.get("source"),
                "url": hit.payload.get("url"),
                "authors": hit.payload.get("authors", []),
                "published": hit.payload.get("published", ""),
                "score": hit.score,
            }
        )
    return results


def delete_paper(collection_name: str, pdf_url: str) -> int:
    """Delete all chunks belonging to a paper identified by its URL. Returns deleted count."""
    client.delete(
        collection_name=collection_name,
        points_selector=Filter(
            must=[FieldCondition(key="url", match=MatchValue(value=pdf_url))]
        ),
    )
    logger.info("Deleted paper with url: %s", pdf_url)
    return 1


def get_indexed_papers(collection_name: str = COLLECTION_NAME):
    """Retrieve unique paper titles and URLs from the collection metadata."""
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    indexed_papers = {}
    for pt in points:
        title = pt.payload.get("source")
        url = pt.payload.get("url")
        if title and title not in indexed_papers:
            indexed_papers[title] = url

    return indexed_papers
