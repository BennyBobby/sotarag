from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


client = QdrantClient(host="qdrant", port=6333)


def init_collection(collection_name: str = "sota_papers"):
    """Create a collection in Qdrant if it doesn't exist."""
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024, distance=Distance.COSINE
            ),  # Size depends on the embedding model
        )
        print(f"Collection '{collection_name}' created.")


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
                    "chunk_index": i,
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    print(f"Successfully indexed {len(points)} chunks in Qdrant.")
