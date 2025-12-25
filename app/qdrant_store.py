from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from .settings import settings

client = QdrantClient(url=settings.qdrant_url)

def shard_collection_name(shard_id: int) -> str:
    return f"{settings.collection_prefix}{shard_id}"

def ensure_collections():
    for shard_id in range(settings.shard_count):
        name = shard_collection_name(shard_id)
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=settings.vector_dim, distance=Distance.COSINE),
            )

def upsert_point(shard_id: int, point_id: str, vector: list[float], payload: dict):
    name = shard_collection_name(shard_id)
    client.upsert(
        collection_name=name,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )

def search_shard(shard_id: int, query_vector: list[float], top_k: int = 5):
    name = shard_collection_name(shard_id)
    return client.search(
        collection_name=name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
