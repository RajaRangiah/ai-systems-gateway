# scaffold.ps1
$ErrorActionPreference = "Stop"

# Create folders
New-Item -ItemType Directory -Force -Path "app" | Out-Null

# docker-compose.yml
@"
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
"@ | Set-Content -Encoding UTF8 "docker-compose.yml"

# requirements.txt
@"
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.8.2
qdrant-client==1.11.1
numpy==2.0.1
"@ | Set-Content -Encoding UTF8 "requirements.txt"

# app/__init__.py
"" | Set-Content -Encoding UTF8 "app/__init__.py"

# app/settings.py
@"
from pydantic import BaseModel

class Settings(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    shard_count: int = 8
    vector_dim: int = 384
    collection_prefix: str = "docs_"

settings = Settings()
"@ | Set-Content -Encoding UTF8 "app/settings.py"

# app/embedder.py
@"
import numpy as np
import hashlib

def embed_text(text: str, dim: int = 384) -> list[float]:
    """
    Deterministic pseudo-embedding for learning:
    - hashes tokens into a fixed-size vector
    - normalizes
    Swap later with real embeddings.
    """
    v = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().split()

    for t in tokens:
        h = hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h, "little") % dim
        v[idx] += 1.0

    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm

    return v.astype(np.float32).tolist()
"@ | Set-Content -Encoding UTF8 "app/embedder.py"

# app/shard_router.py
@"
import hashlib

def _u64(x: bytes) -> int:
    return int.from_bytes(x, "little", signed=False)

def hash_u64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return _u64(h)

def semantic_bucket(embedding: list[float], buckets: int) -> int:
    # Grab a few stable positions -> sign bits -> hash -> bucket
    idxs = [1, 7, 31, 63, 127, 191, 255, 319]
    bits = []
    for i in idxs:
        val = embedding[i] if i < len(embedding) else 0.0
        bits.append("1" if val >= 0 else "0")
    sig = "".join(bits)
    return hash_u64(sig) % buckets

def doc_bucket(doc_id: str, buckets: int) -> int:
    return hash_u64(doc_id) % buckets

def write_shard(doc_id: str, embedding: list[float], shard_count: int) -> int:
    # Neighborhood (semantic) + spread (doc) to avoid hot shards
    a = doc_bucket(doc_id, shard_count)
    b = semantic_bucket(embedding, shard_count)
    return (a ^ b) % shard_count

def candidate_shards(query_embedding: list[float], shard_count: int, fanout: int = 3) -> list[int]:
    # Bounded fanout: deterministic shard set for the query
    base = semantic_bucket(query_embedding, shard_count)
    out: list[int] = []
    for salt in range(fanout * 4):
        s = hash_u64(f"{base}:{salt}") % shard_count
        if s not in out:
            out.append(s)
        if len(out) >= fanout:
            break
    return out
"@ | Set-Content -Encoding UTF8 "app/shard_router.py"

# app/qdrant_store.py
@"
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
"@ | Set-Content -Encoding UTF8 "app/qdrant_store.py"

# app/main.py
@"
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any
import uuid
import time

from .settings import settings
from .embedder import embed_text
from .shard_router import write_shard, candidate_shards
from .qdrant_store import ensure_collections, upsert_point, search_shard

app = FastAPI(title="Gateway + Partitioned Vector Search")

@app.on_event("startup")
def _startup():
    ensure_collections()

class UpsertReq(BaseModel):
    doc_id: str = Field(..., description="Stable document id")
    text: str = Field(..., description="Chunk text to embed and store")
    metadata: dict[str, Any] = Field(default_factory=dict)

class UpsertResp(BaseModel):
    point_id: str
    shard_id: int

@app.post("/v1/upsert", response_model=UpsertResp)
def upsert(req: UpsertReq):
    vec = embed_text(req.text, dim=settings.vector_dim)
    shard_id = write_shard(req.doc_id, vec, settings.shard_count)

    point_id = str(uuid.uuid4())
    payload = {"doc_id": req.doc_id, "text": req.text, **req.metadata}

    upsert_point(shard_id, point_id, vec, payload)
    return UpsertResp(point_id=point_id, shard_id=shard_id)

class SearchReq(BaseModel):
    query: str
    top_k: int = 5
    fanout: int = 3

class SearchHit(BaseModel):
    score: float
    shard_id: int
    payload: dict[str, Any]

class SearchResp(BaseModel):
    latency_ms: float
    shards_queried: list[int]
    hits: list[SearchHit]

@app.post("/v1/search", response_model=SearchResp)
def search(req: SearchReq):
    t0 = time.time()

    qvec = embed_text(req.query, dim=settings.vector_dim)
    shards = candidate_shards(qvec, settings.shard_count, fanout=req.fanout)

    all_hits: list[SearchHit] = []
    for shard_id in shards:
        results = search_shard(shard_id, qvec, top_k=req.top_k)
        for r in results:
            all_hits.append(SearchHit(score=float(r.score), shard_id=shard_id, payload=r.payload or {}))

    all_hits.sort(key=lambda x: x.score, reverse=True)
    hits = all_hits[: req.top_k]

    return SearchResp(
        latency_ms=(time.time() - t0) * 1000.0,
        shards_queried=shards,
        hits=hits,
    )
"@ | Set-Content -Encoding UTF8 "app/main.py"

Write-Host "✅ Scaffold created. Next steps:"
Write-Host "1) docker compose up -d"
Write-Host "2) python -m venv .venv"
Write-Host "3) .\.venv\Scripts\Activate.ps1"
Write-Host "4) pip install -r requirements.txt"
Write-Host "5) uvicorn app.main:app --reload --port 8000"
