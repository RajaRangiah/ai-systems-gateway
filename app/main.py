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
# ==========================
# Experiment 5: Adaptive Fanout
# ==========================

class SearchAdaptiveResp(BaseModel):
    latency_ms: float
    fanout_initial: int
    fanout_final: int
    retried: bool
    shards_queried: list[int]
    hits: list[SearchHit]
    best_score: float


@app.post("/v1/search_adaptive", response_model=SearchAdaptiveResp)
def search_adaptive(req: SearchReq):
    """
    Adaptive fanout search:
    1) Run cheap fanout
    2) If confidence is low, retry with full fanout
    """
    t0 = time.time()

    qvec = embed_text(req.query, dim=settings.vector_dim)

    fanout_initial = req.fanout
    fanout_max = settings.shard_count
    CONF_THRESHOLD = 0.40  # adjust if needed

    def run_search(fanout: int):
        shards = candidate_shards(qvec, settings.shard_count, fanout=fanout)
        hits: list[SearchHit] = []
        for shard_id in shards:
            results = search_shard(shard_id, qvec, top_k=req.top_k)
            for r in results:
                hits.append(
                    SearchHit(
                        score=float(r.score),
                        shard_id=shard_id,
                        payload=r.payload or {},
                    )
                )
        hits.sort(key=lambda x: x.score, reverse=True)
        return shards, hits

    # Pass 1: cheap fanout
    shards1, hits1 = run_search(fanout_initial)
    best1 = hits1[0].score if hits1 else 0.0

    retried = False
    shards_final = shards1
    hits_final = hits1
    best_final = best1
    fanout_final = fanout_initial

    # Pass 2: retry if confidence is low
    if best1 < CONF_THRESHOLD and fanout_initial < fanout_max:
        retried = True
        fanout_final = fanout_max
        shards2, hits2 = run_search(fanout_final)

        # Merge results (avoid duplicates)
        seen = set()
        merged: list[SearchHit] = []
        for h in hits2 + hits1:
            key = (h.payload.get("doc_id"), h.payload.get("text"))
            if key not in seen:
                seen.add(key)
                merged.append(h)

        merged.sort(key=lambda x: x.score, reverse=True)
        hits_final = merged[: req.top_k]
        shards_final = shards2
        best_final = hits_final[0].score if hits_final else 0.0

    return SearchAdaptiveResp(
        latency_ms=(time.time() - t0) * 1000.0,
        fanout_initial=fanout_initial,
        fanout_final=fanout_final,
        retried=retried,
        shards_queried=shards_final,
        hits=hits_final,
        best_score=best_final,
    )

