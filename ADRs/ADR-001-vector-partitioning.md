# ADR-001: Gateway-Level Vector Partitioning to Prevent Hot Shards

## Status
Accepted

## Context
Semantic vector workloads exhibit “high write skew”: documents with similar meaning
produce near-identical embeddings and are repeatedly routed to the same physical shard
when naive semantic or hash-based partitioning is used.

In practice, this leads to:
- hot shards under write-heavy workloads
- elevated p99 latency and throttling on affected shards
- uneven infrastructure cost utilization
- cascading read latency due to overloaded shards

Native vector database sharding alone is insufficient to address this, because semantic
skew is introduced “before” persistence, at the embedding and ingestion layer.

## Decision
Introduce a **FastAPI gateway** responsible for shard-aware routing.

At write time, each document is routed to a single shard using an **XOR-based spread**
between:
- a document-derived bucket (stable per document)
- a semantic bucket derived from the embedding

This intentionally breaks pure semantic locality to evenly distribute writes.

At read time, queries are executed with “bounded fanout” across a small subset of
candidate shards, and results are merged at the gateway. Fanout is increased only when
initial recall confidence is insufficient.

## Consequences
### Positive
- Prevents hot shards caused by semantic skew
- Stabilizes write throughput and shard-level p99 latency
- Improves infrastructure cost balance
- Keeps write path single-shard and efficient

### Negative
- Read path requires fanout and result merging
- Recall depends on sufficient fanout
- Gateway becomes a critical routing component

### Mitigations
- Fanout is bounded and adaptive, not global
- Retry-based fanout escalation recovers recall only when necessary
- Gateway logic remains stateless and horizontally scalable

