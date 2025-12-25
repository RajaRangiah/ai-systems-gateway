# ADR-001: Vector Partitioning to Prevent Hot Shards

## Status
Accepted

## Context
Semantic workloads create skew that leads to hot shards in vector databases.

## Decision
Introduce a FastAPI gateway with XOR-based write spread and fanout-based reads.

## Consequences
- Prevents hot shards
- Requires fanout on reads to maintain recall
