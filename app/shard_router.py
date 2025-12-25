import hashlib
import struct

def _u64(x: bytes) -> int:
    return int.from_bytes(x, "little", signed=False)

def hash_u64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return _u64(h)

def semantic_bucket(embedding: list[float], buckets: int) -> int:
    """
    Robust semantic bucketing:
    - quantize several dimensions of the embedding
    - hash the quantized bytes
    This makes different queries more likely to map to different buckets.
    """
    idxs = [1, 7, 31, 63, 127, 191, 255, 319]
    buf = bytearray()
    for i in idxs:
        x = embedding[i] if i < len(embedding) else 0.0
        # quantize float -> int16-ish bucket (-127..127) to stabilize
        q = int(max(-127, min(127, round(x * 50))))
        buf += struct.pack("b", q)

    h = hashlib.blake2b(bytes(buf), digest_size=8).digest()
    return int.from_bytes(h, "little") % buckets

def doc_bucket(doc_id: str, buckets: int) -> int:
    return hash_u64(doc_id) % buckets

def write_shard(doc_id: str, embedding: list[float], shard_count: int) -> int:
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
