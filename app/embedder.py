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
