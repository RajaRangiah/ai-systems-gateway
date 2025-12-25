from pydantic import BaseModel

class Settings(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    shard_count: int = 8
    vector_dim: int = 384
    collection_prefix: str = "docs_"

settings = Settings()
