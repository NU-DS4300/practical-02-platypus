## DS 4300 Example - from docs
import numpy as np
import utils
import redis
from config import (
    VECTOR_DIM, 
    DOC_PREFIX, 
    INDEX_NAME, 
    REDIS_DISTANCE_METRIC, 
    get_redis_client
)

# Initialize Redis client
redis_client = get_redis_client()

# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {REDIS_DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# store the embedding in Redis
def store_embedding_redis(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {key}")


def pipeline_redis(chunk_size: int = 8000, overlap: int = 100, embedding_model: str = "nomic-embed-text"):
    clear_redis_store()
    create_hnsw_index()
    to_store = utils.process_pdfs("data", chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
    for file, page, chunk, embedding in to_store:
        store_embedding_redis(file=file, page=page, chunk=chunk, embedding=embedding)
    print("\n---Done processing PDFs---\n")


if __name__ == "__main__":
    pipeline_redis()
