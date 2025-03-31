import psycopg2
from pgvector.psycopg2 import register_vector
import chromadb
import redis

# ---------- Shared variables ----------
VECTOR_DIM = 1024
DOC_PREFIX = "doc:"

# ---------- PGVector setup ----------
def get_pg_connection():
    """Returns a configured PostgreSQL connection with pgvector extension."""
    pg_conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="postgres",
    )
    pg_conn.autocommit = True  # Required for creating extensions
    pg_cursor = pg_conn.cursor()
    pg_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(pg_conn)
    return pg_conn, pg_cursor

# ---------- Chroma setup ----------
COLLECTION_NAME = "ds4300_embeddings"

def get_chroma_client():
    """Returns a configured ChromaDB client with collection."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return chroma_client, collection

# ---------- Redis setup ----------
def get_redis_client():
    """Returns a configured Redis client."""
    redis_client = redis.Redis(host="localhost", port=6379, db=0)
    return redis_client

INDEX_NAME = "embedding_index"
REDIS_DISTANCE_METRIC = "COSINE"