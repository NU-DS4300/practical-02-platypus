import ollama
import psycopg2
import os
import fitz
from pgvector.psycopg2 import register_vector
import utils

from config import VECTOR_DIM, DOC_PREFIX, get_pg_connection

# Get the connection
conn, cursor = get_pg_connection()



def clear_postgres_store():
    # Create vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("DROP TABLE IF EXISTS documents;")
    # Create table for storing documents and embeddings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            key TEXT UNIQUE,
            file TEXT,
            page INTEGER,
            chunk TEXT,
            embedding VECTOR(%s)
        );
    """, (VECTOR_DIM,))
    conn.commit()
    print("Clearing existing Postgres store...")
    cursor.execute("DELETE FROM documents;")
    conn.commit()
    print("Postgres store cleared.")

def create_hnsw_index_pg():
    # Create vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Create HNSW index
    cursor.execute("DROP INDEX IF EXISTS embedding_index;")
    cursor.execute("""
        CREATE INDEX embedding_index
        ON documents
        USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()
    print("Index created successfully.")

def store_embedding_pg(file: str, page: int, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}{file}_page_{page}_chunk_{chunk}"
    # Upsert embedding into the documents table
    cursor.execute("""
        INSERT INTO documents (key, file, page, chunk, embedding)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (key) DO UPDATE
        SET embedding = EXCLUDED.embedding;
    """, (key, file, page, chunk, embedding))
    conn.commit()
    print(f"Stored embedding for: {key}")

def pipeline_pgvector(chunk_size: int = 8000, overlap: int = 100, embedding_model: str = "nomic-embed-text" ):
    """
    Pipeline for experiments using pgvector database.

    Args:
        chunk_size (int): Chunk size for PDF processing.
        overlap (int): Overlap between chunks.
        embedding_model (str): Embedding model to use.
    """
    clear_postgres_store()
    create_hnsw_index_pg()
    to_store = utils.process_pdfs("data", chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
    for file, page, chunk, embedding in to_store:
        store_embedding_pg(file=file, page=page, chunk=chunk, embedding=embedding)
    print("\n---Done processing PDFs---\n")

if __name__ == "__main__":
    # Initialize PostgreSQL connection

    pipeline_pgvector()