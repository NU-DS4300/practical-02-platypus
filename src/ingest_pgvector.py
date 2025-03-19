import ollama
import psycopg2
import os
import fitz
from pgvector.psycopg2 import register_vector

# Initialize PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres",
)
conn.autocommit = True  # Required for creating extensions
cursor = conn.cursor()
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
register_vector(conn)

VECTOR_DIM = 768
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "vector_cosine_ops"



def clear_postgres_store():
    # Create vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
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

def create_hnsw_index():
    # Create vector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    # Drop existing index if it exists
    cursor.execute("DROP INDEX IF EXISTS embedding_index;")
    # Create HNSW index
    cursor.execute("""
        CREATE INDEX embedding_index
        ON documents
        USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()
    print("Index created successfully.")

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def store_embedding(file: str, page: int, chunk_idx: int, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}{file}_page_{page}_chunk_{chunk_idx}"
    # Upsert embedding into the documents table
    cursor.execute("""
        INSERT INTO documents (key, file, page, chunk, embedding)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (key) DO UPDATE
        SET embedding = EXCLUDED.embedding;
    """, (key, file, page, chunk, embedding))
    conn.commit()
    print(f"Stored embedding for: {key}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_idx, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=page_num,
                        chunk_idx=chunk_idx,
                        chunk=chunk,
                        embedding=embedding
                    )
            print(f" -----> Processed {file_name}")

def query_pgvector(query_text: str):
    embedding = get_embedding(query_text)
    cursor.execute("""
        SELECT key, file, page, chunk, embedding <=> %s AS distance
        FROM documents
        ORDER BY distance
        LIMIT 5;
    """, (embedding,))
    results = cursor.fetchall()
    for row in results:
        key, file, page, chunk, distance = row
        print(f"Key: {key}\nDistance: {distance}\nChunk: {chunk}\n---")

def main():
    clear_postgres_store()
    create_hnsw_index()
    process_pdfs("data")
    print("\n---Done processing PDFs---\n")
    query_pgvector("What is the capital of France?")

if __name__ == "__main__":
    main()