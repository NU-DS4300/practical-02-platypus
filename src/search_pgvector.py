import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
import ollama

# Initialize PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres",
)

# Set up vector extension
conn.autocommit = True
cursor = conn.cursor()
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
register_vector(conn)
conn.autocommit = False

VECTOR_DIM = 768
DOC_PREFIX = "doc:"

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def search_embeddings(query, top_k=5):
    query_embedding = get_embedding(query)
    
    try:
        # Add ::vector cast to the parameter
        cursor.execute("""
            SELECT file, page, chunk, embedding <=> %s::vector AS similarity
            FROM documents
            ORDER BY similarity
            LIMIT %s;
        """, (query_embedding, top_k))
        
        results = cursor.fetchall()
        top_results = [
            {
                "file": row[0],
                "page": row[1],
                "chunk": row[2],
                "similarity": float(row[3])
            }
            for row in results
        ]

        for result in top_results:
            print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [f"From {res.get('file', 'Unknown')} (page {res.get('page', '?')}, "
         f"chunk {res.get('chunk', '?')} - similarity: {res.get('similarity', 0):.2f}"
         for res in context_results]
    )

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. 
    If the context is not relevant, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model="mistral:latest", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

def interactive_search():
    print("🔍 PostgreSQL RAG Search Interface")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter your search query: ")
        if query.lower() == "exit":
            break
        
        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)
        
        print("\n--- Response ---")
        print(response)
        print("-----------------\n")

if __name__ == "__main__":
    interactive_search()