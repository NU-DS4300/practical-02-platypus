import chromadb
import ollama
import numpy as np

# Initialize Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Ensure collection uses cosine similarity (default is Euclidean)
collection = chroma_client.get_or_create_collection(
    name="embedding_index",
    metadata={"hnsw:space": "cosine"},  # Set cosine similarity for better search
)

VECTOR_DIM = 768  # Match the embedding model's output dimension


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Get a normalized embedding vector."""
    response = ollama.embeddings(model=model, prompt=text)
    embedding = np.array(response["embedding"], dtype=np.float32)
    return embedding / np.linalg.norm(embedding)  # Normalize for cosine similarity


def search_embeddings(query, top_k=3):
    """Perform a fast vector search in ChromaDB."""
    query_embedding = get_embedding(query)  # Generate normalized query embedding

    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],  # Convert to list for Chroma
            n_results=top_k,
        )

        # Ensure query results exist
        if not results or not results["metadatas"]:
            return []

        metadatas = results["metadatas"][0]  # Extract metadata list
        distances = results["distances"][0]  # Extract distances list

        return sorted(
            [
                {
                    "file": metadata.get("file", "Unknown file"),
                    "page": metadata.get("page", "Unknown page"),
                    "chunk": metadata.get("chunk", "Unknown chunk"),
                    "similarity": 1 - distance,  # Convert to similarity score
                }
                for metadata, distance in zip(metadatas, distances)
            ],
            key=lambda x: x["similarity"],
            reverse=True,  # Sort highest similarity first
        )[:top_k]

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    """Generate response with retrieved context."""
    context_str = "\n".join(
        [
            f"From {r['file']} (page {r['page']}, chunk {r['chunk']}) with similarity {r['similarity']:.2f}"
            for r in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query accurately. If the context 
    is not relevant, say 'I don't know'.

    Context:
    {context_str}

    Query: {query}

    Answer:"""

    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search loop."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    # ⚡ Warm up embedding model (avoid cold start)
    get_embedding("test")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


def store_embedding(file, page, chunk, text):
    """Store a document embedding in ChromaDB."""
    embedding = get_embedding(text)  # Generate embedding for the chunk
    doc_id = f"{file}_page_{page}_chunk_{chunk}"

    collection.add(
        ids=[doc_id],
        embeddings=[embedding.tolist()],  # Convert NumPy array to list
        metadatas=[{"file": file, "page": page, "chunk": chunk, "text": text}],
    )


if __name__ == "__main__":
    interactive_search()
