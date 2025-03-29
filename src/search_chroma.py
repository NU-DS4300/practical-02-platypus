import chromadb
import ollama

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="embedding_index")

VECTOR_DIM = 768


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    query_embedding = get_embedding(query)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        # Fix: Extract first element since Chroma returns a list of lists
        metadatas = results["metadatas"][0]  # Extract list of metadata dictionaries
        distances = results["distances"][0]  # Extract list of distances

        top_results = [
            {
                "file": metadata.get("file", "Unknown file"),
                "page": metadata.get("page", "Unknown page"),
                "chunk": metadata.get("chunk", "Unknown chunk"),
                "similarity": distance,
            }
            for metadata, distance in zip(metadatas, distances)
        ]

        return top_results
    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )
    
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.
    
    Context:
    {context_str}
    
    Query: {query}
    
    Answer:"""
    
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]


def interactive_search():
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your search query: ")
        
        if query.lower() == "exit":
            break
        
        context_results = search_embeddings(query)
        response = generate_rag_response(query, context_results)
        
        print("\n--- Response ---")
        print(response)


def store_embedding(file, page, chunk, embedding):
    doc_id = f"{file}_page_{page}_chunk_{chunk}"
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )


if __name__ == "__main__":
    interactive_search()
