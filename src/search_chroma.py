import chromadb
import ollama
import utils

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="embedding_index")

VECTOR_DIM = 768


def search_embeddings(query, top_k=3):
    query_embedding = utils.get_embedding(query)

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





def interactive_search():
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your search query: ")
        
        if query.lower() == "exit":
            break
        
        context_results = search_embeddings(query)
        response = utils.generate_rag_response(query, context_results)
        
        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    interactive_search()
