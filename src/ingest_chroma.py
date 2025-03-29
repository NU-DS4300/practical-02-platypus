import chromadb
import utils

# Clear existing ChromaDB collection
def clear_chroma_store(chroma_client):
    print("Clearing existing ChromaDB store...")
    chroma_client.delete_collection(name=COLLECTION_NAME)
    global collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print("ChromaDB store cleared.")

# Store the embedding in ChromaDB
def store_embedding_chroma(file: str, page: int, chunk: str, embedding: list):
    """Stores chunk embeddings in ChromaDB."""
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk)}"
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
    
    print(f"Stored embedding for: {chunk[:50]}...")


# Main function to run the pipeline
def pipeline_chroma(chroma_client, chunk_size: int = 8000, overlap: int = 100, embedding_model: str = "nomic-embed-text"):
    clear_chroma_store(chroma_client)
    to_store = utils.process_pdfs("data", chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
    store_embedding_chroma(file=to_store[0], page=to_store[1], chunk=to_store[2], embedding=to_store[3])
    print("\n---Done processing PDFs---\n")


if __name__ == "__main__":
    
    VECTOR_DIM = 768
    COLLECTION_NAME = "ds4300_embeddings"

    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Stores data persistently
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    pipeline_chroma(chroma_client)
