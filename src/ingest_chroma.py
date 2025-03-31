import chromadb
import utils
from config import (
    VECTOR_DIM, 
    COLLECTION_NAME, 
    get_chroma_client
)

# Initialize ChromaDB client
chroma_client, collection = get_chroma_client()

# Clear existing ChromaDB collection
def clear_chroma_store(chroma_client, collection):
    print("Clearing existing ChromaDB store...")
    chroma_client.delete_collection(name=COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print("ChromaDB store cleared.")
    return collection

# Store the embedding in ChromaDB
def store_embedding_chroma(collection, file: str, page: int, chunk: str, embedding: list):
    """Stores chunk embeddings in ChromaDB."""
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk)}"
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
    
    print(f"Stored embedding for: {chunk[:50]}...")


# Main function to run the pipeline
def pipeline_chroma(chroma_client, collection, chunk_size: int = 8000, overlap: int = 100, embedding_model: str = "nomic-embed-text"):
    collection = clear_chroma_store(chroma_client, collection)
    to_store = utils.process_pdfs("data", chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
    for file, page, chunk, embedding in to_store:
        store_embedding_chroma(collection, file=file, page=page, chunk=chunk, embedding=embedding)
    print("\n---Done processing PDFs---\n")
    return collection


if __name__ == "__main__":
    pipeline_chroma()
