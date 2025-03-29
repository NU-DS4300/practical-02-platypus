import ollama
import chromadb
import numpy as np
import os
import fitz  # PyMuPDF

VECTOR_DIM = 768
COLLECTION_NAME = "ds4300_embeddings"

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Stores data persistently
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


# Clear existing ChromaDB collection
def clear_chroma_store():
    print("Clearing existing ChromaDB store...")
    chroma_client.delete_collection(name=COLLECTION_NAME)
    global collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print("ChromaDB store cleared.")


# Generate an embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the embedding in ChromaDB
def store_embedding(file: str, page: int, chunk: str, embedding: list):
    """Stores chunk embeddings in ChromaDB."""
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk)}"
    
    collection.add(
        ids=[doc_id],
        embeddings=[embedding],
        metadatas=[{"file": file, "page": page, "chunk": chunk}]
    )
    
    print(f"Stored embedding for: {chunk[:50]}...")


# Extract text from a PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir):
    """Reads PDFs, extracts text, generates embeddings, and stores them in ChromaDB."""
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size=8000, overlap=100)
                
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(file=file_name, page=page_num, chunk=chunk, embedding=embedding)

            print(f" -----> Processed {file_name}")


# Query ChromaDB for similar documents
def query_chroma(query_text: str, top_k=5):
    """Searches for similar text chunks in ChromaDB."""
    embedding = get_embedding(query_text)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    print("\nTop Matches:")
    for i in range(len(results["ids"][0])):
        print(f" - {results['metadatas'][0][i]['chunk'][:100]}... (Page {results['metadatas'][0][i]['page']})")


# Main function to run the pipeline
def main():
    clear_chroma_store()
    process_pdfs("data")
    print("\n---Done processing PDFs---\n")

    query_chroma("What is the capital of France?")


if __name__ == "__main__":
    main()
