import os
import fitz
import ollama

# Generate an embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

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
def process_pdfs(data_dir, chunk_size=8000, overlap=100, embedding_model="nomic-embed-text"):
    """Reads PDFs, extracts text, generates embeddings, and stores them in ChromaDB."""
    to_store = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size=chunk_size, overlap=overlap)
                
                for chunk in chunks:
                    embedding = get_embedding(chunk, embedding_model)
                    to_store.append((file_name, page_num, chunk, embedding))

            print(f" -----> Processed {file_name}")

    return to_store


def generate_rag_response(query, context_results, model="deepseek-r1:1.5b", prompt=None):
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    if prompt is None:
        prompt = f"""You are a helpful AI assistant. 
        Use the following context to answer the query as accurately as possible. If the context is 
        not relevant to the query, say 'I don't know'."""
    prompt += f"""
        Context:
        {context_str}
        
        Query: {query}
        
        Answer:"""
    
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]
