# Practical 02

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Python with requirements.txt installed (`pip install -r requirements.txt`)
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.
- Pgvector stack running in a similar manner on port 5432.
- chroma_db stack running in a similar manner on port 6379

## Source Code
- config.py: set up common variables. If you change the embedding model used, you might need to change vector_dim to 1024/768
- ingest_*.py: ingest the data for that database
- search_*.py: run an interactive search for that database after ingesting the files
- run_experiments.py: runs the experiments and saves data to results folder
- utils.py: shared functionality
- response.py: grade answers and generate plots
