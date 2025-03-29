# Variables to explore:
# Preprocessing: Chunk size/overlap, stop word removal, whitespace adjustment
# Embedding: Three models (nomic-embed-text, all-miniLM, mxbai-embed-large)
# Databases: Three vector databases (pgvector, redis, chroma)
# System prompt tweaks: Basic prompt, few-shot prompt, long role-based prompt
# LLMs: Two LLMs (deepseek-r1 1.5b, llama3 8b)
# topk: 3, 5, 10

import psycopg2
from pgvector.psycopg2 import register_vector
import chromadb
import redis
import ingest_chroma, ingest_pgvector, ingest_redis
import search_chroma, search_pgvector, search_redis
import utils

'''
def run_rag_query_test(query, top_k=3, llm="deepseek-r1:1.5b", prompt: str = None):
    context_results = search_embeddings(query, top_k=top_k)
    response = utils.generate_rag_response(query, context_results, model=llm, prompt=prompt)
    return response'''

# ---------- Shared setup ----------
VECTOR_DIM = 768
DOC_PREFIX = "doc:"

# ---------- PGVector setup ----------
pg_conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres",
)
pg_conn.autocommit = True  # Required for creating extensions
pg_cursor = pg_conn.cursor()
pg_cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
register_vector(pg_conn)

# PG_DISTANCE_METRIC = "vector_cosine_ops" # Not needed ?????????? TODO: Check

# ---------- Chroma setup ----------
COLLECTION_NAME = "ds4300_embeddings"
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Stores data persistently
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# ---------- Redis setup ----------
redis_client = redis.Redis(host="localhost", port=6379, db=0)

INDEX_NAME = "embedding_index"
REDIS_DISTANCE_METRIC = "COSINE"

# ---------- Prompt setup ----------
prompt_prefix_basic = f"""You are a helpful AI assistant. 
Use the following context to answer the query as accurately as possible. If the context is 
not relevant to the query, say 'I don't know'."""
prompt_role_based = f"""You are an expert data engineer with a deep understanding of relational databases and NoSQL databases, including the relational model as well as Redis, Neo4j, Mongo. You are also an expert in AWS as a service. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say that, then answer to the best of your ability anyways."""

prompt_few_shot = """""" # TODO: Make few shot prompt

prompts = [prompt_prefix_basic, prompt_role_based, prompt_few_shot]

# ---------- Run Experiments ----------
# TODO: Make sure it's gradeable somehow <----- most important and hardest of the todos
# Ingestion Variables
for embedding_model in ["nomic-embed-text", "all-MiniLM-L6-v2", "mxbai-embed-large"]:
    for chunk_size in [1000, 5000, 8000]:
        for overlap in [100, 500, 1000]:   
            # Ingestion
            ingest_pgvector.pipeline_pgvector(chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
            ingest_chroma.pipeline_chroma(chroma_client, chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
            ingest_redis.pipeline_redis(chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
            # Search variables
            for topk in [3, 5, 10]:
                for llm in ["deepseek-r1:1.5b", "llama3:8b"]:
                    for prompt in prompts:
                        # Search
                        # TODO: change search function
                        search_pgvector.interactive_search_pgvector(topk=topk, llm=llm, prompt=prompt)
                        search_chroma.interactive_search_chroma(topk=topk, llm=llm, prompt=prompt)
                        search_redis.interactive_search_redis(topk=topk, llm=llm, prompt=prompt)