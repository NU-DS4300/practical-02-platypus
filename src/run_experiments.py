# Variables to explore:
# Preprocessing: Chunk size/overlap, stop word removal, whitespace adjustment
# Embedding: Three models (nomic-embed-text, all-miniLM, mxbai-embed-large)
# Databases: Three vector databases (pgvector, redis, chroma)
# System prompt tweaks: Basic prompt, long role-based prompt
# LLMs: Two LLMs (deepseek-r1 1.5b, llama3 8b)
# topk: 1, 5, 10

import pandas as pd
import numpy as np
import utils
import os
from datetime import datetime

# Database-specific modules
import ingest_chroma
import ingest_pgvector
import ingest_redis
from redis.commands.search.query import Query

# Database connection modules
import psycopg2
from pgvector.psycopg2 import register_vector
import chromadb
import redis

# Import configuration
from config import (
    
    # Database setup
    get_pg_connection,
    get_chroma_client,
    get_redis_client,
    INDEX_NAME,
)

# Get database connections
pg_conn, pg_cursor = get_pg_connection()
chroma_client, collection = get_chroma_client()
redis_client = get_redis_client()

def save_batch(results_list, batch_num):
    """Save a batch of results to a CSV file."""
    if not results_list:  # Skip saving if the list is empty
        return
    
    # Create directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save to CSV
    batch_df = pd.DataFrame(results_list)
    filename = f"results/batch_{batch_num}_{timestamp}.csv"
    batch_df.to_csv(filename, index=False, sep="|")
    print(f"Saved batch {batch_num} with {len(results_list)} results to {filename}")
    


def search_embeddings_redis(query, embed_model, top_k=5):

    query_embedding = utils.get_embedding(query, model=embed_model)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []

def search_embeddings_pgvector(query, model="nomic-embed-text", top_k=5):
    query_embedding = utils.get_embedding(query, model=model)
    
    try:
        # Add ::vector cast to the parameter
        pg_cursor.execute("""
            SELECT file, page, chunk, embedding <=> %s::vector AS similarity
            FROM documents
            ORDER BY similarity
            LIMIT %s;
        """, (query_embedding, top_k))
        
        results = pg_cursor.fetchall()
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


def search_embeddings_chroma(query, embed_model, top_k=3):
    query_embedding = utils.get_embedding(query, model=embed_model)

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

# ---------- Prompt setup ----------
prompt_prefix_basic = f"""You are a helpful AI assistant. 
Use the following context to answer the query as accurately as possible. If the context is 
not relevant to the query, say 'I don't know'."""
prompt_role_based = f"""You are an expert data engineer with a deep understanding of relational databases and NoSQL databases, including the relational model as well as Redis, Neo4j, Mongo. You are also an expert in AWS as a service. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say that, then answer to the best of your ability anyways."""

prompts = [prompt_prefix_basic, prompt_role_based]

questions = ["When was Redis released?", "How many databases does Redis support?", "What kind of imbalances in an AVL tree require multiple rotations?", "What is the EC2 lifecycle?", "When was neo4j's graph query language invented?", "Name the data types supported by Redis for values."]

# ---------- Run Experiments ----------
# Create a list to collect results
results_list = []
batch_num = 1000

for embedding_model in ["mxbai-embed-large"]:
    for chunk_size, overlap in [(64, 10), (320, 32), (500, 50)]:
            # Ingestion
            ingest_pgvector.pipeline_pgvector(chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
            collection = ingest_chroma.pipeline_chroma(chroma_client, collection, chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
            ingest_redis.pipeline_redis(chunk_size=chunk_size, overlap=overlap, embedding_model=embedding_model)
            
            # Search variables
            for topk in [1, 5, 10]:
                for llm in ["deepseek-r1:1.5b", "llama3:8b"]:
                    for prompt in prompts:
                        # Search
                        for question in questions:
                            # Redis search
                            batch_list = []
                            context_results_redis = search_embeddings_redis(question, embedding_model, top_k=topk)
                            response_redis = utils.generate_rag_response(question, context_results_redis, model=llm, prompt=prompt)
                            response_redis.replace("|", " ") 
                            results = {
                                "response": response_redis,
                                "database": "redis",
                                "embedding_model": embedding_model,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "topk": topk,
                                "llm": llm,
                                "prompt": prompt,
                                "question": question
                            }
                            results_list.append(results)
                            batch_list.append(results)
                            
                            # PGVector search
                            context_results_pgvector = search_embeddings_pgvector(question, embedding_model, top_k=topk)
                            response_pgvector = utils.generate_rag_response(question, context_results_pgvector, model=llm, prompt=prompt)
                            response_pgvector.replace("|", " ")
                            results = {
                                "response": response_pgvector,
                                "database": "pgvector",
                                "embedding_model": embedding_model,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "topk": topk,
                                "llm": llm,
                                "prompt": prompt,
                                "question": question
                            }
                            results_list.append(results)
                            batch_list.append(results)
                            
                            # Chroma search
                            context_results_chroma = search_embeddings_chroma(question, embedding_model, top_k=topk)
                            response_chroma = utils.generate_rag_response(question, context_results_chroma, model=llm, prompt=prompt)
                            response_chroma.replace("|", " ")
                            results = {
                                "response": response_chroma,
                                "database": "chroma",
                                "embedding_model": embedding_model,
                                "chunk_size": chunk_size,
                                "overlap": overlap,
                                "topk": topk,
                                "llm": llm,
                                "prompt": prompt,
                                "question": question
                            }
                            results_list.append(results)
                            batch_list.append(results)

                            save_batch(batch_list, batch_num)
                            batch_num += 1

# Create DataFrame from results list
results_df = pd.DataFrame(results_list)

# Save to CSV
results_df.to_csv("results.csv", index=False)