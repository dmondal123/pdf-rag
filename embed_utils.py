import os
from typing import List, Dict, Any
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use OpenAI Python SDK for embedding generation

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        input=chunks,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", 5433),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "password"),
        dbname=os.getenv("PGDATABASE", "pdfrag")
    )

def upsert_chunks_with_embeddings(chunks: List[Dict[str, Any]], embeddings: List[List[float]], source: str):
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Create tables for different content types
    cur.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            source TEXT,
            content_type TEXT,
            content TEXT,
            embedding VECTOR(1536),
            metadata JSONB
        );
    ''')
    
    for chunk, emb in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO documents (source, content_type, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                source,
                chunk['type'],
                chunk['content'],
                emb,
                chunk.get('metadata', {})
            )
        )
    
    conn.commit()
    cur.close()
    conn.close() 