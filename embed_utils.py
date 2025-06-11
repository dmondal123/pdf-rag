import os
from typing import List
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

def upsert_chunks_with_embeddings(chunks: List[str], embeddings: List[List[float]], source: str):
    conn = get_db_connection()
    cur = conn.cursor()
    # Ensure table exists (update dimension to 1536 for text-embedding-3-small)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            source TEXT,
            chunk TEXT,
            embedding VECTOR(1536)
        );
    ''')
    for chunk, emb in zip(chunks, embeddings):
        cur.execute(
            """
            INSERT INTO documents (source, chunk, embedding)
            VALUES (%s, %s, %s)
            """,
            (source, chunk, emb)
        )
    conn.commit()
    cur.close()
    conn.close() 