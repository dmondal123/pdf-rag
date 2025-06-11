import os
from typing import List
import psycopg2
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Choose your embedding model (Anthropic, OpenAI, or HuggingFace)
# For this example, we'll use Anthropic via LangChain

def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-large")

def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    model = get_embedding_model()
    return model.embed_documents(chunks)

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
    # Ensure table exists
    cur.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            source TEXT,
            chunk TEXT,
            embedding VECTOR(3072)
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