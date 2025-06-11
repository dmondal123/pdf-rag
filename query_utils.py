import os
from typing import List, Tuple
import psycopg2
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIM = 1536  # Anthropic embedding size

def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-large")

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", 5433),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "password"),
        dbname=os.getenv("PGDATABASE", "pdfrag")
    )

def get_top_k_chunks(query: str, k: int = 5) -> List[Tuple[str, float]]:
    """Retrieve top-k most relevant chunks for the query."""
    embedder = get_embedding_model()
    query_emb = embedder.embed_query(query)
    # Format as pgvector literal: '[0.1,0.2,...]'
    vector_str = '[' + ','.join(str(x) for x in query_emb) + ']'
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f'''
        SELECT chunk, embedding <#> %s::vector AS distance
        FROM documents
        ORDER BY distance ASC
        LIMIT %s;
    ''', (vector_str, k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [(row[0], row[1]) for row in results]

def generate_answer(query: str, context_chunks: List[str]) -> str:
    """Generate an answer using Anthropic Claude Sonnet with context."""
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert assistant. Use the following context to answer the user's question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {query}
Answer:
"""
    llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = llm.invoke(prompt)
    return response.content.strip() if hasattr(response, 'content') else str(response) 
