import os
from typing import List, Tuple
import psycopg2
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from openai import OpenAI
from dotenv import load_dotenv
from embed_utils import generate_embeddings

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

def get_top_k_chunks(query: str, k: int = 5) -> List[Tuple[str, float]]:
    """Retrieve top-k most relevant chunks for the query."""
    # Use OpenAI SDK embedding from embed_utils
    query_emb = generate_embeddings([query])[0]
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

def generate_answer(query: str, context_chunks: List[str], history_text: str = "") -> str:
    """Generate an answer using Anthropic Claude Sonnet with context and conversation history."""
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are an expert assistant. Use the following context and conversation history to answer the user's question. If the answer is not in the context, say you don't know.

Conversation history:
{history_text}

Context:
{context}

Question: {query}
Answer:
"""
    llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = llm.invoke(prompt)
    return response.content.strip() if hasattr(response, 'content') else str(response) 
