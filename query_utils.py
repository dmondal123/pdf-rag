import os
from typing import List, Tuple
import psycopg2
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from retriever import get_retriever

from openai import OpenAI
from dotenv import load_dotenv
from embed_utils import generate_embeddings
from rerank_utils import rerank_documents

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
    """Get top k most relevant chunks using vector similarity and embedding-based reranking."""
    # Get retriever
    retriever = get_retriever(k=k)
    
    # Get reranked documents
    docs = retriever.invoke(query)
    
    # Convert to expected format
    return [(doc.page_content, doc.metadata.get("similarity", 0.0)) for doc in docs]

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
    llm = ChatOpenAI(
        model="claude-3-7-sonnet-20250219",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    response = llm.invoke(prompt)
    return response.content.strip() if hasattr(response, 'content') else str(response) 
