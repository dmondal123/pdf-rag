from typing import List, Dict, Any
import psycopg2
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class PGVectorRetriever(BaseRetriever):
    """Custom retriever that uses pgvector for similarity search with embedding-based reranking."""
    
    def __init__(self, k: int = 5):
        self.k = k
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def get_db_connection(self):
        return psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=os.getenv("PGPORT", 5433),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "password"),
            dbname=os.getenv("PGDATABASE", "pdfrag")
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Convert embedding to string format for pgvector
        vector_str = str(query_embedding).replace('[', '{').replace(']', '}')
        
        # Get initial results using vector similarity
        conn = self.get_db_connection()
        cur = conn.cursor()
        
        # Get more results than needed for reranking
        cur.execute(f'''
            SELECT id, chunk, embedding <#> %s::vector AS distance
            FROM documents
            ORDER BY distance ASC
            LIMIT %s;
        ''', (vector_str, self.k * 2))  # Get 2x more results for reranking
        
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to LangChain Documents
        documents = [
            Document(
                page_content=chunk,
                metadata={"id": doc_id, "distance": distance}
            )
            for doc_id, chunk, distance in results
        ]
        
        # Rerank using cosine similarity with query embedding
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Add similarity scores to metadata
        for doc, similarity in zip(documents, similarities):
            doc.metadata["similarity"] = float(similarity)
        
        # Sort by similarity score
        reranked_docs = sorted(documents, key=lambda x: x.metadata["similarity"], reverse=True)
        
        # Return top k documents
        return reranked_docs[:self.k]

def get_retriever(k: int = 5) -> PGVectorRetriever:
    """Create a retriever with embedding-based reranking."""
    return PGVectorRetriever(k=k) 