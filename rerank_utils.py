from typing import List, Tuple
from flashrank import Ranker

def rerank_documents(query: str, documents: List[Tuple[str, float]], top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Rerank documents using FlashRank based on relevance to the query.
    
    Args:
        query: The search query
        documents: List of (document_text, similarity_score) tuples
        top_k: Number of top documents to return
        
    Returns:
        List of reranked (document_text, similarity_score) tuples
    """
    # Initialize FlashRank with default model
    ranker = Ranker()
    
    # Prepare documents for reranking
    docs_to_rerank = [{"text": doc[0], "score": doc[1]} for doc in documents]
    
    # Rerank documents
    reranked_docs = ranker.rerank(query, docs_to_rerank, top_k=top_k)
    
    # Convert back to original format
    return [(doc["text"], doc["score"]) for doc in reranked_docs]