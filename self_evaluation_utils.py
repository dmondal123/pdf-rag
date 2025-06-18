import os
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import Counter
import re
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class SelfRAGEvaluator:
    """Self-evaluation metrics for RAG systems without reference answers."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.llm = ChatOpenAI(
            model="gpt-4",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def evaluate_retrieval_relevance(self, query: str, retrieved_docs: List[str]) -> Dict[str, float]:
        """Evaluate if retrieved documents are relevant to the query."""
        if not retrieved_docs:
            return {"relevance_score": 0.0}
        
        # Calculate semantic similarity between query and each document
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = self.embeddings.embed_documents(retrieved_docs)
        
        similarities = []
        for doc_emb in doc_embeddings:
            similarity = np.dot(query_embedding, doc_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb))
            similarities.append(similarity)
        
        return {
            "avg_relevance": np.mean(similarities),
            "min_relevance": np.min(similarities),
            "max_relevance": np.max(similarities),
            "relevance_std": np.std(similarities)
        }
    
    def evaluate_answer_faithfulness(self, query: str, answer: str, retrieved_docs: List[str]) -> Dict[str, float]:
        """Evaluate if the answer is faithful to the retrieved documents."""
        if not retrieved_docs:
            return {"faithfulness_score": 0.0}
        
        # Combine retrieved documents
        context = "\n\n".join(retrieved_docs)
        
        # Use LLM to evaluate faithfulness
        prompt = f"""
        Evaluate if the following answer is faithful to the provided context. 
        The answer should only contain information that can be directly supported by the context.
        
        Query: {query}
        
        Context:
        {context}
        
        Answer: {answer}
        
        Rate the faithfulness on a scale of 0-1, where:
        0 = Answer contains information not in the context
        1 = Answer is completely faithful to the context
        
        Provide only the numerical score:
        """
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return {"faithfulness_score": score}
        except:
            return {"faithfulness_score": 0.5}  # Default score
    
    def evaluate_answer_relevance(self, query: str, answer: str) -> Dict[str, float]:
        """Evaluate if the answer is relevant to the query."""
        prompt = f"""
        Evaluate if the following answer is relevant to the query.
        
        Query: {query}
        Answer: {answer}
        
        Rate the relevance on a scale of 0-1, where:
        0 = Answer is completely irrelevant to the query
        1 = Answer directly addresses the query
        
        Provide only the numerical score:
        """
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return {"answer_relevance": score}
        except:
            return {"answer_relevance": 0.5}
    
    def evaluate_answer_completeness(self, query: str, answer: str) -> Dict[str, float]:
        """Evaluate if the answer is complete."""
        prompt = f"""
        Evaluate if the following answer is complete and comprehensive.
        
        Query: {query}
        Answer: {answer}
        
        Rate the completeness on a scale of 0-1, where:
        0 = Answer is very incomplete or vague
        1 = Answer is comprehensive and complete
        
        Provide only the numerical score:
        """
        
        try:
            response = self.llm.invoke(prompt)
            score = float(response.content.strip())
            return {"completeness_score": score}
        except:
            return {"completeness_score": 0.5}
    
    def evaluate_document_diversity(self, retrieved_docs: List[str]) -> Dict[str, float]:
        """Evaluate diversity of retrieved documents."""
        if len(retrieved_docs) < 2:
            return {"diversity_score": 0.0}
        
        # Calculate pairwise similarities between documents
        doc_embeddings = self.embeddings.embed_documents(retrieved_docs)
        
        similarities = []
        for i in range(len(doc_embeddings)):
            for j in range(i + 1, len(doc_embeddings)):
                similarity = np.dot(doc_embeddings[i], doc_embeddings[j]) / (
                    np.linalg.norm(doc_embeddings[i]) * np.linalg.norm(doc_embeddings[j])
                )
                similarities.append(similarity)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities)
        diversity_score = 1 - avg_similarity
        
        return {
            "diversity_score": diversity_score,
            "avg_doc_similarity": avg_similarity
        }
    
    def evaluate_answer_length(self, answer: str) -> Dict[str, float]:
        """Evaluate answer length appropriateness."""
        word_count = len(answer.split())
        
        # Score based on length (not too short, not too long)
        if word_count < 10:
            length_score = 0.3
        elif word_count < 50:
            length_score = 0.8
        elif word_count < 200:
            length_score = 1.0
        elif word_count < 500:
            length_score = 0.7
        else:
            length_score = 0.4
        
        return {
            "length_score": length_score,
            "word_count": word_count
        }
    
    def comprehensive_self_evaluation(self, 
                                    queries: List[str],
                                    retrieved_docs_list: List[List[str]],
                                    generated_answers: List[str]) -> Dict[str, Any]:
        """Run comprehensive self-evaluation without reference answers."""
        
        all_scores = []
        
        for i, query in enumerate(queries):
            click.echo(f"Evaluating query {i+1}/{len(queries)}")
            
            scores = {}
            
            # Retrieval evaluation
            retrieval_scores = self.evaluate_retrieval_relevance(query, retrieved_docs_list[i])
            scores.update(retrieval_scores)
            
            # Answer evaluation
            faithfulness = self.evaluate_answer_faithfulness(query, generated_answers[i], retrieved_docs_list[i])
            relevance = self.evaluate_answer_relevance(query, generated_answers[i])
            completeness = self.evaluate_answer_completeness(query, generated_answers[i])
            length = self.evaluate_answer_length(generated_answers[i])
            
            scores.update(faithfulness)
            scores.update(relevance)
            scores.update(completeness)
            scores.update(length)
            
            # Document diversity
            diversity = self.evaluate_document_diversity(retrieved_docs_list[i])
            scores.update(diversity)
            
            all_scores.append(scores)
        
        # Aggregate scores
        aggregated_scores = {}
        for key in all_scores[0].keys():
            values = [score[key] for score in all_scores]
            aggregated_scores[f'avg_{key}'] = np.mean(values)
            aggregated_scores[f'std_{key}'] = np.std(values)
        
        return {
            'aggregated_scores': aggregated_scores,
            'detailed_scores': all_scores
        } 