import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RAGEvaluator:
    """Evaluation metrics for RAG systems."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.smoothie = SmoothingFunction().method1
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing for evaluation."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def _get_ngrams(self, text: str, n: int) -> Counter:
        """Get n-grams from text."""
        words = word_tokenize(self._preprocess_text(text))
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(tuple(words[i:i + n]))
        return Counter(ngrams)
    
    def rouge_1(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE-1 scores (unigram overlap)."""
        ref_ngrams = self._get_ngrams(reference, 1)
        cand_ngrams = self._get_ngrams(candidate, 1)
        
        # Calculate overlap
        overlap = sum((ref_ngrams & cand_ngrams).values())
        ref_count = sum(ref_ngrams.values())
        cand_count = sum(cand_ngrams.values())
        
        # Calculate precision, recall, and F1
        precision = overlap / cand_count if cand_count > 0 else 0
        recall = overlap / ref_count if ref_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'rouge-1_precision': precision,
            'rouge-1_recall': recall,
            'rouge-1_f1': f1
        }
    
    def rouge_2(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE-2 scores (bigram overlap)."""
        ref_ngrams = self._get_ngrams(reference, 2)
        cand_ngrams = self._get_ngrams(candidate, 2)
        
        # Calculate overlap
        overlap = sum((ref_ngrams & cand_ngrams).values())
        ref_count = sum(ref_ngrams.values())
        cand_count = sum(cand_ngrams.values())
        
        # Calculate precision, recall, and F1
        precision = overlap / cand_count if cand_count > 0 else 0
        recall = overlap / ref_count if ref_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'rouge-2_precision': precision,
            'rouge-2_recall': recall,
            'rouge-2_f1': f1
        }
    
    def rouge_l(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE-L scores (longest common subsequence)."""
        ref_words = word_tokenize(self._preprocess_text(reference))
        cand_words = word_tokenize(self._preprocess_text(candidate))
        
        # Calculate LCS length
        lcs_length = self._longest_common_subsequence(ref_words, cand_words)
        
        # Calculate precision, recall, and F1
        precision = lcs_length / len(cand_words) if len(cand_words) > 0 else 0
        recall = lcs_length / len(ref_words) if len(ref_words) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'rouge-l_precision': precision,
            'rouge-l_recall': recall,
            'rouge-l_f1': f1
        }
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate the length of the longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def bleu_score(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BLEU score."""
        ref_words = word_tokenize(self._preprocess_text(reference))
        cand_words = word_tokenize(self._preprocess_text(candidate))
        
        # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
        bleu_scores = {}
        for n in range(1, 5):
            try:
                score = sentence_bleu([ref_words], cand_words, 
                                    weights=tuple([1.0/n] * n),
                                    smoothing_function=self.smoothie)
                bleu_scores[f'bleu-{n}'] = score
            except:
                bleu_scores[f'bleu-{n}'] = 0.0
        
        return bleu_scores
    
    def semantic_similarity(self, reference: str, candidate: str) -> float:
        """Calculate semantic similarity using embeddings."""
        try:
            ref_embedding = self.embeddings.embed_query(reference)
            cand_embedding = self.embeddings.embed_query(candidate)
            
            similarity = cosine_similarity([ref_embedding], [cand_embedding])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def precision_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K."""
        if k == 0:
            return 0.0
        
        # Get top-k retrieved documents
        top_k_docs = retrieved_docs[:k]
        
        # Count relevant documents in top-k
        relevant_in_top_k = sum(1 for doc in top_k_docs if doc in relevant_docs)
        
        return relevant_in_top_k / k
    
    def recall_at_k(self, retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_docs:
            return 0.0
        
        # Get top-k retrieved documents
        top_k_docs = retrieved_docs[:k]
        
        # Count relevant documents in top-k
        relevant_in_top_k = sum(1 for doc in top_k_docs if doc in relevant_docs)
        
        return relevant_in_top_k / len(relevant_docs)
    
    def mean_reciprocal_rank(self, retrieved_docs: List[str], relevant_docs: List[str]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)."""
        if not relevant_docs:
            return 0.0
        
        for i, doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def evaluate_retrieval(self, 
                          query: str,
                          retrieved_docs: List[str], 
                          relevant_docs: List[str],
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """Evaluate retrieval performance."""
        results = {}
        
        # Calculate precision and recall at different k values
        for k in k_values:
            results[f'precision@{k}'] = self.precision_at_k(retrieved_docs, relevant_docs, k)
            results[f'recall@{k}'] = self.recall_at_k(retrieved_docs, relevant_docs, k)
        
        # Calculate MRR
        results['mrr'] = self.mean_reciprocal_rank(retrieved_docs, relevant_docs)
        
        return results
    
    def evaluate_generation(self, 
                           reference_answer: str, 
                           generated_answer: str) -> Dict[str, float]:
        """Evaluate answer generation quality."""
        results = {}
        
        # Calculate ROUGE scores
        rouge_1_scores = self.rouge_1(reference_answer, generated_answer)
        rouge_2_scores = self.rouge_2(reference_answer, generated_answer)
        rouge_l_scores = self.rouge_l(reference_answer, generated_answer)
        
        results.update(rouge_1_scores)
        results.update(rouge_2_scores)
        results.update(rouge_l_scores)
        
        # Calculate BLEU scores
        bleu_scores = self.bleu_score(reference_answer, generated_answer)
        results.update(bleu_scores)
        
        # Calculate semantic similarity
        results['semantic_similarity'] = self.semantic_similarity(reference_answer, generated_answer)
        
        return results
    
    def evaluate_rag_system(self,
                           queries: List[str],
                           reference_answers: List[str],
                           retrieved_docs_list: List[List[str]],
                           relevant_docs_list: List[List[str]],
                           generated_answers: List[str]) -> Dict[str, Any]:
        """Comprehensive evaluation of RAG system."""
        
        retrieval_scores = []
        generation_scores = []
        
        for i, query in enumerate(queries):
            # Evaluate retrieval
            retrieval_score = self.evaluate_retrieval(
                query, 
                retrieved_docs_list[i], 
                relevant_docs_list[i]
            )
            retrieval_scores.append(retrieval_score)
            
            # Evaluate generation
            generation_score = self.evaluate_generation(
                reference_answers[i], 
                generated_answers[i]
            )
            generation_scores.append(generation_score)
        
        # Aggregate scores
        aggregated_retrieval = {}
        aggregated_generation = {}
        
        # Average retrieval scores
        for key in retrieval_scores[0].keys():
            aggregated_retrieval[f'avg_{key}'] = np.mean([score[key] for score in retrieval_scores])
            aggregated_retrieval[f'std_{key}'] = np.std([score[key] for score in retrieval_scores])
        
        # Average generation scores
        for key in generation_scores[0].keys():
            aggregated_generation[f'avg_{key}'] = np.mean([score[key] for score in generation_scores])
            aggregated_generation[f'std_{key}'] = np.std([score[key] for score in generation_scores])
        
        return {
            'retrieval_metrics': aggregated_retrieval,
            'generation_metrics': aggregated_generation,
            'detailed_scores': {
                'retrieval': retrieval_scores,
                'generation': generation_scores
            }
        } 