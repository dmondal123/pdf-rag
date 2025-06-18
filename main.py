import click
from typing import List, Tuple, Dict
from multimodal_utils import MultimodalProcessor, chunk_multimodal_content
from evaluation_utils import RAGEvaluator
from self_evaluation_utils import SelfRAGEvaluator

conversation_history = []

@click.group()
def cli():
    """PDF RAG CLI: Upload PDFs and query their content."""
    pass

@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
def upload(pdf_path):
    """Upload a PDF, extract, chunk, embed, and store in DB."""
    from pdf_utils import extract_text_from_pdf, chunk_text
    from embed_utils import generate_embeddings, upsert_chunks_with_embeddings
    from tqdm import tqdm
    import os
    
    click.echo(f"Uploading and processing: {pdf_path}")
    
    # Process multimodal content
    processor = MultimodalProcessor()
    multimodal_content = processor.process_pdf(pdf_path)
    
    # Add regular text content
    text = extract_text_from_pdf(pdf_path)
    text_chunks = chunk_text(text)
    for chunk in text_chunks:
        multimodal_content.append({
            'type': 'text',
            'content': chunk,
            'metadata': {'content_type': 'text'}
        })
    
    # Chunk the content while preserving structure
    chunked_content = chunk_multimodal_content(multimodal_content)
    
    click.echo(f"Extracted {len(chunked_content)} chunks from PDF.")
    click.echo("Generating embeddings...")
    
    # Generate embeddings for all content
    content_texts = [item['content'] for item in chunked_content]
    embeddings = generate_embeddings(content_texts)
    
    click.echo("Storing in database...")
    upsert_chunks_with_embeddings(chunked_content, embeddings, source=os.path.basename(pdf_path))
    click.echo("Upload complete.")

@cli.command()
def ask():
    """Enter interactive Q&A mode with conversational memory and detailed scoring."""
    from query_utils import get_top_k_chunks, generate_answer
    global conversation_history
    click.echo("Entering interactive Q&A mode. Type 'exit' or 'quit' to leave.")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit"}:
            click.echo("Exiting Q&A mode.")
            break
            
        click.echo(f"\nQuerying: {query}")
        results = get_top_k_chunks(query, k=5)
        
        # Display scoring information
        click.echo("\nRetrieved chunks with scores:")
        for i, (chunk, scores) in enumerate(results, 1):
            click.echo(f"\nChunk {i}:")
            click.echo(f"Content Type: {chunk.get('type', 'text')}")
            click.echo(f"Cosine Similarity: {scores['cosine_similarity']:.3f}")
            click.echo(f"Word Overlap: {scores['word_overlap']:.3f}")
            click.echo(f"Keyword Density: {scores['keyword_density']:.3f}")
            click.echo(f"Combined Score: {scores['combined_score']:.3f}")
            click.echo(f"Content: {chunk[:200]}...")  # Show first 200 chars
        
        # Generate answer using the chunks
        context_chunks = [chunk for chunk, _ in results]
        history_text = ""
        for q, a in conversation_history:
            history_text += f"Q: {q}\nA: {a}\n"
        
        answer = generate_answer(query, context_chunks, history_text)
        click.echo("\nAnswer:\n" + answer)
        conversation_history.append((query, answer))

@cli.command()
@click.option('--test-file', type=click.Path(exists=True), help='Path to test data file (JSON)')
@click.option('--output-file', type=click.Path(), help='Path to save evaluation results')
def evaluate(test_file, output_file):
    """Evaluate RAG system performance using test data."""
    import json
    from query_utils import get_top_k_chunks, generate_answer
    
    if not test_file:
        click.echo("Please provide a test file with --test-file option")
        return
    
    # Load test data
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    evaluator = RAGEvaluator()
    
    # Extract test data
    queries = test_data['queries']
    reference_answers = test_data['reference_answers']
    relevant_docs_list = test_data['relevant_docs']
    
    # Run evaluation
    click.echo("Running evaluation...")
    
    retrieved_docs_list = []
    generated_answers = []
    
    for i, query in enumerate(queries):
        click.echo(f"Processing query {i+1}/{len(queries)}: {query}")
        
        # Get retrieved documents
        results = get_top_k_chunks(query, k=10)
        retrieved_docs = [chunk for chunk, _ in results]
        retrieved_docs_list.append(retrieved_docs)
        
        # Generate answer
        context_chunks = retrieved_docs[:5]  # Use top 5 for generation
        answer = generate_answer(query, context_chunks, "")
        generated_answers.append(answer)
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.evaluate_rag_system(
        queries, reference_answers, retrieved_docs_list, 
        relevant_docs_list, generated_answers
    )
    
    # Display results
    click.echo("\n=== EVALUATION RESULTS ===")
    
    click.echo("\nRetrieval Metrics:")
    for metric, value in evaluation_results['retrieval_metrics'].items():
        if metric.startswith('avg_'):
            click.echo(f"{metric[4:]}: {value:.3f}")
    
    click.echo("\nGeneration Metrics:")
    for metric, value in evaluation_results['generation_metrics'].items():
        if metric.startswith('avg_'):
            click.echo(f"{metric[4:]}: {value:.3f}")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        click.echo(f"\nResults saved to {output_file}")

@cli.command()
@click.option('--queries-file', type=click.Path(exists=True), help='Path to queries file (JSON)')
@click.option('--output-file', type=click.Path(), help='Path to save evaluation results')
def self_evaluate(queries_file, output_file):
    """Self-evaluate RAG system without reference answers."""
    import json
    from query_utils import get_top_k_chunks, generate_answer
    
    if not queries_file:
        click.echo("Please provide a queries file with --queries-file option")
        return
    
    # Load queries
    with open(queries_file, 'r') as f:
        data = json.load(f)
    
    queries = data['queries']
    
    # Initialize evaluator
    evaluator = SelfRAGEvaluator()
    
    # Run evaluation
    click.echo("Running self-evaluation...")
    
    retrieved_docs_list = []
    generated_answers = []
    
    for i, query in enumerate(queries):
        click.echo(f"Processing query {i+1}/{len(queries)}: {query}")
        
        # Get retrieved documents
        results = get_top_k_chunks(query, k=10)
        retrieved_docs = [chunk for chunk, _ in results]
        retrieved_docs_list.append(retrieved_docs)
        
        # Generate answer
        context_chunks = retrieved_docs[:5]  # Use top 5 for generation
        answer = generate_answer(query, context_chunks, "")
        generated_answers.append(answer)
    
    # Run self-evaluation
    evaluation_results = evaluator.comprehensive_self_evaluation(
        queries, retrieved_docs_list, generated_answers
    )
    
    # Display results
    click.echo("\n=== SELF-EVALUATION RESULTS ===")
    
    click.echo("\nAggregated Scores:")
    for metric, value in evaluation_results['aggregated_scores'].items():
        if metric.startswith('avg_'):
            click.echo(f"{metric[4:]}: {value:.3f}")
    
    # Save results if output file specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        click.echo(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    cli() 