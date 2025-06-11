import click

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
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    click.echo(f"Extracted {len(chunks)} chunks from PDF.")
    click.echo("Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    click.echo("Storing in database...")
    upsert_chunks_with_embeddings(chunks, embeddings, source=os.path.basename(pdf_path))
    click.echo("Upload complete.")

@cli.command()
def ask():
    """Enter interactive Q&A mode with conversational memory."""
    from query_utils import get_top_k_chunks, generate_answer
    global conversation_history
    click.echo("Entering interactive Q&A mode. Type 'exit' or 'quit' to leave.")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit"}:
            click.echo("Exiting Q&A mode.")
            break
        click.echo(f"Querying: {query}")
        top_chunks = get_top_k_chunks(query, k=5)
        context_chunks = [chunk for chunk, _ in top_chunks]
        history_text = ""
        for q, a in conversation_history:
            history_text += f"Q: {q}\nA: {a}\n"
        answer = generate_answer(query, context_chunks, history_text)
        click.echo("\nAnswer:\n" + answer)
        conversation_history.append((query, answer))

if __name__ == "__main__":
    cli() 