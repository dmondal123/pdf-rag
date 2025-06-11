import click

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
@click.argument('query', type=str)
def ask(query):
    """Query the knowledge base using agentic reasoning."""
    from query_utils import get_top_k_chunks, generate_answer
    click.echo(f"Querying: {query}")
    top_chunks = get_top_k_chunks(query, k=5)
    context_chunks = [chunk for chunk, _ in top_chunks]
    answer = generate_answer(query, context_chunks)
    click.echo("\nAnswer:\n" + answer)

if __name__ == "__main__":
    cli() 