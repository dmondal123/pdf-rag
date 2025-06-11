# PDF RAG CLI

A command-line tool for uploading PDFs, extracting text, chunking, embedding, and querying using pgvector and Anthropic Claude Sonnet via LangChain.

## Features
- PDF text extraction (PyPDF2)
- Document chunking and embedding
- pgvector database integration (Docker)
- Anthropic Claude Sonnet integration (LangChain)
- Agentic querying with memory and reasoning

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start pgvector database (Docker):**
   ```bash
   docker-compose up -d
   ```

3. **Configure environment variables:**
   - Create a `.env` file with your Anthropic API key and database credentials.

4. **Run the CLI:**
   ```bash
   python main.py
   ```

## Usage
- Upload PDFs and query their content via the CLI interface. 