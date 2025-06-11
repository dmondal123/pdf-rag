from PyPDF2 import PdfReader
from typing import List
import re
import tiktoken

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    # Use tiktoken to count tokens
    enc = tiktoken.get_encoding("cl100k_base")  # for OpenAI 3 models
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks 