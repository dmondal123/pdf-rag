from PyPDF2 import PdfReader
from typing import List
import re

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Chunk text into overlapping segments of approximately chunk_size characters."""
    # Split by paragraphs for better semantic chunks
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += (para + "\n\n")
        else:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            current_chunk = current_chunk[-overlap:] + para + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [c for c in chunks if c.strip()] 