import os
from typing import List, Dict, Any, Tuple
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import cv2
import numpy as np
from PIL import Image
import tabula
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class MultimodalProcessor:
    """Process different types of content from PDFs."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    
    def extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using tabula-py."""
        try:
            # Extract all tables from the PDF
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            # Convert tables to text with structure preserved
            table_texts = []
            for i, table in enumerate(tables):
                # Convert table to markdown format
                markdown_table = table.to_markdown(index=False)
                table_texts.append({
                    'type': 'table',
                    'content': markdown_table,
                    'table_number': i + 1
                })
            
            return table_texts
        except Exception as e:
            print(f"Error extracting tables: {e}")
            return []

    def extract_images(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract and process images from PDF."""
        try:
            # Convert PDF pages to images
            images = convert_from_path(pdf_path)
            
            image_texts = []
            for i, image in enumerate(images):
                # Convert PIL image to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Extract text from image using OCR
                text = pytesseract.image_to_string(cv_image)
                
                # Get image description using OpenAI Vision API
                # Note: This requires additional implementation
                
                image_texts.append({
                    'type': 'image',
                    'content': text,
                    'image_number': i + 1
                })
            
            return image_texts
        except Exception as e:
            print(f"Error extracting images: {e}")
            return []

    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process PDF and extract all types of content."""
        # Extract tables
        tables = self.extract_tables(pdf_path)
        
        # Extract images and their text
        images = self.extract_images(pdf_path)
        
        # Combine all content with metadata
        all_content = []
        
        # Add tables
        for table in tables:
            all_content.append({
                'type': 'table',
                'content': table['content'],
                'metadata': {
                    'table_number': table['table_number'],
                    'content_type': 'table'
                }
            })
        
        # Add images
        for image in images:
            all_content.append({
                'type': 'image',
                'content': image['content'],
                'metadata': {
                    'image_number': image['image_number'],
                    'content_type': 'image'
                }
            })
        
        return all_content

    def generate_embeddings(self, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for all content types."""
        for item in content_list:
            # Generate embedding for the content
            embedding = self.embeddings.embed_query(item['content'])
            item['embedding'] = embedding
        
        return content_list

def chunk_multimodal_content(content_list: List[Dict[str, Any]], chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """Chunk multimodal content while preserving structure."""
    chunked_content = []
    
    for item in content_list:
        if item['type'] == 'table':
            # Tables are kept as single chunks to preserve structure
            chunked_content.append(item)
        else:
            # For text content (including OCR text from images), split into chunks
            text = item['content']
            words = text.split()
            
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                chunked_content.append({
                    'type': item['type'],
                    'content': chunk,
                    'metadata': {
                        **item['metadata'],
                        'chunk_number': i // chunk_size + 1
                    }
                })
    
    return chunked_content 