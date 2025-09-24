import os
import re
from typing import List, Dict
from pathlib import Path
import PyPDF2

class DocumentProcessor:
    """Handles document collection, cleaning, and preprocessing."""
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory_path: str) -> List[Dict[str, str]]:
        """Load documents from a directory."""
        documents = []
        directory = Path(directory_path)
        
        # Process text files
        for file_path in directory.rglob('*.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'title': file_path.stem
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Process PDF files
        for file_path in directory.rglob('*.pdf'):
            try:
                content = self.extract_text_from_pdf(file_path)
                if content.strip():  # Only add if content is not empty
                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'title': file_path.stem
                    })
            except Exception as e:
                print(f"Error loading PDF {file_path}: {e}")
        
        # Process Markdown files
        for file_path in directory.rglob('*.md'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'title': file_path.stem
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text content from PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                    
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str, metadata: Dict[str, str]) -> List[Dict[str, str]]:
        """Split text into chunks using sentence-based approach."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'content': self.clean_text(current_chunk),
                        'source': metadata['source'],
                        'title': metadata['title'],
                        'chunk_id': len(chunks)
                    })
                
                # Start new chunk with overlap
                if len(chunks) > 0 and self.chunk_overlap > 0:
                    # Get last few words for overlap
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap:] if len(words) > self.chunk_overlap else words
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'content': self.clean_text(current_chunk),
                'source': metadata['source'],
                'title': metadata['title'],
                'chunk_id': len(chunks)
            })
        
        return chunks
    
    def process_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Process documents into chunks."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc['content'], {
                'source': doc['source'],
                'title': doc['title']
            })
            all_chunks.extend(chunks)
        
        return all_chunks