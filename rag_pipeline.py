from typing import List, Dict
from document_processor import DocumentProcessor
from embedding_generator import EmbeddingGenerator
from vector_store import VectorStore
from llm_interface import LLMInterface
from config import Config
import os

class RAGPipeline:
    """End-to-end RAG pipeline."""
    
    def __init__(self):
        self.doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        self.embedding_generator = EmbeddingGenerator(Config.EMBEDDING_MODEL)
        self.vector_store = VectorStore(self.embedding_generator.embedding_dim)
        self.llm_interface = LLMInterface()
        self.is_indexed = False
    
    def build_knowledge_base(self, documents_directory: str):
        """Build the knowledge base from documents."""
        print("Loading documents...")
        documents = self.doc_processor.load_documents(documents_directory)
        print(f"Loaded {len(documents)} documents")
        
        print("Processing documents into chunks...")
        chunks = self.doc_processor.process_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        print("Generating embeddings...")
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        print("Building vector store...")
        self.vector_store.add_embeddings(embeddings, chunks)
        
        # Save the vector store
        vector_store_path = os.path.join(Config.VECTOR_DB_PATH, 'vector_store')
        self.vector_store.save(vector_store_path)
        print(f"Vector store saved to {vector_store_path}")
        
        self.is_indexed = True
        return len(chunks)
    
    def load_knowledge_base(self):
        """Load existing knowledge base."""
        vector_store_path = os.path.join(Config.VECTOR_DB_PATH, 'vector_store')
        
        if os.path.exists(f"{vector_store_path}.faiss"):
            self.vector_store.load(vector_store_path)
            self.is_indexed = True
            print(f"Loaded vector store with {len(self.vector_store.metadata)} chunks")
            return True
        else:
            print("No existing vector store found")
            return False
    
    def query(self, question: str, top_k: int = None) -> Dict:
        """Query the RAG system."""
        if not self.is_indexed:
            return {
                'error': 'Knowledge base not built. Please run build_knowledge_base() first.'
            }
        
        top_k = top_k or Config.TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(question)
        
        # Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(query_embedding, k=top_k)
        
        # Generate grounded response
        result = self.llm_interface.generate_grounded_response(question, retrieved_chunks)
        
        return result
    
    def compare_responses(self, question: str) -> Dict:
        """Compare grounded vs ungrounded responses."""
        # Get grounded response
        grounded_result = self.query(question)
        
        # Get ungrounded response (without context)
        ungrounded_prompt = f"Question: {question}\n\nAnswer:"
        ungrounded_response = self.llm_interface.generate_response(ungrounded_prompt)
        
        return {
            'question': question,
            'grounded_response': grounded_result.get('response', 'Error'),
            'ungrounded_response': ungrounded_response,
            'sources': grounded_result.get('sources', [])
        }