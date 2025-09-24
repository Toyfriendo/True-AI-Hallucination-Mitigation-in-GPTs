import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pickle
import os

class EmbeddingGenerator:
    """Handles embedding generation using sentence transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query."""
        return self.model.encode([query])[0]
    
    def save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict], filepath: str):
        """Save embeddings and metadata to disk."""
        data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_embeddings(self, filepath: str) -> tuple:
        """Load embeddings and metadata from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        return data['embeddings'], data['metadata']