import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os

class VectorStore:
    """FAISS-based vector store for similarity search."""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        self.metadata = []
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings and metadata to the vector store."""
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
        self.metadata.extend(metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar embeddings."""
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # Valid result
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'embedding_dim': self.embedding_dim
            }, f)
    
    def load(self, filepath: str):
        """Load the vector store from disk."""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}.metadata", 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.embedding_dim = data['embedding_dim']