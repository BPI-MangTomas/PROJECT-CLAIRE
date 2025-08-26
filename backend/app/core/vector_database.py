import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from pathlib import Path
import pickle
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index = None
        self.documents = []
        self.dimension = None
        self.index_path = Path(settings.VECTOR_STORE_PATH) / "faiss_index.bin"
        self.docs_path = Path(settings.VECTOR_STORE_PATH) / "documents.pkl"
        
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build FAISS index from documents"""
        logger.info(f"Building FAISS index for {len(documents)} documents")
        self.documents = documents
        
        # Extract text for embedding
        texts = [doc['content'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create FAISS index
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save index and documents
        self.save_index()
        logger.info(f"Index built successfully. Dimension: {self.dimension}")
        
    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Search for most relevant documents"""
        if self.index is None:
            self.load_index()
            
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
                
        return results
        
    def save_index(self):
        """Save FAISS index and documents to disk"""
        Path(settings.VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            logger.info(f"Index saved to {self.index_path}")
            
    def load_index(self):
        """Load FAISS index and documents from disk"""
        if self.index_path.exists() and self.docs_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            logger.info(f"Index loaded from {self.index_path}")
        else:
            raise FileNotFoundError("Vector index not found. Please build the index first.")