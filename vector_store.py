import asyncio
import logging
import pickle
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# Vector database libraries
import faiss
from sentence_transformers import SentenceTransformer

# Optional Pinecone support
try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    pinecone = None

from models import DocumentChunk, EmbeddingVector, RetrievalResult
from config import settings

logger = logging.getLogger(__name__)

class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the vector store"""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for similar chunks"""
        pass
    
    @abstractmethod
    async def delete_document(self, document_url: str) -> None:
        """Delete all chunks from a specific document"""
        pass

class EmbeddingModel:
    """Handles text embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded model with dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {str(e)}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text to embedding"""
        return self.encode([text])[0]

class FAISSVectorStore(VectorStore):
    """FAISS-based vector store implementation"""
    
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.dimension = settings.embedding_dimension
        self.index = None
        self.chunks_metadata = {}  # Store chunk metadata by index
        self.index_path = settings.faiss_index_path
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        try:
            # Try to load existing index
            if os.path.exists(f"{self.index_path}.index"):
                logger.info("Loading existing FAISS index")
                self.index = faiss.read_index(f"{self.index_path}.index")
                
                # Load metadata
                if os.path.exists(f"{self.index_path}_metadata.pkl"):
                    with open(f"{self.index_path}_metadata.pkl", 'rb') as f:
                        self.chunks_metadata = pickle.load(f)
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors")
            else:
                # Create new index
                logger.info("Creating new FAISS index")
                # Use IndexFlatIP for cosine similarity
                self.index = faiss.IndexFlatIP(self.dimension)
                self.chunks_metadata = {}
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            # Fallback to new index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks_metadata = {}
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to FAISS index"""
        try:
            if not chunks:
                return
            
            logger.info(f"Adding {len(chunks)} chunks to FAISS index")
            
            # Extract texts for embedding
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            start_idx = self.index.ntotal
            self.index.add(embeddings.astype(np.float32))
            
            # Store metadata
            for i, chunk in enumerate(chunks):
                idx = start_idx + i
                self.chunks_metadata[idx] = chunk.dict()
            
            # Save index and metadata
            await self._save_index()
            
            logger.info(f"Successfully added chunks. Total vectors: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to FAISS: {str(e)}")
            raise
    
    async def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for similar chunks using FAISS"""
        try:
            if self.index.ntotal == 0:
                logger.warning("Index is empty, no results to return")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode_single(query)
            query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            k = min(top_k, self.index.ntotal)
            similarities, indices = self.index.search(query_embedding, k)
            
            # Convert results
            results = []
            for rank, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                chunk_data = self.chunks_metadata.get(idx)
                if chunk_data:
                    chunk = DocumentChunk(**chunk_data)
                    result = RetrievalResult(
                        chunk=chunk,
                        similarity_score=float(similarity),
                        rank=rank + 1
                    )
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search FAISS index: {str(e)}")
            raise
    
    async def delete_document(self, document_url: str) -> None:
        """Delete all chunks from a specific document"""
        try:
            # Find indices to remove
            indices_to_remove = []
            for idx, chunk_data in self.chunks_metadata.items():
                if chunk_data.get('document_url') == document_url:
                    indices_to_remove.append(idx)
            
            if not indices_to_remove:
                logger.info(f"No chunks found for document: {document_url}")
                return
            
            # FAISS doesn't support direct deletion, so we need to rebuild
            logger.info(f"Rebuilding index after removing {len(indices_to_remove)} chunks")
            
            # Get all embeddings except those to remove
            all_embeddings = []
            new_metadata = {}
            new_idx = 0
            
            for old_idx in range(self.index.ntotal):
                if old_idx not in indices_to_remove:
                    # Get embedding from index
                    embedding = self.index.reconstruct(old_idx)
                    all_embeddings.append(embedding)
                    
                    # Update metadata
                    if old_idx in self.chunks_metadata:
                        new_metadata[new_idx] = self.chunks_metadata[old_idx]
                        new_idx += 1
            
            # Rebuild index
            if all_embeddings:
                embeddings_array = np.array(all_embeddings).astype(np.float32)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(embeddings_array)
            else:
                self.index = faiss.IndexFlatIP(self.dimension)
            
            self.chunks_metadata = new_metadata
            await self._save_index()
            
            logger.info(f"Successfully removed chunks for document: {document_url}")
            
        except Exception as e:
            logger.error(f"Failed to delete document chunks: {str(e)}")
            raise
    
    async def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else '.', exist_ok=True)
            
            # Save index
            faiss.write_index(self.index, f"{self.index_path}.index")
            
            # Save metadata
            with open(f"{self.index_path}_metadata.pkl", 'wb') as f:
                pickle.dump(self.chunks_metadata, f)
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {str(e)}")
            raise

class PineconeVectorStore(VectorStore):
    """Pinecone-based vector store implementation"""
    
    def __init__(self):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not available. Install with: pip install pinecone-client")
        
        self.embedding_model = EmbeddingModel()
        self.dimension = settings.embedding_dimension
        self.index_name = settings.pinecone_index_name
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            if not settings.pinecone_api_key:
                raise ValueError("Pinecone API key not provided")
            
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Check if index exists, create if not
            existing_indexes = pinecone.list_indexes()
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine"
                )
            
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to Pinecone index"""
        try:
            if not chunks:
                return
            
            logger.info(f"Adding {len(chunks)} chunks to Pinecone index")
            
            # Extract texts for embedding
            texts = [chunk.content for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts)
            
            # Prepare vectors for Pinecone
            vectors = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector = {
                    'id': chunk.chunk_id,
                    'values': embedding.tolist(),
                    'metadata': {
                        'content': chunk.content,
                        'document_url': chunk.document_url,
                        'chunk_number': chunk.metadata.get('chunk_number', i),
                        'character_count': chunk.metadata.get('character_count', len(chunk.content)),
                        'word_count': chunk.metadata.get('word_count', len(chunk.content.split())),
                    }
                }
                vectors.append(vector)
            
            # Upsert to Pinecone in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Successfully added {len(chunks)} chunks to Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to add chunks to Pinecone: {str(e)}")
            raise
    
    async def search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Search for similar chunks using Pinecone"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode_single(query)
            
            # Search Pinecone
            search_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            # Convert results
            results = []
            for rank, match in enumerate(search_results['matches']):
                metadata = match['metadata']
                
                chunk = DocumentChunk(
                    content=metadata['content'],
                    chunk_id=match['id'],
                    document_url=metadata['document_url'],
                    metadata={
                        'chunk_number': metadata.get('chunk_number', 0),
                        'character_count': metadata.get('character_count', 0),
                        'word_count': metadata.get('word_count', 0),
                    }
                )
                
                result = RetrievalResult(
                    chunk=chunk,
                    similarity_score=float(match['score']),
                    rank=rank + 1
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search Pinecone index: {str(e)}")
            raise
    
    async def delete_document(self, document_url: str) -> None:
        """Delete all chunks from a specific document"""
        try:
            # First, find all chunk IDs for this document
            # Note: This is a simplified approach. In production, you might want to maintain
            # a separate mapping of documents to chunk IDs
            logger.info(f"Deleting chunks for document: {document_url}")
            
            # Query to find all chunks for this document
            # This is a workaround since Pinecone doesn't have a direct way to query by metadata
            dummy_query = [0.0] * self.dimension
            search_results = self.index.query(
                vector=dummy_query,
                filter={"document_url": document_url},
                top_k=10000,  # Large number to get all results
                include_metadata=False
            )
            
            # Extract IDs to delete
            ids_to_delete = [match['id'] for match in search_results['matches']]
            
            if ids_to_delete:
                self.index.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for document: {document_url}")
            else:
                logger.info(f"No chunks found for document: {document_url}")
                
        except Exception as e:
            logger.error(f"Failed to delete document chunks from Pinecone: {str(e)}")
            raise

def create_vector_store() -> VectorStore:
    """Factory function to create the appropriate vector store"""
    if settings.use_faiss:
        logger.info("Using FAISS vector store")
        return FAISSVectorStore()
    elif settings.pinecone_api_key and PINECONE_AVAILABLE:
        logger.info("Using Pinecone vector store")
        return PineconeVectorStore()
    else:
        logger.info("Falling back to FAISS vector store")
        return FAISSVectorStore()