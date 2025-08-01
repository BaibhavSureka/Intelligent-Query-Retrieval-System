import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim: int):
        # Use IndexFlatIP for cosine similarity (higher scores = more similar)
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []
        self.embeddings = []

    def add(self, embeddings: list, chunks: list):
        """Add embeddings and chunks to the vector store"""
        if len(embeddings) != len(chunks):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(chunks)} chunks")
        
        # Normalize embeddings for cosine similarity
        arr = np.array(embeddings).astype("float32")
        faiss.normalize_L2(arr)  # Normalize for cosine similarity
        
        self.index.add(arr)
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)
        
        print(f"Added {len(embeddings)} embeddings to vector store. Total: {len(self.chunks)}")

    def search(self, query_emb: list, top_k: int = 5):
        """Search for similar chunks using cosine similarity"""
        if len(self.chunks) == 0:
            print("WARNING: Vector store is empty!")
            return []
        
        # Normalize query embedding for cosine similarity
        query_arr = np.array([query_emb]).astype("float32")
        faiss.normalize_L2(query_arr)
        
        # Search with higher k to allow for filtering
        search_k = min(top_k * 2, len(self.chunks))
        D, I = self.index.search(query_arr, search_k)
        
        results = []
        similarity_threshold = 0.5  # Minimum cosine similarity
        
        for idx, i in enumerate(I[0]):
            if 0 <= i < len(self.chunks):
                similarity_score = float(D[0][idx])  # Higher is better with IndexFlatIP
                if similarity_score >= similarity_threshold:
                    results.append((self.chunks[i], similarity_score))
        
        # Sort by similarity (highest first) and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"Retrieved {len(results)} chunks with similarity >= {similarity_threshold}")
        
        return results[:top_k]
    
    def get_stats(self):
        """Get statistics about the vector store"""
        return {
            "total_chunks": len(self.chunks),
            "index_size": self.index.ntotal,
            "dimension": self.index.d
        }