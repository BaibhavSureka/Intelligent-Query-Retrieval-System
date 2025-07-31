from .embedder import get_embedding
import re

def extract_keywords(query: str) -> list:
    """Extract important keywords from query for better matching"""
    # Remove common words and extract key terms
    stop_words = {'what', 'is', 'the', 'for', 'in', 'of', 'to', 'and', 'or', 'with', 'under', 'this', 'that', 'are', 'does', 'do', 'how', 'when', 'where', 'why', 'which', 'who', 'can', 'will', 'should', 'would', 'could', 'may', 'might', 'must', 'shall'}
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords

def retrieve_relevant_chunks(query: str, vector_store, top_k=12):
    """Retrieve relevant chunks from vector store based on query"""
    query_emb = get_embedding(query)
    results = vector_store.search(query_emb, top_k=top_k)
    
    # Get chunks and scores
    chunks_with_scores = [(chunk, score) for chunk, score in results]
    
    # Enhanced keyword-based filtering for better relevance
    keywords = extract_keywords(query)
    if keywords:
        # Boost chunks that contain query keywords
        filtered_chunks = []
        for chunk, score in chunks_with_scores:
            chunk_lower = chunk.lower()
            keyword_matches = sum(1 for keyword in keywords if keyword in chunk_lower)
            
            if keyword_matches > 0:
                # Enhanced boosting for general documents
                boosted_score = score + (keyword_matches * 0.2)
                filtered_chunks.append((chunk, boosted_score))
        
        # Sort by boosted scores and return top chunks
        filtered_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in filtered_chunks[:top_k]]
    
    return [chunk for chunk, _ in chunks_with_scores]