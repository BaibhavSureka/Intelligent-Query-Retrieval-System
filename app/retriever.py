from .embedder import get_embedding
import re

def extract_keywords(query: str) -> list:
    """Extract important keywords from query for better matching"""
    # Remove common words and extract key terms
    stop_words = {'what', 'is', 'the', 'for', 'in', 'of', 'to', 'and', 'or', 'with', 'under', 'this', 'that', 'are', 'does', 'do', 'how', 'when', 'where', 'why', 'which', 'who', 'can', 'will', 'should', 'would', 'could', 'may', 'might', 'must', 'shall'}
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Add domain-specific insurance terms for better matching
    insurance_terms = ['coverage', 'exclusion', 'waiting', 'period', 'premium', 'claim', 'benefit', 'policy', 'insured', 'maternity', 'organ', 'donor', 'health', 'check', 'grace', 'pre-existing', 'disease', 'expense', 'hospitalization']
    keywords.extend([term for term in insurance_terms if term in query.lower()])
    
    return keywords

def retrieve_relevant_chunks(query: str, vector_store, top_k=15):
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
                # Enhanced boosting for insurance documents
                boosted_score = score + (keyword_matches * 0.3)  # Increased from 0.2
                
                # Additional bonus for exact phrase matches
                exact_phrases = ['not covered', 'excluded', 'waiting period', 'grace period', 'pre-existing disease', 'maternity benefit', 'organ donor']
                for phrase in exact_phrases:
                    if phrase in chunk_lower:
                        boosted_score += 0.5
                
                filtered_chunks.append((chunk, boosted_score))
        
        # Sort by boosted scores and return top chunks
        filtered_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in filtered_chunks[:top_k]]
    
    return [chunk for chunk, _ in chunks_with_scores]