from .embedder import get_embedding
import re

def extract_keywords(query: str) -> list:
    """Extract important keywords from query for better matching"""
    # Remove common words and extract key terms
    stop_words = {'what', 'is', 'the', 'for', 'in', 'of', 'to', 'and', 'or', 'with', 'under', 'this', 'that', 'are', 'does', 'do', 'how', 'when', 'where', 'why', 'which', 'who', 'can', 'will', 'should', 'would', 'could', 'may', 'might', 'must', 'shall'}
    words = re.findall(r'\b\w+\b', query.lower())
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Add domain-specific terms for better matching
    domain_terms = ['coverage', 'exclusion', 'waiting', 'period', 'premium', 'claim', 'benefit', 'policy', 'insured', 'maternity', 'organ', 'donor', 'health', 'check', 'grace', 'pre-existing', 'disease', 'expense', 'hospitalization']
    keywords.extend([term for term in domain_terms if term in query.lower()])
    
    return list(set(keywords))  # Remove duplicates

def retrieve_relevant_chunks(query: str, vector_store, top_k=15):
    """Retrieve relevant chunks from vector store with improved scoring"""
    if len(vector_store.chunks) == 0:
        print("ERROR: Vector store is empty - no chunks to search!")
        return []
    
    print(f"Searching vector store with {len(vector_store.chunks)} chunks...")
    
    # Get query embedding
    try:
        query_emb = get_embedding(query)
    except Exception as e:
        print(f"Error getting query embedding: {e}")
        return []
    
    # Get initial semantic results (higher k for more candidates)
    initial_results = vector_store.search(query_emb, top_k=min(top_k * 3, len(vector_store.chunks)))
    
    if not initial_results:
        print("No semantically similar chunks found!")
        return []
    
    print(f"Found {len(initial_results)} semantically similar chunks")
    
    # Enhanced keyword-based re-ranking
    keywords = extract_keywords(query)
    query_lower = query.lower()
    
    # Re-score chunks with combined semantic + keyword scoring
    rescored_chunks = []
    for chunk, semantic_score in initial_results:
        chunk_lower = chunk.lower()
        
        # Start with semantic similarity score
        final_score = semantic_score
        
        # Keyword matching bonus
        keyword_matches = sum(1 for keyword in keywords if keyword in chunk_lower)
        if keyword_matches > 0:
            final_score += keyword_matches * 0.1
        
        # Enhanced phrase matching with medical patterns and table awareness
        phrase_bonuses = [
            ('maternity', ['maternity', 'pregnancy', 'childbirth', 'delivery']),
            ('room rent', ['room rent', 'boarding expenses', 'accommodation charges', 'room charges', 'daily room', 'bed charges', 'schedule of benefits', 'room and boarding']),
            ('ayush', ['ayush', 'ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy', 'alternative treatment', 'alternative treatments', 'non-allopathic']),
            ('domiciliary', ['domiciliary', 'home treatment', 'treatment at home', 'non-hospitalization', 'domiciliary hospitalization']),
            ('cumulative', ['cumulative bonus', 'bonus benefit', 'claim free', 'sum insured increase', 'rollover benefit']),
            ('waiting period', ['waiting period', 'waiting time', 'wait period']),
            ('pre-existing', ['pre-existing', 'pre existing', 'existing condition']),
            ('coverage', ['covered', 'coverage', 'benefit', 'included']),
            ('exclusion', ['excluded', 'not covered', 'exclusion', 'not included'])
        ]
        
        # CRITICAL: Chunk type boosting - prioritize structured data
        chunk_type_boost = 0
        if chunk.startswith('DEFINITION:'):
            chunk_type_boost = 0.4  # Highest boost for definitions
        elif chunk.startswith('SCHEDULE_SECTION:'):
            chunk_type_boost = 0.35  # High boost for benefit schedules
        elif chunk.startswith('BENEFIT:'):
            chunk_type_boost = 0.3   # Good boost for benefit descriptions
        
        final_score += chunk_type_boost
            
        # Additional boost for definition-like content patterns
        if any(term in chunk_lower for term in ['means', 'refers to', 'defined as', 'is defined as']):
            final_score += 0.2  # Increased from 0.15
            
        # Boost for coverage/benefit language
        if any(term in chunk_lower for term in ['coverage for', 'benefit for', 'covered under']):
            final_score += 0.15
        
        for query_phrase, matching_phrases in phrase_bonuses:
            if query_phrase in query_lower:
                for phrase in matching_phrases:
                    if phrase in chunk_lower:
                        final_score += 0.2
                        break
        
        # Special bonus for negation patterns (critical for accuracy)
        negation_patterns = ['not covered', 'excluded', 'not eligible', 'not applicable', 'does not cover', 'not included']
        for pattern in negation_patterns:
            if pattern in chunk_lower:
                final_score += 0.15
        
        # Bonus for numbers (often contain important facts)
        if re.search(r'\d+', chunk):
            final_score += 0.05
        
        rescored_chunks.append((chunk, final_score))
    
    # Sort by final score and return top chunks
    rescored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return just the chunks (not scores)
    final_chunks = [chunk for chunk, score in rescored_chunks[:top_k]]
    
    print(f"Returning {len(final_chunks)} top-ranked chunks")
    for i, chunk in enumerate(final_chunks[:3]):  # Show first 3 for debugging
        print(f"Chunk {i+1} preview: {chunk[:100]}...")
    
    return final_chunks