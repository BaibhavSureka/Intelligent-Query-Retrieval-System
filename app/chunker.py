import re
import tiktoken

# Initialize tokenizer for token-aware chunking
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text))

def chunk_text(text: str, max_tokens: int = 500, overlap_tokens: int = 100, use_sentence_windows: bool = False) -> list:
    """Balanced chunking - maintains accuracy while improving speed"""
    if not text or not text.strip():
        return []
    
    # Enhanced medical system synonym mapping for better detection
    medical_synonyms = {
        'ayush': ['ayurveda', 'yoga', 'naturopathy', 'unani', 'siddha', 'homeopathy', 'alternative treatment', 'alternative treatments', 'non-allopathic', 'traditional medicine', 'ayush treatment', 'ayush treatments'],
        'room rent': ['boarding expenses', 'accommodation charges', 'room charges', 'daily room', 'bed charges', 'room and boarding', 'hospital room', 'accommodation cost'],
        'domiciliary': ['home treatment', 'treatment at home', 'non-hospitalization treatment', 'domiciliary hospitalization', 'domiciliary treatment', 'home care'],
        'grace period': ['payment grace', 'premium grace', 'installment grace', 'due date extension', 'grace days', 'payment extension'],
        'waiting period': ['coverage waiting', 'benefit waiting', 'treatment waiting', 'policy waiting', 'waiting time', 'exclusion period']
    }
    
    # Pre-process text to normalize medical terms for better matching
    text_lower = text.lower()
    for key_term, synonyms in medical_synonyms.items():
        for synonym in synonyms:
            if synonym in text_lower:
                # Add the key term near the synonym for better retrieval
                text = text.replace(synonym, f"{synonym} ({key_term})")
    
    # Clean and normalize text (improved from speed version)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Normalize multiple newlines
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    
    # TABLE-AWARE EXTRACTION: Capture structured data sections
    table_chunks = []
    
    # Extract Schedule of Benefits sections
    schedule_pattern = r'(Schedule of Benefits[^.]*?(?:Coverage|Benefit|Treatment)[^.]*?(?:\n.*?){0,10})'
    schedule_matches = re.findall(schedule_pattern, text, re.IGNORECASE | re.DOTALL)
    for match in schedule_matches:
        if len(match.strip()) > 50:  # Only meaningful content
            table_chunks.append(f"SCHEDULE_SECTION: {match.strip()}")
    
    # Extract definition-like patterns (for Room Rent, AYUSH, etc.)
    definition_pattern = r'((?:Room Rent|AYUSH Treatment|Domiciliary|Cumulative Bonus)[^.]*?means[^.]*?\.(?:[^.]*?\.){0,2})'
    definition_matches = re.findall(definition_pattern, text, re.IGNORECASE | re.DOTALL)
    for match in definition_matches:
        if len(match.strip()) > 30:
            table_chunks.append(f"DEFINITION: {match.strip()}")
    
    # Extract benefit tables with specific patterns
    benefit_pattern = r'(\b(?:Room Rent|AYUSH|Domiciliary|Alternative Treatment)[^.]*?(?:covered|benefit|treatment|means)[^.]*?\.(?:[^.]*?\.){0,1})'
    benefit_matches = re.findall(benefit_pattern, text, re.IGNORECASE | re.DOTALL)
    for match in benefit_matches:
        if len(match.strip()) > 40:
            table_chunks.append(f"BENEFIT: {match.strip()}")
    
    # Better sentence splitting that preserves context
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_tokens = num_tokens(sentence)
        
        # Handle long sentences by splitting on commas/semicolons
        if sentence_tokens > max_tokens:
            parts = re.split(r'[,;]\s*', sentence)
            for part in parts:
                if num_tokens(part.strip()) <= max_tokens and part.strip():
                    if current_tokens + num_tokens(part) <= max_tokens:
                        if current_chunk:
                            current_chunk += " " + part.strip()
                        else:
                            current_chunk = part.strip()
                        current_tokens = num_tokens(current_chunk)
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part.strip()
                        current_tokens = num_tokens(current_chunk)
            i += 1
            continue
        
        # Check if adding this sentence would exceed the limit
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(current_chunk.strip())
            
            # Create minimal overlap for context continuity (simplified from complex version)
            overlap_chunk = ""
            overlap_chunk_tokens = 0
            
            # Add last sentence for overlap if it fits
            recent_sentences = current_chunk.split('. ')
            if recent_sentences and overlap_tokens > 0:
                last_sent = recent_sentences[-1].strip()
                if num_tokens(last_sent) <= overlap_tokens:
                    overlap_chunk = last_sent
                    overlap_chunk_tokens = num_tokens(overlap_chunk)
            
            # Start new chunk with overlap + current sentence
            if overlap_chunk.strip():
                current_chunk = overlap_chunk.strip() + '. ' + sentence
            else:
                current_chunk = sentence
            current_tokens = num_tokens(current_chunk)
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += ' ' + sentence
            else:
                current_chunk = sentence
            current_tokens += sentence_tokens
        
        i += 1
    
    # Add the last chunk if it exists
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # COMBINE: Add table chunks to regular chunks (high priority)
    all_chunks = table_chunks + chunks  # Table chunks first for higher priority
    
    # Balanced filtering - keep meaningful chunks but not too strict
    meaningful_chunks = []
    for chunk in all_chunks:
        chunk_tokens = num_tokens(chunk)
        # Keep chunks with at least 15 tokens and some meaningful content (relaxed from 20)
        # Special handling for table/definition chunks
        min_tokens = 10 if chunk.startswith(('SCHEDULE_SECTION:', 'DEFINITION:', 'BENEFIT:')) else 15
        min_length = 30 if chunk.startswith(('SCHEDULE_SECTION:', 'DEFINITION:', 'BENEFIT:')) else 40
        
        if chunk_tokens >= min_tokens and len(chunk.strip()) >= min_length:
            meaningful_chunks.append(chunk.strip())
    
    print(f"Balanced chunking: {len(meaningful_chunks)} chunks from {num_tokens(text)} tokens (including {len(table_chunks)} table/definition chunks)")
    return meaningful_chunks

def chunk_text_by_characters(text: str, max_length: int = 1200) -> list:
    """Split text into chunks based on character count (legacy method)"""
    # Split by single newlines, then merge to max_length
    paras = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    chunks = []
    current = ""
    
    for para in paras:
        if len(current) + len(para) < max_length:
            current += " " + para
        else:
            if current:
                chunks.append(current.strip())
            current = para
    
    if current:
        chunks.append(current.strip())
    
    return chunks

def chunk_text_with_sections(text: str, max_tokens: int = 1000) -> list:
    """Enhanced chunking that preserves section structure"""
    # Split by major sections first
    sections = re.split(r'(Section [A-Z]|SECTION [A-Z]|\d+\.\s*[A-Z][^.\n]*)', text)
    chunks = []
    
    for section in sections:
        if not section.strip():
            continue
            
        # If section is small enough, keep it as one chunk
        if num_tokens(section) <= max_tokens:
            chunks.append(section.strip())
        else:
            # Split large sections into smaller chunks
            section_chunks = chunk_text(section, max_tokens)
            chunks.extend(section_chunks)
    
    return chunks