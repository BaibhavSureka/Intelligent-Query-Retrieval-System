import re
import tiktoken

# Initialize tokenizer for token-aware chunking
tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    """Count tokens in text"""
    return len(tokenizer.encode(text))

def chunk_text(text: str, max_tokens: int = 800) -> list:
    """Split text into chunks based on token count for better embedding performance"""
    # Split by paragraphs first
    paras = [p.strip() for p in re.split(r'\n+', text) if p.strip()]
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for para in paras:
        para_tokens = num_tokens(para)
        
        # If adding this paragraph would exceed the limit, save current chunk and start new one
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
            current_tokens = para_tokens
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += " " + para
                current_tokens += para_tokens
            else:
                current_chunk = para
                current_tokens = para_tokens
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_text_by_characters(text: str, max_length: int = 1000) -> list:
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