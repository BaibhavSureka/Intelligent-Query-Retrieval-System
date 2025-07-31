import openai
import os
from dotenv import load_dotenv
import tiktoken

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TOKENS = 8191  # Safe limit for embedding model

tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def split_chunk(text: str, max_tokens: int = MAX_TOKENS) -> list:
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if num_tokens(" ".join(current_chunk)) > max_tokens:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_embedding(text: str, model="text-embedding-ada-002"):
    if num_tokens(text) > MAX_TOKENS:
        raise ValueError(f"Text too long ({num_tokens(text)} tokens). Must be ≤ {MAX_TOKENS} tokens.")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def embed_chunks(chunks: list) -> list:
    """Embed chunks with batch processing for better performance"""
    embedded = []
    
    # Process in batches for better performance
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = []
        
        for chunk in batch:
            if num_tokens(chunk) <= MAX_TOKENS:
                batch_texts.append(chunk)
            else:
                subchunks = split_chunk(chunk)
                batch_texts.extend(subchunks)
        
        # Batch embed for efficiency
        if batch_texts:
            try:
                response = client.embeddings.create(input=batch_texts, model="text-embedding-ada-002")
                embedded.extend([data.embedding for data in response.data])
            except Exception as e:
                # Fallback to individual embedding if batch fails
                for text in batch_texts:
                    embedded.append(get_embedding(text))
    
    return embedded
