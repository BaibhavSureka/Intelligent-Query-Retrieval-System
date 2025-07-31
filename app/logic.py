import openai
import os
from dotenv import load_dotenv
import tiktoken
import re
from typing import List, Dict, Any

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TOKENS = 4096  # GPT-3.5-turbo limit
RESERVED_FOR_ANSWER = 800  # Reduced for faster responses
MAX_CONTEXT_TOKENS = MAX_TOKENS - RESERVED_FOR_ANSWER

tokenizer = tiktoken.get_encoding("cl100k_base")

def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def clean_response_text(text: str) -> str:
    """Clean up response text for better JSON formatting"""
    # Replace multiple newlines with single newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    # Replace single newlines with spaces for better readability
    text = re.sub(r'\n(?!\n)', ' ', text)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove any trailing slashes, backslashes, or unwanted characters
    text = text.strip().rstrip('/').rstrip('\\').rstrip('|').rstrip('-')
    # Remove any remaining trailing characters that might cause issues
    while text.endswith(('/', '\\', '|', '-', ' ')):
        text = text.rstrip('/').rstrip('\\').rstrip('|').rstrip('-').rstrip()
    return text.strip()

def extract_supporting_clauses(context_chunks: List[str]) -> List[Dict[str, str]]:
    """Extract potential supporting clauses from context chunks"""
    clauses = []
    for i, chunk in enumerate(context_chunks):
        # Look for numbered clauses or sections
        clause_matches = re.findall(r'(\d+\.\s*[^.\n]+)', chunk)
        for match in clause_matches:
            clauses.append({
                "text": match,
                "location": f"Chunk {i+1}"
            })
    return clauses

def calculate_confidence_score(relevant_chunks: List[str], answer: str) -> float:
    """Calculate a simple confidence score based on chunk relevance and answer length"""
    if not relevant_chunks:
        return 0.0
    
    # Simple heuristic: more relevant chunks + detailed answer = higher confidence
    chunk_score = min(len(relevant_chunks) / 5.0, 1.0)  # Normalize to 0-1
    answer_score = min(len(answer) / 300.0, 1.0)  # Normalize to 0-1
    
    return (chunk_score + answer_score) / 2.0

def answer_question(query: str, context_chunks: list) -> Dict[str, Any]:
    """General answer function for any type of document"""
    try:
        # Optimize context selection for better accuracy
        selected_chunks = []
        total_tokens = 0
        
        # Score chunks by relevance to query
        query_lower = query.lower()
        key_terms = re.findall(r'\b\w{3,}\b', query_lower)
        
        scored_chunks = []
        for chunk in context_chunks[:12]:  # Limit initial search
            chunk_lower = chunk.lower()
            relevance_score = 0
            
            # Score based on keyword matches
            for term in key_terms:
                if term in chunk_lower:
                    relevance_score += 1
            
            scored_chunks.append((chunk, relevance_score))
        
        # Sort by relevance and select within token limit
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        for chunk, score in scored_chunks:
            chunk_tokens = num_tokens(chunk)
            if total_tokens + chunk_tokens <= MAX_CONTEXT_TOKENS:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                break

        if not selected_chunks:
            return {
                "answer": "No relevant information found in the document to answer this question.",
                "supporting_clauses": [],
                "decision_rationale": "No relevant chunks were retrieved from the document.",
                "confidence_score": 0.0
            }

        context = "\n\n".join(selected_chunks)
        
        # General prompt for any type of document
        prompt = (
            f"You are an expert document analyst. Analyze the following document text and answer the question accurately:\n\n"
            f"DOCUMENT TEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer based ONLY on the provided document text\n"
            f"2. Be specific and accurate with numbers, dates, and details\n"
            f"3. Include relevant section references when available\n"
            f"4. If information is not in the text, say 'The information is not available in the provided document'\n"
            f"5. Keep answer concise but complete\n"
            f"6. Focus on the most relevant details from the document\n"
            f"7. Do not include any slashes, backslashes, or special formatting characters at the end of your response"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Zero temperature for consistency
            max_tokens=250,  # Shorter for faster responses
            timeout=15  # Add timeout for faster failure
        )
        
        answer = clean_response_text(response.choices[0].message.content)
        
        # More aggressive cleaning to remove any slashes or unwanted characters
        answer = answer.strip().rstrip('/').rstrip('\\').rstrip('|')
        # Remove any remaining slashes at the end
        while answer.endswith('/') or answer.endswith('\\'):
            answer = answer.rstrip('/').rstrip('\\')
        
        # Extract supporting clauses
        supporting_clauses = extract_supporting_clauses(selected_chunks)
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(selected_chunks, answer)
        
        return {
            "answer": answer,
            "supporting_clauses": supporting_clauses,
            "decision_rationale": "Analysis based on retrieved document sections.",
            "confidence_score": round(confidence_score, 2)
        }
        
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "supporting_clauses": [],
            "decision_rationale": "An error occurred during processing.",
            "confidence_score": 0.0
        }