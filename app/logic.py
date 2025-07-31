import openai
import os
from dotenv import load_dotenv
import tiktoken
import re
from typing import List, Dict, Any

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TOKENS = 4096  # Reduced for GPT-3.5-turbo
RESERVED_FOR_ANSWER = 1024  # Leave room for the model's answer
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
    answer_score = min(len(answer) / 500.0, 1.0)  # Normalize to 0-1
    
    return (chunk_score + answer_score) / 2.0

def answer_question(query: str, context_chunks: list) -> Dict[str, Any]:
    """Optimized answer function for faster performance"""
    try:
        # Add context chunks until we hit the token limit
        selected_chunks = []
        total_tokens = 0
        for chunk in context_chunks:
            chunk_tokens = num_tokens(chunk)
            if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS:
                break
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens

        if not selected_chunks:
            return {
                "answer": "No relevant information found in the document to answer this question.",
                "supporting_clauses": [],
                "decision_rationale": "No relevant chunks were retrieved from the document.",
                "confidence_score": 0.0
            }

        context = "\n\n".join(selected_chunks)
        prompt = (
            f"Given the following policy/contract clauses:\n{context}\n\n"
            f"Answer the question: '{query}'\n"
            "Provide a concise answer with specific clause references. "
            "Keep your response brief and to the point. "
            "If the information is not available in the provided text, clearly state that."
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Faster model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Lower temperature for more consistent answers
            max_tokens=300,  # Shorter responses for better performance
        )
        
        answer = clean_response_text(response.choices[0].message.content)
        
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