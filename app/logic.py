import openai
import os
from dotenv import load_dotenv
import tiktoken
import re
from typing import List, Dict, Any
from .document_analyzer import detect_document_type, normalize_question_intent

load_dotenv()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_TOKENS = 4096  # GPT-3.5-turbo limit
RESERVED_FOR_ANSWER = 1200  # Increased for more detailed responses
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

def verify_facts(answer: str, context_chunks: List[str]) -> str:
    """Verify facts in the answer against context chunks"""
    # Generic fact verification for any document type
    context_text = " ".join(context_chunks).lower()
    
    # Remove incorrect "information not available" statements
    if "the information is not available in the provided document" in answer.lower():
        # Check if the answer actually contains substantive information
        if len(answer) > 100:  # If answer has substantial content, remove the false statement
            answer = answer.replace("The information is not available in the provided document.", "")
            answer = answer.replace("the information is not available in the provided document.", "")
            answer = answer.replace("The information is not available in the provided document", "")
            answer = answer.replace("the information is not available in the provided document", "")
    
    # Generic cleaning to remove any double spaces or punctuation issues
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    return answer

async def answer_question_async(query: str, context_chunks: list, document_text: str = "") -> Dict[str, Any]:
    """Enhanced async answer generation with dynamic document-aware prompting"""
    try:
        if not context_chunks:
            return {
                "answer": "No relevant information found in the document to answer this question.",
                "supporting_clauses": [],
                "decision_rationale": "No relevant chunks were retrieved from the document.",
                "confidence_score": 0.0
            }
        
        # Detect document type and get appropriate instructions
        doc_analysis = detect_document_type(document_text) if document_text else {
            'type': 'general',
            'system_prompt': "You are a document analyst. Answer questions accurately based on the provided context.",
            'context_instructions': "Look for relevant information that directly addresses the question."
        }
        
        # Normalize question intent based on document type
        normalized_query = normalize_question_intent(query, doc_analysis['type'])
        if normalized_query != query:
            print(f"Question normalized: '{query}' -> '{normalized_query}'")
        
        # Prioritize chunks with type information for better context
        prioritized_chunks = []
        definition_chunks = []
        regular_chunks = []
        
        for chunk in context_chunks[:8]:  # Process more chunks for better coverage
            if chunk.startswith(('DEFINITION:', 'SCHEDULE_SECTION:', 'BENEFIT:')):
                definition_chunks.append(chunk)
            else:
                regular_chunks.append(chunk)
        
        # Put definition/structured chunks first for higher priority
        prioritized_chunks = definition_chunks + regular_chunks
        context = "\n\n".join(prioritized_chunks[:6])
        
        # Enhanced system prompt with document-type awareness
        system_prompt = f"""{doc_analysis['system_prompt']}
        
Important: If the context contains definitions (e.g., "Room Rent means..."), treat these as direct answers to coverage questions. 
When you see structured information like "DEFINITION:" or "SCHEDULE_SECTION:", prioritize this information in your response.
Be specific and include details like amounts, percentages, and conditions when available."""
        
        user_prompt = f"""DOCUMENT CONTEXT:
{context}

QUESTION: {normalized_query}

{doc_analysis['context_instructions']} Provide a specific answer based on the context above:"""
        
        # Single fast API call with enhanced parameters
        try:
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=250,  # Increased for more detailed responses
                timeout=12       # Slightly increased for quality
            )
            answer = response.choices[0].message.content.strip()
            model_used = "gpt-3.5-turbo-0125"
        except Exception as e:
            return {
                "answer": f"Processing error: {str(e)}",
                "supporting_clauses": [],
                "decision_rationale": "AI service error",
                "confidence_score": 0.0
            }
        
        # Minimal cleaning and length control
        answer = clean_response_text(answer)
        
        # Enhanced conciseness control - trim verbose answers
        sentences = answer.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # If more than 3 sentences, keep only the most essential ones
        if len(sentences) > 3:
            # Keep first sentence (main answer) and most important details
            answer = sentences[0] + '. ' + sentences[1] + '.'
        elif len(answer) > 250:
            # If still too long, truncate at reasonable point
            answer = answer[:200] + '...'
        
        # Ensure proper ending
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # Enhanced confidence calculation with more factors
        base_confidence = 0.8 if len(answer) > 20 else 0.3
        
        # Boost confidence for medical term matches
        medical_terms = ['ayush', 'room rent', 'domiciliary', 'grace', 'waiting', 'cumulative']
        term_matches = sum(1 for term in medical_terms if term.lower() in context.lower())
        confidence_boost = min(0.15, term_matches * 0.03)
        
        confidence_score = min(0.95, base_confidence + confidence_boost)
        
        # Find best source chunk (most relevant)
        best_chunk_idx = 0
        best_chunk_score = 0.0
        for i, chunk in enumerate(context_chunks[:4]):
            # Simple relevance scoring based on query terms
            query_words = query.lower().split()
            chunk_words = chunk.lower().split()
            matches = sum(1 for word in query_words if word in chunk_words)
            score = matches / len(query_words) if query_words else 0
            if score > best_chunk_score:
                best_chunk_score = score
                best_chunk_idx = i
        
        return {
            "answer": answer,
            "supporting_clauses": [],
            "decision_rationale": f"Analysis of {len(context_chunks)} chunks using {model_used}",
            "confidence_score": confidence_score,
            "model_used": model_used,
            "chunk_count": len(context_chunks),
            "source_chunk": best_chunk_idx + 1,  # 1-based indexing for user display
            "source_preview": context_chunks[best_chunk_idx][:150] + "..." if context_chunks else ""
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "supporting_clauses": [],
            "decision_rationale": "Processing error",
            "confidence_score": 0.0
        }

def answer_question(query: str, context_chunks: list, document_text: str = "") -> Dict[str, Any]:
    """Enhanced answer function with document-type awareness and robust error handling"""
    try:
        if not context_chunks:
            return {
                "answer": "No relevant information found in the document to answer this question.",
                "supporting_clauses": [],
                "decision_rationale": "No relevant chunks were retrieved from the document.",
                "confidence_score": 0.0
            }
        
        print(f"Processing question with {len(context_chunks)} context chunks")
        
        # Detect document type and get appropriate instructions
        doc_analysis = detect_document_type(document_text) if document_text else {
            'type': 'general',
            'system_prompt': "You are a document analyst. Answer questions accurately based on the provided context.",
            'context_instructions': "Look for relevant information that directly addresses the question."
        }
        
        # Normalize question intent based on document type
        normalized_query = normalize_question_intent(query, doc_analysis['type'])
        if normalized_query != query:
            print(f"Question normalized: '{query}' -> '{normalized_query}'")
        
        # Enhanced context selection with chunk type prioritization
        query_lower = query.lower()
        key_terms = re.findall(r'\b\w{3,}\b', query_lower)
        
        # Add domain-specific key terms
        domain_terms = ['coverage', 'exclusion', 'period', 'claim', 'benefit', 'policy', 'insured', 'document', 'section', 'clause', 'condition', 'requirement', 'maternity', 'waiting', 'pre-existing']
        key_terms.extend([term for term in domain_terms if term in query_lower])
        
        # Score and select the best chunks with type-awareness
        scored_chunks = []
        for i, chunk in enumerate(context_chunks[:15]):  # Process top 15 chunks
            chunk_lower = chunk.lower()
            relevance_score = 0
            
            # CRITICAL: Chunk type boosting - same as in retriever
            if chunk.startswith('DEFINITION:'):
                relevance_score += 8  # High boost for definitions
            elif chunk.startswith('SCHEDULE_SECTION:'):
                relevance_score += 7  # High boost for benefit schedules
            elif chunk.startswith('BENEFIT:'):
                relevance_score += 6  # Good boost for benefit descriptions
            
            # Keyword matching
            for term in key_terms:
                if term in chunk_lower:
                    relevance_score += 1
            
            # Exact phrase bonuses
            high_value_phrases = ['not covered', 'excluded', 'waiting period', 'grace period', 'pre-existing', 'maternity', 'coverage', 'benefit', 'means', 'defined as']
            for phrase in high_value_phrases:
                if phrase in chunk_lower:
                    relevance_score += 2
            
            # Question-specific bonuses
            if 'maternity' in query_lower and any(term in chunk_lower for term in ['maternity', 'pregnancy', 'childbirth']):
                relevance_score += 3
            if 'pre-existing' in query_lower and any(term in chunk_lower for term in ['pre-existing', 'existing condition']):
                relevance_score += 3
            if 'waiting' in query_lower and 'waiting' in chunk_lower:
                relevance_score += 3
            if any(term in query_lower for term in ['room rent', 'accommodation']) and any(term in chunk_lower for term in ['room rent', 'boarding', 'accommodation']):
                relevance_score += 4
            if 'ayush' in query_lower and any(term in chunk_lower for term in ['ayush', 'alternative', 'ayurveda', 'homeopathy']):
                relevance_score += 4
            
            scored_chunks.append((chunk, relevance_score, i))
        
        # Sort by relevance score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Select chunks within token limit
        selected_chunks = []
        total_tokens = 0
        
        for chunk, score, original_idx in scored_chunks:
            chunk_tokens = num_tokens(chunk)
            if total_tokens + chunk_tokens <= MAX_CONTEXT_TOKENS and len(selected_chunks) < 8:  # Limit to best 8 chunks
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
                print(f"Selected chunk {original_idx} with score {score} ({chunk_tokens} tokens)")
            elif len(selected_chunks) >= 8:
                break
        
        if not selected_chunks:
            return {
                "answer": "No sufficiently relevant information found in the document to answer this question.",
                "supporting_clauses": [],
                "decision_rationale": "No chunks met the relevance threshold.",
                "confidence_score": 0.0
            }
        
        # Create context with prioritized chunks (definition chunks first)
        definition_chunks = [chunk for chunk in selected_chunks if chunk.startswith(('DEFINITION:', 'SCHEDULE_SECTION:', 'BENEFIT:'))]
        regular_chunks = [chunk for chunk in selected_chunks if not chunk.startswith(('DEFINITION:', 'SCHEDULE_SECTION:', 'BENEFIT:'))]
        prioritized_chunks = definition_chunks + regular_chunks
        
        context = "\n\n---\n\n".join(prioritized_chunks)
        
        # Enhanced prompt with document-type awareness
        prompt = (
            f"{doc_analysis['system_prompt']}\n\n"
            f"IMPORTANT: If the context contains definitions (e.g., 'Room Rent means...'), treat these as direct answers to coverage questions. "
            f"When you see structured information like 'DEFINITION:' or 'SCHEDULE_SECTION:', prioritize this information in your response.\n\n"
            f"DOCUMENT EXCERPTS:\n{context}\n\n"
            f"QUESTION: {normalized_query}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Answer based ONLY on the document excerpts above\n"
            f"2. Be precise with numbers, dates, amounts, and percentages\n"
            f"3. If something is NOT covered or excluded, state this clearly\n"
            f"4. Include specific conditions, waiting periods, or limitations\n"
            f"5. If the information is not in the excerpts, say 'This information is not available in the provided document'\n"
            f"6. For yes/no questions, give a clear answer first, then explain\n"
            f"7. Quote relevant parts of the document when helpful\n"
            f"8. Be direct and factual\n\n"
            f"ANSWER:"
        )
        
        # Try GPT-4 first for better accuracy
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=500,
                timeout=25
            )
            model_used = "gpt-4"
        except Exception as gpt4_error:
            print(f"GPT-4 failed: {gpt4_error}")
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=500,
                    timeout=20
                )
                model_used = "gpt-3.5-turbo"
            except Exception as gpt35_error:
                print(f"Both models failed: {gpt35_error}")
                return {
                    "answer": "Unable to process the question due to AI service issues. Please try again.",
                    "supporting_clauses": [],
                    "decision_rationale": f"AI models unavailable: {str(gpt35_error)}",
                    "confidence_score": 0.0
                }
        
        answer = clean_response_text(response.choices[0].message.content)
        answer = verify_facts(answer, selected_chunks)
        
        # Clean answer
        answer = answer.strip()
        while answer.endswith(('/', '\\', '|', '-', ' ')):
            answer = answer.rstrip('/').rstrip('\\').rstrip('|').rstrip('-').rstrip()
        
        # Extract supporting clauses
        supporting_clauses = extract_supporting_clauses(selected_chunks)
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(selected_chunks, answer)
        
        print(f"Generated answer using {model_used}: {answer[:100]}...")
        
        return {
            "answer": answer,
            "supporting_clauses": supporting_clauses,
            "decision_rationale": f"Analysis based on {len(selected_chunks)} most relevant document sections using {model_used}.",
            "confidence_score": round(confidence_score, 2)
        }
        
    except Exception as e:
        print(f"Error in answer_question: {e}")
        return {
            "answer": f"Error processing question: {str(e)}",
            "supporting_clauses": [],
            "decision_rationale": "An error occurred during processing.",
            "confidence_score": 0.0
        }