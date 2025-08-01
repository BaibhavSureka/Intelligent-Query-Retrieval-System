import re
from typing import Dict, Any

def detect_document_type(text: str) -> Dict[str, Any]:
    """Dynamically detect document type and return appropriate instructions"""
    text_lower = text.lower()
    
    # Insurance/Policy document indicators
    insurance_indicators = [
        'policy', 'insurance', 'premium', 'coverage', 'benefits', 'claim',
        'insured', 'exclusions', 'waiting period', 'sum insured', 'deductible'
    ]
    
    # Legal document indicators
    legal_indicators = [
        'whereas', 'party of the first part', 'agreement', 'contract',
        'terms and conditions', 'liability', 'jurisdiction', 'breach'
    ]
    
    # Medical/Health document indicators
    medical_indicators = [
        'diagnosis', 'treatment', 'patient', 'medical', 'symptoms',
        'prescription', 'dosage', 'clinical', 'hospital'
    ]
    
    # Academic/Textbook indicators
    academic_indicators = [
        'chapter', 'exercise', 'lesson', 'student', 'learning objectives',
        'quiz', 'homework', 'curriculum', 'syllabus'
    ]
    
    # Count indicators
    insurance_count = sum(1 for indicator in insurance_indicators if indicator in text_lower)
    legal_count = sum(1 for indicator in legal_indicators if indicator in text_lower)
    medical_count = sum(1 for indicator in medical_indicators if indicator in text_lower)
    academic_count = sum(1 for indicator in academic_indicators if indicator in text_lower)
    
    # Determine document type
    counts = {
        'insurance': insurance_count,
        'legal': legal_count,
        'medical': medical_count,
        'academic': academic_count
    }
    
    doc_type = max(counts, key=counts.get)
    confidence = counts[doc_type] / max(1, sum(counts.values()))
    
    # Return appropriate system prompt and instructions
    if doc_type == 'insurance':
        return {
            'type': 'insurance',
            'confidence': confidence,
            'system_prompt': "You are an insurance policy analyst. When answering questions, prioritize definitions, coverage details, and policy terms found in the context. If a definition is provided (e.g., 'Room Rent means...'), use it as the answer even if the question uses different phrasing like 'coverage'.",
            'context_instructions': "Look for definitions, benefit schedules, and policy clauses that directly address the question."
        }
    elif doc_type == 'legal':
        return {
            'type': 'legal',
            'confidence': confidence,
            'system_prompt': "You are a legal document analyst. Interpret clauses strictly based on the exact text provided. Avoid assumptions and stick to what is explicitly stated.",
            'context_instructions': "Focus on specific clauses, terms, and legal provisions."
        }
    elif doc_type == 'medical':
        return {
            'type': 'medical',
            'confidence': confidence,
            'system_prompt': "You are a medical document analyst. Provide accurate information based on medical records, diagnoses, and treatment information in the context.",
            'context_instructions': "Look for diagnostic information, treatment plans, and medical terminology."
        }
    elif doc_type == 'academic':
        return {
            'type': 'academic',
            'confidence': confidence,
            'system_prompt': "You are an educational content analyst helping students understand concepts. Explain clearly using the educational material provided.",
            'context_instructions': "Focus on explanations, examples, and educational content."
        }
    else:
        return {
            'type': 'general',
            'confidence': 0.5,
            'system_prompt': "You are a document analyst. Answer questions accurately based on the provided context. Use definitions and specific information when available.",
            'context_instructions': "Look for relevant information that directly addresses the question."
        }

def normalize_question_intent(question: str, doc_type: str) -> str:
    """Normalize question intent based on document type"""
    question_lower = question.lower()
    
    if doc_type == 'insurance':
        # Map coverage questions to definition/benefit questions
        if 'coverage for' in question_lower or 'covered for' in question_lower:
            # Extract the subject (e.g., "room rent", "ayush")
            subjects = ['room rent', 'ayush', 'domiciliary', 'maternity', 'cumulative']
            for subject in subjects:
                if subject in question_lower:
                    return question.replace('coverage for', f'definition or benefits for').replace('covered for', f'definition or benefits for')
        
        if 'what is' in question_lower and 'coverage' in question_lower:
            return question.replace('coverage', 'definition, benefits, or limits')
    
    return question  # Return original if no normalization needed
