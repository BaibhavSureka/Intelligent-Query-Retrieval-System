import asyncio
import logging
import time
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
import json
import re

from openai import AsyncOpenAI

from models import (
    DocumentChunk, RetrievalResult, ClauseMatch, 
    DecisionRationale, AnswerWithRationale
)
from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for LLM-powered query processing and answer generation"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
        self.max_tokens = settings.max_tokens
        self.temperature = settings.temperature
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
    
    async def process_query(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        document_url: str
    ) -> AnswerWithRationale:
        """
        Process a query using LLM with retrieved context
        
        Args:
            question: The user's question
            retrieval_results: Retrieved document chunks
            document_url: Source document URL
            
        Returns:
            AnswerWithRationale object with answer and reasoning
        """
        start_time = time.time()
        
        try:
            # Extract relevant clauses
            clause_matches = await self._extract_clause_matches(question, retrieval_results)
            
            # Generate answer using LLM
            answer = await self._generate_answer(question, retrieval_results, clause_matches)
            
            # Create decision rationale
            rationale = await self._create_decision_rationale(
                question, answer, retrieval_results, clause_matches
            )
            
            # Count tokens used
            tokens_used = self._count_tokens(question, retrieval_results, answer)
            
            processing_time = time.time() - start_time
            
            return AnswerWithRationale(
                answer=answer,
                rationale=rationale,
                processing_time=processing_time,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    async def _extract_clause_matches(
        self,
        question: str,
        retrieval_results: List[RetrievalResult]
    ) -> List[ClauseMatch]:
        """Extract and match specific clauses relevant to the question"""
        try:
            clause_matches = []
            
            # Define clause patterns for insurance/legal documents
            clause_patterns = [
                r'(?i)(waiting period|grace period|cooling period).*?(\d+\s*(?:days|months|years))',
                r'(?i)(coverage|covers?|covered).*?(?:for|of|includes?)\s+([^.]+)',
                r'(?i)(exclusion|excluded|not covered).*?([^.]+)',
                r'(?i)(condition|requirement|term).*?([^.]+)',
                r'(?i)(benefit|discount|allowance).*?(\d+%?|\$\d+)',
                r'(?i)(limit|maximum|minimum|up to).*?(\$?\d+(?:,\d+)*(?:\.\d+)?)',
                r'(?i)(deductible|co-payment|premium).*?(\$?\d+(?:,\d+)*(?:\.\d+)?)',
            ]
            
            for result in retrieval_results:
                content = result.chunk.content
                
                # Extract clauses using patterns
                for pattern in clause_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        clause_text = match.group(0).strip()
                        
                        # Calculate relevance score based on question similarity
                        relevance_score = await self._calculate_clause_relevance(question, clause_text)
                        
                        if relevance_score > 0.3:  # Threshold for relevance
                            # Get surrounding context
                            start_pos = max(0, match.start() - 100)
                            end_pos = min(len(content), match.end() + 100)
                            context = content[start_pos:end_pos]
                            
                            clause_match = ClauseMatch(
                                clause_text=clause_text,
                                relevance_score=relevance_score,
                                source_location=f"Chunk {result.chunk.chunk_id}",
                                context=context
                            )
                            clause_matches.append(clause_match)
            
            # Sort by relevance score
            clause_matches.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Return top matches
            return clause_matches[:10]
            
        except Exception as e:
            logger.error(f"Error extracting clause matches: {str(e)}")
            return []
    
    async def _calculate_clause_relevance(self, question: str, clause_text: str) -> float:
        """Calculate relevance score between question and clause"""
        try:
            # Simple keyword-based relevance calculation
            question_words = set(question.lower().split())
            clause_words = set(clause_text.lower().split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            question_words = question_words - stop_words
            clause_words = clause_words - stop_words
            
            if not question_words or not clause_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(question_words.intersection(clause_words))
            union = len(question_words.union(clause_words))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating clause relevance: {str(e)}")
            return 0.0
    
    async def _generate_answer(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        clause_matches: List[ClauseMatch]
    ) -> str:
        """Generate answer using LLM"""
        try:
            # Prepare context from retrieval results
            context_chunks = []
            for i, result in enumerate(retrieval_results[:5]):  # Use top 5 results
                context_chunks.append(f"Context {i+1} (Similarity: {result.similarity_score:.3f}):\n{result.chunk.content}")
            
            context_text = "\n\n".join(context_chunks)
            
            # Prepare clause information
            clause_text = ""
            if clause_matches:
                clause_text = "\n\nRelevant Clauses:\n"
                for i, clause in enumerate(clause_matches[:3]):  # Use top 3 clauses
                    clause_text += f"Clause {i+1}: {clause.clause_text}\n"
            
            # Create prompt
            prompt = f"""You are an expert document analysis assistant specializing in insurance, legal, HR, and compliance documents. Your task is to answer questions based on the provided document context with high accuracy and clarity.

**Question:** {question}

**Document Context:**
{context_text}
{clause_text}

**Instructions:**
1. Answer the question directly and comprehensively based ONLY on the provided context
2. If the context contains specific information, quote it directly
3. Include relevant numbers, percentages, timeframes, and conditions
4. If information is not available in the context, clearly state this
5. Be precise and avoid speculation
6. For insurance/legal questions, focus on coverage, conditions, exclusions, and requirements
7. Structure your answer clearly with specific details

**Answer:**"""

            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise document analysis expert. Provide accurate, detailed answers based strictly on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer for question: {question[:50]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise
    
    async def _create_decision_rationale(
        self,
        question: str,
        answer: str,
        retrieval_results: List[RetrievalResult],
        clause_matches: List[ClauseMatch]
    ) -> DecisionRationale:
        """Create explainable decision rationale"""
        try:
            # Generate reasoning explanation
            reasoning_prompt = f"""Explain the reasoning behind this answer in a clear, structured way:

**Question:** {question}
**Answer:** {answer}

**Available Evidence:**
{len(retrieval_results)} document chunks were analyzed
{len(clause_matches)} relevant clauses were identified

Provide a brief explanation of:
1. What information was used to answer the question
2. How confident you are in this answer
3. Any limitations or assumptions made

Keep the explanation concise but informative."""

            reasoning_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are explaining the reasoning behind a document analysis decision. Be clear and logical."
                    },
                    {
                        "role": "user",
                        "content": reasoning_prompt
                    }
                ],
                max_tokens=300,
                temperature=0.1
            )
            
            reasoning = reasoning_response.choices[0].message.content.strip()
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                question, answer, retrieval_results, clause_matches
            )
            
            return DecisionRationale(
                reasoning=reasoning,
                evidence_chunks=retrieval_results[:5],  # Top 5 as evidence
                confidence_score=confidence_score,
                clause_matches=clause_matches[:3]  # Top 3 clause matches
            )
            
        except Exception as e:
            logger.error(f"Error creating decision rationale: {str(e)}")
            # Fallback rationale
            return DecisionRationale(
                reasoning="Answer generated based on document analysis and retrieval.",
                evidence_chunks=retrieval_results[:5],
                confidence_score=0.7,
                clause_matches=clause_matches[:3]
            )
    
    def _calculate_confidence_score(
        self,
        question: str,
        answer: str,
        retrieval_results: List[RetrievalResult],
        clause_matches: List[ClauseMatch]
    ) -> float:
        """Calculate confidence score for the answer"""
        try:
            confidence_factors = []
            
            # Factor 1: Quality of retrieval results
            if retrieval_results:
                avg_similarity = sum(r.similarity_score for r in retrieval_results) / len(retrieval_results)
                confidence_factors.append(avg_similarity)
            
            # Factor 2: Number of relevant clauses found
            clause_factor = min(1.0, len(clause_matches) / 3.0)  # Normalize to max 1.0
            confidence_factors.append(clause_factor)
            
            # Factor 3: Answer specificity (length and detail)
            answer_specificity = min(1.0, len(answer.split()) / 50.0)  # Normalize
            confidence_factors.append(answer_specificity)
            
            # Factor 4: Check for uncertainty phrases in answer
            uncertainty_phrases = ['not clear', 'unclear', 'not specified', 'not mentioned', 'may', 'might', 'possibly']
            uncertainty_penalty = 0.0
            for phrase in uncertainty_phrases:
                if phrase in answer.lower():
                    uncertainty_penalty += 0.1
            
            # Calculate weighted average
            if confidence_factors:
                base_confidence = sum(confidence_factors) / len(confidence_factors)
                final_confidence = max(0.1, base_confidence - uncertainty_penalty)
            else:
                final_confidence = 0.5  # Default moderate confidence
            
            return min(1.0, final_confidence)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.7  # Default confidence
    
    def _count_tokens(
        self,
        question: str,
        retrieval_results: List[RetrievalResult],
        answer: str
    ) -> int:
        """Count tokens used in the LLM request"""
        try:
            # Count tokens in question
            question_tokens = len(self.tokenizer.encode(question))
            
            # Count tokens in context
            context_text = ""
            for result in retrieval_results[:5]:
                context_text += result.chunk.content + "\n"
            context_tokens = len(self.tokenizer.encode(context_text))
            
            # Count tokens in answer
            answer_tokens = len(self.tokenizer.encode(answer))
            
            # Add approximate prompt tokens (instructions, formatting, etc.)
            prompt_overhead = 500
            
            total_tokens = question_tokens + context_tokens + answer_tokens + prompt_overhead
            
            return total_tokens
            
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            return 1000  # Estimate
    
    async def extract_structured_query(self, natural_query: str) -> Dict[str, Any]:
        """Extract structured information from natural language query"""
        try:
            extraction_prompt = f"""Extract key information from this query for document search:

Query: "{natural_query}"

Extract and return a JSON object with:
- "intent": The main intent (coverage, waiting_period, exclusions, benefits, conditions, etc.)
- "entities": List of key entities mentioned (surgery types, conditions, amounts, etc.)
- "question_type": Type of question (yes_no, specific_info, comparison, explanation)
- "domain": Document domain (insurance, legal, hr, compliance)

Example:
{{"intent": "coverage", "entities": ["knee surgery"], "question_type": "yes_no", "domain": "insurance"}}

JSON:"""

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query analysis expert. Extract structured information from natural language queries and return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": extraction_prompt
                    }
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            
            # Try to parse JSON
            try:
                structured_query = json.loads(result)
                return structured_query
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return {
                    "intent": "general_query",
                    "entities": [],
                    "question_type": "explanation",
                    "domain": "general"
                }
            
        except Exception as e:
            logger.error(f"Error extracting structured query: {str(e)}")
            return {
                "intent": "general_query",
                "entities": [],
                "question_type": "explanation",
                "domain": "general"
            }