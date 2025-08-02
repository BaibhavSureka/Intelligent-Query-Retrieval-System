import asyncio
import logging
import time
from typing import List, Dict, Any, Optional

from document_processor import DocumentProcessor
from vector_store import create_vector_store, VectorStore
from llm_service import LLMService
from models import DocumentChunk, RetrievalResult, QueryRequest, QueryResponse, AnswerWithRationale

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Main service that orchestrates the entire query processing pipeline"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = create_vector_store()
        self.llm_service = LLMService()
        self.document_cache = {}  # Cache processed documents
        
    async def process_query_request(self, request: QueryRequest) -> QueryResponse:
        """
        Process a complete query request with document and questions
        
        Args:
            request: QueryRequest containing document URL and questions
            
        Returns:
            QueryResponse with answers to all questions
        """
        try:
            start_time = time.time()
            
            logger.info(f"Processing query request with {len(request.questions)} questions")
            logger.info(f"Document URL: {request.documents}")
            
            # Step 1: Process and index the document
            await self._ensure_document_indexed(request.documents)
            
            # Step 2: Process each question
            answers = []
            for i, question in enumerate(request.questions):
                logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:100]}...")
                
                answer = await self._process_single_question(question, request.documents)
                answers.append(answer)
            
            processing_time = time.time() - start_time
            logger.info(f"Completed processing {len(request.questions)} questions in {processing_time:.2f}s")
            
            return QueryResponse(answers=answers)
            
        except Exception as e:
            logger.error(f"Error processing query request: {str(e)}")
            raise
    
    async def _ensure_document_indexed(self, document_url: str) -> None:
        """Ensure document is processed and indexed in vector store"""
        try:
            # Check if document is already processed
            if document_url in self.document_cache:
                logger.info(f"Document already indexed: {document_url}")
                return
            
            # Process document
            logger.info(f"Processing and indexing document: {document_url}")
            chunks = await self.document_processor.process_document_from_url(document_url)
            
            if not chunks:
                raise ValueError(f"No content could be extracted from document: {document_url}")
            
            # Add to vector store
            await self.vector_store.add_chunks(chunks)
            
            # Cache the document
            self.document_cache[document_url] = {
                'chunks': chunks,
                'indexed_at': time.time()
            }
            
            logger.info(f"Successfully indexed {len(chunks)} chunks from document")
            
        except Exception as e:
            logger.error(f"Error indexing document {document_url}: {str(e)}")
            raise
    
    async def _process_single_question(self, question: str, document_url: str) -> str:
        """
        Process a single question against the document
        
        Args:
            question: The question to answer
            document_url: URL of the source document
            
        Returns:
            The answer string
        """
        try:
            # Step 1: Extract structured query information
            structured_query = await self.llm_service.extract_structured_query(question)
            logger.debug(f"Structured query: {structured_query}")
            
            # Step 2: Enhance question for better retrieval
            enhanced_question = await self._enhance_question_for_retrieval(question, structured_query)
            
            # Step 3: Retrieve relevant chunks
            retrieval_results = await self.vector_store.search(enhanced_question, top_k=10)
            
            if not retrieval_results:
                logger.warning(f"No relevant chunks found for question: {question}")
                return "I couldn't find relevant information in the document to answer this question."
            
            # Step 4: Filter results by document URL (if multiple documents indexed)
            filtered_results = [
                result for result in retrieval_results 
                if result.chunk.document_url == document_url
            ]
            
            if not filtered_results:
                logger.warning(f"No relevant chunks found in specified document for question: {question}")
                return "I couldn't find relevant information in the specified document to answer this question."
            
            # Step 5: Re-rank results based on question relevance
            reranked_results = await self._rerank_results(question, filtered_results, structured_query)
            
            # Step 6: Process with LLM to generate answer
            answer_with_rationale = await self.llm_service.process_query(
                question, reranked_results[:5], document_url
            )
            
            # Step 7: Post-process answer for consistency
            final_answer = self._post_process_answer(answer_with_rationale.answer, question)
            
            logger.info(f"Generated answer with confidence: {answer_with_rationale.rationale.confidence_score:.2f}")
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {str(e)}")
            return f"I encountered an error while processing this question: {str(e)}"
    
    async def _enhance_question_for_retrieval(self, question: str, structured_query: Dict[str, Any]) -> str:
        """Enhance question with domain-specific terms for better retrieval"""
        try:
            # Add domain-specific terms based on intent
            enhancement_terms = []
            
            intent = structured_query.get('intent', '')
            domain = structured_query.get('domain', '')
            entities = structured_query.get('entities', [])
            
            # Domain-specific enhancements
            if domain == 'insurance':
                if intent == 'coverage':
                    enhancement_terms.extend(['coverage', 'benefits', 'covered', 'includes'])
                elif intent == 'waiting_period':
                    enhancement_terms.extend(['waiting period', 'grace period', 'months', 'days'])
                elif intent == 'exclusions':
                    enhancement_terms.extend(['exclusions', 'excluded', 'not covered', 'limitations'])
            
            # Add entity synonyms
            entity_expansions = {
                'knee surgery': ['knee', 'orthopedic', 'surgical procedure'],
                'maternity': ['pregnancy', 'childbirth', 'delivery'],
                'cataract': ['eye surgery', 'vision', 'ophthalmology'],
                'AYUSH': ['alternative medicine', 'ayurveda', 'homeopathy'],
            }
            
            for entity in entities:
                if entity.lower() in entity_expansions:
                    enhancement_terms.extend(entity_expansions[entity.lower()])
            
            # Combine original question with enhancement terms
            if enhancement_terms:
                enhanced_question = f"{question} {' '.join(enhancement_terms)}"
            else:
                enhanced_question = question
            
            return enhanced_question
            
        except Exception as e:
            logger.error(f"Error enhancing question: {str(e)}")
            return question
    
    async def _rerank_results(
        self,
        question: str,
        results: List[RetrievalResult],
        structured_query: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Re-rank results based on question-specific criteria"""
        try:
            # Calculate additional relevance scores
            for result in results:
                additional_score = 0.0
                content = result.chunk.content.lower()
                question_lower = question.lower()
                
                # Boost score for exact entity matches
                entities = structured_query.get('entities', [])
                for entity in entities:
                    if entity.lower() in content:
                        additional_score += 0.1
                
                # Boost score for intent-related keywords
                intent = structured_query.get('intent', '')
                intent_keywords = {
                    'coverage': ['coverage', 'covers', 'covered', 'benefits', 'includes'],
                    'waiting_period': ['waiting', 'period', 'grace', 'months', 'days'],
                    'exclusions': ['exclusion', 'excluded', 'not covered', 'limitations'],
                    'conditions': ['conditions', 'requirements', 'terms'],
                    'benefits': ['benefits', 'discount', 'allowance', 'reimbursement']
                }
                
                if intent in intent_keywords:
                    for keyword in intent_keywords[intent]:
                        if keyword in content:
                            additional_score += 0.05
                
                # Boost score for question word matches
                question_words = set(question_lower.split())
                content_words = set(content.split())
                word_overlap = len(question_words.intersection(content_words))
                additional_score += word_overlap * 0.01
                
                # Update similarity score
                result.similarity_score = min(1.0, result.similarity_score + additional_score)
            
            # Sort by updated similarity score
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Update ranks
            for i, result in enumerate(results):
                result.rank = i + 1
            
            return results
            
        except Exception as e:
            logger.error(f"Error re-ranking results: {str(e)}")
            return results
    
    def _post_process_answer(self, answer: str, question: str) -> str:
        """Post-process answer for consistency and clarity"""
        try:
            # Remove any system prompts or artifacts
            answer = answer.strip()
            
            # Remove common LLM artifacts
            artifacts_to_remove = [
                "Based on the provided context:",
                "According to the document:",
                "The document states:",
                "**Answer:**",
                "Answer:"
            ]
            
            for artifact in artifacts_to_remove:
                answer = answer.replace(artifact, "").strip()
            
            # Ensure answer starts with capital letter
            if answer and not answer[0].isupper():
                answer = answer[0].upper() + answer[1:]
            
            # Ensure answer ends with proper punctuation
            if answer and answer[-1] not in '.!?':
                answer += '.'
            
            return answer
            
        except Exception as e:
            logger.error(f"Error post-processing answer: {str(e)}")
            return answer
    
    async def get_document_info(self, document_url: str) -> Dict[str, Any]:
        """Get information about a processed document"""
        try:
            if document_url not in self.document_cache:
                return {"error": "Document not found in cache"}
            
            doc_info = self.document_cache[document_url]
            chunks = doc_info['chunks']
            
            return {
                "document_url": document_url,
                "num_chunks": len(chunks),
                "total_characters": sum(len(chunk.content) for chunk in chunks),
                "total_words": sum(chunk.metadata.get('word_count', 0) for chunk in chunks),
                "indexed_at": doc_info['indexed_at'],
                "chunk_details": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "character_count": chunk.metadata.get('character_count', 0),
                        "word_count": chunk.metadata.get('word_count', 0),
                    }
                    for chunk in chunks[:10]  # Show first 10 chunks
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}")
            return {"error": str(e)}
    
    async def clear_document_cache(self, document_url: Optional[str] = None) -> bool:
        """Clear document cache"""
        try:
            if document_url:
                if document_url in self.document_cache:
                    del self.document_cache[document_url]
                    # Also remove from vector store
                    await self.vector_store.delete_document(document_url)
                    logger.info(f"Cleared cache for document: {document_url}")
                    return True
                else:
                    logger.warning(f"Document not found in cache: {document_url}")
                    return False
            else:
                # Clear all cache
                self.document_cache.clear()
                logger.info("Cleared all document cache")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing document cache: {str(e)}")
            return False