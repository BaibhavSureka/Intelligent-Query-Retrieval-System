from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field
from datetime import datetime

class QueryRequest(BaseModel):
    """Request model for the /hackrx/run endpoint"""
    documents: str = Field(..., description="URL to the document blob")
    questions: List[str] = Field(..., description="List of questions to answer")

class QueryResponse(BaseModel):
    """Response model for query answers"""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")

class DocumentChunk(BaseModel):
    """Model for document chunks with metadata"""
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the chunk")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_url: str = Field(..., description="Source document URL")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    section: Optional[str] = Field(None, description="Section name if identified")

class EmbeddingVector(BaseModel):
    """Model for embedding vectors"""
    vector: List[float] = Field(..., description="Embedding vector")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Associated metadata")
    chunk_id: str = Field(..., description="Reference to document chunk")

class RetrievalResult(BaseModel):
    """Model for retrieval results"""
    chunk: DocumentChunk = Field(..., description="Retrieved document chunk")
    similarity_score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Ranking position")

class ClauseMatch(BaseModel):
    """Model for clause matching results"""
    clause_text: str = Field(..., description="Matched clause text")
    relevance_score: float = Field(..., description="Relevance score to the query")
    source_location: str = Field(..., description="Location in source document")
    context: str = Field(..., description="Surrounding context")

class DecisionRationale(BaseModel):
    """Model for explainable decision rationale"""
    reasoning: str = Field(..., description="Explanation of the decision logic")
    evidence_chunks: List[RetrievalResult] = Field(..., description="Supporting evidence")
    confidence_score: float = Field(..., description="Confidence in the answer")
    clause_matches: List[ClauseMatch] = Field(default_factory=list, description="Specific clause matches")

class AnswerWithRationale(BaseModel):
    """Enhanced answer model with decision rationale"""
    answer: str = Field(..., description="The generated answer")
    rationale: DecisionRationale = Field(..., description="Decision reasoning and evidence")
    processing_time: float = Field(..., description="Time taken to process the query")
    tokens_used: int = Field(..., description="Number of tokens consumed")

class HealthCheck(BaseModel):
    """Health check response model"""
    status: str = Field(default="healthy", description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field(default="1.0.0", description="API version")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")