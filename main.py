import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from models import (
    QueryRequest, QueryResponse, HealthCheck, ErrorResponse
)
from query_processor import QueryProcessor
from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global query processor instance
query_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global query_processor
    
    # Startup
    logger.info("Starting LLM-Powered Query Retrieval System...")
    try:
        query_processor = QueryProcessor()
        logger.info("Query processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize query processor: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="A comprehensive system for processing documents and answering queries using LLM and vector search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the bearer token"""
    if credentials.credentials != settings.bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def get_query_processor() -> QueryProcessor:
    """Dependency to get query processor instance"""
    if query_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query processor not initialized"
        )
    return query_processor

# API Endpoints

@app.get("/", response_model=HealthCheck)
async def root():
    """Root endpoint with health check"""
    return HealthCheck(
        status="healthy",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        version="1.0.0"
    )

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx_submission(
    request: QueryRequest,
    token: str = Depends(verify_token),
    processor: QueryProcessor = Depends(get_query_processor)
):
    """
    Main endpoint for HackRx submission processing
    
    Processes a document and answers multiple questions about it.
    """
    try:
        start_time = time.time()
        
        logger.info(f"Received HackRx submission with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Validate request
        if not request.documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document URL is required"
            )
        
        if not request.questions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one question is required"
            )
        
        if len(request.questions) > 50:  # Reasonable limit
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many questions. Maximum 50 questions allowed."
            )
        
        # Process the request
        response = await processor.process_query_request(request)
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully processed submission in {processing_time:.2f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing HackRx submission: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/v1/documents/{document_url:path}/info")
async def get_document_info(
    document_url: str,
    token: str = Depends(verify_token),
    processor: QueryProcessor = Depends(get_query_processor)
):
    """Get information about a processed document"""
    try:
        # URL decode the document_url
        import urllib.parse
        decoded_url = urllib.parse.unquote(document_url)
        
        info = await processor.get_document_info(decoded_url)
        return info
        
    except Exception as e:
        logger.error(f"Error getting document info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving document info: {str(e)}"
        )

@app.delete("/api/v1/documents/{document_url:path}")
async def delete_document(
    document_url: str,
    token: str = Depends(verify_token),
    processor: QueryProcessor = Depends(get_query_processor)
):
    """Delete a document from cache and vector store"""
    try:
        # URL decode the document_url
        import urllib.parse
        decoded_url = urllib.parse.unquote(document_url)
        
        success = await processor.clear_document_cache(decoded_url)
        
        if success:
            return {"message": f"Document deleted successfully: {decoded_url}"}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )

@app.delete("/api/v1/cache")
async def clear_cache(
    token: str = Depends(verify_token),
    processor: QueryProcessor = Depends(get_query_processor)
):
    """Clear all document cache"""
    try:
        await processor.clear_document_cache()
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing cache: {str(e)}"
        )

@app.post("/api/v1/query")
async def process_single_query(
    query_data: dict,
    token: str = Depends(verify_token),
    processor: QueryProcessor = Depends(get_query_processor)
):
    """Process a single query against a document (for testing/debugging)"""
    try:
        document_url = query_data.get("document_url")
        question = query_data.get("question")
        
        if not document_url or not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both document_url and question are required"
            )
        
        # Create a query request with single question
        request = QueryRequest(
            documents=document_url,
            questions=[question]
        )
        
        response = await processor.process_query_request(request)
        
        return {
            "document_url": document_url,
            "question": question,
            "answer": response.answers[0] if response.answers else "No answer generated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing single query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/api/v1/config")
async def get_config(token: str = Depends(verify_token)):
    """Get current configuration (non-sensitive data only)"""
    return {
        "api_version": settings.api_version,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "use_faiss": settings.use_faiss,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "log_level": settings.log_level
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred"
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=getattr(exc, 'detail', None)
        ).dict()
    )

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower()
    )