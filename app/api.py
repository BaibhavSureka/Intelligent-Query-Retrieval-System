from fastapi import FastAPI, HTTPException, Depends, Header
from .schemas import QueryRequest, QueryResponse, Answer
from .document_loader import load_document
from .chunker import chunk_text
from .embedder import embed_chunks
from .vector_store import VectorStore
from .retriever import retrieve_relevant_chunks
from .logic import answer_question
import time
from typing import List
import hashlib

app = FastAPI()

# Simple in-memory cache for embeddings
embedding_cache = {}

def get_cache_key(text: str) -> str:
    """Generate cache key for text"""
    return hashlib.md5(text.encode()).hexdigest()

async def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.replace("Bearer ", "")
    expected_token = "88def649851a3e0861a60905001a92f0a9cdd621ade7686aa6be07cc91f1ed9b"
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

def validate_input(request: QueryRequest):
    """Validate input parameters"""
    if not request.documents:
        raise HTTPException(status_code=400, detail="Document URL is required")
    
    if not request.questions or len(request.questions) == 0:
        raise HTTPException(status_code=400, detail="At least one question is required")
    
    # Validate document URL format
    if not request.documents.startswith(('http://', 'https://')):
        raise HTTPException(status_code=400, detail="Invalid document URL format")
    
    # Validate questions
    for i, question in enumerate(request.questions):
        if not question.strip():
            raise HTTPException(status_code=400, detail=f"Question {i+1} cannot be empty")

@app.get("/")
async def root():
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "description": "Universal document analysis and question answering system",
        "status": "running",
        "endpoint": "/api/v1/hackrx/run",
        "version": "1.0.0",
        "supported_formats": ["PDF", "DOCX", "Email"],
        "document_types": ["Legal", "Technical", "Academic", "Business", "Medical", "Any text-based document"]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG Pipeline"}

@app.post("/api/v1/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, token: str = Depends(verify_token)):
    start_time = time.time()
    
    try:
        # Validate input
        validate_input(request)
        
        # 1. Load and parse document
        try:
            text = load_document(request.documents)
            if not text.strip():
                raise HTTPException(status_code=400, detail="Document is empty or could not be parsed")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error loading document: {str(e)}")
        
        # 2. Chunk text
        chunks = chunk_text(text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No meaningful content could be extracted from the document")
        
        # 3. Embed chunks (with caching)
        try:
            # Check cache first
            cache_key = get_cache_key(text)
            if cache_key in embedding_cache:
                embeddings = embedding_cache[cache_key]
            else:
                embeddings = embed_chunks(chunks)
                # Cache the embeddings (limit cache size)
                if len(embedding_cache) < 10:  # Keep only 10 cached documents
                    embedding_cache[cache_key] = embeddings
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")
        
        # 4. Build vector store
        dim = len(embeddings[0])
        vector_store = VectorStore(dim)
        vector_store.add(embeddings, chunks)
        
        # 5. For each question, retrieve and answer
        answers = []
        
        for q in request.questions:
            try:
                relevant_chunks = retrieve_relevant_chunks(q, vector_store)
                result = answer_question(q, relevant_chunks)
                
                # Extract only the answer text for HackRx format and clean it
                answer = result["answer"]
                # Ensure clean response - remove any trailing characters
                answer = answer.strip()
                while answer.endswith(('/', '\\', '|', '-', ' ')):
                    answer = answer.rstrip('/').rstrip('\\').rstrip('|').rstrip('-').rstrip()
                answers.append(answer)
                
            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                answers.append(error_msg)
        
        # Return only the answers array as expected by HackRx
        return {"answers": answers}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")