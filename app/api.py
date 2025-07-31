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

app = FastAPI()

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
        
        # 3. Embed chunks
        try:
            embeddings = embed_chunks(chunks)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")
        
        # 4. Build vector store
        dim = len(embeddings[0])
        vector_store = VectorStore(dim)
        vector_store.add(embeddings, chunks)
        
        # 5. For each question, retrieve and answer
        answers = []
        detailed_answers = []
        
        for q in request.questions:
            try:
                relevant_chunks = retrieve_relevant_chunks(q, vector_store)
                result = answer_question(q, relevant_chunks)
                
                # Extract simple answer for backward compatibility
                answers.append(result["answer"])
                
                # Create detailed answer object
                detailed_answer = Answer(
                    answer=result["answer"],
                    supporting_clauses=result["supporting_clauses"],
                    decision_rationale=result["decision_rationale"],
                    confidence_score=result["confidence_score"]
                )
                detailed_answers.append(detailed_answer)
                
            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                answers.append(error_msg)
                detailed_answers.append(Answer(
                    answer=error_msg,
                    supporting_clauses=[],
                    decision_rationale="An error occurred during processing",
                    confidence_score=0.0
                ))
        
        processing_time = round(time.time() - start_time, 2)
        
        return QueryResponse(
            answers=answers,
            detailed_answers=detailed_answers,
            processing_time=processing_time,
            document_processed=request.documents
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")