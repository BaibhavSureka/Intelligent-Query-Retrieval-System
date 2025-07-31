from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: str  # URL or list of URLs
    questions: List[str]

class Clause(BaseModel):
    text: str
    location: Optional[str] = None

class Answer(BaseModel):
    answer: str
    supporting_clauses: List[Clause]
    decision_rationale: str
    confidence_score: Optional[float] = None

class QueryResponse(BaseModel):
    answers: List[str]