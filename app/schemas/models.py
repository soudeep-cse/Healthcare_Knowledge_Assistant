from pydantic import BaseModel
from typing import List, Optional

class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    files_uploaded: List[str]

class RetrieveRequest(BaseModel):
    query: str
    k: Optional[int] = 3

class DocumentResult(BaseModel):
    content: str
    distance: float
    similarity_percent: float
    metadata: dict

class RetrieveResponse(BaseModel):
    query: str
    documents: List[DocumentResult]

class GenerateRequest(BaseModel):
    query: str
    k: Optional[int] = 3
    output_language: Optional[str] = None

class GenerateResponse(BaseModel):
    query: str
    answer: str
    language: str