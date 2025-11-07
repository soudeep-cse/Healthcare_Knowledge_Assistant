from fastapi import APIRouter, HTTPException
from app.schemas.models import RetrieveRequest, RetrieveResponse
from app.services.medical_service import MedicalChatbotService

router = APIRouter()
service = MedicalChatbotService()

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_documents(request: RetrieveRequest):
    try:
        documents = service.retrieve_documents(request.query, request.k)
        return RetrieveResponse(
            query=request.query,
            documents=documents
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))