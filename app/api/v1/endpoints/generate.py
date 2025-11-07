from fastapi import APIRouter, HTTPException
from langdetect import detect
from app.schemas.models import GenerateRequest, GenerateResponse
from app.services.medical_service import MedicalChatbotService

router = APIRouter()
service = MedicalChatbotService()

@router.post("/generate", response_model=GenerateResponse)
async def generate_answer(request: GenerateRequest):
    try:
        answer = service.generate_answer(
            request.query, 
            request.k, 
            request.output_language
        )
        try:
            detected_lang = "ja" if detect(answer) == "ja" else "en"
        except:
            detected_lang = "en"
        
        return GenerateResponse(
            query=request.query,
            answer=answer,
            language=detected_lang
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))