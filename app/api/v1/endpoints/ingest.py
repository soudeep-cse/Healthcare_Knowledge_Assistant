from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import os
import tempfile
from app.schemas.models import IngestResponse
from app.services.medical_service import MedicalChatbotService

router = APIRouter()
service = MedicalChatbotService()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    try:
        temp_paths = []
        uploaded_files = []
        
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                content = await file.read()
                tmp.write(content)
                temp_paths.append(tmp.name)
                uploaded_files.append(file.filename)
        
        docs_count = service.ingest_documents(temp_paths)
        
        for path in temp_paths:
            os.unlink(path)
            
        return IngestResponse(
            message="Documents ingested successfully",
            documents_processed=docs_count,
            files_uploaded=uploaded_files
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))