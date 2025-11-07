from fastapi import APIRouter
from app.api.v1.endpoints import ingest, retrieve, generate

api_router = APIRouter()
api_router.include_router(ingest.router, prefix="/medical", tags=["ingest"])
api_router.include_router(retrieve.router, prefix="/medical", tags=["retrieve"])
api_router.include_router(generate.router, prefix="/medical", tags=["generate"])