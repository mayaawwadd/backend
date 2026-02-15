from fastapi import APIRouter
from app.config import settings
from app.schema import HealthResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", tags=["health"], response_model=HealthResponse)
async def health():
    logger.debug("Health check requested")
    return HealthResponse(status="ok", version=settings.version)

@router.get("/", tags=["health"])
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.version,
        "api_docs": "/docs",
        "api_base": settings.api_base,
    }
