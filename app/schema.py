"""
Pydantic models for request/response validation.
"""

from typing import Generic, TypeVar, Optional
from pydantic import BaseModel, ConfigDict
from fastapi import HTTPException, status

T = TypeVar("T")


class CamelCaseModel(BaseModel):
    """Base model with standard configuration."""
    model_config = ConfigDict(populate_by_name=True)


class AppResponse(CamelCaseModel, Generic[T]):
    """Generic response wrapper for successful requests."""
    success: bool = True
    data: T
    message: Optional[str] = None
    model_config = ConfigDict(populate_by_name=True)


class ErrorResponse(CamelCaseModel):
    """Error response model."""
    success: bool = False
    error: str
    message: str
    status_code: int
    model_config = ConfigDict(populate_by_name=True)


class RaisedException(HTTPException):
    """Custom HTTPException for consistent error handling."""
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "Internal Server Error",
        headers: Optional[dict] = None,
    ):
        if headers is None:
            super().__init__(status_code=status_code, detail=detail)
        else:
            super().__init__(status_code=status_code, detail=detail, headers=headers)


# Health check response model
class HealthResponse(CamelCaseModel):
    """Health check response."""
    status: str
    version: str
    model_config = ConfigDict(populate_by_name=True)
