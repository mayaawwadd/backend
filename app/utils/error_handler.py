
from fastapi import Request
from fastapi.responses import JSONResponse
from app.schema import RaisedException, ErrorResponse
import logging
import asyncio

logger = logging.getLogger(__name__)

async def http_exception_handler(request: Request, exc: RaisedException) -> JSONResponse:
    error_response = ErrorResponse(
        success=False,
        error=exc.detail,
        message=f"Error on {request.method} {request.url.path}",
        status_code=exc.status_code,
    )
    
    logger.error(
        f"Exception: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(by_alias=True),
    )


def raised_exception_handler(request, exc):
    if isinstance(exc, RaisedException):
        return asyncio.get_event_loop().run_until_complete(http_exception_handler(request, exc))
    return JSONResponse(status_code=500, content={"detail": str(exc)})