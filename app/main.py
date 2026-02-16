from contextlib import asynccontextmanager
from app.infrastructure.cosmos_db_client import CosmosDBClient
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schema import RaisedException

from app.utils.fast_api import  include_applications

from app.utils.error_handler import raised_exception_handler

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

tags_metadata = [
    {
        "name": "health",
        "description": "Health check endpoints",
    },
    {
        "name": "articles",
        "description": "News articles endpoints",
    },
    {
        "name": "scraper",
        "description": "Web scraper and vector ingestion endpoints",
    },
]


def create_app() -> FastAPI:
    """
    Purpose: Initialize and configure the FastAPI application
    Unified pipeline for fetching, scraping, and vectorizing AI news
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """
        To ensure CosmosDBClient is initialized and available in app.state for the entire lifespan of the application.
        """
        cosmos_client = CosmosDBClient()
        await cosmos_client.init()
        app.state.cosmos_client = cosmos_client
        yield
        

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description="AI Pulse Backend API - Aggregates AI news from credible sources",
        terms_of_service="https://example.com/terms",
        contact={
            "name": "AI Pulse Team",
            "url": "https://example.com",
            "email": "support@example.com",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        openapi_tags=tags_metadata,
        debug=settings.debug,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ... exception handlers moved to utils/error_handler.py ...
    app.add_exception_handler(RaisedException, raised_exception_handler)

    # ...routes moved to routers/health.py...

    logger.info(f"Loading routers from {settings.api_base}...")
    include_applications(app, api_base=settings.api_base, router_package="app.routers")

    logger.info(f"Application {settings.app_name} initialized successfully")
    return app


app = create_app()
