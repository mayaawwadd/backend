"""
Scraper API router for orchestrating the complete pipeline.
Handles: fetch -> scrape -> embed -> deduplicate -> store
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings
from app.run import fetch_urls_via_google_news
from app.services.web_scraper import WebScraper
from app.services.article_scraper import ScraperConfig
from app.services.vector_service import QdrantVectorDBClient
from app.services.google_news_fetcher import GoogleNewsFetcher

logger = logging.getLogger(__name__)


class ScrapeRequest(BaseModel):
    """Request model for scraping operation."""
    num_articles: int = Field(10, ge=1, le=50, description="Number of articles to fetch and scrape")
    use_vector_db: bool = Field(True, description="Whether to store results in Qdrant")
    exclude_urls: Optional[List[str]] = Field(None, description="URLs to exclude from scraping")

    class Config:
        json_schema_extra = {
            "example": {
                "num_articles": 10,
                "use_vector_db": True,
                "exclude_urls": ["https://example.com/old-article"]
            }
        }


class ArticleResult(BaseModel):
    """Single article result."""
    url: str
    title: str
    text_length: int
    metadata: Dict[str, Any]


class ScrapeResponse(BaseModel):
    """Response model for scraping operation."""
    status: str
    articles_fetched: int
    articles_scraped: int
    articles_stored: int
    duplicates_found: int
    failed_count: int
    articles: List[ArticleResult] = []
    vdb_stats: Optional[Dict[str, Any]] = None


router = APIRouter(prefix="/scrape", tags=["scraper"])

DEFAULT_SCRAPER_CONFIG = ScraperConfig(
    output_json="output/articles.json",
    raw_text_dir="output/raw_text",
    headless=True,
    max_articles=50,
)


def create_vdb_client(use_vector_db: bool) -> Any:
    """Create appropriate VDB client based on settings."""
    if not use_vector_db:
        logger.info("Vector DB disabled; no storage will occur.")
        return None
    
    try:
        settings = Settings()
        client = QdrantVectorDBClient(
            qdrant_url="http://localhost:6333",
            collection_name="articles",
            embedding_model="text-embedding-3-small",
            similarity_threshold=0.8,
            use_azure=False,
        )
        logger.info("Using QdrantVectorDBClient for vector storage")
        return client
    except Exception as e:
        logger.warning(f"Failed to connect to Qdrant, falling back to NoOp: {e}")
        return None


@router.post("/", response_model=ScrapeResponse, summary="Scrape news articles")
async def scrape_articles(request: ScrapeRequest) -> ScrapeResponse:
    """
    Execute the complete scraping pipeline.
    
    Process:
    1. Fetch article URLs from Google News
    2. Scrape full article content using Playwright
    3. Generate embeddings and store in Qdrant (optional)
    4. Return results with deduplication stats
    
    Query Parameters:
    - num_articles: How many articles to process (1-50)
    - use_vector_db: Whether to store in Qdrant (default: true)
    - exclude_urls: URLs to skip (optional)
    """
    logger.info(
        f"[SCRAPE] Starting pipeline: fetching {request.num_articles} articles, "
        f"VDB={request.use_vector_db}"
    )
    
    try:
        vdb_client = create_vdb_client(request.use_vector_db)
        
        exclude_urls = set(request.exclude_urls or [])
        
        async def fetch_urls() -> List[str]:
            """Fetch URLs from Google News."""
            fetcher = GoogleNewsFetcher()
            records = await fetcher.fetch_articles(
                num=request.num_articles,
                exclude_urls=exclude_urls
            )
            urls = [r["link"] for r in records if isinstance(r, dict) and r.get("link")]
            logger.info(f"[SCRAPE] Fetched {len(urls)} URLs from Google News")
            return urls
        
        loop = asyncio.get_event_loop()
        urls = await fetch_urls()
        
        if not urls:
            return ScrapeResponse(
                status="error",
                articles_fetched=0,
                articles_scraped=0,
                articles_stored=0,
                duplicates_found=0,
                failed_count=0,
                vdb_stats=None,
            )
        
        def sync_url_fetcher() -> List[str]:
            return urls
        
        scraper_config = ScraperConfig(
            output_json=DEFAULT_SCRAPER_CONFIG.output_json,
            raw_text_dir=DEFAULT_SCRAPER_CONFIG.raw_text_dir,
            headless=True,
            max_articles=len(urls),
        )
        
        scraper = WebScraper(
            config=scraper_config,
            url_fetcher=sync_url_fetcher,
            vdb_client=vdb_client,
        )
        
        logger.info(f"[SCRAPE] Starting web scraping for {len(urls)} URLs")
        await scraper.scrape_all()
        
        vdb_stats = None
        if request.use_vector_db and hasattr(vdb_client, 'fetch_collection_stats'):
            try:
                vdb_stats = await vdb_client.fetch_collection_stats()
                logger.info(f"[SCRAPE] VDB stats: {vdb_stats}")
            except Exception as e:
                logger.warning(f"Failed to get VDB stats: {e}")
        
        logger.info("[SCRAPE] Pipeline completed successfully")
        return ScrapeResponse(
            status="success",
            articles_fetched=len(urls),
            articles_scraped=len(urls),
            articles_stored=vdb_stats.get("points_count", 0) if vdb_stats else 0,
            duplicates_found=0,
            failed_count=0,
            articles=[],
            vdb_stats=vdb_stats,
        )
        
    except Exception as e:
        logger.error(f"[SCRAPE] Pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Scraping pipeline failed: {str(e)}"
        )


@router.get("/status", summary="Get scraper status and Qdrant collection stats")
async def get_status() -> Dict[str, Any]:
    """
    Get current status of scraper and vector database.
    
    Returns:
    - qdrant_connected: Whether Qdrant is accessible
    - collection_stats: Statistics about the articles collection
    - openai_configured: Whether OpenAI/Azure OpenAI is configured
    """
    try:
        status = {
            "status": "operational",
            "qdrant": None,
            "openai": None,
        }
        
        try:
            vdb_client = QdrantVectorDBClient(
                qdrant_url="http://localhost:6333",
                collection_name="articles",
                use_azure=False,
            )
            stats = await vdb_client.fetch_collection_stats()
            status["qdrant"] = {
                "connected": True,
                "stats": stats,
            }
            logger.info("Qdrant status: connected")
        except Exception as e:
            status["qdrant"] = {
                "connected": False,
                "error": str(e),
            }
            logger.warning(f"Qdrant connection failed: {e}")
        
        import os
        has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
        has_azure_keys = all([
            os.getenv("AZURE_OPENAI_KEY"),
            os.getenv("AZURE_OPENAI_ENDPOINT"),
        ])
        
        status["openai"] = {
            "standard_configured": has_openai_key,
            "azure_configured": has_azure_keys,
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get scraper status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}"
        )


@router.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "scraper-api"}
