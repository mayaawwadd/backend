import logging
from typing import List, Dict, Any

from app.services.vector_service import QdrantVectorDBClient
from app.services.summarization import summarize_articles

logger = logging.getLogger(__name__)


async def run_summarization_stage() -> List[Dict[str, Any]]:
    logger.info("Starting summarization stage")

    vector_service = QdrantVectorDBClient()
    all_articles = await vector_service.fetch_all_articles_payload()

    if not all_articles:
        raise RuntimeError("Summarization aborted: no articles found in Qdrant.")

    summarized = summarize_articles(all_articles)

    logger.info(
        "Summarization completed: %s articles processed",
        len(summarized),
    )

    return summarized