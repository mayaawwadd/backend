import logging
from typing import List, Dict, Any

from app.infrastructure.vector.vector_repository import QdrantVectorRepository

logger = logging.getLogger(__name__)


async def run_vector_stage(scraped_results: List[Dict[str, Any]]) -> None:
    logger.info("Starting vector ingestion stage")

    if not scraped_results:
        raise RuntimeError("Vector stage aborted: no scraped results available.")

    vector_repo = QdrantVectorRepository()

    try:
        await vector_repo.ingest_scraped_results(scraped_results)

        logger.info(
            "Vector ingestion completed successfully. %s documents stored.",
            len(scraped_results),
        )

    except Exception as e:
        logger.error("Vector ingestion failed: %s", e)
        raise