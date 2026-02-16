import logging
from typing import List, Dict, Any

from app.services.vector_service import QdrantVectorDBClient

logger = logging.getLogger(__name__)


async def run_vector_stage(scraped_results: List[Dict[str, Any]]) -> None:
    logger.info("Starting vector ingestion stage")

    if not scraped_results:
        raise RuntimeError("Vector stage aborted: no scraped results available.")

    vdb_client = QdrantVectorDBClient()

    await vdb_client.test_connection()
    await vdb_client.ingest_scraped_results(scraped_results)

    logger.info("Vector ingestion completed successfully.")
