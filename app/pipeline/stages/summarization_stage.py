import logging
from typing import List, Dict, Any

from app.infrastructure.vector.vector_repository import QdrantVectorRepository
from app.infrastructure.llm.article_summarizer import ArticleSummarizer

logger = logging.getLogger(__name__)


async def run_summarization_stage() -> List[Dict[str, Any]]:
    logger.info("Starting summarization stage")

    vector_repo = QdrantVectorRepository()
    summarizer = ArticleSummarizer()

    all_articles = await vector_repo.fetch_all_articles_payload()

    if not all_articles:
        raise RuntimeError("Summarization aborted: no articles found in Qdrant.")

    summarized_results: List[Dict[str, Any]] = []

    for article in all_articles:
        try:
            enriched_article = await summarizer.summarize_article(article)
            if enriched_article:
                summarized_results.append(enriched_article)

        except Exception as e:
            logger.warning(
                "Failed to summarize article %s: %s",
                article.get("id"),
                e,
            )

    logger.info(
        "Summarization completed: %s articles processed",
        len(summarized_results),
    )

    return summarized_results
