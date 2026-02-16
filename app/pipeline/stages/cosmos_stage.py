import logging
import uuid
import json
from pathlib import Path
from typing import List, Dict, Any

from app.infrastructure.storage.cosmos_db_client import CosmosDBClient

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "summarized_articles.json"


async def run_cosmos_stage(
    articles: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    logger.info("Starting Cosmos DB stage")

    if not articles:
        raise RuntimeError(
            "Cosmos stage aborted: no summarized results available."
        )

    cosmos = CosmosDBClient()
    await cosmos.init()

    cosmos_written_items: List[Dict[str, Any]] = []
    upserted = 0

    try:
        deleted = await cosmos.reset_container()
        logger.info(
            "Deleted %s existing articles from Cosmos DB",
            deleted,
        )

        for article in articles:
            try:
                article_id = (
                    article.get("id")
                    or article.get("unique_id")
                    or str(uuid.uuid4())
                )

                cosmos_item = {
                    "id": str(article_id),
                    "title": article.get("title", ""),
                    "url": article.get("url") or article.get("link", ""),
                    "summary": article.get("summary", ""),
                    "date": (
                        article.get("published_at")
                        or article.get("iso_date")
                        or ""
                    ),
                    "categories": article.get("categories") or ["AI"],
                    "image_url": article.get("image_url", ""),
                    "local_image_path": article.get("local_image_path", ""),
                    "source": article.get("source", ""),
                }

                await cosmos.upsert_news(cosmos_item)
                cosmos_written_items.append(cosmos_item)
                upserted += 1

            except Exception as e:
                logger.warning(
                    "Failed to upsert article %s: %s",
                    article.get("id"),
                    e,
                )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(
                cosmos_written_items,
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(
            "Cosmos DB stage completed. %s articles inserted and written to JSON.",
            upserted,
        )

    finally:
        await cosmos.close()

    return cosmos_written_items