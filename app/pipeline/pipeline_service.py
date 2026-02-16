import logging
from typing import List, Dict, Any
import asyncio
import uuid
import json
import os

from app.services.google_news_fetcher import GoogleNewsFetcher
from app.services.article_scraper import ScraperConfig
from app.services.web_scraper import WebScraper
from app.services.vector_service import QdrantVectorDBClient
from app.services.summarization import summarize_articles

from app.services.Image_generator import (
    to_safe_concept_prompt,
    call_image_api,
    extract_image_url,
    save_image_from_url,
)

from app.infrastructure.cosmos_db_client import CosmosDBClient

logger = logging.getLogger(__name__)


class PipelineService:

    def __init__(self):
        self.scraped_results: List[Dict[str, Any]] = []
        self.summarized_results: List[Dict[str, Any]] = []

    async def run(self) -> None:
        logger.info("Pipeline started")

        await self._scrape_stage()
        await self._vector_stage()
        await self._summarization_stage()
        await self._image_stage()
        await self._cosmos_stage()

        logger.info("Pipeline finished")

    async def _scrape_stage(self):
        logger.info("Starting scraping stage")

        config = ScraperConfig(
            headless=False,
            max_articles=10
        )

        fetcher = GoogleNewsFetcher()

        async def url_fetcher(exclude_urls):
            return await fetcher.fetch_articles(
                num=10,
                exclude_urls=exclude_urls
            )

        scraper = WebScraper(
            config=config,
            url_fetcher=url_fetcher
        )

        results = await scraper.scrape_all()

        successful = [
            r for r in results
            if not r.get("skipped") and r.get("text")
        ]

        if len(successful) < 10:
            raise RuntimeError(
                f"Pipeline aborted: only {len(successful)} successful articles scraped."
            )

        self.scraped_results = successful[:10]

        logger.info("Scraping completed. 10 successful articles locked.")

    async def _vector_stage(self):
        logger.info("Starting vector ingestion stage")

        if not self.scraped_results:
            raise RuntimeError("Vector stage aborted: no scraped results available.")

        vdb_client = QdrantVectorDBClient()

        await vdb_client.test_connection()
        await vdb_client.ingest_scraped_results(self.scraped_results)

        logger.info("Vector ingestion completed successfully.")

    async def _summarization_stage(self):
        logger.info("Starting summarization stage")

        vector_service = QdrantVectorDBClient()
        all_articles = await vector_service.fetch_all_articles_payload()

        if not all_articles:
            raise RuntimeError("Summarization aborted: no articles found in Qdrant.")

        self.summarized_results = summarize_articles(all_articles)

        logger.info(
            "Summarization completed: %s articles processed",
            len(self.summarized_results),
        )

    async def _image_stage(self):
        logger.info("Starting image generation stage")

        if not self.summarized_results:
            raise RuntimeError(
                "Image stage aborted: no summarized results available."
            )

        processed = 0

        for article in self.summarized_results:
            try:
                title = (article.get("title") or "").strip()

                if not title:
                    article["image_url"] = ""
                    article["local_image_path"] = ""
                    continue

                curated_prompt = to_safe_concept_prompt(title)

                raw = await asyncio.to_thread(
                    call_image_api,
                    curated_prompt
                )

                image_url, is_default = extract_image_url(raw)

                if image_url:
                    saved = await asyncio.to_thread(
                        save_image_from_url,
                        image_url,
                        title
                    )

                    article["image_url"] = "" if is_default else image_url
                    article["local_image_path"] = saved or ""
                else:
                    article["image_url"] = ""
                    article["local_image_path"] = ""

                processed += 1

            except Exception as e:
                logger.warning(
                    "Image generation failed for article %s: %s",
                    article.get("id"),
                    e
                )
                article["image_url"] = ""
                article["local_image_path"] = ""

        logger.info(
            "Image generation completed for %s articles",
            processed
        )

    async def _cosmos_stage(self):
        logger.info("Starting Cosmos DB stage")

        if not self.summarized_results:
            raise RuntimeError("Cosmos stage aborted: no summarized results available.")

        cosmos = CosmosDBClient()
        await cosmos.init()

        try:
            deleted = await cosmos.reset_container()
            logger.info("Deleted %s existing articles from Cosmos DB", deleted)

            upserted = 0
            cosmos_written_items = []

            for article in self.summarized_results:
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
                        "source": article.get("source", "")
                    }

                    await cosmos.upsert_news(cosmos_item)

                    cosmos_written_items.append(cosmos_item)
                    upserted += 1

                except Exception as e:
                    logger.warning(
                        "Failed to upsert article %s: %s",
                        article.get("id"),
                        e
                    )

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, "summarized_articles.json")

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cosmos_written_items, f, indent=2, ensure_ascii=False)

            logger.info(
                "Cosmos DB stage completed. %s articles inserted and written to JSON.",
                upserted
            )

        finally:
            await cosmos.close()
