import logging
from typing import List, Dict, Any

from app.services.google_news_fetcher import GoogleNewsFetcher
from app.services.article_scraper import ScraperConfig
from app.services.web_scraper import WebScraper
from app.services.vector_service import QdrantVectorDBClient

from app.services.summarization import summarize_articles
from qdrant_client import QdrantClient
from app.config import settings

from app.services.Image_generator import (
    to_safe_concept_prompt,
    call_image_api,
    extract_image_url,
    save_image_from_url,
)

import asyncio


logger = logging.getLogger(__name__)


class PipelineService:

    def __init__(self):
        self.scraped_results: List[Dict[str, Any]] = []

    async def run(self) -> None:
        logger.info("Pipeline started")

        await self._scrape_stage()
        await self._vector_stage()
        await self._summarization_stage()

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

        # Keep only first 10
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

        qdrant_client = QdrantClient(url=settings.qdrant_url or "http://localhost:6333")
        collection_name = settings.qdrant_collection

        all_articles = []
        next_page = None

        while True:
            points, next_page = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=None,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=next_page,
            )

            if not points:
                break

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                flat = {**payload, **metadata}
                all_articles.append(flat)

            if next_page is None:
                break

        if not all_articles:
            raise RuntimeError("Summarization aborted: no articles found in Qdrant.")

        self.summarized_results = summarize_articles(all_articles)

        logger.info("Summarization completed: %s articles processed", len(self.summarized_results))

    async def _image_stage(self):
        logger.info("Starting image generation stage")

        if not self.summarized_results:
            raise RuntimeError("Image stage aborted: no summarized results available.")

        processed = 0

        for article in self.summarized_results:
            try:
                title = (article.get("title") or "").strip()
                text = (article.get("text") or "").strip()

                if not title or not text:
                    article["image_url"] = ""
                    article["local_image_path"] = ""
                    continue

                curated_prompt = to_safe_concept_prompt(title)

                # 🔥 Run blocking image API call in background thread
                raw = await asyncio.to_thread(call_image_api, curated_prompt)

                image_url, is_default = extract_image_url(raw)

                if image_url:
                    saved = await asyncio.to_thread(
                        save_image_from_url,
                        image_url,
                        title
                    )

                    if is_default:
                        article["image_url"] = ""
                        article["local_image_path"] = saved or ""
                    else:
                        article["image_url"] = image_url
                        article["local_image_path"] = saved or ""
                else:
                    article["image_url"] = ""
                    article["local_image_path"] = ""

                processed += 1

            except Exception as e:
                logger.warning(
                    f"Image generation failed for article {article.get('id')}: {e}"
                )
                article["image_url"] = ""
                article["local_image_path"] = ""

        logger.info("Image generation completed for %s articles", processed)


