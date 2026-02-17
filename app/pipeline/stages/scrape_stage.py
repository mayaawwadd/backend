import logging
from typing import List, Dict, Any

from app.services.google_news_fetcher import GoogleNewsFetcher
from app.services.article_scraper import ScraperConfig
from app.services.web_scraper import WebScraper

logger = logging.getLogger(__name__)


async def run_scrape_stage() -> List[Dict[str, Any]]:
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

    logger.info("Scraping completed. 10 successful articles locked.")

    return successful[:10]