import logging
from typing import List, Dict, Any, Set

from app.services.google_news_fetcher import GoogleNewsFetcher
from app.services.article_scraper import ScraperConfig
from app.services.web_scraper import WebScraper

logger = logging.getLogger(__name__)


TARGET_ARTICLES = 10


async def run_scrape_stage() -> List[Dict[str, Any]]:
    logger.info("Starting scraping stage")

    config = ScraperConfig(
        headless=False,
        max_articles=TARGET_ARTICLES,
    )

    fetcher = GoogleNewsFetcher()

    async def url_fetcher(exclude_urls: Set[str]):
        return await fetcher.fetch_articles(
            num=TARGET_ARTICLES,
            exclude_urls=exclude_urls,
        )

    scraper = WebScraper(
        config=config,
        url_fetcher=url_fetcher,
    )

    results = await scraper.scrape_all()

    successful = [
        r for r in results
        if not r.get("skipped") and r.get("text")
    ]

    if len(successful) < TARGET_ARTICLES:
        raise RuntimeError(
            f"Pipeline aborted: only {len(successful)} successful articles scraped "
            f"(required {TARGET_ARTICLES})."
        )

    logger.info(
        "Scraping completed successfully. %s articles ready for vector stage.",
        TARGET_ARTICLES,
    )

    return successful[:TARGET_ARTICLES]