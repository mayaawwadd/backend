import logging
from typing import Any, Callable, Dict, List, Optional, Set, Awaitable

from app.services.article_scraper import PlaywrightArticleScraper, ScraperConfig
from app.services.vector_service import VectorDBClient

logger = logging.getLogger(__name__)


class WebScraper:

    def __init__(
        self,
        config: ScraperConfig,
        url_fetcher: Callable[[Set[str]], Awaitable[List[Dict[str, Any]]]],
        vdb_client: Optional[VectorDBClient] = None,
    ) -> None:
        self.config = config
        self.fetch_urls = url_fetcher
        self.scraper = PlaywrightArticleScraper(
            headless=config.headless,
            selectors=config.selectors,
            viewport=config.viewport,
            user_agent=config.user_agent,
        )
        self.vdb = vdb_client

    async def scrape_all(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()
        target_success_count = 15
        success_count = 0

        max_fetch_rounds = 5  
        fetch_round = 0

        while success_count < target_success_count and fetch_round < max_fetch_rounds:

            fetch_round += 1

            records = await self.fetch_urls(seen_urls)

            if not records:
                logger.warning("No records returned from fetcher.")
                break

            for record in records:
                url = record.get("link") if isinstance(record, dict) else record

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)

                if success_count >= target_success_count:
                    break

                try:
                    res = await self.scraper.scrape(url)
                except Exception as e:
                    logger.warning("Scrape failed for %s: %s", url, e)
                    continue

                res_dict = res.to_dict()

                if isinstance(record, dict):
                    field_map = {
                        "unique_id": "gn_id",
                        "title": "gn_title",
                        "iso_date": "published_at",
                        "position": "gn_position",
                        "source": "source",
                    }

                    for src_key, dst_key in field_map.items():
                        if src_key in record and (
                            dst_key not in res_dict
                            or res_dict.get(dst_key) in (None, "")
                        ):
                            res_dict[dst_key] = record.get(src_key)

                results.append(res_dict)

                if not res.skipped and res.text and res.text.strip():
                    success_count += 1

            logger.info(
                "Fetch round %s complete. Success count: %s/%s",
                fetch_round,
                success_count,
                target_success_count,
            )

        if fetch_round >= max_fetch_rounds and success_count < target_success_count:
            logger.warning(
                "Stopped due to max_fetch_rounds limit. Success: %s/%s",
                success_count,
                target_success_count,
            )

        return results