import asyncio
import hashlib
import logging
import os
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse
from app.utils.scraper_utils import URLTools
from app.services.google_news_fetcher import GoogleNewsFetcher
from app.services.article_scraper import PlaywrightArticleScraper, ScraperConfig
from app.services.vector_service import VectorDBClient

logger = logging.getLogger(__name__)


class WebScraper:

    def __init__(
        self,
        config: ScraperConfig,
        url_fetcher: Callable[[], List[str]],
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
        seen_urls = set()
        fetcher = GoogleNewsFetcher()
        target_success_count = 10
        success_count = 0

        while success_count < target_success_count:
            request_num = self.config.max_articles if (self.config.max_articles and self.config.max_articles > 0) else 10
            records = await fetcher.fetch_articles(num=request_num, exclude_urls=seen_urls)
            urls = [r["link"] for r in records if isinstance(r, dict) and r.get("link")]

            # Build a mapping of normalized URL -> Google News metadata for merging
            def _norm(u: str) -> str:
                return (u or "").strip().rstrip("/")

            gn_by_url = { _norm(r.get("link", "")): r for r in records if isinstance(r, dict) and r.get("link") }

            if not urls:
                break

            for url in urls:
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                if success_count >= target_success_count:
                    break
                res = await self.scraper.scrape(url)
                res_dict = res.to_dict()

                gn_meta = gn_by_url.get(_norm(url))
                if gn_meta and isinstance(gn_meta, dict):
                    # Primary mappings
                    field_map = {
                        "unique_id": "gn_id",
                        "title": "gn_title",
                        "iso_date": "published_at",
                        "position": "gn_position",
                        "source": "source",
                    }
                    for src_key, dst_key in field_map.items():
                        if src_key in gn_meta and (dst_key not in res_dict or res_dict.get(dst_key) in (None, "")):
                            res_dict[dst_key] = gn_meta.get(src_key)

                    for k, v in gn_meta.items():
                        if k not in field_map:
                            gn_key = f"gn_{k}" if not k.startswith("gn_") else k
                            if gn_key not in res_dict or res_dict.get(gn_key) is None:
                                res_dict[gn_key] = v

                results.append(res_dict)
                if not res.skipped and res.text and res.text.strip():
                    success_count += 1

            if success_count >= target_success_count:
                break

        return results


