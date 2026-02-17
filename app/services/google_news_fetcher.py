import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from uuid import NAMESPACE_URL, uuid5

import httpx
from dotenv import load_dotenv

from app.infrastructure.qdrant_client import QdrantCloudClient

# Ensure dotenv is loaded globally
# Already present in this file, no changes needed.

load_dotenv()

logger = logging.getLogger(__name__)


SEARCH_TERMS = (
    'intitle:"artificial intelligence" OR '
    'intitle:"AI" OR '
    'intitle:"machine learning" OR '
    'intitle:"large language model" OR '
    'intitle:"AI policy" OR '
    'intitle:"AI regulation" OR '
    '"Artificial Intelligence" OR '
    '"AI"'
)

CREDIBLE_SITES = (
    "site:apnews.com OR site:bbc.coms OR site:washingtonpost.com OR "
    "site:theguardian.com OR site:npr.org OR site:financialtimes.com OR site:bloomberg.com OR site:wsj.com OR "
    "site:cnbc.com OR site:forbes.com OR "
    "site:wired.com OR site:theverge.com OR site:techcrunch.com OR site:venturebeat.com OR "
    "site:technologyreview.com OR site:nature.com OR site:sciencedaily.com OR site:scientificamerican.com OR "
    "site:spectrum.ieee.org OR "
    "site:businessinsider.com OR site:fortune.com"
)


class GoogleNewsFetcher:
    BASE_URL = "https://www.searchapi.io/api/v1/search"

    def __init__(self):
        self.api_key = QdrantCloudClient.get_searchapi_key()
        self.url = self.BASE_URL  # Define the base URL for the search API

    def extract_records(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        records = []
        for item in data.get("organic_results", []):
            link = item.get("link")
            title = item.get("title")
            iso_date = item.get("iso_date")
            position = item.get("position")
            source = item.get("source")
            unique_id = str(uuid5(NAMESPACE_URL, link))
            records.append({
                "unique_id": unique_id,
                "title": title,
                "iso_date": iso_date,
                "link": link,
                "position": position,
                "source": source,
            })
        return records

    def pretty_print_records(self, records: List[Dict[str, Any]]) -> None:
        """Print records in a readable, formatted way."""
        if not records:
            print("No records to display.")
            return
        
        print("\n" + "="*100)
        print(f"{'Total Records: ' + str(len(records)):^100}")
        print("="*100 + "\n")
        
        for i, record in enumerate(records, 1):
            print(f"Record {i}:")
            print(f"  Position:   {record.get('position', 'N/A')}")
            print(f"  Title:      {record.get('title', 'N/A')}")
            print(f"  Source:     {record.get('source', 'N/A')}")
            print(f"  Date:       {record.get('iso_date', 'N/A')}")
            print(f"  URL:        {record.get('link', 'N/A')}")
            print(f"  ID:         {record.get('unique_id', 'N/A')}")
            print("-" * 100)



    async def fetch_articles(self, num: int = 10, exclude_urls: Optional[Set[str]] = None) -> List[Dict[str, Any]]:

        async def run_query(params):
            headers = {
                "Authorization": f"Bearer {self.api_key.strip()}"
            }
            max_retries = 5
            backoff_base = 1
            for attempt in range(1, max_retries + 1):
                try:
                    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                        response = await client.get(self.url, params=params, headers=headers)
                        response.raise_for_status()
                        raw_json = response.json()
                        return self.extract_records(raw_json)
                except httpx.HTTPStatusError as e:
                    status = e.response.status_code if e.response is not None else None
                    if status == 429 and attempt < max_retries:
                        retry_after = e.response.headers.get("Retry-After") if e.response is not None else None
                        try:
                            wait = int(retry_after) if retry_after else backoff_base * (2 ** (attempt - 1))
                        except Exception:
                            wait = backoff_base * (2 ** (attempt - 1))
                        logger.warning("Received 429 from search API; backing off %s seconds (attempt %s/%s)", wait, attempt, max_retries)
                        await asyncio.sleep(wait)
                        continue
                    raise
                except (httpx.RequestError, asyncio.TimeoutError) as e:
                    if attempt < max_retries:
                        wait = backoff_base * (2 ** (attempt - 1))
                        logger.warning("Network error when querying search API; retrying in %s seconds (attempt %s/%s): %s", wait, attempt, max_retries, e)
                        await asyncio.sleep(wait)
                        continue
                    raise

        base_params = {
            "engine": "google_news",
            "q": f"({SEARCH_TERMS}) AND ({CREDIBLE_SITES})",
            "num": num,
            "lang": "en",
            "hl": "en",
            "filter": 1,
            "gl": "us",
            "location": "United States",
            "api_key": self.api_key
        }

        time_periods = ["last_day", "last_week"]
        loosen_queries = [
            f"{SEARCH_TERMS}",
            "AI OR Artificial Intelligence",
        ]

        # Track unique URLs to avoid duplicates across different queries
        seen_urls = set()
        unique_records = []
        exclude_urls = exclude_urls or set()

        for period in time_periods:
            params = dict(base_params)
            params["time_period"] = period
            records = await run_query(params)
            if records is None:
                continue
            for record in records:
                link = record.get("link")
                if not link:
                    continue
                if link in exclude_urls:
                    continue
                if link and link not in seen_urls:
                    seen_urls.add(link)
                    unique_records.append(record)
                    if len(unique_records) >= num:
                        return unique_records[:num]

        for q in loosen_queries:
            params = dict(base_params)
            params["time_period"] = "last_week"
            params["q"] = q
            records = await run_query(params)
            if records is None:
                continue
            for record in records:
                link = record.get("link")
                if not link:
                    continue
                if link in exclude_urls:
                    continue
                if link and link not in seen_urls:
                    seen_urls.add(link)
                    unique_records.append(record)
                    if len(unique_records) >= num:
                        return unique_records[:num]

        return unique_records[:num]


async def main():
    fetcher = GoogleNewsFetcher()
    records = await fetcher.fetch_articles(num=10)
    fetcher.pretty_print_records(records)
    return records


if __name__ == "__main__":
    asyncio.run(main())