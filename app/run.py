import asyncio
import logging
import os
from typing import List, Dict

import httpx
from app.config import settings
from app.services.google_news_fetcher import GoogleNewsFetcher
from app.services.article_scraper import ScraperConfig
from app.services.web_scraper import WebScraper
from app.services.vector_service import QdrantVectorDBClient
from app.services.cosmos_db_client import CosmosDBClient
from app.services.summarization import summarize_articles
import glob
from datetime import datetime, timedelta, timezone
from app.services.document_manager import DocumentManager
from app.services.Image_generator import process_scraper_results
import json

logger = logging.getLogger(__name__)

HEADLESS = False
MAX_ARTICLES = 0
USE_VECTOR_DB = True

node_ca = getattr(settings, "node_extra_ca_certs", None)
ssl_certificate = getattr(settings, 'ssl_certificate', None)
if node_ca:
    os.environ["NODE_EXTRA_CA_CERTS"] = node_ca
    logger.info("Set NODE_EXTRA_CA_CERTS from config.ini: %s", node_ca)
else:
    if ssl_certificate:
        # save the raw cert fingerprint to env for other tooling, but do not use as NODE_EXTRA_CA_CERTS
        os.environ["SSL_CERTIFICATE"] = ssl_certificate
        logger.info("SSL certificate fingerprint found in config.ini; set SSL_CERTIFICATE env var.")
    else:
        logger.warning("No node_extra_ca_certs or SSL certificate found in config.ini; continuing without extra CA certs.")

async def fetch_urls_via_google_news(num: int) -> List[str]:
    fetcher = GoogleNewsFetcher()
    return await fetcher.fetch_articles(num=num)


async def main() -> None:

    cosmos_client = CosmosDBClient()
    logger.info("Connecting to Cosmos DB...")
    await cosmos_client.init()
    logger.info("Connected to Cosmos DB")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    try:
        config = ScraperConfig(
            headless=HEADLESS,
            max_articles=MAX_ARTICLES,
        )

        num_to_fetch = config.max_articles if (config.max_articles and config.max_articles > 0) else 10
        url_fetcher = lambda: asyncio.run(fetch_urls_via_google_news(num=num_to_fetch))
        vdb_client = QdrantVectorDBClient() if USE_VECTOR_DB else None
        files_dir = os.path.join(os.path.dirname(__file__), "files")
        dm = DocumentManager(base_dir=files_dir)
        dm.ensure_dir()
        # Look for scraper results in the dedicated scraper_results subfolder
        scraper_dir = os.path.join(files_dir, "scraper_results")
        os.makedirs(scraper_dir, exist_ok=True)
        pattern = os.path.join(scraper_dir, "scraper_results_*.txt")
        candidates = glob.glob(pattern)
        legacy_path = os.path.join(scraper_dir, "scraper_results.txt")
        if os.path.exists(legacy_path):
            candidates.append(legacy_path)

        # Always fetch fresh results
        logger.info("Starting web scraping...")
        scraper = WebScraper(config=config, url_fetcher=url_fetcher)
        try:
            results = await scraper.scrape_all()
            logger.info("Scraped %d articles", len(results) if results else 0)
            # --- Store all scraped articles to output/scraped_articles.json ---
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
            os.makedirs(output_dir, exist_ok=True)
            scraped_json_path = os.path.join(output_dir, "scraped_articles.json")
            with open(scraped_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info("Saved all scraped articles to %s", scraped_json_path)
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error when fetching articles: %s", e)
            return
        except Exception as e:
            logger.exception("Unexpected error during scraping: %s", e)
            return

        try:
            dm = DocumentManager(base_dir=os.path.join(os.path.dirname(__file__), "files"))
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            filename = f"scraper_results_{timestamp}.txt"
            # save_results will place scraper_results files into the scraper_results subfolder
            out_path = dm.save_results(filename, results, overwrite=True)
            logger.info("Saved scraped results to %s", out_path)
            try:
                logger.info("Generating images for scraped results at %s", out_path)
                processed = process_scraper_results(out_path)
                logger.info("Image generation processed %d items and updated file: %s", processed, out_path)
            except Exception:
                logger.exception("Failed to generate images for scraped results")
        except Exception as e:
            logger.exception("Failed to save results to file: %s", e)
        
        if USE_VECTOR_DB and vdb_client:
            logger.info("Ingesting articles to Qdrant...")
            try:
                await vdb_client.test_connection()
                await vdb_client.ingest_scraped_results(results)
                logger.info("Successfully ingested articles to Qdrant")
            except Exception:
                logger.exception("[QdrantVDB] Vector operations failed")
                return
        
        # Load articles from Qdrant collection instead of articles.json
        logger.info("Loading articles from Qdrant collection...")
        try:
            from qdrant_client import QdrantClient
            qdrant_client = QdrantClient(url=settings.qdrant_url or "http://localhost:6333")
            collection_name = settings.qdrant_collection
            
            # Scroll through all points in the collection
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
                    if point.payload:
                        all_articles.append(point.payload)
                
                if next_page is None:
                    break
            
            logger.info("Loaded %d articles from Qdrant", len(all_articles))
        except Exception as e:
            logger.exception("Failed to load articles from Qdrant: %s", e)
            all_articles = []
        
        # Generate summaries from Qdrant articles
        logger.info("Starting summarization of %d articles...", len(all_articles))
        try:
            llm_results = summarize_articles(all_articles)
            logger.info("Successfully generated summaries for %d articles", len(llm_results))
            # --- Store all summarized articles to output/summarized_articles.json ---
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
            os.makedirs(output_dir, exist_ok=True)
            summarized_json_path = os.path.join(output_dir, "summarized_articles.json")
            with open(summarized_json_path, "w", encoding="utf-8") as f:
                json.dump(llm_results, f, ensure_ascii=False, indent=2)
            logger.info("Saved all summarized articles to %s", summarized_json_path)
        except Exception as e:
            logger.exception("Failed to generate summaries: %s", e)
            llm_results = []
        
        # Clear existing data from Cosmos DB
        logger.info("Clearing existing articles from Cosmos DB...")
        try:
            deleted_count = await cosmos_client.delete_all_news()
            logger.info("Deleted %d articles from Cosmos DB", deleted_count)
        except Exception as e:
            logger.error(f"Failed to delete articles from Cosmos DB: {e}")
        
        # Upsert summarized articles to Cosmos DB
        upserted_count = 0
        skipped_count = 0
        for article in llm_results:
            try:
                # Articles are already enriched from summarize_articles
                if article and isinstance(article, dict):
                    # Ensure required fields exist
                    if all(k in article for k in ['id', 'url', 'title', 'summary', 'categories']):
                        await cosmos_client.upsert_news(article)
                        upserted_count += 1
                    else:
                        logger.debug("Skipping article missing required fields: %s", article.get('id', 'unknown'))
                        skipped_count += 1
            except Exception as e:
                logger.error(f"Failed to upsert article to Cosmos DB: {e}")
                skipped_count += 1
        
        logger.info("Upserted %d articles to Cosmos DB (%d skipped)", upserted_count, skipped_count)
        
        # Fetch and display final results from Cosmos DB
        logger.info("Fetching final results from Cosmos DB...")
        try:
            final_articles = await cosmos_client.get_all_news()
            print("\n" + "="*120)
            print("FINAL COSMOS DB RESULTS (FULL PAYLOAD)".center(120))
            print("="*120 + "\n")
            
            if final_articles:
                for idx, article in enumerate(final_articles, 1):
                    print(f"[{idx}]")
                    # Pretty-print the entire document including id, url, and all other fields
                    print(json.dumps(article, ensure_ascii=False, indent=2, sort_keys=True, default=str))
                    print("-" * 120)

                print("="*120)
                print(f"Total articles in Cosmos DB: {len(final_articles)}".center(120))
                print("="*120)
            else:
                print("No articles found in Cosmos DB")
        except Exception as e:
            logger.error(f"Failed to fetch final results from Cosmos DB: {e}")

    finally:
        await cosmos_client.close()

if __name__ == "__main__":
    asyncio.run(main())