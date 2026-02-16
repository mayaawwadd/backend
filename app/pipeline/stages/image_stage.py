import logging
import asyncio
from typing import List, Dict, Any

from app.services.image_generator import (
    build_safe_editorial_prompt,
    call_image_api,
    extract_image_url,
    save_image_from_url,
)

logger = logging.getLogger(__name__)


async def run_image_stage(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logger.info("Starting image generation stage")

    if not articles:
        raise RuntimeError("Image stage aborted: no summarized results available.")

    processed = 0

    for article in articles:
        try:
            title = (article.get("title") or "").strip()

            if not title:
                article["image_url"] = ""
                article["local_image_path"] = ""
                continue

            curated_prompt = build_safe_editorial_prompt(article)

            raw = await asyncio.to_thread(call_image_api, curated_prompt)
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

            await asyncio.sleep(20)  # 3 per minute rate limit

        except Exception as e:
            logger.warning(
                "Image generation failed for article %s: %s",
                article.get("id"),
                e
            )
            article["image_url"] = ""
            article["local_image_path"] = ""

    logger.info("Image generation completed for %s articles", processed)

    return articles
