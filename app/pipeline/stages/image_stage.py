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


async def run_image_stage(
    articles: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    logger.info("Starting image generation stage")

    if not articles:
        raise RuntimeError(
            "Image stage aborted: no summarized results available."
        )

    processed = 0
    enriched_articles: List[Dict[str, Any]] = []

    for article in articles:
        try:
            updated = article.copy()
            title = (updated.get("title") or "").strip()

            if not title:
                updated["image_url"] = ""
                updated["local_image_path"] = ""
                enriched_articles.append(updated)
                continue

            curated_prompt = build_safe_editorial_prompt(updated)

            raw = await asyncio.to_thread(call_image_api, curated_prompt)
            image_url, is_default = extract_image_url(raw)

            if image_url:
                saved = await asyncio.to_thread(
                    save_image_from_url,
                    image_url,
                    title
                )

                updated["image_url"] = "" if is_default else image_url
                updated["local_image_path"] = saved or ""
            else:
                updated["image_url"] = ""
                updated["local_image_path"] = ""

            processed += 1

            # 3 requests per minute
            await asyncio.sleep(20)

        except Exception as e:
            logger.warning(
                "Image generation failed for article %s: %s",
                article.get("id"),
                e,
            )
            updated = article.copy()
            updated["image_url"] = ""
            updated["local_image_path"] = ""

        enriched_articles.append(updated)

    logger.info(
        "Image generation completed for %s articles",
        processed,
    )

    return enriched_articles