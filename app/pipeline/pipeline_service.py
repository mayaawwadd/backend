import logging
from app.pipeline.stages import (
    run_scrape_stage,
    run_vector_stage,
    run_summarization_stage,
    run_image_stage,
    run_cosmos_stage,
)

logger = logging.getLogger(__name__)

class PipelineService:

    async def run(self) -> None:
        logger.info("Pipeline started")

        scraped = await run_scrape_stage()
        await run_vector_stage(scraped)
        summarized = await run_summarization_stage()
        with_images = await run_image_stage(summarized)
        await run_cosmos_stage(with_images)

        logger.info("Pipeline finished")