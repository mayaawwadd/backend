import asyncio
import logging

from app.pipeline.pipeline_service import PipelineService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

if __name__ == "__main__":
    asyncio.run(PipelineService().run())
