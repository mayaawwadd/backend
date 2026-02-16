import logging
from typing import List
from openai import AzureOpenAI
from app.config import settings
import asyncio

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        self.model = settings.azure_openai_embed_model

    def generate_embedding(self, text: str) -> List[float]:
        try:
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid text for embedding: {text}")
                return []

            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding if response.data else []
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def generate_embedding_async(self, text: str) -> List[float]:
        try:
            text = text[:8000]  # Truncate text to 8000 characters
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embeddings.create(input=text, model=self.model)
            )
            return response.data[0].embedding if response.data else []
        except Exception as e:
            logger.error(f"Error generating embedding asynchronously: {str(e)}")
            raise
