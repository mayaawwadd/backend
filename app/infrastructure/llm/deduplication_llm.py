import logging
from typing import List, Dict, Any
from app.config import settings

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

class DeduplicationLLM:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.chat_endpoint,
            api_version=settings.chat_api_version,
            azure_deployment=settings.chat_deployment,
            api_key=settings.chat_key,
            max_tokens=2000,
            timeout=60,
        )

    def _build_prompt(self, articles: List[Dict[str, Any]], k: int) -> str:
        items = [
            f"{i+1}. Title: {a.get('title', '')}\n   Source: {a.get('source', '')}\n   Link: {a.get('link', '')}"
            for i, a in enumerate(articles)
        ]
        joined = "\n".join(items)
        return (
            f"You are given a list of {len(articles)} news articles. "
            f"Some may be duplicates (same story from different sources, or very similar titles/content). "
            f"Your job is to select the {k} most important, unique news stories. "
            f"Return a JSON list of exactly {k} article objects, each with 'title', 'link', and 'source'.\n"
            f"Here are the articles:\n{joined}\n"
            f"Respond ONLY with the JSON list."
        )

    async def select_unique_news(self, articles: List[Dict[str, Any]], k: int = 10) -> List[Dict[str, Any]]:
        """
        Use LLM to deduplicate and select the k most valuable news stories from a list of articles.
        Each article should have at least 'title' and 'link'.
        Returns a list of exactly k articles.
        """
        import json
        prompt = self._build_prompt(articles, k)
        logger.info("Sending deduplication prompt to Azure OpenAI...")
        messages = [
            SystemMessage(content="You are a helpful news deduplication assistant."),
            HumanMessage(content=prompt),
        ]
        raw = await self.llm.ainvoke(messages)
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            selected = json.loads(text)
            if not isinstance(selected, list):
                raise ValueError("LLM did not return a list")
            if len(selected) != k:
                raise ValueError(f"LLM did not return exactly {k} articles")
            return selected
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}\nOutput: {text}")
            raise
