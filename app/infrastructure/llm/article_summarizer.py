import json
import logging
import time
from typing import Dict, Any, List, Tuple

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings
from app.infrastructure.llm.prompts import (
    build_system_prompt,
    build_user_prompt,
    BASE_CATEGORIES,
    MAX_SUMMARY_WORDS,
)

logger = logging.getLogger(__name__)


class ArticleSummarizer:
    def __init__(self) -> None:
        self.llm = AzureChatOpenAI(
            azure_endpoint=settings.chat_endpoint,
            api_version=settings.chat_api_version,
            azure_deployment=settings.chat_deployment,
            api_key=settings.chat_key,
            max_tokens=500,
            timeout=60,
        )

    # -----------------------------
    # Public API
    # -----------------------------
    async def summarize(self, article: Dict[str, Any]) -> Dict[str, Any]:
        content = self._extract_content(article)
        if not content:
            raise ValueError("No valid content found for summarization")

        summary, categories = await self._call_llm(content)

        enriched = article.copy()
        enriched.update(
            {
                "summary": summary,
                "categories": categories,
                "ingested_at": int(time.time()),
            }
        )

        return enriched

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _extract_content(self, article: Dict[str, Any]) -> str:
        for key in ("content", "text", "body", "article", "raw"):
            value = article.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        metadata = article.get("metadata", {})
        if isinstance(metadata, dict):
            for key in ("content", "text", "body"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        return ""

    async def _call_llm(self, content: str) -> Tuple[str, List[str]]:
        messages = [
            SystemMessage(content=build_system_prompt()),
            HumanMessage(content=build_user_prompt(content)),
        ]

        try:
            response = self.llm.invoke(messages)
            raw_text = response.content if hasattr(response, "content") else str(response)

            data = self._parse_json_response(raw_text)

            summary = str(data.get("summary", "")).strip()
            categories = [
                c for c in (data.get("categories") or [])
                if c in BASE_CATEGORIES
            ]

            if not categories:
                categories = ["General"]

            summary = self._enforce_word_limit(summary)

            return summary, categories

        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            raise

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end + 1])
            raise ValueError("LLM did not return valid JSON")

    def _enforce_word_limit(self, summary: str) -> str:
        words = summary.split()
        if len(words) > MAX_SUMMARY_WORDS:
            return " ".join(words[:MAX_SUMMARY_WORDS])
        return summary
