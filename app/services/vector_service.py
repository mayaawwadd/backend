import logging
import uuid
from typing import Any, Dict, List, Optional

from app.services.embedding_service import EmbeddingService
from app.services.document_mapper import document_to_point
from app.infrastructure.qdrant_client import QdrantCloudClient, initialize_embedding_client

logger = logging.getLogger(__name__)


class VectorDBClient:
    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        raise NotImplementedError


class QdrantVectorDBClient(VectorDBClient):
    async def ingest_scraped_results(self, results: List[Dict[str, Any]]):
        import hashlib
        from urllib.parse import urlparse
        from app.utils.scraper_utils import URLTools

        vdb_documents: List[Dict[str, Any]] = []
        for res in results:
            res_url = res.get("url") or res.get("link")
            text = (res.get("text") or "")[:100000]
            
            # Skip only if article is skipped AND lacks valuable Google News metadata
            has_gn_metadata = res.get("gn_id") or res.get("gn_title") or res.get("published_at")
            if res.get("skipped") and not has_gn_metadata:
                logger.debug(f"[QdrantVDB] Skipping {res_url}: {res.get('skipped_reason')}")
                continue
            
            # Must have URL; text not strictly required if we have good metadata
            if not res_url:
                continue
            
            # Skip if both text is empty AND no Google News metadata
            if not text and not has_gn_metadata:
                continue

            doc_id = URLTools.safe_name_from_url(res_url)
            checksum = hashlib.sha1((res_url + text).encode("utf-8")).hexdigest() if text else None
            source_from_gn = res.get("source")
            fallback_source = urlparse(res_url).netloc if res_url else None

            # Preserve ALL fields explicitly to prevent loss through the pipeline
            meta: Dict[str, Any] = {
                # Direct fields
                "url": res_url,
                "text_preview": text[:300] if text else "",
                "checksum": checksum,
                
                # Scraper-derived fields
                "title": res.get("title"),
                "selector": res.get("selector"),
                "method": res.get("method"),
                "skipped": res.get("skipped", False),
                "skipped_reason": res.get("skipped_reason"),
                
                # Google News enrichment fields (highest priority)
                "gn_id": res.get("gn_id"),
                "gn_title": res.get("gn_title"),
                "gn_position": res.get("gn_position"),
                "source": source_from_gn or fallback_source,
                "published_at": res.get("published_at") or res.get("iso_date"),
            }

            # Preserve any other fields not explicitly handled
            for k, v in res.items():
                if k not in meta and k not in ("text", "url"):
                    meta[k] = v
            
            # Only add to documents if we have text OR valuable metadata
            if text or has_gn_metadata:
                vdb_documents.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "iso_date": res.get("iso_date"),
                        "published_at": res.get("published_at") or res.get("iso_date"),
                        "metadata": meta,
                    }
                )

        if vdb_documents:
            logger.info(f"[QdrantVDB] Processing {len(vdb_documents)} documents with preserved fields")
            await self.perform_operations(vdb_documents)

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
        use_azure: bool = False,
    ) -> None:
        from app.config import settings

        self.qdrant_url = qdrant_url or settings.qdrant_url or "http://127.0.0.1:6333"
        self.qdrant_api_key = qdrant_api_key or settings.qdrant_api_key or None
        self.collection_name = collection_name or settings.qdrant_collection or "articles"
        self.vector_name = getattr(settings, "qdrant_vector_name", None) or "article_vectors"
        self.dim = getattr(settings, "qdrant_dim", None) or 1536
        self.embedding_model = embedding_model or getattr(settings, "embedding_model", None) or "text-embedding-3-small"
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else float(getattr(settings, "qdrant_threshold", 0.8) or 0.8)
        )

        self.qdrant_client = QdrantCloudClient(
            collection=self.collection_name,
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            vector_name=self.vector_name,
            dim=self.dim,
        )
        self.openai_client, self.azure_embed_model = initialize_embedding_client(use_azure, self.embedding_model)

    async def _generate_embedding(self, text: str) -> List[float]:
        embedding_service = EmbeddingService()
        return await embedding_service.generate_embedding_async(text)

    async def test_connection(self):
        try:
            stats = await self.fetch_collection_stats()
            logger.info(f"[QdrantVDB] Connection successful. Collection stats: {stats}")
        except Exception as e:
            logger.error(f"[QdrantVDB] Connection test failed: {e}")

    async def process_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            logger.info("[QdrantVDB] No documents to process")
            return

        logger.info(f"[QdrantVDB] Processing {len(documents)} documents")

        points_to_upsert = []
        for doc in documents:
            try:
                doc_id = doc.get("id", str(uuid.uuid4()))
                text = doc.get("text", "")
                if not text:
                    logger.warning(f"[QdrantVDB] Skipping document {doc_id}: empty text")
                    continue

                embedding = await self._generate_embedding(text)
                points_to_upsert.append(document_to_point(doc, embedding, doc_id))
            except Exception as e:
                logger.error(f"[QdrantVDB] Error processing document {doc.get('id')}: {e}")

        if points_to_upsert:
            try:
                self.qdrant_client.ensure_collection_and_vector()
                self.qdrant_client.upsert_points(points_to_upsert)
                logger.info(f"[QdrantVDB] Upsert complete: {len(points_to_upsert)} points stored")
            except Exception as e:
                logger.error(f"[QdrantVDB] Batch upsert failed: {e}")
                raise

    async def query_documents(self, query_text: str, limit: int = 5, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        try:
            embedding = await self._generate_embedding(query_text)
            results = self.qdrant_client.search_points(embedding, limit, score_threshold or self.similarity_threshold)
            return [
                {
                    "id": hit["id"],
                    "score": hit["score"],
                    "text_preview": (hit["payload"].get("text") or "")[:200],
                    "metadata": hit["payload"].get("metadata", {}),
                }
                for hit in results
            ]
        except Exception as e:
            logger.error(f"[QdrantVDB] Query failed: {e}")
            return []

    async def fetch_collection_stats(self) -> Dict[str, Any]:
        try:
            return self.qdrant_client.get_collection_stats()
        except Exception as e:
            logger.error(f"[QdrantVDB] Failed to fetch collection stats: {e}")
            return {}

    async def perform_operations(self, documents: List[Dict[str, Any]]):
        try:
            try:
                self.qdrant_client.ensure_collection_and_vector()
                if hasattr(self.qdrant_client, "clear_points"):
                    self.qdrant_client.clear_points()
                    logger.info("[QdrantVDB] Cleared existing points in collection '%s'", self.collection_name)
                else:
                    if hasattr(self.qdrant_client, "drop_collection"):
                        self.qdrant_client.drop_collection()
                        logger.info("[QdrantVDB] Dropped collection '%s' to reset", self.collection_name)
                    self.qdrant_client.ensure_collection_and_vector()
            except Exception as e:
                logger.error(f"[QdrantVDB] Failed to clear/reset collection: {e}")
                raise

            await self.process_documents(documents)
            logger.info("[QdrantVDB] Vector operations completed successfully.")
        except Exception as e:
            logger.error(f"[QdrantVDB] Vector operations failed: {e}")

    async def fetch_all_articles_payload(self) -> list:
        """
        Fetch all stored article payloads from Qdrant.
        """
        all_articles = []
        next_page = None

        while True:
            points, next_page = self.qdrant_client.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=100,
                with_payload=True,
                with_vectors=False,
                offset=next_page,
            )

            if not points:
                break

            for point in points:
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                flat = {**payload, **metadata}
                all_articles.append(flat)

            if next_page is None:
                break

        return all_articles
