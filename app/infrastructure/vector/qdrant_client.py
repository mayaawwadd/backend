"""
QdrantCloudClient: Utility class for interacting with Qdrant (local or cloud)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional, cast

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings
import os
import uuid
from typing import Union

logger = logging.getLogger(__name__)


class QdrantCloudClient:
    """
    Provides methods for interacting with Qdrant, including upserting vectors,
    searching, and managing collections.
    """

    # ---------- Construction / wiring ----------

    @staticmethod
    def initialize_qdrant_client(qdrant_url: str, qdrant_api_key: Optional[str]) -> QdrantClient:
        # prefer_grpc=False keeps things simple on Windows/corp networks; REST only
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30, prefer_grpc=False)

    @classmethod
    def from_config(cls) -> "QdrantCloudClient":
        """
        Factory method to create QdrantCloudClient using config.ini only.
        """
        qdrant_url = settings.qdrant_url
        qdrant_api_key = settings.qdrant_api_key
        collection = settings.qdrant_collection
        dim = settings.qdrant_dim
        vector_name = settings.qdrant_vector_name

        if not qdrant_url:
            raise RuntimeError("QDRANT_URL must be set in config.ini.")

        return cls(
            collection=collection,
            dim=dim,
            vector_name=vector_name,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )

    @classmethod
    def get_searchapi_key(cls) -> str:
        """Retrieve the searchapi.io API key from config.ini."""
        # Prefer environment variable (e.g., .env or runtime env) so users can override easily.
        # Prefer explicit SEARCH_API_KEY4 if present, then fall back to other env vars.
        env_key = (
            os.environ.get("SEARCH_API_KEY4")
            or os.environ.get("SEARCH_API_KEY")
            or os.environ.get("SEARCH_API_KEY2")
            or os.environ.get("SEARCH_API_KEY3")
        )
        if env_key:
            return env_key.strip()

        try:
            api_key = settings.searchapi_key
        except Exception:
            api_key = ""
        if not api_key:
            api_key = ""
            logger.warning("No search API key found in env or config.ini. Set SEARCH_API_KEY or update config.ini.")
        return api_key

    def __init__(
        self,
        collection: Optional[str] = None,
        dim: Optional[int] = None,
        vector_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ) -> None:
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.qdrant_api_key = qdrant_api_key or settings.qdrant_api_key
        self.collection = collection or settings.qdrant_collection or "articles"
        self.dim = int(dim or settings.qdrant_dim or 1536)
        self.vector_name = vector_name or settings.qdrant_vector_name or "article_vectors"

        if not self.qdrant_url:
            raise RuntimeError("QDRANT_URL must be set.")

        self.client: QdrantClient = QdrantCloudClient.initialize_qdrant_client(self.qdrant_url, self.qdrant_api_key)

    # ---------- Collection management ----------

    def create_collection(self, distance: str = "cosine") -> None:
        """Create collection with a **named vector** schema."""
        dist = getattr(qm.Distance, distance.upper(), qm.Distance.COSINE)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={self.vector_name: qm.VectorParams(size=self.dim, distance=dist)},
        )
        logger.info(
            "Created collection '%s' with named vector '%s' (dim=%d, distance=%s)",
            self.collection, self.vector_name, self.dim, dist.name
        )

    def drop_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
            logger.info("Deleted collection '%s'", self.collection)

    def ensure_collection_and_vector(self, distance: str = "cosine") -> None:
        """
        Ensure the collection exists and has the expected named-vector schema.
        (This creates it if missing. For schema drift, extend with validation if needed.)
        """
        if not self.client.collection_exists(self.collection):
            self.create_collection(distance=distance)
        else:
            logger.info(
                "Collection '%s' already exists. (expected vector='%s', dim=%d)",
                self.collection, self.vector_name, self.dim
            )

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return a simplified stats dict with points count and vectors config info.
        Includes None-guards to satisfy strict type checkers like Pylance.
        """
        info = self.client.get_collection(self.collection)
        vectors_field = info.config.params.vectors  # may be dict[str, VectorParams] or VectorParams

        vectors_map: Dict[str, qm.VectorParams] = {}
        if isinstance(vectors_field, dict):
            # Only keep well-typed VectorParams values
            for k, v in vectors_field.items():
                if isinstance(v, qm.VectorParams):
                    vectors_map[k] = v
        elif isinstance(vectors_field, qm.VectorParams):
            vectors_map[self.vector_name] = vectors_field

        vec = vectors_map.get(self.vector_name)
        distance_name: Optional[str] = None
        if isinstance(vec, qm.VectorParams) and getattr(vec, "distance", None) is not None:
            distance_name = vec.distance.name  # type: ignore[assignment]

        # Build a sanitized dict for all vectors
        norm_vectors = {
            k: {
                "size": int(v.size),
                "distance": (v.distance.name if getattr(v, "distance", None) is not None else None),
            }
            for k, v in vectors_map.items()
        }

        return {
            "points_count": getattr(info, "points_count", None),
            "vectors": norm_vectors,
            "distance": distance_name,
        }

    # ---------- Data operations ----------


    def upsert_point(self, point_id: Any, vector: List[float], payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Upsert a single point using the **named vector** schema.
        NOTE: point_id must be an unsigned int or a UUID string.
        """
        valid_id = self._ensure_valid_point_id(point_id)
        point = qm.PointStruct(id=valid_id, vector={self.vector_name: vector}, payload=payload or {})
        self.client.upsert(collection_name=self.collection, points=[point])


    def upsert_points(self, points: List[qm.PointStruct]) -> None:
        """
        Upsert multiple points (each must already carry a named vector in its 'vector' field).
        """
        # Ensure all point IDs are valid
        for p in points:
            p.id = self._ensure_valid_point_id(p.id)
        self.client.upsert(collection_name=self.collection, points=points)

    def _ensure_valid_point_id(self, point_id: Any) -> Union[int, str]:
        """
        Ensures the point_id is a valid Qdrant point ID (int or UUID string).
        If a string is provided that is not a valid UUID, it will be converted to a UUID5.
        """
        if isinstance(point_id, int):
            return point_id
        try:
            # If already a valid UUID string, return as is
            uuid_obj = uuid.UUID(str(point_id))
            return str(uuid_obj)
        except Exception:
            # Convert any string to a UUID5 (namespace-based)
            return str(uuid.uuid5(uuid.NAMESPACE_URL, str(point_id)))

    def search_points(
        self,
        query_vector: List[float],
        limit: int,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity using the **Query API** (named vectors):
        - query: dense vector (list[float])
        - using: name of the vector space (self.vector_name)
        """
        res = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,         # dense vector query
            using=self.vector_name,     # pick the named vector space
            limit=limit,
            with_payload=True,
            score_threshold=score_threshold,
        )

        # Normalize return type and satisfy Pylance with an explicit cast.
        points = getattr(res, "points", res)  # res.points is List[ScoredPoint] in 1.16.x
        scored = cast(List[qm.ScoredPoint], points)

        out: List[Dict[str, Any]] = []
        for h in scored:
            # getattr() fallbacks keep Pylance happy even if stubs are slightly off
            out.append({
                "id": getattr(h, "id", None),
                "score": float(getattr(h, "score", 0.0)),
                "payload": getattr(h, "payload", None),
            })
        return out


def initialize_embedding_client(use_azure: bool, embedding_model: str):
    """
    Initialize the embedding client based on the configuration.
    Keep as a simple stub if you swap between Azure/OpenAI elsewhere.
    """
    if use_azure:
        azure_embed_model = f"azure-{embedding_model}"
        return None, azure_embed_model
    else:
        openai_client = "openai-client-instance"
        return openai_client, None