"""Document payload and filter mapping helpers."""

from typing import Dict, Any, Optional, TypedDict, List, Sequence
from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct, Condition

METADATA_KEY_PREFIX = "metadata."
TOP_LEVEL_FILTER_KEYS = {"document_id", "title", "source"}

class DocumentDict(TypedDict, total=False):
    content: str
    metadata: Dict[str, Any]
    document_id: str
    title: str
    source: str

def extract_content(document: DocumentDict) -> str:
    return document.get("content", "")

def build_payload(document: Dict[str, Any], content: str, point_id: str) -> Dict[str, Any]:
    metadata = document.get("metadata", {}).copy()
    # Add date_published from iso_date if present (enforce only iso_date)
    date_val = document.get("iso_date") or metadata.get("iso_date")
    if date_val:
        metadata["date_published"] = date_val

    document_id = document.get("document_id")
    payload = {
        "content": content,
        "metadata": metadata,
        "document_id": str(document_id) if document_id else point_id
    }
    # Do NOT duplicate metadata fields at the top level or as metadata.<field> keys
    return payload

def build_search_filter(filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
    if not filters:
        return None

    conditions: List[Condition] = [
        FieldCondition(
            key=f"{METADATA_KEY_PREFIX}{key}" if key not in TOP_LEVEL_FILTER_KEYS and not key.startswith(METADATA_KEY_PREFIX) else key,
            match=MatchValue(value=value)
        )
        for key, value in filters.items()
    ]
    return Filter(must=conditions) if conditions else None

def format_search_result(result) -> Dict[str, Any]:
    return {
        "id": result.id,
        "score": result.score,
        "content": result.payload.get("content", ""),
        "metadata": result.payload.get("metadata", {}),
        "document_id": result.payload.get("document_id", ""),
        "title": result.payload.get("title", ""),
        "source": result.payload.get("source", "")
    }

def format_document(point) -> Dict[str, Any]:
    payload = point.payload or {}
    return {
        "id": point.id,
        "content": payload.get("content", ""),
        "metadata": payload.get("metadata", {}),
        "document_id": payload.get("document_id", ""),
        "title": payload.get("title", ""),
        "source": payload.get("source", "")
    }

def document_to_point(document: Dict[str, Any], embedding: List[float], point_id: str) -> PointStruct:
    from app.config import settings
    from qdrant_client.http import models as qm
    payload = build_payload(document, document.get("text", ""), point_id)
    vector_name = settings.qdrant_vector_name or "article_vectors"
    return qm.PointStruct(id=point_id, vector={vector_name: embedding}, payload=payload)
