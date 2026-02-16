import os
import json
import hashlib
import time
from typing import Dict, Any, List, Optional, Tuple

from qdrant_client import QdrantClient

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import settings

# ----------------------------
# Config / constants
# ----------------------------
QDRANT_URL = settings.qdrant_url or "http://localhost:6333"
QDRANT_COLLECTION = settings.qdrant_collection

AZURE_OPENAI_ENDPOINT = settings.chat_endpoint
AZURE_OPENAI_API_KEY = settings.chat_key
AZURE_OPENAI_API_VERSION = settings.chat_api_version
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = settings.chat_deployment

# Local JSON storage
JSON_STORAGE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'summaries')
os.makedirs(JSON_STORAGE_DIR, exist_ok=True)
JSON_STORAGE_FILE = os.path.join(JSON_STORAGE_DIR, 'summaries.json')

# Summarization constraints
MAX_SUMMARY_WORDS = 20
CONTENT_CHAR_LIMIT = 12000  # avoid long prompts; tune as needed

# Base categories (your list)
BASE_CATEGORIES = {
    "General",
    "ML",
    "GenAI",
    "Leading Companies",
    "Research",
    "Responsible AI",
    "Products & Features",
    "Policy & Regulation",
    "Use Cases",
    "Tools & Frameworks",
    "Startups & Funding",
    "Security & Privacy",
    }

SYSTEM_PROMPT = f"""You are an expert editor for an internal, organization-wide AI newsletter. 
Your goals:
1) Write a concise, engaging, and informative summary suitable for busy professionals.
2) Assign one or more categories from the allowed list.
3) When helpful, also suggest categories from an extended taxonomy.

Rules:
- Summary: 1 sentence, max {MAX_SUMMARY_WORDS} words, neutral tone, high-signal, no fluff.
- Audience: enterprise, cross-functional (execs, product, eng, data, GTM).
- If content is too short or unclear, do your best but stay faithful to the source.
- Output ONLY valid JSON with keys: "summary", "categories".
- "categories" MUST be from the BASE list.

BASE CATEGORIES (use one or more):
- {', '.join(sorted(BASE_CATEGORIES))}

"""

HUMAN_PROMPT_TEMPLATE = """Content:
\"\"\"
{content}
\"\"\"

Return valid JSON ONLY:
{{
  "summary": "...",
  "categories": ["..."]               // Subset of BASE only
}}
"""

# ----------------------------
# Azure OpenAI (LangChain) setup
# ----------------------------
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    api_key=AZURE_OPENAI_API_KEY,
    max_tokens=2000,  
    timeout=60,
)

# Local JSON storage helper functions
def load_summaries() -> Dict[str, Any]:
    """Load all summaries from local JSON storage."""
    if os.path.exists(JSON_STORAGE_FILE):
        try:
            with open(JSON_STORAGE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_summaries(data: Dict[str, Any]) -> None:
    """Save summaries to local JSON storage."""
    with open(JSON_STORAGE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def normalize_content(payload: Dict[str, Any]) -> Optional[str]:
    """
    Returns the text content to summarize.
    Looks for common keys. Edit as needed for your schema.
    """
    if not payload:
        return None
    for key in ("content", "text", "body", "article", "raw"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # some stores keep content under a nested 'document' or 'metadata'
    meta = payload.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("content", "text", "body"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def extract_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract required metadata fields. Supports top-level or payload['metadata'] placement.
    """
    fields = ["url", "title", "gn_id", "source", "published_at"]
    result = {k: None for k in fields}

    def get_from(d: Dict[str, Any], key: str):
        if isinstance(d, dict):
            return d.get(key)
        return None

    meta = payload.get("metadata") if isinstance(payload, dict) else None

    for f in fields:
        result[f] = get_from(payload, f) or (get_from(meta, f) if isinstance(meta, dict) else None)
        if isinstance(result[f], str):
            result[f] = result[f].strip() or None

    return result


def build_doc_id(url: Optional[str], gn_id: Optional[str], point_id: Optional[str]) -> str:
    """
    Make a stable unique id for documents. Prefer gn_id, else hash of URL, else hash of point_id.
    """
    if gn_id:
        return str(gn_id)
    if url:
        return hashlib.sha256(url.encode("utf-8")).hexdigest()
    if point_id:
        return str(point_id)
    return hashlib.sha256(os.urandom(16)).hexdigest()


def call_llm(content: str) -> Tuple[str, List[str], List[str]]:
    """
    Call Azure OpenAI via LangChain to get summary and categories (base + suggested).
    Returns (summary, categories).
    """
    # Trim content (avoid huge prompts)
    if len(content) > CONTENT_CHAR_LIMIT:
        content = content[:CONTENT_CHAR_LIMIT] + "\n\n[Truncated for summarization]"

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=HUMAN_PROMPT_TEMPLATE.format(content=content)),
    ]

    raw = llm.invoke(messages)
    text = raw.content if hasattr(raw, "content") else str(raw)

    # Parse strict JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage JSON if model adds extra text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            raise ValueError(f"LLM did not return valid JSON: {text}")

    summary = str(data.get("summary", "")).strip()
    categories = [c for c in (data.get("categories") or []) if c in BASE_CATEGORIES]
    # If the model missed base categories, try a fallback heuristic
    if not categories:
        # extremely light heuristic fallback (optional)
        lc = content.lower()
        if any(k in lc for k in ("fine-tune", "training", "supervised", "unsupervised", "regression", "classification")):
            categories.append("ML")
        if any(k in lc for k in ("llm", "prompt", "diffusion", "genai", "gpt", "image generation", "chatbot")):
            if "GenAI" not in categories:
                categories.append("GenAI")
        if not categories:
            categories.append("General")

    # Hard constraints on summary length (approx words)
    if summary:
        words = summary.split()
        if len(words) > MAX_SUMMARY_WORDS:
            summary = " ".join(words[:MAX_SUMMARY_WORDS])

    return summary, categories


def upsert_summary(doc: Dict[str, Any]) -> None:
    """
    Upsert the summary document into local JSON storage.
    """
    try:
        all_summaries = load_summaries()
        doc_id = doc.get('id')
        all_summaries[doc_id] = doc
        save_summaries(all_summaries)
    except Exception as e:
        raise RuntimeError(f"Failed to save summary: {e}")


def summarize_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Takes a list of articles and generates summaries for each one.
    Returns a list of enriched articles with summary and categories.
    
    Args:
        articles: List of article dictionaries with 'url', 'title', 'text', etc.
    
    Returns:
        List of articles with added 'summary' and 'categories' fields, plus
        Cosmos DB required fields: 'id', 'date', 'url', 'title'.
    """
    summarized_articles = []
    processed = 0
    failures = 0
    
    for article in articles:
        try:
            # Skip articles that should be skipped or have no content
            if article.get('skipped', False):
                continue
                
            content = normalize_content(article)
            if not content:
                # Skip if no content to summarize
                continue

            meta = extract_metadata(article)
            summary, categories = call_llm(content)

            # Build enriched document
            doc_id = build_doc_id(meta.get("url"), meta.get("gn_id"), article.get("qdrant_point_id"))
            enriched_article = article.copy()
            enriched_article.update({
                "id": doc_id,
                "url": meta.get("url") or enriched_article.get("url"),
                "title": meta.get("title") or enriched_article.get("title"),
                "gn_id": meta.get("gn_id") or enriched_article.get("gn_id"),
                "source": meta.get("source") or enriched_article.get("source"),
                "published_at": meta.get("published_at") or enriched_article.get("published_at"),
                "date": meta.get("published_at") or enriched_article.get("published_at"),  # For Cosmos DB
                "summary": summary,
                "categories": categories,  # Array of categories
                "ingested_at": int(time.time()),
            })

            summarized_articles.append(enriched_article)
            processed += 1

        except Exception as e:
            failures += 1
            print(f"[WARN] Failed to summarize article: {e}")

    print(f"Summarization complete. Processed={processed}, Failures={failures}")
    return summarized_articles


# ----------------------------
# Main pipeline
# ----------------------------
def process_collection(
    qdrant_url: str,
    collection_name: str,
    batch_size: int = 128,
    sleep_between_batches: float = 0.0,  # set small sleep if you need to rate-limit
):
    client = QdrantClient(url=qdrant_url)

    next_page = None
    processed = 0
    failures = 0

    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=next_page,
        )

        if not points:
            break

        for p in points:
            try:
                payload = p.payload or {}
                content = normalize_content(payload)
                if not content:
                    # Skip if no content to summarize
                    continue

                meta = extract_metadata(payload)
                summary, categories = call_llm(content)

                # Build final document for storage
                doc_id = build_doc_id(meta.get("url"), meta.get("gn_id"), getattr(p, "id", None))
                doc = {
                    "id": doc_id,
                    "url": meta.get("url"),
                    "title": meta.get("title"),
                    "gn_id": meta.get("gn_id"),
                    "source": meta.get("source"),
                    "published_at": meta.get("published_at"),
                    "summary": summary,
                    "categories": categories,                       # required (from BASE)
                    "qdrant_point_id": getattr(p, "id", None),
                    "ingested_at": int(time.time()),                 # unix timestamp
                }

                upsert_summary(doc)
                processed += 1
                
                # Print readable output
                print(f"\n{'='*100}")
                print(f"Title: {meta.get('title', 'N/A')}")
                print(f"URL: {meta.get('url', 'N/A')}")
                print(f"Summary: {summary}")
                print(f"Categories: {', '.join(categories) if categories else 'N/A'}")
                print(f"{'='*100}")

            except Exception as e:
                failures += 1
                # You may replace with logging
                print(f"[WARN] Failed for point {getattr(p, 'id', None)}: {e}")

        if sleep_between_batches > 0:
            time.sleep(sleep_between_batches)

        if next_page is None:
            break

    print(f"Done. Processed={processed}, Failures={failures}")


if __name__ == "__main__":
    if not QDRANT_COLLECTION:
        raise ValueError("Please set QDRANT_COLLECTION environment variable.")
    process_collection(QDRANT_URL, QDRANT_COLLECTION)