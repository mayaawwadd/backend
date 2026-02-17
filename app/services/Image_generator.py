import requests
import os
import re
import random
import time
import shutil
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
RESPONSES_DIR = IMAGES_DIR / "responses"
DEFAULTS_DIR = IMAGES_DIR / "defaults"
DEBUG_DIR = OUTPUT_DIR / "debug"

RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
DEFAULTS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = DEBUG_DIR / f"image_debug_{int(time.time())}.log"



def _write_debug(msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{ts} | {msg}\n")
    except Exception:
        pass

IMAGE_API_URL = os.getenv("NORA_API_URL") or os.getenv("AZURE_OPENAI_ENDPOINT")
IMAGE_API_KEY = os.getenv("NORA_API_KEY") or os.getenv("AZURE_OPENAI_KEY")


def build_safe_editorial_prompt(article: dict) -> str:
    """
    Converts article content into a neutral, safe conceptual prompt.
    Prevents moderation failures from political names, brands, etc.
    """

    summary = (article.get("summary") or "").strip().lower()
    title = (article.get("title") or "").strip().lower()

    text = summary if len(summary) > 40 else title

    political_keywords = [
        "obama", "trump", "biden", "government",
        "department", "election", "congress",
        "racist", "senate"
    ]

    corporate_keywords = [
        "disney", "google", "tiktok", "bloomberg",
        "techcrunch", "wired", "new york times",
        "guardian", "npr"
    ]

    if any(word in text for word in political_keywords):
        theme = "AI-generated misinformation and political communication risks"

    elif any(word in text for word in corporate_keywords):
        theme = "The societal impact of artificial intelligence on media and corporations"

    elif "health" in text:
        theme = "AI systems influencing public health information"

    elif "research" in text or "science" in text:
        theme = "AI accelerating scientific research and innovation"

    elif "funding" in text or "startup" in text:
        theme = "Investment trends in artificial intelligence infrastructure"

    elif "agriculture" in text:
        theme = "Artificial intelligence transforming global agriculture"

    else:
        theme = "The growing influence of artificial intelligence on modern society"

    return (
        f"Create a clean, magazine-style conceptual editorial illustration about {theme}. "
        "No logos. No brand names. No public figures. No text overlays. "
        "Modern digital art style. High quality. Neutral and professional."
    )


# Backwards compatibility (in case anything still calls it)
def to_safe_concept_prompt(user_prompt: str) -> str:
    return (
        "Create a modern editorial illustration about artificial intelligence "
        "and technology. No logos. No brand names. No text overlays."
    )

def call_image_api(prompt: str) -> dict:
    url = f"{IMAGE_API_URL}/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {IMAGE_API_KEY}"
    }

    body = {
        "model": "dall-e-3",
        "prompt": prompt,
        "size": "1024x1024",
        "style": "vivid",
        "quality": "standard",
        "n": 1
    }

    try:
        _write_debug(f'Sending image request with prompt "{prompt}"')

        response = requests.post(url, headers=headers, json=body)

        try:
            data = response.json()
        except Exception:
            _write_debug("Failed to parse response JSON")
            return {"error": {"message": "invalid_json"}}

        if response.status_code != 200:
            _write_debug(
                f"Image API error | status={response.status_code} | body={data}"
            )

        return data

    except Exception as e:
        _write_debug(f"Image API request failed: {str(e)}")
        return {"error": {"message": "request_failed"}}



def extract_image_url(response_json: dict) -> Tuple[Optional[str], bool]:
    """
    Returns (url_or_path, is_default).
    Guarantees a default image if generation fails.
    """

    # If API error → return default image
    if response_json.get("error"):
        default_images = list(DEFAULTS_DIR.glob("*"))
        if default_images:
            chosen = random.choice(default_images)
            _write_debug(f"Using default image fallback: {chosen}")
            return str(chosen.resolve()), True
        return None, False

    try:
        data = response_json.get("data", [])
        if data and "url" in data[0]:
            return data[0]["url"], False
    except Exception:
        pass

    # If no URL returned → fallback
    default_images = list(DEFAULTS_DIR.glob("*"))
    if default_images:
        chosen = random.choice(default_images)
        _write_debug(f"Using default image (no URL in response): {chosen}")
        return str(chosen.resolve()), True

    return None, False


def _sanitize_filename(s: str, ext: str = ".png", max_len: int = 120) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\\/:*?\"<>|]+", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)

    if not s:
        s = f"image_{int(time.time())}"

    if len(s) > max_len:
        s = s[:max_len]

    if not s.endswith(ext):
        s += ext

    return s


def save_image_from_url(url: str, title: Optional[str] = None) -> Optional[str]:
    try:
        parsed = urlparse(url)

        ext = ".png"
        if parsed.path:
            path_ext = Path(parsed.path).suffix
            if path_ext:
                ext = path_ext

        filename = _sanitize_filename(title or f"img_{int(time.time())}", ext)
        dest = RESPONSES_DIR / filename

        if dest.exists():
            dest = RESPONSES_DIR / f"{dest.stem}_{int(time.time())}{ext}"

        # HTTP download
        if parsed.scheme in ("http", "https"):
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return str(dest.resolve())

        # Local file copy (default fallback case)
        if parsed.scheme == "" and Path(url).exists():
            shutil.copyfile(url, dest)
            return str(dest.resolve())

    except Exception as e:
        _write_debug(f"Failed saving image: {e}")
        return None

    return None