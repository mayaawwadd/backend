from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import re
import random
import json
from typing import Optional
from dotenv import load_dotenv
import time
import shutil
import glob
from urllib.parse import urlparse
from app.services.document_manager import DocumentManager

# Setup debug log file in app/files/my_debug_files
DEBUG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files", "my_debug_files"))
os.makedirs(DEBUG_DIR, exist_ok=True)
LOG_FILE = os.path.join(DEBUG_DIR, f"terminal_output_{int(time.time())}.txt")


def _write_debug(msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"{ts} | {msg}\n")
    except Exception:
        # swallowing any logging errors to avoid interfering with main flow
        pass

# Load the .env file located in the app/ folder so env vars are available
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # fallback to default behavior (search current working directory)
    load_dotenv()

app = FastAPI()

# Prefer Nora API settings if present, otherwise fall back to Azure OpenAI env names
IMAGE_API_URL = os.getenv("NORA_API_URL") or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_ENDPOINT")
IMAGE_API_KEY = os.getenv("NORA_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_API_KEY")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
def generate_image(data: PromptRequest):
    # curate the prompt, perform the API call and return a JSON object with the url when available
    try:
        curated_prompt = to_safe_concept_prompt(data.prompt)
    except Exception:
        curated_prompt = data.prompt

    raw = call_image_api(curated_prompt)
    image_url, is_default = extract_image_url(raw)
    if image_url:
        saved = save_image_from_url(image_url, title=data.prompt)
        if is_default:
            return {"image_url": "", "local_image_path": saved or ""}
        return {"image_url": image_url, "local_image_path": saved or ""}
    return raw


def call_image_api(prompt: str) -> dict:
    """Make the image generation POST request and return the raw JSON response."""
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

    # Debug: show the prompt being sent
    try:
        _write_debug(f'request to generate image was send with prompt "{prompt}"')
    except Exception:
        pass

    try:
        response = requests.post(url, headers=headers, json=body)
        try:
            resp_json = response.json()
        except Exception:
            resp_json = {"error": {"message": "invalid_json_response"}, "raw_text": response.text}

        # Debug: show the raw JSON response received
        try:
            _write_debug(f'response recieved to generate images JSOn recieved is "{json.dumps(resp_json, ensure_ascii=False)}"')
        except Exception:
            _write_debug('response recieved, but failed to stringify JSON')

        return resp_json
    except Exception:
        _write_debug('request_failed: exception while calling image API')
        return {"error": {"message": "request_failed"}}


def extract_image_url(response_json: dict) -> tuple[str | None, bool]:
    """Extract the generated image URL from the API response.

    Returns the URL string if present, otherwise None.
    """
    # If the API returned an error (e.g., content policy violation),
    # fall back to selecting a random default image file from
    # app/files/images/Default
    if response_json.get("error"):
        try:
            images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files", "images", "Default"))
            if os.path.isdir(images_dir):
                files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
                if files:
                    chosen = random.choice(files)
                    chosen_path = os.path.abspath(os.path.join(images_dir, chosen))
                    _write_debug(f'unable to extract URL, default image file used "{chosen_path}"')
                    # Return the local default path and mark as default
                    return chosen_path, True
        except Exception:
            _write_debug('unable to extract URL, no default image available')
            return None, False

    try:
        data = response_json.get("data")
        if not data or not isinstance(data, list):
            return None, False
        first = data[0]
        url = first.get("url")
        if isinstance(url, str) and url:
            try:
                _write_debug(f'Extracted URL is "{url}"')
            except Exception:
                pass
            return url, False
    except Exception:
        return None, False
    return None, False


def _sanitize_filename(s: str, ext: str = ".png", max_len: int = 120) -> str:
    # Basic filename sanitizer: keep alphanumerics, dash, underscore; replace spaces
    s = s.strip().lower()
    s = re.sub(r"[\\/:*?\"<>|]+", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-\.]+", "", s)
    if not s:
        s = f"image_{int(time.time())}"
    if len(s) > max_len:
        s = s[:max_len]
    if not s.endswith(ext):
        s = s + ext
    return s


def save_image_from_url(url: str, title: Optional[str] = None, dest_dir: Optional[str] = None) -> str | None:
    """Save an image from an HTTP(S) URL or copy a local file into the
    `Responses` folder. File is named using the provided `title` (sanitized).
    Returns the absolute path to the saved file or None on failure.
    """
    try:
        # Use DocumentManager to determine and ensure base files dir
        dm = DocumentManager(base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files")))
        base_dir = dm.base_dir
        # Store responses under images/responses
        responses_dir = dest_dir or os.path.join(base_dir, "images", "responses")
        dm.ensure_dir(responses_dir)

        parsed = urlparse(url)
        # Determine extension
        ext = ".png"
        if parsed.scheme in ("http", "https"):
            # try to infer from URL path
            path_ext = os.path.splitext(parsed.path)[1]
            if path_ext:
                ext = path_ext
        else:
            # local file
            _, path_ext = os.path.splitext(url)
            if path_ext:
                ext = path_ext

        # Build filename from title
        if title:
            fname = _sanitize_filename(title, ext=ext)
        else:
            fname = f"img_{int(time.time())}_{random.randint(1000,9999)}{ext}"

        dest = os.path.join(responses_dir, fname)
        # Avoid overwriting
        if os.path.exists(dest):
            name, e = os.path.splitext(fname)
            dest = os.path.join(responses_dir, f"{name}_{int(time.time())}{e}")

        # If it's a local file path, copy it
        if parsed.scheme == "" and os.path.isfile(url):
            try:
                shutil.copyfile(url, dest)
                return os.path.abspath(dest)
            except Exception:
                return None

        # Otherwise attempt to download via HTTP
        if parsed.scheme in ("http", "https"):
            try:
                resp = requests.get(url, stream=True, timeout=15)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(1024 * 8):
                        if chunk:
                            f.write(chunk)
                return os.path.abspath(dest)
            except Exception:
                return None

    except Exception:
        return None

    return None


def to_safe_concept_prompt(user_prompt: str) -> str:
    # 1) Strip domains and outlet names (basic heuristics)
    banned_tokens = ["reuters.com", "forbes.com", "bloomberg", "the new york times", 
                     "wired", "britannica", "mit news", "google cloud blog", 
                     "foreign affairs", "cnbc", "npr"]
    p = user_prompt.lower()
    for t in banned_tokens:
        p = p.replace(t, "")
    # 2) If it's just a URL or nearly empty after stripping, produce a generic concept
    if len(p.strip()) < 15:
        return ("Create an artistic, magazine-style conceptual illustration on modern "
                "technology and society, with no logos, no text, and no specific brands.")
    # 3) Wrap into a neutral concept with explicit constraints
    return (f"Create a conceptual editorial illustration inspired by the idea: '{p.strip()}'. "
            "Avoid logos, trademarks, specific company names, or reproducing any website layout. "
            "No text overlays. Depict diverse people and a modern setting. Use a tasteful, "
            "stylized approach suitable for a general-audience magazine.")


def process_scraper_results(json_path: str, output_path: Optional[str] = None) -> int:
    """Read a JSON file containing a list of scraped results, generate images
    from each `title` (if present), extract the image URL, and attach it as
    the `url` field on each item. Writes back to `output_path` (or overwrites
    `json_path` if not provided).

    Returns the number of items processed.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return 0

    if not isinstance(items, list):
        return 0

    # Count how many items actually need an image request (have a non-empty title)
    total_titles = 0
    for item in items:
        try:
            if (item.get("title") or "").strip():
                total_titles += 1
        except Exception:
            continue

    processed_calls = 0

    for item in items:
        try:
            title = (item.get("title") or "").strip()
        except Exception:
            title = ""

        # If item is marked skipped, do not generate and ensure empty image_url and local_image_path
        if item.get("skipped"):
            item["image_url"] = ""
            item["local_image_path"] = ""
            continue

        # If text field is empty, do not generate and ensure empty image_url and local_image_path
        try:
            if (item.get("text") or "").strip() == "":
                item["image_url"] = ""
                item["local_image_path"] = ""
                continue
        except Exception:
            pass

        if title:
            try:
                curated = to_safe_concept_prompt(title)
            except Exception:
                curated = title
            # Check if an image already exists for this prompt in responses dir
            try:
                dm = DocumentManager(base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files")))
                base_dir = dm.base_dir
                responses_dir = os.path.join(base_dir, "images", "responses")
                os.makedirs(responses_dir, exist_ok=True)
                # build a filename stem and look for any matching file (any extension)
                stem = _sanitize_filename(title, ext="")
                pattern = os.path.join(responses_dir, f"{stem}*")
                matches = glob.glob(pattern)
                if matches:
                    saved_existing = os.path.abspath(matches[0])
                    _write_debug(f'Existing image found for prompt "{title}" at "{saved_existing}", skipping generation')
                    item["image_url"] = ""
                    item["local_image_path"] = saved_existing
                    continue
            except Exception:
                # if anything goes wrong with the existence check, proceed to generate
                pass
            raw = call_image_api(curated)
            url, is_default = extract_image_url(raw)
            if url:
                saved = save_image_from_url(url, title=title)
                if is_default:
                    item["image_url"] = ""
                    item["local_image_path"] = saved or ""
                else:
                    item["image_url"] = url or ""
                    item["local_image_path"] = saved or ""
            else:
                item["image_url"] = ""
                item["local_image_path"] = ""

            # Track and throttle API calls in batches of 3
            try:
                processed_calls += 1
                if processed_calls % 3 == 0 and processed_calls < total_titles:
                    _write_debug(f"sent {processed_calls} image requests, waiting 1 minute before next batch...")
                    time.sleep(60)
            except Exception:
                pass
        else:
            item.setdefault("image_url", "")
            item.setdefault("local_image_path", "")

    out_path = output_path or json_path
    try:
        # Use DocumentManager to save the resulting JSON to ensure dirs exist
        dm = DocumentManager(base_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files")))
        DocumentManager.save_json_file(out_path, items, overwrite=True)
    except Exception:
        return 0

    return len(items)