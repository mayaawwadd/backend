import json
import re
import hashlib
from pathlib import Path
from typing import Any

class URLTools:

    HTTP_RE = re.compile(r"^https?://", re.IGNORECASE)

    @staticmethod
    def safe_name_from_url(url: str, max_len: int = 80) -> str:
        h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
        no_scheme = re.sub(r"^https?://", "", url)
        tokens = re.sub(r"[^A-Za-z0-9]+", "-", no_scheme).strip("-")
        if len(tokens) > max_len:
            tokens = tokens[:max_len].rstrip("-")
        return f"{tokens}--{h}" if tokens else h

class FileManager:

    @staticmethod
    def ensure_dir(path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_json(path: str, data: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def write_text(path: str, text: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")


