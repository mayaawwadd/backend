import os
import json
from typing import Any, Optional


class DocumentManager:
    """Utility class for creating, replacing, and reading document files.

    Place an instance in front of a folder (base_dir) and use convenience
    methods to read/write JSON or text files. Includes a static method to
    create or replace a file.
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = os.path.abspath(base_dir)
        else:
            # default to repository files folder adjacent to app package
            self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "files"))

    def ensure_dir(self, path: Optional[str] = None) -> str:
        dirpath = path or self.base_dir
        os.makedirs(dirpath, exist_ok=True)
        return dirpath

    @staticmethod
    def save_json_file(path: str, data: Any, overwrite: bool = True, encoding: str = "utf-8") -> str:
        """Write a JSON-serializable object to `path`.

        If `overwrite` is True the file will be replaced. If False and the
        file exists, a FileExistsError will be raised.
        Returns the path written to.
        """
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        mode = "w" if overwrite else "x"
        with open(path, mode, encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return path

    def save_results(self, filename: str, data: Any, overwrite: bool = True) -> str:
        """Convenience wrapper that ensures base dir exists and writes file."""
        self.ensure_dir(self.base_dir)
        # If saving scraper results, place them in a dedicated subfolder
        name = os.path.basename(filename)
        if name.startswith("scraper_results"):
            target_dir = os.path.join(self.base_dir, "scraper_results")
            os.makedirs(target_dir, exist_ok=True)
            out_path = os.path.join(target_dir, name)
        else:
            out_path = os.path.join(self.base_dir, filename)

        return self.save_json_file(out_path, data, overwrite=overwrite)

    def read_json_file(self, path: str, encoding: str = "utf-8") -> Any:
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)
