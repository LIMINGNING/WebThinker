import json
import os
from typing import Any


def save_json_cache(path: str, value: Any) -> None:
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as cache_file:
            json.dump(value, cache_file, ensure_ascii=True, indent=2)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
