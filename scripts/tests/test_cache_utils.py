import json
import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from cache_utils import save_json_cache  # noqa: E402


class CacheUtilsTests(unittest.TestCase):
    def test_saves_lone_surrogate_as_json_escape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "cache.json")

            save_json_cache(path, {"page": "text\udf5d"})

            with open(path, encoding="utf-8") as cache_file:
                self.assertEqual(json.load(cache_file), {"page": "text\udf5d"})

    def test_failed_save_preserves_existing_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "cache.json")
            with open(path, "w", encoding="utf-8") as cache_file:
                json.dump({"stable": True}, cache_file)

            with self.assertRaises(TypeError):
                save_json_cache(path, {"invalid": object()})

            with open(path, encoding="utf-8") as cache_file:
                self.assertEqual(json.load(cache_file), {"stable": True})
            self.assertFalse(os.path.exists(f"{path}.tmp"))


if __name__ == "__main__":
    unittest.main()
