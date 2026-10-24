import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from painter_rl.model_assets import read_lock, download_part

ROOT = Path(__file__).resolve().parents[1]


class ReleaseAssets(unittest.TestCase):
    def fixture(self):
        raw = b"not actual model weights"
        digest = hashlib.sha256(raw).hexdigest()
        return {"schema": "github_release_chunked_model_v1", "step": 520, "epoch": 2,
                "marker_token_id": 248077, "repository": "owner/repo", "release_tag": "v1",
                "files": {"test.bin": {"bytes": len(raw), "sha256": digest,
                    "parts": [{"asset": "test.bin", "bytes": len(raw), "sha256": digest}]}}}

    def test_release_lock_rejects_path_traversal_and_bad_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"lock.json"
            good = self.fixture()
            path.write_text(json.dumps(good))
            self.assertEqual(read_lock(path)["step"], 520)
            good["files"]["../escape"] = good["files"].pop("test.bin")
            path.write_text(json.dumps(good))
            with self.assertRaises(ValueError):
                read_lock(path)
            bad = self.fixture()
            bad["files"]["test.bin"]["sha256"] = "unverified"
            path.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                read_lock(path)

    def test_already_verified_part_never_downloads(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.bin"
            path.write_bytes(b"not actual model weights")
            record = self.fixture()["files"]["test.bin"]["parts"][0]
            with patch("urllib.request.urlopen", side_effect=AssertionError("unneeded network")):
                download_part("https://example.test/test.bin", path, record)

    def test_declared_baseline_identical_files_really_match(self):
        lock = json.loads((ROOT/"artifacts/baseline.lock.json").read_text())
        self.assertEqual(lock["commit"], "392a5d99beb5edada502ca8806190e9ecc49a703")
        for name, expected in lock["unchanged_files"].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
