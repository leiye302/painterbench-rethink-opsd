import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from painter_rl.model_assets import read_lock, download_part, prepare_model

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

    def test_server_ignoring_range_restarts_instead_of_appending(self):
        raw = b"not actual model weights"
        class Response(io.BytesIO):
            status = 200
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.bin"
            path.with_name(path.name+".downloading").write_bytes(raw[:5])
            record = self.fixture()["files"]["test.bin"]["parts"][0]
            with patch("urllib.request.urlopen", return_value=Response(raw)):
                download_part("https://example.test/test.bin", path, record)
            self.assertEqual(path.read_bytes(), raw)

    def test_bad_download_is_never_marked_complete(self):
        class Response(io.BytesIO):
            status = 200
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"test.bin"
            record = self.fixture()["files"]["test.bin"]["parts"][0]
            with patch("urllib.request.urlopen", return_value=Response(b"wrong")):
                with self.assertRaises(ValueError):
                    download_part("https://example.test/test.bin", path, record)
            self.assertFalse(path.exists())

    def test_lossless_assembly_and_repeat_verification(self):
        pieces = {"weights.part-000": b"exact tensor ", "weights.part-001": b"bytes",
                  "stage1_provenance.json": b'{"step":520,"epoch":2}'}
        def item(raw):
            return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        lock = self.fixture()
        whole = pieces["weights.part-000"]+pieces["weights.part-001"]
        lock["files"] = {"weights.safetensors": {**item(whole), "parts": [
            {"asset": n, **item(pieces[n])} for n in ("weights.part-000", "weights.part-001")]},
            "stage1_provenance.json": {**item(pieces["stage1_provenance.json"]), "parts": [
                {"asset": "stage1_provenance.json", **item(pieces["stage1_provenance.json"])}]}}
        def fake_fetch(url, path, record):
            path.write_bytes(pieces[url.rsplit("/", 1)[1]])
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/"model"
            with patch("painter_rl.model_assets.download_part", side_effect=fake_fetch):
                result = prepare_model(target, lock)
            self.assertEqual(result["step"], 520)
            self.assertEqual((target/"weights.safetensors").read_bytes(), whole)
            self.assertFalse(target.with_name("model.partial").exists())
            with patch("painter_rl.model_assets.download_part", side_effect=AssertionError("redownload")):
                self.assertEqual(prepare_model(target, lock)["step"], 520)


if __name__ == "__main__":
    unittest.main()
