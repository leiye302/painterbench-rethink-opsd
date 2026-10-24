"""Download the released exact step-520 model, without a private host or token."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "artifacts/route_sft_step520.lock.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lock(path=LOCK_PATH):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    if (lock["schema"] != "github_release_chunked_model_v1" or lock["step"] != 520 or
            lock["epoch"] != 2 or lock["marker_token_id"] != 248077 or
            not re.fullmatch(r"[\w.-]+/[\w.-]+", lock["repository"]) or
            not re.fullmatch(r"[\w.-]+", lock["release_tag"])):
        raise ValueError("Invalid model lock")
    names = set()
    for name, item in lock["files"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
            raise ValueError("Unsafe model file name")
        if not item["parts"] or item["bytes"] != sum(p["bytes"] for p in item["parts"]):
            raise ValueError("Model chunk sizes do not add up")
        for part in item["parts"]:
            if (not re.fullmatch(r"[A-Za-z0-9_.-]+", part["asset"]) or part["asset"] in names or
                    part["asset"] in (".", "..") or part["bytes"] < 1):
                raise ValueError("Unsafe or duplicate model asset")
            names.add(part["asset"])
        for entry in [item, *item["parts"]]:
            if not re.fullmatch("[a-f0-9]{64}", entry["sha256"]):
                raise ValueError("Model checksum missing")
    return lock


def verify_initialization(directory, lock=None):
    directory = Path(directory)
    lock = read_lock() if lock is None else lock
    for name, record in lock["files"].items():
        path = directory / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["bytes"] or
                sha(path) != record["sha256"]):
            raise ValueError("Step-520 initialization failed SHA verification: " + name)
    provenance = json.loads((directory / "stage1_provenance.json").read_text())
    if provenance["step"] != 520 or provenance["epoch"] != 2:
        raise ValueError("Not the published epoch-2 Route-SFT model")
    return {"step": 520, "model_lock_sha256": sha(LOCK_PATH), "files_verified": len(lock["files"])}


def download_part(url, path, record):
    if path.exists() and path.stat().st_size == record["bytes"] and sha(path) == record["sha256"]:
        return
    pending = path.with_name(path.name + ".downloading")
    offset = pending.stat().st_size if pending.exists() else 0
    if offset > record["bytes"]:
        raise ValueError("Partial download is larger than its fixed asset")
    headers = {"User-Agent": "painterbench-rethink-opsd/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    if offset < record["bytes"]:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=180) as response:
            # A server may ignore Range; never append a full file to a partial one.
            append = offset > 0 and response.status == 206
            with pending.open("ab" if append else "wb") as handle:
                while block := response.read(4 << 20):
                    handle.write(block)
    if pending.stat().st_size != record["bytes"] or sha(pending) != record["sha256"]:
        raise ValueError("Download checksum mismatch; preserve partial evidence: " + path.name)
    pending.replace(path)


def prepare_model(directory, lock=None):
    directory = Path(directory)
    lock = read_lock() if lock is None else lock
    if directory.exists():
        return verify_initialization(directory, lock)
    if directory.is_symlink():
        raise ValueError("Model directory must not be a link")
    pending = directory.with_name(directory.name + ".partial")
    if pending.is_symlink():
        raise ValueError("Partial model directory must not be a link")
    directory.parent.mkdir(parents=True, exist_ok=True)
    pending.mkdir(exist_ok=True)
    parts = pending / ".parts"
    parts.mkdir(exist_ok=True)
    remaining = sum(x["bytes"] for name, x in lock["files"].items() if not (pending / name).exists())
    if shutil.disk_usage(pending).free < remaining + max(x["bytes"] for x in lock["files"].values()) + (2 << 30):
        raise RuntimeError("Insufficient space for the exact FP32 model plus download assembly headroom")
    prefix = f"https://github.com/{lock['repository']}/releases/download/{lock['release_tag']}/"
    for name, record in lock["files"].items():
        path = pending / name
        if path.exists():
            if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
                raise ValueError("Existing partial-model file changed: " + name)
            continue
        for part in record["parts"]:
            download_part(prefix + part["asset"], parts / part["asset"], part)
        assembling = pending / (name + ".assembling")
        with assembling.open("wb") as target:
            for part in record["parts"]:
                with (parts / part["asset"]).open("rb") as source:
                    shutil.copyfileobj(source, target, length=4 << 20)
        if assembling.stat().st_size != record["bytes"] or sha(assembling) != record["sha256"]:
            raise ValueError("Assembled model file checksum mismatch: " + name)
        assembling.replace(path)
        for part in record["parts"]:
            (parts / part["asset"]).unlink()
        print(json.dumps({"verified_file": name, "bytes": record["bytes"]}), flush=True)
    parts.rmdir()
    result = verify_initialization(pending, lock)
    pending.replace(directory)
    return result
