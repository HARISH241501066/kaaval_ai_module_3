"""
Fetch the local helmet YOLO weights (no cloud API key, no hosted inference).

    python -m app.ai.fetch_model           # download -> AI_MODEL_PATH, verify pinned SHA-256
    python -m app.ai.fetch_model --info    # additionally load the model and print its classes

The model card (source, license, classes, hash) lives in app.ai.config.MODEL_METADATA
and is documented in the README.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

from app.ai.config import MODEL_METADATA, get_ai_config

log = logging.getLogger("sentinel.m3.ai.fetch")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(dest: str | None = None, force: bool = False) -> Path:
    cfg = get_ai_config()
    target = Path(dest or cfg.model_path)
    expected = str(MODEL_METADATA["sha256"])
    url = str(MODEL_METADATA["source_url"])

    if target.exists() and not force:
        actual = _sha256(target)
        if actual == expected:
            log.info("model already present and verified: %s", target)
            return target
        raise SystemExit(
            f"{target} exists but its SHA-256 ({actual}) does not match the pinned hash "
            f"({expected}); re-run with --force to re-download"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    log.info("downloading %s -> %s", url, target)
    try:
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:  # noqa: S310 pinned https URL
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"download failed ({e.__class__.__name__}: {e}); inference stays local, no API fallback")
    actual = _sha256(tmp)
    if actual != expected:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"downloaded file hash {actual} != pinned {expected}; refusing to use it")
    tmp.replace(target)
    log.info("model verified (sha256 ok, %d bytes)", target.stat().st_size)
    return target


def print_model_card() -> None:
    print("=" * 64)
    print("KAAVAL AI HELMET MODEL CARD")
    print("=" * 64)
    for key in ("model_name", "model_version", "base_model", "source", "source_url", "format",
                "classes", "license", "sha256", "metrics", "cloud_api_required", "notes"):
        print(f"{key:>18}: {MODEL_METADATA.get(key)}")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Download / verify the local helmet YOLO weights")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--info", action="store_true", help="load the model and print its class names")
    ap.add_argument("--dest", default=None, help="override AI_MODEL_PATH")
    args = ap.parse_args(argv)

    path = fetch_model(args.dest, force=args.force)
    print_model_card()
    print(f"{'weights':>18}: {path}")
    if args.info:
        from ultralytics import YOLO

        print(f"{'loaded classes':>18}: {YOLO(str(path)).names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
