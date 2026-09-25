"""
Real object detection for live frames: YOLOv4-tiny (COCO) through OpenCV's DNN module, on CPU.

It finds vehicles and people. The tracker (tracker.py) turns those into Sentinel events.
Model files are downloaded once into LIVE_MODEL_DIR (git-ignored) and checked against pinned
SHA-256 hashes:

    python -m app.live.detector --fetch
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("sentinel.m3.detector")

MODEL_VERSION = "yolov4-tiny-coco"
INPUT_SIZE = (416, 416)
_BASE = "https://raw.githubusercontent.com/AlexeyAB/darknet/master"
# name -> (url, sha256). Hashes were recorded from the upstream files on first download.
MODEL_FILES = {
    "yolov4-tiny.cfg": (f"{_BASE}/cfg/yolov4-tiny.cfg",
                        "f858e3724962eedf3ac44e3b6cb3f0c3d9ed067c306bb831f539c578b924c90e"),
    "yolov4-tiny.weights": ("https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
                            "cf9fbfd0f6d4869b35762f56100f50ed05268084078805f0e7989efe5bb8ca87"),
    "coco.names": (f"{_BASE}/data/coco.names",
                   "634a1132eb33f8091d60f2c346ababe8b905ae08387037aed883953b7329af84"),
}
# COCO label -> Sentinel vehicle class
VEHICLES = {"car": "CAR", "motorbike": "TWO_WHEELER", "bus": "BUS", "truck": "TRUCK"}
KEEP = set(VEHICLES) | {"person"}


class ModelMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Detection:
    label: str                       # COCO label: person, car, motorbike, bus, truck
    conf: float
    box: tuple[int, int, int, int]   # x, y, w, h in frame pixels


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_models(model_dir: str | Path) -> Path:
    d = Path(model_dir)
    d.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in MODEL_FILES.items():
        target = d / name
        if target.exists() and _sha256(target) == sha:
            continue
        log.info("downloading %s", name)
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:   # noqa: S310 (pinned https URLs)
            while chunk := r.read(1 << 20):
                f.write(chunk)
        if _sha256(tmp) != sha:
            tmp.unlink(missing_ok=True)
            raise ModelMissing(f"{name}: downloaded file does not match the pinned SHA-256; refusing to use it")
        tmp.replace(target)
    return d


class Detector:
    """One instance per thread: OpenCV's DNN network is not thread-safe."""

    def __init__(self, model_dir: str | Path, min_conf: float = 0.45, nms: float = 0.4):
        d = Path(model_dir)
        missing = [n for n in MODEL_FILES if not (d / n).exists()]
        if missing:
            raise ModelMissing(f"model files missing in '{d}': {', '.join(missing)}. Run: python -m app.live.detector --fetch")
        self.names = (d / "coco.names").read_text().split()
        self.min_conf, self.nms = min_conf, nms
        net = cv2.dnn_DetectionModel(str(d / "yolov4-tiny.cfg"), str(d / "yolov4-tiny.weights"))
        net.setInputParams(size=INPUT_SIZE, scale=1 / 255.0, swapRB=True)
        net.setNmsAcrossClasses(False)
        self._net = net

    def detect(self, frame: np.ndarray) -> list[Detection]:
        class_ids, scores, boxes = self._net.detect(frame, self.min_conf, self.nms)
        out: list[Detection] = []
        for cid, score, box in zip(np.array(class_ids).reshape(-1), np.array(scores).reshape(-1), boxes):
            label = self.names[int(cid)]
            if label in KEEP:
                x, y, w, h = (int(v) for v in box)
                out.append(Detection(label, float(score), (x, y, w, h)))
        return out


if __name__ == "__main__":
    from app.core.config import get_settings

    ap = argparse.ArgumentParser(description="Download / verify the detector model files")
    ap.add_argument("--fetch", action="store_true", help="download missing model files into LIVE_MODEL_DIR")
    ap.add_argument("--dir", default=get_settings().live_model_dir)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(f"model files ready in {fetch_models(args.dir)}" if args.fetch else "use --fetch")
