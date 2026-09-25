"""
OPTIONAL / MANUAL integration test: the REAL YOLO weights against the REAL road video.

Skipped automatically when the local weights are not present — fetch them first:

    python -m app.ai.fetch_model

Unit tests (tests/test_ai_helmet.py) never need this; this file proves the actual
model produces helmet-related detections from camera_feed/camera_main.mp4 on CPU
(no GPU required), and optionally runs the full vertical slice when
RUN_AI_FULL_TESTS=1 (slow: processes the whole clip).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

MODEL = Path("models/helmet_model.pt")
ROAD_VIDEO = Path("camera_feed/camera_main.mp4")

pytestmark = pytest.mark.skipif(
    not MODEL.is_file(),
    reason="local helmet weights missing — run: python -m app.ai.fetch_model",
)


def _sample_frames(path: Path, count: int = 6):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cannot open {path}"
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for pos in [int(total * i / (count + 1)) for i in range(1, count + 1)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            yield pos, frame
    cap.release()


def test_real_model_loads_and_reports_health():
    from app.ai.helmet_detector import UltralyticsHelmetDetector

    det = UltralyticsHelmetDetector()
    health = det.health_check()
    assert health["status"] == "READY"
    assert health["model_name"] == "yolov8n-helmet-detection"
    assert set(health["classes"]) == {"With Helmet", "Without Helmet"}


def test_real_model_detects_helmets_in_road_video():
    from app.ai.helmet_detector import UltralyticsHelmetDetector
    from app.ai.schemas import HELMET_CLASSES

    assert ROAD_VIDEO.is_file(), f"road video missing: {ROAD_VIDEO}"
    det = UltralyticsHelmetDetector()
    found = []
    for pos, frame in _sample_frames(ROAD_VIDEO, count=6):
        for d in det.detect_frame(frame, frame_number=pos, timestamp=pos / 30.0):
            found.append(d)
            assert 0.0 <= d.confidence <= 1.0
            x1, y1, x2, y2 = d.bbox
            assert x2 > x1 and y2 > y1
    classes = {d.class_name for d in found}
    assert found, "expected at least one helmet-related detection in camera_main.mp4"
    assert classes & set(HELMET_CLASSES), f"expected helmet classes, got {classes}"


@pytest.mark.skipif(
    os.environ.get("RUN_AI_FULL_TESTS") != "1",
    reason="full vertical slice (whole video, real model) — set RUN_AI_FULL_TESTS=1 to run",
)
def test_full_vertical_slice_on_real_video(tmp_path, monkeypatch):
    """Real model + real road video -> confirmed events -> M3 -> Mock M4."""
    from app.ai.topology import ensure_ai_video_topology
    from app.ai.video_processor import VideoProcessor
    from app.db.database import SessionLocal
    from app.db.repositories import EventRepository
    from app.services.publisher import get_mock_m4_consumer

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "evidence"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    ensure_ai_video_topology()

    outcome = VideoProcessor().process("CAM-001", str(ROAD_VIDEO))
    metrics = outcome["metrics"]
    assert metrics["frames_decoded"] > 100
    assert metrics["detections"] > 0
    assert metrics["violations"] >= 1

    ingest = outcome["ingest_results"][0]
    assert ingest.status in ("ACCEPTED", "DUPLICATE")
    with SessionLocal() as db:
        record = EventRepository(db).get_by_event_id(ingest.canonical_event_id)
        assert record is not None
        assert record.event_type == "HELMET_VIOLATION"
        assert record.provider == "KAAVAL_AI"
        assert record.camera_id == "CAM-001"
        assert record.processing_node_id == "RDK-X5-001"
        assert record.evidence_ref.get("type") == "local_file"
        assert Path(record.evidence_ref["uri"]).is_file()
    m4 = get_mock_m4_consumer().get_events(limit=50)
    assert any(e["event"]["event_id"] == ingest.canonical_event_id for e in m4)
    get_settings.cache_clear()
