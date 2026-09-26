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


@pytest.fixture(scope="module", autouse=True)
def _database():
    """The real-model tests ingest through the EXISTING M3 pipeline, so the schema and seed
    topology must exist even when this file is run on its own (not just after test_ai_helmet)."""
    from app.db.database import SessionLocal, init_db
    from app.db.seed import seed_initial_topology

    init_db()
    with SessionLocal() as db:
        seed_initial_topology(db)


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


class _RecordingDetector:
    """Wraps the real detector and records every observation, so a test can prove that
    detections happened at many different timestamps — not on a single frame."""

    def __init__(self, inner):
        self.inner = inner
        self.seen = []          # (frame_number, timestamp, class_name, confidence, track_id)
        self.last_latency_ms = None
        self.model_name = inner.model_name
        self.model_version = inner.model_version

    def reset_tracker(self):
        self.inner.reset_tracker()

    def health_check(self):
        return self.inner.health_check()

    def detect_frame(self, frame, frame_number: int = 0, timestamp: float = 0.0):
        detections = self.inner.detect_frame(frame, frame_number=frame_number, timestamp=timestamp)
        self.last_latency_ms = self.inner.last_latency_ms
        for d in detections:
            self.seen.append((frame_number, timestamp, d.class_name, d.confidence, d.track_id))
        return detections


def test_two_videos_back_to_back_in_one_process_survive_the_tracker_reset():
    """Regression for the server bug: the SECOND video processed by one detector used to die
    with `IndexError: list index out of range` because `reset_tracker()` emptied
    `predictor.trackers`, which Ultralytics indexes as `trackers[0]` on the next frame."""
    from app.ai.helmet_detector import UltralyticsHelmetDetector

    detector = UltralyticsHelmetDetector()
    frame = next(iter(_sample_frames(ROAD_VIDEO, count=1)))[1]
    first = detector.detect_frame(frame, frame_number=0, timestamp=0.0)     # sets up ByteTrack
    detector.reset_tracker()                                               # a new video starts
    second = detector.detect_frame(frame, frame_number=0, timestamp=0.0)   # must not raise
    assert isinstance(first, list) and isinstance(second, list)
    assert detector.tracker_resets == 1


def test_real_model_covers_every_frame_of_a_bounded_run(tmp_path, monkeypatch):
    """Fast real-model smoke test of the frame loop: 60 decoded frames -> 60 inferences."""
    from app.ai.helmet_detector import UltralyticsHelmetDetector
    from app.ai.video_processor import VideoProcessor

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "evidence"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    detector = _RecordingDetector(UltralyticsHelmetDetector())
    outcome = VideoProcessor(detector=detector).process("CAM-001", str(ROAD_VIDEO), max_frames=60)
    metrics = outcome["metrics"]
    assert metrics["frames_processed"] == 60 and metrics["frames_decoded"] == 60
    assert metrics["frames_skipped"] == 0 and metrics["sampling_mode"] == "every_frame"
    assert metrics["reached_last_frame"] is False          # stopped by max_frames on purpose
    assert len(detector.seen) == metrics["detections"]     # no detection is lost in normalization
    # counting invariant: people can never outnumber the frame-level detections behind them
    assert metrics["unique_violators"] <= metrics["violation_detections"]
    get_settings.cache_clear()


@pytest.mark.skipif(
    os.environ.get("RUN_AI_FULL_TESTS") != "1",
    reason="full vertical slice (whole video, real model) — set RUN_AI_FULL_TESTS=1 to run",
)
def test_full_vertical_slice_on_real_video(tmp_path, monkeypatch):
    """MANUAL ACCEPTANCE TEST: real model + the whole real road video, beginning to end.

        video -> every frame -> YOLO -> unique-rider aggregation -> evidence
        -> existing M3 ingestion -> canonical event -> SQLite -> Mock M4

    Asserts the acceptance list: the last frame is reached, detections occur at many
    timestamps, the same rider in many frames counts once, evidence is one image per unique
    violator, and the events really land in M3 and M4.
    """
    from app.ai.helmet_detector import UltralyticsHelmetDetector
    from app.ai.topology import ensure_ai_video_topology
    from app.ai.video_processor import VideoProcessor
    from app.db.database import SessionLocal
    from app.db.repositories import EventRepository
    from app.services.publisher import get_mock_m4_consumer

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "evidence"))
    from app.core.config import get_settings

    get_settings.cache_clear()
    ensure_ai_video_topology()

    detector = _RecordingDetector(UltralyticsHelmetDetector())
    outcome = VideoProcessor(detector=detector).process("CAM-001", str(ROAD_VIDEO))
    metrics = outcome["metrics"]

    # --- coverage: the WHOLE video, frame by frame ---
    assert metrics["frames_decoded"] > 100
    assert metrics["total_frames"] == metrics["frames_decoded"]
    assert metrics["frames_processed"] == metrics["frames_decoded"]
    assert metrics["frames_skipped"] == 0
    assert metrics["reached_last_frame"] is True and metrics["decode_complete"] is True
    assert metrics["detections"] > 0

    # --- detections at MULTIPLE timestamps throughout the video ---
    without = [s for s in detector.seen if s[2] == "WITHOUT_HELMET"]
    assert without, "expected at least one WITHOUT_HELMET observation in the real video"
    timestamps = sorted({round(s[1], 2) for s in without})
    assert len(timestamps) >= 3, f"violations seen on too few frames: {timestamps[:5]}"
    assert timestamps[-1] - timestamps[0] > 0.5, "violations seen only around one moment"

    # --- ONE violation per unique rider, never one per frame ---
    assert metrics["violation_detections"] >= metrics["unique_violators"]
    assert metrics["violations"] == metrics["unique_violators"]
    assert len(outcome["violations"]) == metrics["unique_violators"]
    assert metrics["unique_violators"] >= 1
    track_ids = [s[4] for s in without if s[4] is not None]
    if track_ids:
        assert metrics["unique_violators"] <= len(set(track_ids))
    for violation in outcome["violations"]:
        assert violation["observations"] >= 3                 # confirmation rule
        assert violation["violation_detections"] >= violation["observations"]
        assert 0.0 <= violation["first_seen"] <= violation["last_seen"] <= metrics["video_duration_seconds"]

    # --- one evidence image per unique violator (not one per frame) ---
    evidence_files = sorted((tmp_path / "evidence").rglob("*.jpg"))
    assert len(evidence_files) == metrics["evidence_frames"] == metrics["unique_violators"]

    # --- every unique violator becomes a canonical event in M3 and is pushed to Mock M4 ---
    assert len(outcome["ingest_results"]) == metrics["unique_violators"]
    accepted = 0
    m4 = get_mock_m4_consumer().get_events(limit=200)
    m4_ids = {e["event"]["event_id"] for e in m4}
    for ingest in outcome["ingest_results"]:
        assert ingest.status in ("ACCEPTED", "DUPLICATE")
        accepted += 1 if ingest.status == "ACCEPTED" else 0
        with SessionLocal() as db:
            record = EventRepository(db).get_by_event_id(ingest.canonical_event_id)
            assert record is not None
            assert record.event_type == "HELMET_VIOLATION"
            assert record.provider == "KAAVAL_AI"
            assert record.camera_id == "CAM-001"
            assert record.processing_node_id == "RDK-X5-001"
            assert record.evidence_ref.get("type") == "local_file"
            assert Path(record.evidence_ref["uri"]).is_file()
            assert record.attributes.get("unique_violator") is True
            assert record.attributes.get("violation_detections", 0) >= 3
        assert ingest.canonical_event_id in m4_ids
    assert accepted == metrics["events_ingested"]
    get_settings.cache_clear()
