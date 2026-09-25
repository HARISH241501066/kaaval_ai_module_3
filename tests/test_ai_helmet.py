"""
KAAVAL AI local helmet detection layer — automated tests.

Covers the acceptance list: config loading, graceful model/video failures, normalized
detections, confidence filtering, violation confirmation, cooldown (no per-frame
events), evidence frames, ingestion through the EXISTING M3 pipeline, canonical
provenance, independent YOLO health, API endpoints.

YOLO inference is mocked here (no GPU, no model download); the real model is exercised
by the optional tests in tests/test_ai_integration.py.
"""
from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.ai.config import get_ai_config, parse_video_sources
from app.ai.evidence import save_evidence_frame
from app.ai.helmet_detector import (
    ModelMissing,
    UltralyticsHelmetDetector,
    normalize_class_name,
    reset_detector_for_tests,
)
from app.ai.kaaval_logic import HelmetViolationLogic, _gate_px
from app.ai.schemas import (
    WITH_HELMET,
    WITHOUT_HELMET,
    ConfirmedViolation,
    Detection,
    JobStatus,
    filter_detections,
)
from app.ai.video_source import LocalVideoSource, VideoError
from app.core.config import get_settings


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _database():
    from app.db.database import init_db, SessionLocal
    from app.db.seed import seed_initial_topology

    init_db()
    with SessionLocal() as db:
        seed_initial_topology(db)


@pytest.fixture
def ai_env(tmp_path, monkeypatch):
    """AI settings pointed at a tmp evidence dir + a synthetic clip; restores globals after."""
    clip = _write_clip(tmp_path / "clip.mp4")
    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("AI_VIDEO_SOURCES", f"CAM-001={clip}")
    monkeypatch.setenv("AI_MODEL_PATH", str(tmp_path / "model.pt"))  # missing unless a test writes it
    monkeypatch.setenv("AI_PROCESSING_NODE_ID", "RDK-X5-001")
    get_settings.cache_clear()
    reset_detector_for_tests()
    yield tmp_path
    reset_detector_for_tests()
    get_settings.cache_clear()


@pytest.fixture
def client(ai_env):
    from app.main import app

    with TestClient(app) as c:
        yield c


def _write_clip(path: Path, frames: int = 40, fps: float = 10.0, size: int = 128) -> Path:
    """Tiny synthetic mp4 (4 s @ 10 fps) so tests never depend on the repo's road videos."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    assert w.isOpened(), "OpenCV could not create the test clip (mp4v codec)"
    for i in range(frames):
        img = np.full((size, size, 3), 40, np.uint8)
        x = 10 + (i * 2) % 60
        cv2.rectangle(img, (x, 40), (x + 40, 100), (200, 200, 200), -1)
        w.write(img)
    w.release()
    return path


class FakeDetector:
    """Stands in for the YOLO model: a rider head that never wears a helmet."""

    def __init__(self, conf: float = 0.91, bbox=(100, 80, 160, 160), track_id: int = 7,
                 classes=(WITHOUT_HELMET,)):
        self.conf = conf
        self.bbox = bbox
        self.track_id = track_id
        self.classes = classes
        self.last_latency_ms = 11.5
        self.model_name = "fake-helmet-model"
        self.model_version = "0.0-test"

    def detect_frame(self, frame, frame_number: int = 0, timestamp: float = 0.0):
        return [
            Detection(class_name=c, confidence=self.conf, bbox=self.bbox,
                      frame_number=frame_number, timestamp=timestamp, track_id=self.track_id)
            for c in self.classes
        ]

    def health_check(self):
        return {"status": "READY", "model_name": self.model_name}


def _mk(class_name: str, conf: float, frame: int, ts: float, bbox=(100, 80, 160, 160),
        track_id: int | None = 7) -> Detection:
    return Detection(class_name, conf, bbox, frame, ts, track_id)


# ---------------------------------------------------------------------------
# 1. configuration
# ---------------------------------------------------------------------------
def test_model_configuration_loads():
    cfg = get_ai_config()
    assert cfg.confidence_threshold == 0.50
    assert cfg.inference_fps == 2.0
    assert cfg.image_size == 1280
    assert isinstance(cfg.device, str) and cfg.device          # default "cpu", env may override
    assert cfg.confirmation_frames == 3
    assert cfg.cooldown_s == 10.0
    assert "CAM-001" in cfg.video_sources
    card = cfg.as_public_dict()["model_card"]
    assert card["license"] == "MIT" and card["cloud_api_required"] is False
    assert card["classes"] == ["With Helmet", "Without Helmet"]


def test_inference_size_scales_with_source_resolution():
    """A global imgsz is the small-object bug: 4K must not be squeezed into 640 px."""
    cfg = get_ai_config()
    assert cfg.image_size_auto is True
    assert cfg.inference_size(2160, 3840) == 1280      # 4K wide shot: the reference size
    assert cfg.inference_size(3840, 2160) == 1280      # orientation does not matter
    assert cfg.inference_size(1080, 1920) == 640       # 1080p stays at the accuracy floor
    assert cfg.inference_size(720, 1280) == 640
    assert cfg.inference_size(2160, 2560) == 864       # in between: proportional, stride-aligned
    assert cfg.inference_size(2160, 3840) % 32 == 0    # always a multiple of the model stride
    # a smaller frame is never upscaled past the configured ceiling
    assert cfg.inference_size(2160, 7680) == cfg.image_size

    # explicit fixed size: AI_IMAGE_SIZE_AUTO=false must be honoured verbatim (benchmarks/tests)
    fixed = dataclasses.replace(cfg, image_size_auto=False, image_size=1920)
    assert fixed.inference_size(2160, 3840) == 1920
    assert fixed.inference_size(240, 320) == 1920

    # a deliberate low ceiling wins over the accuracy floor (CPU-constrained runs)
    cheap = dataclasses.replace(cfg, image_size=320)
    assert cheap.inference_size(2160, 3840) == 320


def test_inference_size_is_reported_in_config():
    cfg = get_ai_config()
    public = cfg.as_public_dict()
    assert public["image_size"] == 1280 and public["image_size_auto"] is True
    assert public["inference_size_4k"] == 1280


def test_video_source_map_parsing_and_resolution(ai_env):
    assert parse_video_sources(" CAM-001=a.mp4 ; CAM-002=b.mp4 ") == {"CAM-001": "a.mp4", "CAM-002": "b.mp4"}
    with pytest.raises(ValueError):
        parse_video_sources("CAM-001-no-equals-sign")
    cfg = get_ai_config()
    cam, path = cfg.resolve_video("CAM-001")
    assert cam == "CAM-001" and path.endswith("clip.mp4")
    with pytest.raises(ValueError):
        cfg.resolve_video("CAM-999")          # unknown camera
    # explicit video_path with a single configured camera is unambiguous
    assert cfg.resolve_video(None, "other.mp4") == ("CAM-001", "other.mp4")
    # ...but ambiguous when several cameras are configured
    multi = dataclasses.replace(cfg, video_sources={"CAM-001": "a.mp4", "CAM-002": "b.mp4"})
    with pytest.raises(ValueError):
        multi.resolve_video(None, "other.mp4")
    with pytest.raises(ValueError):
        multi.resolve_video()                 # nothing given at all


# ---------------------------------------------------------------------------
# 2. model missing -> graceful failure, independent health
# ---------------------------------------------------------------------------
def test_model_file_missing_fails_gracefully(ai_env):
    cfg = get_ai_config()
    assert not cfg.model_file_exists
    with pytest.raises(ModelMissing) as ei:
        UltralyticsHelmetDetector(cfg)
    assert "fetch_model" in str(ei.value)      # actionable message, no cloud fallback

    reset_detector_for_tests()
    from app.ai.helmet_detector import get_detector

    with pytest.raises(ModelMissing):
        get_detector(cfg, record=False)
    from app.ai.helmet_detector import detector_status

    status = detector_status(record=False, config=cfg)
    assert status["status"] == "FAILED" and status["error"]


# ---------------------------------------------------------------------------
# 3. video missing / broken -> graceful failure
# ---------------------------------------------------------------------------
def test_video_file_missing_fails_gracefully(tmp_path):
    src = LocalVideoSource(str(tmp_path / "nope.mp4"))
    with pytest.raises(VideoError) as ei:
        list(src.frames())
    assert "not found" in str(ei.value)


def test_empty_video_file_fails_gracefully(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(VideoError):
        list(LocalVideoSource(str(empty)).frames())


def test_unreadable_video_fails_gracefully(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is definitely not a video container")
    with pytest.raises(VideoError):
        list(LocalVideoSource(str(junk)).frames())


# ---------------------------------------------------------------------------
# 4. normalized detections + confidence filtering + class vocabulary
# ---------------------------------------------------------------------------
def test_class_names_normalise_to_canonical_vocabulary():
    assert normalize_class_name("With Helmet") == WITH_HELMET
    assert normalize_class_name("with_helmet") == WITH_HELMET
    assert normalize_class_name("helmet") == WITH_HELMET
    assert normalize_class_name("Without Helmet") == WITHOUT_HELMET
    assert normalize_class_name("no_helmet") == WITHOUT_HELMET
    assert normalize_class_name("head") == WITHOUT_HELMET
    assert normalize_class_name("person") == "PERSON"
    assert normalize_class_name("motorcycle") == "MOTORCYCLE"


def test_detector_contract_returns_normalized_detections():
    det = FakeDetector(conf=0.9, bbox=(1, 2, 3, 4))
    out = det.detect_frame(np.zeros((64, 64, 3), np.uint8), frame_number=5, timestamp=2.5)
    assert len(out) == 1
    d = out[0]
    assert isinstance(d, Detection)
    assert d.class_name == WITHOUT_HELMET
    assert d.confidence == pytest.approx(0.9)
    assert d.bbox == (1, 2, 3, 4) and d.frame_number == 5 and d.timestamp == 2.5
    as_dict = d.to_dict()
    assert as_dict["class_name"] == "WITHOUT_HELMET" and as_dict["bbox"] == [1, 2, 3, 4]


def test_low_confidence_detections_are_filtered():
    dets = [_mk(WITHOUT_HELMET, 0.30, 0, 0.0), _mk(WITHOUT_HELMET, 0.79, 0, 0.0)]
    kept = filter_detections(dets, 0.50)
    assert len(kept) == 1 and kept[0].confidence == pytest.approx(0.79)


def test_rule_engine_ignores_low_confidence_helmet_detections():
    cfg = get_ai_config()
    logic = HelmetViolationLogic("CAM-001", cfg)
    for i in range(5):
        assert logic.observe([_mk(WITHOUT_HELMET, 0.40, i, i * 0.5)], i, i * 0.5) == []
    assert logic.active_states() == []


# ---------------------------------------------------------------------------
# 5. violation confirmation (no event per frame) + cooldown
# ---------------------------------------------------------------------------
def test_helmet_violation_requires_confirmation_frames():
    cfg = get_ai_config()
    logic = HelmetViolationLogic("CAM-001", cfg)
    assert logic.observe([_mk(WITHOUT_HELMET, 0.9, 1, 0.5)], 1, 0.5) == []   # 1st observation
    assert logic.observe([_mk(WITHOUT_HELMET, 0.9, 2, 1.0)], 2, 1.0) == []   # 2nd observation
    events = logic.observe([_mk(WITHOUT_HELMET, 0.9, 3, 1.5)], 3, 1.5)       # 3rd confirms
    assert len(events) == 1
    assert isinstance(events[0], ConfirmedViolation)
    assert events[0].event_id.startswith("EVT-")
    assert events[0].camera_id == "CAM-001"
    assert events[0].confidence == pytest.approx(0.9)
    # ...and the same continuous violation never fires again
    for i in range(4, 12):
        assert logic.observe([_mk(WITHOUT_HELMET, 0.9, i, i * 0.5)], i, i * 0.5) == []


def test_with_helmet_resets_the_episode():
    cfg = get_ai_config()
    logic = HelmetViolationLogic("CAM-001", cfg)
    logic.observe([_mk(WITHOUT_HELMET, 0.9, 1, 0.5)], 1, 0.5)
    logic.observe([_mk(WITHOUT_HELMET, 0.9, 2, 1.0)], 2, 1.0)
    logic.observe([_mk(WITH_HELMET, 0.9, 3, 1.5)], 3, 1.5)                   # helmet appears
    events = logic.observe([_mk(WITHOUT_HELMET, 0.9, 4, 2.0)], 4, 2.0)       # count restarts at 1
    assert events == []
    assert logic.active_states()[0]["hits"] == 1


def test_cooldown_prevents_duplicate_violations():
    cfg = get_ai_config()          # cooldown = 10 s of video time
    logic = HelmetViolationLogic("CAM-001", cfg)

    def episode(start_ts: float, bbox, track: int):
        """Confirm a fresh violation episode starting at start_ts."""
        out = []
        for k in range(cfg.confirmation_frames):
            ts = start_ts + k * 0.5
            out += logic.observe([_mk(WITHOUT_HELMET, 0.9, int(ts * 10), ts, bbox, track)], int(ts * 10), ts)
        return out

    first = episode(0.0, (100, 80, 160, 160), track=7)
    assert len(first) == 1
    # a second rider confirmed 1.5 s later is suppressed by the cooldown
    second = episode(1.5, (400, 80, 460, 160), track=8)
    assert second == [] and logic.suppressed == 1
    # after the cooldown a new violation is reported again
    third = episode(12.0, (700, 80, 760, 160), track=9)
    assert len(third) == 1


def test_association_gate_scales_with_frame_size():
    """Pixel motion is meaningless without the frame scale: a 60 px head on a 4K frame may
    legitimately travel 150 px between samples, which is 6x its own box."""
    det = _mk(WITHOUT_HELMET, 0.9, 0, 0.0, (2306, 1648, 2354, 1710))   # real 4K detection
    assert _gate_px(det, (1280, 720)) == pytest.approx(1.5 * 62)       # relative to the box
    assert _gate_px(det, (3840, 2160)) == pytest.approx(0.06 * 3840)   # ...but the frame dominates
    assert _gate_px(det, None) == pytest.approx(1.5 * 62)              # no frame info: box only


def test_fast_rider_stays_one_episode_on_a_large_frame():
    """Regression for the CAM-002 bug: without a frame-scaled gate every sample opened a new
    episode, hits never reached AI_HELMET_CONFIRMATION_FRAMES, and a 605-frame 4K video
    produced zero violations. The rider here moves 140 px per sample with no track id."""
    cfg = get_ai_config()
    logic = HelmetViolationLogic("CAM-002", cfg)
    bbox = (2306, 1648, 2354, 1710)
    events = []
    for k in range(cfg.confirmation_frames):
        moved = (bbox[0], bbox[1] + k * 140, bbox[2], bbox[3] + k * 140)
        events += logic.observe([_mk(WITHOUT_HELMET, 0.78, k, k * 0.5, moved, track_id=None)],
                                k, k * 0.5, frame_size=(3840, 2160))
    assert len(events) == 1
    assert events[0].confirmations == cfg.confirmation_frames
    assert events[0].camera_id == "CAM-002"
    # a rider 400 px away on the same frame is a DIFFERENT episode, not a hit on this one
    logic2 = HelmetViolationLogic("CAM-002", cfg)
    logic2.observe([_mk(WITHOUT_HELMET, 0.9, 0, 0.0, bbox, track_id=None)], 0, 0.0, frame_size=(3840, 2160))
    logic2.observe([_mk(WITHOUT_HELMET, 0.9, 1, 0.5, (bbox[0] + 400, bbox[1], bbox[2] + 400, bbox[3]), None)],
                   1, 0.5, frame_size=(3840, 2160))
    assert [s["hits"] for s in logic2.active_states()] == [1, 1]


def test_tracking_ids_keep_one_episode_across_frames():
    cfg = get_ai_config()
    logic = HelmetViolationLogic("CAM-001", cfg)
    # rider moves fast (boxes never overlap) but keeps the same track id
    for i in range(cfg.confirmation_frames):
        bbox = (100 + i * 120, 80, 160 + i * 120, 160)
        out = logic.observe([_mk(WITHOUT_HELMET, 0.9, i, i * 0.5, bbox, track_id=42)], i, i * 0.5)
    assert len(out) == 1
    assert logic.active_states()[0]["key"] == "trk:42"


# ---------------------------------------------------------------------------
# 6. evidence frames
# ---------------------------------------------------------------------------
def test_evidence_frame_is_created(ai_env):
    cfg = get_ai_config()
    frame = np.full((240, 320, 3), 30, np.uint8)
    detections = [_mk(WITHOUT_HELMET, 0.91, 142, 5.67)]
    violation = ConfirmedViolation(
        event_id="EVT-TESTEVID01", camera_id="CAM-001", detection=detections[0],
        confidence=0.91, frame_number=142, timestamp=5.67,
    )
    uri = save_evidence_frame(frame, detections, violation, cfg, model_name="unit-test")
    norm_uri = (uri or "").replace("\\", "/")
    assert norm_uri.startswith(cfg.evidence_dir.replace("\\", "/"))
    assert norm_uri.endswith("EVT-TESTEVID01.jpg")
    on_disk = Path(uri)
    assert on_disk.is_file() and on_disk.stat().st_size > 0
    img = cv2.imread(str(on_disk))
    assert img is not None and img.shape == frame.shape


# ---------------------------------------------------------------------------
# 7. end-to-end: event enters the EXISTING M3 pipeline and reaches Mock M4
# ---------------------------------------------------------------------------
def test_violation_flows_through_existing_m3_pipeline(ai_env, monkeypatch):
    from app.ai.video_processor import VideoProcessor
    from app.db.database import SessionLocal
    from app.db.repositories import EventRepository
    from app.services.publisher import get_mock_m4_consumer
    from app.ai.topology import ensure_ai_video_topology

    monkeypatch.setattr("app.ai.video_processor.get_detector", lambda cfg=None: FakeDetector())
    ensure_ai_video_topology()

    clip = str(ai_env / "clip.mp4")
    outcome = VideoProcessor().process("CAM-001", clip)

    metrics = outcome["metrics"]
    assert metrics["frames_decoded"] == 40
    assert metrics["inference_frames"] == 8                 # 4 s @ 2 fps: not every frame
    assert metrics["violations"] == 1                       # one episode -> one event
    assert metrics["processing_time"] > 0 and metrics["inference_fps_actual"] > 0

    ingest = outcome["ingest_results"][0]
    assert ingest.status == "ACCEPTED"
    canonical_id = ingest.canonical_event_id
    assert canonical_id.startswith("evt-")

    with SessionLocal() as db:
        record = EventRepository(db).get_by_event_id(canonical_id)
        assert record is not None
        # --- canonical envelope provenance ---
        assert record.event_type == "HELMET_VIOLATION"
        assert record.camera_id == "CAM-001"                      # camera preserved
        assert record.processing_node_id == "RDK-X5-001"          # processing node preserved
        assert record.provider == "KAAVAL_AI"
        assert record.adapter == "KAAVAL_EDGE"
        prov = record.ai_provenance
        assert prov["model_name"] == "yolov8n-helmet-detection"
        assert prov["model_version"] == "1.0.0"
        # --- attributes & evidence ---
        assert record.attributes["helmet_status"] == "WITHOUT_HELMET"
        assert record.attributes["violation"] == "HELMET_VIOLATION"
        assert record.evidence_ref["type"] == "local_file"
        assert record.evidence_ref["uri"].endswith(f"/{ingest.source_event_id}.jpg")
        assert Path(record.evidence_ref["uri"]).is_file()

    # --- Mock M4 received the canonical event ---
    m4_events = get_mock_m4_consumer().get_events(limit=50)
    assert any(e["event"]["event_id"] == canonical_id for e in m4_events)
    received = next(e["event"] for e in m4_events if e["event"]["event_id"] == canonical_id)
    assert received["source"]["provider"] == "KAAVAL_AI"
    assert received["processing"]["processing_node_id"] == "RDK-X5-001"
    assert received["processing"]["model_name"] == "yolov8n-helmet-detection"


def test_events_are_deduplicated_by_the_pipeline(ai_env, monkeypatch):
    """The same Kaaval evt_id pushed twice is suppressed by the existing dedup filter."""
    from app.ai.video_processor import build_kaaval_event_payload, ingest_to_m3
    from app.ai.schemas import ConfirmedViolation as CV

    det = _mk(WITHOUT_HELMET, 0.91, 142, 5.67)
    violation = CV(event_id="EVT-DEDUPTEST1", camera_id="CAM-001", detection=det,
                   confidence=0.91, frame_number=142, timestamp=5.67)
    payload = build_kaaval_event_payload(violation, get_ai_config(), 2.0, 11.5, None)
    first = ingest_to_m3(payload)
    second = ingest_to_m3(payload)
    assert first.status == "ACCEPTED"
    assert second.status == "DUPLICATE" and second.canonical_event_id == first.canonical_event_id


# ---------------------------------------------------------------------------
# 8. API endpoints (Swagger-documented)
# ---------------------------------------------------------------------------
def test_process_endpoint_starts_background_job(client, ai_env, monkeypatch):
    from app.ai.jobs import get_job_registry

    monkeypatch.setattr("app.ai.video_processor.get_detector", lambda cfg=None: FakeDetector())

    resp = client.post("/ai/helmet/process", json={"camera_id": "CAM-001"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["camera_id"] == "CAM-001" and body["status"] == "STARTED"
    job_id = body["job_id"]

    deadline = time.time() + 60
    data = {}
    while time.time() < deadline:
        data = client.get(f"/ai/helmet/jobs/{job_id}").json()
        if data["status"] in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.2)
    assert data["status"] == "COMPLETED", data.get("error")
    assert data["metrics"]["violations"] == 1
    assert data["violations"][0]["event_id"].startswith("EVT-")
    assert get_job_registry().get(job_id).status == JobStatus.COMPLETED


def test_process_endpoint_rejects_bad_requests(client):
    assert client.post("/ai/helmet/process", json={}).status_code == 400
    assert client.post("/ai/helmet/process", json={"camera_id": "CAM-404"}).status_code == 400
    resp = client.post("/ai/helmet/process", json={"camera_id": "CAM-001", "video_path": "missing/file.mp4"})
    assert resp.status_code == 400 and "not found" in resp.json()["detail"]
    assert client.get("/ai/helmet/jobs/job-does-not-exist").status_code == 404


def test_config_and_health_endpoints(client, ai_env, monkeypatch):
    cfg = client.get("/ai/helmet/config").json()
    assert cfg["model_name"] == "yolov8n-helmet-detection"
    assert cfg["model_card"]["source_url"].startswith("https://huggingface.co/")
    assert cfg["inference_fps"] == 2.0 and cfg["confirmation_frames"] == 3

    # model file deliberately missing (ai_env points AI_MODEL_PATH at a tmp path)
    health = client.get("/ai/helmet/health").json()
    assert health["yolo_model"]["status"] == "FAILED"
    assert health["kaaval_ai"] == "ONLINE" and health["event_pipeline"] == "ONLINE"
    # ...while the camera stays ONLINE: layers are independent
    cam001 = next(c for c in health["cameras"] if c["camera_id"] == "CAM-001")
    assert cam001["camera_status"] == "ONLINE"


def test_yolo_failure_never_marks_camera_offline_in_m3_health(client, ai_env):
    client.get("/ai/helmet/health")          # first load attempt records YOLO_MODEL = FAILED
    health = client.get("/health").json()
    by_id = {c["entity_id"]: c for c in health["components"]}
    assert by_id["CAM-001"]["status"] == "ONLINE"
    yolo = [c for c in health["components"] if c["entity_type"] == "YOLO_MODEL"]
    assert yolo and yolo[0]["status"] == "FAILED"      # model missing -> YOLO FAILED, camera ONLINE
    assert health["summary"]["yolo_models_ready"] == 0


def test_swagger_documents_the_ai_endpoints(client):
    from app.main import app

    paths = app.openapi()["paths"]
    assert "/ai/helmet/process" in paths
    assert "/ai/helmet/health" in paths
    assert "/ai/helmet/config" in paths
    assert "/ai/helmet/jobs/{job_id}" in paths
    assert "/ai/helmet/evidence" in paths


# ---------------------------------------------------------------------------
# 9. browser console served by M3 itself
# ---------------------------------------------------------------------------

def test_console_page_and_assets_are_served(client):
    page = client.get("/ui/ai")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "Content-Security-Policy" in page.headers          # the page ships no inline script
    assert "KAAVAL AI" in page.text

    js = client.get("/ui/ai/app.js")
    css = client.get("/ui/ai/styles.css")
    assert js.status_code == 200 and "javascript" in js.headers["content-type"]
    assert css.status_code == 200 and "css" in css.headers["content-type"]
    # the route is an allow-list, not a directory listing
    assert client.get("/ui/ai/nope.js").status_code == 404
    assert client.get("/ui/ai/relay/generated/mediamtx.yml").status_code == 404


def test_evidence_endpoint_serves_the_frame_but_nothing_else(client, ai_env, tmp_path):
    """The console shows evidence frames through an endpoint that cannot read arbitrary files."""
    cfg = get_ai_config()
    det = _mk(WITHOUT_HELMET, 0.9, 3, 1.5, (10, 20, 60, 80))
    violation = ConfirmedViolation(event_id="EVT-EVIDENCE01", camera_id="CAM-001", detection=det,
                                   confidence=0.9, frame_number=3, timestamp=1.5)
    uri = save_evidence_frame(np.full((120, 160, 3), 25, np.uint8), [det], violation, cfg,
                              model_name="unit-test")
    assert uri, "the real evidence writer should have produced a frame"

    served = client.get("/ai/helmet/evidence", params={"uri": uri})
    assert served.status_code == 200
    assert served.headers["content-type"].startswith("image/jpeg")
    assert served.content[:2] == b"\xff\xd8"                  # JPEG magic bytes

    # a real image OUTSIDE the evidence directory is still refused
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"\xff\xd8nope")
    assert client.get("/ai/helmet/evidence", params={"uri": str(outside)}).status_code == 404

    # traversal, a non-image inside the evidence dir, a missing frame, and a missing param
    assert client.get("/ai/helmet/evidence", params={"uri": "../sentinel_m3.db"}).status_code == 404
    assert client.get("/ai/helmet/evidence",
                      params={"uri": str(Path(cfg.evidence_dir) / "notes.txt")}).status_code == 404
    assert client.get("/ai/helmet/evidence", params={"uri": "evidence/1970/01/01/EVT-NOPE.jpg"}).status_code == 404
    assert client.get("/ai/helmet/evidence").status_code == 422
