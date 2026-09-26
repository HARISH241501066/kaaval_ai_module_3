"""
VideoAnalysisProcessor tests — the whole Google-path pipeline with NO live Google API.

Covered here (the provider doubles live in tests/test_ai_google.py):

* tracked riders -> second-stage helmet classification -> UNIQUE violators (not frames)
* confirmation rule (too-few-observation tracks are never classified)
* classification budget (google_max_classifications)
* evidence frame per unique violator, rendered from the real video file
* ANPR: one OCR call per violator, capped by ANPR_MAX_PER_JOB, skipped when disabled
* existing M3 ingestion: payload provenance + ACCEPTED/REJECTED counting
* PROCESSING_COMPLETED vs PROCESSING_COMPLETED_ZERO_VIOLATIONS vs a FAILED job

The analysis provider, helmet classifier, ANPR provider and the M3 ingestion call are all
injected, so the only real IO is the tiny synthetic clip (same helper as test_ai_helmet.py).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from app.ai.config import AIConfig, get_ai_config
from app.ai.kaaval_logic import HelmetViolationLogic  # noqa: F401  (re-exported contract)
from app.ai.providers import ProviderSelection
from app.ai.providers.base import (
    ANPR_FOUND,
    AnprResult,
    ProviderError,
    TrackObservation,
    TrackedObject,
    VideoAnalysis,
)
from app.ai.schemas import WITH_HELMET, WITHOUT_HELMET, Detection
from app.core.config import get_settings


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def ai_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("AI_MODEL_PATH", str(tmp_path / "missing-model.pt"))
    get_settings.cache_clear()
    (tmp_path / "clip.mp4")
    _write_clip(tmp_path / "clip.mp4")
    yield tmp_path
    get_settings.cache_clear()


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


def _cfg(**overrides) -> AIConfig:
    return dataclasses.replace(get_ai_config(), **overrides)


# --------------------------------------------------------------------------- doubles
class FakeAnalysisProvider:
    """Returns a canned VideoAnalysis; records the call like the real provider would."""

    name = "GOOGLE_VIDEO_INTELLIGENCE"

    def __init__(self, analysis=None, capabilities=None, error=None):
        self.analysis = analysis
        self.capabilities = capabilities or {
            "person_detection": True, "object_tracking": True,
            "helmet_status": False, "unique_identity": True,
        }
        self.error = error
        self.calls: List[Dict[str, Any]] = []

    def capabilities_method(self):  # noqa: D401 (unused by the processor)
        return self.capabilities

    def health(self):
        return {"status": "AVAILABLE", "provider": self.name, "error": None}

    def analyze_video(self, video_path, camera_id="CAM-001", on_progress=None,
                      max_observations_per_track=None):
        self.calls.append({"video_path": video_path, "camera_id": camera_id,
                           "max_observations": max_observations_per_track})
        if self.error is not None:
            raise self.error
        return self.analysis


def _analysis(tracks, provider="GOOGLE_VIDEO_INTELLIGENCE", helmet_status=False,
              fps=10.0, total_frames=40, duration=4.0):
    return VideoAnalysis(
        provider=provider,
        model_name="google-cloud-video-intelligence" if provider.startswith("GOOGLE") else "yolov8n-helmet",
        model_version="v1" if provider.startswith("GOOGLE") else "1.0.0",
        capabilities={"person_detection": True, "object_tracking": True, "helmet_status": helmet_status},
        video_path="clip.mp4",
        duration_s=duration,
        video_fps=fps,
        total_frames=total_frames,
        frames_analyzed=sum(len(t.observations) for t in tracks),
        tracks=tracks,
        extra={"api_calls": 1},
    )


def _rider(key: str, timestamps, box=(0.1, 0.2, 0.3, 0.6), normalized=True,
           helmet_status=None, helmet_confidence=0.0):
    """One tracked rider seen at several video times (the unit of uniqueness)."""
    return TrackedObject(
        key=key, object_type="PERSON", label="RIDER", confidence=0.9,
        observations=[
            TrackObservation(timestamp=t, bbox=box, confidence=0.9,
                             frame_number=(5 + 5 * i) if not normalized else -1,
                             bbox_normalized=normalized,
                             helmet_status=helmet_status, helmet_confidence=helmet_confidence)
            for i, t in enumerate(timestamps)
        ],
    )


class StubClassifier:
    """Second stage that answers from a plan keyed by track id (defaults to WITHOUT_HELMET)."""

    name = "stub-helmet"
    model_name = "stub-helmet"
    model_version = "0.0"

    def __init__(self, default=(WITHOUT_HELMET, 0.90), plan: Optional[Dict] = None):
        self.default = default
        self.plan = plan or {}
        self.calls: List[Tuple] = []

    def capabilities(self):
        return {"helmet_status": True, "classes": [WITH_HELMET, WITHOUT_HELMET]}

    def health(self):
        return {"status": "READY", "classifier": self.name}

    def classify(self, frame, bbox=None, frame_number=0, timestamp=0.0, track_id=None):
        self.calls.append((track_id, frame_number, round(timestamp, 3),
                           tuple(int(v) for v in bbox) if bbox is not None else None))
        cls, conf = self.plan.get(track_id, self.default)
        x1, y1, x2, y2 = bbox if bbox is not None else (10, 10, 60, 60)
        return [Detection(class_name=cls, confidence=conf, bbox=(int(x1), int(y1), int(x2), int(y2)),
                          frame_number=frame_number, timestamp=timestamp, track_id=track_id)]


class FakeAnpr:
    """ONE OCR call per confirmed violator — records the image path and the ROI."""

    name = "GOOGLE_CLOUD_VISION"

    def __init__(self, plate="GJ01AB1234", status=ANPR_FOUND):
        self.plate, self.status = plate, status
        self.calls: List[Tuple[Any, Optional[Dict]]] = []

    def capabilities(self):
        return {"ocr": True, "plate_detector": False}

    def health(self):
        return {"status": "AVAILABLE", "provider": self.name}

    def recognize(self, image, context=None):
        self.calls.append((image, context))
        return AnprResult(status=self.status, plate_number=self.plate,
                          raw_text="GJ 01 AB 1234" if self.plate else None,
                          provider=self.name)


class IngestRecorder:
    """Stands in for ingest_to_m3: records payloads, answers like the M3 pipeline."""

    def __init__(self, status: str = "ACCEPTED"):
        self.status = status
        self.payloads: List[Dict[str, Any]] = []

    def __call__(self, payload: Dict[str, Any]) -> Any:
        self.payloads.append(payload)
        return SimpleNamespace(status=self.status, canonical_event_id=f"CE-{len(self.payloads):04d}")


_UNSET = object()


def _run(ai_env, provider, classifier=_UNSET, anpr=_UNSET, ingest=_UNSET, config=None, selection=None):
    from app.ai.video_analysis import VideoAnalysisProcessor

    return VideoAnalysisProcessor(
        config=config or _cfg(provider="yolo"),
        provider=provider,
        classifier=StubClassifier() if classifier is _UNSET else classifier,
        anpr=FakeAnpr() if anpr is _UNSET else anpr,
        ingest_fn=IngestRecorder() if ingest is _UNSET else ingest,
        selection=selection,
    ).process("CAM-001", str(ai_env / "clip.mp4"))


# --------------------------------------------------------------------------- 1. hybrid path
def test_tracked_riders_become_unique_violators_with_evidence_anpr_and_m3_events(ai_env):
    """3 tracked riders -> 2 candidates -> 8 classifications -> 2 UNIQUE violators -> 2 events."""
    analysis = _analysis([
        _rider("person:0", [0.5, 1.0, 1.5, 2.0, 2.5]),            # 5 obs -> violator
        _rider("person:1", [0.6, 1.1, 1.6], box=(0.6, 0.2, 0.8, 0.6)),  # 3 obs -> violator
        _rider("person:2", [0.7, 1.2]),                            # 2 obs -> unconfirmable
    ])
    provider = FakeAnalysisProvider(analysis)
    classifier = StubClassifier()
    anpr = FakeAnpr()
    ingest = IngestRecorder()

    outcome = _run(ai_env, provider, classifier=classifier, anpr=anpr, ingest=ingest)
    metrics = outcome["metrics"]

    # --- unique violators, not frame detections
    assert metrics["result"]["unique_violators"] == 2
    assert metrics["result"]["violation_events"] == 2
    assert metrics["unique_violators"] == 2 and metrics["violation_detections"] == 8
    assert metrics["outcome"] == "PROCESSING_COMPLETED"
    assert metrics["tracks_seen"] == 3
    assert metrics["rider_tracks"] == 2                     # the 2-observation track was dropped
    assert metrics["helmet_classifications"] == 8           # 5 + 3, bounded per track
    assert metrics["tracks_with_provider_helmet_status"] == 0
    assert metrics["evidence_frames"] == 2
    assert metrics["anpr_attempts"] == 2 and metrics["anpr_matches"] == 2
    assert metrics["events_ingested"] == 2
    assert len(ingest.payloads) == 2

    # --- the provider analyzed the WHOLE video in one call
    assert provider.calls[0]["video_path"].endswith("clip.mp4")
    assert metrics["analysis_provider"] == "GOOGLE_VIDEO_INTELLIGENCE"
    assert metrics["analysis_model"] == "google-cloud-video-intelligence v1"

    # --- classification actually used the rider's pixel box (normalized -> 128px frame)
    track_one = [c for c in classifier.calls if c[0] == 1]
    assert len(track_one) == 5
    assert track_one[0][3] == (13, 26, 38, 77)              # (0.1, 0.2, 0.3, 0.6) * 128

    # --- payload provenance: honest about who did what
    by_track = {p["data"].get("track_id"): p for p in ingest.payloads}
    assert set(by_track) == {1, 2}
    payload = by_track[1]
    assert payload["type"] == "HELMET_VIOLATION"
    assert payload["data"]["unique_violator"] is True
    assert payload["data"]["violation_detections"] == 5
    assert payload["data"]["analytics_provider"] == "GOOGLE_VIDEO_INTELLIGENCE"
    assert payload["data"]["helmet_classifier"] == "stub-helmet"
    assert payload["data"]["plate_status"] == "FOUND"
    assert payload["data"]["plate"] == "GJ01AB1234"
    assert payload["analytics_provider"] == "GOOGLE_VIDEO_INTELLIGENCE"
    assert payload["helmet_classifier"] == "stub-helmet"
    # evidence file really exists on disk, referenced by the canonical data block
    uri = payload["data"]["evidence_ref"]["uri"]
    assert Path(uri).is_file() and Path(uri).suffix == ".jpg"
    assert payload["data"]["bbox"] == [13, 26, 38, 77]

    # --- ANPR ran on the evidence frame with the rider box as ROI
    assert len(anpr.calls) == 2
    image_arg, context = anpr.calls[0]
    assert str(image_arg).endswith(".jpg") and Path(image_arg).is_file()
    assert context["roi"] == (13, 26, 38, 77)


def test_local_provider_helmet_status_needs_no_second_stage(ai_env):
    """helmet_status=True providers already carry the verdict: zero classifier calls."""
    tracks = [
        _rider("trk:7", [0.5, 1.0, 1.5, 2.0], helmet_status=WITHOUT_HELMET, helmet_confidence=0.91),
        _rider("trk:9", [0.6, 1.1, 1.6], box=(0.6, 0.2, 0.8, 0.6),
               helmet_status=WITHOUT_HELMET, helmet_confidence=0.88),
    ]
    analysis = _analysis(tracks, provider="LOCAL_YOLO", helmet_status=True)
    classifier = StubClassifier()
    ingest = IngestRecorder()

    metrics = _run(ai_env, FakeAnalysisProvider(analysis), classifier=classifier,
                   anpr=FakeAnpr(), ingest=ingest)["metrics"]

    assert classifier.calls == []                           # never re-classified locally
    assert metrics["tracks_with_provider_helmet_status"] == 2
    assert metrics["helmet_classifications"] == 0
    assert metrics["unique_violators"] == 2
    assert metrics["analysis_provider"] == "LOCAL_YOLO"
    assert len(ingest.payloads) == 2


def test_unconfirmable_tracks_are_never_classified(ai_env):
    """A track with fewer observations than the confirmation rule cannot confirm — skip it."""
    analysis = _analysis([_rider("person:0", [0.5, 1.0])])       # 2 obs < 3 confirmations
    classifier = StubClassifier()

    metrics = _run(ai_env, FakeAnalysisProvider(analysis), classifier=classifier)["metrics"]
    assert metrics["rider_tracks"] == 0
    assert metrics["helmet_classifications"] == 0
    assert metrics["unique_violators"] == 0
    assert classifier.calls == []


def test_classification_budget_bounds_cloud_path_work(ai_env):
    """google_max_classifications stops the second stage; later riders are reported, not faked."""
    analysis = _analysis([
        _rider("person:0", [0.5, 1.0, 1.5, 2.0, 2.5]),
        _rider("person:1", [0.6, 1.1, 1.6], box=(0.6, 0.2, 0.8, 0.6)),
    ])
    classifier = StubClassifier()
    cfg = _cfg(provider="yolo", google_max_classifications=4)

    metrics = _run(ai_env, FakeAnalysisProvider(analysis), classifier=classifier, config=cfg)["metrics"]
    assert metrics["helmet_classifications"] == 4           # budget consumed by the first rider
    assert metrics["unique_violators"] == 1                 # rider 2 never classified


def test_zero_helmet_findings_are_reported_as_zero_violations(ai_env):
    """Every rider WITH_HELMET -> completed with zero violations (never FAILED, never faked)."""
    analysis = _analysis([_rider("person:0", [0.5, 1.0, 1.5, 2.0])])
    classifier = StubClassifier(default=(WITH_HELMET, 0.95))
    anpr = FakeAnpr()
    ingest = IngestRecorder()

    metrics = _run(ai_env, FakeAnalysisProvider(analysis), classifier=classifier,
                   anpr=anpr, ingest=ingest)["metrics"]
    assert metrics["outcome"] == "PROCESSING_COMPLETED_ZERO_VIOLATIONS"
    assert metrics["unique_violators"] == 0
    assert metrics["evidence_frames"] == 0
    assert metrics["anpr_attempts"] == 0
    assert ingest.payloads == []


# --------------------------------------------------------------------------- 2. ANPR bounding
def test_anpr_is_capped_per_job(ai_env):
    """ANPR_MAX_PER_JOB bounds OCR cost: 2 violators + cap 1 -> exactly one OCR call."""
    analysis = _analysis([
        _rider("person:0", [0.5, 1.0, 1.5, 2.0]),
        _rider("person:1", [0.6, 1.1, 1.6], box=(0.6, 0.2, 0.8, 0.6)),
    ])
    anpr = FakeAnpr()
    cfg = _cfg(provider="yolo", anpr_max_per_job=1)

    metrics = _run(ai_env, FakeAnalysisProvider(analysis), anpr=anpr, config=cfg)["metrics"]
    assert metrics["result"]["unique_violators"] == 2
    assert metrics["anpr_attempts"] == 1 and metrics["anpr_matches"] == 1
    assert len(metrics["anpr_results"]) == 1


def test_anpr_disabled_leaves_the_payload_honestly_plate_free(ai_env):
    analysis = _analysis([_rider("person:0", [0.5, 1.0, 1.5])])
    ingest = IngestRecorder()
    cfg = _cfg(provider="yolo", anpr_enabled=False)

    # anpr=None -> the processor resolves it from the config -> NoAnprProvider
    metrics = _run(ai_env, FakeAnalysisProvider(analysis), anpr=None,
                   ingest=ingest, config=cfg)["metrics"]
    assert metrics["anpr_attempts"] == 0
    assert metrics["anpr_provider"] != "GOOGLE_CLOUD_VISION"     # NoAnprProvider reported
    payload = ingest.payloads[0]
    assert "plate" not in payload["data"]
    assert "plate_status" not in payload["data"]


# --------------------------------------------------------------------------- 3. M3 results
def test_rejected_ingestion_is_counted_not_hidden(ai_env):
    analysis = _analysis([_rider("person:0", [0.5, 1.0, 1.5])])
    ingest = IngestRecorder(status="REJECTED")

    metrics = _run(ai_env, FakeAnalysisProvider(analysis), ingest=ingest)["metrics"]
    assert metrics["unique_violators"] == 2 - 1 == 1        # riders confirmed regardless of M3
    assert metrics["violation_events"] == 1
    assert metrics["events_ingested"] == 0                  # ...but M3 said no
    assert metrics["outcome"] == "PROCESSING_COMPLETED"


def test_failed_ingestion_raises_and_never_reports_zero_violations(ai_env):
    analysis = _analysis([_rider("person:0", [0.5, 1.0, 1.5])])

    def boom(payload):
        raise RuntimeError("sqlite is on fire")

    with pytest.raises(RuntimeError, match="sqlite is on fire"):
        _run(ai_env, FakeAnalysisProvider(analysis), ingest=boom)


def test_provider_failure_raises_and_never_reports_zero_violations(ai_env):
    provider = FakeAnalysisProvider(error=ProviderError("Cloud Video Intelligence quota exceeded"))
    ingest = IngestRecorder()

    with pytest.raises(ProviderError, match="quota"):
        _run(ai_env, provider, ingest=ingest)
    assert ingest.payloads == []


# --------------------------------------------------------------------------- 4. reporting
def test_fallback_selection_is_reported_in_the_metrics(ai_env):
    """requested=google / effective=local_yolo must be visible in the result, never silent."""
    analysis = _analysis([_rider("person:0", [0.5, 1.0, 1.5])])
    selection = ProviderSelection(
        requested="google", effective="local_yolo", provider=FakeAnalysisProvider(analysis),
        fallback_reason="NOT_CONFIGURED: no credentials",
    )
    metrics = _run(ai_env, FakeAnalysisProvider(analysis), selection=selection)["metrics"]
    assert metrics["analysis_requested"] == "google"
    assert metrics["analysis_effective"] == "local_yolo"
    assert metrics["analysis_fallback"] is True
    assert "NOT_CONFIGURED" in metrics["analysis_fallback_reason"]


def test_job_runner_marks_a_failed_video_job_not_zero_violations(ai_env, monkeypatch):
    """The /ai/video/job path: an exception becomes FAILED with PROCESSING_FAILED metrics."""
    from app.ai import video_analysis
    from app.ai.jobs import JobRegistry
    from app.ai.schemas import JobStatus

    class ExplodingProcessor:
        def __init__(self, *a, **k):
            pass

        def process(self, *a, **k):
            raise ProviderError("Google Video Intelligence unavailable")

    monkeypatch.setattr(video_analysis, "VideoAnalysisProcessor", ExplodingProcessor)
    # keep the runner off the database: topology is irrelevant to this failure mode
    from app.ai import topology as ai_topology
    monkeypatch.setattr(ai_topology, "ensure_ai_video_topology", lambda *a, **k: None)
    from app.ai import jobs as ai_jobs
    monkeypatch.setattr(ai_jobs, "get_ai_config", lambda: _cfg(provider="yolo"))
    job = JobRegistry().create("CAM-001", str(ai_env / "clip.mp4"), mode="video")
    JobRegistry().run_job(job)                              # synchronous, like tests do

    assert job.status == JobStatus.FAILED
    assert "Google Video Intelligence unavailable" in job.error
    assert job.metrics["outcome"] == "PROCESSING_FAILED"    # never "0 violations"
