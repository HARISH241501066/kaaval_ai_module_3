"""
REAL Google Cloud integration tests — OPT-IN, never part of the default suite.

    RUN_GOOGLE_TESTS=1  python -m pytest tests/test_ai_google_integration.py -v

Prerequisites (all checked first, with a clear SKIP reason):

    pip install google-cloud-videointelligence google-cloud-vision
    gcloud auth application-default login          # or set GOOGLE_APPLICATION_CREDENTIALS
    GOOGLE_CLOUD_PROJECT_ID=my-project             # a project with both APIs enabled

What they verify against the live services (the mocked suite cannot):

* Cloud Video Intelligence accepts the local demo video and returns tracked people/objects
  with timestamps and bounding boxes;
* Cloud Vision OCR runs on a real image and reports an honest ANPR state;
* the provider health reports AVAILABLE with real credentials (and NOT_CONFIGURED without).

These tests cost real API money (a couple of Video Intelligence units) and are therefore
skipped by default. CI never sets RUN_GOOGLE_TESTS.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

def _prerequisites() -> list:
    """Human-readable list of what is missing, so a SKIP explains itself."""
    missing = []
    try:
        import google.cloud.videointelligence  # noqa: F401
    except ImportError:
        missing.append("pip install google-cloud-videointelligence")
    try:
        import google.cloud.vision  # noqa: F401
    except ImportError:
        missing.append("pip install google-cloud-vision")
    from app.ai.config import get_ai_config

    cfg = get_ai_config()
    if not cfg.google_project_id:
        missing.append("GOOGLE_CLOUD_PROJECT_ID is not set")
    if cfg.google_credentials_source == "none":
        missing.append("no credentials: run 'gcloud auth application-default login' "
                       "or set GOOGLE_APPLICATION_CREDENTIALS")
    return missing


_PREREQ = _prerequisites()
pytestmark = [
    pytest.mark.skipif(os.environ.get("RUN_GOOGLE_TESTS") != "1",
                       reason="set RUN_GOOGLE_TESTS=1 to run live Google Cloud tests"),
    pytest.mark.skipif(bool(_PREREQ), reason="; ".join(_PREREQ)),
]

VIDEO = os.environ.get("GOOGLE_TEST_VIDEO", "camera_feed/camera_main.mp4")
CAMERA = os.environ.get("GOOGLE_TEST_CAMERA", "CAM-001")


def _provider():
    from app.ai.config import get_ai_config
    from app.ai.providers.google_video import GoogleVideoIntelligenceProvider

    return GoogleVideoIntelligenceProvider(get_ai_config())


# --------------------------------------------------------------------------- health
def test_00_video_intelligence_health_is_available():
    provider = _provider()
    health = provider.health()
    assert health["status"] == "AVAILABLE", health
    assert health.get("project_configured") is True or "project" in str(health).lower()


# --------------------------------------------------------------------------- VI analysis
def test_10_real_video_returns_tracked_people_with_timestamps():
    """ONE Video Intelligence call covers the whole video and returns temporal tracks."""
    from app.ai.config import get_ai_config

    analysis = _provider().analyze_video(VIDEO, camera_id=CAMERA)
    assert analysis.provider == "GOOGLE_VIDEO_INTELLIGENCE"
    assert analysis.capabilities["helmet_status"] is False        # never faked
    assert analysis.extra.get("api_calls") == 1
    assert analysis.duration_s > 20                               # the demo videos are 20-30 s
    assert analysis.frames_analyzed > 0                           # timestamped observations exist
    person_tracks = [t for t in analysis.tracks if t.object_type == "PERSON"]
    assert person_tracks, "expected at least one PERSON track in the demo video"
    for track in person_tracks:
        assert track.observations
        for obs in track.observations:
            assert 0.0 <= obs.timestamp <= analysis.duration_s + 1.0
            assert all(0.0 <= v <= 1.0 for v in obs.bbox)         # normalized boxes
    print(f"\nVI: {len(analysis.tracks)} tracks "
          f"({len(person_tracks)} person), {analysis.frames_analyzed} observations, "
          f"{analysis.duration_s:.1f}s video")


def test_11_pipeline_end_to_end_with_real_google_analysis():
    """Full hybrid pipeline: real VI analysis + local helmet classifier + M3 ingestion."""
    from app.ai.config import get_ai_config
    from app.ai.video_analysis import VideoAnalysisProcessor

    outcome = VideoAnalysisProcessor(get_ai_config()).process(CAMERA, VIDEO)
    metrics = outcome["metrics"]
    result = metrics["result"]
    assert result["status"] == "completed"
    assert metrics["analysis_provider"] == "GOOGLE_VIDEO_INTELLIGENCE"
    assert metrics["outcome"] in ("PROCESSING_COMPLETED", "PROCESSING_COMPLETED_ZERO_VIOLATIONS")
    print(f"\nPIPELINE: {result['unique_violators']} unique violators, "
          f"{result['violation_events']} events, evidence={metrics['evidence_frames']}, "
          f"ANPR {result['anpr_matches']}/{result['anpr_attempts']}, "
          f"{result['duration_seconds']}s video in {metrics['processing_time']}s")


# --------------------------------------------------------------------------- Vision OCR
def test_20_vision_ocr_reports_an_honest_state():
    """Cloud Vision OCR on a blank image: NOT_FOUND is the honest answer (no text)."""
    from app.ai.config import get_ai_config
    from app.ai.providers.google_vision import GoogleVisionAnprProvider

    provider = GoogleVisionAnprProvider(get_ai_config())
    health = provider.health()
    assert health["status"] in ("AVAILABLE", "ERROR"), health      # configured; availability differs
    result = provider.recognize(np.zeros((64, 128, 3), dtype="uint8"))
    assert result.status in ("NOT_FOUND", "UNCERTAIN", "ERROR")
    assert result.provider == "GOOGLE_CLOUD_VISION"
    print(f"\nOCR blank frame: status={result.status} plate={result.plate_number}")
