"""
Google Cloud analysis layer — provider tests with NO live Google API calls.

The SDKs are not required: both Google classes accept an injected client, so these tests drive
them with duck-typed doubles that mirror the documented protobuf field names
(`person_detection_annotations[].tracks[]`, `object_annotations[].frames[]`, ...).

A manual test that calls the REAL APIs lives in tests/test_ai_google_integration.py and is
skipped unless RUN_GOOGLE_TESTS=1 and credentials are configured.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from app.ai.config import AIConfig, get_ai_config
from app.ai.providers import (
    get_anpr_provider,
    get_helmet_classifier,
    provider_status,
    select_video_provider,
)
from app.ai.providers.base import (
    ANPR_FOUND,
    ANPR_NOT_CONFIGURED,
    ANPR_NOT_FOUND,
    ANPR_UNCERTAIN,
    ProviderNotConfigured,
    extract_plate_candidate,
    normalize_plate,
)
from app.ai.providers.google_video import GoogleVideoIntelligenceProvider, _duration_seconds
from app.ai.providers.google_vision import GoogleVisionAnprProvider, NoAnprProvider
from app.ai.providers.local_yolo import LocalYOLOClassifier, LocalYOLOProvider
from app.core.config import get_settings


# --------------------------------------------------------------------------- doubles
class FakeDuration:
    def __init__(self, seconds: float):
        self.seconds = int(seconds)
        self.nanos = int((seconds - int(seconds)) * 1e9)


class FakeBox:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom


class FakeStamped:
    def __init__(self, t, box, attributes=()):
        self.time_offset = FakeDuration(t)
        self.normalized_bounding_box = box
        self.attributes = list(attributes)


class FakeSegment:
    def __init__(self, start, end):
        self.start_time_offset = FakeDuration(start)
        self.end_time_offset = FakeDuration(end)


class FakePersonTrack:
    def __init__(self, stamps, confidence=0.9, start=0.0, end=1.0):
        self.timestamped_objects = list(stamps)
        self.confidence = confidence
        self.segment = FakeSegment(start, end)


class FakePersonAnnotation:
    def __init__(self, tracks, version="builtin/stable"):
        self.tracks = list(tracks)
        self.version = version


class FakeEntity:
    def __init__(self, description, entity_id="/m/0"):
        self.description = description
        self.entity_id = entity_id


class FakeObjectFrame:
    def __init__(self, t, box):
        self.time_offset = FakeDuration(t)
        self.normalized_bounding_box = box


class FakeObjectAnnotation:
    def __init__(self, entity, frames, confidence=0.8, start=0.0, end=1.0, version="builtin/stable"):
        self.entity = entity
        self.frames = list(frames)
        self.confidence = confidence
        self.segment = FakeSegment(start, end)
        self.version = version


class FakeResult:
    def __init__(self, persons=(), objects=(), error=None):
        self.person_detection_annotations = list(persons)
        self.object_annotations = list(objects)
        self.error = error


class FakeOperation:
    def __init__(self, results, metadata=None):
        self._results = results
        self.metadata = metadata

    def done(self) -> bool:
        return True

    def result(self):
        class _Response:
            annotation_results = self._results

        return _Response()


class FakeVideoClient:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    def annotate_video(self, request=None, **kwargs):
        self.requests.append(request or kwargs)
        return FakeOperation(self.results)


class FakeText:
    def __init__(self, description, vertices=()):
        self.description = description

        class _Poly:
            pass

        poly = _Poly()
        poly.vertices = [type("V", (), {"x": x, "y": y}) for x, y in vertices]
        self.bounding_poly = poly


class FakeVisionResponse:
    def __init__(self, texts, error=None):
        self.text_annotations = [FakeText(t, v) for t, v in texts]
        self.error = error


class FakeVisionClient:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def text_detection(self, image=None, **kwargs):
        self.calls += 1
        return self.response


class StubDetector:
    """Stands in for the local helmet model when only the plumbing is under test."""

    def __init__(self, classes=("WITHOUT_HELMET",), confidence=0.91):
        self.classes = classes
        self.confidence = confidence

    model_name = "stub-helmet"
    model_version = "0.0"

    def detect_frame(self, frame, frame_number=0, timestamp=0.0):
        from app.ai.schemas import Detection

        return [Detection(c, self.confidence, (10, 10, 60, 60), frame_number, timestamp) for c in self.classes]

    def health_check(self):
        return {"status": "READY", "model_name": self.model_name}


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def ai_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("AI_MODEL_PATH", str(tmp_path / "missing-model.pt"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def _cfg(**overrides) -> AIConfig:
    base = get_ai_config()
    return dataclasses.replace(base, **overrides)


# --------------------------------------------------------------------------- 1. configuration
def test_provider_configuration_and_aliases(ai_env):
    cfg = _cfg(provider="google", helmet_provider="yolo", anpr_provider="google_vision")
    assert cfg.video_provider == "google"
    assert cfg.helmet_provider == "yolo"
    assert cfg.anpr_provider == "google_vision"
    # Google-era env names are the same knobs as the Kaaval-era ones
    assert cfg.helmet_confidence_threshold == cfg.helmet_min_confidence == 0.60
    assert cfg.confirmation_observations == cfg.confirmation_frames == 3
    assert cfg.google_features == ["PERSON_DETECTION", "OBJECT_TRACKING"]
    assert cfg.google_configured is False           # no credentials in this environment


def test_env_aliases_are_accepted(ai_env, monkeypatch):
    monkeypatch.setenv("HELMET_CONFIDENCE_THRESHOLD", "0.72")
    monkeypatch.setenv("HELMET_CONFIRMATION_OBSERVATIONS", "5")
    get_settings.cache_clear()
    try:
        cfg = get_ai_config()
        assert cfg.helmet_min_confidence == pytest.approx(0.72)
        assert cfg.confirmation_frames == 5
    finally:
        monkeypatch.delenv("HELMET_CONFIDENCE_THRESHOLD", raising=False)
        monkeypatch.delenv("HELMET_CONFIRMATION_OBSERVATIONS", raising=False)
        get_settings.cache_clear()


def test_public_config_never_leaks_credentials(ai_env):
    cfg = _cfg(google_project_id="sentinel-demo", google_credentials="/secret/path/sa.json")
    public = cfg.as_public_dict()
    google = public["google"]
    assert google["project_id_configured"] is True
    assert google["credentials_source"] == "service_account_file"
    assert google["credentials_file_present"] is False        # path exists? -> boolean only
    flat = str(public)
    assert "/secret/path" not in flat and "sa.json" not in flat
    assert google["configured"] is True
    assert public["provider"] == cfg.video_provider


# --------------------------------------------------------------------------- 2. credentials
def test_missing_credentials_reported_with_actionable_error(ai_env):
    provider = GoogleVideoIntelligenceProvider(_cfg(google_project_id="", google_credentials=""))
    health = provider.health()
    assert health["status"] in ("NOT_CONFIGURED", "ERROR")
    assert health["error"]
    if health["status"] == "NOT_CONFIGURED":
        assert "GOOGLE_CLOUD_PROJECT_ID" in health["error"] or "credentials" in health["error"].lower()


def test_selecting_google_without_configuration_falls_back_and_says_so(ai_env):
    cfg = _cfg(provider="google", allow_local_fallback=True,
               google_project_id="", google_credentials="")
    selection = select_video_provider(cfg, detector=StubDetector())
    assert selection.requested == "google"
    assert selection.effective == "local_yolo"
    assert selection.fallback_reason                      # never silent
    assert isinstance(selection.provider, LocalYOLOProvider)
    assert selection.to_dict()["fallback"] is True


def test_no_fallback_means_explicit_failure(ai_env):
    cfg = _cfg(provider="google", allow_local_fallback=False,
               google_project_id="", google_credentials="")
    with pytest.raises(ProviderNotConfigured):
        select_video_provider(cfg, detector=StubDetector())


def test_provider_yolo_is_used_verbatim(ai_env):
    cfg = _cfg(provider="yolo")
    selection = select_video_provider(cfg, detector=StubDetector())
    assert selection.requested == "yolo" and selection.effective == "yolo"
    assert selection.fallback_reason is None


# --------------------------------------------------------------------------- 3. health
def test_health_of_every_layer_never_raises(ai_env):
    status = provider_status(_cfg(google_project_id="", google_credentials=""), detector=StubDetector())
    assert status["analysis"]["requested"] == "google"
    assert status["analysis"]["effective"] in ("local_yolo(fallback)", "google")
    for key in ("video_intelligence", "local_yolo_provider", "helmet_classifier", "anpr"):
        assert key in status and "status" in status[key]
    assert status["local_yolo_provider"]["status"] == "READY"     # injected stub detector
    assert status["helmet_classifier"]["status"] == "READY"
    assert status["video_intelligence"]["sdk_installed"] in (True, False)
    assert "deprecation_note" in status["video_intelligence"]


def test_helmet_provider_google_is_refused_with_a_reason(ai_env):
    with pytest.raises(ProviderNotConfigured) as ei:
        get_helmet_classifier(_cfg(helmet_provider="google"))
    assert "does NOT classify helmet status" in str(ei.value)


def test_classifier_and_anpr_selection(ai_env):
    assert isinstance(get_helmet_classifier(_cfg(), detector=StubDetector()), LocalYOLOClassifier)
    assert isinstance(get_anpr_provider(_cfg()), GoogleVisionAnprProvider)
    assert isinstance(get_anpr_provider(_cfg(anpr_enabled=False)), NoAnprProvider)
    assert isinstance(get_anpr_provider(_cfg(anpr_provider="none")), NoAnprProvider)
    assert isinstance(get_anpr_provider(_cfg(anpr_provider="something_else")), NoAnprProvider)


# --------------------------------------------------------------------------- 4. normalization
def test_google_response_normalization_keeps_identity_and_timestamps(ai_env):
    person = FakePersonAnnotation([
        FakePersonTrack([FakeStamped(0.5, FakeBox(0.1, 0.2, 0.3, 0.6)),
                         FakeStamped(1.5, FakeBox(0.12, 0.22, 0.32, 0.62)),], confidence=0.88),
        FakePersonTrack([FakeStamped(0.4, FakeBox(0.6, 0.1, 0.8, 0.5))], confidence=0.77),
    ])
    bike = FakeObjectAnnotation(FakeEntity("Motorcycle"), [FakeObjectFrame(0.5, FakeBox(0.1, 0.4, 0.3, 0.9))])
    car = FakeObjectAnnotation(FakeEntity("Car"), [FakeObjectFrame(0.6, FakeBox(0.5, 0.3, 0.9, 0.8))])
    # camera_main.mp4 is ~75 MB, above the default 50 MB inline limit — raise it for this
    # test so the inline-bytes path (the one a local demo actually uses) is what's exercised.
    provider = GoogleVideoIntelligenceProvider(
        _cfg(google_project_id="p", google_video_inline_max_mb=200),
        client=FakeVideoClient([FakeResult([person], [bike, car])]),
    )

    analysis = provider.analyze_video("camera_feed/camera_main.mp4", camera_id="CAM-001")
    assert analysis.provider == "GOOGLE_VIDEO_INTELLIGENCE"
    assert analysis.capabilities["helmet_status"] is False        # never faked
    assert analysis.capabilities["person_detection"] is True
    assert len(analysis.tracks) == 4
    keys = [t.key for t in analysis.tracks]
    assert len(set(keys)) == 4                                    # identity preserved, no collapsing
    person_tracks = [t for t in analysis.tracks if t.object_type == "PERSON"]
    assert len(person_tracks) == 2
    assert sorted(len(t.observations) for t in person_tracks) == [1, 2]
    first, second = sorted(person_tracks, key=lambda t: -len(t.observations))
    assert first.first_seen == pytest.approx(0.5) and first.last_seen == pytest.approx(1.5)
    types = {t.object_type for t in analysis.tracks}
    assert {"PERSON", "MOTORCYCLE", "VEHICLE"} <= types
    # normalized boxes convert to pixels only when the frame size is known
    obs = first.observations[0]
    assert obs.bbox_normalized is True
    assert obs.pixels(1000, 500) == (100, 100, 300, 300)
    assert analysis.extra["api_calls"] == 1
    # every time-stamped annotation across all 4 tracks (2+1+1+1) is an analyzed instant
    assert analysis.frames_analyzed == 5


def test_duration_conversion_handles_proto_and_floats():
    assert _duration_seconds(FakeDuration(1.5)) == pytest.approx(1.5)
    assert _duration_seconds(None) == 0.0
    assert _duration_seconds(2.25) == pytest.approx(2.25)


def test_one_video_request_covers_the_whole_video(ai_env):
    client = FakeVideoClient([FakeResult([FakePersonAnnotation([FakePersonTrack([FakeStamped(0.1, FakeBox(0, 0, 1, 1))])])])])
    provider = GoogleVideoIntelligenceProvider(
        _cfg(google_project_id="p", google_video_inline_max_mb=200), client=client)
    provider.analyze_video("camera_feed/camera_main.mp4")
    assert len(client.requests) == 1                              # ONE call for the whole video
    request = client.requests[0]
    assert "input_content" in request and request["input_content"]     # local file -> inline bytes
    assert request["features"]                                     # features requested


def test_large_local_file_is_refused_with_instructions(ai_env):
    provider = GoogleVideoIntelligenceProvider(
        _cfg(google_project_id="p", google_video_inline_max_mb=0), client=FakeVideoClient([]))
    # 0 MB would allow everything; force the limit down instead
    provider.config = dataclasses.replace(provider.config, google_video_inline_max_mb=1)
    with pytest.raises(ProviderNotConfigured) as ei:
        provider.analyze_video("camera_feed/camera_main.mp4")
    assert "gs://" in str(ei.value) and "AI_GOOGLE_VIDEO_INLINE_MAX_MB" in str(ei.value)


def test_gcs_mode_accepts_a_gs_uri(ai_env):
    client = FakeVideoClient([FakeResult([])])
    provider = GoogleVideoIntelligenceProvider(
        _cfg(google_project_id="p", google_video_input_mode="gcs"), client=client)
    provider.analyze_video("gs://sentinel-demo/road1.mp4")
    assert client.requests[0]["input_uri"] == "gs://sentinel-demo/road1.mp4"
    assert "input_content" not in client.requests[0]


# --------------------------------------------------------------------------- 5. plate / OCR
@pytest.mark.parametrize("raw,expected", [
    ("GJ 01 AB 1234", "GJ01AB1234"),
    ("gj-01-ab-1234", "GJ01AB1234"),
    ("MH12DE1433", "MH12DE1433"),
    ("", None),
    (None, None),
])
def test_plate_normalization(raw, expected):
    assert normalize_plate(raw) == expected


def test_plate_candidate_extraction_states():
    assert extract_plate_candidate("Vehicle GJ01AB1234 seen") == ("GJ01AB1234", "")
    candidate, reason = extract_plate_candidate("HELLO WORLD 99")
    assert reason and candidate in (None, "WORLD99", "HELLO")     # never invents a plate
    assert extract_plate_candidate(None) == (None, "no text detected")


def test_ocr_found_uncertain_and_not_found(ai_env):
    cfg = _cfg(google_project_id="p", anpr_min_plate_length=6)
    frame = np.zeros((40, 60, 3), dtype="uint8")

    found = GoogleVisionAnprProvider(cfg, client=FakeVisionClient(
        FakeVisionResponse([("GJ 01 AB 1234", [(0, 0), (10, 0), (10, 5), (0, 5)])])))
    result = found.recognize(frame)
    assert result.status == ANPR_FOUND and result.plate_number == "GJ01AB1234"
    assert result.provider == "GOOGLE_CLOUD_VISION" and result.raw_text == "GJ 01 AB 1234"
    assert found.calls == 1

    # a 6+ char alphanumeric token with a digit: plate-like enough to read, not a valid
    # Indian plate shape -> UNCERTAIN (4-char tokens like "AB12" are just NOT_FOUND)
    uncertain = GoogleVisionAnprProvider(cfg, client=FakeVisionClient(FakeVisionResponse([("WORLD99", [])])))
    assert uncertain.recognize(frame).status == ANPR_UNCERTAIN

    nothing = GoogleVisionAnprProvider(cfg, client=FakeVisionClient(FakeVisionResponse([])))
    assert nothing.recognize(frame).status == ANPR_NOT_FOUND


def test_ocr_prefers_text_inside_the_rider_region(ai_env):
    cfg = _cfg(google_project_id="p", anpr_min_plate_length=6)
    response = FakeVisionResponse([
        ("GJ01AB1234 GJ99XY0000", []),
        ("GJ99XY0000", [(5, 5), (25, 5), (25, 15), (5, 15)]),        # inside the ROI
        ("GJ01AB1234", [(50, 50), (90, 50), (90, 70), (50, 70)]),    # outside the ROI
    ])
    provider = GoogleVisionAnprProvider(cfg, client=FakeVisionClient(response))
    result = provider.recognize(np.zeros((100, 100, 3), dtype="uint8"), context={"roi": (0, 0, 40, 40)})
    assert result.plate_number == "GJ99XY0000"


def test_anpr_without_configuration_is_explicit(ai_env):
    provider = GoogleVisionAnprProvider(_cfg(google_project_id="", google_credentials=""))
    health = provider.health()
    assert health["status"] in ("NOT_CONFIGURED", "ERROR") and health["error"]
    result = provider.recognize(np.zeros((10, 10, 3), dtype="uint8"))
    assert result.status in ("ERROR", "NOT_CONFIGURED")
    assert result.plate_number is None
    disabled = NoAnprProvider()
    assert disabled.recognize(None).status == ANPR_NOT_CONFIGURED


def test_ocr_missing_file_is_an_error_not_a_plate(ai_env):
    provider = GoogleVisionAnprProvider(_cfg(google_project_id="p"), client=FakeVisionClient(FakeVisionResponse([])))
    result = provider.recognize(str(Path("does/not/exist.jpg")))
    assert result.status == "ERROR" and result.plate_number is None
