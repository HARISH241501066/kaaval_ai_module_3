"""
Google Cloud Video Intelligence provider — person/object detection and tracking.

What this provider DOES, verified against the public API reference
(google.cloud.videointelligence.v1):

* `PERSON_DETECTION`  -> `PersonDetectionAnnotation.tracks[]` (a `Track` per person, with a
  `VideoSegment`, `timestamped_objects[]`, per-frame `normalized_bounding_box` and optional
  attributes such as attire/hair colour).
* `OBJECT_TRACKING`   -> `ObjectTrackingAnnotation[]`; in BATCH (non-streaming) mode each
  annotation is ONE tracked object instance with a `VideoSegment` and `frames[]`
  (`track_id` is documented as streaming-mode only, so batch identity is the annotation
  itself — see `_object_tracks`).

What it does NOT do: classify helmet status. There is no "without helmet" class in these
annotations, so this provider reports `helmet_status: False` and the pipeline runs the local
helmet classifier on the tracked rider's frame (the documented hybrid flow).

Cost/limits handled explicitly:
* one `annotate_video` call per video, covering the WHOLE video (no per-frame calls);
* `input_content` (inline bytes) for local files up to `AI_GOOGLE_VIDEO_INLINE_MAX_MB`,
  otherwise the file must already live at a `gs://` URI (`input_uri`, the documented path for
  large videos) — the refusal message says exactly what to do;
* the operation is polled with a timeout, and progress is reported from
  `AnnotateVideoProgress`.

API lifecycle note: Cloud Video Intelligence is deprecated as of 2026-09-14 (usable until
2027-09-14, Google recommends migrating to Gemini). This is surfaced in `health()` and the
README so nobody is surprised later.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.ai.config import AIConfig, get_ai_config
from app.ai.providers.base import (
    ProviderError,
    ProviderNotConfigured,
    TrackObservation,
    TrackedObject,
    VideoAnalysis,
    VideoAnalyticsProvider,
)

log = logging.getLogger("sentinel.m3.ai.providers.google")

PROVIDER_NAME = "GOOGLE_VIDEO_INTELLIGENCE"
MODEL_NAME = "google-cloud-video-intelligence"
MODEL_VERSION = "v1"
API_DEPRECATION_NOTE = (
    "Cloud Video Intelligence API is deprecated since 2026-09-14 and shuts down "
    "2027-09-14 (Google recommends migrating to Gemini-family models)."
)
POLL_INTERVAL_S = 3.0


# ---------------------------------------------------------------------------
# dependency / configuration probes (never raise)
# ---------------------------------------------------------------------------
def sdk_available() -> bool:
    """Is the google-cloud-videointelligence package importable?"""
    try:
        import google.cloud.videointelligence  # noqa: F401

        return True
    except Exception:
        return False


def _duration_seconds(value: Any) -> float:
    """google.protobuf.duration_pb2.Duration (seconds+nanos) -> float seconds."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    total = getattr(value, "total_seconds", None)
    if callable(total):
        return float(total())
    seconds = getattr(value, "seconds", None)
    if seconds is not None:
        return float(seconds) + float(getattr(value, "nanos", 0) or 0) / 1e9
    try:  # pragma: no cover - defensive
        return float(value)
    except Exception:
        return 0.0


def validate_features(features: List[str]) -> List[str]:
    """Keep only supported Video Intelligence features (names, not SDK enums)."""
    supported = GoogleVideoIntelligenceProvider.GOOGLE_VIDEO_FEATURES
    unknown = [f for f in features if f.upper() not in supported]
    if unknown:
        raise ProviderNotConfigured(
            f"unsupported Video Intelligence feature(s) {unknown}. Supported here: {list(supported)}"
        )
    return [f.upper() for f in features]


def map_google_error(exc: Exception, stage: str) -> Exception:
    """Translate SDK exceptions into actionable provider errors (never a silent failure)."""
    name = exc.__class__.__name__
    text = str(exc)
    if name in ("DefaultCredentialsError", "RefreshError", "TransportError") or "credentials" in text.lower():
        return ProviderNotConfigured(
            f"{stage} failed: Google credentials were not found or are invalid ({name}). "
            "Set GOOGLE_APPLICATION_CREDENTIALS to a service-account JSON (never commit it) or run "
            "'gcloud auth application-default login'. Original error: " + text
        )
    if name in ("PermissionDenied", "Forbidden"):
        return ProviderError(
            f"{stage} failed: PERMISSION_DENIED — the Video Intelligence (or Vision) API is probably "
            f"not enabled for the project, or the service account lacks roles/videointelligence.admin. "
            f"Original error: {text}"
        )
    if name in ("ResourceExhausted", "TooManyRequests"):
        return ProviderError(f"{stage} failed: QUOTA_EXCEEDED / rate limited. Retry later or lower "
                             f"AI_GOOGLE_MAX_OBSERVATIONS_PER_TRACK. Original error: {text}")
    if name in ("DeadlineExceeded", "TimeoutError", "RetryError", "ServiceUnavailable", "InternalServerError"):
        return ProviderError(f"{stage} timed out or the service was unavailable. Increase "
                             f"GOOGLE_API_TIMEOUT_S. Original error: {text}")
    if name in ("InvalidArgument", "BadRequest", "FailedPrecondition"):
        return ProviderError(
            f"{stage} rejected the request: {text}. For local files check the codec/container and that "
            f"the file is within the inline limit (AI_GOOGLE_VIDEO_INLINE_MAX_MB); for large videos pass a "
            f"gs:// URI and set AI_GOOGLE_VIDEO_INPUT_MODE=gcs."
        )
    return ProviderError(f"{stage} failed ({name}): {text}")


class GoogleVideoIntelligenceProvider(VideoAnalyticsProvider):
    """Person/object detection + tracking through Cloud Video Intelligence."""

    def __init__(self, config: Optional[AIConfig] = None, client: Any = None):
        self.config = config or get_ai_config()
        self._client = client                      # injectable for tests
        self._client_error: Optional[str] = None
        self.last_progress: Dict[str, Any] = {}

    # ------------------------------------------------------------------ identity
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def capabilities(self) -> Dict[str, Any]:
        return {
            "video_analysis": "person_object_tracking",
            "person_detection": True,
            "object_tracking": True,
            "helmet_status": False,             # NOT provided by Google — local classifier does it
            "unique_identity": True,
            "bbox_units": "normalized",
            "cloud_api_calls": 1,               # one annotate_video per video (plus polling)
            "premium_features": ["PERSON_DETECTION", "OBJECT_TRACKING"],
            "deprecated": True,
            "deprecation_note": API_DEPRECATION_NOTE,
        }

    def health(self) -> Dict[str, Any]:
        """AVAILABLE / NOT_CONFIGURED / ERROR with an actionable reason. Never raises."""
        configured = self.config.google_configured
        source = self.config.google_credentials_source
        base = {
            "provider": PROVIDER_NAME,
            "sdk_installed": sdk_available(),
            "project_id_configured": bool(self.config.google_project_id),
            "credentials_source": source,
            "credentials_file_present": bool(self.config.google_credentials_path
                                             and Path(self.config.google_credentials_path).is_file()),
            "region": self.config.google_video_region or "(auto)",
            "features": self.config.google_features,
            "input_mode": self.config.google_video_input_mode,
            "deprecation_note": API_DEPRECATION_NOTE,
        }
        if not sdk_available():
            return {**base, "status": "ERROR", "error":
                    "google-cloud-videointelligence is not installed. Install it with "
                    "'pip install -r requirements.txt' (google-cloud-videointelligence)"}
        if not self.config.google_project_id:
            return {**base, "status": "NOT_CONFIGURED", "error":
                    "GOOGLE_CLOUD_PROJECT_ID is not set — Cloud Video Intelligence cannot be used."}
        if source == "none":
            return {**base, "status": "NOT_CONFIGURED", "error":
                    "Google credentials not found: set GOOGLE_APPLICATION_CREDENTIALS to a service-account "
                    "JSON (git-ignored) or run 'gcloud auth application-default login'."}
        if self._client_error:
            return {**base, "status": "ERROR", "error": self._client_error}
        return {**base, "status": "AVAILABLE", "error": None}

    # ------------------------------------------------------------------ client
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not sdk_available():
            raise ProviderNotConfigured(
                "google-cloud-videointelligence is not installed — "
                "run 'pip install -r requirements.txt'"
            )
        if not self.config.google_project_id:
            raise ProviderNotConfigured(
                "GOOGLE_CLOUD_PROJECT_ID is not set — Cloud Video Intelligence needs a project."
            )
        if self.config.google_credentials_path:
            # The SDK reads GOOGLE_APPLICATION_CREDENTIALS; make the configured path win without
            # ever logging or persisting the secret itself.
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", self.config.google_credentials_path)
        try:
            from google.cloud import videointelligence

            self._client = videointelligence.VideoIntelligenceServiceClient()
        except Exception as e:
            err = map_google_error(e, "Video Intelligence client setup")
            self._client_error = str(err)
            raise err
        return self._client

    # ------------------------------------------------------------------ analysis
    def analyze_video(
        self,
        video_path: str,
        camera_id: str = "CAM-001",
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_observations_per_track: Optional[int] = None,
    ) -> VideoAnalysis:
        """Analyze the COMPLETE video in ONE annotate_video request and normalize the tracks."""
        from app.ai.frames import FrameExtractor

        started = time.perf_counter()
        # Input validation first: a too-large local file must fail with instructions BEFORE any
        # client/credential work is attempted.
        input_field, media_meta = self._input_for(video_path)
        request = self.build_request(input_field)
        client = self._get_client()

        log.info("GOOGLE provider: annotate_video (%s) features=%s region=%s timeout=%ss",
                 media_meta["input_mode"], self.config.google_features,
                 self.config.google_video_region or "auto", self.config.google_api_timeout_s)
        try:
            operation = client.annotate_video(request=request)
        except Exception as e:
            raise map_google_error(e, "annotate_video")

        last_announced = -1
        deadline = time.time() + max(30, int(self.config.google_api_timeout_s))
        while not operation.done():
            progress = self._progress_percent(operation)
            if on_progress is not None and progress != last_announced:
                last_announced = progress
                self.last_progress = {
                    "stage": "video_analysis", "analysis_provider": PROVIDER_NAME,
                    "progress_percent": progress,
                    "elapsed_s": round(time.perf_counter() - started, 2),
                    "operation": getattr(operation, "operation", {}).get("name") if hasattr(
                        getattr(operation, "operation", None), "get") else None,
                }
                on_progress(self.last_progress)
            if time.time() > deadline:
                raise ProviderError(
                    f"Video Intelligence did not finish within {self.config.google_api_timeout_s}s "
                    f"(still at {progress}%). Raise GOOGLE_API_TIMEOUT_S or use a shorter clip."
                )
            time.sleep(POLL_INTERVAL_S)
        try:
            response = operation.result()
        except Exception as e:
            raise map_google_error(e, "annotate_video result")

        results = list(getattr(response, "annotation_results", None) or [])
        if not results:
            raise ProviderError("Video Intelligence returned no annotation results for this video")
        for result in results:
            annotated_error = getattr(result, "error", None)
            if annotated_error is not None and (getattr(annotated_error, "code", 0) or getattr(annotated_error, "message", "")):
                raise ProviderError(
                    f"Video Intelligence reported an error for this video: "
                    f"{getattr(annotated_error, 'message', annotated_error) or annotated_error}"
                )

        tracks = self._normalize(results)
        extractor = FrameExtractor(video_path) if Path(str(video_path)).is_file() else None
        duration = 0.0
        fps = 0.0
        total_frames = 0
        if extractor is not None:
            extractor.open()
            try:
                duration = extractor.frame_count / extractor.fps if extractor.fps else 0.0
                fps = extractor.fps
                total_frames = extractor.frame_count
            finally:
                extractor.close()

        warnings: List[str] = []
        if not tracks:
            warnings.append("Video Intelligence returned no person/object tracks for this video")
        elapsed = time.perf_counter() - started
        log.info("GOOGLE provider done: %d tracks (%d rider candidates) in %.1fs",
                 len(tracks), len([t for t in tracks if t.object_type == "PERSON"]), elapsed)
        return VideoAnalysis(
            provider=PROVIDER_NAME,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            capabilities=self.capabilities(),
            video_path=str(video_path),
            duration_s=round(duration, 3),
            video_fps=round(fps, 3),
            total_frames=total_frames,
            frames_analyzed=sum(len(t.observations) for t in tracks),
            tracks=sorted(tracks, key=lambda t: t.first_seen),
            segments_analyzed=len(results),
            warnings=warnings,
            extra={
                "camera_id": camera_id,
                "analysis_time_s": round(elapsed, 2),
                "input_mode": media_meta["input_mode"],
                "input_bytes": media_meta.get("input_bytes"),
                "features": self.config.google_features,
                "api_calls": 1,
                **{f"feature_version_{i}": v for i, v in enumerate(self._feature_versions(results))},
            },
        )

    # ------------------------------------------------------------------ request
    # Keys are the documented protobuf field names (see the API reference); passing them as a
    # plain dict lets the GAPIC layer build the message and keeps this provider testable with an
    # injected client even when the SDK is not installed here.
    GOOGLE_VIDEO_FEATURES = ("PERSON_DETECTION", "OBJECT_TRACKING", "LABEL_DETECTION", "TEXT_DETECTION")

    def build_request(self, input_field: Dict[str, Any]) -> Dict[str, Any]:
        """One annotate_video request covering the WHOLE video (no segments -> full duration)."""
        features = validate_features(self.config.google_features)
        video_context: Dict[str, Any] = {}
        if "PERSON_DETECTION" in features:
            video_context["person_detection_config"] = {
                "include_bounding_boxes": True,       # needed: boxes are how riders are tracked
                "include_attributes": True,           # attire/hair attributes are free context
                "include_pose_landmarks": False,      # not needed for helmet status
            }
        if "OBJECT_TRACKING" in features:
            video_context["object_tracking_config"] = {}
        request: Dict[str, Any] = {"features": features, "video_context": video_context}
        request.update(input_field)
        if self.config.google_video_region:
            request["location_id"] = self.config.google_video_region
        return request

    # ------------------------------------------------------------------ input
    def _input_for(self, video_path: str) -> (Dict[str, Any], Dict[str, Any]):
        """Decide between `input_uri` (gs://) and `input_content` (inline bytes)."""
        path = str(video_path)
        mode = self.config.google_video_input_mode
        if path.startswith("gs://"):
            return {"input_uri": path}, {"input_mode": "gcs", "input_bytes": None}
        if mode == "gcs":
            raise ProviderNotConfigured(
                f"AI_GOOGLE_VIDEO_INPUT_MODE=gcs needs a gs:// video URI, got '{path}'. "
                "Upload the file to a bucket and pass gs://bucket/object.mp4 (or switch input mode to auto/inline)."
            )
        p = Path(path)
        if not p.is_file():
            raise ProviderError(f"video file not found: '{path}'")
        size_bytes = p.stat().st_size
        limit = max(0, int(self.config.google_video_inline_max_mb)) * 1024 * 1024
        if limit and size_bytes > limit:
            raise ProviderNotConfigured(
                f"'{p.name}' is {size_bytes / 1048576:.1f} MB, above the inline limit of "
                f"{self.config.google_video_inline_max_mb} MB for Video Intelligence input_content. "
                f"Upload it to Cloud Storage and pass gs://bucket/{p.name} "
                f"(AI_GOOGLE_VIDEO_INPUT_MODE=gcs), or raise AI_GOOGLE_VIDEO_INLINE_MAX_MB if you "
                f"know the size is acceptable."
            )
        with open(path, "rb") as fh:
            content = fh.read()
        return ({"input_content": content},
                {"input_mode": "inline", "input_bytes": size_bytes})

    # ------------------------------------------------------------------ normalize
    @staticmethod
    def _progress_percent(operation: Any) -> Optional[int]:
        try:
            metadata = getattr(operation, "metadata", None)
            entries = list(getattr(metadata, "annotation_progress", None) or [])
            if entries:
                return int(getattr(entries[0], "progress_percent", 0) or 0)
        except Exception:  # pragma: no cover - progress must never break the run
            pass
        return None

    @staticmethod
    def _feature_versions(results: List[Any]) -> List[str]:
        versions: List[str] = []
        for result in results:
            for annotation in list(getattr(result, "person_detection_annotations", None) or []):
                version = getattr(annotation, "version", None)
                if version:
                    versions.append(f"person:{version}")
            for annotation in list(getattr(result, "object_annotations", None) or []):
                version = getattr(annotation, "version", None)
                if version:
                    versions.append(f"object:{version}")
        return versions

    def _normalize(self, results: List[Any]) -> List[TrackedObject]:
        """Duck-typed normalization (works on real protos and on test doubles)."""
        tracks: List[TrackedObject] = []
        for result in results:
            tracks.extend(self._person_tracks(result))
            tracks.extend(self._object_tracks(result))
        return tracks

    def _person_tracks(self, result: Any) -> List[TrackedObject]:
        """PERSON_DETECTION: one `Track` per person, with per-frame boxes and attributes."""
        out: List[TrackedObject] = []
        for annotation in list(getattr(result, "person_detection_annotations", None) or []):
            for index, track in enumerate(list(getattr(annotation, "tracks", None) or [])):
                segment = getattr(track, "segment", None)
                observations: List[TrackObservation] = []
                for stamped in list(getattr(track, "timestamped_objects", None) or []):
                    box = getattr(stamped, "normalized_bounding_box", None)
                    if box is None:
                        continue
                    attributes = {
                        getattr(attr, "name", ""): (getattr(attr, "value", "") or round(
                            float(getattr(attr, "confidence", 0.0) or 0.0), 3))
                        for attr in list(getattr(stamped, "attributes", None) or [])
                        if getattr(attr, "name", None)
                    }
                    observations.append(TrackObservation(
                        timestamp=_duration_seconds(getattr(stamped, "time_offset", None)),
                        bbox=self._normalized_box(box),
                        bbox_normalized=True,
                        confidence=float(getattr(track, "confidence", 0.0) or 0.0) or 1.0,
                        attributes=attributes,
                    ))
                if not observations:
                    continue
                out.append(TrackedObject(
                    key=f"person:{index}",
                    object_type="PERSON",
                    label="PERSON",
                    confidence=float(getattr(track, "confidence", 0.0) or 0.0),
                    observations=observations,
                    attributes={
                        "segment_start": _duration_seconds(getattr(segment, "start_time_offset", None)),
                        "segment_end": _duration_seconds(getattr(segment, "end_time_offset", None)),
                        "frame_count": len(observations),
                    },
                ))
        return out

    def _object_tracks(self, result: Any) -> List[TrackedObject]:
        """OBJECT_TRACKING (batch): each annotation is one tracked object instance."""
        out: List[TrackedObject] = []
        for index, annotation in enumerate(list(getattr(result, "object_annotations", None) or [])):
            entity = getattr(annotation, "entity", None)
            description = (getattr(entity, "description", None) or "").strip() or "OBJECT"
            segment = getattr(annotation, "segment", None)
            observations: List[TrackObservation] = []
            for frame in list(getattr(annotation, "frames", None) or []):
                box = getattr(frame, "normalized_bounding_box", None)
                if box is None:
                    continue
                observations.append(TrackObservation(
                    timestamp=_duration_seconds(getattr(frame, "time_offset", None)),
                    bbox=self._normalized_box(box),
                    bbox_normalized=True,
                    confidence=float(getattr(annotation, "confidence", 0.0) or 0.0) or 1.0,
                ))
            if not observations:
                continue
            out.append(TrackedObject(
                key=f"obj:{description}:{index}",
                object_type=self._canonical_object_type(description),
                label=description,
                confidence=float(getattr(annotation, "confidence", 0.0) or 0.0),
                observations=observations,
                attributes={
                    "entity_id": getattr(entity, "entity_id", None),
                    "segment_start": _duration_seconds(getattr(segment, "start_time_offset", None)),
                    "segment_end": _duration_seconds(getattr(segment, "end_time_offset", None)),
                    "frame_count": len(observations),
                },
            ))
        return out

    @staticmethod
    def _canonical_object_type(description: str) -> str:
        text = (description or "").upper()
        if "PERSON" in text or "PEOPLE" in text or "RIDER" in text:
            return "PERSON"
        if "MOTOR" in text or "BIKE" in text or "SCOOTER" in text:
            return "MOTORCYCLE"
        if "CAR" in text or "VEHICLE" in text or "TRUCK" in text or "BUS" in text:
            return "VEHICLE"
        return "OBJECT"

    @staticmethod
    def _normalized_box(box: Any) -> tuple:
        """Google's normalized [0,1] box as a 4-tuple, kept normalized until a frame size is known
        (`TrackObservation.pixels(width, height)` performs that conversion)."""
        return (float(getattr(box, "left", 0.0) or 0.0), float(getattr(box, "top", 0.0) or 0.0),
                float(getattr(box, "right", 0.0) or 0.0), float(getattr(box, "bottom", 0.0) or 0.0))
