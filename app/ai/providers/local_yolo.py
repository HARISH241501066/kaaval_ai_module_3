"""
Local YOLO provider + local helmet classifier (the offline half of the hybrid design).

* `LocalYOLOProvider` walks the COMPLETE video with the existing
  `UltralyticsHelmetDetector` (same weights, same `AI_FRAME_SKIP` policy as
  `/ai/helmet/process`) and groups per-frame detections into tracked riders with ByteTrack
  ids. Because the local model *is* a helmet model, each observation already carries
  `helmet_status`, so `capabilities()["helmet_status"]` is True and the pipeline needs no
  second stage. It declares `person_detection: False` on purpose — the two-class helmet model
  detects rider *heads*, not persons, and pretending otherwise would be a lie.

* `LocalYOLOClassifier` is the second stage for the Google path: Cloud Video Intelligence
  tracks the rider, this classifies the helmet status of that rider's crop with the same
  local weights (see the hybrid flow in the task's section 33).

Both are bounded: one frame at a time, no frame buffering, no extra model weights.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.ai.config import AIConfig, get_ai_config
from app.ai.helmet_detector import DetectorError, HelmetDetector, get_detector
from app.ai.kaaval_logic import VIOLATOR_RECENCY_S, _centre, _gate_px
from app.ai.providers.base import (
    Box,
    HelmetClassifier,
    ProviderError,
    TrackObservation,
    TrackedObject,
    VideoAnalysis,
    VideoAnalyticsProvider,
)
from app.ai.schemas import Detection, HELMET_CLASSES, WITH_HELMET, WITHOUT_HELMET
from app.ai.video_source import VideoError, open_source

log = logging.getLogger("sentinel.m3.ai.providers.local")

PROVIDER_NAME = "LOCAL_YOLO"
CLASSIFIER_NAME = "LOCAL_YOLO_HELMET"

# The rider crop handed to the helmet model is padded so the head is never clipped.
CROP_PADDING = 0.15


def _crop_with_padding(frame: np.ndarray, bbox: Box, padding: float = CROP_PADDING) -> Tuple[np.ndarray, int, int]:
    """Crop `bbox` from `frame` with padding; returns (crop, offset_x, offset_y)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    pad_x, pad_y = int(round(bw * padding)), int(round(bh * padding))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        cx1, cy1, cx2, cy2 = x1, y1, x2, y2
    return frame[cy1:cy2, cx1:cx2].copy(), cx1, cy1


class LocalYOLOProvider(VideoAnalyticsProvider):
    """Video analysis with the local helmet model (frame loop + ByteTrack identities)."""

    def __init__(self, config: Optional[AIConfig] = None, detector: Optional[HelmetDetector] = None):
        self.config = config or get_ai_config()
        self.detector = detector

    # ------------------------------------------------------------------ identity
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def capabilities(self) -> Dict[str, Any]:
        return {
            "video_analysis": "helmet_model_frame_loop",
            "person_detection": False,          # two-class helmet model: heads, not persons
            "object_tracking": True,
            "tracker": "bytetrack",
            "helmet_status": True,              # the local model IS a helmet classifier
            "unique_identity": True,
            "bbox_units": "pixels",
            "offline": True,
            "cloud_api_calls": 0,
            "classes": list(HELMET_CLASSES),
        }

    def health(self) -> Dict[str, Any]:
        """READY only when the local weights can be loaded; never raises."""
        try:
            detector = self.detector or get_detector(self.config, record=False)
            return {"status": "READY", "provider": PROVIDER_NAME, "error": None,
                    **{k: v for k, v in detector.health_check().items() if k != "status"}}
        except DetectorError as e:
            return {"status": "FAILED", "provider": PROVIDER_NAME, "error": str(e)}

    def reset(self) -> None:
        if self.detector is not None and hasattr(self.detector, "reset_tracker"):
            self.detector.reset_tracker()

    # ------------------------------------------------------------------ analysis
    def analyze_video(
        self,
        video_path: str,
        camera_id: str = "CAM-001",
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_observations_per_track: Optional[int] = None,
    ) -> VideoAnalysis:
        detector = self.detector or get_detector(self.config)
        self.reset()
        source = open_source(video_path)
        source.open()
        info = dict(source.info)
        video_fps = float(info.get("video_fps") or 0.0) or 25.0
        total_frames = int(info.get("total_frames") or 0)
        interval = self.config.sample_interval_s
        legacy_gate = self.config.frame_skip <= 0 and self.config.inference_fps > 0
        next_sample_at = 0.0

        tracks: Dict[str, TrackedObject] = {}
        frames_analyzed = 0
        frames_decoded = 0
        started = time.perf_counter()
        warnings: List[str] = []

        log.info("LOCAL provider analyzing %s (%d frames @ %.3f fps) as %s",
                 video_path, total_frames, video_fps, camera_id)
        try:
            for frame, frame_number, timestamp in source.frames():
                frames_decoded += 1
                if not self.config.should_process_frame(frame_number):
                    continue
                if legacy_gate and timestamp + 1e-6 < next_sample_at:
                    continue
                next_sample_at = timestamp + interval

                detections = detector.detect_frame(frame, frame_number=frame_number, timestamp=timestamp)
                frames_analyzed += 1
                frame_size = (frame.shape[1], frame.shape[0])
                for det in detections:
                    if det.class_name not in HELMET_CLASSES:
                        continue
                    key, track_id = self._identity(det, tracks, timestamp, frame_size)
                    track = tracks.get(key)
                    if track is None:
                        track = TrackedObject(key=key, object_type="PERSON", label="RIDER",
                                              confidence=det.confidence,
                                              attributes={"track_id": track_id})
                        tracks[key] = track
                    track.confidence = max(track.confidence, float(det.confidence))
                    track.observations.append(TrackObservation(
                        timestamp=float(timestamp), bbox=tuple(det.bbox), confidence=float(det.confidence),
                        frame_number=int(frame_number), helmet_status=det.class_name,
                        helmet_confidence=float(det.confidence),
                        attributes={"track_id": track_id},
                    ))
                if on_progress is not None and frames_analyzed % 25 == 0:
                    on_progress({
                        "frames_decoded": frames_decoded,
                        "frames_processed": frames_analyzed,
                        "total_frames": total_frames,
                        "progress_percent": round(100.0 * frames_decoded / total_frames, 1) if total_frames else None,
                        "tracks": len(tracks),
                        "analysis_provider": PROVIDER_NAME,
                        "stage": "video_analysis",
                    })
        finally:
            source.close()

        if frames_decoded == 0:
            raise VideoError(f"video '{video_path}' contains zero decodable frames")

        elapsed = time.perf_counter() - started
        log.info("LOCAL provider done: %d frames, %d tracks, %.1fs", frames_analyzed, len(tracks), elapsed)
        return VideoAnalysis(
            provider=PROVIDER_NAME,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            capabilities=self.capabilities(),
            video_path=str(video_path),
            duration_s=float(info.get("video_duration_seconds") or 0.0),
            video_fps=video_fps,
            total_frames=total_frames,
            frames_analyzed=frames_analyzed,
            tracks=sorted(tracks.values(), key=lambda t: t.first_seen),
            segments_analyzed=1,
            warnings=warnings,
            extra={
                "camera_id": camera_id,
                "frames_decoded": frames_decoded,
                "analysis_time_s": round(elapsed, 2),
                "sampling_mode": self.config.sampling_mode,
                "inference_size": getattr(detector, "last_image_size", None),
            },
        )

    # ------------------------------------------------------------------ helpers
    def _identity(self, det: Detection, tracks: Dict[str, TrackedObject], timestamp: float,
                  frame_size: Tuple[int, int]) -> Tuple[str, Optional[int]]:
        """Stable per-video identity: the ByteTrack id, or a spatial match when there is none."""
        if det.track_id is not None:
            key = f"trk:{det.track_id}"
            if key in tracks:
                return key, int(det.track_id)
            recent = self._nearest_recent(det, tracks, timestamp)
            if recent is not None:                       # absorbed an id switch on the same rider
                return recent, int(det.track_id)
            return key, int(det.track_id)

        recent = self._nearest_recent(det, tracks, timestamp)
        if recent is not None:
            return recent, None
        return f"loc:{det.center[0]:.0f},{det.center[1]:.0f}@{det.frame_number}", None

    def _nearest_recent(self, det: Detection, tracks: Dict[str, TrackedObject],
                        timestamp: float) -> Optional[str]:
        gate = _gate_px(det, None)
        cx, cy = _centre(det.bbox)
        best, best_dist = None, None
        for key, track in tracks.items():
            if not track.observations:
                continue
            last = track.observations[-1]
            if (timestamp - last.timestamp) > VIOLATOR_RECENCY_S:
                continue
            if det.track_id is not None and track.attributes.get("track_id") not in (None, det.track_id):
                continue
            sx, sy = _centre(last.bbox)
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            limit = gate if (det.track_id is None or track.attributes.get("track_id") is None) else 0.5 * gate
            if dist > limit:
                continue
            if best_dist is None or dist < best_dist:
                best, best_dist = key, dist
        return best


class LocalYOLOClassifier(HelmetClassifier):
    """
    Helmet classifier used as the second stage of the Google pipeline.

    Given the frame and the rider box that Cloud Video Intelligence tracked, it crops the
    rider region (with padding so the head is never clipped) and runs the same local
    YOLO helmet model on it, mapping the head boxes back to full-frame coordinates.
    """

    def __init__(self, config: Optional[AIConfig] = None, detector: Optional[HelmetDetector] = None):
        self.config = config or get_ai_config()
        self.detector = detector

    @property
    def name(self) -> str:
        return CLASSIFIER_NAME

    def capabilities(self) -> Dict[str, Any]:
        return {
            "helmet_status": True,
            "classes": list(HELMET_CLASSES),
            "runs_on": "rider_crop",
            "crop_padding": CROP_PADDING,
            "min_confidence": self.config.helmet_min_confidence,
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "offline": True,
            "cloud_api_calls": 0,
        }

    def health(self) -> Dict[str, Any]:
        try:
            detector = self.detector or get_detector(self.config, record=False)
            return {"status": "READY", "classifier": CLASSIFIER_NAME, "error": None,
                    "model_name": detector.model_name, "model_version": detector.model_version}
        except DetectorError as e:
            return {"status": "FAILED", "classifier": CLASSIFIER_NAME, "error": str(e)}

    def classify(
        self,
        frame: np.ndarray,
        bbox: Optional[Box] = None,
        frame_number: int = 0,
        timestamp: float = 0.0,
        track_id: Optional[int] = None,
    ) -> List[Detection]:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ProviderError("helmet classifier received an empty frame")
        detector = self.detector or get_detector(self.config)
        if bbox is None:
            return detector.detect_frame(frame, frame_number=frame_number, timestamp=timestamp)

        crop, ox, oy = _crop_with_padding(frame, bbox)
        if crop.size == 0:
            raise ProviderError(f"empty rider crop for bbox={bbox}")
        raw = detector.detect_frame(crop, frame_number=frame_number, timestamp=timestamp)
        out: List[Detection] = []
        for det in raw:
            x1, y1, x2, y2 = det.bbox
            out.append(Detection(
                class_name=det.class_name,
                confidence=det.confidence,
                bbox=(int(x1 + ox), int(y1 + oy), int(x2 + ox), int(y2 + oy)),
                frame_number=frame_number,
                timestamp=timestamp,
                track_id=track_id if track_id is not None else det.track_id,
            ))
        return out
