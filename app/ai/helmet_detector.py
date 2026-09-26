"""
YOLO helmet detector — the only place the Ultralytics API is allowed to appear.

Produces normalized Detection objects (see schemas.Detection); nothing downstream sees
Ultralytics Results, so the weights can later be replaced by a custom Kaaval model, an
ONNX/TensorRT export or an edge-accelerator runtime without changing M3.

Failure semantics are explicit and independent from camera health:
  * weights missing        -> ModelMissing  (YOLO model: FAILED, camera stays ONLINE)
  * ultralytics not import -> DetectorError
  * inference exception    -> DetectorError with context (never silently swallowed)
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.ai.config import AIConfig, get_ai_config
from app.ai.schemas import (
    MOTORCYCLE,
    PERSON,
    WITH_HELMET,
    WITHOUT_HELMET,
    Detection,
    filter_detections,
)

log = logging.getLogger("sentinel.m3.ai.detector")

# Health entity reported to M3 C6 — kept separate from CAMERA / PROCESSING_NODE /
# AI_RUNTIME / EVENT_PIPE so a YOLO failure never marks a physical camera offline.
YOLO_HEALTH_TYPE = "YOLO_MODEL"


class DetectorError(RuntimeError):
    """Inference-layer failure with context (model load, import or predict)."""


class ModelMissing(DetectorError):
    """The local weights file is not where the configuration says it is."""


def normalize_class_name(raw: str) -> str:
    """
    Map arbitrary model class labels onto the canonical Kaaval vocabulary:
      'With Helmet' / 'helmet' / 'with_helmet'  -> WITH_HELMET
      'Without Helmet' / 'no_helmet' / 'head'   -> WITHOUT_HELMET  (head = no helmet in most datasets)
      'person' / 'rider'                        -> PERSON
      'motorcycle' / 'motorbike' / 'bike'       -> MOTORCYCLE
    """
    s = str(raw).strip().lower().replace("-", " ").replace("_", " ")
    if "without" in s or s.startswith("no ") or s in {"nohelmet", "head", "bare head"}:
        return WITHOUT_HELMET
    if "helmet" in s:
        return WITH_HELMET
    if s in {"person", "rider", "people"}:
        return PERSON
    if s in {"motorcycle", "motorbike", "bike", "two wheeler", "twowheeler"}:
        return MOTORCYCLE
    return str(raw).upper()


class HelmetDetector:
    """
    Detector interface used by the video processor and the tests.

    Implementations: UltralyticsHelmetDetector (this prototype), later ONNX/TensorRT/
    custom Kaaval models — or a stub in unit tests.
    """

    def detect_frame(self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0) -> List[Detection]:
        raise NotImplementedError

    def reset_tracker(self) -> None:
        """Clear per-video tracking state. Track ids are only valid inside one video/session,
        so every job starts from an empty tracker (no-op for detectors without a tracker)."""

    def health_check(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def model_version(self) -> str:
        raise NotImplementedError


class UltralyticsHelmetDetector(HelmetDetector):
    """Local YOLO implementation. One instance per process; inference is serialized
    behind a lock because a shared Ultralytics model is not thread-safe."""

    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config or get_ai_config()
        self._lock = threading.Lock()
        self._track_disabled_reason: Optional[str] = None
        self.tracker_resets = 0                          # videos started with a fresh tracker
        self.last_latency_ms: Optional[float] = None
        self.last_image_size: Optional[int] = None      # effective imgsz of the last frame

        model_path = Path(self.config.model_path)
        if not model_path.is_file():
            raise ModelMissing(
                f"helmet model weights not found at '{model_path}'. "
                "Fetch them locally with: python -m app.ai.fetch_model "
                "(no cloud API key required)"
            )
        try:
            from ultralytics import YOLO  # lazy: keeps M3 importable without the ML stack
        except ImportError as e:  # pragma: no cover - exercised only when ML deps absent
            raise DetectorError(
                "ultralytics is not installed. Install the AI dependencies: "
                "pip install -r requirements.txt (ultralytics + opencv-python)"
            ) from e
        try:
            self._model = YOLO(str(model_path))
        except Exception as e:
            raise DetectorError(f"failed to load model '{model_path}': {e}") from e
        self._names: Dict[int, str] = {int(k): str(v) for k, v in dict(self._model.names).items()}
        log.info(
            "loaded helmet model %s v%s (%s classes: %s) on device=%s",
            self.config.model_name, self.config.model_version,
            len(self._names), ", ".join(self._names.values()), self.config.device,
        )

    # ------------------------------------------------------------------ props
    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def model_version(self) -> str:
        return self.config.model_version

    # -------------------------------------------------------------- inference
    def detect_frame(self, frame: np.ndarray, frame_number: int = 0, timestamp: float = 0.0) -> List[Detection]:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise DetectorError(f"empty frame passed to detector (frame_number={frame_number})")
        conf = self.config.confidence_threshold
        # Resolution-aware size: a fixed imgsz is what makes 4K wide shots blind (see
        # AIConfig.inference_size and the measurements in app/ai/config.py).
        imgsz = self.config.inference_size(frame.shape[0], frame.shape[1])
        self.last_image_size = imgsz
        kwargs: Dict[str, Any] = dict(
            conf=conf, imgsz=imgsz, device=self.config.device, verbose=False
        )
        started = time.perf_counter()
        try:
            with self._lock:
                use_track = self.config.tracking_enabled and self._track_disabled_reason is None
                if use_track:
                    try:
                        results = self._model.track(
                            frame, persist=True, tracker="bytetrack.yaml", **kwargs
                        )
                    except Exception as e:  # tracker config unsupported -> degrade, don't fail the job
                        self._track_disabled_reason = f"{e.__class__.__name__}: {e}"
                        log.warning("YOLO tracking unavailable, falling back to plain detection (%s)",
                                    self._track_disabled_reason)
                        self._disable_tracking()          # really remove tracking, then predict
                        results = self._model.predict(frame, **kwargs)
                else:
                    results = self._model.predict(frame, **kwargs)
        except DetectorError:
            raise
        except Exception as e:
            raise DetectorError(
                f"inference failed on frame {frame_number} (device={self.config.device}, "
                f"imgsz={imgsz}, frame={frame.shape[0]}x{frame.shape[1]}): {e}"
            ) from e
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0

        detections: List[Detection] = []
        try:
            for res in results or []:
                boxes = getattr(res, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                track_ids = None
                if getattr(boxes, "id", None) is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)
                for i, (box, cf, ci) in enumerate(zip(xyxy, confs, clss)):
                    raw_label = self._names.get(int(ci), str(ci))
                    detections.append(
                        Detection(
                            class_name=normalize_class_name(raw_label),
                            confidence=float(cf),
                            bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
                            frame_number=frame_number,
                            timestamp=timestamp,
                            track_id=int(track_ids[i]) if track_ids is not None else None,
                        )
                    )
        except Exception as e:
            raise DetectorError(f"invalid model output on frame {frame_number}: {e}") from e
        return filter_detections(detections, conf)

    # ----------------------------------------------------------------- tracker
    def reset_tracker(self) -> None:
        """
        Start this video with an empty tracker: track ids are only valid inside one video.

        Ultralytics keeps ByteTrack state on the predictor, and its
        `on_predict_postprocess_end` callback unconditionally reads `predictor.trackers[0]`,
        so the tracker list must never be emptied by hand (that raises
        `IndexError: list index out of range` on the very next frame). Dropping the predictor
        instead lets Ultralytics rebuild it — with a brand-new tracker — on the next call.
        """
        with self._lock:
            self.tracker_resets += 1
            if getattr(self._model, "predictor", None) is not None:
                self._model.predictor = None
                log.debug("tracker state cleared for a new video")

    def _disable_tracking(self) -> None:
        """Remove the tracker callbacks and predictor state so the rest of the video runs as
        plain detection (used when ByteTrack cannot run in this environment)."""
        try:
            callbacks = getattr(self._model, "callbacks", None) or {}
            for hook in ("on_predict_start", "on_predict_postprocess_end"):
                registered = list(callbacks.get(hook) or [])
                callbacks[hook] = [
                    cb for cb in registered
                    if not getattr(getattr(cb, "func", cb), "__module__", "").startswith("ultralytics.trackers")
                ]
            self._model.predictor = None
        except Exception as e:  # pragma: no cover - best effort, never fatal
            log.debug("could not detach the tracker callbacks: %s", e)

    # ----------------------------------------------------------------- health
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "READY",
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "model_path": self.config.model_path,
            "device": self.config.device,
            "tracking_enabled": self.config.tracking_enabled,
            "tracking_fallback": self._track_disabled_reason,
            "image_size": self.config.image_size,
            "image_size_auto": self.config.image_size_auto,
            "inference_size_4k": self.config.inference_size(2160, 3840),
            "classes": list(self._names.values()),
            "error": None,
        }


# ---------------------------------------------------------------------------
# Process-wide detector manager (health is reported independently of cameras)
# ---------------------------------------------------------------------------
_manager_lock = threading.Lock()
_detector: Optional[HelmetDetector] = None
_state: Dict[str, Any] = {"status": "NOT_LOADED", "error": None, "detail": {}}


def get_detector(config: Optional[AIConfig] = None, record: bool = True) -> HelmetDetector:
    """
    Load (once) and return the process-wide detector.
    Raises ModelMissing / DetectorError on failure; records YOLO_MODEL health either way.
    """
    global _detector
    cfg = config or get_ai_config()
    with _manager_lock:
        if _detector is not None:
            return _detector
        try:
            det = UltralyticsHelmetDetector(cfg)
        except DetectorError as e:
            _state.update(status="FAILED", error=str(e), detail={"model_path": cfg.model_path,
                                                                 "model_name": cfg.model_name,
                                                                 "device": cfg.device})
            log.error("YOLO helmet detector FAILED: %s", e)
            if record:
                record_yolo_health(_state)
            raise
        _detector = det
        _state.update(status="READY", error=None, detail=det.health_check())
        if record:
            record_yolo_health(_state)
        return det


def detector_status(record: bool = False, config: Optional[AIConfig] = None) -> Dict[str, Any]:
    """Snapshot of YOLO model health. Attempts a first load when never tried."""
    global _detector
    with _manager_lock:
        status = _state["status"]
    if status == "NOT_LOADED":
        try:
            get_detector(config, record=record)
            status = _state["status"]
        except DetectorError:
            status = _state["status"]
    if record:
        record_yolo_health(_state)
    return {"status": status, "error": _state["error"], **(_state.get("detail") or {})}


def record_yolo_health(state: Optional[Dict[str, Any]] = None) -> None:
    """Upsert the YOLO_MODEL health record so /health shows it as its own dimension."""
    state = _state if state is None else state
    try:
        from app.db.database import SessionLocal
        from app.db.repositories import HealthRepository

        status = "READY" if state["status"] == "READY" else (
            "FAILED" if state["status"] == "FAILED" else "NOT_LOADED"
        )
        with SessionLocal() as db:
            HealthRepository(db).upsert(
                entity_id=str(state.get("detail", {}).get("model_name") or get_ai_config().model_name),
                entity_type=YOLO_HEALTH_TYPE,
                status=status,
                details={"error": state.get("error"), **{
                    k: v for k, v in (state.get("detail") or {}).items() if v is not None
                }},
            )
    except Exception as e:  # health bookkeeping must never take the caller down
        log.warning("could not record YOLO health: %s", e)


def reset_detector_for_tests() -> None:
    """Drop the cached detector/state (used by the test suite)."""
    global _detector
    with _manager_lock:
        _detector = None
        _state.update(status="NOT_LOADED", error=None, detail={})
