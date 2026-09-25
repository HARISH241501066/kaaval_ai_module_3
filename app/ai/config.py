"""
Configuration for the KAAVAL AI local inference layer.

All runtime values come from the existing M3 Settings (env / .env, prefix AI_*), so there
is no second configuration system. This module adds:

* MODEL_METADATA — the model card (name, version, source URL, format, classes, license,
  pinned SHA-256) recorded before integration, per the hackathon rules.
* AIConfig — an immutable, typed view of the AI settings plus the demo video->camera map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from app.core.config import get_settings

# ---------------------------------------------------------------------------
# Model card (downloaded once, never a cloud/API dependency)
# ---------------------------------------------------------------------------
MODEL_METADATA: Dict[str, object] = {
    "model_name": "yolov8n-helmet-detection",
    "model_version": "1.0.0",
    "base_model": "Ultralytics YOLOv8n (COCO pretrained, fine-tuned)",
    "source": "Hugging Face — iam-tsr/yolov8n-helmet-detection",
    "source_url": "https://huggingface.co/iam-tsr/yolov8n-helmet-detection/resolve/main/best.pt",
    "format": "Ultralytics PyTorch .pt (YOLOv8n detection weights)",
    "classes": ["With Helmet", "Without Helmet"],
    "canonical_classes": ["WITH_HELMET", "WITHOUT_HELMET"],
    "license": "MIT",
    "sha256": "c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d",
    "size_bytes": 6199658,
    "metrics": {"precision": 0.855, "recall": 0.808, "mAP50": 0.881},
    "cloud_api_required": False,
    "notes": (
        "Two-class rider-head detector (helmet status only). It does NOT natively detect "
        "person or motorcycle boxes; see README 'Known limitations' for the association impact."
    ),
}


# --- inference size scaling -------------------------------------------------
# YOLO letterboxes every frame into a square imgsz. A single global value is wrong for
# mixed sources: the same 640 that is generous for a 1280px feed makes a 3840x2160 frame
# 6x smaller and the helmet model stops seeing rider heads at all. Measured on the demo
# wide feed (camera_sub.mp4, 3840x2160), same sample frames, conf>=0.25:
#
#     imgsz=640   ->  2 detections      (the bug)
#     imgsz=1280  -> 41 detections      ~0.28 s/frame on CPU
#     imgsz=1920  -> 94 detections      ~0.57 s/frame on CPU
#
# So 1280 is the default for a 4K source (2x the detections for 2x the time of 640, and
# half the time of 1920), and the size scales down proportionally for smaller sources.
REFERENCE_FRAME_LONG_SIDE = 3840     # the resolution the reference size was tuned on (4K)
REFERENCE_INFERENCE_SIZE = 1280      # inference size a 4K frame gets by default
MIN_INFERENCE_SIZE = 640             # never go below this: 640 is the accuracy floor
YOLO_STRIDE = 32                     # model stride; keep the input a multiple of it


def parse_video_sources(raw: str) -> Dict[str, str]:
    """Parse 'CAM-001=path/one.mp4;CAM-002=path/two.mp4' into a dict (whitespace tolerated)."""
    mapping: Dict[str, str] = {}
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"AI_VIDEO_SOURCES entry '{chunk}' must look like CAM-001=path/to/video.mp4")
        cam_id, path = chunk.split("=", 1)
        cam_id, path = cam_id.strip(), path.strip()
        if not cam_id or not path:
            raise ValueError(f"AI_VIDEO_SOURCES entry '{chunk}' must look like CAM-001=path/to/video.mp4")
        mapping[cam_id] = path
    return mapping


@dataclass(frozen=True)
class AIConfig:
    """Typed view of every AI_* setting used by the inference layer."""

    model_path: str
    model_name: str
    model_version: str
    confidence_threshold: float
    inference_fps: float
    image_size: int
    device: str
    tracking_enabled: bool
    helmet_min_confidence: float
    confirmation_frames: int
    cooldown_s: float
    evidence_dir: str
    processing_node_id: str
    video_sources: Dict[str, str] = field(default_factory=dict)
    # Keep new fields last: this dataclass has non-default fields before video_sources.
    image_size_auto: bool = True

    @classmethod
    def from_settings(cls, settings=None) -> "AIConfig":
        s = settings or get_settings()
        return cls(
            model_path=s.ai_model_path,
            model_name=s.ai_model_name,
            model_version=s.ai_model_version,
            confidence_threshold=s.ai_confidence_threshold,
            inference_fps=s.ai_inference_fps,
            image_size=s.ai_image_size,
            image_size_auto=s.ai_image_size_auto,
            device=s.ai_device,
            tracking_enabled=s.ai_tracking_enabled,
            helmet_min_confidence=s.ai_helmet_min_confidence,
            confirmation_frames=s.ai_helmet_confirmation_frames,
            cooldown_s=s.ai_helmet_event_cooldown_s,
            evidence_dir=s.ai_evidence_dir,
            processing_node_id=s.ai_processing_node_id,
            video_sources=parse_video_sources(s.ai_video_sources),
        )

    # ------------------------------------------------------------- convenience
    @property
    def model_file_exists(self) -> bool:
        return Path(self.model_path).is_file()

    @property
    def sample_interval_s(self) -> float:
        """Seconds of video between two inference observations."""
        return 1.0 / self.inference_fps if self.inference_fps > 0 else 0.0

    def inference_size(self, frame_height: int, frame_width: int) -> int:
        """
        Model input size for one frame, derived from the source resolution.

        A fixed size is the classic small-object bug: at imgsz=640 a 3840x2160 frame is
        downscaled 6x and distant riders disappear (see the measurements above). So the size
        tracks the long side of the frame, snapped to the YOLO stride and clamped to
        [MIN_INFERENCE_SIZE, image_size] — the configured value is the ceiling, never the
        target. With AI_IMAGE_SIZE_AUTO=false the configured value is used verbatim.
        """
        configured = max(YOLO_STRIDE, int(self.image_size))
        if not self.image_size_auto:
            return configured
        long_side = max(int(frame_height), int(frame_width))
        scaled = int(round(long_side * (REFERENCE_INFERENCE_SIZE / REFERENCE_FRAME_LONG_SIDE)))
        if scaled % YOLO_STRIDE:
            scaled += YOLO_STRIDE - (scaled % YOLO_STRIDE)      # keep the input stride-aligned
        floor = min(MIN_INFERENCE_SIZE, configured)             # a deliberate lower cap wins
        return max(floor, min(scaled, configured))

    def resolve_video(self, camera_id: Optional[str] = None, video_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Resolve (camera_id, video_path) from an explicit request, falling back to the
        configured demo map. Raises ValueError when the request cannot be resolved.
        """
        if camera_id and video_path:
            return camera_id, video_path
        if camera_id:
            if camera_id in self.video_sources:
                return camera_id, self.video_sources[camera_id]
            raise ValueError(
                f"No video configured for camera '{camera_id}'. "
                f"Known cameras: {', '.join(sorted(self.video_sources)) or '(none)'} "
                "or pass an explicit video_path."
            )
        if video_path:
            if len(self.video_sources) == 1:
                return next(iter(self.video_sources)), video_path
            raise ValueError("camera_id is required when AI_VIDEO_SOURCES maps several cameras.")
        raise ValueError("Provide camera_id (from AI_VIDEO_SOURCES) or a video_path.")

    def as_public_dict(self) -> Dict[str, object]:
        """Effective config for GET /ai/helmet/config — no secrets, ever."""
        return {
            "model_path": self.model_path,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_present": self.model_file_exists,
            "model_card": MODEL_METADATA,
            "confidence_threshold": self.confidence_threshold,
            "inference_fps": self.inference_fps,
            "image_size": self.image_size,
            "image_size_auto": self.image_size_auto,
            "inference_size_4k": self.inference_size(2160, 3840),   # what a 4K frame gets
            "device": self.device,
            "tracking_enabled": self.tracking_enabled,
            "helmet_min_confidence": self.helmet_min_confidence,
            "confirmation_frames": self.confirmation_frames,
            "event_cooldown_s": self.cooldown_s,
            "evidence_dir": self.evidence_dir,
            "processing_node_id": self.processing_node_id,
            "video_sources": dict(self.video_sources),
        }


def get_ai_config() -> AIConfig:
    """Cached-free factory: settings are process-stable, config objects are cheap."""
    return AIConfig.from_settings()
