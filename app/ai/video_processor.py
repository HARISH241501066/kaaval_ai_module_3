"""
Video processing orchestrator for the KAAVAL AI layer.

    VideoSource --frames--> sampled at AI_INFERENCE_FPS --frame--> HelmetDetector
        --detections--> HelmetViolationLogic --confirmed--> evidence frame
        --Kaaval event payload--> EXISTING M3 C4 ingestion pipeline
        --> adapter normalization / dedup / persistence --> Mock M4

It never writes to the database or to M4 directly: the only exit is the existing
C4IngestionPipeline (the same one sentinel_hwsim.push and the live worker use).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.ai.config import AIConfig, get_ai_config
from app.ai.evidence import save_evidence_frame
from app.ai.helmet_detector import HelmetDetector, get_detector
from app.ai.kaaval_logic import HelmetViolationLogic
from app.ai.schemas import ConfirmedViolation, Detection, Job, filter_detections
from app.ai.video_source import LocalVideoSource, VideoError, open_source

log = logging.getLogger("sentinel.m3.ai.processor")


# ---------------------------------------------------------------------------
# Kaaval event payload -> existing M3 ingestion
# ---------------------------------------------------------------------------
def build_kaaval_event_payload(
    violation: ConfirmedViolation,
    config: AIConfig,
    inference_fps: float,
    latency_ms: Optional[float],
    evidence_uri: Optional[str],
) -> Dict[str, Any]:
    """A raw Kaaval edge event, shaped exactly like what a real Kaaval node pushes
    (evt_id / node / cam / type / ts_ms / conf / model_ver / data) so the existing
    KAAVAL_EDGE adapter normalizes it like any other edge event."""
    data: Dict[str, Any] = {
        "helmet_status": "WITHOUT_HELMET",
        "object_type": "MOTORCYCLE_RIDER",
        "riders": 1,
        "bbox": list(violation.detection.bbox),
        "frame_number": violation.frame_number,
        "video_timestamp": round(violation.timestamp, 3),
        "confirmations": violation.confirmations,
        "detection_model": config.model_name,
    }
    if violation.track_id is not None:
        data["track_id"] = violation.track_id
    if violation.vehicle_bbox:
        data["vehicle_class"] = violation.vehicle_class or "TWO_WHEELER"
        data["vehicle_bbox"] = list(violation.vehicle_bbox)
    if evidence_uri:
        data["evidence_ref"] = {"type": "local_file", "uri": evidence_uri}

    payload: Dict[str, Any] = {
        "kind": "event",
        "evt_id": violation.event_id,
        "node": config.processing_node_id,
        "cam": violation.camera_id,
        "type": "HELMET_VIOLATION",
        "ts_ms": int(time.time() * 1000),
        "conf": round(violation.confidence, 4),
        "model_name": config.model_name,
        "model_ver": config.model_version,
        "data": data,
    }
    if inference_fps:
        payload["fps"] = round(float(inference_fps), 2)
    if latency_ms:
        payload["latency_ms"] = round(float(latency_ms), 1)
    return payload


def ingest_to_m3(payload: Dict[str, Any]) -> Any:
    """
    Send one Kaaval event through the EXISTING M3 pipeline
    (validate -> identity resolution -> dedup -> canonical envelope -> SQLite -> Mock M4).
    Runs the async pipeline on its own event loop so background jobs can call it synchronously.
    """
    from app.db.database import SessionLocal
    from app.services.pipeline import get_pipeline

    loop = asyncio.new_event_loop()
    try:
        with SessionLocal() as db:
            return loop.run_until_complete(
                get_pipeline().ingest_single_event(
                    db=db,
                    raw_payload=payload,
                    provider="KAAVAL_AI",
                    adapter_name="KAAVAL_EDGE",
                )
            )
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------
class VideoProcessor:
    """Processes one video for one camera. One instance per job."""

    def __init__(
        self,
        config: Optional[AIConfig] = None,
        detector: Optional[HelmetDetector] = None,
        ingest_fn: Callable[[Dict[str, Any]], Any] = ingest_to_m3,
    ):
        self.config = config or get_ai_config()
        self.detector = detector
        self.ingest_fn = ingest_fn

    def process(
        self,
        camera_id: str,
        video_path: str,
        job: Optional[Job] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Decode -> sample -> detect -> decide -> evidence -> ingest.
        Returns the metrics dict; raises VideoError / DetectorError on hard failures.
        """
        detector = self.detector or get_detector(self.config)
        logic = HelmetViolationLogic(camera_id, self.config)
        source = open_source(video_path)
        started = time.perf_counter()

        frames_decoded = 0
        inference_frames = 0
        detections_total = 0
        violations: List[Dict[str, Any]] = []
        ingested: List[Dict[str, Any]] = []
        latencies: List[float] = []
        next_sample_at = 0.0
        interval = self.config.sample_interval_s

        log.info(
            "processing %s as %s | model=%s v%s device=%s inference_fps=%.1f conf>=%.2f",
            source.describe, camera_id, self.config.model_name, self.config.model_version,
            self.config.device, self.config.inference_fps, self.config.confidence_threshold,
        )

        try:
            for frame, frame_number, timestamp in source.frames():
                frames_decoded += 1
                if max_frames is not None and frames_decoded > max_frames:
                    frames_decoded -= 1
                    break
                if timestamp + 1e-6 < next_sample_at:
                    continue
                next_sample_at = timestamp + interval

                detections = detector.detect_frame(frame, frame_number=frame_number, timestamp=timestamp)
                inference_frames += 1
                detections_total += len(detections)
                if getattr(detector, "last_latency_ms", None):
                    latencies.append(detector.last_latency_ms)

                # frame size scales the rider-association gate (see app/ai/kaaval_logic.py)
                confirmed = logic.observe(detections, frame_number, timestamp,
                                          frame_size=(frame.shape[1], frame.shape[0]))
                for violation in confirmed:
                    record = self._handle_violation(camera_id, frame, detections, violation, detector)
                    violations.append(violation.to_dict())
                    ingested.append(record)

                if job is not None and inference_frames % 5 == 0:
                    job.metrics.update(self._partial_metrics(
                        frames_decoded, inference_frames, detections_total, len(violations), started))
        except VideoError:
            raise
        except Exception as e:
            # keep the original failure type (VideoError / DetectorError) visible to the job runner
            if isinstance(e, (VideoError,)):
                raise
            log.error("processing failed on %s after %d frames: %s", video_path, frames_decoded, e, exc_info=True)
            raise

        elapsed = time.perf_counter() - started
        if frames_decoded == 0:
            raise VideoError(f"video '{video_path}' contains zero decodable frames")

        metrics = {
            "camera_id": camera_id,
            "video_path": video_path,
            "model": f"{self.config.model_name} v{self.config.model_version}",
            "device": self.config.device,
            "inference_fps_target": self.config.inference_fps,
            "inference_size": getattr(detector, "last_image_size", None),
            "image_size_configured": self.config.image_size,
            "confidence_threshold": self.config.confidence_threshold,
            "frames_decoded": frames_decoded,
            "frames_processed": frames_decoded,
            "inference_frames": inference_frames,
            "detections": detections_total,
            "violations": len(violations),
            "events_ingested": sum(1 for r in ingested if getattr(r, "status", None) == "ACCEPTED"),
            "processing_time": round(elapsed, 2),
            "inference_fps_actual": round(inference_frames / elapsed, 2) if elapsed > 0 else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "cooldown_suppressed": logic.suppressed,
        }
        log.info(
            "done %s: %d frames, %d inference observations, %d detections, %d violation event(s), %.1fs",
            video_path, frames_decoded, inference_frames, detections_total, len(violations), elapsed,
        )
        if job is not None:
            job.metrics.update(metrics)
            job.violations = violations
        return {"metrics": metrics, "violations": violations, "ingest_results": ingested}

    # ------------------------------------------------------------------ helpers
    def _handle_violation(
        self,
        camera_id: str,
        frame,
        detections: List[Detection],
        violation: ConfirmedViolation,
        detector: HelmetDetector,
    ) -> Any:
        """Evidence frame first (its uri goes into the event), then the M3 pipeline."""
        evidence_uri = save_evidence_frame(
            frame, detections, violation, self.config,
            model_name=getattr(detector, "model_name", self.config.model_name),
        )
        violation.evidence_uri = evidence_uri
        payload = build_kaaval_event_payload(
            violation,
            self.config,
            inference_fps=self.config.inference_fps,
            latency_ms=getattr(detector, "last_latency_ms", None),
            evidence_uri=evidence_uri,
        )
        log.info("sending %s to M3 ingestion (camera=%s, node=%s, evidence=%s)",
                 violation.event_id, camera_id, self.config.processing_node_id, evidence_uri)
        result = self.ingest_fn(payload)
        status = getattr(result, "status", "UNKNOWN")
        canonical_id = getattr(result, "canonical_event_id", None)
        log.info("M3 ingestion result for %s: %s (canonical=%s)", violation.event_id, status, canonical_id)
        return result

    def _partial_metrics(self, frames, inference_frames, detections, violations, started) -> Dict[str, Any]:
        elapsed = time.perf_counter() - started
        return {
            "frames_decoded": frames,
            "frames_processed": frames,
            "inference_frames": inference_frames,
            "detections": detections,
            "violations": violations,
            "processing_time": round(elapsed, 2),
            "inference_fps_actual": round(inference_frames / elapsed, 2) if elapsed > 0 else 0.0,
        }
