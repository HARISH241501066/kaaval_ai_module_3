"""
Video processing orchestrator for the KAAVAL AI layer.

    VideoSource --every frame--> HelmetDetector --detections--> HelmetViolationLogic
        (unique-rider registry + temporal confirmation)
        --one violation per UNIQUE rider + evidence frame--> EXISTING M3 C4 ingestion
        --> adapter normalization / dedup / persistence --> Mock M4

The loop reads the video frame by frame until `cap.read()` returns False (never "the first
frame", never "one result for the whole video"). By default EVERY decoded frame is run
through YOLO; `AI_FRAME_SKIP=N` is the explicit opt-in to sample every (N+1)-th frame.

Counting is deliberately two-layered, because they are different numbers:

    violation_detections   frame-level WITHOUT_HELMET detections   (e.g. 137)
    unique_violators       distinct tracked riders that violated   (e.g. 4)

It never writes to the database or to M4 directly: the only exit is the existing
C4IngestionPipeline (the same one sentinel_hwsim.push and the live worker use).
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.ai.config import AIConfig, get_ai_config
from app.ai.evidence import draw_detections, draw_frame_banner, save_evidence_frame
from app.ai.helmet_detector import HelmetDetector, get_detector
from app.ai.kaaval_logic import HelmetViolationLogic, ViolatorRecord
from app.ai.schemas import ConfirmedViolation, Detection
from app.ai.video_source import VideoError, open_source

log = logging.getLogger("sentinel.m3.ai.processor")

# Progress is written to the job (not per frame) so polling the API stays cheap.
PROGRESS_EVERY_FRAMES = 25


# ---------------------------------------------------------------------------
# Kaaval event payload -> existing M3 ingestion
# ---------------------------------------------------------------------------
def build_kaaval_event_payload(
    violation: ConfirmedViolation,
    config: AIConfig,
    inference_fps: float,
    latency_ms: Optional[float],
    evidence_uri: Optional[str],
    analytics_provider: Optional[str] = None,
    analytics_model_name: Optional[str] = None,
    analytics_model_version: Optional[str] = None,
    helmet_classifier: Optional[str] = None,
    anpr: Optional[Any] = None,
) -> Dict[str, Any]:
    """A raw Kaaval edge event, shaped exactly like what a real Kaaval node pushes
    (evt_id / node / cam / type / ts_ms / conf / model_ver / data) so the existing
    KAAVAL_EDGE adapter normalizes it like any other edge event.

    One payload is built per UNIQUE violator (not per frame), carrying the rider's
    whole-video counts in `data`.

    The extra keyword arguments carry *provenance* for externally analyzed videos: which
    video-analysis provider found the rider (Cloud Video Intelligence or the local YOLO
    model), which helmet classifier decided the status, and the bounded OCR result.
    They are optional so the local-only `/ai/helmet/process` path is unchanged.
    """
    data: Dict[str, Any] = {
        "helmet_status": "WITHOUT_HELMET",
        "object_type": "MOTORCYCLE_RIDER",
        "riders": 1,
        "unique_violator": True,
        "bbox": list(violation.detection.bbox),
        "frame_number": violation.frame_number,
        "video_timestamp": round(violation.timestamp, 3),
        "first_seen": round(violation.first_seen, 3),
        "last_seen": round(violation.last_seen, 3),
        "observations": violation.observations,
        "violation_detections": violation.violation_detections,
        "confirmations": violation.confirmations,
        "detection_model": config.model_name,
    }
    if analytics_provider:
        data["analytics_provider"] = analytics_provider
    if helmet_classifier:
        data["helmet_classifier"] = helmet_classifier
    if anpr is not None:
        # The plate is only asserted when OCR produced one; the status is always explicit so
        # "no plate" can never masquerade as "plate unknown".
        data["plate_status"] = getattr(anpr, "status", None)
        data["anpr_provider"] = getattr(anpr, "provider", None)
        if getattr(anpr, "plate_number", None):
            data["plate"] = anpr.plate_number
        if getattr(anpr, "raw_text", None):
            data["plate_raw_text"] = str(anpr.raw_text)[:120]
        if getattr(anpr, "confidence", None) is not None:
            data["anpr_confidence"] = anpr.confidence
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
        # model_name/version stay the HELMET model (the model that made the violation call);
        # the analysis backend is reported separately as external-provider provenance.
        "model_name": config.model_name,
        "model_ver": config.model_version,
        "analytics_provider": analytics_provider,
        "analytics_model_name": analytics_model_name,
        "analytics_model_version": analytics_model_version,
        "helmet_classifier": helmet_classifier,
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


def _violation_shell(record: ViolatorRecord, detection: Detection) -> ConfirmedViolation:
    """A render-only ConfirmedViolation for the evidence writer (same schema as an event)."""
    return ConfirmedViolation(
        event_id=record.event_id,
        camera_id=record.camera_id,
        detection=detection,
        confidence=record.max_confidence or detection.confidence,
        frame_number=detection.frame_number,
        timestamp=detection.timestamp,
        track_id=record.track_id,
        confirmations=record.confirmations or record.observations,
        first_seen=record.first_seen,
        last_seen=record.last_seen,
        observations=record.observations,
        violation_detections=record.violation_detections,
    )


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

    # --------------------------------------------------------------------------
    def process(
        self,
        camera_id: str,
        video_path: str,
        job: Optional[Any] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Decode -> detect (every frame by default) -> aggregate per rider -> evidence -> ingest.

        Returns {"metrics", "violations", "ingest_results"}; raises VideoError / DetectorError
        on hard failures, so a failed video can never be reported as "0 violations".
        """
        detector = self.detector or get_detector(self.config)
        logic = HelmetViolationLogic(camera_id, self.config)
        source = open_source(video_path)
        source.open()                                  # real metadata BEFORE the loop
        info = dict(source.info)
        video_fps = float(info.get("video_fps") or 0.0) or 25.0
        total_frames_meta = int(info.get("total_frames") or 0)
        video_duration = float(info.get("video_duration_seconds") or 0.0)
        frame_size = (int(info.get("video_width") or 0), int(info.get("video_height") or 0))
        started = time.perf_counter()

        frames_decoded = 0
        frames_processed = 0
        frames_skipped = 0
        detections_total = 0
        violations: List[Dict[str, Any]] = []
        ingested: List[Dict[str, Any]] = []
        latencies: List[float] = []
        stopped_early = False
        next_sample_at = 0.0
        interval = self.config.sample_interval_s
        legacy_fps_gate = self.config.frame_skip <= 0 and self.config.inference_fps > 0
        annotated = self._open_annotated_writer(video_path, info)

        if hasattr(detector, "reset_tracker"):
            detector.reset_tracker()                   # track ids are per video/session

        log.info(
            "processing %s as %s | model=%s v%s device=%s sampling=%s conf>=%.2f "
            "helmet_conf=%.2f confirm=%d frames",
            source.describe, camera_id, self.config.model_name, self.config.model_version,
            self.config.device, self.config.sampling_mode, self.config.confidence_threshold,
            self.config.helmet_min_confidence, self.config.confirmation_frames,
        )
        log.info(
            "video metadata: %d frames @ %.3f fps (%.2fs), %dx%d | %s",
            total_frames_meta, video_fps, video_duration,
            frame_size[0], frame_size[1], "every frame" if self.config.processes_every_frame
            else f"sampling {self.config.sampling_mode}",
        )

        try:
            for frame, frame_number, timestamp in source.frames():
                frames_decoded += 1
                if max_frames is not None and frames_decoded > max_frames:
                    frames_decoded -= 1
                    stopped_early = True
                    break

                # --- frame selection: AI_FRAME_SKIP wins, otherwise every frame ---
                if not self.config.should_process_frame(frame_number):
                    frames_skipped += 1
                    if annotated is not None:
                        annotated.write(frame)
                    continue
                if legacy_fps_gate and timestamp + 1e-6 < next_sample_at:
                    frames_skipped += 1
                    if annotated is not None:
                        annotated.write(frame)
                    continue
                next_sample_at = timestamp + interval

                # --- inference on THIS frame ---
                detections = detector.detect_frame(
                    frame, frame_number=frame_number, timestamp=timestamp
                )
                frames_processed += 1
                detections_total += len(detections)
                if getattr(detector, "last_latency_ms", None):
                    latencies.append(detector.last_latency_ms)

                # --- decide + aggregate (unique rider registry, not per-frame events) ---
                apply_logic = frame_size if frame_size != (0, 0) else None
                logic.observe(detections, frame_number, timestamp, frame_size=apply_logic)

                # --- one evidence frame per unique violator, from its best-confidence frame ---
                for record, detection in logic.evidence_candidates():
                    uri = save_evidence_frame(
                        frame, detections, _violation_shell(record, detection), self.config,
                        model_name=getattr(detector, "model_name", self.config.model_name),
                    )
                    logic.mark_evidence(record, uri)
                    log.info("evidence frame %s for unique violator %s (track=%s)",
                             uri or "(write failed)", record.event_id, record.track_id)

                # --- optional annotated output video ---
                if annotated is not None:
                    annotated.write(draw_frame_banner(
                        draw_detections(frame, detections), camera_id, timestamp, frame_number))

                if job is not None and frames_processed % PROGRESS_EVERY_FRAMES == 0:
                    job.metrics.update(self._partial_metrics(
                        frames_decoded, frames_processed, frames_skipped, total_frames_meta,
                        detections_total, logic, started))
        except VideoError:
            raise
        except Exception as e:
            if isinstance(e, VideoError):
                raise
            log.error("processing failed on %s after %d frames: %s", video_path, frames_decoded, e,
                      exc_info=True)
            raise
        finally:
            # resources are released even when decoding/inference raises
            if annotated is not None:
                annotated.release()
            source.close()

        elapsed = time.perf_counter() - started
        if frames_decoded == 0:
            raise VideoError(f"video '{video_path}' contains zero decodable frames")

        # ------------------------------------------------------------------
        # FINAL AGGREGATION: one Kaaval event per UNIQUE confirmed violator,
        # each with the single evidence frame written for that rider.
        # ------------------------------------------------------------------
        processing_fps = round(frames_processed / elapsed, 2) if elapsed > 0 else 0.0
        unique_events = logic.finalize_violations()
        for violation in unique_events:
            record = self._emit_violation(violation, detector, processing_fps)
            violations.append(violation.to_dict())
            ingested.append(record)

        accepted = sum(1 for r in ingested if getattr(r, "status", None) == "ACCEPTED")
        evidence_frames = sum(1 for v in violations if (v.get("evidence_ref") or {}).get("uri"))
        violation_detections = logic.violation_detections
        total_frames = total_frames_meta or frames_decoded
        decode_complete = (not stopped_early) and frames_decoded >= total_frames

        result = {
            "status": "completed",
            "camera_id": camera_id,
            "video": Path(video_path).name,
            "video_path": str(video_path),
            "duration_seconds": round(video_duration, 2),
            "video_fps": round(video_fps, 3),
            "total_frames": total_frames,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "detections": violation_detections,
            "violation_detections": violation_detections,
            "unique_violators": len(violations),
            "violations": len(violations),
            "evidence_frames": evidence_frames,
            "processing_fps": processing_fps,
            "processing_time": round(elapsed, 2),
        }
        metrics: Dict[str, Any] = {
            # --- spec block (the numbers shown to the user) ---
            "result": result,
            "outcome": ("PROCESSING_COMPLETED" if violations
                        else "PROCESSING_COMPLETED_WITH_ZERO_VIOLATIONS"),
            "camera_id": camera_id,
            "video_path": str(video_path),
            "video": Path(video_path).name,
            "video_fps": round(video_fps, 3),
            "video_duration_seconds": round(video_duration, 2),
            "video_width": frame_size[0],
            "video_height": frame_size[1],
            "total_frames": total_frames,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "decode_complete": decode_complete,
            "reached_last_frame": not stopped_early,
            # --- counting: frame detections vs unique people (different metrics) ---
            "detections": detections_total,
            "violation_detections": violation_detections,
            "unique_violators": len(violations),
            "violations": len(violations),
            "evidence_frames": evidence_frames,
            "tracked_riders": len(logic.violators),
            # --- throughput / model provenance ---
            "sampling_mode": self.config.sampling_mode,
            "processing_fps": processing_fps,
            "processing_time": round(elapsed, 2),
            "model": f"{self.config.model_name} v{self.config.model_version}",
            "device": self.config.device,
            "inference_fps_target": self.config.inference_fps,
            "frame_skip": self.config.frame_skip,
            "inference_size": getattr(detector, "last_image_size", None),
            "image_size_configured": self.config.image_size,
            "confidence_threshold": self.config.confidence_threshold,
            "helmet_min_confidence": self.config.helmet_min_confidence,
            "confirmation_frames": self.config.confirmation_frames,
            "frames_decoded": frames_decoded,
            "inference_frames": frames_processed,
            "inference_fps_actual": processing_fps,
            "events_ingested": accepted,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "cooldown_suppressed": logic.suppressed,
            "annotated_video": self._annotated_uri if annotated is not None else None,
        }
        log.info(
            "done %s: %d/%d frames processed (%d skipped), %d detections, %d frame-level "
            "violations, %d UNIQUE violators, %d evidence frame(s), %.1fs (%.2f fps)",
            video_path, frames_processed, frames_decoded, frames_skipped, detections_total,
            violation_detections, len(violations), evidence_frames, elapsed, processing_fps,
        )
        if job is not None:
            job.metrics.update(metrics)
            job.violations = violations
        return {"metrics": metrics, "violations": violations, "ingest_results": ingested,
                "violators": logic.violators_summary()}

    # ------------------------------------------------------------------ helpers
    def _emit_violation(self, violation: ConfirmedViolation, detector: HelmetDetector,
                        processing_fps: float) -> Any:
        """Ingest ONE unique violator through the existing M3 pipeline (evidence already written)."""
        payload = build_kaaval_event_payload(
            violation,
            self.config,
            inference_fps=processing_fps,
            latency_ms=getattr(detector, "last_latency_ms", None),
            evidence_uri=violation.evidence_uri,
        )
        log.info("sending %s to M3 ingestion (camera=%s, track=%s, node=%s, evidence=%s)",
                 violation.event_id, violation.camera_id, violation.track_id,
                 self.config.processing_node_id, violation.evidence_uri)
        result = self.ingest_fn(payload)
        log.info("M3 ingestion result for %s: %s (canonical=%s)", violation.event_id,
                 getattr(result, "status", "UNKNOWN"), getattr(result, "canonical_event_id", None))
        return result

    def _open_annotated_writer(self, video_path: str, info: Dict[str, Any]):
        """Optional annotated result video (AI_ANNOTATED_VIDEO=true). Never fatal."""
        self._annotated_uri: Optional[str] = None
        if not self.config.annotated_video:
            return None
        width, height = int(info.get("video_width") or 0), int(info.get("video_height") or 0)
        fps = float(info.get("video_fps") or 0.0)
        if width <= 0 or height <= 0 or fps <= 0:
            log.warning("annotated video skipped: unknown frame size/fps for %s", video_path)
            return None
        try:
            import cv2

            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{Path(video_path).stem}_annotated.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                log.warning("annotated video skipped: OpenCV could not open a writer for %s", path)
                return None
            self._annotated_uri = path.as_posix()
            log.info("annotated video will be written to %s", path)
            return writer
        except Exception as e:                                  # never fail the job for this
            log.warning("annotated video disabled after an error: %s", e)
            return None

    def _partial_metrics(self, frames_decoded: int, frames_processed: int, frames_skipped: int,
                         total_frames: int, detections: int, logic: HelmetViolationLogic,
                         started: float) -> Dict[str, Any]:
        elapsed = time.perf_counter() - started
        total = total_frames or frames_decoded
        return {
            "frames_decoded": frames_decoded,
            "frames_processed": frames_processed,
            "frames_skipped": frames_skipped,
            "total_frames": total,
            "sampling_mode": self.config.sampling_mode,
            "progress_percent": round(100.0 * frames_decoded / total, 1) if total else None,
            "detections": detections,
            "violation_detections": logic.violation_detections,
            "unique_violators": logic.unique_violators,
            "violations": logic.unique_violators,
            "processing_time": round(elapsed, 2),
            "processing_fps": round(frames_processed / elapsed, 2) if elapsed > 0 else 0.0,
            "inference_fps_actual": round(frames_processed / elapsed, 2) if elapsed > 0 else 0.0,
        }
