"""
Provider-based video analysis: the Google Cloud path (and the same path with the local provider).

    video
      |
      v
    VideoAnalyticsProvider.analyze_video()        Cloud Video Intelligence: PERSON_DETECTION +
      |                                           OBJECT_TRACKING over the WHOLE video (1 call)
      v
    tracked riders (stable per-video identity)
      |
      v
    helmet classifier on the tracked rider         Cloud VI has no helmet verdict, so the local
      |                                            YOLO helmet model classifies the rider crop
      v
    Kaaval decision logic (reuses app/ai/kaaval_logic.py)
      |  temporal confirmation + unique-rider registry (one record per tracked rider)
      v
    evidence frame (best-confidence frame of that rider, extracted locally)
      |
      v
    ANPR: ONE Cloud Vision OCR call per confirmed violator (never per frame)
      |
      v
    Kaaval event -> EXISTING M3 C4 ingestion -> canonical envelope -> SQLite -> Mock M4

Nothing here bypasses M3, and nothing here is a second event pipeline: the payload builder,
the evidence writer, the decision logic, the ingestion call and the canonical adapter are the
same ones `/ai/helmet/process` uses.

Cost control (task section 26):
* 1 Video Intelligence operation per video (whole video, no per-frame uploads);
* helmet classification runs on the local model and is bounded per track and per job;
* OCR runs once per confirmed unique violator and is hard-capped by ANPR_MAX_PER_JOB.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from app.ai.config import AIConfig, get_ai_config
from app.ai.evidence import save_evidence_frame
from app.ai.frames import FrameExtractionError, FrameExtractor
from app.ai.kaaval_logic import HelmetViolationLogic
from app.ai.providers import (
    ProviderSelection,
    get_anpr_provider,
    get_helmet_classifier,
    select_video_provider,
)
from app.ai.providers.base import (
    ANPR_FOUND,
    AnprResult,
    TrackedObject,
    VideoAnalysis,
    VideoAnalyticsProvider,
)
from app.ai.schemas import Detection, HELMET_CLASSES
from app.ai.video_processor import build_kaaval_event_payload, ingest_to_m3

log = logging.getLogger("sentinel.m3.ai.video_analysis")

PROGRESS_EVERY_TRACKS = 5


class VideoAnalysisProcessor:
    """
    Provider-backed analyzer for one video/job.

    The Google and local backends share this class: the provider decides *what* is in the
    video, the helmet classifier decides *helmet status*, and `HelmetViolationLogic` decides
    *how many unique riders violated the rule*.
    """

    def __init__(
        self,
        config: Optional[AIConfig] = None,
        provider: Optional[VideoAnalyticsProvider] = None,
        classifier: Optional[Any] = None,
        anpr: Optional[Any] = None,
        ingest_fn: Callable[[Dict[str, Any]], Any] = ingest_to_m3,
        selection: Optional[ProviderSelection] = None,
    ):
        self.config = config or get_ai_config()
        self._provider = provider
        self._classifier = classifier
        self._anpr = anpr
        self.ingest_fn = ingest_fn
        self.selection = selection

    # --------------------------------------------------------------------------
    def process(self, camera_id: str, video_path: str, job: Optional[Any] = None) -> Dict[str, Any]:
        """
        Analyze the complete video and emit ONE event per unique helmet violator.

        Raises ProviderError / VideoError / DetectorError on hard failures, so a failed run can
        never be reported as "zero violations".
        """
        cfg = self.config
        started = time.perf_counter()

        # 1. resolve the analysis provider (with the reported fallback policy)
        selection = self.selection or select_video_provider(cfg, detector=None)
        provider = self._provider or selection.provider
        classifier = self._classifier if self._classifier is not None else get_helmet_classifier(cfg)
        anpr = self._anpr if self._anpr is not None else get_anpr_provider(cfg)

        log.info(
            "video analysis on %s as %s | requested=%s effective=%s%s | helmet_classifier=%s | anpr=%s",
            video_path, camera_id, selection.requested, selection.effective,
            f" (fallback: {selection.fallback_reason})" if selection.fallback_reason else "",
            classifier.name, anpr.name,
        )

        # 2. analyze the WHOLE video with the provider
        analysis: VideoAnalysis = provider.analyze_video(
            video_path,
            camera_id=camera_id,
            on_progress=self._progress_callback(job),
            max_observations_per_track=cfg.google_max_observations_per_track,
        )
        video_fps = analysis.video_fps or 0.0
        frame_size = self._frame_size(video_path, analysis)

        logic = HelmetViolationLogic(camera_id, cfg)
        detections_total = 0
        classifications = 0
        classification_failures = 0
        tracks_with_helmet_status = 0
        tracks_skipped = 0
        evidence_frames = 0
        anpr_attempts = 0
        anpr_matches = 0
        anpr_results: List[Dict[str, Any]] = []
        violations: List[Dict[str, Any]] = []
        ingested: List[Dict[str, Any]] = []

        candidate_tracks = self._candidate_tracks(analysis, cfg)
        log.info("analysis: %d tracks, %d candidate rider tracks (%s)",
                 len(analysis.tracks), len(candidate_tracks), analysis.provider)

        extractor: Optional[FrameExtractor] = None
        try:
            if candidate_tracks:
                extractor = FrameExtractor(video_path)
                extractor.open()
                if frame_size == (0, 0):
                    frame_size = extractor.size

            for track_index, track in enumerate(candidate_tracks, start=1):
                track_id = track_index
                samples = self._samples(track, analysis)
                if not samples:
                    tracks_skipped += 1
                    continue

                if analysis.helmet_status_known():
                    tracks_with_helmet_status += 1
                    detections = self._detections_from_track(track, samples, frame_size, track_id)
                    detections_total += len(detections)
                    if detections:
                        self._observe(logic, detections, frame_size)
                else:
                    for sample in samples:
                        if classifications >= cfg.google_max_classifications:
                            log.warning("classification budget (%d) reached — remaining riders in %s "
                                        "are not classified", cfg.google_max_classifications, track.key)
                            break
                        frame, frame_number = self._frame_for(extractor, sample, video_fps)
                        if frame is None:
                            classification_failures += 1
                            continue
                        if frame_size == (0, 0):
                            frame_size = (frame.shape[1], frame.shape[0])
                        box = sample.pixels(frame.shape[1], frame.shape[0])
                        try:
                            found = classifier.classify(
                                frame, bbox=box, frame_number=frame_number,
                                timestamp=sample.timestamp, track_id=track_id,
                            )
                        except Exception as e:                      # one bad frame must not kill the job
                            classification_failures += 1
                            log.warning("helmet classification failed for %s at t=%.2fs: %s",
                                        track.key, sample.timestamp, e)
                            continue
                        classifications += 1
                        detections_total += len(found)
                        if not found:
                            continue
                        self._observe(logic, found, frame_size)

                if job is not None and track_index % PROGRESS_EVERY_TRACKS == 0:
                    job.metrics.update(self._partial_metrics(
                        analysis, track_index, len(candidate_tracks), classifications, logic,
                        anpr_attempts, anpr_matches, started))
        finally:
            if extractor is not None:
                extractor.close()

        # 3. FINAL AGGREGATION: one event per UNIQUE confirmed violator
        unique_events = logic.finalize_violations()
        for violation in unique_events:
            # ONE evidence image per unique violator, rendered from the original video at that
            # rider's best-confidence frame (never one image per frame, never the whole video).
            evidence_uri = violation.evidence_uri or self._write_evidence(video_path, violation, cfg)
            violation.evidence_uri = evidence_uri
            if evidence_uri:
                evidence_frames += 1

            anpr_result: Optional[AnprResult] = None
            if cfg.anpr_enabled and evidence_uri and anpr_attempts < cfg.anpr_max_per_job:
                anpr_result = self._run_anpr(anpr, violation, evidence_uri)
                anpr_attempts += 1
                if anpr_result.status == ANPR_FOUND:
                    anpr_matches += 1
                anpr_results.append({"event_id": violation.event_id, **anpr_result.to_dict()})

            record = self._emit_violation(violation, analysis, classifier, anpr_result, cfg)
            violations.append(violation.to_dict())
            ingested.append(record)

        elapsed = time.perf_counter() - started
        accepted = sum(1 for r in ingested if getattr(r, "status", None) == "ACCEPTED")
        duration = self._video_duration(video_path, analysis, video_fps)
        result = {
            "status": "completed",
            "camera_id": camera_id,
            "video": Path(video_path).name,
            "video_path": str(video_path),
            "duration_seconds": round(duration, 2),
            "unique_violators": len(violations),
            "violation_events": len(violations),
            "anpr_attempts": anpr_attempts,
            "anpr_matches": anpr_matches,
        }
        metrics: Dict[str, Any] = {
            "result": result,
            "outcome": ("PROCESSING_COMPLETED" if violations
                        else "PROCESSING_COMPLETED_ZERO_VIOLATIONS"),
            "camera_id": camera_id,
            "video": Path(video_path).name,
            "video_path": str(video_path),
            "duration_seconds": round(duration, 2),
            "video_fps": round(video_fps, 3),
            "total_frames": analysis.total_frames,
            "frames_analysed_by_provider": analysis.frames_analyzed,
            "analysis_provider": analysis.provider,
            "analysis_model": f"{analysis.model_name} {analysis.model_version}",
            "analysis_requested": selection.requested,
            "analysis_effective": selection.effective,
            "analysis_fallback": bool(selection.fallback_reason),
            "analysis_fallback_reason": selection.fallback_reason,
            "analysis_capabilities": analysis.capabilities,
            "analysis_warnings": analysis.warnings,
            "tracks_seen": len(analysis.tracks),
            "rider_tracks": len(candidate_tracks),
            "tracks_skipped_unconfirmable": tracks_skipped,
            "tracks_with_provider_helmet_status": tracks_with_helmet_status,
            "helmet_classifier": classifier.name,
            "helmet_classifications": classifications,
            "helmet_classification_failures": classification_failures,
            "helmet_confidence_threshold": cfg.helmet_confidence_threshold,
            "confirmation_observations": cfg.confirmation_observations,
            "observations": sum(len(t.observations) for t in analysis.tracks),
            "detections": detections_total,
            "violation_detections": logic.violation_detections,
            "unique_violators": len(violations),
            "violation_events": len(violations),
            "evidence_frames": evidence_frames,
            "anpr_provider": anpr.name,
            "anpr_enabled": cfg.anpr_enabled,
            "anpr_attempts": anpr_attempts,
            "anpr_matches": anpr_matches,
            "anpr_results": anpr_results,
            "events_ingested": accepted,
            "processing_time": round(elapsed, 2),
            "processing_fps": round(analysis.frames_analyzed / elapsed, 2) if elapsed > 0 else 0.0,
        }
        log.info(
            "video analysis done: %s | %d tracks, %d classifications, %d frame-level violations, "
            "%d UNIQUE violators, %d evidence frame(s), ANPR %d/%d, %d M3 event(s) in %.1fs",
            video_path, len(analysis.tracks), classifications, logic.violation_detections,
            len(violations), evidence_frames, anpr_matches, anpr_attempts, accepted, elapsed,
        )
        if job is not None:
            job.metrics.update(metrics)
            job.violations = violations
        return {"metrics": metrics, "violations": violations, "ingest_results": ingested,
                "analysis": analysis.summary(), "anpr": anpr_results}

    # ------------------------------------------------------------------ analysis helpers
    def _candidate_tracks(self, analysis: VideoAnalysis, cfg: AIConfig) -> List[TrackedObject]:
        """Rider candidates: persons (and bikes, when the provider labels them separately).

        Tracks with fewer observations than the confirmation rule can never confirm a
        violation, so they are dropped before any classifier/OCR work is done.
        """
        candidates: List[TrackedObject] = []
        for track in analysis.rider_tracks:
            if len(track.observations) < max(1, cfg.confirmation_frames):
                continue
            candidates.append(track)
        return candidates

    def _samples(self, track: TrackedObject, analysis: VideoAnalysis) -> List[Any]:
        """Which observations of this rider get classified (bounded, cost-aware)."""
        observations = sorted(track.observations, key=lambda o: o.timestamp)
        if analysis.helmet_status_known():
            return observations                       # local provider: already classified, no cost
        interval = max(0.0, float(self.config.google_classification_interval_s))
        limit = max(1, int(self.config.google_max_observations_per_track))
        chosen: List[Any] = []
        last_ts = None
        for obs in observations:
            if last_ts is not None and interval and (obs.timestamp - last_ts) < interval:
                continue
            chosen.append(obs)
            last_ts = obs.timestamp
            if len(chosen) >= limit:
                break
        return chosen

    def _detections_from_track(self, track: TrackedObject, samples: List[Any],
                               frame_size: Tuple[int, int], track_id: int) -> List[Detection]:
        out: List[Detection] = []
        for sample in samples:
            if sample.helmet_status not in HELMET_CLASSES:
                continue
            box = sample.pixels(frame_size[0], frame_size[1]) if frame_size != (0, 0) else sample.pixels(1, 1)
            out.append(Detection(
                class_name=sample.helmet_status,
                confidence=sample.helmet_confidence or sample.confidence,
                bbox=box,
                frame_number=sample.frame_number,
                timestamp=sample.timestamp,
                track_id=track_id,
            ))
        return out

    def _observe(self, logic: HelmetViolationLogic, detections: List[Detection],
                 frame_size: Tuple[int, int]) -> None:
        """Feed one frame's detections to the shared Kaaval logic (per timestamp, in order).

        The Kaaval logic is the same instance type `/ai/helmet/process` uses, so the
        confirmation rule and the unique-rider registry behave identically on both paths.
        """
        groups: Dict[Tuple[int, float], List[Detection]] = {}
        for det in detections:
            groups.setdefault((det.frame_number, round(det.timestamp, 3)), []).append(det)
        for (frame_number, timestamp), group in sorted(groups.items(), key=lambda kv: kv[0][1]):
            logic.observe(group, frame_number, timestamp,
                          frame_size=frame_size if frame_size != (0, 0) else None)

    def _frame_size(self, video_path: str, analysis: VideoAnalysis) -> Tuple[int, int]:
        try:
            with FrameExtractor(video_path) as extractor:
                return extractor.size
        except Exception:
            return (0, 0)

    def _video_duration(self, video_path: str, analysis: VideoAnalysis, video_fps: float) -> float:
        if analysis.duration_s:
            return analysis.duration_s
        try:
            with FrameExtractor(video_path) as extractor:
                return extractor.frame_count / extractor.fps if extractor.fps else 0.0
        except Exception:
            return 0.0

    def _frame_for(self, extractor: Optional[FrameExtractor], sample: Any,
                   video_fps: float) -> Tuple[Optional[np.ndarray], int]:
        """Decode the frame of one observation (by index when known, else by timestamp)."""
        if extractor is None:
            return None, -1
        try:
            if sample.frame_number is not None and sample.frame_number >= 0:
                return extractor.frame(int(sample.frame_number))
            if video_fps > 0:
                return extractor.frame_at_time(float(sample.timestamp))
            return extractor.frame(0)
        except FrameExtractionError as e:
            log.warning("frame extraction failed at t=%.2fs: %s", sample.timestamp, e)
            return None, -1

    # ------------------------------------------------------------------ evidence / ANPR
    def _write_evidence(self, video_path: str, violation: Any, cfg: AIConfig) -> Optional[str]:
        """Render the evidence frame for a confirmed rider from the original video."""
        frame_number = violation.frame_number
        try:
            with FrameExtractor(video_path) as extractor:
                if frame_number is None or frame_number < 0:
                    frame, frame_number = extractor.frame_at_time(violation.timestamp)
                else:
                    frame, frame_number = extractor.frame(int(frame_number))
                detections = [Detection(class_name=violation.detection.class_name,
                                        confidence=violation.confidence,
                                        bbox=tuple(violation.detection.bbox),
                                        frame_number=frame_number,
                                        timestamp=violation.timestamp,
                                        track_id=violation.track_id)]
                return save_evidence_frame(frame, detections, violation, cfg,
                                           model_name=self.config.model_name)
        except Exception as e:
            log.error("evidence frame failed for %s: %s", violation.event_id, e)
            return None

    def _run_anpr(self, anpr: Any, violation: Any, evidence_uri: str) -> AnprResult:
        """ONE OCR attempt for one confirmed violator (the evidence frame is the input)."""
        context = {
            "event_id": violation.event_id,
            "camera_id": violation.camera_id,
            "timestamp": violation.timestamp,
            "roi": tuple(violation.detection.bbox),
        }
        try:
            result = anpr.recognize(evidence_uri, context=context)
        except Exception as e:                                # OCR must never kill an event
            log.warning("ANPR failed for %s: %s", violation.event_id, e)
            result = AnprResult(status="ERROR", provider=getattr(anpr, "name", ""), notes=str(e))
        log.info("ANPR %s for %s: %s", result.status, violation.event_id, result.plate_number or "-")
        return result

    # ------------------------------------------------------------------ emission
    def _emit_violation(self, violation: Any, analysis: VideoAnalysis, classifier: Any,
                        anpr_result: Optional[AnprResult], cfg: AIConfig) -> Any:
        payload = build_kaaval_event_payload(
            violation,
            cfg,
            inference_fps=self._payload_fps(analysis),
            latency_ms=None,
            evidence_uri=violation.evidence_uri,
            analytics_provider=analysis.provider,
            analytics_model_name=analysis.model_name,
            analytics_model_version=analysis.model_version,
            helmet_classifier=classifier.name,
            anpr=anpr_result,
        )
        log.info("sending %s to M3 ingestion (camera=%s, track=%s, analysis=%s, plate=%s, evidence=%s)",
                 violation.event_id, violation.camera_id, violation.track_id, analysis.provider,
                 (anpr_result.plate_number if anpr_result else None), violation.evidence_uri)
        result = self.ingest_fn(payload)
        log.info("M3 ingestion result for %s: %s (canonical=%s)", violation.event_id,
                 getattr(result, "status", "UNKNOWN"), getattr(result, "canonical_event_id", None))
        return result

    @staticmethod
    def _payload_fps(analysis: VideoAnalysis) -> float:
        elapsed = float((analysis.extra or {}).get("analysis_time_s") or 0.0)
        return round(analysis.frames_analyzed / elapsed, 2) if elapsed > 0 else 0.0

    # ------------------------------------------------------------------ progress
    def _progress_callback(self, job: Optional[Any]) -> Optional[Callable[[Dict[str, Any]], None]]:
        if job is None:
            return None

        def _report(info: Dict[str, Any]) -> None:
            job.metrics.update({
                "stage": info.get("stage", "video_analysis"),
                "analysis_provider": info.get("analysis_provider"),
                "progress_percent": info.get("progress_percent"),
                "frames_processed": info.get("frames_processed"),
                "frames_decoded": info.get("frames_decoded"),
                "total_frames": info.get("total_frames"),
                "tracks_seen": info.get("tracks"),
            })

        return _report

    def _partial_metrics(self, analysis: VideoAnalysis, track_index: int, total_tracks: int,
                         classifications: int, logic: HelmetViolationLogic, anpr_attempts: int,
                         anpr_matches: int, started: float) -> Dict[str, Any]:
        elapsed = time.perf_counter() - started
        return {
            "stage": "helmet_classification",
            "analysis_provider": analysis.provider,
            "progress_percent": round(100.0 * track_index / total_tracks, 1) if total_tracks else None,
            "tracks_seen": len(analysis.tracks),
            "rider_tracks": total_tracks,
            "tracks_processed": track_index,
            "helmet_classifications": classifications,
            "violation_detections": logic.violation_detections,
            "unique_violators": logic.unique_violators,
            "progress_anpr_attempts": anpr_attempts,
            "progress_anpr_matches": anpr_matches,
            "processing_time": round(elapsed, 2),
        }
