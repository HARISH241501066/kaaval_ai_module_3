"""
Cloud video analysis API — the provider-based counterpart of `/ai/helmet/*`.

    POST /ai/video/process        start a background analysis job  -> {job_id, status: STARTED}
    GET  /ai/video/jobs[/{id}]    job status, live progress and the final result block
    GET  /ai/video/health         independent health: Video Intelligence / helmet classifier /
                                  ANPR / M3 pipeline / cameras
    GET  /ai/video/config         effective provider routing + Google settings (no secrets)

The flow these endpoints trigger is the ONE M3 pipeline (no second event path):

    video -> analysis provider (Cloud Video Intelligence) -> tracked riders
          -> helmet classifier (local YOLO) -> Kaaval decision logic -> unique violators
          -> evidence frame -> bounded Cloud Vision OCR -> Kaaval event
          -> EXISTING C4 ingestion -> canonical envelope -> SQLite -> Mock M4

Requests never block on the analysis: the job runs on a daemon thread and the caller polls
`/ai/video/jobs/{job_id}`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ai.config import get_ai_config
from app.ai.jobs import get_job_registry
from app.ai.providers import provider_status
from app.ai.topology import ensure_ai_video_topology

log = logging.getLogger("sentinel.m3.api.video")

router = APIRouter(prefix="/ai/video", tags=["Cloud Video Analysis — Google Cloud / local"])


# --------------------------------------------------------------------- schemas
class VideoProcessRequest(BaseModel):
    camera_id: Optional[str] = Field(None, example="CAM-001",
                                     description="Camera from AI_VIDEO_SOURCES (its video is used).")
    video_path: Optional[str] = Field(None, example="camera_feed/camera_main.mp4",
                                      description="Local .mp4 path, or a gs:// Cloud Storage URI for "
                                                  "large videos (AI_GOOGLE_VIDEO_INPUT_MODE=gcs).")

    class Config:
        json_schema_extra = {"example": {"camera_id": "CAM-001", "video_path": "camera_feed/camera_main.mp4"}}


class VideoProcessResponse(BaseModel):
    job_id: str = Field(..., example="job-8f2c1a9b0d31")
    camera_id: str = Field(..., example="CAM-001")
    video_path: str = Field(..., example="camera_feed/camera_main.mp4")
    status: str = Field(..., example="STARTED")
    analysis_provider: str = Field(..., example="google",
                                   description="Requested analysis backend. The job reports "
                                               "analysis_effective + analysis_fallback_reason when "
                                               "configuration forces the local fallback.")
    message: str = Field(..., example="Video accepted for background cloud analysis.")


# -------------------------------------------------------------------- endpoints
@router.post("/process", response_model=VideoProcessResponse, status_code=202,
             summary="Analyze a road video with Google Cloud Video Intelligence (non-blocking)")
def process_video(request: VideoProcessRequest):
    """
    Starts a background job:

    **video → Cloud Video Intelligence (person/object tracking) → tracked riders →
    local helmet classifier → Kaaval logic → unique violators → evidence frame →
    Cloud Vision OCR (one call per violator) → existing M3 ingestion → canonical → Mock M4**

    Returns immediately with a `job_id`; poll `GET /ai/video/jobs/{job_id}` for
    `progress_percent`, `unique_violators`, `violation_events`, `anpr_attempts` and
    `anpr_matches`.
    """
    cfg = get_ai_config()
    try:
        camera_id, video_path = cfg.resolve_video(request.camera_id, request.video_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    is_gcs = str(video_path).lower().startswith("gs://")
    if not is_gcs and not Path(video_path).is_file():
        raise HTTPException(
            status_code=400,
            detail=f"video file not found: '{video_path}' (check AI_VIDEO_SOURCES / camera_id)",
        )

    try:
        ensure_ai_video_topology(config=cfg)      # camera + node rows must exist for provenance
    except Exception as e:                        # topology problems must not block the job
        log.warning("AI topology bootstrap failed (continuing): %s", e)

    registry = get_job_registry()
    job = registry.create(camera_id=camera_id, video_path=str(video_path), mode="video")
    registry.start(job)
    return VideoProcessResponse(
        job_id=job.job_id,
        camera_id=camera_id,
        video_path=str(video_path),
        status="STARTED",
        analysis_provider=cfg.video_provider,
        message=("Video accepted for background cloud analysis. Google Cloud Video Intelligence "
                 "detects/tracks riders; helmet status comes from the local helmet classifier "
                 "when the configured backend cannot provide it."),
    )


@router.get("/jobs", summary="List recent video-analysis jobs")
def list_jobs(limit: int = 20):
    jobs = [j for j in get_job_registry().list(limit=limit * 2) if j.mode == "video"][:limit]
    return {"count": len(jobs), "jobs": [j.to_dict() for j in jobs]}


@router.get("/jobs/{job_id}", summary="Video-analysis job status, progress and result")
def get_job(job_id: str):
    job = get_job_registry().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return job.to_dict()


@router.get("/health", summary="Independent health: Google video analysis / classifier / ANPR / M3")
def health():
    """
    Every layer reports on its own, and a Google problem never marks a camera offline:

    * `video_intelligence` — AVAILABLE / NOT_CONFIGURED / ERROR (+ whether the SDK is installed,
      where credentials come from, and the API deprecation note)
    * `local_yolo_provider` / `helmet_classifier` — READY / FAILED (local weights)
    * `anpr` — AVAILABLE / NOT_CONFIGURED / ERROR (Cloud Vision OCR)
    * `analysis` — requested vs effective backend and the fallback policy
    """
    cfg = get_ai_config()
    status = provider_status(cfg)
    cameras = []
    try:
        from app.db.database import SessionLocal
        from app.db.repositories import CameraRepository, HealthRepository

        with SessionLocal() as db:
            cam_repo, health_repo = CameraRepository(db), HealthRepository(db)
            rows = {r.entity_id: r for r in health_repo.list_all() if r.entity_type == "CAMERA"}
            for cam_id, path in cfg.video_sources.items():
                row = cam_repo.get(cam_id)
                cameras.append({
                    "camera_id": cam_id,
                    "video_path": path,
                    "video_present": Path(path).is_file(),
                    "camera_status": (rows.get(cam_id).status if cam_id in rows
                                      else (row.status if row else "NOT_REGISTERED")),
                })
    except Exception as e:                        # DB trouble -> report, don't 500
        log.warning("camera health lookup failed: %s", e)

    from app.services.pipeline import get_pipeline

    return {
        **status,
        "event_pipeline": "ONLINE" if get_pipeline() is not None else "FAILED",
        "camera_health_source": "M3 C6 (independent of every AI provider)",
        "cameras": cameras,
        "jobs": get_job_registry().summary(),
    }


@router.get("/config", summary="Effective provider routing + Google settings (no secrets)")
def config():
    """What the analysis layer is configured with. Credentials are never echoed — only their
    *source* and whether the file exists."""
    cfg = get_ai_config()
    return {
        "provider": cfg.video_provider,
        "allow_local_fallback": cfg.allow_local_fallback,
        "helmet_provider": cfg.helmet_provider,
        "anpr_provider": cfg.anpr_provider if cfg.anpr_enabled else "none",
        "helmet_confidence_threshold": cfg.helmet_confidence_threshold,
        "confirmation_observations": cfg.confirmation_observations,
        "video": {
            "sources": dict(cfg.video_sources),
            "frame_skip": cfg.frame_skip,
            "sampling_mode": cfg.sampling_mode,
        },
        "google": cfg.as_public_dict()["google"],
        "classifier": {
            "model_name": cfg.model_name,
            "model_version": cfg.model_version,
            "model_present": cfg.model_file_exists,
            "min_confidence": cfg.helmet_min_confidence,
        },
        "cost_controls": {
            "max_observations_per_track": cfg.google_max_observations_per_track,
            "classification_interval_s": cfg.google_classification_interval_s,
            "max_classifications": cfg.google_max_classifications,
            "anpr_max_per_job": cfg.anpr_max_per_job,
            "video_api_calls_per_job": 1,
        },
    }
