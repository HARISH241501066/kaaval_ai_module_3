"""
KAAVAL AI — local YOLO helmet detection API.

Adds four endpoints to the EXISTING FastAPI app (no second application, no job queue):

    POST /ai/helmet/process      start a background video job -> {job_id, status: QUEUED}
    GET  /ai/helmet/health       independent YOLO / KAAVAL AI / pipeline health
    GET  /ai/helmet/config       effective AI configuration + model card (no secrets)
    GET  /ai/helmet/jobs[/{id}]  job status and metrics (QUEUED/RUNNING/COMPLETED/FAILED)

Events produced by a job enter the existing C4 ingestion pipeline; this router never
touches the database or Module 4 directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.ai.config import get_ai_config
from app.ai.helmet_detector import detector_status
from app.ai.jobs import get_job_registry
from app.ai.schemas import JobStatus
from app.ai.topology import ensure_ai_video_topology

log = logging.getLogger("sentinel.m3.api.ai")

router = APIRouter(prefix="/ai/helmet", tags=["KAAVAL AI — Local Helmet Detection"])

# Evidence frames are local JPEGs; only these extensions are ever served, and only from
# inside AI_EVIDENCE_DIR (see the /evidence endpoint).
EVIDENCE_SUFFIXES = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------- schemas
class HelmetProcessRequest(BaseModel):
    camera_id: Optional[str] = Field(None, example="CAM-001",
                                     description="Camera from AI_VIDEO_SOURCES (video is taken from the map).")
    video_path: Optional[str] = Field(None, example="camera_feed/camera_main.mp4",
                                      description="Explicit local .mp4; overrides the configured path for camera_id.")

    class Config:
        json_schema_extra = {
            "example": {"camera_id": "CAM-001"}
        }


class HelmetProcessResponse(BaseModel):
    job_id: str = Field(..., example="job-8f2c1a9b0d31")
    camera_id: str = Field(..., example="CAM-001")
    video_path: str = Field(..., example="camera_feed/camera_main.mp4")
    status: str = Field(..., example="STARTED",
                        description="Job accepted and started; poll /ai/helmet/jobs/{job_id} "
                                    "for QUEUED | RUNNING | COMPLETED | FAILED.")
    message: str = Field(..., example="Video accepted for background processing.")


# -------------------------------------------------------------------- endpoints
@router.post("/process", response_model=HelmetProcessResponse, status_code=202,
             summary="Process a road video with the local YOLO helmet detector (non-blocking)")
def process_video(request: HelmetProcessRequest):
    """
    Starts a background job:

    **video -> YOLO detections -> Kaaval AI decision logic -> HELMET_VIOLATION
    -> existing M3 ingestion -> canonical envelope -> SQLite -> Mock M4**

    Returns immediately with a `job_id`; poll `GET /ai/helmet/jobs/{job_id}` for
    frames_processed / detections / violations / processing_time.
    """
    cfg = get_ai_config()
    try:
        camera_id, video_path = cfg.resolve_video(request.camera_id, request.video_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not str(video_path).lower().startswith("rtsp://") and not Path(video_path).is_file():
        raise HTTPException(
            status_code=400,
            detail=f"video file not found: '{video_path}' (check AI_VIDEO_SOURCES / camera_id)",
        )

    try:
        ensure_ai_video_topology(config=cfg)      # camera + node + assignment rows exist first
    except Exception as e:                        # topology problems must not block the job
        log.warning("AI topology bootstrap failed (continuing): %s", e)

    registry = get_job_registry()
    job = registry.create(camera_id=camera_id, video_path=str(video_path))
    registry.start(job)
    return HelmetProcessResponse(
        job_id=job.job_id,
        camera_id=camera_id,
        video_path=str(video_path),
        status="STARTED",
        message="Video accepted for background processing.",
    )


@router.get("/health", summary="Independent health: YOLO model / KAAVAL AI / event pipeline / cameras")
def health():
    """
    Health layers are independent by design: a FAILED YOLO model must never mark the
    physical camera OFFLINE. Reports:

    * `yolo_model`  — READY / FAILED / NOT_LOADED (loads the model on first call)
    * `kaaval_ai`   — the decision layer (ONLINE when importable)
    * `event_pipeline` — the existing C4 pipeline singleton
    * `cameras`     — per configured camera: stored camera health + video file presence
    * `jobs`        — job counts by status
    """
    cfg = get_ai_config()
    yolo = detector_status(record=True, config=cfg)

    from app.ai.video_processor import ingest_to_m3  # noqa: F401  (importability check)
    from app.services.pipeline import get_pipeline

    cameras = []
    try:
        from app.db.database import SessionLocal
        from app.db.repositories import CameraRepository, HealthRepository

        with SessionLocal() as db:
            cam_repo, health_repo = CameraRepository(db), HealthRepository(db)
            health_rows = {r.entity_id: r for r in health_repo.list_all() if r.entity_type == "CAMERA"}
            for cam_id, path in cfg.video_sources.items():
                row = cam_repo.get(cam_id)
                cameras.append({
                    "camera_id": cam_id,
                    "video_path": path,
                    "video_present": Path(path).is_file(),
                    "camera_status": (health_rows.get(cam_id).status if cam_id in health_rows
                                      else (row.status if row else "NOT_REGISTERED")),
                    "processing_node_id": cfg.processing_node_id,
                })
    except Exception as e:  # DB trouble -> report, don't 500
        log.warning("camera health lookup failed: %s", e)

    return {
        "kaaval_ai": "ONLINE",
        "yolo_model": yolo,
        "event_pipeline": "ONLINE" if get_pipeline() is not None else "FAILED",
        "camera_health_source": "M3 C6 (independent of YOLO model state)",
        "cameras": cameras,
        "jobs": get_job_registry().summary(),
    }


@router.get("/evidence", summary="Serve one evidence frame (only files inside AI_EVIDENCE_DIR)")
def evidence(uri: str = Query(..., description="evidence_ref.uri from a confirmed violation")):
    """
    Returns the annotated JPEG written when a violation was confirmed, so the console can
    show the frame the event came from.

    Path safety is explicit: the request is resolved against the project root and must land
    **inside** the configured evidence directory, with a JPEG/PNG extension. Anything else is
    a 404, so this endpoint cannot read arbitrary files off the host.
    """
    cfg = get_ai_config()
    root = Path(cfg.evidence_dir).resolve()
    candidate = Path(uri)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate        # evidence uris are written relative to the cwd
    target = candidate.resolve()
    if (not target.is_relative_to(root)
            or target.suffix.lower() not in EVIDENCE_SUFFIXES
            or not target.is_file()):
        raise HTTPException(status_code=404, detail="evidence frame not found")
    return FileResponse(target, media_type="image/jpeg")


@router.get("/config", summary="Effective AI configuration and model card (no secrets)")
def config():
    """Everything the inference layer is configured with: model path/card, thresholds,
    inference FPS, image size, device, tracking, cooldown, video->camera map."""
    return get_ai_config().as_public_dict()


@router.get("/jobs", summary="List recent processing jobs")
def list_jobs(limit: int = 20):
    return {"count": len(get_job_registry().list(limit=limit)),
            "jobs": [j.to_dict() for j in get_job_registry().list(limit=limit)]}


@router.get("/jobs/{job_id}", summary="Job status and metrics")
def get_job(job_id: str):
    job = get_job_registry().get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"job '{job_id}' not found")
    return job.to_dict()
