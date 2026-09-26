"""
In-memory background job registry for video processing (hackathon prototype).

POST /ai/helmet/process returns immediately with a job_id; a daemon thread runs
VideoProcessor -> existing M3 pipeline. No queue/broker/microservice — one process,
a dict of jobs and a thread, which is safe for this FastAPI app (no event loop is
blocked: each job runs in its own thread and its own asyncio loop per ingest call).
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, List, Optional

from app.ai.config import get_ai_config
from app.ai.schemas import Job, JobStatus

log = logging.getLogger("sentinel.m3.ai.jobs")


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ CRUD
    def create(self, camera_id: str, video_path: str, mode: str = "helmet") -> Job:
        job = Job(
            job_id=f"job-{uuid.uuid4().hex[:12]}",
            camera_id=camera_id,
            video_path=video_path,
            mode=mode,
            created_at=time.time(),
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> List[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def summary(self) -> Dict[str, int]:
        with self._lock:
            counts = {"total": len(self._jobs)}
        for status in JobStatus:
            counts[status.value.lower()] = 0
        for job in self.list(limit=len(self._jobs) or 1):
            counts[job.status.value.lower()] = counts.get(job.status.value.lower(), 0) + 1
        return counts

    # ----------------------------------------------------------------- run
    def start(self, job: Job) -> Job:
        """Kick the job onto a daemon thread; returns immediately (HTTP does not block)."""
        thread = threading.Thread(target=self.run_job, args=(job,), name=f"ai-job-{job.job_id}", daemon=True)
        thread.start()
        return job

    def run_job(self, job: Job) -> Job:
        """Synchronous runner (also called directly by tests)."""
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        try:
            from app.ai.topology import ensure_ai_video_topology

            ensure_ai_video_topology()          # camera/node rows exist before ingestion
            config = get_ai_config()
            if job.mode == "video":
                # provider-based path: Google Cloud Video Intelligence (or the local fallback)
                from app.ai.video_analysis import VideoAnalysisProcessor

                processor = VideoAnalysisProcessor(config=config)
            else:
                from app.ai.video_processor import VideoProcessor

                processor = VideoProcessor(config=config)
            outcome = processor.process(job.camera_id, job.video_path, job=job)
            job.metrics = outcome["metrics"]
            job.violations = outcome["violations"]
            job.status = JobStatus.COMPLETED
            log.info("job %s COMPLETED: %d violation event(s)", job.job_id, len(job.violations))
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            # A failed video is NEVER reported as "0 violations" (no video -> no result).
            job.metrics = {"outcome": "PROCESSING_FAILED", "error": str(e)}
            log.error("job %s FAILED: %s", job.job_id, e, exc_info=True)
        finally:
            job.finished_at = time.time()
        return job


_registry = JobRegistry()


def get_job_registry() -> JobRegistry:
    return _registry
