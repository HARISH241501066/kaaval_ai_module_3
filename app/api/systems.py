"""
Systems, Nodes, Cameras, and Assignments Router (C1, C2, C3).
Handles inventory federation and failure-impact relationships:
Camera -> Processing Node -> AI Model.
"""
from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import (
    SystemRepository,
    NodeRepository,
    CameraRepository,
    AssignmentRepository,
)
from app.schemas.system import (
    SystemCreate,
    SystemRead,
    ProcessingNodeCreate,
    ProcessingNodeRead,
    CameraCreate,
    CameraRead,
    StreamSourceRead,
    CameraAssignmentCreate,
    CameraAssignmentRead,
    NodeWithCamerasRead,
)

router = APIRouter(prefix="/api/v1", tags=["Inventory & Topology (C1, C2, C3)"])


# --- Systems ---
@router.get("/systems", response_model=List[SystemRead])
def list_systems(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all registered upstream systems (VMS, NVR, Kaaval fleet)."""
    return SystemRepository(db).list(skip=skip, limit=limit)


@router.post("/systems", response_model=SystemRead, status_code=status.HTTP_201_CREATED)
def create_system(system: SystemCreate, db: Session = Depends(get_db)):
    """Register a new upstream VMS, NVR, or Kaaval system."""
    repo = SystemRepository(db)
    if repo.get(system.id):
        raise HTTPException(status_code=400, detail=f"System with ID '{system.id}' already exists")
    return repo.create(system)


# --- Processing Nodes ---
@router.get("/nodes", response_model=List[ProcessingNodeRead])
def list_nodes(db: Session = Depends(get_db)):
    """List all AI processing nodes (RDK-X5 edge boxes and central GPU servers)."""
    return NodeRepository(db).list()


@router.post("/nodes", response_model=ProcessingNodeRead, status_code=status.HTTP_201_CREATED)
def create_node(node: ProcessingNodeCreate, db: Session = Depends(get_db)):
    """Register an AI processing node."""
    repo = NodeRepository(db)
    if repo.get(node.id):
        raise HTTPException(status_code=400, detail=f"Node with ID '{node.id}' already exists")
    return repo.create(node)


@router.get("/nodes/{node_id}/cameras", response_model=NodeWithCamerasRead)
def get_node_failure_impact(node_id: str, db: Session = Depends(get_db)):
    """
    Failure Impact Analysis:
    Answers 'If this processing node fails, which cameras are affected?'
    """
    node_repo = NodeRepository(db)
    node = node_repo.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    assigned_cameras = node_repo.get_assigned_cameras(node_id)
    return NodeWithCamerasRead(node=node, assigned_cameras=assigned_cameras)


# --- Cameras ---
@router.get("/cameras", response_model=List[CameraRead])
def list_cameras(system_id: Optional[str] = Query(None), db: Session = Depends(get_db)):
    """List registered cameras, optionally filtered by system."""
    return CameraRepository(db).list(system_id=system_id)


@router.post("/cameras", response_model=CameraRead, status_code=status.HTTP_201_CREATED)
def create_camera(camera: CameraCreate, db: Session = Depends(get_db)):
    """Register a camera in the federation layer."""
    repo = CameraRepository(db)
    if repo.get(camera.id):
        raise HTTPException(status_code=400, detail=f"Camera with ID '{camera.id}' already exists")
    return repo.create(camera)


@router.get("/cameras/{camera_id}/stream-source", response_model=StreamSourceRead)
def get_stream_source(camera_id: str, db: Session = Depends(get_db)):
    """
    C3 Stream Source Broker:
    Returns internal relay URL for Module 2 or AI workers without exposing upstream vendor credentials.
    """
    repo = CameraRepository(db)
    cam = repo.get(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found")

    # Relay URL abstraction hides vendor RTSP credentials
    relay_url = f"rtsp://sentinel-m3-relay:8554/live/{cam.id}/sub"
    return StreamSourceRead(
        camera_id=cam.id,
        relay_url=relay_url,
        codec="H264",
        width=1920,
        height=1080,
        fps=cam.fps,
        status="AVAILABLE",
    )


# --- Camera Assignments ---
@router.get("/assignments", response_model=List[CameraAssignmentRead])
def list_assignments(db: Session = Depends(get_db)):
    """List all active camera-to-node analytics assignments."""
    return AssignmentRepository(db).list()


@router.post("/assignments", response_model=CameraAssignmentRead, status_code=status.HTTP_201_CREATED)
def create_assignment(assignment: CameraAssignmentCreate, db: Session = Depends(get_db)):
    """Assign an AI processing node to analyze a specific camera stream."""
    cam_repo = CameraRepository(db)
    node_repo = NodeRepository(db)
    if not cam_repo.get(assignment.camera_id):
        raise HTTPException(status_code=404, detail=f"Camera '{assignment.camera_id}' not found")
    if not node_repo.get(assignment.node_id):
        raise HTTPException(status_code=404, detail=f"Node '{assignment.node_id}' not found")

    return AssignmentRepository(db).create(assignment)
