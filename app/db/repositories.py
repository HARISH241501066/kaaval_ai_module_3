"""
Repository abstractions for clean database access.
Supports SQLite and PostgreSQL through SQLAlchemy 2.0 sessions.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, func

from app.models.system import SystemModel, ProcessingNodeModel, CameraModel, CameraAssignmentModel
from app.models.event import CanonicalEventRecord
from app.models.health import HealthRecordModel
from app.schemas.system import SystemCreate, ProcessingNodeCreate, CameraCreate, CameraAssignmentCreate


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SystemRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, system_id: str) -> Optional[SystemModel]:
        return self.db.execute(select(SystemModel).where(SystemModel.id == system_id)).scalar_one_or_none()

    def list(self, skip: int = 0, limit: int = 100) -> List[SystemModel]:
        return list(self.db.execute(select(SystemModel).offset(skip).limit(limit)).scalars().all())

    def create(self, system: SystemCreate) -> SystemModel:
        db_obj = SystemModel(
            id=system.id,
            name=system.name,
            vendor=system.vendor.value if hasattr(system.vendor, "value") else str(system.vendor),
            protocol=system.protocol.value if hasattr(system.protocol, "value") else str(system.protocol),
            connection_params=system.connection_params,
            status=system.status,
        )
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj


class NodeRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, node_id: str) -> Optional[ProcessingNodeModel]:
        return self.db.execute(select(ProcessingNodeModel).where(ProcessingNodeModel.id == node_id)).scalar_one_or_none()

    def list(self) -> List[ProcessingNodeModel]:
        return list(self.db.execute(select(ProcessingNodeModel)).scalars().all())

    def create(self, node: ProcessingNodeCreate) -> ProcessingNodeModel:
        db_obj = ProcessingNodeModel(
            id=node.id,
            name=node.name,
            node_type=node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
            system_id=node.system_id,
            ip_address=node.ip_address,
            status=node.status,
            metrics=node.metrics,
            last_heartbeat=utcnow(),
        )
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

    def update_heartbeat(self, node_id: str, metrics: Optional[dict] = None, status: str = "ONLINE") -> Optional[ProcessingNodeModel]:
        node = self.get(node_id)
        if node:
            node.last_heartbeat = utcnow()
            node.status = status
            if metrics:
                node.metrics = {**(node.metrics or {}), **metrics}
            self.db.commit()
            self.db.refresh(node)
        return node

    def get_assigned_cameras(self, node_id: str) -> List[CameraModel]:
        stmt = (
            select(CameraModel)
            .join(CameraAssignmentModel, CameraModel.id == CameraAssignmentModel.camera_id)
            .where(and_(CameraAssignmentModel.node_id == node_id, CameraAssignmentModel.status == "ACTIVE"))
        )
        return list(self.db.execute(stmt).scalars().all())


class CameraRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, camera_id: str) -> Optional[CameraModel]:
        return self.db.execute(select(CameraModel).where(CameraModel.id == camera_id)).scalar_one_or_none()

    def get_by_external_id(self, external_id: str) -> Optional[CameraModel]:
        return self.db.execute(select(CameraModel).where(CameraModel.external_id == external_id)).scalar_one_or_none()

    def resolve_camera(self, cam_identifier: str) -> Optional[CameraModel]:
        """Resolves camera by canonical ID first, then by external ID."""
        cam = self.get(cam_identifier)
        if not cam:
            cam = self.get_by_external_id(cam_identifier)
        return cam

    def list(self, system_id: Optional[str] = None) -> List[CameraModel]:
        stmt = select(CameraModel)
        if system_id:
            stmt = stmt.where(CameraModel.system_id == system_id)
        return list(self.db.execute(stmt).scalars().all())

    def create(self, camera: CameraCreate) -> CameraModel:
        db_obj = CameraModel(
            id=camera.id,
            external_id=camera.external_id or camera.id,
            name=camera.name,
            system_id=camera.system_id,
            rtsp_url=camera.rtsp_url,
            location_name=camera.location_name,
            latitude=camera.latitude,
            longitude=camera.longitude,
            status=camera.status,
            resolution=camera.resolution,
            fps=camera.fps,
        )
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

    def get_assigned_node(self, camera_id: str) -> Optional[ProcessingNodeModel]:
        stmt = (
            select(ProcessingNodeModel)
            .join(CameraAssignmentModel, ProcessingNodeModel.id == CameraAssignmentModel.node_id)
            .where(and_(CameraAssignmentModel.camera_id == camera_id, CameraAssignmentModel.status == "ACTIVE"))
        )
        return self.db.execute(stmt).scalar_one_or_none()


class AssignmentRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, assignment: CameraAssignmentCreate) -> CameraAssignmentModel:
        db_obj = CameraAssignmentModel(
            camera_id=assignment.camera_id,
            node_id=assignment.node_id,
            pipeline_profile=assignment.pipeline_profile,
            status=assignment.status,
            assigned_at=utcnow(),
        )
        self.db.add(db_obj)
        self.db.commit()
        self.db.refresh(db_obj)
        return db_obj

    def list(self) -> List[CameraAssignmentModel]:
        return list(self.db.execute(select(CameraAssignmentModel)).scalars().all())


class EventRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, record: CanonicalEventRecord) -> CanonicalEventRecord:
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_by_event_id(self, event_id: str) -> Optional[CanonicalEventRecord]:
        return self.db.execute(
            select(CanonicalEventRecord).where(CanonicalEventRecord.event_id == event_id)
        ).scalar_one_or_none()

    def find_duplicate(self, provider: str, source_event_id: str, window_s: int = 3600) -> Optional[CanonicalEventRecord]:
        cutoff = utcnow() - timedelta(seconds=window_s)
        stmt = select(CanonicalEventRecord).where(
            and_(
                CanonicalEventRecord.provider == provider,
                CanonicalEventRecord.source_event_id == source_event_id,
                CanonicalEventRecord.created_at >= cutoff,
            )
        )
        return self.db.execute(stmt).scalar_one_or_none()

    def list(
        self,
        camera_id: Optional[str] = None,
        node_id: Optional[str] = None,
        event_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[CanonicalEventRecord]:
        stmt = select(CanonicalEventRecord)
        if camera_id:
            stmt = stmt.where(CanonicalEventRecord.camera_id == camera_id)
        if node_id:
            stmt = stmt.where(CanonicalEventRecord.processing_node_id == node_id)
        if event_type:
            stmt = stmt.where(CanonicalEventRecord.event_type == event_type)
        stmt = stmt.order_by(CanonicalEventRecord.timestamp.desc()).offset(skip).limit(limit)
        return list(self.db.execute(stmt).scalars().all())

    def count(self) -> int:
        return self.db.execute(select(func.count(CanonicalEventRecord.id))).scalar_one() or 0


class HealthRepository:
    def __init__(self, db: Session):
        self.db = db

    def upsert(self, entity_id: str, entity_type: str, status: str, details: Optional[dict] = None) -> HealthRecordModel:
        record = self.db.execute(
            select(HealthRecordModel).where(
                and_(HealthRecordModel.entity_id == entity_id, HealthRecordModel.entity_type == entity_type)
            )
        ).scalar_one_or_none()

        if record:
            record.status = status
            record.details = details or {}
            record.last_observed = utcnow()
        else:
            record = HealthRecordModel(
                entity_id=entity_id,
                entity_type=entity_type,
                status=status,
                details=details or {},
                last_observed=utcnow(),
            )
            self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def list_all(self) -> List[HealthRecordModel]:
        return list(self.db.execute(select(HealthRecordModel).order_by(HealthRecordModel.entity_type, HealthRecordModel.entity_id)).scalars().all())
