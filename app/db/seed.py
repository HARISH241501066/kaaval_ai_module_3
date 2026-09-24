"""
Initial Database Topology Seeder for Sentinel Gujarat CCTV.
Populates standard systems, processing nodes, cameras, and assignments on fresh startup.
"""
from __future__ import annotations

import logging
from sqlalchemy.orm import Session

from app.db.repositories import (
    SystemRepository,
    NodeRepository,
    CameraRepository,
    AssignmentRepository,
    HealthRepository,
)
from app.schemas.system import (
    SystemCreate,
    ProcessingNodeCreate,
    CameraCreate,
    CameraAssignmentCreate,
)
from app.schemas.common import SystemVendor, SystemProtocol, NodeType

logger = logging.getLogger("sentinel.m3.seed")


def seed_initial_topology(db: Session) -> None:
    system_repo = SystemRepository(db)
    node_repo = NodeRepository(db)
    cam_repo = CameraRepository(db)
    assign_repo = AssignmentRepository(db)
    health_repo = HealthRepository(db)

    # 1. Systems
    systems = [
        SystemCreate(
            id="VMS-AHMEDABAD-HQ",
            name="Ahmedabad Smart City Central VMS",
            vendor=SystemVendor.HIKVISION,
            protocol=SystemProtocol.ONVIF,
            connection_params={"host": "10.0.1.10", "port": 80, "auth": "credential_ref://vault/vms-ahm"},
            status="ACTIVE",
        ),
        SystemCreate(
            id="KAAVAL-EDGE-SYSTEM",
            name="Kaaval AI RDK-X5 Edge Cluster",
            vendor=SystemVendor.KAAVAL,
            protocol=SystemProtocol.KAAVAL_EDGE,
            connection_params={"fleet": "gujarat-traffic-rdk", "auth": "hmac"},
            status="ACTIVE",
        ),
        SystemCreate(
            id="KAAVAL-SERVER-SYSTEM",
            name="Kaaval AI Central GPU Analytics Cluster",
            vendor=SystemVendor.KAAVAL,
            protocol=SystemProtocol.KAAVAL_SERVER,
            connection_params={"cluster_endpoint": "https://kaaval-gpu.gujarat.police:8443"},
            status="ACTIVE",
        ),
    ]

    for sys_data in systems:
        if not system_repo.get(sys_data.id):
            system_repo.create(sys_data)
            logger.info(f"Seeded System: {sys_data.id}")

    # 2. Processing Nodes
    nodes = [
        ProcessingNodeCreate(
            id="RDK-X5-AHM-01",
            name="RDK-X5 Edge Unit 01 (SG Highway)",
            node_type=NodeType.EDGE_RDK_X5,
            system_id="KAAVAL-EDGE-SYSTEM",
            ip_address="192.168.1.101",
            status="ONLINE",
            metrics={"cpu": 42.5, "mem": 58.2, "temp_c": 51.0, "fps": 15.2},
        ),
        ProcessingNodeCreate(
            id="RDK-X5-AHM-02",
            name="RDK-X5 Edge Unit 02 (Gandhinagar)",
            node_type=NodeType.EDGE_RDK_X5,
            system_id="KAAVAL-EDGE-SYSTEM",
            ip_address="192.168.1.102",
            status="ONLINE",
            metrics={"cpu": 39.0, "mem": 52.1, "temp_c": 49.5, "fps": 14.8},
        ),
        ProcessingNodeCreate(
            id="KAAVAL-SRV-GANDHINAGAR",
            name="Central GPU Server 01 (Gandhinagar Data Center)",
            node_type=NodeType.SERVER_GPU,
            system_id="KAAVAL-SERVER-SYSTEM",
            ip_address="10.0.12.50",
            status="ONLINE",
            metrics={"cpu": 65.0, "mem": 72.0, "temp_c": 62.0, "fps": 30.0},
        ),
    ]

    for node_data in nodes:
        if not node_repo.get(node_data.id):
            node_repo.create(node_data)
            logger.info(f"Seeded Node: {node_data.id}")
            # Initialize health records for node, AI runtime, and pipe
            health_repo.upsert(node_data.id, "PROCESSING_NODE", "ONLINE", node_data.metrics)
            health_repo.upsert(f"KAAVAL-AI-{node_data.id}", "AI_RUNTIME", "RUNNING", {"fps": node_data.metrics.get("fps", 15.0)})
            health_repo.upsert(f"PIPE-{node_data.id}", "EVENT_PIPE", "UP", {"status": "CONNECTED"})

    # 3. Cameras
    cameras = [
        CameraCreate(
            id="CAM-AHM-SG-001",
            external_id="CAM_01",
            name="SG Highway Junction 1 - North Face",
            system_id="VMS-AHMEDABAD-HQ",
            rtsp_url="rtsp://10.0.1.101:554/live/ch0",
            location_name="SG Highway, Bodakdev Junction, Ahmedabad",
            latitude=23.0378,
            longitude=72.5097,
            status="ONLINE",
            resolution="1920x1080",
            fps=25.0,
        ),
        CameraCreate(
            id="CAM-AHM-SG-002",
            external_id="CAM_02",
            name="SG Highway Junction 2 - South Face",
            system_id="VMS-AHMEDABAD-HQ",
            rtsp_url="rtsp://10.0.1.102:554/live/ch0",
            location_name="SG Highway, Iscon Junction, Ahmedabad",
            latitude=23.0270,
            longitude=72.5085,
            status="ONLINE",
            resolution="1920x1080",
            fps=25.0,
        ),
        CameraCreate(
            id="CAM-GNR-SECT-010",
            external_id="CAM_03",
            name="Sector 10 Circle - East Face",
            system_id="VMS-AHMEDABAD-HQ",
            rtsp_url="rtsp://10.0.2.103:554/live/ch0",
            location_name="Sector 10 Circle, Gandhinagar",
            latitude=23.2156,
            longitude=72.6369,
            status="ONLINE",
            resolution="1920x1080",
            fps=25.0,
        ),
        CameraCreate(
            id="CAM-LEGACY-DVR-04",
            external_id="CH-07",
            name="Delhi Darwaja Gate (Legacy DVR Ch 7)",
            system_id="VMS-AHMEDABAD-HQ",
            rtsp_url="rtsp://10.0.3.50:554/ch7",
            location_name="Delhi Darwaja, Old City, Ahmedabad",
            latitude=23.0333,
            longitude=72.5850,
            status="ONLINE",
            resolution="1280x720",
            fps=15.0,
        ),
    ]

    for cam_data in cameras:
        if not cam_repo.get(cam_data.id):
            cam_repo.create(cam_data)
            logger.info(f"Seeded Camera: {cam_data.id}")
            health_repo.upsert(cam_data.id, "CAMERA", "ONLINE", {"resolution": cam_data.resolution})

    # 4. Camera Assignments (Topology mapping)
    assignments = [
        CameraAssignmentCreate(camera_id="CAM-AHM-SG-001", node_id="RDK-X5-AHM-01", pipeline_profile="TRAFFIC_ANALYTICS"),
        CameraAssignmentCreate(camera_id="CAM-AHM-SG-002", node_id="RDK-X5-AHM-01", pipeline_profile="TRAFFIC_ANALYTICS"),
        CameraAssignmentCreate(camera_id="CAM-GNR-SECT-010", node_id="RDK-X5-AHM-02", pipeline_profile="ANPR_SPEED"),
        CameraAssignmentCreate(camera_id="CAM-LEGACY-DVR-04", node_id="KAAVAL-SRV-GANDHINAGAR", pipeline_profile="CENTRAL_ANALYTICS"),
    ]

    existing_assignments = assign_repo.list()
    if not existing_assignments:
        for asgn in assignments:
            assign_repo.create(asgn)
            logger.info(f"Seeded Assignment: {asgn.camera_id} -> {asgn.node_id}")
