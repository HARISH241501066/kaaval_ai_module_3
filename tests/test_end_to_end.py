"""
End-to-End Integration Test Suite for Sentinel Gujarat Module 3.
Validates the complete vertical slice:
1. Multi-dimension health monitoring (C6)
2. Inventory & failure impact topology (C1, C2, C3)
3. Stream source broker (C3)
4. Kaaval edge event ingestion with HMAC authentication (C4)
5. Canonical Sentinel Event Envelope normalization
6. Deduplication filter (provider + source_event_id)
7. Live AI provenance tracking
8. Downstream Module 4 delivery verification
"""
from __future__ import annotations

import json
import time
import hmac
import hashlib
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import get_settings
from app.db.database import init_db, SessionLocal
from app.db.seed import seed_initial_topology
from app.services.publisher import get_mock_m4_consumer

settings = get_settings()


@pytest.fixture(scope="session", autouse=True)
def initialize_database():
    """Ensure database and seed data are ready before any tests run."""
    init_db()
    with SessionLocal() as db:
        seed_initial_topology(db)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def make_hmac_headers(node_id: str, body_bytes: bytes, secret: str) -> dict[str, str]:
    """Helper to compute HMAC headers matching sentinel_hwsim.push.sign()."""
    ts = str(int(time.time()))
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body_bytes, hashlib.sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-Kaaval-Node": node_id,
        "X-Kaaval-Timestamp": ts,
        "X-Kaaval-Signature": sig,
    }


def test_01_health_monitoring_four_dimensions(client):
    """
    Test C6: Independent tracking of Camera, Node, AI Runtime, and Event Pipe health.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("HEALTHY", "DEGRADED")
    assert data["database"] == "CONNECTED"
    summary = data["summary"]
    assert summary["cameras_total"] >= 4
    assert summary["nodes_total"] >= 3
    assert summary["ai_runtimes_running"] >= 1
    assert summary["event_pipes_up"] >= 1

    # Verify component types exist
    entity_types = {c["entity_type"] for c in data["components"]}
    assert "CAMERA" in entity_types
    assert "PROCESSING_NODE" in entity_types
    assert "AI_RUNTIME" in entity_types
    assert "EVENT_PIPE" in entity_types


def test_02_inventory_and_failure_impact(client):
    """
    Test C2: Answers 'If this processing node fails, which cameras are affected?'
    """
    # 1. Systems list
    sys_resp = client.get("/api/v1/systems")
    assert sys_resp.status_code == 200
    assert len(sys_resp.json()) >= 3

    # 2. Nodes list
    node_resp = client.get("/api/v1/nodes")
    assert node_resp.status_code == 200
    assert len(node_resp.json()) >= 3

    # 3. Failure impact on RDK-X5-AHM-01
    impact_resp = client.get("/api/v1/nodes/RDK-X5-AHM-01/cameras")
    assert impact_resp.status_code == 200
    impact_data = impact_resp.json()
    assert impact_data["node"]["id"] == "RDK-X5-AHM-01"
    assigned_cam_ids = [c["id"] for c in impact_data["assigned_cameras"]]
    assert "CAM-AHM-SG-001" in assigned_cam_ids
    assert "CAM-AHM-SG-002" in assigned_cam_ids


def test_03_c3_stream_source_broker(client):
    """
    Test C3: Broker returns internal relay URL without exposing vendor credentials.
    """
    resp = client.get("/api/v1/cameras/CAM-AHM-SG-001/stream-source")
    assert resp.status_code == 200
    data = resp.json()
    assert data["camera_id"] == "CAM-AHM-SG-001"
    assert "rtsp://sentinel-m3-relay:8554" in data["relay_url"]
    assert "password" not in data["relay_url"]
    assert data["codec"] == "H264"


import uuid

SHARED_TEST_EVT_ID = f"test-e2e-helmet-{uuid.uuid4().hex[:8]}"


def test_04_kaaval_event_ingestion_and_normalization(client):
    """
    Test C4: Full ingestion pipeline with HMAC signature verification,
    normalization into Canonical Sentinel Event Envelope, and Module 4 delivery.
    """
    node_id = "RDK-X5-AHM-01"
    source_evt_id = SHARED_TEST_EVT_ID

    raw_batch = {
        "node": node_id,
        "batch": [
            {
                "kind": "event",
                "evt_id": source_evt_id,
                "node": node_id,
                "cam": "CAM_01",
                "type": "HELMET_VIOLATION",
                "ts_ms": int(time.time() * 1000),
                "conf": 0.945,
                "data": {
                    "plate": "GJ01AB9999",
                    "riders": 2,
                    "bbox": [100, 200, 300, 250],
                },
                "model_ver": "kaaval-traffic-v2.4",
            },
            {
                "kind": "heartbeat",
                "node": node_id,
                "runtime": "RUNNING",
                "fps": 15.2,
                "ts_ms": int(time.time() * 1000),
            },
        ],
    }

    body_bytes = json.dumps(raw_batch, separators=(",", ":")).encode()
    headers = make_hmac_headers(node_id, body_bytes, settings.kaaval_hmac_secret)

    # 1. Post to ingestion endpoint
    resp = client.post("/mock/kaaval/events", content=body_bytes, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["processed"] == 2
    assert data["accepted"] == 1
    assert data["heartbeats"] == 1

    event_result = [r for r in data["results"] if r["source_event_id"] == source_evt_id][0]
    canonical_id = event_result["canonical_event_id"]
    assert canonical_id.startswith("evt-")

    # 2. Retrieve Canonical Event and verify envelope structure
    get_resp = client.get(f"/api/v1/events/{canonical_id}")
    assert get_resp.status_code == 200
    canonical_event = get_resp.json()

    # Verify Envelope Contract
    assert canonical_event["event_id"] == canonical_id
    assert canonical_event["camera_id"] == "CAM-AHM-SG-001"  # Resolved from external ID "CAM_01"
    assert canonical_event["event_type"] == "HELMET_VIOLATION"
    assert canonical_event["confidence"] == 0.945

    # Verify Source
    assert canonical_event["source"]["provider"] == "KAAVAL_AI"
    assert canonical_event["source"]["adapter"] == "KAAVAL_EDGE"
    assert canonical_event["source"]["source_event_id"] == source_evt_id

    # Verify AI Provenance
    proc = canonical_event["processing"]
    assert proc["processing_node_id"] == node_id
    assert proc["model_name"] == "KAAVAL_HELMET_VIOLATION"
    assert proc["model_version"] == "kaaval-traffic-v2.4"
    assert proc["inference_fps"] == 15.0

    # Verify Object & Attributes
    assert canonical_event["object"]["plate_number"] == "GJ01AB9999"
    assert canonical_event["object"]["bounding_box"] == [100, 200, 300, 250]
    assert canonical_event["attributes"]["riders"] == 2

    # Verify Evidence Ref
    assert f"mock://evidence/kaaval/{source_evt_id}.jpg" in canonical_event["evidence_ref"]["image"]


def test_05_deduplication_filter(client):
    """
    Test C4: Deduplication suppresses duplicate events with matching provider + source_event_id.
    """
    node_id = "RDK-X5-AHM-01"
    source_evt_id = SHARED_TEST_EVT_ID  # Re-sending the same event from test_04

    raw_batch = {
        "node": node_id,
        "batch": [
            {
                "kind": "event",
                "evt_id": source_evt_id,
                "node": node_id,
                "cam": "CAM_01",
                "type": "HELMET_VIOLATION",
                "ts_ms": int(time.time() * 1000),
                "conf": 0.945,
                "data": {"plate": "GJ01AB9999"},
                "model_ver": "kaaval-traffic-v2.4",
            }
        ],
    }

    body_bytes = json.dumps(raw_batch, separators=(",", ":")).encode()
    headers = make_hmac_headers(node_id, body_bytes, settings.kaaval_hmac_secret)

    resp = client.post("/mock/kaaval/events", content=body_bytes, headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["accepted"] == 0
    assert data["duplicates"] == 1
    assert data["results"][0]["duplicate"] is True
    assert data["results"][0]["status"] == "DUPLICATE"


def test_06_live_ai_provenance_tracking(client):
    """
    Test: Query full AI execution provenance for a canonical event.
    """
    # Query latest events
    events_resp = client.get("/api/v1/events?limit=1")
    assert events_resp.status_code == 200
    events = events_resp.json()
    assert len(events) > 0
    test_evt_id = events[0]["event_id"]

    # Query provenance
    prov_resp = client.get(f"/api/v1/provenance/{test_evt_id}")
    assert prov_resp.status_code == 200
    prov_data = prov_resp.json()

    assert prov_data["event_id"] == test_evt_id
    assert "camera" in prov_data
    assert "processing_node" in prov_data
    assert "ai_lineage" in prov_data
    assert "evidence" in prov_data
    assert prov_data["audit"]["raw_payload_preserved"] is True


def test_07_mock_m4_consumer_verification(client):
    """
    Test: Verifies that downstream Module 4 Consumer received the canonical events.
    """
    m4_resp = client.get("/api/v1/mock-m4/events")
    assert m4_resp.status_code == 200
    m4_data = m4_resp.json()
    assert m4_data["consumer"] == "SENTINEL_MODULE_4_INTELLIGENCE"
    assert m4_data["count"] > 0
    first_received = m4_data["delivered_events"][0]
    assert "received_at" in first_received
    assert "event" in first_received
    assert "event_id" in first_received["event"]


def test_08_seed_demo_events(client):
    """
    Test: Seeding demo events for hackathon live evaluator testing.
    """
    seed_resp = client.post("/api/v1/events/seed", json={"count": 3})
    assert seed_resp.status_code == 200
    data = seed_resp.json()
    assert len(data["results"]) == 3
    for res in data["results"]:
        assert res["status"] == "ACCEPTED"
        assert res["canonical_event_id"].startswith("evt-")
