"""
Interactive Demonstration Script for Sentinel Gujarat Module 3.
Runs the complete vertical slice demonstration against a running M3 server (default: http://localhost:8000).
"""
from __future__ import annotations

import sys
import time
import json
import hmac
import hashlib
import urllib.request
import urllib.error

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
SECRET = "dev-secret"


def req(method: str, path: str, data: dict | None = None, headers: dict | None = None):
    url = f"{BASE_URL}{path}"
    headers = headers or {}
    body = None
    if data is not None:
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"

    r = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            content = resp.read().decode("utf-8")
            return resp.status, json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        content = e.read().decode("utf-8")
        return e.code, json.loads(content) if content else {}


def header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print(f"Connecting to Sentinel Module 3 Backend at: {BASE_URL}")

    # 1. Health Monitoring (C6)
    header("1. C6 Health Monitoring (4 Independent Dimensions)")
    status, health = req("GET", "/health")
    print(f"HTTP Status: {status}")
    print(f"System State: {health.get('status')}, Database: {health.get('database')}")
    print("Health Summary Counts:", json.dumps(health.get("summary", {}), indent=2))
    print(f"Total Components Monitored: {len(health.get('components', []))}")

    # 2. Topology & Failure Impact (C1, C2)
    header("2. Inventory Topology & Failure Impact Analysis")
    status, nodes = req("GET", "/api/v1/nodes")
    print(f"Registered Processing Nodes ({len(nodes)}):")
    for n in nodes:
        print(f"  - [{n['id']}] {n['name']} (Type: {n['node_type']}, IP: {n['ip_address']})")

    node_id = "RDK-X5-AHM-01"
    status, impact = req("GET", f"/api/v1/nodes/{node_id}/cameras")
    print(f"\nFailure Impact Analysis for Node [{node_id}]:")
    print(f"  If {node_id} fails, the following cameras lose AI analytics:")
    for cam in impact.get("assigned_cameras", []):
        print(f"  - {cam['id']}: {cam['name']} ({cam['location_name']})")

    # 3. Stream Source Broker (C3)
    header("3. C3 Stream Source Broker (Credential Isolation)")
    status, stream = req("GET", "/api/v1/cameras/CAM-AHM-SG-001/stream-source")
    print("Stream Broker Response:")
    print(json.dumps(stream, indent=2))
    print("-> Notice: Upstream vendor credentials are HIDDEN; internal relay URL is provided.")

    # 4. Ingest Event with HMAC Signature (C4)
    header("4. C4 Event Ingestion with HMAC-SHA256 Signature (Kaaval Edge)")
    source_evt_id = f"demo-{int(time.time())}"
    batch_payload = {
        "node": node_id,
        "batch": [
            {
                "kind": "event",
                "evt_id": source_evt_id,
                "node": node_id,
                "cam": "CAM_01",
                "type": "HELMET_VIOLATION",
                "ts_ms": int(time.time() * 1000),
                "conf": 0.942,
                "data": {
                    "plate": "GJ01AB1234",
                    "riders": 2,
                    "bbox": [120, 240, 220, 180],
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

    body_bytes = json.dumps(batch_payload, separators=(",", ":")).encode("utf-8")
    ts = str(int(time.time()))
    sig = hmac.new(SECRET.encode(), f"{ts}.".encode() + body_bytes, hashlib.sha256).hexdigest()

    hmac_headers = {
        "X-Kaaval-Node": node_id,
        "X-Kaaval-Timestamp": ts,
        "X-Kaaval-Signature": sig,
    }

    status, ingest_resp = req("POST", "/mock/kaaval/events", batch_payload, hmac_headers)
    print(f"Ingest Status: {status}")
    print(json.dumps(ingest_resp, indent=2))
    canonical_id = ingest_resp["results"][0]["canonical_event_id"]

    # 5. Deduplication Test
    header("5. C4 Deduplication Filter Test")
    print("Re-sending the exact same event payload with identical source_event_id...")
    ts2 = str(int(time.time()))
    sig2 = hmac.new(SECRET.encode(), f"{ts2}.".encode() + body_bytes, hashlib.sha256).hexdigest()
    hmac_headers2 = {
        "X-Kaaval-Node": node_id,
        "X-Kaaval-Timestamp": ts2,
        "X-Kaaval-Signature": sig2,
    }
    status, dedup_resp = req("POST", "/mock/kaaval/events", batch_payload, hmac_headers2)
    print(f"Ingest Status: {status}")
    print(json.dumps(dedup_resp, indent=2))
    assert dedup_resp["duplicates"] == 1, "Expected duplicate suppression!"
    print("-> Deduplication successfully suppressed duplicate event!")

    # 6. Retrieve Normalized Canonical Event Envelope
    header("6. Normalized Canonical Sentinel Event Envelope")
    status, canonical_event = req("GET", f"/api/v1/events/{canonical_id}")
    print(json.dumps(canonical_event, indent=2))

    # 7. Live AI Provenance
    header("7. Live AI Provenance & Lineage Tracking")
    status, prov = req("GET", f"/api/v1/provenance/{canonical_id}")
    print(json.dumps(prov, indent=2))

    # 8. Downstream Module 4 Delivery Verification
    header("8. Module 4 (Intelligence) Consumer Verification")
    status, m4_data = req("GET", "/api/v1/mock-m4/events?limit=3")
    print(f"Module 4 Consumer Buffer (Total Events: {m4_data['count']}):")
    if m4_data["delivered_events"]:
        latest = m4_data["delivered_events"][0]
        print(f"  Delivered at: {latest['received_at']}")
        print(f"  Canonical Event ID: {latest['event']['event_id']}")
        print(f"  Camera ID: {latest['event']['camera_id']}")
        print(f"  Event Type: {latest['event']['event_type']}")
        print(f"  Confidence: {latest['event']['confidence']}")

    header("ALL DEMONSTRATION STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
