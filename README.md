# sentinel-hwsim

Simulated hardware for Sentinel Module 3 (VMS Federation). It provides **IP cameras** (ONVIF-style) and **Kaaval AI nodes** (RDK X5 edge / analytics server) that behave like the real devices, including latency, drifting sensors, crashes, garbage payloads and dropped sessions, so M3 can be built and tested with no hardware on the desk.

```
pip install -e ".[dev]"
pytest                                   # 41 tests, ~7 s
python examples/01_camera_basics.py
```

## Architecture: why the swap is a config change

```
 application code (M3-C1 adapters, C6 probes, C7 connection test)
        │  imports ONLY sentinel_hwsim.contract  (DTOs, errors, CameraDriver / AINodeDriver)
        ▼
 create_driver(config)  ── M3_DRIVER_MODE=mock ──► MockCameraDriver / MockKaavalDriver
        │                                                │ maps vendor payload -> strict DTO
        │                                                ▼
        │                                         SimTransport  (latency, deadline, faults, trace)
        │                                                ▼
        │                                         CameraModel / KaavalNodeModel (state + physics,
        │                                                        vendor-shaped payloads)
        └────────────── M3_DRIVER_MODE=real ──► OnvifCameraDriver / KaavalHttpDriver (skeletons)
```

The layers have separate jobs:

- **Models** hold device state and physics. They answer with vendor-shaped payloads (ONVIF-like keys, Kaaval JSON), never canonical DTOs.
- **Transport** is the wire. It applies latency, enforces the deadline, injects faults and records a trace.
- **Drivers** own the connection state machine and map vendor payloads to strict Pydantic DTOs. A malformed payload always becomes a `MappingError`, never a silent wrong value.
- **Contract** is shared by mock and real drivers. `tests/test_conformance.py` is the kit every real driver must pass.

## What is simulated

| Device | Reads | Writes | Behaviour over time |
|---|---|---|---|
| Camera | status, stream URI (MAIN/SUB), snapshot (real PNG), PTZ position, clock | encoder fps/bitrate, PTZ absolute move | fps/bitrate converge to new settings in ~2 s; bitrate follows a traffic cycle; PTZ travels at 0.5 units/s; RTC drifts |
| Kaaval node | capabilities, model info, AI health, event stream | load model (restart, version change) | CPU/memory follow load; temperature follows CPU and ambient; throttles at ≥85 °C (fps −40%, latency up); Poisson events per camera; heartbeats |

**Kaaval failure layers**, matching the M3-C6 health layers:

| Control | Effect |
|---|---|
| `crash_runtime()` | Events stop; the agent keeps heartbeating `CRASHED`. |
| `stall_inference()` | Runtime reports `RUNNING` but fps is 0. |
| `cut_pipe()` | Events and heartbeats buffer on the node, then flush late and out of order when the pipe returns. |
| `offline()` | Node is powered off. Calls time out, and the buffer is lost. |
| `set_ambient(58)` | Heat stress that triggers thermal throttling. |

**Event types:** PLATE_READ (Gujarat plates, e.g. `GJ01AB1234`), HELMET_VIOLATION, TRIPLE_RIDING, VEHICLE_DETECTION, and WATCHLIST_HIT (after `set_watchlist`). Only types for the node's enabled functions are emitted.

## Usage

### Initialise and send commands

```python
from sentinel_hwsim import DriverConfig, Simulator, create_driver

sim = Simulator(speed=10, seed=7)       # 10x time compression, deterministic
cam = create_driver(DriverConfig(kind="camera", device_id="CAM-001",
                                 sim={"faults": "healthy", "clock_offset_s": 37,
                                      "rtsp_base": "rtsp://mediamtx:8554/sim/cam001"}),
                    simulator=sim)

async with cam:                                     # connect() / disconnect()
    src = await cam.get_stream_uri("SUB")           # StreamSource DTO
    await cam.set_encoder("MAIN", fps=10, bitrate_kbps=1500)   # write
    await cam.ptz_move(pan=0.75, tilt=-0.3, zoom=0.4)          # write
    while (pos := await cam.ptz_get_position()).moving:        # poll
        await asyncio.sleep(0.1)
```

Point `rtsp_base` at your MediaMTX/FFmpeg simulated streams. The URIs that M3-C3 resolves will then actually play video.

### Receive AI data

```python
node = sim.kaaval_node("EDGE-NODE-001", cameras=["CAM-001", "CAM-002"])
await node.connect()
async for item in node.stream(poll_interval_s=1.0):   # AIEvent | Heartbeat
    if isinstance(item, AIEvent):
        print(item.event_type, item.attributes.get("plate"), item.model_version)
# node.last_heartbeat_at, node.dead_letters  -> for C6 pipe checks and C4 DLQ tests
```

### Handle error states

```python
from sentinel_hwsim import DeviceError, MappingError, AuthFailed

try:
    await cam.get_status()
except MappingError as e:        # garbage payload: do not retry, send e.detail["sample"] to the DLQ
    ...
except AuthFailed:               # credentials: flag auth_ok=false (C6), no retry
    ...
except DeviceError as e:         # everything else carries .code (ErrorCode) and .retryable
    if e.retryable: ...          # VENDOR_TIMEOUT, DEVICE_DISCONNECTED, CONNECTION_FAILED
```

`examples/03_resilience.py` shows a full retry-with-backoff, auto-reconnect, circuit-breaker and DLQ loop against every failure mode.

## Configuring faults

**Presets:**

| Preset | Behaviour |
|---|---|
| `ideal` | 0 ms latency, no faults. Use for unit tests. |
| `healthy` | ~80 ms latency with occasional spikes. |
| `flaky` | 5% timeouts, 2% garbage payloads, 2% dropped sessions, 10% connect failures. |
| `hostile` | Much worse on every axis. |

**Custom profile:**

```python
from sentinel_hwsim import FaultProfile, LatencyProfile
sim.camera("CAM-009", faults=FaultProfile(latency=LatencyProfile(base_ms=300, spike_probability=0.1),
                                          timeout_rate=0.1, invalid_data_rate=0.05))
```

**Deterministic injection** for tests and demos:

```python
ctl = sim.controls("CAM-001")
ctl.fail_next("timeout")            # also: drop_connection, auth_failure, invalid_data, connect_failure
ctl.offline(); ctl.revoke_credentials(); ctl.heal()
```

Queued faults apply to the next operation that can exhibit them, which may be a `connect()` or a call.

`invalid_data` corrupts payloads in one of three ways: an HTML error page, `{}`, or every scalar stringified plus a missing section (the classic XML-to-dict bug). Each is guaranteed to fail strict mapping.

## Tracing

```python
from sentinel_hwsim import enable_trace_logging
enable_trace_logging(logging.DEBUG)   # every wire call; logging.WARNING = faults only
for entry in cam.trace: print(entry)  # ring buffer of the last 1000 calls
```

Example trace output:

```
#36    04:00:38.025 CAM-099   status              5000.0ms timeout
#101   04:01:16.780 CAM-099   status               298.3ms invalid_data
```

## Push mode: Kaaval to M3-C4 ingest, with live chaos

```
python -m sentinel_hwsim.push --url http://localhost:8000/federation/ingest/EDGE-NODE-001 \
    --secret "$KAAVAL_HMAC_SECRET" --cameras CAM-001,CAM-002,CAM-003,CAM-004
```

While it runs, type commands on stdin: `crash`, `restart`, `stall`, `cut`, `restore`, `off`, `on`, `heat 58`, `watch GJ01AB1234`, `badrate 0.2`, `status`. This is your on-stage "kill Kaaval" control. Failed deliveries are re-buffered in order (store-and-forward), and `--dry-run` prints batches without a server.

Your C4 receiver verifies each request like this:

```python
expected = hmac.new(secret, f"{ts}.".encode() + raw_body, sha256).hexdigest()
ok = hmac.compare_digest(expected, request.headers["X-Kaaval-Signature"]) and abs(now - int(ts)) <= 300
```

## Swapping to real hardware

1. Implement the methods in `drivers/real.py`. Each one names the ONVIF operation or Kaaval endpoint it must call, and credentials come from a `CredentialProvider` (C7/Vault).
2. Run `tests/test_conformance.py` against the real driver. It must pass unchanged.
3. Set `M3_DRIVER_MODE=real`. No application code changes.

To add a vendor (e.g. Hikvision ISAPI), call `register_driver("camera", "hikvision", builder)` and set `mode="hikvision"` for those devices in config.

**The rule that keeps the swap clean:** application code never imports `Simulator`, `SimControls` or `driver.sim`. Those belong in tests and demo scripts only.

## Limits (be honest in the demo)

- There is no video decoding. Stream URIs point wherever `rtsp_base` says, so pair this with MediaMTX + FFmpeg loops for real pixels.
- Events are statistically plausible but not derived from frames.
- Recording search and playback (C5) are not modelled yet.
- The real drivers are skeletons until hardware arrives.

---

## KAAVAL AI — local YOLO helmet detection (`app/ai/`)

A **local, CPU-first YOLO inference layer** that turns road traffic videos into
`HELMET_VIOLATION` events which flow through the **existing** M3 C4 pipeline. Three
different concerns, kept strictly apart:

| Layer | Job | Where |
|---|---|---|
| **YOLO detection** | pixels → `WITH_HELMET` / `WITHOUT_HELMET` boxes + confidence | `app/ai/helmet_detector.py` |
| **Kaaval AI decision logic** | temporal confirmation, cooldown, rider/vehicle association → **one event per violation episode**, never per frame | `app/ai/kaaval_logic.py` |
| **M3 event federation** | validation, identity resolution, dedup, canonical envelope, SQLite, health, Mock M4 | existing `KAAVAL_EDGE` adapter + `app/services/pipeline.py` |

```
LOCAL VIDEO → VideoSource → YOLO HelmetDetector → Kaaval AI Logic
    → HELMET_VIOLATION (Kaaval event) → existing Kaaval adapter / C4 ingestion
    → Validate · Dedup · Identity → Canonical Sentinel Event Envelope → SQLite → Mock M4
```

Nothing here bypasses M3: there is no `YOLO → database` or `YOLO → M4` path.

### 1. Install

```bash
pip install -r requirements.txt        # adds: ultralytics, opencv-python
python -m app.ai.fetch_model           # downloads the weights once, verifies pinned SHA-256
```

No API key, no cloud inference, no hosted service: `fetch_model` downloads from a public
URL and refuses any file whose SHA-256 does not match the pin in `app/ai/config.py`.

### 2. Model weights and model card

Weights live (git-ignored) at `models/helmet_model.pt` — override with `AI_MODEL_PATH`.
The full model card is stored in `app/ai/config.py` (`MODEL_METADATA`) and printed by
`python -m app.ai.fetch_model --info`:

| Field | Value |
|---|---|
| Model | `yolov8n-helmet-detection` v1.0.0 (Ultralytics YOLOv8n, fine-tuned) |
| Source | https://huggingface.co/iam-tsr/yolov8n-helmet-detection (`best.pt`) |
| Format | Ultralytics PyTorch `.pt` |
| Classes | `With Helmet`, `Without Helmet` → `WITH_HELMET`, `WITHOUT_HELMET` |
| License | **MIT** |
| SHA-256 | `c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d` |
| Accuracy | P 0.855 / R 0.808 / mAP@50 0.881 |

### 3. Supported detection classes

`WITH_HELMET`, `WITHOUT_HELMET` (from the selected two-class helmet model). Class names
are normalised, so `helmet`, `no_helmet`, `head`, `With Helmet`, … all map onto the same
canonical vocabulary (`app/ai/helmet_detector.py: normalize_class_name`). The detector
interface also understands `PERSON` / `MOTORCYCLE` labels so a future multi-class Kaaval
model drops in without changes (see *Known limitations*).

### 4. Configuration (all env / `.env`, prefix `AI_`)

```dotenv
AI_MODEL_PATH=models/helmet_model.pt
AI_CONFIDENCE_THRESHOLD=0.50     # detections below this are dropped
AI_INFERENCE_FPS=2               # sample 30 fps video down to 2 inferences/s
AI_IMAGE_SIZE=1280               # CEILING for the model input
AI_IMAGE_SIZE_AUTO=true          # ...the real size follows the source resolution
AI_DEVICE=cpu                    # AI_DEVICE=0 for CUDA — never hardcoded
AI_TRACKING_ENABLED=true         # YOLO ByteTrack, auto-falls back if unavailable
AI_HELMET_MIN_CONFIDENCE=0.60    # rule engine threshold
AI_HELMET_CONFIRMATION_FRAMES=3  # observations before a violation is confirmed
AI_HELMET_EVENT_COOLDOWN_S=10    # min video-seconds between two events per camera
AI_EVIDENCE_DIR=evidence
AI_PROCESSING_NODE_ID=RDK-X5-001
AI_VIDEO_SOURCES=CAM-001=camera_feed/camera_main.mp4;CAM-002=camera_feed/camera_sub.mp4
```

### 5. Video → camera mapping (simulated CCTV → swap to RTSP later)

`AI_VIDEO_SOURCES` maps a camera id to a local `.mp4`:

| Camera | Video |
|---|---|
| `CAM-001` | `camera_feed/camera_main.mp4` (street-level traffic) |
| `CAM-002` | `camera_feed/camera_sub.mp4` (wide intersection) |

On startup (and on every job) M3 registers these cameras, the `RDK-X5-001` processing
node and their assignments through the existing repositories — provenance comes from the
database topology, not from hard-coded strings. The detector only ever sees
`VideoSource → frames`, so replacing the `.mp4` path with an `rtsp://` URI later
(`RTSPVideoSource` is the reserved slot; ONVIF discovery is out of scope here) changes
nothing else.

### 6. Run helmet detection

```bash
# demo script (console transcript below)
python scripts/demo_helmet_detection.py --camera CAM-001

# or the API — start the server first
uvicorn app.main:app --port 8000
```

**Browser console: <http://localhost:8000/ui/ai>** — served by M3 itself, no build step and no
separate frontend process. Pick a camera, start a job, watch the live counters
(`frames · detections · violations`), and see every confirmed violation with its annotated
evidence frame. Three static files under `app/web/static/ai/`, served by `app/api/ui.py`; the
page calls only this app's own `/ai/helmet/*` API, so nothing is exposed to CORS.

Example API request (Swagger UI at `/docs` → *KAAVAL AI — Local Helmet Detection*):

```bash
curl -X POST http://localhost:8000/ai/helmet/process \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "CAM-001"}'
# 202 {"job_id": "job-8f2c1a9b0d31", "camera_id": "CAM-001",
#      "video_path": "camera_feed/camera_main.mp4", "status": "STARTED"}

curl http://localhost:8000/ai/helmet/jobs/job-8f2c1a9b0d31   # QUEUED→RUNNING→COMPLETED/FAILED
curl http://localhost:8000/ai/helmet/health                  # YOLO READY/FAILED, cameras, pipeline
curl http://localhost:8000/ai/helmet/config                  # effective config + model card
```

Job metrics: `frames_decoded`, `inference_frames`, `detections`, `violations`,
`processing_time`, `inference_fps_actual`, `avg_latency_ms`, `model`, `camera_id`.

### 7. How the event enters M3

1. Kaaval AI logic confirms a violation → evidence frame written to
   `evidence/<yyyy>/<mm>/<dd>/<EVT-ID>.jpg` (boxes, labels, confidences, camera, timestamp).
2. A raw **Kaaval edge event** (`evt_id / node / cam / type / ts_ms / conf / model_ver / data`)
   is built and pushed into the **existing** `C4IngestionPipeline`
   (`POST`-equivalent call, provider `KAAVAL_AI`, adapter `KAAVAL_EDGE`).
3. The existing pipeline validates → resolves camera & node → deduplicates → normalizes
   through `KaavalEdgeAdapter` → persists the canonical envelope in SQLite → publishes to
   the Mock M4 consumer. Evidence and real model provenance ride along as optional
   payload fields; legacy payloads keep their old defaults untouched.

Example canonical event (as delivered to Mock M4):

```json
{
  "event_id": "evt-6f0c1b2a9d31",
  "camera_id": "CAM-001",
  "event_type": "HELMET_VIOLATION",
  "timestamp": "2026-09-25T12:34:56Z",
  "source": {
    "provider": "KAAVAL_AI",
    "adapter": "KAAVAL_EDGE",
    "source_event_id": "EVT-8F2C1A9B04"
  },
  "processing": {
    "processing_node_id": "RDK-X5-001",
    "analytics_version": "kaaval-agent-1.4.0",
    "model_name": "yolov8n-helmet-detection",
    "model_version": "1.0.0",
    "inference_fps": 2.0,
    "latency_ms": 143.2
  },
  "confidence": 0.91,
  "object": {"type": "MOTORCYCLE_RIDER", "bounding_box": [412, 183, 488, 321]},
  "attributes": {
    "helmet_status": "WITHOUT_HELMET",
    "violation": "HELMET_VIOLATION",
    "riders": 1,
    "frame_number": 142,
    "video_timestamp": 5.67
  },
  "evidence_ref": {
    "type": "local_file",
    "uri": "evidence/2026/09/25/EVT-8F2C1A9B04.jpg"
  },
  "delivery_status": "DELIVERED"
}
```

### 8. Health: four M3 dimensions + an independent YOLO dimension

`GET /health` now also surfaces a `YOLO_MODEL` component (`READY` / `FAILED` /
`NOT_LOADED`) next to CAMERA / PROCESSING_NODE / AI_RUNTIME / EVENT_PIPE:

```
Camera:        ONLINE
Processing Node: ONLINE
Kaaval AI:     ONLINE
YOLO Model:    READY   (or FAILED if the weights are missing — the camera stays ONLINE)
Event Pipeline: UP
```

A missing/failed model never marks a physical camera offline.

### 9. Tests

```bash
python -m pytest tests -q                      # full suite incl. AI tests (YOLO mocked)
python -m app.ai.fetch_model                   # once
python -m pytest tests/test_ai_integration.py  # optional: REAL model + REAL road video
RUN_AI_FULL_TESTS=1 python -m pytest tests/test_ai_integration.py  # + full vertical slice
```

### 10. Known limitations (honest demo notes)

- **Two-class model.** The selected MIT model detects rider *heads* (`With/Without Helmet`)
  only — no `person`/`motorcycle` boxes. A violation therefore proves a rider without a
  helmet in frame, but **no helmet box is guaranteed to belong to a specific motorcycle**.
  Vehicle association in `kaaval_logic.py` only activates when a model that emits
  `MOTORCYCLE`/`PERSON` classes is configured.
- **Inference size is derived per frame, never fixed.** YOLO letterboxes every frame into a
  square `imgsz`, so a single global value is wrong for mixed sources: at `imgsz=640` a
  `3840x2160` frame is downscaled 6x and a rider's head lands at ~15 px, below what the model
  can see. `AI_IMAGE_SIZE` is therefore a **ceiling**, and the detector scales the real size to
  the source (`AIConfig.inference_size`): 1280 for 4K, 640 for anything up to 1280 px wide,
  snapped to the model stride. Measured on `camera_sub.mp4` (same sample frames, conf ≥ 0.25):
  **2 detections at 640, 41 at 1280 (`~0.28 s/frame` on CPU), 94 at 1920 (`~0.57 s/frame`)**.
  Set `AI_IMAGE_SIZE_AUTO=false` to pin a fixed size instead (benchmarks, CPU-constrained runs).
- **Wide/aerial views still find less than street-level ones.** At the fixed size the wide
  `CAM-002` view now yields ~39 detections per video (it was 1), but its riders are far smaller
  than `CAM-001`'s, so expect fewer confirmed events than the street-level feed. Raising
  `AI_IMAGE_SIZE` to 1920 helps at roughly double the processing time.
- **Confirmation is a temporal contract.** A violation needs `AI_HELMET_CONFIRMATION_FRAMES`
  observations, so at `AI_INFERENCE_FPS=2` a rider must stay visible for ~1.5 s video-seconds.
  Riders crossing in fewer samples are never confirmed — raise `AI_INFERENCE_FPS` to catch
  faster traffic (cost: more CPU per video).
- **Cooldown is per camera**: a second rider confirmed within `AI_HELMET_EVENT_COOLDOWN_S`
  video-seconds of the previous event is suppressed (raise the interval or lower the
  cooldown if that matters for your demo).
- **Prototype job runner**: in-memory job registry + daemon thread. No queue/broker —
  deliberately out of scope.
- **Evidence** is a local JPEG directory; object storage would replace
  `app/ai/evidence.py: save_evidence_frame` only.
- **RTSP**: `RTSPVideoSource` is a reserved slot — this task processes local MP4; ONVIF
  discovery is not implemented.
