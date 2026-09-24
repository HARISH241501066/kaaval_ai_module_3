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
