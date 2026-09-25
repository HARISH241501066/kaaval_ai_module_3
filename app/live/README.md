# Live feed detection

Reads the Sentinel Camera Grid's live CCTV streams, detects events in them with a real object
detector, and sends those events into Module 3's ingestion pipeline (C4) in place of the mock
Kaaval events.

```
grid camera (RTSP, TCP)
   -> reader thread     keeps draining the stream; times everything on the stream's own timestamps
   -> detector          YOLOv4-tiny (COCO), CPU, through OpenCV
   -> tracker           one event per real object, not one per frame
   -> signed batches    same HMAC scheme as a Kaaval node -> /federation/ingest/<node>
   -> M3 (dedup, storage, provenance, delivery to Module 4)
```

## Set up

1. Put your grid account in `.env` (this file is git-ignored):
   ```
   SENTINEL_GRID_EMAIL=you@example.com
   SENTINEL_GRID_PASSWORD=XXXX-XXXX-XXXX
   ```
   Everything else in the "Sentinel Camera Grid" and "Live detection worker" blocks has a working default.
   `LIVE_CAMERAS` picks the cameras (the grid gives every connection its own stream copy, so keep it small;
   `LIVE_MAX_CAMERAS` is a hard cap). `KAAVAL_HMAC_SECRET` must be the same value the API uses.
2. Install and fetch the detector model (once):
   ```
   pip install -r requirements.txt
   python -m app.live.detector --fetch
   ```

## Run

```
uvicorn app.main:app --port 8000          # terminal 1: the M3 API
python -m app.live.worker                 # terminal 2: the live worker
```

The worker first runs the connection test (TCP reach, login, catalogue). Only if that passes does it register the grid,
its detector node and the chosen cameras in M3 (`GRID-CAM01` ... with external id `cam01`), then it starts watching.
Look at the results in `http://localhost:8000/docs`: `GET /api/v1/events`, `GET /health`, `GET /api/v1/provenance/summary/stats`.

Try it without the grid, on a video file that loops like a live feed:
```
python -m app.live.worker --source cam01=camera_feed/camera_main.mp4 --dry-run
```

## What it detects

| Event | How |
|---|---|
| `VEHICLE_DETECTION` | a car, motorbike, bus or truck followed for 3 sampled frames (once per vehicle) |
| `TRIPLE_RIDING` | a motorbike carrying 3 or more people for 3 sampled frames (once per bike) |

`HELMET_VIOLATION` and `PLATE_READ` need dedicated models (a helmet classifier, a plate detector and OCR). The general
COCO model used here cannot produce them, so they are not faked.

## How it behaves on a live feed

* Timing uses the stream's presentation timestamps, never arrival time or `CAP_PROP_FPS`. `LIVE_SAMPLE_FPS=4` means
  four detections per second of *video*.
* A loop point (timestamps jump back) resets the tracker and is logged. It is never counted as a failure.
* Decoder warnings on join are logged, not fatal. Only a failed open or a 5 s read timeout counts as a disconnect.
* Reconnect uses exponential backoff, 2 s doubling to 30 s with jitter, and only resets once the stream proved healthy.
* Reading never waits for detection. If detection cannot keep up, sampled frames are skipped (the "frame(s) skipped"
  counter in the 30-second stats line) instead of letting the connection back up and the server discard data.

## Credentials

They are read only from the environment. The only place a credentialed URL exists is `GridClient.rtsp_url`, in memory.
Cameras are registered in M3 with credential-free URLs, and every log record and traceback is scrubbed (`redact.py`).

## Limits

* Detection quality is that of a small general-purpose model; distant or small objects are missed.
* 4K feeds at high bitrate are heavy to decode: watch the "skipped" counter and the reconnect count.
* Some H.265 streams report no timestamps to OpenCV; the reader then counts frames at 25 fps and logs a warning.
* Not built: the MediaMTX relay and `/web` transcoding, the credential-free stream broker for Module 2, scheduled
  catalogue re-sync, active camera health probes, and the HLS fallback when RTSP port 8554 is blocked.
