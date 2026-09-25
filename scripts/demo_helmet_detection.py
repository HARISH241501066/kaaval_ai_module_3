#!/usr/bin/env python
"""
KAAVAL AI — HELMET DETECTION DEMO (vertical slice)

    road video -> VideoSource -> YOLO -> Kaaval AI logic -> HELMET_VIOLATION
        -> existing M3 Kaaval ingestion -> canonical envelope -> SQLite -> Mock M4

Usage:
    python scripts/demo_helmet_detection.py                       # CAM-001 / camera_feed/camera_main.mp4
    python scripts/demo_helmet_detection.py --camera CAM-002
    python scripts/demo_helmet_detection.py --video some/road1.mp4 --camera CAM-001

Prerequisites:
    pip install -r requirements.txt
    python -m app.ai.fetch_model          # local weights, no cloud API key
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from app.ai.config import MODEL_METADATA, get_ai_config  # noqa: E402
from app.ai.topology import ensure_ai_video_topology  # noqa: E402
from app.ai.video_processor import VideoProcessor  # noqa: E402

BAR = "=" * 65


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sentinel M3 / KAAVAL AI helmet detection demo")
    ap.add_argument("--camera", default=None, help="camera id from AI_VIDEO_SOURCES (default: first entry)")
    ap.add_argument("--video", default=None, help="explicit .mp4 path (overrides the configured video)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N decoded frames (smoke run)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    cfg = get_ai_config()
    try:
        camera_id, video_path = cfg.resolve_video(args.camera, args.video)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(BAR)
    print("KAAVAL AI HELMET DETECTION")
    print(BAR)
    print(f"Camera:        {camera_id}")
    print(f"Video:         {Path(video_path).name}  ({video_path})")
    print(f"Model:         {cfg.model_name} v{cfg.model_version}  [{MODEL_METADATA['license']}]")
    print(f"Weights:       {cfg.model_path} ({'found' if cfg.model_file_exists else 'MISSING'})")
    print(f"Device:        {cfg.device.upper()}")
    print(f"Inference FPS: {cfg.inference_fps}")
    print(f"Thresholds:    conf>={cfg.confidence_threshold}, helmet>={cfg.helmet_min_confidence}, "
          f"confirm={cfg.confirmation_frames} frames, cooldown={cfg.cooldown_s}s")
    print(f"Node:          {cfg.processing_node_id}  (provider KAAVAL_AI)")
    if not cfg.model_file_exists:
        print("\nModel weights missing. Fetch them first:\n    python -m app.ai.fetch_model")
        return 2

    ensure_ai_video_topology()

    processor = VideoProcessor(config=cfg)
    outcome = processor.process(camera_id, video_path, max_frames=args.max_frames)
    metrics = outcome["metrics"]

    print(BAR)
    print("KAAVAL AI DECISION LAYER")
    print(BAR)
    if not outcome["violations"]:
        print("No confirmed helmet violations in this video "
              f"({metrics['detections']} detections over {metrics['inference_frames']} inference frames).")
    for v in outcome["violations"]:
        print(f"\nFrame {v['frame_number']}  (t={v['timestamp']:.2f}s)")
        print("Rider detected")
        print(f"Helmet status: {v['class_name']}")
        print(f"Confidence:    {v['confidence']:.2f}")
        print("Violation confirmed")
        print(f"Event ID:      {v['event_id']}")
        print(f"Evidence:      {(v.get('evidence_ref') or {}).get('uri', '(none)')}")

    print(BAR)
    print("M3 EVENT INGESTION  (existing C4 pipeline)")
    print(BAR)
    for res in outcome["ingest_results"]:
        print(f"Provider:         KAAVAL_AI")
        print(f"Camera:           {res.canonical_event_id and camera_id}")
        print(f"Processing Node:  {cfg.processing_node_id}")
        print(f"Event:            HELMET_VIOLATION -> {res.status} ({res.canonical_event_id})")
        if res.message:
            print(f"Pipeline:         {res.message}")

    print(BAR)
    print("M4 CANONICAL EVENTS (mock Module 4 consumer)")
    print(BAR)
    from app.services.publisher import get_mock_m4_consumer

    for item in get_mock_m4_consumer().get_events(limit=5):
        ev = item["event"]
        print(f"Event ID: {ev['event_id']}  cam={ev['camera_id']}  type={ev['event_type']}  "
              f"conf={ev['confidence']:.2f}  node={ev['processing']['processing_node_id']}  "
              f"model={ev['processing']['model_name']} v{ev['processing']['model_version']}  "
              f"evidence={ev['evidence_ref'].get('uri')}")

    print(BAR)
    print("METRICS")
    print(BAR)
    for key in ("frames_decoded", "inference_frames", "detections", "violations", "events_ingested",
                "processing_time", "inference_fps_actual", "avg_latency_ms", "cooldown_suppressed"):
        print(f"{key:>22}: {metrics.get(key)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
