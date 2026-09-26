"""
M3 application settings.
All values can be overridden through environment variables or a .env file.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- database ---
    database_url: str = "sqlite:///./sentinel_m3.db"

    # --- security / auth ---
    # Secret used to verify HMAC-SHA256 signatures from push.py / real Kaaval nodes.
    # Must match the --secret argument passed to sentinel_hwsim.push.
    kaaval_hmac_secret: str = "dev-secret"

    # Maximum allowed clock skew (seconds) for HMAC timestamp check
    kaaval_hmac_max_skew_s: int = 300

    # --- deduplication ---
    # Time window (seconds) within which duplicate events are suppressed
    dedup_window_s: int = 3600

    # --- logging ---
    log_level: str = "INFO"

    # --- project meta ---
    project_name: str = "Sentinel Gujarat — Module 3 Federation"
    version: str = "0.1.0"

    # --- mock M4 ---
    # 'log'  → print to console (prototype default)
    # 'http' → POST to m4_endpoint_url
    m4_delivery_mode: str = "log"
    m4_endpoint_url: str = "http://localhost:9000/events"

    # --- Sentinel Camera Grid (live CCTV feeds) ---
    # Credentials come only from the environment / .env and are never returned by the API or logged.
    sentinel_grid_email: str = ""
    sentinel_grid_password: SecretStr = SecretStr("")
    sentinel_grid_host: str = "103.250.160.189"
    sentinel_grid_rtsp_port: int = 8554
    sentinel_grid_web_base: str = "https://cctv.corp8.cloud"

    sentinel_grid_whep_port: int = 8889
    grid_sync_interval_s: int = 900   # C2 re-sync period; 0 = only when POST /api/v1/grid/sync is called

    # --- stream relay (C3): our MediaMTX, the only thing browsers and Module 2 ever talk to ---
    relay_enabled: bool = True
    relay_bin: str = "relay/bin/mediamtx.exe"
    relay_config_path: str = "relay/generated/mediamtx.yml"      # holds grid credentials: git-ignored
    relay_api_url: str = "http://127.0.0.1:9997"
    relay_public_host: str = "localhost"                          # host the BROWSER uses to reach the relay
    relay_rtsp_port: int = 8654
    relay_hls_port: int = 8888
    relay_webrtc_port: int = 8889
    relay_allow_origins: str = "http://127.0.0.1:8765,http://localhost:8765"   # who may call the relay from a browser
    relay_idle_close_s: int = 60      # upstream is closed this long after the last viewer leaves
    relay_probe_timeout_s: int = 12   # how long a source gets to start before we try the next protocol
    # Some grid cameras send a keyframe only every 20+ s, and the grid replays that whole GOP as a burst when a client connects.
    # MediaMTX drops packets beyond this queue (default 512), which corrupts the picture: keep it above the largest burst.
    relay_write_queue_size: int = 8192
    ffmpeg_path: str = ""             # empty = look on PATH

    # --- live detection worker (python -m app.live.worker) ---
    live_cameras: str = ""            # comma separated grid ids; empty = first live_max_cameras of the catalogue
    live_max_cameras: int = 3         # every open camera costs the grid one stream copy
    live_sample_fps: float = 4.0      # detections per second of video, on the stream's own timestamps
    live_min_confidence: float = 0.45
    live_node_id: str = "LIVE-DETECTOR-001"
    m3_ingest_base_url: str = "http://localhost:8000"
    live_model_dir: str = "models"

    # --- KAAVAL AI: local YOLO helmet detection layer (app.ai) ---
    # Local weights for the helmet detector (fetched with: python -m app.ai.fetch_model).
    ai_model_path: str = "models/helmet_model.pt"
    ai_model_name: str = "yolov8n-helmet-detection"   # provenance: model_name in the canonical event
    ai_model_version: str = "1.0.0"                   # provenance: model_version in the canonical event
    ai_confidence_threshold: float = 0.50             # detections below this are dropped
    # --- frame selection (demo coverage first) ---
    # AI_FRAME_SKIP=0 (default) => run YOLO on EVERY decoded frame (maximum coverage).
    # AI_FRAME_SKIP=1 => every 2nd frame, AI_FRAME_SKIP=4 => every 5th frame, ...
    ai_frame_skip: int = 0
    # Legacy time-based throttle: inference observations per second of video. 0 = disabled.
    # Kept for the AI_INFERENCE_FPS env var and for machines that cannot run every frame; the
    # frame loop processes every frame unless AI_FRAME_SKIP or AI_INFERENCE_FPS asks otherwise.
    ai_inference_fps: float = 0.0
    # Model input size (pixels). YOLO letterboxes every frame into a square imgsz, so one
    # global 640 silently destroys detection on a 4K feed: a 3840x2160 frame is scaled by
    # 1/6 and a rider's head lands at ~15 px, below what the helmet model can see. This is
    # the UPPER bound; with ai_image_size_auto the detector derives the actual size from the
    # source resolution (~1280 for 4K, 640 for anything <= 1280 px wide).
    ai_image_size: int = 1280
    # true  -> inference size follows the source resolution (see AIConfig.inference_size)
    # false -> always use ai_image_size verbatim (fixed cost, useful for benchmarks/tests)
    ai_image_size_auto: bool = True
    ai_device: str = "cpu"                            # "cpu", "0"/"1" for CUDA, "mps" for Apple silicon
    ai_tracking_enabled: bool = True                  # YOLO persistent tracking (ByteTrack) when available
    # --- Kaaval AI decision logic (temporal confirmation, not per-frame events) ---
    ai_helmet_min_confidence: float = Field(
        default=0.60,
        # The Kaaval-era name and the Google-era name are the same knob; both env vars work.
        validation_alias=AliasChoices("AI_HELMET_MIN_CONFIDENCE", "HELMET_CONFIDENCE_THRESHOLD"),
    )
    ai_helmet_confirmation_frames: int = Field(
        default=3,
        validation_alias=AliasChoices("AI_HELMET_CONFIRMATION_FRAMES", "HELMET_CONFIRMATION_OBSERVATIONS"),
    )
    ai_helmet_event_cooldown_s: float = 10.0          # min video-seconds between two events on the same camera

    # --- AI provider routing: Google Cloud primary, local YOLO fallback ---------
    # AI_PROVIDER=google  -> Cloud Video Intelligence analysis (falls back locally when the
    #                        Google APIs are not configured, if AI_ALLOW_LOCAL_FALLBACK=true)
    # AI_PROVIDER=yolo    -> the local YOLO helmet pipeline only
    ai_provider: str = "google"
    ai_allow_local_fallback: bool = True
    # HELMET_PROVIDER=yolo: Cloud Video Intelligence detects/tracks people and objects but
    # does NOT classify helmet status, so helmet status always comes from a classifier.
    helmet_provider: str = "yolo"
    # ANPR_PROVIDER=google_vision (Cloud Vision OCR) | none
    anpr_provider: str = "google_vision"
    anpr_enabled: bool = True
    anpr_min_plate_length: int = 6
    anpr_max_per_job: int = 20                        # hard bound on Vision OCR calls per job

    # --- Google Cloud (credentials must never be committed) ---------------------
    google_cloud_project_id: str = ""                 # GOOGLE_CLOUD_PROJECT_ID
    google_application_credentials: str = ""          # GOOGLE_APPLICATION_CREDENTIALS (path to the SA JSON)
    google_vision_credentials: str = ""               # GOOGLE_VISION_CREDENTIALS (optional separate key)
    google_video_region: str = ""                     # us-east1 | us-west1 | europe-west1 | asia-east1 ("" = auto)
    google_video_features: str = "PERSON_DETECTION,OBJECT_TRACKING"
    google_video_input_mode: str = "auto"             # auto | inline | gcs  (gcs = video lives at gs://)
    google_video_inline_max_mb: int = 50              # larger local files are refused with an actionable error
    google_api_timeout_s: int = 600                   # operation poll budget for one video
    google_max_observations_per_track: int = 8        # helmet classifications per tracked rider (cost bound)
    google_classification_interval_s: float = 0.25    # min video-seconds between classifications of one rider
    google_max_classifications: int = 400             # hard bound on frame extractions/classifications per job
    # --- evidence frames & topology ---
    ai_evidence_dir: str = "evidence"                 # evidence/<yyyy>/<mm>/<dd>/<event_id>.jpg
    # Optional annotated result video (boxes + status + confidence + track id + timestamp).
    # Off by default: encoding a 4K clip costs as much as the inference itself.
    ai_annotated_video: bool = False
    ai_output_dir: str = "outputs"                    # <dir>/<video-stem>_annotated.mp4
    ai_processing_node_id: str = "RDK-X5-001"         # provenance processing_node_id for locally processed events
    # Demo video -> camera map: "CAM-001=path/one.mp4;CAM-002=path/two.mp4".
    # Swap .mp4 for an RTSP URI later without changing the detector interface.
    ai_video_sources: str = "CAM-001=camera_feed/camera_main.mp4;CAM-002=camera_feed/camera_sub.mp4"


@lru_cache
def get_settings() -> Settings:
    return Settings()
