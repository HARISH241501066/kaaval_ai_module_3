"""
    python -m app.relay fetch     # once: download MediaMTX into relay/bin (SHA-256 verified)
    python -m app.relay config    # write relay/generated/mediamtx.yml from the camera registry
    python -m app.relay run       # write the config, then run MediaMTX with it (Ctrl+C to stop)

The registry is filled by the M3 API (POST /api/v1/grid/sync, or its background sync). If it is empty,
`config` and `run` fill it themselves from the grid catalogue first.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import platform
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import httpx

from app.core.config import get_settings

VERSION = "v1.21.1"
BASE = f"https://github.com/bluenviron/mediamtx/releases/download/{VERSION}"
PINNED = {                                    # from the release's own checksums.sha256
    f"mediamtx_{VERSION}_windows_amd64.zip": "faa97974861eb75a68b5aa326c78e7e7a6f670b5ef191bace78e715130381f23",
    f"mediamtx_{VERSION}_linux_amd64.tar.gz": "653abc672a3e693f8d3b2717752492fdcfb8072291ec108d03d3dd857411b0ee",
    f"mediamtx_{VERSION}_darwin_arm64.tar.gz": "25e20ed41611f1f3103b8359585210b29b11b69fa0d9e11bd11b92f7bbcb42ef",
}


def _asset() -> str:
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    return f"mediamtx_{VERSION}_{system}_{arch}." + ("zip" if system == "windows" else "tar.gz")


def fetch() -> int:
    s = get_settings()
    dest = Path(s.relay_bin)
    asset = _asset()
    print(f"downloading {asset} ...")
    with httpx.Client(follow_redirects=True, timeout=120) as http:
        data = http.get(f"{BASE}/{asset}").raise_for_status().content
        expected = PINNED.get(asset)
        if expected is None:                                    # other platforms: the release's published checksum list
            for line in http.get(f"{BASE}/checksums.sha256").raise_for_status().text.splitlines():
                digest, _, name = line.partition(" *")
                if name.strip() == asset:
                    expected = digest
    got = hashlib.sha256(data).hexdigest()
    if not expected or got != expected:
        print(f"CHECKSUM MISMATCH for {asset}: expected {expected}, got {got}. Not installing.", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    member = dest.name
    if asset.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            dest.write_bytes(z.read(member))
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            dest.write_bytes(t.extractfile("mediamtx").read())
    dest.chmod(0o755)
    print(f"installed {dest} ({VERSION}, sha256 verified)")
    return 0


def _prepare() -> tuple[Path, int]:
    from app.adapters.sentinel_grid import SentinelGridAdapter  # noqa: F401  (registers nothing; import check)
    from app.db.database import SessionLocal, init_db
    from app.live.redact import Redactor, install_log_redaction
    from app.models.grid import CameraMappingModel
    from app.relay.manager import RelayManager
    from app.services import grid_sync

    init_db()
    mgr = RelayManager()
    install_log_redaction(Redactor(*mgr.adapter.grid.secrets()))
    with SessionLocal() as db:
        if not db.query(CameraMappingModel).count():
            print("the registry is empty: reading the grid catalogue first ...")
            grid_sync.register_system(db, mgr.adapter)
            grid_sync.sync_inventory(db, mgr.adapter)
        path = mgr.write_config(db)
        count = db.query(CameraMappingModel).filter_by(status="ACTIVE").count()
    return path, count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.relay")
    ap.add_argument("command", choices=["fetch", "config", "run"])
    cmd = ap.parse_args(argv).command
    if cmd == "fetch":
        return fetch()
    path, count = _prepare()
    print(f"wrote {path} for {count} camera(s) (contains the grid login: git-ignored, do not share)")
    if cmd == "config":
        return 0
    s = get_settings()
    exe = Path(s.relay_bin)
    if not exe.exists():
        print(f"MediaMTX not found at {exe}. Run: python -m app.relay fetch", file=sys.stderr)
        return 2
    try:
        return subprocess.call([str(exe), str(path)])
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
