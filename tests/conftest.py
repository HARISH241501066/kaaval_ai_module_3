"""
Run the suite against a throw-away database, so tests never write into the tracked sentinel_m3.db.
Must run before `app` is imported (pytest loads conftest.py first). Set DATABASE_URL yourself to override.
"""
import os
import tempfile

os.environ.setdefault("DATABASE_URL", "sqlite:///" + os.path.join(tempfile.mkdtemp(prefix="m3-tests-"), "test.db").replace("\\", "/"))

# Tests must never touch the real grid, whatever is in the developer's .env: blank credentials, no background sync, no relay.
for _k, _v in {"SENTINEL_GRID_EMAIL": "", "SENTINEL_GRID_PASSWORD": "", "GRID_SYNC_INTERVAL_S": "0", "RELAY_ENABLED": "false"}.items():
    os.environ.setdefault(_k, _v)
