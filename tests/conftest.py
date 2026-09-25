"""
Run the suite against a throw-away database, so tests never write into the tracked sentinel_m3.db.
Must run before `app` is imported (pytest loads conftest.py first). Set DATABASE_URL yourself to override.
"""
import os
import tempfile

os.environ.setdefault("DATABASE_URL", "sqlite:///" + os.path.join(tempfile.mkdtemp(prefix="m3-tests-"), "test.db").replace("\\", "/"))
