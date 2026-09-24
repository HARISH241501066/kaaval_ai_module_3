import hashlib
import hmac
import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from conftest import run

from sentinel_hwsim.clock import SimClock
from sentinel_hwsim.models.kaaval import KaavalNodeModel
from sentinel_hwsim.push import Pusher

SECRET = "test-secret"


class Ingest(BaseHTTPRequestHandler):
    """Minimal stand-in for M3-C4: verifies HMAC exactly as the real receiver should."""
    received: list[dict] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        ts = self.headers["X-Kaaval-Timestamp"]
        expected = hmac.new(SECRET.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
        ok = hmac.compare_digest(expected, self.headers["X-Kaaval-Signature"])
        if ok:
            Ingest.received.extend(json.loads(body)["batch"])
        self.send_response(202 if ok else 401)
        self.end_headers()

    def log_message(self, *a):
        pass


def _server():
    srv = HTTPServer(("127.0.0.1", 0), Ingest)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _node():
    return KaavalNodeModel("EDGE-NODE-001", SimClock(speed=100), random.Random(1),
                           heartbeat_interval_s=5, event_rate_multiplier=5)


def test_signed_batches_are_accepted():
    Ingest.received = []
    srv = _server()
    node = _node()
    p = Pusher(node, f"http://127.0.0.1:{srv.server_port}/ingest", SECRET, interval_s=10, dry_run=False)

    async def go():
        await node.clock.sleep(30)
        await p._deliver(node.drain()["items"])
    run(go())
    srv.shutdown()
    assert Ingest.received and p.failed_posts == 0


def test_wrong_secret_is_rejected_and_rebuffered():
    Ingest.received = []
    srv = _server()
    node = _node()
    p = Pusher(node, f"http://127.0.0.1:{srv.server_port}/ingest", "wrong", interval_s=10, dry_run=False)

    async def go():
        await node.clock.sleep(30)
        items = node.drain()["items"]
        await p._deliver(items)
        return len(items)
    n = run(go())
    srv.shutdown()
    assert not Ingest.received and p.failed_posts == 1 and len(node.outbox) >= n


def test_server_down_keeps_items_in_order():
    node = _node()
    p = Pusher(node, "http://127.0.0.1:9/unreachable", SECRET, interval_s=10, dry_run=False)

    async def go():
        await node.clock.sleep(30)
        items = node.drain()["items"]
        await p._deliver(items)
        return items
    items = run(go())
    assert list(node.outbox)[: len(items)] == items
