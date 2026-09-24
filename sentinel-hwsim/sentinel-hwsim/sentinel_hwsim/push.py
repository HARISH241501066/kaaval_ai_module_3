"""
Kaaval node in PUSH mode: POSTs HMAC-signed event/heartbeat batches to the
M3-C4 ingest endpoint, exactly like a real node would. Type chaos commands on
stdin while it runs (this is the "kill Kaaval on stage" demo control).

    python -m sentinel_hwsim.push --url http://localhost:8000/federation/ingest/EDGE-NODE-001 \
        --secret dev-secret --cameras CAM-001,CAM-002,CAM-003,CAM-004

    python -m sentinel_hwsim.push --dry-run      # print batches, no server needed

Signature (verify the same way in M3-C4):
    X-Kaaval-Timestamp: <unix seconds>
    X-Kaaval-Signature: hex(HMAC_SHA256(secret, f"{timestamp}.{raw_body}"))
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import sys
import time
import urllib.error
import urllib.request

from .clock import SimClock
from .models.kaaval import KaavalNodeModel

log = logging.getLogger("sentinel_hwsim.push")

HELP = """commands:
  crash | restart            Kaaval runtime down / restart (agent keeps heartbeating)
  stall | unstall            inference stuck at 0 fps, runtime still RUNNING
  cut | restore              event pipe down / back (node buffers, then flushes late)
  off | on                   power the whole node off / on (buffer lost)
  heat <celsius>             set ambient temperature (>= ~55 triggers throttling)
  rate <x>                   event rate multiplier
  watch <PLATE>              add plate to watchlist (WATCHLIST_HIT events)
  badrate <0..1>             fraction of garbled events (DLQ testing)
  status | help | quit"""


def sign(secret: str, timestamp: str, body: bytes) -> str:
    return hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()


def _post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> int:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


class Pusher:
    def __init__(self, node: KaavalNodeModel, url: str | None, secret: str, *, interval_s: float, dry_run: bool):
        self.node, self.url, self.secret = node, url, secret
        self.interval_s, self.dry_run = interval_s, dry_run
        self.sent_events = self.sent_heartbeats = self.failed_posts = 0
        self.running = True

    async def run(self) -> None:
        while self.running:
            items = self.node.drain(500)["items"]
            if items:
                await self._deliver(items)
            await self.node.clock.sleep(self.interval_s)

    async def _deliver(self, items: list[dict]) -> None:
        body = json.dumps({"node": self.node.node_id, "batch": items}, separators=(",", ":")).encode()
        ts = str(int(time.time()))
        headers = {"Content-Type": "application/json", "X-Kaaval-Node": self.node.node_id,
                   "X-Kaaval-Timestamp": ts, "X-Kaaval-Signature": sign(self.secret, ts, body)}
        if self.dry_run:
            for it in items:
                print(json.dumps(it))
            self._count(items)
            return
        try:
            status = await asyncio.to_thread(_post, self.url, body, headers, 5.0)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            status, err = None, e
        if status is not None and 200 <= status < 300:
            self._count(items)
            log.debug("delivered %d items (%s)", len(items), status)
            return
        # store-and-forward: put the batch back at the front, keep order
        self.failed_posts += 1
        self.node.outbox.extendleft(reversed(items))
        reason = f"HTTP {status}" if status else f"{type(err).__name__}"
        log.warning("delivery failed (%s); %d items re-buffered, outbox=%d",
                    reason, len(items), len(self.node.outbox))
        if status in (401, 403):
            log.error("ingest rejected the signature: check --secret and clock skew")

    def _count(self, items: list[dict]) -> None:
        hb = sum(1 for i in items if i.get("kind") == "heartbeat")
        self.sent_heartbeats += hb
        self.sent_events += len(items) - hb

    def status(self) -> str:
        n = self.node
        h = n.health()
        return (f"{n.node_id} powered={n.powered} runtime={n.runtime} stalled={n.inference_stalled} "
                f"pipe={'UP' if n.pipe_connected else 'CUT'} fps={h['inference']['fps']} "
                f"cpu={h['system']['cpu']}% temp={h['system']['temp_c']}C throttled={h['system']['throttled']} "
                f"model={n.model_version} outbox={len(n.outbox)} dropped={n.outbox_dropped} "
                f"sent_events={self.sent_events} sent_heartbeats={self.sent_heartbeats} failed_posts={self.failed_posts}")


def handle_command(line: str, p: Pusher) -> bool:
    n = p.node
    parts = line.strip().split()
    if not parts:
        return True
    cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else None)
    actions = {
        "crash": n.crash_runtime, "restart": n.restart_runtime,
        "stall": lambda: n.stall_inference(True), "unstall": lambda: n.stall_inference(False),
        "cut": lambda: n.cut_pipe(True), "restore": lambda: n.cut_pipe(False),
        "off": lambda: n.power(False), "on": lambda: n.power(True),
    }
    try:
        if cmd in actions:
            actions[cmd]()
        elif cmd == "heat" and arg:
            n.ambient_c = float(arg)
        elif cmd == "rate" and arg:
            n.event_rate_multiplier = float(arg)
        elif cmd == "watch" and arg:
            n.watchlist.add(arg.upper())
        elif cmd == "badrate" and arg:
            n.malformed_event_rate = float(arg)
        elif cmd == "help":
            print(HELP)
            return True
        elif cmd == "quit":
            p.running = False
            return False
        elif cmd != "status":
            print(f"unknown command {line.strip()!r}; type help")
            return True
    except ValueError as e:
        print(f"bad argument: {e}")
    print(p.status())
    return True


async def repl(p: Pusher) -> None:
    loop = asyncio.get_running_loop()
    print(HELP)
    while p.running:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:  # stdin closed (e.g. running under a process manager): keep pushing
            return
        if not handle_command(line, p):
            return


async def amain(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="M3-C4 ingest URL")
    ap.add_argument("--secret", default=os.getenv("KAAVAL_HMAC_SECRET", "dev-secret"))
    ap.add_argument("--node", default="EDGE-NODE-001")
    ap.add_argument("--node-type", default="KAAVAL_EDGE", choices=["KAAVAL_EDGE", "KAAVAL_SERVER"])
    ap.add_argument("--cameras", default="CAM-001,CAM-002,CAM-003,CAM-004")
    ap.add_argument("--functions", default="ANPR,HELMET_DETECTION,TRIPLE_RIDING,WATCHLIST_MATCH")
    ap.add_argument("--interval", type=float, default=2.0, help="push interval, simulated seconds")
    ap.add_argument("--heartbeat", type=float, default=10.0, help="heartbeat interval, simulated seconds")
    ap.add_argument("--rate", type=float, default=1.0, help="event rate multiplier")
    ap.add_argument("--clock-offset", type=float, default=0.0, help="node clock skew in seconds")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if not a.url and not a.dry_run:
        ap.error("--url is required unless --dry-run")

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    clock = SimClock(speed=a.speed)
    node = KaavalNodeModel(a.node, clock, random.Random(a.seed), node_type=a.node_type,
                           hardware="RDK X5" if a.node_type == "KAAVAL_EDGE" else "x86 GPU server",
                           cameras=a.cameras.split(","), functions=a.functions.split(","),
                           heartbeat_interval_s=a.heartbeat, event_rate_multiplier=a.rate,
                           clock_offset_s=a.clock_offset)
    pusher = Pusher(node, a.url, a.secret, interval_s=a.interval, dry_run=a.dry_run)
    log.info("node %s pushing to %s", a.node, a.url or "<dry-run stdout>")
    push_task = asyncio.create_task(pusher.run())
    await repl(pusher)
    if not pusher.running:
        push_task.cancel()
    await asyncio.gather(push_task, return_exceptions=True)


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
