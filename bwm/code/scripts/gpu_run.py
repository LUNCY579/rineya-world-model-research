"""Run a GPU script under the shared-card lease: python scripts/gpu_run.py <label> -- <script.py> [args...]

Holds C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock for the child's whole lifetime and registers the child
pid in worker_pids so the lock cannot be reclaimed as stale while the child runs.
"""
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import gpu_lease  # noqa: E402

YIELD = ROOT / "runs" / ".yield.json"


def honour_yield():
    """Explicit turn-taking with a peer session. If runs/.yield.json exists ({"label": L, "max_wait_s": S}), wait until
    the peer lease labelled L has been seen holding the lock and then released, or S seconds, whichever comes first;
    then consume the yield so later launches proceed normally."""
    if not YIELD.exists():
        return
    try:
        y = json.loads(YIELD.read_text())
    except Exception:
        return
    import time
    label = y.get("label")
    prefix = y.get("label_prefix")
    start = time.time()
    deadline = start + float(y.get("max_wait_s", 3000))
    idle_s = float(y.get("give_up_if_not_seen_s", 1e9))
    seen = False
    # D-63 (QP-2): announce BWM as a waiter during the yield too, so a peer re-call after its cap deferral sees a live
    # BWM marker and alternates (wait_for_turn rewrites the marker; acquire removes it)
    import os
    WAITING.write_text(json.dumps({"pid": os.getpid(), "label": "bwm:yield", "since": time.time()}))
    print(f"  [gpu-run] yielding to peer lease '{label or prefix + '*'}' (max {y.get('max_wait_s')} s)", flush=True)
    while time.time() < deadline:
        try:
            cur = json.loads(gpu_lease.LOCK.read_text(encoding="utf-8")).get("label")
        except Exception:
            cur = None
        match = cur is not None and ((label is not None and cur == label) or (prefix is not None and cur.startswith(prefix)))
        if match:
            seen = True
        elif seen:
            break
        elif time.time() - start > idle_s:
            break  # the peer did not take the card: it is idle
        time.sleep(10)
    print(f"  [gpu-run] yield ended (peer seen={seen})", flush=True)
    try:
        YIELD.rename(YIELD.with_name(f".yield_consumed_{int(time.time())}.json"))
    except Exception:
        pass


# Conditional one-for-one alternation with the RCWM-T1 session (their proposal, 2026-10-06):
#  - while waiting for the card, keep rineya-gpu-0.waiting.bwm.json {pid,label,since}; delete it once acquired;
#  - on every release write rineya-gpu-0.turn.json {last: "bwm", label, released_unix};
#  - before acquiring: if turn.last == "bwm" and the peer's waiting marker exists with a live pid, wait until
#    turn.last == "rcwm" (or 45 min, or their marker disappears). Otherwise acquire normally (no idle GPU).
TMP = gpu_lease.LOCK.parent
WAITING = TMP / "rineya-gpu-0.waiting.bwm.json"
PEER_WAITING = TMP / "rineya-gpu-0.waiting.rcwm.json"
TURN = TMP / "rineya-gpu-0.turn.json"


def _read(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_turn(label, bundle_start=None):
    import time
    try:
        rec = {"last": "bwm", "label": f"bwm:{label}", "released_unix": time.time()}
        if bundle_start is not None:
            rec["bundle_start"] = bundle_start
        TURN.write_text(json.dumps(rec))
    except Exception as exc:
        print(f"  [gpu-run] could not write turn file: {exc}", flush=True)


def wait_for_turn(label):
    import os, time
    WAITING.write_text(json.dumps({"pid": os.getpid(), "label": f"bwm:{label}", "since": time.time()}))
    t0 = time.time()
    announced = False
    while time.time() - t0 < 2700:
        turn = _read(TURN) or {}
        peer = _read(PEER_WAITING)
        peer_live = bool(peer) and isinstance(peer.get("pid"), int) and gpu_lease._pid_alive(peer["pid"])
        if turn.get("last") != "bwm" or not peer_live:
            break
        if not announced:
            print(f"  [gpu-run] alternation: last turn was ours and RCWM ({peer.get('label')}) is waiting; yielding", flush=True)
            announced = True
        time.sleep(15)


def main():
    label = sys.argv[1]
    honour_yield()
    wait_for_turn(label)
    import time
    # courtesy gap only between two of OUR consecutive jobs (lets a waiting peer's lease poll win that race); when the
    # last turn was the peer's, take the card at once (on 2026-10-08 the gap let a second peer job jump the queue)
    if (_read(TURN) or {}).get("last") == "bwm":
        time.sleep(30)
    assert sys.argv[2] == "--"
    cmd = [sys.executable] + sys.argv[3:]
    # bundles (scripts/bundle.py): label carries "bundle-", the lock record and turn.json carry bundle_start
    is_bundle = len(sys.argv) > 3 and sys.argv[3].replace("\\", "/").endswith("bundle.py")
    lease_label = f"bwm:bundle-{label}" if is_bundle else f"bwm:{label}"
    bundle_start = None
    with gpu_lease.lease(lease_label, " ".join(sys.argv[3:])[:200], timeout_seconds=6 * 3600) as L:
        try:
            WAITING.unlink()
        except FileNotFoundError:
            pass
        acquired = time.time()
        env = dict(__import__("os").environ, BWM_LEASE_ACQUIRED=str(acquired))
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
        try:
            rec = json.loads(gpu_lease.LOCK.read_text(encoding="utf-8"))
            if rec.get("token") == L.token:
                rec["worker_pids"] = [p.pid]
                if is_bundle:
                    bundle_start = acquired
                    rec["bundle_start"] = acquired
                    rec["bundle_budget_s"] = 16200
                gpu_lease.LOCK.write_text(json.dumps(rec), encoding="utf-8")
        except Exception as exc:  # the lease itself is still held
            print(f"  [gpu-run] could not record worker pid: {exc}", flush=True)
        rc = p.wait()
    write_turn(label, bundle_start)
    sys.exit(rc)

if __name__ == "__main__":
    main()
