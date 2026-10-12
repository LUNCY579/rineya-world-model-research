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


def honour_yield(clock=None, sleep=None):
    """Explicit turn-taking with a peer session. If runs/.yield.json exists ({"label": L, "max_wait_s": S}), wait until
    the peer lease labelled L has been seen holding the lock and then released, or S seconds, whichever comes first;
    then consume the yield so later launches proceed normally. D-84: inside a quiet window the peer cannot take the card
    (RCWM's night guard), so the yield's clocks are frozen there and a window never uses the yield up."""
    if not YIELD.exists():
        return
    y = _read(YIELD)
    if not isinstance(y, dict) or not (y.get("label") or y.get("label_prefix")):
        print(f"  [gpu-run] ignoring an unreadable or label-less {YIELD.name}", flush=True)
        return
    import time
    clock, sleep = clock or time.time, sleep or time.sleep
    label = y.get("label")
    prefix = y.get("label_prefix")
    start = clock()
    try:
        max_wait = float(y.get("max_wait_s") or 3000)
        idle_s = float(y.get("give_up_if_not_seen_s") or 1e9)
    except (TypeError, ValueError):
        max_wait, idle_s = 3000.0, 1e9
    deadline = start + max_wait
    seen = False
    # D-63 (QP-2): announce BWM as a waiter during the yield too, so a peer re-call after its cap deferral sees a live
    # BWM marker and alternates (wait_for_turn rewrites the marker; acquire removes it)
    _mark_waiting("bwm:yield")
    print(f"  [gpu-run] yielding to peer lease '{label or prefix + '*'}' (max {y.get('max_wait_s')} s)", flush=True)
    while clock() < deadline:
        now = clock()
        if not seen and _quiet_end_safe(now) is not None:   # D-84: frozen inside a window
            start, deadline = now, max(deadline, now + max_wait)
        try:
            cur = json.loads(gpu_lease.LOCK.read_text(encoding="utf-8")).get("label")
        except Exception:
            cur = None
        match = cur is not None and ((label is not None and cur == label) or (prefix is not None and cur.startswith(prefix)))
        if match:
            seen = True
        elif seen:
            break
        elif clock() - start > idle_s:
            break  # the peer did not take the card: it is idle
        sleep(10)
    print(f"  [gpu-run] yield ended (peer seen={seen})", flush=True)
    try:
        YIELD.rename(YIELD.with_name(f".yield_consumed_{int(clock())}.json"))
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

# D-84: agreed GPU-idle night windows (RCX's E01R CPU trainings need >= 12.5 GB free RAM, which a GPU bundle leaves at
# about 10 GB; RCWM honours the same windows with its own night guard, which stops its chain from 22:50 to 07:00).
#  - No NEW BWM GPU job starts inside a listed window (local time, start inclusive, end exclusive); a running job
#    finishes. The list is dated, so it expires by itself; RCX ends it early by writing QUIET_RELEASE. A missing or
#    unreadable list never blocks the card (fail open), but every launch logs what it read.
#  - After a window: while RCX's CPU-lane block that started inside it still runs, BWM does not load a GPU job next to
#    the training (31 GB box; a bundle takes about 8 GB; I-12), for at most LANE_HOLD_S after the window's end and not
#    once RCX has released the windows; and, independent of the release (RCWM's guard runs regardless), if BWM had the
#    last turn, BWM gives RCWM's guard-relaunched chain up to GRACE_S to claim the turn it is owed.
QUIET = ROOT / "runs" / "gpu_quiet_windows.json"
QUIET_RELEASE = TMP / "rineya-quiet-release.rcx.json"
LANE = TMP / "rineya-cpu-lane.json"
GRACE_S = 900.0
LANE_HOLD_S = 4 * 3600.0


def _read(p):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _mark_waiting(label):
    """The waiting marker only advises RCWM's alternation; failing to write it must never fail the GPU job."""
    import os, time
    try:
        WAITING.write_text(json.dumps({"pid": os.getpid(), "label": label, "since": time.time()}))
    except OSError as exc:
        print(f"  [gpu-run] could not write the waiting marker: {exc}", flush=True)


def _local_unix(s):
    import time
    return time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M"))


MAX_WINDOW_S = 12 * 3600.0   # a longer window is a typo (e.g. a wrong end date), not a night: skipped as malformed


def _windows(cfg=None):
    """[(start, end)] of the parseable windows; malformed entries are skipped (report_quiet_list shows them)."""
    cfg = _read(QUIET) if cfg is None else cfg
    ws = cfg.get("windows") if isinstance(cfg, dict) else None
    out = []
    for w in ws if isinstance(ws, list) else []:
        try:
            a, b = _local_unix(w["start"]), _local_unix(w["end"])
        except Exception:
            continue
        if 0 < b - a <= MAX_WINDOW_S:
            out.append((a, b))
    return out


def quiet_window_end(now, cfg=None, release=None):
    """Unix end of the agreed window containing `now`, or None: no window, the list is missing or unreadable, or the
    peer has released the windows."""
    release = QUIET_RELEASE if release is None else Path(release)
    if release.exists():
        return None
    for a, b in _windows(cfg):
        if a <= now < b:
            return b
    return None


def _quiet_end_safe(now):
    try:
        return quiet_window_end(now)
    except Exception:   # fail open
        return None


def last_window_end(now, cfg=None):
    """End of the most recent window that has ended by `now` (the release marker does not matter here), or None."""
    ends = [b for a, b in _windows(cfg) if b <= now]
    return max(ends) if ends else None


def report_quiet_list(now=None):
    """One line per launch, so a missing, unreadable or released list is visible in the job log."""
    import time
    now = time.time() if now is None else now
    if not QUIET.exists():
        print("  [gpu-run] D-84: no windows file; quiet windows off", flush=True)
        return
    cfg = _read(QUIET)
    ws = cfg.get("windows") if isinstance(cfg, dict) else None
    if not isinstance(ws, list):
        print("  [gpu-run] D-84: windows file UNREADABLE; failing open (no quiet windows)", flush=True)
        return
    ok = _windows(cfg)
    nxt = min((a for a, b in ok if b > now), default=None)
    print(f"  [gpu-run] D-84: {len(ok)} window(s), {len(ws) - len(ok)} skipped as malformed, next start "
          f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(nxt)) if nxt else 'none'}"
          f"{'; RELEASED by RCX' if QUIET_RELEASE.exists() else ''}", flush=True)


def honour_quiet_windows(label, clock=None, sleep=None, cfg=None, release=None):
    """Block while `now` is inside an agreed window (re-reading the list and the release marker every minute). Keeps
    BWM's waiting marker meanwhile (D-63 practice; RCWM confirmed its side ignores it at night and yields to it at 07:00
    when it went last). Returns True if it waited."""
    import time
    clock, sleep = clock or time.time, sleep or time.sleep
    end = quiet_window_end(clock(), cfg, release)
    if end is None:
        return False
    _mark_waiting("bwm:quiet-window")
    print(f"  [gpu-run] D-84 quiet window until {time.strftime('%Y-%m-%d %H:%M', time.localtime(end))}: "
          f"not starting {label}", flush=True)
    last_end = end
    while end is not None:
        last_end = end
        sleep(min(60.0, max(1.0, end - clock())))
        end = quiet_window_end(clock(), cfg, release)
    print(f"  [gpu-run] quiet window over at {time.strftime('%H:%M:%S', time.localtime(clock()))}", flush=True)
    if clock() < last_end:   # ended early: released by RCX, list edited, or list unreadable (fail open) - say which
        print("  [gpu-run] D-84: the window ended EARLY; the list now reads:", flush=True)
        report_quiet_list(clock())
    return True


def _pid_alive_since(pid, started):
    """gpu_lease's liveness, except that a live pid whose process was created after the lock (started + 2 s) is a reused
    pid, i.e. the holder is dead (scripts/planbench_qwen.py rule). Unknown -> alive."""
    if not gpu_lease._pid_alive(pid):
        return False
    try:
        import psutil
        return psutil.Process(pid).create_time() <= float(started) + 2.0
    except Exception:
        return True


def _lane_held_by_rcx(started_by):
    """RCX's live CPU-lane block that started by `started_by` (it straddles the window's end). Only RCX, the party
    D-84 was agreed with; nothing once RCX has released the windows (its chain then has no more E01R blocks)."""
    rec = _read(LANE)
    if not isinstance(rec, dict) or rec.get("holder") != "rcx" or not isinstance(rec.get("pid"), int) or \
            QUIET_RELEASE.exists():
        return False
    if not (isinstance(rec.get("started"), (int, float)) and rec["started"] <= started_by):
        return False
    return _pid_alive_since(rec["pid"], rec["started"])


def hold_for_peer_lane_after_window(clock=None, sleep=None, cfg=None):
    """After a window, do not load a GPU job beside RCX's CPU-lane block that started inside it and is still running
    (RAM), for at most LANE_HOLD_S after the window's end. Blocks started after the window are outside the agreement and
    do not hold the card. This is BWM's own courtesy: BWM withdraws its waiting marker meanwhile, so RCWM (whose yield to
    a live BWM marker is capped at 2700 s) makes its own call and the card never idles while both sides wait; BWM
    re-registers in wait_for_turn afterwards. Returns True if it waited."""
    import os, time
    clock, sleep = clock or time.time, sleep or time.sleep
    end = last_window_end(clock(), cfg)
    if end is None or clock() >= end + LANE_HOLD_S or not _lane_held_by_rcx(started_by=end):
        return False
    try:
        if (_read(WAITING) or {}).get("pid") == os.getpid():
            WAITING.unlink()
    except OSError as exc:
        print(f"  [gpu-run] could not withdraw the waiting marker: {exc}", flush=True)
    rec = _read(LANE) or {}
    print(f"  [gpu-run] D-84: the window ended but {rec.get('holder')} still holds the CPU lane ({rec.get('job')}); "
          f"holding until it releases, at most until "
          f"{time.strftime('%H:%M', time.localtime(end + LANE_HOLD_S))}", flush=True)
    while clock() < end + LANE_HOLD_S and _lane_held_by_rcx(started_by=end):
        sleep(60.0)
    print(f"  [gpu-run] lane hold over at {time.strftime('%H:%M:%S', time.localtime(clock()))}", flush=True)
    return True


def _rcwm_claimed():
    """RCWM is back in the alternation: a live waiting marker, an RCWM lease, or a turn record that is not ours."""
    peer = _read(PEER_WAITING)
    if peer and isinstance(peer.get("pid"), int) and gpu_lease._pid_alive(peer["pid"]):
        return True
    lock = _read(gpu_lease.LOCK)
    if lock and str(lock.get("label", "")).startswith("rcwm"):
        return True
    return (_read(TURN) or {}).get("last") != "bwm"


def post_window_grace(clock=None, sleep=None, cfg=None):
    """RCWM's night guard kills its waiting job (leaving a dead marker) and relaunches its chain only at 07:00, so a
    plain turn check at 07:00 would let BWM take a second turn in a row. If BWM went last, wait up to GRACE_S after the
    window's end for RCWM to claim the card or register as a waiter. Returns True if it waited."""
    import time
    clock, sleep = clock or time.time, sleep or time.sleep
    end = last_window_end(clock(), cfg)
    if end is None or clock() >= end + GRACE_S or _rcwm_claimed():
        return False
    print(f"  [gpu-run] D-84: BWM went last; giving RCWM until "
          f"{time.strftime('%H:%M', time.localtime(end + GRACE_S))} to claim its turn", flush=True)
    while clock() < end + GRACE_S and not _rcwm_claimed():
        sleep(10.0)
    print(f"  [gpu-run] grace over at {time.strftime('%H:%M:%S', time.localtime(clock()))} "
          f"(RCWM claimed={_rcwm_claimed()})", flush=True)
    return True


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
    import time
    _mark_waiting(f"bwm:{label}")
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


def acquire_outside_quiet(label, lease_label, job, acquire=None, release=None, clock=None, sleep=None):
    """Quiet windows, the post-window lane hold and grace, alternation, courtesy gap, then the lease. Waiting for a
    peer's lease can run into a window, so the window is checked again once the card is ours: inside one, the card is
    released unused (turn.json untouched) and we wait again."""
    import time
    acquire, release = acquire or gpu_lease.acquire, release or gpu_lease.release
    clock, sleep = clock or time.time, sleep or time.sleep
    while True:
        honour_quiet_windows(label, clock=clock, sleep=sleep)
        hold_for_peer_lane_after_window(clock=clock, sleep=sleep)
        post_window_grace(clock=clock, sleep=sleep)
        wait_for_turn(label)
        # courtesy gap only between two of OUR consecutive jobs (lets a waiting peer's lease poll win that race); when
        # the last turn was the peer's, take the card at once (on 2026-10-08 the gap let a second peer job jump the queue)
        if (_read(TURN) or {}).get("last") == "bwm":
            sleep(30)
        token = acquire(lease_label, job, timeout_seconds=6 * 3600)
        try:
            inside = quiet_window_end(clock()) is not None
        except Exception as exc:   # fail open, and never leave the token unreleased
            print(f"  [gpu-run] D-84: post-acquire window check failed ({exc}); failing open", flush=True)
            inside = False
        if not inside:
            return token
        release(token)
        print("  [gpu-run] the card came free inside a D-84 quiet window; released it unused", flush=True)


def main():
    label = sys.argv[1]
    report_quiet_list()
    honour_yield()
    import time
    assert sys.argv[2] == "--"
    cmd = [sys.executable] + sys.argv[3:]
    # bundles (scripts/bundle.py): label carries "bundle-", the lock record and turn.json carry bundle_start
    is_bundle = len(sys.argv) > 3 and sys.argv[3].replace("\\", "/").endswith("bundle.py")
    lease_label = f"bwm:bundle-{label}" if is_bundle else f"bwm:{label}"
    bundle_start = None
    token = acquire_outside_quiet(label, lease_label, " ".join(sys.argv[3:])[:200])
    try:
        try:
            WAITING.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:   # e.g. a sharing violation while a peer reads it: advisory only, never fail the job
            print(f"  [gpu-run] could not remove the waiting marker: {exc}", flush=True)
        acquired = time.time()
        env = dict(__import__("os").environ, BWM_LEASE_ACQUIRED=str(acquired))
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
        try:
            rec = json.loads(gpu_lease.LOCK.read_text(encoding="utf-8"))
            if rec.get("token") == token:
                rec["worker_pids"] = [p.pid]
                if is_bundle:
                    bundle_start = acquired
                    rec["bundle_start"] = acquired
                    rec["bundle_budget_s"] = 16200
                gpu_lease.LOCK.write_text(json.dumps(rec), encoding="utf-8")
        except Exception as exc:  # the lease itself is still held
            print(f"  [gpu-run] could not record worker pid: {exc}", flush=True)
        rc = p.wait()
    finally:
        gpu_lease.release(token)
    write_turn(label, bundle_start)
    sys.exit(rc)

if __name__ == "__main__":
    main()
