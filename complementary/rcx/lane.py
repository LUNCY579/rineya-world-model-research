"""Shared single-CPU-lane lock (agreed with BWM, D-76 / CX D-36): one heavy CPU job on the machine at a time (BWM I-12).

Files in %TEMP% (override with RINEYA_LANE_DIR, tests only):
  rineya-cpu-lane.json            the lock: {holder, job, pid, started, expected_end, token}; created with O_EXCL
  rineya-cpu-lane.waiting.<party> a party that wants the lane; the holder yields at its next safe point
Staleness: a lock whose holder pid is dead is reclaimable (renamed to rineya-cpu-lane.json.stale-<pid>.bak).
`pid` is the Windows pid of the long-lived process that owns the job (for a Git Bash chain: `cat /proc/$$/winpid`).

RCX acquire policy: if another party holds the lane, write our waiting marker and wait (its job yields between
instances); if the lane is free, take it only when available RAM >= need_ram (GPU turns can hold RAM; waiting then
blocks nobody, so no marker is written). CLI (from complementary/):
  python -B -m rcx.lane acquire --job "E01R step A" --pid <winpid> --expected-min 100 [--need-ram 12.5]
  python -B -m rcx.lane release --pid <winpid>
  python -B -m rcx.lane status
"""
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True

PARTY = "rcx"


def lane_dir():
    return Path(os.environ.get("RINEYA_LANE_DIR") or os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp")


def lock_path():
    return lane_dir() / "rineya-cpu-lane.json"


def waiting_path(party=PARTY):
    return lane_dir() / f"rineya-cpu-lane.waiting.{party}"


def pid_alive(pid):
    try:
        import psutil
        return psutil.pid_exists(int(pid))
    except ImportError:  # Windows fallback without psutil; never os.kill (it terminates on Windows)
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return ctypes.windll.kernel32.GetLastError() == 5  # access denied: exists
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE


def read_lock():
    try:
        return json.loads(lock_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _reclaim_if_stale(lock):
    if lock and not pid_alive(lock.get("pid", -1)):
        try:
            lock_path().rename(lock_path().with_name(f"rineya-cpu-lane.json.stale-{lock.get('pid')}.bak"))
        except OSError:
            pass
        return True
    return False


def try_acquire(job, pid, expected_min, party=PARTY):
    rec = {"holder": party, "job": job, "pid": int(pid), "started": time.time(),
           "expected_end": time.time() + 60 * float(expected_min), "token": uuid.uuid4().hex}
    try:
        fd = os.open(lock_path(), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    return rec


def acquire(job, pid, expected_min, need_ram=None, poll_s=60, log=print, party=PARTY, ram_fn=None, max_wait_s=None):
    """Block until the lane is ours; returns the lock record."""
    if ram_fn is None and need_ram is not None:
        from rcx.guard import available_ram_gb as ram_fn
    t0, said = time.time(), None
    while True:
        lock = read_lock()
        if lock and _reclaim_if_stale(lock):
            log(f"lane: reclaimed stale lock of {lock.get('holder')} pid {lock.get('pid')}")
            lock = None
        if lock and lock.get("holder") == party and int(lock.get("pid", -1)) == int(pid):
            waiting_path(party).unlink(missing_ok=True)
            return lock
        if lock:
            waiting_path(party).write_text(json.dumps({"party": party, "job": job, "since": time.time()}),
                                           encoding="utf-8")
            msg = f"lane: held by {lock.get('holder')} ({lock.get('job')}); waiting marker written"
        else:
            ram = ram_fn() if need_ram is not None else None
            if need_ram is None or (ram or 0) >= need_ram:
                rec = try_acquire(job, pid, expected_min, party)
                if rec:
                    waiting_path(party).unlink(missing_ok=True)
                    log(f"lane: acquired for {job!r}")
                    return rec
                continue
            waiting_path(party).unlink(missing_ok=True)
            msg = f"lane: free, but RAM {ram:.1f} GB < {need_ram} GB; waiting (no marker)"
        if msg != said:
            log(msg)
            said = msg
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            raise TimeoutError(msg)
        time.sleep(poll_s)


def release(pid, party=PARTY):
    lock = read_lock()
    if lock and lock.get("holder") == party and int(lock.get("pid", -1)) == int(pid):
        lock_path().unlink(missing_ok=True)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--job", required=True)
    a.add_argument("--pid", type=int, required=True)
    a.add_argument("--expected-min", type=float, required=True)
    a.add_argument("--need-ram", type=float, default=None)
    r = sub.add_parser("release")
    r.add_argument("--pid", type=int, required=True)
    sub.add_parser("status")
    args = ap.parse_args()
    if args.cmd == "acquire":
        def log(m):
            print(time.strftime("%H:%M:%S"), m, flush=True)
        acquire(args.job, args.pid, args.expected_min, args.need_ram, log=log)
    elif args.cmd == "release":
        print("released" if release(args.pid) else "not held by this pid")
    else:
        print(json.dumps({"lock": read_lock(), "waiting": sorted(p.name for p in lane_dir().glob(
            "rineya-cpu-lane.waiting.*"))}, indent=1))


if __name__ == "__main__":
    main()
