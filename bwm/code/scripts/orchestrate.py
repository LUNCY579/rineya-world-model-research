"""Restart-safe sequential job runner for the shared GPU.

Reads runs/jobs.json on every iteration (so jobs can be appended while it runs). A job is
  {"name": str, "marker": relative path that exists when the job is done, "cmd": [script, args...], "log": relative path}
Each pending job runs under the shared GPU lease (scripts/gpu_run.py). Exit code 0 -> marker is touched (unless the
job writes it itself). Failure -> output preserved under runs/_failed and one retry; a second failure writes
runs/_failed/<name>.failed and the job is skipped. Stops when every job is done or failed, or when runs/.stop exists.

Launch detached (survives the launching session):
  powershell -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='<python> scripts/orchestrate.py'; CurrentDirectory='<native_v1>'}"
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANE = "cpu2" if "--cpu2" in sys.argv else "cpu" if "--cpu" in sys.argv else "gpu"
CPU = LANE.startswith("cpu")  # CPU lanes: no GPU lease; jobs run with BWM_DEVICE=cpu, 8 threads, no duty cap
_SUFFIX = "" if LANE == "gpu" else "_" + LANE
JOBS = ROOT / "runs" / f"jobs{_SUFFIX}.json"
LOG = ROOT / "runs" / f"orchestrator{_SUFFIX}.log"
FAILED = ROOT / "runs" / "_failed"
STOP = ROOT / "runs" / f".stop{_SUFFIX}"


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg + "\n")


def harden():
    """Immune to console control events: a Ctrl-C delivered to a shared console killed two earlier runs."""
    import signal
    for sig in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), signal.SIG_IGN)
    sys.stderr = open(ROOT / "runs" / f"orchestrator{_SUFFIX}.err", "a", encoding="utf-8",
                      buffering=1)


# new process group (Ctrl-C disabled for the whole job tree) and no console window
CHILD_FLAGS = 0x00000200 | 0x08000000 if os.name == "nt" else 0


ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002


def keep_awake(on: bool) -> None:
    """User-approved (2026-10-06): while a job runs, hold 'system + display required' so the laptop does not enter
    Modern Standby (a standby transition under load bugchecked the machine with 0x19C at 19:10). Released when idle."""
    if os.name != "nt":
        return
    import ctypes
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def main():
    harden()
    FAILED.mkdir(parents=True, exist_ok=True)
    log(f"orchestrator start pid={os.getpid()}")
    attempts: dict[str, int] = {}
    while True:
        if STOP.exists():
            log("stop file present; exiting")
            return
        try:
            jobs = json.loads(JOBS.read_text(encoding="utf-8"))
        except Exception as exc:
            log(f"cannot read jobs.json: {exc}; retry in 60 s")
            time.sleep(60)
            continue
        pending = [j for j in jobs if not (ROOT / j["marker"]).exists() and not (FAILED / f"{j['name']}.failed").exists()]
        if not pending:
            keep_awake(False)
            log("no pending jobs; exiting")
            return
        # optional dependency: a job with "after": <marker> runs only once that marker exists
        runnable = [j for j in pending if not j.get("after") or (ROOT / j["after"]).exists()]
        if not runnable:
            keep_awake(False)
            time.sleep(120)
            continue
        j = runnable[0]
        keep_awake(True)
        name = j["name"]
        out_log = ROOT / j.get("log", f"runs/{name}.stdout")
        out_log.parent.mkdir(parents=True, exist_ok=True)
        log(f"start {name}")
        t0 = time.time()
        with open(out_log, "a", encoding="utf-8") as fh:
            if CPU:
                # I-12 (hard reset 2026-10-07 09:31 under two 8-thread lanes): 6 threads, one lane at a time, and a
                # 12 GB free-RAM floor before each model load (>= 6 GB stays free for either session's GPU job).
                env = dict(os.environ, BWM_DEVICE="cpu", BWM_DUTY_RATIO="0", BWM_YIELD_S="0", OMP_NUM_THREADS="6",
                           MKL_NUM_THREADS="6", CUDA_VISIBLE_DEVICES="", BWM_RAM_FLOOR_GB="12")
                p = subprocess.Popen([sys.executable] + j["cmd"], cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                     creationflags=CHILD_FLAGS, env=env)
            else:
                p = subprocess.Popen([sys.executable, "scripts/gpu_run.py", name, "--"] + j["cmd"], cwd=str(ROOT),
                                     stdout=fh, stderr=subprocess.STDOUT, creationflags=CHILD_FLAGS)
            log(f"  child pid {p.pid}")
            while True:
                try:
                    rc = p.wait(timeout=600)
                    break
                except subprocess.TimeoutExpired:
                    log(f"  heartbeat {name} {time.time() - t0:.0f}s")
        dt = time.time() - t0
        marker = ROOT / j["marker"]
        if rc == 0:
            if not marker.exists():
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(json.dumps({"job": name, "finished": time.time(), "wall_s": dt}))
            log(f"done {name} in {dt:.0f}s")
            continue
        if rc == 75:  # scripts/bundle.py: the turn's 4.5 h budget ended with sub-jobs left -> requeue, not a failure
            log(f"yielded {name} after {dt:.0f}s (bundle budget used; continues next turn)")
            time.sleep(60)
            continue
        attempts[name] = attempts.get(name, 0) + 1
        stamp = time.strftime("%H%M%S")
        try:
            shutil.copy(out_log, FAILED / f"{name}_{stamp}.stdout")
        except Exception:
            pass
        log(f"FAILED {name} rc={rc} after {dt:.0f}s (attempt {attempts[name]})")
        if attempts[name] >= 2:
            (FAILED / f"{name}.failed").write_text(json.dumps({"rc": rc, "attempts": attempts[name]}))
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        log("orchestrator crashed: " + traceback.format_exc())
        raise
