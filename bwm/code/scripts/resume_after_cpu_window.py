"""One-shot: keep the GPU idle while a peer's CPU-lane job runs (D-76 lane lock), then clear BWM's own stale GPU lock
and relaunch the BWM orchestrator.

Used after the 2026-10-11 21:54 reboot (D-79): RCX's E01R step A (operator-approved GPU-idle window) holds
%TEMP%/rineya-cpu-lane.json. Waits until that lock is gone or its holder pid is dead, or until --deadline (unix
seconds). It then archives rineya-gpu-0.lock only if it is BWM's own (label starts with "bwm:") and both its holder pid
and every worker pid are dead, and starts runs/start_orchestrator.sh. Logs to runs/resume_after_cpu_window.log.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(os.environ.get("TEMP", "C:/Users/clclc/AppData/Local/Temp"))
LANE = TMP / "rineya-cpu-lane.json"
GPU_LOCK = TMP / "rineya-gpu-0.lock"
LOG = ROOT / "runs/resume_after_cpu_window.log"


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION; never os.kill
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def read(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", type=float, required=True)
    a = ap.parse_args()
    log(f"start; deadline {time.strftime('%H:%M', time.localtime(a.deadline))}")
    while time.time() < a.deadline:
        lane = read(LANE)
        if not lane or lane.get("holder") == "bwm" or not pid_alive(lane.get("pid")):
            log(f"lane free (lock={'none' if not lane else lane.get('holder')})")
            break
        time.sleep(30)
    else:
        log("deadline reached with the lane still held; proceeding (RCX's 2.5 h fallback)")
    rec = read(GPU_LOCK)
    if rec:
        label = str(rec.get("label", ""))
        pids = [rec.get("pid")] + list(rec.get("worker_pids") or [])
        if label.startswith("bwm:") and not any(pid_alive(p) for p in pids):
            dst = GPU_LOCK.with_name(f"rineya-gpu-0.lock.stale-{rec.get('pid')}.bak")
            GPU_LOCK.replace(dst)
            log(f"archived own stale lock {label} pids {pids} -> {dst.name}")
        else:
            log(f"GPU lock is not BWM's stale lock ({label}, live={[p for p in pids if pid_alive(p)]}); not touching it")
    out = subprocess.run(["C:/Program Files/Git/bin/bash.exe", "runs/start_orchestrator.sh"], cwd=str(ROOT),
                         capture_output=True, text=True)
    log(f"orchestrator relaunch: rc {out.returncode} {out.stdout.strip()[:200]} {out.stderr.strip()[:200]}")


if __name__ == "__main__":
    main()
