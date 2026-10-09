"""Detached machine-health logger (incident I-12: hard reset without bugcheck at 09:31 on 2026-10-07).

Every 30 s appends one line to runs/health.log and fsyncs it, so the state just before any future reset survives:
  time  tz01 C (ACPI thermal zone, CPU/chassis)  CPU %  available RAM GB  available commit GB  GPU C / W / util %
Also rewrites runs/health_now.json (latest sample) for the CPU-lane thermal guard in bwmn/lm.py.
Stops when runs/.health_stop exists. Launch: runs/start_health.sh (WMI -> bash -> python, like the orchestrators).
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import nvml  # noqa: E402

LOG = ROOT / "runs" / "health.log"
NOW = ROOT / "runs" / "health_now.json"
STOP = ROOT / "runs" / ".health_stop"
PERIOD = 30


class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def ram():
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30, m.ullAvailPageFile / 2 ** 30


def gpu():
    t = nvml.temperature()
    if t is not None:  # NVML: no subprocess per sample (incident I-14)
        return t, nvml.power_w(), nvml.utilization()
    try:
        out = subprocess.run(["C:/Windows/System32/nvidia-smi.exe",
                              "--query-gpu=temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, creationflags=0x08000000).stdout.strip()
        t, w, u = [x.strip() for x in out.splitlines()[0].split(",")]
        return int(t), float(w), int(u)
    except Exception:
        return None, None, None


def main():
    import win32pdh
    q = win32pdh.OpenQuery()
    tz_paths = win32pdh.ExpandCounterPath(r"\Thermal Zone Information(*)\Temperature")
    tz = [win32pdh.AddCounter(q, p) for p in tz_paths]
    cpu = win32pdh.AddCounter(q, r"\Processor(_Total)\% Processor Time")
    win32pdh.CollectQueryData(q)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" health logger start pid={os.getpid()} zones={tz_paths}\n")
    while not STOP.exists():
        time.sleep(PERIOD)
        try:
            sample(q, tz, cpu)
        except Exception:  # never die: a reader holding health_now.json made os.replace fail once (15:31 on 10-07)
            import traceback
            with open(ROOT / "runs" / "health.err", "a", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + traceback.format_exc())


def sample(q, tz, cpu):
    import win32pdh
    try:
        win32pdh.CollectQueryData(q)
        temps = [round(win32pdh.GetFormattedCounterValue(h, win32pdh.PDH_FMT_DOUBLE)[1] - 273.15, 1) for h in tz]
        c = round(win32pdh.GetFormattedCounterValue(cpu, win32pdh.PDH_FMT_DOUBLE)[1])
    except Exception:
        temps, c = [None], None
    phys, commit = ram()
    gt, gw, gu = gpu()
    rec = {"t": time.time(), "tz": temps[0], "cpu": c, "ram": round(phys, 2), "commit": round(commit, 2),
           "gpu_t": gt, "gpu_w": gw, "gpu_u": gu}
    line = (time.strftime("%Y-%m-%d %H:%M:%S") + f" tz={temps[0]} cpu={c}% ram={phys:.1f} commit={commit:.1f} "
            f"gpu={gt}C/{gw}W/{gu}%\n")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    tmp = NOW.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    for _ in range(10):
        try:
            os.replace(tmp, NOW)
            break
        except PermissionError:
            time.sleep(0.2)


if __name__ == "__main__":
    main()
