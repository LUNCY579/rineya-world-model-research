"""Independent GPU safety watchdog (DECISIONS I-12, I-18, I-19). Shared by both Claude sessions.

Every 1.5 s:
  * thermal: at >= 76 C create runs/.thermal_pause; once <= 66 C (and on AC) remove it.
  * power:   on battery -> create the pause flag (no GPU work on battery) and drop the power request;
             on AC, ONLY while a GPU job holds the shared lease -> hold a Windows power request
             (SetThreadExecutionState ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED | ES_CONTINUOUS). On Modern
             Standby (S0ix) laptops the idle transition is driven by the display timeout, so DISPLAY_REQUIRED
             is what prevents it (peer finding; bugcheck 0x19C WIN32K_POWER_WATCHDOG_TIMEOUT at the 19:10
             crash). When no lease is held the request is released and the screen times out normally.
             No setting is changed; a closed lid or a user-chosen Sleep is not prevented.
  * memory (v3.2, I-23): when available host RAM < 3 GB, create the pause flag (GPU jobs stop between
             batches and none can load a model); a RAM-caused pause resumes at >= 5 GB, any other pause needs >= 3 GB (v3.5).
             Added after an unclean reset at 09:31 on 2026-10-07 that followed host-RAM exhaustion.
  * CPU zone (v3.3, agreed with the peer's CPU guard): ACPI \_TZ.TZ01 sits at ~95 C under any load (the
             CPU boost ceiling); >= 98 C sets the pause flag. Every resume needs tz01 < 98; a pause that
             tz01 itself caused also needs <= 90 C (v3.4: reason-specific, I-25). It is polled every 30 s
             in a thread, so it is a coarse guard only.
  * sensors (v3.6, I-27): GPU via NVML and tz01 via PDH, both in-process (scripts/sensors_noproc.py); no
             subprocess per sample, nvidia-smi only as a <= 2/min fallback. Spawning ~40 processes/min fed a
             kernel-memory leak on this machine (an elevated helper kept every exited process alive).
Logs one line per minute (GPU temperature, utilisation, power, AC, available RAM, CPU load, tz01) and on
every state change to runs/thermal_log.txt. Exits when runs/.thermal_stop exists.
"""
import ctypes
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAG, STOP, LOG = ROOT / "runs" / ".thermal_pause", ROOT / "runs" / ".thermal_stop", ROOT / "runs" / "thermal_log.txt"
ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
LEASE = Path(r"C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock")


class SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]


def on_ac() -> bool:
    s = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(s)):
        return True
    return s.ACLineStatus != 0          # 1 = online, 255 = unknown (desktop) -> treat as AC


import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import sensors_noproc  # noqa: E402  (v3.6, I-27: NVML/PDH in-process, no subprocess per sample)

_last_smi = [0.0]


def temp():
    t, u, p = sensors_noproc.gpu()
    if t is not None:
        return t, u, p
    if time.time() - _last_smi[0] < 30:            # NVML unavailable: slow nvidia-smi fallback, <= 2 spawns/min
        return None, None, None
    _last_smi[0] = time.time()
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout
        t, u, p = [x.strip() for x in out.splitlines()[0].split(",")]
        return int(t), int(u), float(p)
    except Exception:
        return None, None, None


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail_gb():
    """Available physical memory (includes standby), GB."""
    s = MEMORYSTATUSEX()
    s.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
        return None
    return s.ullAvailPhys / 2**30


_tz = {"c": None}


def _tz_poller():
    """ACPI thermal zone \\_TZ.TZ01 (CPU/chassis; peer finding 2026-10-07: 86 C right after the 09:31
    reset). Read in-process via PDH (v3.6) every 30 s in this thread."""
    while True:                                     # v3.6: PDH in-process (was a PowerShell spawn every 30 s)
        _tz["c"] = sensors_noproc.tz01_c()
        time.sleep(30)


threading.Thread(target=_tz_poller, daemon=True).start()
_cpu_prev = [None]


def cpu_pct():
    """System CPU load (%) since the previous call, from GetSystemTimes."""
    idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
    if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
        return "?"
    cur, prev = (idle.value, kern.value, user.value), _cpu_prev[0]
    _cpu_prev[0] = cur
    if prev is None:
        return "?"
    di, dk, du = (c - p for c, p in zip(cur, prev))
    tot = dk + du                                 # kernel time includes idle time
    return "?" if tot <= 0 else round(100.0 * (tot - di) / tot)


def _flag_reason() -> str:
    try:
        return FLAG.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def log(msg):
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


last_log, holding, ac_prev = 0.0, False, None
while not STOP.exists():
    ac = on_ac()
    want = ac and LEASE.exists()
    if want and not holding:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        holding = True
        log("power request HELD (AC + GPU lease held): idle sleep and display-off blocked")
    elif not want and holding:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        holding = False
        log(f"power request RELEASED ({'battery' if not ac else 'no GPU lease held'})")
    if ac != ac_prev:
        log(f"AC power: {ac}")
        ac_prev = ac
    t, u, p = temp()
    ram = avail_gb()
    tz = _tz["c"]
    state = None
    if not ac and not FLAG.exists():
        FLAG.write_text("battery"); state = "PAUSE (battery)"
    elif t is not None and t >= 76 and not FLAG.exists():
        FLAG.write_text(str(t)); state = "PAUSE"
    elif ram is not None and ram < 3.0 and not FLAG.exists():
        FLAG.write_text(f"ram {ram:.1f}"); state = f"PAUSE (RAM {ram:.1f} GB)"
    elif tz is not None and tz >= 98 and not FLAG.exists():
        FLAG.write_text(f"tz01 {tz}"); state = f"PAUSE (tz01 {tz} C)"
    elif (FLAG.exists() and ac and t is not None and t <= 66
          and (ram is None or ram >= 3.0) and (ram is None or ram >= 5.0 or not _flag_reason().startswith("ram"))
          and (tz is None or tz < 98) and (tz is None or tz <= 90 or not _flag_reason().startswith("tz01"))):
        # Hysteresis is reason-specific: RAM >= 5 GB only for pauses low RAM caused, tz01 <= 90 only for pauses
        # tz01 caused. Otherwise a GPU-temperature pause stalls while RAM sits at 3-5 GB (05:55-06:05, I-26) or
        # tz01 at its ~95 C load ceiling (14:03-14:35, I-25). Every resume still needs both out of their pause zone.
        FLAG.unlink(); state = "RESUME"
    now = time.time()
    if state or now - last_log >= 60:
        ram_s = "?" if ram is None else f"{ram:.1f}"
        log(f"t={t} util={u} power={p} ac={ac} ram={ram_s} cpu={cpu_pct()} tz01={_tz['c']}{' ' + state if state else ''}"
            f"{' (paused)' if FLAG.exists() and not state else ''}")
        last_log = now
    time.sleep(1.5)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
log("watchdog stopped; power request released")
