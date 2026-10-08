"""In-process sensor readers for the watchdog (I-27): no subprocess per sample.

Spawning nvidia-smi every 1.5 s and PowerShell every 30 s created ~40 short-lived processes per minute. With an
elevated helper on this machine holding a handle to every exited process (RevoUninHelper, 380k handles on
2026-10-08), each one stays in kernel memory, and commit charge ran out overnight.

  gpu()    -> (temp_C, util_pct, power_W) via NVML (nvml.dll; management API, no CUDA context)
  tz01_c() -> ACPI \\_TZ.TZ01 temperature in C via PDH (pdh.dll)
"""
from __future__ import annotations

import ctypes
from ctypes import byref, c_uint, c_ulong, c_void_p, wintypes


class _Util(ctypes.Structure):
    _fields_ = [("gpu", c_uint), ("memory", c_uint)]


class _NVML:
    def __init__(self):
        self.lib = None
        self.handle = c_void_p()
        for path in ("nvml.dll", r"C:\Windows\System32\nvml.dll",
                     r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"):
            try:
                self.lib = ctypes.WinDLL(path)
                break
            except OSError:
                continue
        if self.lib is None:
            raise OSError("nvml.dll not found")
        if self.lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit_v2 failed")
        if self.lib.nvmlDeviceGetHandleByIndex_v2(c_uint(0), byref(self.handle)) != 0:
            raise OSError("nvmlDeviceGetHandleByIndex_v2 failed")

    def read(self):
        t, p, u = c_uint(), c_uint(), _Util()
        temp = t.value if self.lib.nvmlDeviceGetTemperature(self.handle, c_uint(0), byref(t)) == 0 else None
        util = u.gpu if self.lib.nvmlDeviceGetUtilizationRates(self.handle, byref(u)) == 0 else None
        power = round(p.value / 1000.0, 2) if self.lib.nvmlDeviceGetPowerUsage(self.handle, byref(p)) == 0 else None
        return temp, util, power


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]


class _PDH:
    PDH_FMT_DOUBLE = 0x00000200

    def __init__(self, path=r"\Thermal Zone Information(\_TZ.TZ01)\High Precision Temperature"):
        self.lib = ctypes.WinDLL("pdh.dll")
        self.query, self.counter = c_void_p(), c_void_p()
        if self.lib.PdhOpenQueryW(None, None, byref(self.query)) != 0:
            raise OSError("PdhOpenQueryW failed")
        if self.lib.PdhAddEnglishCounterW(self.query, ctypes.c_wchar_p(path), None, byref(self.counter)) != 0:
            raise OSError("PdhAddEnglishCounterW failed")
        self.lib.PdhCollectQueryData(self.query)

    def read_c(self):
        if self.lib.PdhCollectQueryData(self.query) != 0:
            return None
        v = _PDH_FMT_COUNTERVALUE()
        if self.lib.PdhGetFormattedCounterValue(self.counter, c_ulong(self.PDH_FMT_DOUBLE), None, byref(v)) != 0:
            return None
        return round(v.doubleValue / 10.0 - 273.15, 1)       # tenths of kelvin -> C


_nv, _pdh = [None], [None]


def gpu():
    try:
        if _nv[0] is None:
            _nv[0] = _NVML()
        return _nv[0].read()
    except Exception:
        _nv[0] = None
        return None, None, None


def tz01_c():
    try:
        if _pdh[0] is None:
            _pdh[0] = _PDH()
        return _pdh[0].read_c()
    except Exception:
        _pdh[0] = None
        return None


if __name__ == "__main__":
    import time
    for _ in range(3):
        print("gpu", gpu(), "tz01", tz01_c(), flush=True)
        time.sleep(1.5)
