"""GPU temperature / power / utilization through the driver's NVML library (ctypes; no subprocess, no install).

Replaces repeated `nvidia-smi` launches (incident I-14: a third-party helper holding a handle to every process ever
started on the machine turned ~10k short-lived nvidia-smi processes per day into a kernel-memory leak). One process-
wide NVML session; every reader returns None on any failure so callers keep their fail-safe behaviour.
"""
from __future__ import annotations

import ctypes

_lib = None
_dev = None


class _Util(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def _handle():
    global _lib, _dev
    if _dev is not None:
        return _dev
    try:
        if _lib is None:
            _lib = ctypes.WinDLL("nvml.dll") if hasattr(ctypes, "WinDLL") else ctypes.CDLL("libnvidia-ml.so.1")
            if _lib.nvmlInit_v2() != 0:
                _lib = None
                return None
        dev = ctypes.c_void_p()
        if _lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev)) != 0:
            return None
        _dev = dev
        return _dev
    except Exception:
        return None


def temperature() -> int | None:
    d = _handle()
    if d is None:
        return None
    t = ctypes.c_uint()
    return int(t.value) if _lib.nvmlDeviceGetTemperature(d, 0, ctypes.byref(t)) == 0 else None


def power_w() -> float | None:
    d = _handle()
    if d is None:
        return None
    mw = ctypes.c_uint()
    return round(mw.value / 1000, 2) if _lib.nvmlDeviceGetPowerUsage(d, ctypes.byref(mw)) == 0 else None


def utilization() -> int | None:
    d = _handle()
    if d is None:
        return None
    u = _Util()
    return int(u.gpu) if _lib.nvmlDeviceGetUtilizationRates(d, ctypes.byref(u)) == 0 else None
