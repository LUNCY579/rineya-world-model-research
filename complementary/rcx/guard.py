"""Thermal/RAM courtesy guard for RCX CPU jobs that share the laptop with peer GPU jobs.

The guard only reads the RCWM watchdog's log tail and checks its pause flag; it never spawns a
subprocess (I-27 process-spawn hygiene) and never writes inside the RCWM tree.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

RUNS_DIR = Path('C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05'
                '/experiments/RCWM-T1/runs')
DEFAULT_LOG = RUNS_DIR / 'thermal_log.txt'
DEFAULT_FLAG = RUNS_DIR / '.thermal_pause'
TAIL_BYTES = 2048
_DAY = 86400
_FUTURE_SKEW_S = 120
_STATUS_EVERY_S = 300.0


class _MemStatus(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


def available_ram_gb() -> float | None:
    """Available host RAM in GiB (GlobalMemoryStatusEx on Windows, sysconf elsewhere); None if unknown."""
    try:
        if sys.platform == 'win32':
            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            return status.ullAvailPhys / 2**30
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2**30
    except (OSError, ValueError, AttributeError):
        return None


def _hms_seconds(ts: str) -> int | None:
    parts = ts.split(':')
    if len(parts) != 3 or not all(p.isdigit() and len(p) == 2 for p in parts):
        return None
    h, m, s = map(int, parts)
    return h * 3600 + m * 60 + s if h < 24 and m < 60 and s < 60 else None


def tod_age_s(ts: str, now: datetime | None = None) -> float:
    """Seconds elapsed from local time-of-day `ts` ('HH:MM:SS') to `now`, wrapping across midnight.

    A timestamp up to two minutes in the future (clock skew) counts as age 0.
    """
    secs = _hms_seconds(ts)
    if secs is None:
        raise ValueError(f'bad timestamp {ts!r}')
    now = now or datetime.now()
    age = (now.hour * 3600 + now.minute * 60 + now.second + now.microsecond / 1e6 - secs) % _DAY
    return 0.0 if age > _DAY - _FUTURE_SKEW_S else age


def _num(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def parse_line(line: str) -> dict | None:
    """Parse one watchdog line into dict(ts, t, tz01, ram); None unless it has a timestamp and tz01."""
    parts = line.split()
    if not parts or _hms_seconds(parts[0]) is None:
        return None
    kv = dict(p.split('=', 1) for p in parts[1:] if '=' in p)
    tz01 = _num(kv.get('tz01'))
    if tz01 is None:
        return None
    return dict(ts=parts[0], t=_num(kv.get('t')), tz01=tz01, ram=_num(kv.get('ram')))


def latest_reading(log_path: str | os.PathLike = DEFAULT_LOG, tail_bytes: int = TAIL_BYTES) -> dict | None:
    """Return the newest reading dict(t, tz01, ram, ts, age_s) from the log tail, or None if unreadable.

    Only the last `tail_bytes` are read and an unterminated last line is ignored. age_s compares the
    HH:MM:SS stamp with local time; the file mtime is used only to count whole days elapsed, so a log that
    stopped exactly a day ago is still stale.
    """
    try:
        with open(log_path, 'rb') as handle:
            size = handle.seek(0, os.SEEK_END)
            start = max(0, size - tail_bytes)
            handle.seek(start)
            chunk = handle.read(tail_bytes)
            mtime = os.fstat(handle.fileno()).st_mtime
    except OSError:
        return None
    lines = chunk.decode('utf-8', 'replace').split('\n')[:-1]
    for line in reversed(lines[1:] if start > 0 else lines):
        reading = parse_line(line)
        if reading:
            age = tod_age_s(reading['ts'])
            days = max(0, round((time.time() - mtime - age) / _DAY))
            reading['age_s'] = age + days * _DAY
            return reading
    return None


class ThermalGuard:
    """Courtesy guard: call check() between work units.

    Pauses while the RCWM pause flag exists, tz01 >= tz_pause or host RAM < ram_floor_gb. Hysteresis is
    reason-specific (as in the watchdog, I-25/I-26): resuming needs the flag gone and every condition out of its
    pause zone, plus tz01 <= tz_resume only if tz01 reached tz_pause during this pause, and RAM >= ram_floor_gb + 1
    only if RAM fell below the floor during it. A stale or unreadable log never blocks: it throttles by
    `throttle_s` per check instead (the flag and tz01 are then untrusted, RAM is still checked live).
    paused_s is all guard-induced sleep; throttled_s is the stale-log share of it.
    """

    def __init__(self, tz_pause: float = 97.0, tz_resume: float = 93.0, ram_floor_gb: float = 3.0,
                 honor_flag: bool = True, poll_s: float = 20, max_age_s: float = 180,
                 log: Callable[[str], object] = print, *, log_path: str | os.PathLike = DEFAULT_LOG,
                 flag_path: str | os.PathLike = DEFAULT_FLAG, throttle_s: float = 5.0,
                 max_pause_s: float | None = None,
                 ram_fn: Callable[[], float | None] | None = available_ram_gb,
                 sleep: Callable[[float], object] = time.sleep):
        if tz_resume > tz_pause:
            raise ValueError('tz_resume must not exceed tz_pause')
        self.tz_pause, self.tz_resume, self.ram_floor_gb = tz_pause, tz_resume, ram_floor_gb
        self.honor_flag, self.poll_s, self.max_age_s, self.log = honor_flag, poll_s, max_age_s, log
        self.log_path, self.flag_path = Path(log_path), Path(flag_path)
        self.throttle_s, self.max_pause_s, self.ram_fn, self._sleep = throttle_s, max_pause_s, ram_fn, sleep
        self.paused_s = self.throttled_s = 0.0
        self.pauses = 0
        self.last_reading: dict | None = None
        self._degraded: str | None = None
        self._warned_at = 0.0

    def latest_reading(self) -> dict | None:
        """Newest watchdog reading from this guard's log (see module-level latest_reading)."""
        return latest_reading(self.log_path)

    def ram_gb(self, reading: dict | None = None) -> float | None:
        """Live host RAM via ram_fn, falling back to the log's ram= value of a fresh reading."""
        live = self.ram_fn() if self.ram_fn else None
        return live if live is not None else (reading or {}).get('ram')

    def _say(self, msg: str) -> None:
        try:
            self.log(f'{datetime.now():%H:%M:%S} [guard] {msg}')
        except Exception:
            pass

    def _flag_reason(self) -> str | None:
        try:
            with open(self.flag_path, 'rb') as handle:
                text = handle.read(120).decode('utf-8', 'replace').strip()
        except FileNotFoundError:
            return None
        except OSError:
            text = ''
        return f'pause flag ({text})' if text else 'pause flag'

    def assess(self, resuming: bool = False) -> tuple[list[str], str | None]:
        """Return (reasons to be paused, stale-log description or None) using pause or resume thresholds."""
        reasons, stale = self._assess(('tz', 'ram') if resuming else ())
        return list(reasons.values()), stale

    def _assess(self, latched=()) -> tuple[dict[str, str], str | None]:
        reading = self.last_reading = self.latest_reading()
        if reading is None:
            stale = 'log unreadable or has no tz01 reading'
        elif reading['age_s'] > self.max_age_s:
            stale = f"log stale ({reading['age_s']:.0f}s > {self.max_age_s:g}s, last {reading['ts']})"
        else:
            stale = None
        reasons = {}
        if stale is None:
            flag = self._flag_reason() if self.honor_flag else None
            if flag:
                reasons['flag'] = flag
            tz, hold = reading['tz01'], 'tz' in latched
            if (tz > self.tz_resume) if hold else (tz >= self.tz_pause):
                reasons['tz'] = f'tz01={tz:g}' + (f'>{self.tz_resume:g}' if hold else f'>={self.tz_pause:g}')
        floor = self.ram_floor_gb + (1.0 if 'ram' in latched else 0.0)
        ram = self.ram_gb(None if stale else reading)
        if ram is not None and ram < floor:
            reasons['ram'] = f'ram={ram:.1f}GB<{floor:g}'
        return reasons, stale

    def check(self) -> float:
        """Block while pause conditions hold (or throttle on a stale log); return seconds slept. Never raises."""
        try:
            return self._check()
        except Exception as exc:
            self._say(f'WARNING guard error {exc!r}; throttling {self.throttle_s:g}s')
            return self._throttle()

    def _throttle(self) -> float:
        self._sleep(self.throttle_s)
        self.throttled_s += self.throttle_s
        self.paused_s += self.throttle_s
        return self.throttle_s

    def _check(self) -> float:
        reasons, stale = self._assess()
        if not reasons:
            if stale is None:
                if self._degraded:
                    self._say('log fresh again; throttling off')
                self._degraded = None
                return 0.0
            kind = 'unreadable' if self.last_reading is None else 'stale'
            if kind != self._degraded or time.monotonic() - self._warned_at > 600:
                flag = ' (pause flag present but untrusted)' if self.honor_flag and self.flag_path.exists() else ''
                self._say(f'WARNING {stale}{flag}; fail-safe throttling {self.throttle_s:g}s per check')
                self._degraded, self._warned_at = kind, time.monotonic()
            return self._throttle()
        t0 = last_status = time.monotonic()
        self.pauses += 1
        latched = set(reasons)
        self._say(f'PAUSE: {", ".join(reasons.values())}')
        while True:
            self._sleep(self.poll_s)
            reasons, stale = self._assess(latched)
            latched |= set(reasons)
            why = ', '.join(reasons.values())
            now = time.monotonic()
            if not reasons:
                break
            if self.max_pause_s is not None and now - t0 >= self.max_pause_s:
                self._say(f'WARNING max_pause_s={self.max_pause_s:g} reached; resuming despite {why}')
                break
            if now - last_status >= _STATUS_EVERY_S:
                self._say(f'still paused {now - t0:.0f}s: {why}')
                last_status = now
        dt = time.monotonic() - t0
        self.paused_s += dt
        note = f'; {stale}' if stale else ''
        self._say(f'RESUME after {dt:.1f}s (total paused {self.paused_s:.1f}s){note}')
        return dt
