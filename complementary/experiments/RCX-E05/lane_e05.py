"""RCX-E05 lane gate (PREREG_E05_DRAFT.md §10.2, binding; DESIGN.md rev 3 §9.3). OPERATIONAL, outside the code digest
(M1): it decides only *when* a phase runs, never *what* it computes, and imports nothing from the E05 outcome code
(soft.py, families_e05, enum_e05, learners_e05, run_e05). A change here is a logged operational deviation and needs
tests/test_e05_lane.py to pass.

Party `rcx-e05`, through rcx.lane.try_acquire ONLY: E05 never calls rcx.lane.acquire (which writes a waiting marker),
never writes a `rineya-cpu-lane.waiting.*` marker, any `rineya-gpu-0.*` file or the quiet-release file. A refused start
writes nothing at all (a stale lane lock is reclaimed only when every other condition holds, just before
try_acquire). Every input is re-read at each phase start and at each safe point; nothing about dates is hard-coded.

Inputs (Paths; tests point them at temporary files; %TEMP% = os.environ TEMP, the lane dir = rcx.lane.lane_dir()):
  windows        runs/RCX-E01R/quiet_windows.json (required; rcx.lane.load_windows format) united with
                 runs/RCX-E05/extra_windows.json (optional, same format; absent = no extra windows). Unreadable or
                 unparseable -> refuse (at a safe point: yield).
  GPU lease      %TEMP%/rineya-gpu-0.lock (read only). Live holder iff its `pid` or any `worker_pids` entry is alive
                 (rcx.lane.pid_alive). No file = no holder; unreadable/unparseable -> refuse (safe point: yield).
  E01R state     %TEMP%/rineya-quiet-release.rcx.json; runs/RCX-E01R/cells/train_chain.log and post_train_chain.log;
                 the lane lock's holder (E01R holds the lane as party `rcx`).
  guard          rcx.guard (watchdog log tail: tz01; RCWM `.thermal_pause` flag); RAM via rcx.guard.available_ram_gb.

Start condition (all must hold; `start_check`):
  1. the lane is free (no lock, or only a stale one: dead holder pid) and no waiting marker of ANY party exists;
  2. free RAM >= 10.0 GB (BWM D-84 condition; unknown -> refuse);
  3. rcx.guard: tz01 < 97 C (>= 93 C counts as hot too if E05's previous exit was a guard yield: "resumed at <= 93 C")
     and no .thermal_pause file (a stale or unreadable watchdog log is not a reason: rcx.guard never blocks on it);
  4. no stray python process (RCX job command lines, STRAY_PATTERNS; unknown -> refuse);
  5. projected phase length L (work at the G-P5 rate, 19 GFLOP/s before G-P5, x 1.2) <= 15 min; the lock's
     expected_end = start + 15 min;
  6. day mode: now outside every window, the GPU lease has no live holder, start + L <= next window start - 10 min;
     or night-tail mode: now inside window W, E01R has no remaining block in W, no live GPU holder, start + L <=
     W.end - end_margin_min (5). "E01R has no remaining block in W" iff the quiet-release file exists, or the newest
     `lane:` line of E01R's ACTIVE chain log (the post-train log once it has started -- i.e. once it has a `lane:` line
     or the training log says "TRAIN CHAIN DONE" -- else the training log) is rcx.lane.acquire's refusal "inside the
     window ending <W.end>, but a ... block would end too late" for THIS window's end, and E01R does not hold the lane.
     Anything else, including an unreadable log, counts as "has a remaining block". If now lies inside several
     (overlapping) windows, the night-tail condition must hold for each of them.

Safe points (M2; `Session.check`): after every optimizer step or evaluation cell once >= 60 s have passed since the
previous check, and at every epoch end and batch end (force=True). The job checkpoints, releases the lane and exits
if: any waiting marker exists; 15 min have elapsed; day mode: the GPU lease has a live holder (unreadable: yield), or
the next window opens within 10 min (or now is inside a window: a window added since the start); night-tail mode:
W.end - margin is reached; the guard asks to pause (tz01 >= 97 C or the flag); free RAM < 6.0 GB (unknown: yield);
the lane lock is no longer ours (fail-safe). Units are sized so that none exceeds 30 s at the planning rate; checks
are then <= 90 s apart at that rate and <= 120 s at half of it (`validate_units`, `max_check_gap`).

E05's own lane events (acquire / exit with reason) are appended to runs/RCX-E05/LANE_EVENTS.jsonl (only after the
lane is held; a refusal writes nothing). The newest exit decides the guard hysteresis of condition 3.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx import guard as GD  # noqa: E402
from rcx import lane as LN  # noqa: E402

PARTY = "rcx-e05"
E01R_PARTY = "rcx"
START_RAM_GB, EXIT_RAM_GB = 10.0, 6.0
MAX_PHASE_MIN, LENGTH_FACTOR, PLAN_GFLOPS = 15.0, 1.2, 19.0
PRE_WINDOW_MIN = 10.0
CHECK_EVERY_S, MAX_UNIT_S, MAX_GAP_S, MAX_GAP_SLOW_S, SLOW_FACTOR = 60.0, 30.0, 90.0, 120.0, 2.0
TZ_PAUSE, TZ_RESUME, WATCHDOG_MAX_AGE_S = 97.0, 93.0, 180.0
STRAY_PATTERNS = ("experiments/RCX-", "experiments\\RCX-", "run_e05", "unittest", "pytest", "-m rcx.bw")
LANE_LINE = re.compile(r"^(?:\d\d:\d\d:\d\d\s+)?lane:\s")
E01R_REFUSAL = re.compile(r"^(?:\d\d:\d\d:\d\d\s+)?lane: not acquiring \(inside the window ending "
                          r"(\d\d-\d\d \d\d:\d\d), but a [^()]* block would end too late\)")
TRAIN_DONE = "TRAIN CHAIN DONE"


class Refused(Exception):
    """The start condition does not hold; nothing was written. `.decision` holds the reasons."""

    def __init__(self, decision):
        super().__init__("; ".join(decision.reasons) or "refused")
        self.decision = decision


class InputError(Exception):
    """An operational input (windows, GPU lease) is unreadable or unparseable."""


def temp_dir() -> Path:
    return Path(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp")


@dataclass
class Paths:
    windows: Path = ROOT / "runs" / "RCX-E01R" / "quiet_windows.json"
    extra_windows: Path = ROOT / "runs" / "RCX-E05" / "extra_windows.json"
    train_log: Path = ROOT / "runs" / "RCX-E01R" / "cells" / "train_chain.log"
    post_log: Path = ROOT / "runs" / "RCX-E01R" / "cells" / "post_train_chain.log"
    events: Path = ROOT / "runs" / "RCX-E05" / "LANE_EVENTS.jsonl"
    thermal_flag: Path = GD.DEFAULT_FLAG
    gpu_lock: Path | None = None   # default %TEMP%/rineya-gpu-0.lock (resolved at read time)
    release: Path | None = None    # default %TEMP%/rineya-quiet-release.rcx.json

    def gpu(self) -> Path:
        return self.gpu_lock or temp_dir() / "rineya-gpu-0.lock"

    def release_file(self) -> Path:
        return self.release or temp_dir() / "rineya-quiet-release.rcx.json"


def default_guard(resuming: bool = False) -> tuple[list, str | None]:
    """(reasons to pause, note) from rcx.guard's reader of the RCWM watchdog log. tz01 >= 97 C is a reason; while
    resuming after a guard yield, tz01 > 93 C is. A stale or unreadable log is only noted (rcx.guard never blocks on
    it). The pause flag is checked separately (Paths.thermal_flag)."""
    r = GD.latest_reading(GD.DEFAULT_LOG)
    if r is None:
        return [], "watchdog log unreadable"
    if r["age_s"] > WATCHDOG_MAX_AGE_S:
        return [], f"watchdog log stale ({r['age_s']:.0f} s)"
    tz = r["tz01"]
    if (tz > TZ_RESUME) if resuming else (tz >= TZ_PAUSE):
        return [f"guard: tz01={tz:g} " + (f"> {TZ_RESUME:g} (resuming)" if resuming else f">= {TZ_PAUSE:g}")], None
    return [], None


def default_strays(pid: int | None = None) -> list:
    """Python processes running RCX job code (STRAY_PATTERNS) other than this process, its ancestors and children.
    Peers' processes (GPU bundles, the watchdog, waiting rcx.lane chains) do not match. Raises if psutil is missing."""
    import psutil
    me = psutil.Process(pid or os.getpid())
    mine = {me.pid} | {p.pid for p in me.parents()} | {p.pid for p in me.children(recursive=True)}
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        name = (p.info.get("name") or "").lower()
        if not name.startswith("python") or p.info["pid"] in mine:
            continue
        cmd = " ".join(p.info.get("cmdline") or [])
        if any(s in cmd for s in STRAY_PATTERNS):
            out.append((p.info["pid"], cmd[:160]))
    return out


@dataclass
class Env:
    """Injectable world: clock, RAM, pid liveness, guard, stray-process scan, file paths and our pid."""
    paths: Paths = field(default_factory=Paths)
    now: Callable[[], dt.datetime] = dt.datetime.now
    ram: Callable[[], float | None] = GD.available_ram_gb
    pid_alive: Callable[[int], bool] = LN.pid_alive
    guard: Callable[[bool], tuple] = default_guard
    strays: Callable[[], list] = default_strays
    pid: int = field(default_factory=os.getpid)


# ----------------------------------------------------------------------------- readers

@dataclass(frozen=True)
class Window:
    start: dt.datetime
    end: dt.datetime
    margin_min: float
    source: str


def read_windows(paths: Paths) -> list:
    """quiet_windows.json (required) united with extra_windows.json (optional), sorted by start. InputError if a
    present file is unreadable or unparseable (or the required one is absent)."""
    out = []
    for p, required in ((Path(paths.windows), True), (Path(paths.extra_windows), False)):
        if not required and not p.exists():
            continue
        try:
            ws, margin = LN.load_windows(p)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
            raise InputError(f"windows file {p.name} unreadable or unparseable ({type(e).__name__}: {e})") from None
        for a, b in ws:
            if not a < b:
                raise InputError(f"windows file {p.name}: window {a:%m-%d %H:%M}-{b:%m-%d %H:%M} is empty")
            out.append(Window(a, b, margin, p.name))
    return sorted(out, key=lambda w: (w.start, w.end))


def gpu_holder(env: Env) -> dict | None:
    """The GPU lease's live holder {label, live_pids} or None (no file, or no live pid). InputError if unreadable."""
    p = env.paths.gpu()
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as e:
        raise InputError(f"GPU lease unreadable ({e})") from None
    try:
        d = json.loads(text)
        pids = [int(d["pid"])] + [int(x) for x in (d.get("worker_pids") or [])]
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise InputError(f"GPU lease unparseable ({type(e).__name__})") from None
    live = [x for x in pids if env.pid_alive(x)]
    return {"label": d.get("label"), "live_pids": live} if live else None


def lane_markers() -> list:
    return sorted(p.name for p in LN.lane_dir().glob("rineya-cpu-lane.waiting.*"))


def lane_state(env: Env) -> tuple[str, dict | None]:
    """('free' | 'stale' | 'held', lock). An existing but unparseable lock file counts as held."""
    lock = LN.read_lock()
    if lock is None:
        return ("held", None) if LN.lock_path().exists() else ("free", None)
    try:
        alive = env.pid_alive(int(lock.get("pid", -1)))
    except (TypeError, ValueError, OverflowError):
        alive = False
    return ("held" if alive else "stale"), lock


def _lines(p: Path) -> list:
    return Path(p).read_text(encoding="utf-8", errors="replace").splitlines()


def newest_lane_line(lines) -> str | None:
    return next((ln.strip() for ln in reversed(lines) if LANE_LINE.match(ln.strip())), None)


def e01r_active_log(paths: Paths) -> tuple[str, list]:
    """(name, lines) of E01R's active chain log: the post-train log once it has started (it has a `lane:` line, or
    the training log says TRAIN CHAIN DONE), else the training log. OSError if the active log is unreadable."""
    post = None
    try:
        post = _lines(paths.post_log)
    except OSError:
        pass
    if post is not None and newest_lane_line(post) is not None:
        return "post_train_chain.log", post
    train = _lines(paths.train_log)
    if any(TRAIN_DONE in ln for ln in train):
        if post is None:
            raise OSError("post_train_chain.log unreadable after TRAIN CHAIN DONE")
        return "post_train_chain.log", post
    return "train_chain.log", train


def e01r_no_remaining_block(env: Env, w: Window, lock: dict | None) -> tuple[bool, str]:
    """(True, why) iff E01R has no remaining block in window w (module docstring, condition 6)."""
    if env.paths.release_file().exists():
        return True, "quiet-release file present"
    if lock and lock.get("holder") == E01R_PARTY:
        return False, "E01R holds the lane"
    try:
        name, lines = e01r_active_log(env.paths)
    except OSError as e:
        return False, f"E01R chain log unreadable ({e})"
    line = newest_lane_line(lines)
    m = E01R_REFUSAL.match(line) if line else None
    want = f"{w.end:%m-%d %H:%M}"
    if m and m.group(1) == want:
        return True, f"{name}: E01R refused the rest of the window ending {want}"
    return False, f"{name}: newest lane line is not E01R's too-late refusal for the window ending {want} ({line!r})"


def last_exit(paths: Paths) -> dict | None:
    try:
        lines = _lines(paths.events)
    except OSError:
        return None
    for ln in reversed(lines):
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("event") == "exit":
            return ev
    return None


def _append_event(paths: Paths, ev: dict) -> None:
    p = Path(paths.events)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(ev, sort_keys=True, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ----------------------------------------------------------------------------- start condition

def projected_min(gflop: float = 0.0, seconds: float = 0.0, rate_gflops: float = PLAN_GFLOPS) -> float:
    """Projected phase length L in minutes: (GFLOP at the rate + other seconds) x 1.2 (condition 5)."""
    return (gflop / rate_gflops + seconds) / 60.0 * LENGTH_FACTOR


@dataclass
class Decision:
    ok: bool
    mode: str | None
    reasons: list
    windows: list
    deadline: dt.datetime | None
    lane: str
    lock: dict | None
    info: dict


def mode_of(windows: list, now: dt.datetime) -> tuple[str, list, dt.datetime | None]:
    """('day', [], next window start - 10 min or None) or ('night-tail', [windows containing now], min(W.end -
    margin))."""
    inside = [w for w in windows if w.start <= now < w.end]
    if inside:
        return "night-tail", inside, min(w.end - dt.timedelta(minutes=w.margin_min) for w in inside)
    nxt = min((w.start for w in windows if w.start > now), default=None)
    return "day", [], (nxt - dt.timedelta(minutes=PRE_WINDOW_MIN) if nxt else None)


def start_check(env: Env, L_min: float) -> Decision:
    """Evaluate conditions 1-6 (reads only; writes nothing). Every failing condition is listed."""
    now, reasons, info = env.now(), [], {}
    lane, lock = lane_state(env)
    if lane == "held":
        reasons.append(f"lane held ({(lock or {}).get('holder')!r}: {(lock or {}).get('job')!r})")
    marks = lane_markers()
    if marks:
        reasons.append(f"waiting marker(s) {marks}")
    ram = env.ram()
    info["ram_gb"] = ram
    if ram is None:
        reasons.append("free RAM unknown")
    elif ram < START_RAM_GB:
        reasons.append(f"free RAM {ram:.1f} GB < {START_RAM_GB:g} GB")
    prev = last_exit(env.paths)
    resuming = bool(prev and prev.get("guard_yield"))
    g_reasons, note = env.guard(resuming)
    reasons += list(g_reasons)
    info["guard_note"] = note
    if Path(env.paths.thermal_flag).exists():
        reasons.append(".thermal_pause flag present")
    try:
        strays = env.strays()
    except Exception as e:  # noqa: BLE001 - fail closed on an unknown process table
        strays = [("?", f"process scan failed: {e!r}")]
    if strays:
        reasons.append(f"stray python process(es) {strays[:3]}")
    if not 0 < L_min <= MAX_PHASE_MIN:
        reasons.append(f"projected length {L_min:.2f} min not in (0, {MAX_PHASE_MIN:g}]")
    windows, mode, deadline = [], None, None
    try:
        windows = read_windows(env.paths)
    except InputError as e:
        reasons.append(str(e))
    else:
        mode, inside, deadline = mode_of(windows, now)
        for w in inside:
            ok, why = e01r_no_remaining_block(env, w, lock if lane != "stale" else None)
            info.setdefault("e01r", []).append(why)
            if not ok:
                reasons.append(f"night-tail: E01R may still take a block in the window ending {w.end:%m-%d %H:%M} "
                               f"({why})")
        if deadline is not None and now + dt.timedelta(minutes=L_min) > deadline:
            reasons.append(f"{mode}: start + L = {now + dt.timedelta(minutes=L_min):%m-%d %H:%M:%S} > "
                           f"{deadline:%m-%d %H:%M:%S}")
    try:
        gpu = gpu_holder(env)
    except InputError as e:
        reasons.append(str(e))
    else:
        info["gpu"] = gpu
        if gpu:
            reasons.append(f"GPU lease has a live holder ({gpu['label']!r}, pids {gpu['live_pids']})")
    return Decision(not reasons, mode, reasons, windows, deadline, lane, lock, info)


def slice_budget_min(env: Env) -> float:
    """Largest L (minutes) the time rule allows now: min(15, deadline - now); 0 if the windows are unreadable."""
    try:
        _, _, deadline = mode_of(read_windows(env.paths), env.now())
    except InputError:
        return 0.0
    room = MAX_PHASE_MIN if deadline is None else (deadline - env.now()).total_seconds() / 60.0
    return max(0.0, min(MAX_PHASE_MIN, room))


# ----------------------------------------------------------------------------- the held lane and its safe points

class Session:
    """The lane held by this process for one phase slice. Use `Session.open`; call `check()` after every optimizer
    step / evaluation cell and `check(force=True)` at epoch and batch ends; on a non-None reason, checkpoint, then
    `release(reason)` and exit."""

    def __init__(self, env: Env, job: str, decision: Decision, rec: dict, L_min: float):
        self.env, self.job, self.decision, self.rec, self.L_min = env, job, decision, rec, L_min
        self.mode, self.started = decision.mode, env.now()
        self.last_check, self.checks, self.gaps = self.started, 0, []
        self.reason, self.reasons = None, []
        self.released = False

    @classmethod
    def open(cls, env: Env, job: str, L_min: float) -> "Session":
        d = start_check(env, L_min)
        if not d.ok:
            raise Refused(d)
        if d.lane == "stale":
            LN._reclaim_if_stale(d.lock)
        rec = LN.try_acquire(job, env.pid, MAX_PHASE_MIN, party=PARTY)
        if rec is None:
            d.ok, d.reasons = False, ["lane taken by another party between the check and try_acquire"]
            raise Refused(d)
        s = cls(env, job, d, rec, L_min)
        _append_event(env.paths, {"event": "acquire", "party": PARTY, "job": job, "mode": d.mode, "L_min": L_min,
                                  "deadline": d.deadline, "at": s.started, "pid": env.pid, "token": rec["token"],
                                  "ram_gb": d.info.get("ram_gb"), "guard_note": d.info.get("guard_note")})
        return s

    def due(self) -> bool:
        return (self.env.now() - self.last_check).total_seconds() >= CHECK_EVERY_S

    def check(self, force: bool = False) -> str | None:
        """A safe point: None to continue, else the yield reason (also kept in .reason)."""
        if self.reason:
            return self.reason
        if not force and not self.due():
            return None
        now = self.env.now()
        self.gaps.append((now - self.last_check).total_seconds())
        self.last_check, self.checks = now, self.checks + 1
        why = self.yield_reasons(now)
        if why:
            self.reasons, self.reason = why, "; ".join(why)
        return self.reason

    def yield_reasons(self, now: dt.datetime) -> list:
        env, out = self.env, []
        marks = lane_markers()
        if marks:
            out.append(f"waiting marker(s) {marks}")
        if (now - self.started).total_seconds() >= 60.0 * MAX_PHASE_MIN:
            out.append(f"{MAX_PHASE_MIN:g} min elapsed")
        lock = LN.read_lock()
        if not lock or lock.get("token") != self.rec["token"]:
            out.append("lane lock is no longer ours")
        try:
            windows = read_windows(env.paths)
        except InputError as e:
            out.append(f"windows: {e}")
        else:
            if self.mode == "day":
                mode, _, deadline = mode_of(windows, now)
                if mode != "day":
                    out.append("day mode: now inside a window")
                elif deadline is not None and now >= deadline:
                    out.append(f"day mode: next window opens within {PRE_WINDOW_MIN:g} min")
            else:
                ends = [w.end - dt.timedelta(minutes=w.margin_min) for w in self.decision.windows + windows
                        if w.start <= self.started < w.end or w.start <= now < w.end]
                if not ends or now >= min(ends):
                    out.append("night-tail: W.end - margin reached")
        if self.mode == "day":
            try:
                gpu = gpu_holder(env)
            except InputError as e:
                out.append(f"day mode: {e}")
            else:
                if gpu:
                    out.append(f"day mode: GPU lease has a live holder ({gpu['label']!r})")
        g_reasons, _ = env.guard(False)
        if g_reasons:
            out.append("guard: " + ", ".join(g_reasons))
        if Path(env.paths.thermal_flag).exists():
            out.append("guard: .thermal_pause flag present")
        ram = env.ram()
        if ram is None or ram < EXIT_RAM_GB:
            out.append(f"free RAM {'unknown' if ram is None else f'{ram:.1f} GB'} < {EXIT_RAM_GB:g} GB")
        return out

    def release(self, reason: str = "done") -> bool:
        """Release the lane (only our own lock) and log the exit; idempotent."""
        if self.released:
            return False
        self.released = True
        ok = LN.release(self.env.pid, party=PARTY)
        _append_event(self.env.paths, {"event": "exit", "party": PARTY, "job": self.job, "reason": reason,
                                       "guard_yield": any(r.startswith("guard") for r in self.reasons),
                                       "at": self.env.now(), "checks": self.checks,
                                       "max_gap_s": max(self.gaps, default=0.0), "released": ok})
        return ok


# ----------------------------------------------------------------------------- safe-point plan (M2)

def check_gaps(unit_s, boundaries=(), every: float = CHECK_EVERY_S) -> list:
    """Gaps (s) between consecutive safe-point checks (and from the start to the first check) for units of the given
    durations run back to back: a check follows unit i if >= `every` s have passed since the previous check or i is
    an epoch/batch end (index in `boundaries`)."""
    t = last = 0.0
    gaps, bset = [], set(boundaries)
    for i, u in enumerate(unit_s):
        t += float(u)
        if t - last >= every or i in bset:
            gaps.append(t - last)
            last = t
    return gaps


def max_check_gap(unit_s, boundaries=(), scale: float = 1.0) -> float:
    return max(check_gaps([u * scale for u in unit_s], boundaries), default=0.0)


def validate_units(costs: dict, n_repeat: int = 200) -> dict:
    """costs = {unit name: seconds per uninterruptible unit at the planning rate}. Returns {"ok", "rejected" (units >
    30 s), "max_gap_s", "max_gap_slow_s"} where the gaps are the worst over each unit run alone (n_repeat times) and
    over all units interleaved, at the planning rate and at 2x slower."""
    rejected = {k: v for k, v in costs.items() if not float(v) <= MAX_UNIT_S}
    seqs = [[v] * n_repeat for v in costs.values()]
    seqs.append([v for _ in range(n_repeat // max(1, len(costs)) + 1) for v in costs.values()])
    g1 = max((max_check_gap(s) for s in seqs), default=0.0)
    g2 = max((max_check_gap(s, scale=SLOW_FACTOR) for s in seqs), default=0.0)
    ok = not rejected and g1 <= MAX_GAP_S and g2 <= MAX_GAP_SLOW_S
    return {"ok": ok, "rejected": rejected, "max_gap_s": g1, "max_gap_slow_s": g2}


# Uninterruptible units of the E05 plan (PREREG §5.1-5.3, §10.2): one optimizer step of a (model, B, batch) and one
# (arm, family-kind) evaluation cell. Parameter counts: §5.1 (PGV width 128 = 43,009 for the extension / R_ext).
MODEL_PARAMS = {"SF": 180_537, "UVFA": 166_145, "UVFA+aux": 180_794, "PGV256": 151_553, "PGV128": 43_009,
                "PGV64": 13_313}
STEP_UNITS = ([(m, 1, 256) for m in ("SF", "UVFA", "UVFA+aux")]
              + [(m, B, bs) for B in (16, 64) for m, bs in
                 (("PGV64", 64), ("PGV256", 64), ("PGV256", 128), ("PGV256", 256), ("PGV128", 64), ("PGV128", 128),
                  ("PGV128", 256), ("UVFA", 64), ("SF", 64))])
EVAL_ARMS = ("SF-GPI", "SF-noGPI", "SF-L", "SF-GPI-5", "SF-shuffled", "UVFA", "UVFA+aux", "PGV", "FT-UVFA", "SF-FT",
             "ENUM-MIS±", "ENUM-H*±", "GNN-ENUM±", "GREEDY", "STOP", "RAND", "ORACLE", "EXACT-GPI(L)", "BEST-SINGLE(L)")


def step_name(model: str, B: int, batch: int) -> str:
    return f"step:{model}:B{B}:b{batch}"


def plan_costs(step_s: dict | None = None, cell_s: dict | None = None, rate_gflops: float = PLAN_GFLOPS,
               default_cell_s: float = 1.0) -> dict:
    """{unit name: seconds} for every planned unit: optimizer steps from measured `step_s` (keyed by step_name) or
    6 x params x batch x B FLOP at `rate_gflops`; evaluation cells from measured `cell_s` (keyed "cell:<arm>") or
    `default_cell_s`."""
    step_s, cell_s, out = step_s or {}, cell_s or {}, {}
    for m, B, bs in STEP_UNITS:
        k = step_name(m, B, bs)
        out[k] = float(step_s.get(k, 6.0 * MODEL_PARAMS[m] * bs * B / (rate_gflops * 1e9)))
    for a in EVAL_ARMS:
        out[f"cell:{a}"] = float(cell_s.get(f"cell:{a}", default_cell_s))
    return out


def reference_shas() -> dict:
    """sha256 of the operational files (recorded in every phase's provenance; not part of the code digest)."""
    out = {}
    for rel in ("experiments/RCX-E05/lane_e05.py", "rcx/lane.py", "rcx/guard.py"):
        p = ROOT / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return out


def main():  # status only (reads; writes nothing)
    env = Env()
    d = start_check(env, 1.0)
    print(json.dumps({"ok": d.ok, "mode": d.mode, "reasons": d.reasons, "deadline": d.deadline, "lane": d.lane,
                      "budget_min": slice_budget_min(env), "info": d.info}, indent=1, default=str))


if __name__ == "__main__":
    main()
