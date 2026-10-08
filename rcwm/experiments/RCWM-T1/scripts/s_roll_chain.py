"""Idempotent stage runner for S-roll (operational helper; computes nothing itself).

Runs `scripts/s_roll.py` one stage per process (each stage < 25 min on 2 threads), skipping any stage
whose output already carries the current sha256 of s_roll.py, waits while the shared thermal-pause flag
(runs/.thermal_pause) exists, and, once every per-family output, the depth check and both criteria are
current (plus s_roll_long.py for the DEV-regret arms), renders the report (`s_roll.py --report-only`). Equivalent to `python scripts/s_roll.py`
but restartable and split so no single process runs long.

  python scripts/s_roll_chain.py                  # EVERYTHING (both criteria, depth + schedule checks, report)
  python scripts/s_roll_chain.py --crit exact     # L/P/R select+fit (spec criterion), then s_roll_deep.py
  python scripts/s_roll_chain.py --crit regret    # L/P/R select+fit (DEV-regret criterion), then s_roll_long.py
  (two chains, one per criterion, can run in parallel: 2 processes x 2 threads)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bank_pilot" / "analysis"
SROLL = ROOT / "scripts" / "s_roll.py"
DEEP = ROOT / "scripts" / "s_roll_deep.py"
LONG = ROOT / "scripts" / "s_roll_long.py"
PAUSE = ROOT / "runs" / ".thermal_pause"
TAG = {"exact": "", "regret": "_regsel"}
LOCK = Path(os.environ.get("S_ROLL_LOCK_DIR", OUT)) / "s_roll_chain_report.lock.log"
CHAIN_LOG = OUT / "s_roll_chain.log"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [pid {os.getpid()}] {msg}"
    print(line, flush=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def current(p: Path) -> bool:
    if not p.exists():
        return False
    try:
        m = json.loads(p.read_text(encoding="utf-8")).get("meta", {}).get("code_sha256", {})
    except Exception:
        return False
    return any(k.replace("\\", "/").endswith("scripts/s_roll.py") and v == sha(SROLL) for k, v in m.items())


def deep_current(name="deep") -> bool:
    """s_roll_deep.json (crit exact) / s_roll_long.json (crit regret) is current: same code, all families,
    newer than the per-family outputs it compares against."""
    p, script, keys, tag = ((OUT / "s_roll_deep.json", DEEP, ("ledger_replay", "prerequisite_plan", "register_machine"), "")
                            if name == "deep" else (OUT / "s_roll_long.json", LONG, ("L", "P", "R"), "_regsel"))
    if not p.exists():
        return False
    d = json.loads(p.read_text(encoding="utf-8"))
    ok = current(p) and d.get("meta", {}).get(f"{name}_code_sha256") == sha(script) and all(f in d for f in keys)
    mains = [OUT / f"s_roll{tag}_{f}.json" for f in "LPR"]
    if not all(m.exists() for m in mains):
        return False
    return ok and p.stat().st_mtime > max(m.stat().st_mtime for m in mains)


def run_helper(name):
    script = DEEP if name == "deep" else LONG
    wait_thermal()
    log(f"RUN {script.name}")
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", str(script)], cwd=str(ROOT), stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""})
    log(f"END {script.name} rc={r.returncode} ({time.time() - t0:.0f}s)")
    if r.returncode != 0:
        log("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-30:]))
        raise SystemExit(r.returncode)


def wait_thermal():
    t0 = time.time()
    while PAUSE.exists():
        time.sleep(15)
    if time.time() - t0 > 1:
        log(f"thermal pause cleared after {time.time() - t0:.0f}s")


def run(args):
    wait_thermal()
    cmd = [sys.executable, "-u", str(SROLL)] + args
    log("RUN " + " ".join(args))
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2"}
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                       encoding="utf-8", errors="replace")
    log(f"END {' '.join(args)} rc={r.returncode} ({time.time() - t0:.0f}s)")
    if r.returncode != 0:
        log("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-30:]))
        raise SystemExit(r.returncode)


def all_current() -> bool:
    return (all(current(OUT / f"s_roll{TAG[c]}_{f}.json") for c in TAG for f in "LPR") and deep_current("deep")
            and deep_current("long"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crit", default="both", choices=("exact", "regret", "both"))
    a = ap.parse_args()
    crits = ("exact", "regret") if a.crit == "both" else (a.crit,)
    for crit in crits:
        for f in "LPR":
            if not current(OUT / f"s_roll_select{TAG[crit]}_{f}.json"):
                run(["--crit", crit, "--family", f, "--stage", "select"])
            else:
                log(f"skip select {crit} {f} (current)")
            if not current(OUT / f"s_roll{TAG[crit]}_{f}.json"):
                run(["--crit", crit, "--family", f, "--stage", "fit"])
            else:
                log(f"skip fit {crit} {f} (current)")
        helper = "deep" if crit == "exact" else "long"
        if not deep_current(helper):
            run_helper(helper)
        else:
            log(f"skip {helper} (current)")
    if all_current():
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            log("report already being rendered by the other chain; done")
            return
        try:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            # s_roll.report() directly: `s_roll.py --report-only` with default flags also counts as a full
            # run and re-executes s_roll_deep.py (deterministic, ~6 min), which the chain has already done.
            wait_thermal()
            log("RUN s_roll.report('')")
            t0 = time.time()
            r = subprocess.run([sys.executable, "-u", "-c", "import sys; sys.path.insert(0, 'scripts'); "
                                "import s_roll; s_roll.report('')"], cwd=str(ROOT), stdout=subprocess.DEVNULL,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": ""})
            log(f"END report rc={r.returncode} ({time.time() - t0:.0f}s)")
            if r.returncode != 0:
                log("STDERR tail:\n" + "\n".join(r.stderr.splitlines()[-30:]))
                raise SystemExit(r.returncode)
        finally:
            LOCK.unlink(missing_ok=True)
        log("CHAIN COMPLETE: report rendered")
    else:
        log("chain finished; other outputs not yet current, report not rendered by this chain")


if __name__ == "__main__":
    main()
