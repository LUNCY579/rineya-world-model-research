"""Run a bundle of GPU sub-jobs inside ONE lease turn (agreed with the RCWM-T1 session, 2026-10-07).

Conditions of the agreement: at most `budget_s` (default 4.5 h) of wall time counted from the first acquire of the turn
(gpu_run passes it as BWM_LEASE_ACQUIRED); a sub-job starts only if its estimate fits the remaining budget (the first
sub-job of a turn always starts); a sub-job still running at the cap is terminated (trainings resume from their 50-step
checkpoints on the next turn); any crash ends the bundle and releases the lease at once; every sub-job keeps the shared
flag, duty cycle and RAM gate (they live in bwmn/lm.py and apply unchanged).

Exit codes: 0 = every sub-job's marker exists; 75 = the budget ended this turn with sub-jobs left (the orchestrator
requeues the bundle without counting a failure); otherwise the failing sub-job's code.

Usage (as an orchestrator GPU job): python scripts/bundle.py runs/bundles/<name>.json
Bundle file: {"budget_s": 16200, "jobs": [{"name", "marker", "cmd": [script, args...], "est_s", "log"}, ...]}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPFAIL = 75


def main() -> int:
    spec = json.loads((ROOT / sys.argv[1]).read_text(encoding="utf-8"))
    budget = float(spec.get("budget_s", 16200))
    t_start = float(os.environ.get("BWM_LEASE_ACQUIRED", time.time()))
    deadline = t_start + budget
    ran = 0
    for j in spec["jobs"]:
        marker = ROOT / j["marker"]
        if marker.exists():
            continue
        left = deadline - time.time()
        if ran and j.get("est_s", 0) > left:
            print(f"[bundle] {j['name']}: estimate {j.get('est_s')} s exceeds remaining {left:.0f} s; ending turn",
                  flush=True)
            return TEMPFAIL
        if left <= min(60.0, 0.02 * budget):
            print(f"[bundle] budget exhausted before {j['name']}; ending turn", flush=True)
            return TEMPFAIL
        log = ROOT / j.get("log", f"runs/{j['name']}.stdout")
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[bundle] {time.strftime('%H:%M:%S')} start {j['name']} (remaining budget {left:.0f} s)", flush=True)
        t0 = time.time()
        with open(log, "a", encoding="utf-8") as fh:
            p = subprocess.Popen([sys.executable] + j["cmd"], cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
            try:
                rc = p.wait(timeout=max(1.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.terminate()
                p.wait()
                print(f"[bundle] {time.strftime('%H:%M:%S')} {j['name']} stopped at the {budget:.0f} s cap after "
                      f"{time.time() - t0:.0f} s; resumes next turn", flush=True)
                return TEMPFAIL
        ran += 1
        if rc != 0:
            print(f"[bundle] {j['name']} FAILED rc={rc} after {time.time() - t0:.0f} s; ending bundle", flush=True)
            return rc
        if not marker.exists():
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"job": j["name"], "finished": time.time(), "wall_s": time.time() - t0}))
        print(f"[bundle] {time.strftime('%H:%M:%S')} done {j['name']} in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
