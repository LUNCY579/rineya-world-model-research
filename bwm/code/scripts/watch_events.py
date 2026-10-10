"""One long-lived event watcher for the Monitor tool (replaces shell loops that spawned tail/grep/wc/stat every 20 s:
incident I-14, where a third-party helper leaked a kernel handle per process ever started).

Prints one line per new relevant event, then sleeps: orchestrator logs (job start/done/failed/yielded), bundle
progress, planner-selection results, training milestones, and health alerts (tz01 >= 98 C, RAM < 4 GB, stale health
file). Usage: python scripts/watch_events.py [--period 30]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


class Tail:
    def __init__(self, path: Path):
        self.path, self.pos = path, path.stat().st_size if path.exists() else 0

    def new_lines(self):
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self.pos:
            self.pos = 0
        if size == self.pos:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos = size
        return data.decode("utf-8", "replace").splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=30)
    a = ap.parse_args()
    tails = {}
    rules = [
        ("orchestrator.log", "[gpu]", re.compile(r"start |done |FAILED|yielded|stop file|crashed")),
        ("orchestrator_cpu.log", "[cpu]", re.compile(r"start |done |FAILED|yielded|stop file|crashed")),
        ("BW_SELECT_v1.stdout", "[select]", re.compile(r"config_id|Traceback|Error")),
        ("BW_TEST_B2.stdout", "[tests]", re.compile(r"^\[tests\]|Traceback|Error")),
        ("BW_DISTILL_r1.search.stdout", "[distill]", re.compile(r"^chunk |Traceback|Error|verified")),
    ]
    stale = False
    while True:
        for name, tag, rx in rules:
            t = tails.setdefault(name, Tail(RUNS / name))
            for line in t.new_lines():
                if rx.search(line):
                    print(f"{tag} {line[:400]}", flush=True)
        for bp in sorted((RUNS / "bundles").glob("bundle*.stdout")):
            t = tails.setdefault(str(bp), Tail(bp))
            for line in t.new_lines():
                if "[bundle]" in line or "acquired" in line or "Traceback" in line:
                    print(f"[{bp.stem}] {line[:300]}", flush=True)
        for tp in RUNS.glob("BW_*_s*.stdout"):
            t = tails.setdefault(str(tp), Tail(tp))
            for line in t.new_lines():
                if re.search(r"step \d*00/|Traceback|Error|^done|resumed", line):
                    print(f"[{tp.stem}] {line[:200]}", flush=True)
        try:
            rec = json.loads((RUNS / "health_now.json").read_text())
            age = time.time() - rec["t"]
            if age > 150 and not stale:
                print(f"[health] STALE health_now.json ({age:.0f}s)", flush=True)
                stale = True
            elif age <= 150:
                stale = False
                if rec.get("tz") is not None and rec["tz"] >= 98:
                    print(f"[health] HOT tz01={rec['tz']}", flush=True)
                if rec.get("ram") is not None and rec["ram"] < 4:
                    print(f"[health] LOW RAM {rec['ram']} GB (commit {rec.get('commit')})", flush=True)
        except Exception:
            pass
        time.sleep(a.period)


if __name__ == "__main__":
    main()
