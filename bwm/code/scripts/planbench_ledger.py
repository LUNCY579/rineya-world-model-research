"""Re-score published PlanBench Task 1 plans with the local VAL (same validator for every system).

Writes bench/planbench/LEDGER.json: per model and prompt type -> {n, val_success, file_flag_success, per_instance}.
Per-instance VAL verdicts allow paired McNemar tests against our systems on identical instance IDs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402

RES = B.BENCH / "results"


def plan_from_text(t):
    if isinstance(t, list):
        t = chr(10).join(map(str, t))
    acts = []
    for line in (t or "").strip().splitlines():
        line = line.strip().strip("()").strip()
        if not line:
            continue
        acts.append(tuple(line.lower().split()))
    return acts


def main():
    ledger = {}
    for task_file in ("task_1_plan_generation_zero_shot.json", "task_1_plan_generation.json"):
        for sub, inst_dir in (("blocksworld", "generated_basic"), ("blocksworld_3", "generated_basic_3")):
            for mdir in sorted((RES / sub).iterdir()):
                f = mdir / task_file
                if not f.exists():
                    continue
                d = json.loads(f.read_text(encoding="utf-8"))
                key = f"{mdir.name}|{task_file.replace('.json', '')}"
                L = ledger.setdefault(key, {"n": 0, "val_success": 0, "file_flag_success": 0, "per_instance": {}})
                for r in d["instances"]:
                    iid = r["instance_id"]
                    p = B.BENCH / "blocksworld" / inst_dir / f"instance-{iid}.pddl"
                    if not p.exists():
                        continue
                    plan = plan_from_text(r.get("extracted_llm_plan", ""))
                    ok = B.val_check(p, plan) if plan else False
                    L["n"] += 1
                    L["val_success"] += ok
                    L["file_flag_success"] += str(r.get("llm_correct")) == "True"
                    L["per_instance"][f"{inst_dir}/instance-{iid}"] = ok
                print(key, L["n"], L["val_success"], L["file_flag_success"], flush=True)
    out = B.BENCH / "LEDGER.json"
    out.write_text(json.dumps(ledger, indent=0))
    print("written", out)


if __name__ == "__main__":
    main()
