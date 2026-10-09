"""Apply the pre-registered selection rule (PREREG_PLANBENCH.md, addendum B2) to runs/BW_SELECT_v1/select.jsonl and
write the freeze records used by blocks_eval.py for the test sets. No test instance is read here.

Rule: among beam configurations evaluated on BOTH generated sets (val_large, val_v2), pick the highest mean success
subject to <= 2,000 unique F queries per instance on each set; ties -> fewer proposer tokens per instance. S-direct is
the direct planner with room for a complete plan (M = 256). Records adapter SHA-256s and this PREREG file's SHA-256.
Writes runs/FREEZE_B2_beam.json and runs/FREEZE_B2_direct.json and prints the choice.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import lm  # noqa: E402

PA, FWD = "runs/BW_Pa_v2_s1/adapter", "runs/BW_Fdelta_s0/adapter"


def main():
    res = {}
    for line in (ROOT / "runs/BW_SELECT_v1/select.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        res.setdefault(r["config_id"], {})[r["set"]] = r
    cands = []
    for cid, by in res.items():
        cfg = json.loads(cid)
        if cfg["planner"] != "beam" or not {"val_large", "val_v2"} <= set(by):
            continue
        if any(by[s]["f_queries_per_instance"] > 2000 for s in by):
            continue
        mean = (by["val_large"]["success"] + by["val_v2"]["success"]) / 2
        tok = (by["val_large"]["proposer_tokens_per_instance"] + by["val_v2"]["proposer_tokens_per_instance"]) / 2
        cands.append((mean, -tok, cid, cfg))
    if not cands:
        sys.exit("no eligible beam configuration with results on both sets")
    cands.sort(reverse=True)
    mean, ntok, cid, cfg = cands[0]
    print("eligible:", [(round(m, 3), round(-t), c) for m, t, c, _ in cands])
    print("selected:", cid, "mean success", round(mean, 3))
    common = {"proposer": "lm", "pa": PA, "mode": "actions", "eval_seed": 0, "max_depth": 80, "step_cap": 24}
    beam = dict(common, fwd="lm", fwd_adapter=FWD, fwd_fmt=cfg.get("fwd", "delta"), K=cfg["K"], planner="beam",
                W=cfg["W"], E=cfg["E"], R=cfg.get("R", 0), M=cfg.get("M", 0), P=cfg.get("P", 0))
    direct = dict(common, fwd="exact", fwd_adapter=None, fwd_fmt="state", K=1, planner="direct", W=4, E=48, R=0,
                  M=256, P=0)
    meta = {"prereg_sha256": lm.file_sha256(ROOT / "PREREG_PLANBENCH.md"),
            "pa_sha256": lm.file_sha256(ROOT / PA / "adapter_model.safetensors"),
            "fwd_sha256": lm.file_sha256(ROOT / FWD / "adapter_model.safetensors"),
            "selected_from": cid, "selection_mean_success": mean, "rule": __doc__.split("\n\n")[1].strip()}
    (ROOT / "runs/FREEZE_B2_beam.json").write_text(json.dumps({"args": beam, **meta}, indent=1))
    (ROOT / "runs/FREEZE_B2_direct.json").write_text(json.dumps({"args": direct, **meta}, indent=1))
    print(json.dumps({"beam": beam, "direct": direct, **{k: v for k, v in meta.items() if k != "rule"}}, indent=1))


if __name__ == "__main__":
    main()
