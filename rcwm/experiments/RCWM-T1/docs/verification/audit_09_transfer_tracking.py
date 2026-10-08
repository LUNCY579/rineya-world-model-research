"""Audit 9: transfer at feature level (is 'no transfer' near chance?) and sparse event-boundary tracking (C4)."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, fidelity, controls
rows = data.candidate_table(data.load_bank("bank_pilot"))
rows_one = [dict(r, item_id=r["cand_id"] + "-one") for r in rows if r["k"] == 1]
sp, su = data.Arm("bank_pilot", "sparse"), data.Arm("bank_pilot", "supa")
fn = lambda r: fidelity.delta_of("ledger_replay", r["s0"], r["states"][-1])
for a, b, an, bn in ((sp, su, "sparse", "supa"), (su, sp, "supa", "sparse")):
    _, aa = controls._fit_eval(a, a, rows_one, "ledger_replay", 1, fn, layer=16)
    _, ab = controls._fit_eval(a, b, rows_one, "ledger_replay", 1, fn, layer=16, restandardize=True)
    print(f"{an}->{an}: exact {aa['exact_delta']:.3f} feat {aa['feat_acc']:.3f} bal {aa['balanced_acc']:.3f} | {an}->{bn} restd: exact {ab['exact_delta']:.3f} feat {ab['feat_acc']:.3f} bal {ab['balanced_acc']:.3f} per-feat {np.round(ab['per_feature_acc'], 2).tolist()}")
te = [r for r in rows_one if r["family"] == "ledger_replay" and r["split"] == "test"]
D = np.stack([fn(r) for r in te]); print("copy-s0 feat acc", float((D == 0).mean()))
for fam in ("ledger_replay", "register_machine", "prerequisite_plan"):
    t = fidelity.tracking(sp, rows, fam, 4)
    print("sparse tracking", fam, {j: (round(v["exact_delta"], 3), round(v["copy_floor_exact"], 3), v["layer"]) for j, v in t.items()})
