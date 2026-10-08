"""Audit 11: parent-level bootstrap CI for the C2 ledger selectivity on discriminating items."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, fidelity, controls, decide
rows = data.candidate_table(data.load_bank("bank_pilot"))
rows_one = [dict(r, item_id=r["cand_id"] + "-one") for r in rows if r["k"] == 1]
fam = "ledger_replay"
for armname in ("sparse", "supa"):
    arm = data.Arm("bank_pilot", armname)
    for tag, rr in (("one", rows_one), ("k1", rows)):
        c = controls.control_labels(arm, rr, fam, 1)
        te = [r for r in rr if r["family"] == fam and r["k"] == 1 and r["split"] == "test"]
        disc = np.array([not np.array_equal(fidelity.delta_of(fam, r["s0"], r["states"][-1]),
                                            fidelity.delta_of(fam, r["s0"], controls.rollout_cf(fam, r["s0"], r["events"]))) for r in te])
        t = np.array(c["per_item_true"])[disc]; f = np.array(c["per_item_cf"])[disc]
        tr = [r for r, m in zip(te, disc) if m]
        bs = decide.paired_bootstrap(fidelity._parent_mean(tr, t), fidelity._parent_mean(tr, f))
        print(f"{armname} {tag}: sel_disc item-mean {c['selectivity_discriminating']:+.3f} | parent-mean {bs['mean_diff']:+.3f} CI {np.round(bs['ci95'], 3).tolist()} n_parents {bs['n']}")
