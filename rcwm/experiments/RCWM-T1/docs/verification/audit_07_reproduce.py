"""Audit 7: reproduce C1 (ledger one-frame ridge/MLP, both arms, parent bootstrap) and C2 (rotation
selectivity) with the current code."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, fidelity, controls, decide
rows = data.candidate_table(data.load_bank("bank_pilot"))
rows_one = [dict(r, item_id=r["cand_id"] + "-one") for r in rows if r["k"] == 1]
res = {}
for armname in ("sparse", "supa"):
    arm = data.Arm("bank_pilot", armname)
    r = fidelity.analyse(arm, rows_one, "ledger_replay", 1, None, mlp=True, symbolic=False)
    res[armname] = r
    print(armname, "ridge", round(r["latent_ridge"]["exact_delta"], 3), "L", r["latent_ridge"]["layer"], "lam", r["latent_ridge"]["lambda"],
          "mlp", round(r["latent_mlp"]["exact_delta"], 3), "| touched floor", round(r["ladder"]["touched_rule"]["exact_delta"], 3))
    c = controls.control_labels(arm, rows_one, "ledger_replay", 1)
    print("   control: true", round(c["true_exact"], 3), "cf", round(c["cf_exact"], 3), "sel_disc", round(c["selectivity_discriminating"], 3), "n_disc", c["n_discriminating"])
te = [r for r in rows_one if r["family"] == "ledger_replay" and r["split"] == "test"]
for m in ("latent_ridge", "latent_mlp"):
    pa = fidelity._parent_mean(te, np.array(res["sparse"]["_per_item"][m]))
    pb = fidelity._parent_mean(te, np.array(res["supa"]["_per_item"][m]))
    print(m, "sparse-supa", decide.paired_bootstrap(pa, pb))
