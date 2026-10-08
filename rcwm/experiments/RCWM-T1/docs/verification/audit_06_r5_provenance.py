"""Audit 6: which item set reproduces the RESULTS_PILOT R5 table, and within- vs pooled-family CKA."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rcwmt import data, controls
sys.path.insert(0, str(ROOT / "scripts"))
from mechanism_report import cka, shift
sp, su = data.Arm("bank_pilot", "sparse"), data.Arm("bank_pilot", "supa")
it = su.items
sets = {
 "test_one_all": [x["item_id"] for x in it if x["split"] == "test" and x["format"] == "one"],
 "test_one_L": [x["item_id"] for x in it if x["split"] == "test" and x["format"] == "one" and x["family"] == "ledger_replay"],
 "test_one_P": [x["item_id"] for x in it if x["split"] == "test" and x["format"] == "one" and x["family"] == "prerequisite_plan"],
 "test_one_R": [x["item_id"] for x in it if x["split"] == "test" and x["format"] == "one" and x["family"] == "register_machine"],
 "test_all_items": [x["item_id"] for x in it if x["split"] == "test"],
 "test_k1_all": [x["item_id"] for x in it if x["split"] == "test" and x["format"] == "multi" and x["k"] == 1],
 "all_one": [x["item_id"] for x in it if x["format"] == "one"],
}
print("R5 target: relL2(all) .16/.40/.65/.75/.57; noMassive .24/.53/.81/1.15/.73; med|z| 1.03/1.58/1.61/1.61/1.36; CKA .991/.953/.965/.925/.937")
for name, ids in sets.items():
    out = []
    for L in (12, 16, 20, 24, 28):
        A = su.X_last(ids, L).astype(np.float64); B = sp.X_last(ids, L).astype(np.float64)
        massive = np.argsort(-np.abs(A).mean(0))[:5]
        s = shift(A, B, massive)
        out.append(f"L{L} {np.linalg.norm(B-A)/np.linalg.norm(A):.2f}/{s['rel_l2_nomassive']:.2f}/{s['median_abs_z']:.2f}/{s['cka']:.3f}")
    print(f"{name:<15} n={len(ids):>5} " + " | ".join(out))
# what does the controls.state_shift in h2b_controls_v2 (rel_l2 .1645 at L12) correspond to?
for name in ("test_one_all", "test_one_L", "test_k1_all", "test_all_items"):
    r = controls.state_shift(su, sp, sets[name], layers=[12, 24])
    print("state_shift", name, {L: round(v["rel_l2"], 4) for L, v in r.items()})
