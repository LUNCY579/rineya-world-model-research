import importlib.util
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("decide_branch", ROOT / "experiments/RCX-E01R/decide_branch.py")
DB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DB)

FORMS = list(DB.FORMS.values())


def battery(n_par=12, tmpl_per_form=3):
    """Synthetic records: per form and polarity, n_par parents with 1 no-op and 1 apply item on p2, plus `in` items."""
    recs = []
    for pol, op in (("POS", "S+"), ("NEG", "S-")):
        for f in FORMS:
            for j in range(n_par):
                pid = f"{pol}-{f}-{j}"
                tmpl = f"{pol[0]}{f[:3]}{j % tmpl_per_form}"
                for dec in ("noop", "apply"):
                    for surf in ("in", "p2"):
                        recs.append({"surface": surf, "rule_ast": {"op": op}, "gold": {"decision": dec},
                                     "parent_id": pid, "p2_form": f if surf == "p2" else None,
                                     "p2_template": tmpl if surf == "p2" else None})
    return recs


def preds_for(recs, pos_rate, in_ok=1.0, neg_rate=0.95):
    """decision_ok per record: POS p2 no-op items correct by form-specific rule pos_rate(form, parent index)."""
    out = []
    for r in recs:
        if r["gold"]["decision"] == "apply":
            ok = 1
        elif r["surface"] == "in":
            ok = 1 if in_ok >= 1 else int(hash(r["parent_id"]) % 100 < in_ok * 100)
        elif r["rule_ast"]["op"] == "S+":
            j = int(r["parent_id"].rsplit("-", 1)[1])
            ok = int(j < round(pos_rate[r["p2_form"]] * 12))
        else:
            j = int(r["parent_id"].rsplit("-", 1)[1])
            ok = int(j < round(neg_rate * 12))
        out.append({"decision_ok": ok})
    return out


class TestBranch(unittest.TestCase):
    def run_case(self, rates, **kw):
        recs = battery()
        p = preds_for(recs, dict(zip(FORMS, rates)), **kw)
        return DB.decide(recs, {t: p for t in DB.TAGS}, [])

    def test_missing_seed_is_b0(self):
        self.assertEqual(DB.decide(battery(), {}, ["N-CFSD_s2 missing"])["branch"], "B0")

    def test_h_alt(self):
        self.assertEqual(self.run_case((0.9, 0.85, 0.83))["branch"], "B1")

    def test_form_specific_before_near_ceiling(self):
        q = self.run_case((1.0, 1.0, 0.25))
        self.assertGreaterEqual(q["pbar"], 0.65)
        self.assertEqual(q["branch"], "B3")
        self.assertEqual(q["arms"], ["R1", "R3"])

    def test_near_ceiling_uniform(self):
        self.assertEqual(self.run_case((0.75, 0.7, 0.7))["branch"], "B2")

    def test_uniform_deficit(self):
        q = self.run_case((0.15, 0.2, 0.1))
        self.assertEqual(q["branch"], "B4")
        self.assertEqual(q["arms"], ["R2", "R3"])

    def test_low_anchor_is_b0(self):
        recs = battery()
        p = preds_for(recs, dict(zip(FORMS, (0.2, 0.2, 0.2))))
        for r, x in zip(recs, p):
            if r["surface"] == "in" and r["gold"]["decision"] == "noop":
                x["decision_ok"] = 0
        self.assertEqual(DB.decide(recs, {t: p for t in DB.TAGS}, [])["branch"], "B0")


if __name__ == "__main__":
    unittest.main()
