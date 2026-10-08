"""Exact-state generative accuracy per arm / family / horizon / mode on the TEST split."""
import json, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcwmt import data, env

def summarize(bank_stem, armname):
    rows = data.candidate_table(data.load_bank(bank_stem))
    d = ROOT / "runs" / bank_stem / armname
    out = {}
    e2e = {r["item_id"]: r for r in map(json.loads, (d / "gen_e2e.jsonl").read_text(encoding="utf-8").splitlines())} if (d / "gen_e2e.jsonl").exists() else {}
    chains = {}
    for m in ("chain_one", "chain_m1"):
        p = d / f"gen_{m}.jsonl"
        if p.exists():
            chains[m] = {r["cand_id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    for fam in env.FAMILIES:
        te = [r for r in rows if r["family"] == fam and r["split"] == "test"]
        for k in (1, 2, 4):
            tk = [r for r in te if r["k"] == k]
            res = {}
            if e2e:
                g = [e2e.get(r["item_id"]) for r in tk]
                res["e2e"] = sum(1 for r, x in zip(tk, g) if x and x["exact"]) / len(tk)
                res["e2e_parse"] = sum(1 for x in g if x and x["parsed"]) / len(tk)
                if k == 1:
                    o = [e2e.get(r["cand_id"] + "-one") for r in tk]
                    res["one_frame"] = sum(1 for x in o if x and x["exact"]) / len(tk)
            for m, c in chains.items():
                res[m] = sum(1 for r in tk if c.get(r["cand_id"]) and c[r["cand_id"]]["states"][k] == r["gold"]) / len(tk)
            out[f"{fam}|k{k}"] = res
    return out

if __name__ == "__main__":
    bank = sys.argv[1]
    for arm in sys.argv[2:]:
        s = summarize(bank, arm)
        for key, v in s.items():
            print(f"{arm:<8} {key:<24} " + " ".join(f"{m}={x:.3f}" for m, x in v.items()))
