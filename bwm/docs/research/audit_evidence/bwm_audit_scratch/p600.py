import json
from pathlib import Path
from scipy.stats import binomtest
R = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
L = json.loads((R / "bench/planbench/LEDGER.json").read_text(encoding="utf-8"))
reg = set(L["deepseek-r1-api|task_1_plan_generation_zero_shot"]["per_instance"])
allsame = all(set(v["per_instance"]) == reg for k, v in L.items() if v.get("n") == 600)
print("registered 600 ids:", len(reg), "all n=600 ledgers share ids:", allsame)
def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")}
b = load(R / "runs/BW_TEST_B2/planbench_beam/plans.jsonl")
d = load(R / "runs/BW_TEST_B2/planbench_direct/plans.jsonl")
extra = sorted(set(b) - reg); missing = sorted(reg - set(b))
print("extra in run:", extra, "missing from run:", missing)
for i in extra:
    print(i, "beam", b[i].get("val"), "direct", d[i].get("val"), "n_blocks", b[i].get("n_blocks"))
ok = lambda r: bool(r.get("val", r["success"]))
kb = sum(ok(b[i]) for i in reg); kd = sum(ok(d[i]) for i in reg)
plus = sum(ok(b[i]) and not ok(d[i]) for i in reg); minus = sum(ok(d[i]) and not ok(b[i]) for i in reg)
print(f"registered-600: beam {kb}/600 direct {kd}/600 +{plus}/-{minus} p={binomtest(plus, plus+minus, .5).pvalue:.3g}")
for k in ("deepseek-r1-api|task_1_plan_generation_zero_shot", "o1-preview_chat|task_1_plan_generation_zero_shot"):
    pi = L[k]["per_instance"]
    p_ = sum(ok(b[i]) and not pi[i] for i in reg); m_ = sum(pi[i] and not ok(b[i]) for i in reg)
    print(k, L[k]["val_success"], f"beam-only {p_} pub-only {m_} p={binomtest(p_, p_+m_, .5).pvalue:.3g}")
