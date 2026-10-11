"""Per-step executor vs VAL -v trace on all published plans (normalised rendering)."""
import sys, json
from collections import Counter
from pathlib import Path
HERE = Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_item9_scratch/impl/review_sem")
src = (HERE / "fuzz_val.py").read_text().split("t0 = time.time()")[0]
sys.argv = ["x", "1", "0"]
exec(src)  # defines L, val_trace, compare, ...

stats = Counter()
bad = []
probs = {}
for f in sorted(L.RESULTS_DIR.rglob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for x in d["instances"]:
        iid = int(x["instance_id"])
        if iid not in probs:
            probs[iid] = L.parse_problem(L.planbench_path(iid))
        objs, init, goal = probs[iid]
        raw = x.get("extracted_llm_plan")
        raw = "\n".join(raw) if isinstance(raw, list) else (raw or "")
        acts = L.plan_from_pddl(raw)
        if any(a and (a[0] not in L.ARITY or len(a) - 1 != L.ARITY[a[0]]) for a in acts) or any(not a for a in acts):
            # VAL crashes / errors on these; compare only the prefix before the first malformed action
            k = next(i for i, a in enumerate(acts) if not a or a[0] not in L.ARITY or len(a) - 1 != L.ARITY[a[0]])
            stats["malformed_truncated"] += 1
            acts = acts[:k]
            # outcome of the truncated prefix may legitimately differ from the full plan; compare traces only
        compare(L.planbench_path(iid), init, goal, acts, "pub", stats, bad)
print(dict(stats), "bad", len(bad))
for b in bad[:5]:
    print(json.dumps(b, default=str)[:1500])
