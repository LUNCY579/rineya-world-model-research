import json, sys, collections
a = json.load(open(sys.argv[1], encoding="utf-8"))["instances"]
b = json.load(open(sys.argv[2], encoding="utf-8"))
print("task2 keys", list(b.keys()), "prompt_type", b.get("prompt_type")); b = b["instances"]
print("task2 instance keys", list(b[0].keys()))
ex = collections.Counter(tuple(x["example_instance_ids"]) if isinstance(x["example_instance_ids"], list) else x["example_instance_ids"] for x in a)
print("example_instance_ids (top5)", ex.most_common(5))
L = lambda s: len([l for l in s.strip().splitlines() if l.strip() and not l.startswith(";")])
ma = {x["instance_id"]: L(x["ground_truth_plan"]) for x in a}
mb = {x["instance_id"]: L(x["ground_truth_plan"]) for x in b if x.get("ground_truth_plan")}
common = sorted(set(ma) & set(mb))
diff = [(i, ma[i], mb[i]) for i in common if ma[i] != mb[i]]
print("common", len(common), "length mismatches", len(diff), diff[:10])
# first 200 ids subset stats
ids = sorted(ma); first200 = ids[:200]
import statistics
print("first-200-ids range", first200[0], first200[-1], "mean len %.2f" % statistics.mean(ma[i] for i in first200), "max", max(ma[i] for i in first200))
