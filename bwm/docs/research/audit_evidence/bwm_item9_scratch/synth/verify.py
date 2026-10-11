"""Synthesis verification (read-only on the clone; pure python; no torch). Recomputes the PB Logistics strata,
candidate test lists and published per-instance counts that the item-9 prereg draft quotes."""
import json, re, sys, hashlib, collections, statistics as st
from pathlib import Path
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
CL = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning")
GB = CL / "plan-bench/instances/logistics/generated_basic"
GT = {int(x["instance_id"]): [l for l in x["ground_truth_plan"].strip().splitlines() if l.strip()]
      for x in json.loads((CL / "plan-bench/prompts/logistics/task_1_plan_generation.json").read_text())["instances"]}


def parse(p):
    t = p.read_text().lower()
    init = t.split("(:init")[1].split("(:goal")[0]
    goal = t.split("(:goal")[1]
    facts = [tuple(f.split()) for f in re.findall(r"\(([^()]+)\)", init)]
    gf = [tuple(f.split()) for f in re.findall(r"\(([^()]+)\)", goal) if not f.strip().startswith("and")]
    ty = collections.defaultdict(list)
    for f in facts:
        if len(f) == 2:
            ty[f[0]].append(f[1])
    incity = collections.Counter(f[2] for f in facts if f[0] == "in-city")
    return {"np": len(ty["obj"]), "nc": len(ty["city"]), "na": len(ty["airplane"]), "nt": len(ty["truck"]),
            "nl": len(ty["location"]), "size": max(incity.values()), "goal": gf, "nfacts": len(facts),
            "nobj": sum(len(v) for k, v in ty.items() if k in ("obj", "truck", "airplane", "location", "city"))}


I = {int(p.stem.split("-")[1]): parse(p) for p in GB.glob("instance-*.pddl")}
print("instances", len(I), "GT ids", min(GT), max(GT), len(GT))
pk = lambda i: I[i]["np"]
ids = range(2, 287)
print("t1 2-286 by pk", sorted(collections.Counter(pk(i) for i in ids).items()))
OUT = [i for i in ids if pk(i) >= 4]
OUTA = [i for i in OUT if i <= 200]
OUTB = [i for i in OUT if i >= 201]
IN = [i for i in ids if pk(i) <= 3]
print("PB-OUT", len(OUT), sorted(collections.Counter(pk(i) for i in OUT).items()))
print("PB-OUT-A (2-200)", len(OUTA), sorted(collections.Counter(pk(i) for i in OUTA).items()))
print("PB-OUT-B (201-286)", len(OUTB), sorted(collections.Counter(pk(i) for i in OUTB).items()), "201 pk", pk(201))
print("PB-IN", len(IN), sorted(collections.Counter(pk(i) for i in IN).items()))
print("1-pkg ids in 202-286", [i for i in range(202, 287) if pk(i) < 7])
gl = lambda S: [len(GT[i]) for i in S]
for nm, S in [("PB-OUT", OUT), ("PB-OUT-A", OUTA), ("PB-OUT-B", OUTB), ("PB-IN", IN)]:
    g = gl(S)
    print(nm, "GT len min/median/mean/max", min(g), st.median(g), round(st.mean(g), 2), max(g))
for k in range(1, 8):
    g = [len(GT[i]) for i in ids if pk(i) == k]
    print(" pk", k, "n", len(g), "GT", min(g), max(g), round(st.mean(g), 1))
print("goal == one 'at' per package:", all(len(I[i]["goal"]) == pk(i) and all(f[0] == "at" for f in I[i]["goal"]) for i in I))
print("trucks==cities", all(I[i]["nt"] == I[i]["nc"] for i in I), "planes<=cities", all(I[i]["na"] <= I[i]["nc"] for i in I))
print("max objs", max(I[i]["nobj"] for i in I), "max init facts", max(I[i]["nfacts"] for i in I))

# lean list
def every_k(pool, per):
    out = []
    for k in (4, 5, 6, 7):
        s = sorted(i for i in pool if pk(i) == k)
        out += [s[int(j * len(s) / per)] for j in range(min(per, len(s)))]
    return out
lean = every_k(OUTA, 10)
print("lean40", lean, hashlib.sha256(json.dumps(lean).encode()).hexdigest())
# synthesis candidate: 20 per package count from the whole PB-OUT, every-k-th by id
s80 = every_k(OUT, 20)
print("synth80", s80)
print("synth80 sha", hashlib.sha256(json.dumps(s80).encode()).hexdigest())
print("synth80 in 2-200:", sum(i <= 200 for i in s80), "by pk", {k: sum(1 for i in s80 if pk(i) == k and i <= 200) for k in (4, 5, 6, 7)})
g = gl(s80); print("synth80 GT min/mean/max", min(g), round(st.mean(g), 1), max(g),
                   {k: (min(len(GT[i]) for i in s80 if pk(i) == k), round(st.mean([len(GT[i]) for i in s80 if pk(i) == k]), 1), max(len(GT[i]) for i in s80 if pk(i) == k)) for k in (4, 5, 6, 7)})
print("synth80 (cities,planes)", sorted(collections.Counter((I[i]["nc"], I[i]["na"]) for i in s80).items()))
REST = [i for i in OUT if i not in s80]
print("PB-OUT-REST", len(REST), sorted(collections.Counter(pk(i) for i in REST).items()))
R = CL / "llm_planning_analysis/results/logistics"
files = {"o1-preview 0-shot PDDL": "o1-preview_chat/task_1_plan_generation_zero_shot_pddl.json",
         "GPT-4 1-shot NL": "gpt-4_chat/task_1_plan_generation.json",
         "GPT-4 1-shot PDDL": "gpt-4_chat/task_1_plan_generation_pddl.json",
         "GPT-4 0-shot NL": "gpt-4_chat/task_1_zero_shot_plan_generation.json",
         "GPT-4 0-shot PDDL": "gpt-4_chat/task_1_zero_shot_plan_generation_pddl.json",
         "LLaMA-3.1-405B 0-shot PDDL": "llama-3.1-405b_aws/task_1_plan_generation_zero_shot_pddl.json",
         "GPT-3.5-instruct 1-shot NL": "gpt-3.5-turbo-instruct/task_1_plan_generation.json"}
for name, f in files.items():
    ins = json.loads((R / f).read_text())["instances"]
    fl = {int(x["instance_id"]): bool(x.get("llm_correct", x.get("correct"))) for x in ins}
    def c(S):
        cov = [i for i in S if i in fl]
        return f"{sum(fl[i] for i in cov)}/{len(cov)}"
    print(f"{name}: file {c(list(fl))} ids {min(fl)}-{max(fl)} | PB-OUT {c(OUT)} | OUT-A {c(OUTA)} | lean40 {c(lean)} | synth80 {c(s80)} | PB-IN {c(IN)}")
bp = json.loads((R / "gpt-4_chat/task_1_plan_generation_backprompting.json").read_text())["instances"]
print("backprompting file n", len(bp), "ids", min(int(x["instance_id"]) for x in bp), max(int(x["instance_id"]) for x in bp))
json.dump({"OUT": OUT, "OUTA": OUTA, "OUTB": OUTB, "IN": IN, "lean40": lean, "synth80": s80, "REST": REST},
          open(Path(__file__).with_name("sets.json"), "w"))
