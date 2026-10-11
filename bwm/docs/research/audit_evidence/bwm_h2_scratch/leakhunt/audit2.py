"""Characterise distill-row / hard sub-problem coincidences against a chance baseline (mirror, b3fresh)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(sys.argv[1])
HERE = Path(__file__).parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from bwmn import blocks as B  # noqa: E402
from canon import canon, goal_canon_n  # noqa: E402

assert "torch" not in sys.modules
R = lambda p: REPO / p  # noqa: E731
FS = lambda xs: frozenset(map(tuple, xs))  # noqa: E731
jl = lambda p: [json.loads(l) for l in R(p).read_text(encoding="utf-8").splitlines() if l.strip()]  # noqa: E731

hard_paths = sorted((B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"), key=lambda p: int(p.stem.split("-")[1]))
sets = {"hard": {f"hard/{p.stem}": B.parse_problem(p) for p in hard_paths}}
for name, path in (("mirror", "runs/BW_val_large/instances_mirror_hard.json"),
                   ("b3fresh", "runs/BW_val_large/instances_b3_fresh.json")):
    sets[name] = {f"{name}/{i}": (r["objs"], FS(r["init"]), FS(r["goal"])) for i, r in
                  enumerate(json.loads(R(path).read_text()))}
info = json.loads((B.BENCH / "blocksworld_hard" / "info.json").read_text())

srcs = {"hard": {"b2beam": "runs/BW_TEST_B2/hard_beam/plans.jsonl", "b2direct": "runs/BW_TEST_B2/hard_direct/plans.jsonl",
                 "v3direct": "runs/BW_B3_HARD/h1_direct/plans.jsonl", "v3beam(H2)": "runs/BW_B3_HARD/h2_beam/plans.jsonl"},
        "mirror": {"b2beam": "runs/BW_TEST_B2/mirror_beam/plans.jsonl", "b2direct": "runs/BW_TEST_B2/mirror_direct/plans.jsonl",
                   "v3direct": "runs/BW_B3_EVAL/plans_v3_mirror.jsonl"},
        "b3fresh": {"v2direct": "runs/BW_B3_EVAL/plans_v2_b3fresh.jsonl", "v3direct": "runs/BW_B3_EVAL/plans_v3_b3fresh.jsonl"}}


def traj(inst, plan):
    objs, s, g = inst
    yield 0, canon(s, g, objs)
    for t, a in enumerate(plan):
        if not B.applicable(s, a):
            return
        s, _ = B.step(s, a)
        yield t + 1, canon(s, g, objs)


sub = {}  # set -> source -> canon -> {inst ids}
for sname, d in srcs.items():
    sub[sname] = {}
    for src, p in d.items():
        m = defaultdict(set)
        for r in jl(p):
            for t, c in traj(sets[sname][r["id"]], [tuple(x.split()) for x in r["plan"]]):
                m[c].add(r["id"])
        sub[sname][src] = m
m = defaultdict(set)
for k, inst in sets["hard"].items():
    plan = [tuple(x.strip("() ").split()) for x in info[k.split("/")[1] + ".pddl"]["plan"].splitlines() if x.strip()]
    for t, c in traj(inst, plan):
        m[c].add(k)
sub["hard"]["ref_optimal"] = m

drows = jl("runs/BW_DISTILL_r1/distill_rows.jsonl")
parsed = []
for r in drows:
    u = r["prompt"].split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    d = json.loads(u)
    s = B.state_from_json(d["state"])
    g = frozenset(("on", x, y) for x, y in d["goal"])
    parsed.append((canon(s, g, B.blocks_of(s)), len(json.loads(r["target"])["actions"]), len(B.blocks_of(s))))
rem_all = Counter(min(x[1], 99) for x in parsed)
res = {}
for sname in sub:
    for src, mp in sub[sname].items():
        hit = [(i, x) for i, x in enumerate(parsed) if x[0] in mp]
        insts = set().union(*[mp[x[0]] for _, x in hit]) if hit else set()
        rem = Counter(x[1] for _, x in hit)
        res[f"{sname}:{src}"] = {"distill_rows_matching": len(hit), "instances_touched": len(insts),
                                 "remaining_actions_of_matching_rows": dict(sorted(rem.items())),
                                 "max_remaining": max(rem) if rem else None}
for k, v in res.items():
    print(k, v)
n10 = lambda s: [k for k, v in sets[s].items() if len(v[0]) <= 10]  # noqa: E731
print("distill rows remaining-actions distribution (all 4254):", dict(sorted(rem_all.items())))

# goal-config overlap baseline (n <= 10 only, the distill range)
sols = jl("runs/BW_DISTILL_r1/solutions.jsonl")
dgoals = Counter(goal_canon_n(FS(r["goal"]), r["objs"]) for r in sols)
for s in sets:
    ks = n10(s)
    hit = [k for k in ks if goal_canon_n(sets[s][k][2], sets[s][k][0]) in dgoals]
    print(f"goal-config(with n) overlap with distill240 goals: {s}: {len(hit)}/{len(ks)} instances with n<=10;"
          f" all n: {sum(goal_canon_n(v[2], v[0]) in dgoals for v in sets[s].values())}/{len(sets[s])}")

# H1/H2 discordant pairs vs goal-overlap flag (descriptive)
def load(p):
    return {r["id"]: bool(r.get("val", r["success"])) for r in jl(p)}


flag = {k: goal_canon_n(v[2], v[0]) in dgoals for k, v in sets["hard"].items()}
for name, new, old in (("H1", "runs/BW_B3_HARD/h1_direct/plans.jsonl", "runs/BW_TEST_B2/hard_direct/plans.jsonl"),
                       ("H2", "runs/BW_B3_HARD/h2_beam/plans.jsonl", "runs/BW_TEST_B2/hard_beam/plans.jsonl")):
    x, y = load(new), load(old)
    tab = Counter()
    for k in x:
        cell = "v3_only" if x[k] and not y[k] else "b2_only" if y[k] and not x[k] else "both" if x[k] else "neither"
        tab[(cell, "goal_in_distill" if flag[k] else "goal_not_in_distill")] += 1
    print(name, dict(sorted(tab.items())), "succ v3", sum(x.values()), "succ b2", sum(y.values()))
    # restricted to n <= 10 and n > 10
    for lo, hi in ((6, 10), (11, 15)):
        b = sum(x[k] and not y[k] for k in x if lo <= len(sets["hard"][k][0]) <= hi)
        c = sum(y[k] and not x[k] for k in x if lo <= len(sets["hard"][k][0]) <= hi)
        print(f"   {name} n {lo}-{hi}: v3_only {b} b2_only {c}")
