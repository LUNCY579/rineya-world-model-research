"""Leak hunt: Blocksworld-hard vs B3 distillation data (read-only on the repo; outputs to stdout + this dir)."""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(sys.argv[1])
HERE = Path(__file__).parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from bwmn import blocks as B  # noqa: E402
import blocks_val_large as VL  # noqa: E402
from canon import canon, goal_canon_n, goal_canon_chain  # noqa: E402

assert "torch" not in sys.modules
R = lambda p: REPO / p  # noqa: E731
FS = lambda xs: frozenset(map(tuple, xs))  # noqa: E731
out = {}

# ------------------------------------------------------------------ hard set
hard_paths = sorted((B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"), key=lambda p: int(p.stem.split("-")[1]))
hard = {p.stem: B.parse_problem(p) for p in hard_paths}
hard_can = {k: canon(h[1], h[2], h[0]) for k, h in hard.items()}
hard_inv = {k: VL.invariant(*h) for k, h in hard.items()}
hard_gn = {k: goal_canon_n(h[2], h[0]) for k, h in hard.items()}
hard_gc = {k: goal_canon_chain(h[2]) for k, h in hard.items()}
can2hard = defaultdict(list)
for k, c in hard_can.items():
    can2hard[c].append(k)
gn2hard, gc2hard = defaultdict(list), defaultdict(list)
for k in hard:
    gn2hard[hard_gn[k]].append(k)
    gc2hard[hard_gc[k]].append(k)
out["hard_distinct_goal_configs_with_n"] = len(gn2hard)
out["hard_distinct_goal_configs_chain_only"] = len(gc2hard)


def act(s):
    return tuple(s.strip("() ").split())


# hard sub-problems: (s_t, goal) along the reference optimal plan (info.json), and along B2 / H2 submitted plans
info = json.loads((B.BENCH / "blocksworld_hard" / "info.json").read_text())
sub_can = defaultdict(set)  # canon -> {(hard id, source, t)}


def add_traj(k, plan, src):
    objs, s, g = hard[k]
    for t, a in enumerate(plan):
        if not B.applicable(s, a):
            return
        s, _ = B.step(s, a)
        sub_can[canon(s, g, objs)].add((k, src, t + 1))


for k in hard:
    add_traj(k, [act(x) for x in info[k + ".pddl"]["plan"].splitlines() if x.strip()], "ref")
for src, d in (("b2beam", "runs/BW_TEST_B2/hard_beam"), ("b2direct", "runs/BW_TEST_B2/hard_direct"),
               ("h2", "runs/BW_B3_HARD/h2_beam"), ("h1", "runs/BW_B3_HARD/h1_direct")):
    for line in R(d + "/plans.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        add_traj(r["id"].split("/")[1], [tuple(x.split()) for x in r["plan"]], src)
for c, ks in can2hard.items():
    sub_can[c].add((ks[0], "init", 0))
out["hard_subproblem_classes_incl_init"] = len(sub_can)

# ------------------------------------------------------------------ (1) rebuild the 240 distillation problems
# verbatim copy of scripts/bw_distill.py problems() (that module imports torch via blocks_planner -> lm)
SEED = 20261010


def problems(n):
    hardl = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
    sigs = B.hard_signatures()
    inv = {VL.invariant(*h) for h in hardl}
    for fn in ("instances_val_large.json", "instances_mirror_hard.json"):
        for r in json.loads((REPO / "runs/BW_val_large" / fn).read_text()):
            inv.add(VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
    rng = random.Random(SEED)
    sizes = [6, 7, 8, 8, 9, 9, 10, 10]
    o = VL.make(rng, [sizes[i % len(sizes)] for i in range(n)], sigs, inv)
    return [{"id": f"distill/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
             "goal": frozenset(map(tuple, r["goal"]))} for i, r in enumerate(o)]


dist = problems(240)
sols = [json.loads(l) for l in R("runs/BW_DISTILL_r1/solutions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
sol_by_id = {r["id"]: r for r in sols}
mismatch = [it["id"] for it in dist if it["id"] not in sol_by_id or (
    sol_by_id[it["id"]]["objs"] != it["objs"] or FS(sol_by_id[it["id"]]["init"]) != it["init"]
    or FS(sol_by_id[it["id"]]["goal"]) != it["goal"])]
out["rebuild_vs_solutions"] = {"rebuilt": len(dist), "solutions": len(sols), "unique_ids": len(sol_by_id),
                               "mismatches": mismatch}
iso = []
inv_share = []
for it in dist:
    c = canon(it["init"], it["goal"], it["objs"])
    if c in can2hard:
        iso.append((it["id"], can2hard[c]))
    v = VL.invariant(it["objs"], it["init"], it["goal"])
    inv_share += [(it["id"], k) for k, hv in hard_inv.items() if hv == v]
out["distill240_iso_to_hard"] = iso
out["distill240_pairs_sharing_coarse_invariant_with_hard"] = inv_share
out["distill240_blocks"] = dict(sorted(Counter(len(it["objs"]) for it in dist).items()))
# distill problem init vs any hard sub-problem (state along ref/B2/H2/H1 plans, same goal)
out["distill240_init_iso_to_hard_subproblem"] = [(it["id"], sorted(sub_can[c])[:3]) for it in dist
                                                  if (c := canon(it["init"], it["goal"], it["objs"])) in sub_can]

# (2) goal configuration overlap: distill problems
d_gn = Counter()
d_gc = Counter()
hard_hit_gn, hard_hit_gc = set(), set()
for it in dist:
    a, b = goal_canon_n(it["goal"], it["objs"]), goal_canon_chain(it["goal"])
    if a in gn2hard:
        d_gn[it["id"]] += 1
        hard_hit_gn.update(gn2hard[a])
    if b in gc2hard:
        d_gc[it["id"]] += 1
        hard_hit_gc.update(gc2hard[b])
out["goal_overlap_distill240"] = {
    "distill_problems_whose_goal_config_with_n_equals_a_hard_goal": len(d_gn),
    "hard_instances_whose_goal_config_with_n_appears_in_distill240": len(hard_hit_gn),
    "hard_ids_with_n": sorted(hard_hit_gn, key=lambda s: int(s.split("-")[1])),
    "distill_problems_goal_chain_only_equal_a_hard_goal": len(d_gc),
    "hard_instances_goal_chain_only_in_distill240": len(hard_hit_gc)}


# ------------------------------------------------------------------ row prompts
def parse_prompt(p):
    u = p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    d = json.loads(u)
    s = B.state_from_json(d["state"])
    g = frozenset(("on", x, y) for x, y in d["goal"])
    return s, g, B.blocks_of(s), d.get("outcome")


def scan_rows(path, label, limit=None):
    rows = [json.loads(l) for l in R(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    res = {"rows": len(rows), "iso_hard_problem": [], "iso_hard_subproblem": [], "blocks": Counter(),
           "outcome": Counter()}
    gnr, gcr = 0, 0
    hard_gn_hit, hard_gc_hit = set(), set()
    goal_cache = {}
    for i, r in enumerate(rows):
        s, g, objs, oc = parse_prompt(r["prompt"])
        res["blocks"][len(objs)] += 1
        res["outcome"][oc] += 1
        c = canon(s, g, objs)
        if c in can2hard:
            res["iso_hard_problem"].append((i, can2hard[c]))
        if c in sub_can:
            res["iso_hard_subproblem"].append((i, sorted(sub_can[c])[:3]))
        key = (g, tuple(objs))
        if key not in goal_cache:
            goal_cache[key] = (goal_canon_n(g, objs), goal_canon_chain(g))
        a, b = goal_cache[key]
        if a in gn2hard:
            gnr += 1
            hard_gn_hit.update(gn2hard[a])
        if b in gc2hard:
            gcr += 1
            hard_gc_hit.update(gc2hard[b])
    res["blocks"] = dict(sorted(res["blocks"].items()))
    res["outcome"] = dict(res["outcome"])
    res["rows_goal_config_with_n_equal_hard"] = gnr
    res["hard_instances_goal_config_with_n_hit"] = sorted(hard_gn_hit, key=lambda s: int(s.split("-")[1]))
    res["rows_goal_chain_only_equal_hard"] = gcr
    res["hard_instances_goal_chain_only_hit"] = len(hard_gc_hit)
    res["distinct_goals"] = len(goal_cache)
    out[label] = res
    return rows


mix_v3 = scan_rows("runs/BW_DISTILL_r1/mix_v3.jsonl", "mix_v3")
mix_v2c = scan_rows("runs/BW_DISTILL_r1/mix_v2c.jsonl", "mix_v2c")
scan_rows("runs/BW_DISTILL_r1/distill_rows.jsonl", "distill_rows")
v2all = scan_rows("runs/BW_data_v2/proposer_actions_train.jsonl", "v2_all_train_rows_(Pa_v2_init_data)")

# ------------------------------------------------------------------ provenance: rows <- solutions, mix <- rows + v2
drows = [json.loads(l) for l in R("runs/BW_DISTILL_r1/distill_rows.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rebuilt = []
for r in sols:
    if not r["verified"]:
        continue
    goal = FS(r["goal"])
    states = [FS(r["init"])] + [FS(s) for s in r["f_states"]]
    plan = [tuple(x) for x in r["plan"]]
    for t in range(len(plan)):
        rebuilt.append({"prompt": B.proposer_prompt(states[t], goal), "target": B.target_actions(plan[t:])})
out["distill_rows_rebuilt_equal"] = (len(rebuilt) == len(drows) and all(
    a["prompt"] == b["prompt"] and a["target"] == b["target"] for a, b in zip(rebuilt, drows)))
v2 = v2all
rng = random.Random(0)
pick = rng.sample(range(len(v2)), 2 * len(drows))
v3 = [{"prompt": r["prompt"], "target": r["target"]} for r in drows] + [
    {"prompt": v2[i]["prompt"], "target": v2[i]["target"]} for i in pick[:len(drows)]]
v2c = [{"prompt": v2[i]["prompt"], "target": v2[i]["target"]} for i in pick]
out["mix_v3_rebuilt_equal"] = v3 == [{"prompt": r["prompt"], "target": r["target"]} for r in mix_v3]
out["mix_v2c_rebuilt_equal"] = v2c == [{"prompt": r["prompt"], "target": r["target"]} for r in mix_v2c]
# f_states vs exact trajectory (labels the student saw)
fdiff = 0
for r in sols:
    if not r["verified"]:
        continue
    s = FS(r["init"])
    for a, fs in zip(r["plan"], r["f_states"]):
        s, _ = B.step(s, tuple(a))
        fdiff += s != FS(fs)
out["f_states_differing_from_exact"] = fdiff

# ------------------------------------------------------------------ distill F cache: any state outside the 240 problems?
dist_objsets = {tuple(it["objs"]) for it in dist}
fc_blocks = Counter()
fc_foreign = 0
for line in R("runs/BW_DISTILL_r1/fcache.jsonl").read_text(encoding="utf-8").splitlines():
    try:
        r = json.loads(line)
    except Exception:
        continue
    bl = tuple(B.blocks_of(FS(r["s"])))
    fc_blocks[len(bl)] += 1
    fc_foreign += bl not in dist_objsets
out["distill_fcache"] = {"keys_by_blocks": dict(sorted(fc_blocks.items())),
                         "keys_whose_block_set_is_not_a_distill_problem_block_set": fc_foreign}

json.dump(out, open(HERE / "audit_out.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str)[:20000])
