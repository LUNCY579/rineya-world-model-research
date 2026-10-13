"""Contamination audit, METHOD B (independent canonical form; no repository helpers imported).

Read-only on the project. Pure stdlib, single process, streams every jsonl.
Problem -> labelled typed digraph on blocks:
  unary fact  (p x)    in init/goal -> vertex label "I:p" / "G:p"
  binary fact (p x y)  in init/goal -> typed edge "I:p" / "G:p"  x -> y
  nullary fact (p)     in init/goal -> global label "I:p" / "G:p"
Two problems are equal up to a bijective renaming of blocks  <=>  their labelled graphs are isomorphic
(every fact is encoded exactly once, and a vertex bijection preserves all labels/edges iff it maps the init fact set
onto the init fact set and the goal fact set onto the goal fact set).
Canonical form = lexicographically minimal leaf certificate of an individualisation-refinement search tree
(1-WL colour refinement over typed in/out edges; target cell = lowest-coloured non-singleton cell; EVERY vertex
of the target cell is tried, pruned only by automorphisms already found that fix the current prefix pointwise).
Every reported match is then confirmed by an explicit isomorphism check: the bijection obtained from the two
canonical labellings is applied to the training facts and compared to the eval facts.
"""
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict

ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
OUT = sys.argv[1]
LEAF_CAP = 200000
STATS = {"max_leaves": 0, "cap_hits": 0, "canon_calls": 0}


# ------------------------------------------------------------------ canonical form
def _refine(col, n, adj):
    k = len(set(col))
    while True:
        sig = [(col[v], tuple(sorted((t, d, col[w]) for t, d, w in adj[v]))) for v in range(n)]
        order = {s: i for i, s in enumerate(sorted(set(sig)))}
        new = [order[s] for s in sig]
        if len(order) == k:
            return new
        k = len(order)
        col = new


def canon(verts, facts_by_part):
    """verts: list of block names. facts_by_part: list of (prefix, iterable of fact tuples).
    Returns (certificate, labelling dict name->canonical index)."""
    STATS["canon_calls"] += 1
    idx = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    vlab = [[] for _ in range(n)]
    glob = []
    edges = []
    adj = [[] for _ in range(n)]
    for pre, facts in facts_by_part:
        for f in facts:
            p = pre + ":" + f[0]
            if len(f) == 1:
                glob.append(p)
            elif len(f) == 2:
                vlab[idx[f[1]]].append(p)
            elif len(f) == 3:
                u, w = idx[f[1]], idx[f[2]]
                edges.append((p, u, w))
                adj[u].append((p, 0, w))
                adj[w].append((p, 1, u))
            else:
                raise ValueError(f)
    vlab = [tuple(sorted(x)) for x in vlab]
    glob = tuple(sorted(glob))
    order = {s: i for i, s in enumerate(sorted(set(vlab)))}
    col0 = [order[s] for s in vlab]
    best = [None, None]
    leaves = {}   # certificate -> leaf colouring (first seen), for automorphism discovery
    autos = []
    nleaves = [0]

    def cert_of(col):
        inv = [0] * n
        for v in range(n):
            inv[col[v]] = v
        return (glob, tuple(vlab[inv[c]] for c in range(n)),
                tuple(sorted((t, col[u], col[w]) for t, u, w in edges)))

    def search(col, prefix):
        col = _refine(col, n, adj)
        cells = defaultdict(list)
        for v in range(n):
            cells[col[v]].append(v)
        target = None
        for c in sorted(cells):
            if len(cells[c]) > 1:
                target = cells[c]
                break
        if target is None:
            nleaves[0] += 1
            if nleaves[0] > LEAF_CAP:
                raise RuntimeError("leaf cap")
            cert = cert_of(col)
            if cert in leaves:
                c1 = leaves[cert]
                pos = {c1[y]: y for y in range(n)}
                g = [pos[col[x]] for x in range(n)]   # automorphism: vertex x (leaf2) -> same colour in leaf1
                if any(g[i] != i for i in range(n)):
                    autos.append(g)
            elif len(leaves) < 5000:
                leaves[cert] = col
            if best[0] is None or cert < best[0]:
                best[0], best[1] = cert, col
            return
        explored = []
        for v in target:
            if explored:
                # orbit pruning: generators fixing the prefix pointwise
                par = list(range(n))

                def find(a):
                    while par[a] != a:
                        par[a] = par[par[a]]
                        a = par[a]
                    return a
                for g in autos:
                    if all(g[p] == p for p in prefix):
                        for i in range(n):
                            a, b = find(i), find(g[i])
                            if a != b:
                                par[a] = b
                rv = find(v)
                if any(find(e) == rv for e in explored):
                    continue
            sig = [(col[x], 0 if x == v else 1) for x in range(n)]
            o = {s: i for i, s in enumerate(sorted(set(sig)))}
            search([o[s] for s in sig], prefix + [v])
            explored.append(v)

    try:
        search(col0, [])
    except RuntimeError:
        STATS["cap_hits"] += 1
        raise
    STATS["max_leaves"] = max(STATS["max_leaves"], nleaves[0])
    lab = {verts[v]: best[1][v] for v in range(n)}
    return best[0], lab


def H(cert):
    return hashlib.sha1(repr(cert).encode()).digest()


def forms(objs, init, goal):
    """(full, init-only, goal-only) canonical certificates + labellings."""
    return (canon(objs, [("I", init), ("G", goal)]), canon(objs, [("I", init)]), canon(objs, [("G", goal)]))


def mapping(lab_t, lab_e):
    inv = {c: v for v, c in lab_e.items()}
    return {x: inv[c] for x, c in lab_t.items()}


def ren(fs, m):
    return frozenset((f[0],) + tuple(m[a] for a in f[1:]) for f in fs)


# ------------------------------------------------------------------ own simulator (plan replay check only)
def step(s, a):
    op = a[0]
    if op == "pick-up":
        x = a[1]
        if ("clear", x) in s and ("ontable", x) in s and ("handempty",) in s:
            return (s - {("clear", x), ("ontable", x), ("handempty",)}) | {("holding", x)}
    elif op == "put-down":
        x = a[1]
        if ("holding", x) in s:
            return (s - {("holding", x)}) | {("clear", x), ("handempty",), ("ontable", x)}
    elif op == "stack":
        x, y = a[1], a[2]
        if x != y and ("clear", y) in s and ("holding", x) in s:
            return (s - {("clear", y), ("holding", x)}) | {("handempty",), ("clear", x), ("on", x, y)}
    elif op == "unstack":
        x, y = a[1], a[2]
        if ("on", x, y) in s and ("clear", x) in s and ("handempty",) in s:
            return (s - {("on", x, y), ("clear", x), ("handempty",)}) | {("holding", x), ("clear", y)}
    return None


def replay(init, goal, plan):
    s = frozenset(init)
    for a in plan:
        s = step(s, a)
        if s is None:
            return False
    return goal <= s


# ------------------------------------------------------------------ raw parsers
def sexp(text):
    toks = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text.lower()))
    stack = [[]]
    for t in toks:
        if t == "(":
            stack.append([])
        elif t == ")":
            x = stack.pop()
            stack[-1].append(x)
        else:
            stack[-1].append(t)
    assert len(stack) == 1
    return stack[0]


def parse_pddl(path):
    tree = sexp(open(path, encoding="utf-8").read())
    prob = [x for x in tree if isinstance(x, list) and x and x[0] == "define"][0]
    objs = init = goal = None
    for sec in prob:
        if isinstance(sec, list) and sec and sec[0] == ":objects":
            objs = [t for t in sec[1:] if t != "-" and t != "object"]
        elif isinstance(sec, list) and sec and sec[0] == ":init":
            init = frozenset(tuple(f) for f in sec[1:])
        elif isinstance(sec, list) and sec and sec[0] == ":goal":
            g = sec[1]
            goal = frozenset(tuple(f) for f in g[1:]) if g[0] == "and" else frozenset([tuple(g)])
    for f in init | goal:
        assert all(isinstance(t, str) for t in f), (path, f)
    return sorted(objs), init, goal


def blocks_in(fs):
    return {a for f in fs for a in f[1:]}


def state_from(d):
    s = set()
    for x in d["clear"]:
        s.add(("clear", x))
    if d["hand"] == "empty":
        s.add(("handempty",))
    else:
        s.add(("holding", d["hand"]))
    for x, y in d["on"]:
        s.add(("on", x, y))
    for x in d["table"]:
        s.add(("ontable", x))
    return frozenset(s)


def user_json(prompt):
    u = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    return u, json.loads(u)


# ------------------------------------------------------------------ eval sets
def load_evals():
    B = ROOT + "/bench/planbench"
    sets = {}
    pb = []
    for sub in ("generated_basic_3", "generated_basic"):
        d = f"{B}/blocksworld/{sub}"
        fs = sorted((f for f in os.listdir(d) if f.endswith(".pddl")), key=lambda f: int(f.split("-")[1].split(".")[0]))
        for f in fs:
            o, i, g = parse_pddl(f"{d}/{f}")
            pb.append((f"{sub}/{f[:-5]}", o, i, g))
    sets["PB-602"] = pb
    d = f"{B}/blocksworld_hard/generated"
    sets["hard"] = [(f"hard/{f[:-5]}",) + parse_pddl(f"{d}/{f}") for f in
                    sorted((f for f in os.listdir(d) if f.endswith(".pddl")), key=lambda f: int(f.split("-")[1].split(".")[0]))]
    d = f"{B}/unsolvable_blocksworld/generated_basic"
    sets["unsolvable-101(extra)"] = [(f"unsolv/{f[:-5]}",) + parse_pddl(f"{d}/{f}") for f in
                                     sorted((f for f in os.listdir(d) if f.endswith(".pddl")),
                                            key=lambda f: int(f.split("-")[1].split(".")[0]))]

    def js(path, prefix):
        out = []
        for k, r in enumerate(json.load(open(ROOT + "/" + path, encoding="utf-8"))):
            out.append((f"{prefix}/{k}", sorted(r["objs"]), frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
        return out
    sets["mirror"] = js("runs/BW_val_large/instances_mirror_hard.json", "mirror")
    sets["b3fresh"] = js("runs/BW_val_large/instances_b3_fresh.json", "b3fresh")
    sets["val_large"] = js("runs/BW_val_large/instances_val_large.json", "val_large")
    sets["seeds_fresh"] = js("runs/BW_val_large/instances_seeds_fresh.json", "seeds_fresh")
    sets["val_v2(select)"] = js("runs/BW_data_v2/instances_val.json", "val_v2")
    sets["val_v1"] = js("runs/BW_data/instances_val.json", "val")
    dist = []
    for k, line in enumerate(open(ROOT + "/runs/BW_DISTILL_r1/solutions.jsonl", encoding="utf-8")):
        r = json.loads(line)
        dist.append((r["id"], sorted(r["objs"]), frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
    sets["distill-240"] = dist
    # sanity: objs == blocks of init, goal blocks subset
    for name, insts in sets.items():
        for iid, o, i, g in insts:
            assert set(o) == blocks_in(i), (name, iid)
            assert blocks_in(g) <= set(o), (name, iid)
    return sets


TRAIN = [
    # path, kind, used_by
    ("runs/BW_data_v2/proposer_actions_train.jsonl", "proposer", "BW_Pa_v2_s1,s2,s3 (and init adapter of Pa_v3_s1/Pa_v2c_s1)"),
    ("runs/BW_DISTILL_r1/distill_rows.jsonl", "proposer", "subset of mix_v3 (all 4254 rows kept, 0 dropped)"),
    ("runs/BW_DISTILL_r1/mix_v3.jsonl", "proposer", "BW_Pa_v3_s1 (B3/H1/H2)"),
    ("runs/BW_DISTILL_r1/mix_v2c.jsonl", "proposer", "BW_Pa_v2c_s1 (B3 control)"),
    ("runs/BW_data/forward_delta_train.jsonl", "forward", "BW_Fdelta_s0,s1,s2"),
    ("runs/BW_data/forward_delta_ko_train.jsonl", "forward", "BW_Fdelta_ko_s0 (FK, D-54)"),
    ("runs/BW_data/forward_train.jsonl", "forward", "BW_F_s0"),
    ("runs/BW_data/proposer_actions_train.jsonl", "proposer", "none found (v1 proposer data)"),
    ("runs/BW_data/proposer_states_train.jsonl", "proposer", "none found (v1 states-target data)"),
    ("runs/BW_data_v2/proposer_states_train.jsonl", "proposer", "none found (v2 states-target data)"),
    ("runs/_superseded/BW_data_v1_suffix_leak/proposer_actions_train.jsonl", "proposer", "none (superseded, never trained)"),
    ("runs/BW_B4_SCREEN/rcx_snapshot/runs/RCX-E07/main/data/instances.json", "instjson", "RCX-E07 GNN gnn_s20261276 (peer model used by the B4 screen); train+val splits only"),
]


def main():
    t0 = time.time()
    sets = load_evals()
    idx = {"full": defaultdict(list), "init": defaultdict(list), "goal": defaultdict(list)}
    einfo = {}
    for name, insts in sets.items():
        for iid, o, i, g in insts:
            F = forms(o, i, g)
            einfo[iid] = (name, o, i, g, F)
            for form, (c, lab) in zip(("full", "init", "goal"), F):
                idx[form][H(c)].append(iid)
    FI = {"full": 0, "init": 1, "goal": 2}

    def iso_ok(form, objs_a, s_a, g_a, m, objs_b, s_b, g_b):
        if len(objs_a) != len(objs_b) or sorted(m.values()) != sorted(objs_b) or sorted(m) != sorted(objs_a):
            return False
        if form in ("full", "init") and ren(s_a, m) != s_b:
            return False
        if form in ("full", "goal") and ren(g_a, m) != g_b:
            return False
        return True
    # verify every eval bucket explicitly (all members isomorphic to the first) -> transitivity for row checks
    cross = defaultdict(int)
    cross_ex = defaultdict(list)
    for form in idx:
        for h, mem in idx[form].items():
            n0, o0, i0, g0, F0 = einfo[mem[0]]
            for iid in mem[1:]:
                n1, o1, i1, g1, F1 = einfo[iid]
                assert F1[FI[form]][0] == F0[FI[form]][0]
                m = mapping(F1[FI[form]][1], F0[FI[form]][1])
                assert iso_ok(form, o1, i1, g1, m, o0, i0, g0), (form, mem[0], iid)
            if form == "full" and len(mem) > 1:
                for a in range(len(mem)):
                    for b in range(a + 1, len(mem)):
                        k = tuple(sorted((einfo[mem[a]][0], einfo[mem[b]][0])))
                        cross[k] += 1
                        if len(cross_ex[k]) < 10:
                            cross_ex[k].append((mem[a], mem[b]))
    print("eval sets loaded+verified", {k: len(v) for k, v in sets.items()}, round(time.time() - t0, 1), "s", flush=True)
    res = {"eval_sizes": {k: len(v) for k, v in sets.items()},
           "eval_block_counts": {k: sorted({len(x[1]) for x in v}) for k, v in sets.items()},
           "eval_goal_preds": {k: sorted({f[0] for x in v for f in x[3]}) for k, v in sets.items()},
           "eval_init_preds": {k: sorted({f[0] for x in v for f in x[2]}) for k, v in sets.items()},
           "eval_eval_full_duplicate_pairs": {"|".join(k): {"pairs": v, "first10": cross_ex[k]} for k, v in cross.items()},
           "files": {}}
    v2_line_of = {}

    def match_row(ln, s, goal, objs, plan, outcome, rk, todo, per):
        for form, (cert, lt) in todo:
            mem = idx[form].get(H(cert))
            if not mem:
                continue
            n0, o0, i0, g0, F0 = einfo[mem[0]]
            m0 = mapping(lt, F0[FI[form]][1])
            if F0[FI[form]][0] != cert or not iso_ok(form, objs, s, goal, m0, o0, i0, g0):
                per[("_any_", form + "_UNCONFIRMED")]["rows"] += 1
                continue
            hit_sets = set()
            for iid in mem:
                name, eo, ei, eg, FE = einfo[iid]
                P = per[(name, form)]
                if name not in hit_sets:
                    P["rows"] += 1
                    hit_sets.add(name)
                new_inst = iid not in P["inst"]
                P["inst"].add(iid)
                if form == "full":
                    m = mapping(lt, FE[0][1])
                    assert iso_ok("full", objs, s, goal, m, eo, ei, eg)   # explicit per-instance check
                    solves = None if plan is None else replay(ei, eg, [(a[0],) + tuple(m[x] for x in a[1:]) for a in plan])
                    P["solves"] = P.get("solves", 0) + int(bool(solves))
                    P["pairs"] = P.get("pairs", 0) + 1
                    kk = (rk.split(":")[-1] if rk.startswith("v2line") else rk.split(":")[0]) + "|outcome=" + str(outcome)
                    P.setdefault("by_rowkind", defaultdict(int))
                    P["by_rowkind"][kk] += 1
                    P.setdefault("inst_by_kind", defaultdict(set))
                    P["inst_by_kind"][kk].add(iid)
                    if len(P["ex"]) < 20:
                        P["ex"].append({"eval": iid, "line": ln, "row_kind": rk, "outcome": outcome,
                                        "plan_len": None if plan is None else len(plan), "target_plan_solves_eval": solves,
                                        "n_blocks": len(objs), "map_train_to_eval": dict(sorted(m.items()))})
                elif new_inst and len(P["ex"]) < 20:
                    P["ex"].append({"eval": iid, "line": ln, "row_kind": rk})

    def ikey(x):
        a, b = x.rsplit("/", 1)
        d = re.sub(r"\D", "", b)
        return (a, int(d) if d else 0, b)

    for path, kind, used in TRAIN:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            res["files"][path] = {"missing": True}
            continue
        tf = time.time()
        stat = {"used_by": used, "kind": kind, "rows": 0, "block_counts": defaultdict(int), "row_kinds": defaultdict(int),
                "skipped_unparsed": 0, "hand_holding_rows": 0}
        per = defaultdict(lambda: {"rows": 0, "inst": set(), "ex": []})
        if kind == "instjson":
            for k, x in enumerate(json.load(open(full, encoding="utf-8")), 1):
                if x["split"] not in ("train", "val"):
                    continue
                stat["rows"] += 1
                s = frozenset(map(tuple, x["init"]))
                goal = frozenset(map(tuple, x["goal"]))
                objs = sorted(x["objs"])
                assert set(objs) == blocks_in(s)
                stat["row_kinds"]["rcx-" + x["split"]] += 1
                stat["block_counts"][len(objs)] += 1
                F = forms(objs, s, goal)
                match_row(k, s, goal, objs, None, None, "rcx-" + x["split"] + ":" + x["id"],
                          (("full", F[0]), ("init", F[1]), ("goal", F[2])), per)
        else:
            grp0 = None
            with open(full, encoding="utf-8") as fh:
                for ln, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    stat["rows"] += 1
                    if "Blocksworld" not in r["prompt"]:
                        stat["skipped_unparsed"] += 1
                        continue
                    ukey, u = user_json(r["prompt"])
                    s = state_from(u["state"])
                    if ("handempty",) not in s:
                        stat["hand_holding_rows"] += 1
                    outcome = u.get("outcome")
                    if kind == "proposer":
                        goal = frozenset(("on", x, y) for x, y in u["goal"])
                        tgt = json.loads(r["target"])
                        plan = [tuple(a.split()) for a in tgt["actions"]] if "actions" in tgt else \
                            [tuple(st["action"].split()) for st in tgt["steps"]]
                        # row kind: per training instance the generator writes optimal suffix rows t=0..L-1 (some t>0
                        # rows may be dropped by the repo's leak guard), then one noisy-plan row from the init.
                        rk = None
                        if grp0 is not None and grp0["goal"] == goal:
                            if s == grp0["s0"]:
                                rk = ("noisy-" + str(outcome)) if plan != grp0["plan0"] else "repeat-of-t0"
                            else:
                                t = grp0["idx"].get(s)
                                if t is not None and t > 0 and plan == grp0["plan0"][t:]:
                                    rk = "suffix(t>0)"
                        if rk is None:
                            rk = "start(t=0)"
                            traj = {}
                            cur = s
                            for t, a in enumerate(plan):
                                traj.setdefault(cur, t)
                                cur = step(cur, a)
                                if cur is None:
                                    break
                            grp0 = {"goal": goal, "s0": s, "plan0": plan, "idx": traj}
                        hk = hashlib.sha1((r["prompt"] + r["target"]).encode()).digest()
                        if path.endswith("BW_data_v2/proposer_actions_train.jsonl"):
                            v2_line_of[hk] = (ln, rk)
                        if "mix_" in path:
                            origin = v2_line_of.get(hk)
                            rk = ("v2line%d:%s" % origin) if origin else "distill"
                            stat["row_kinds"]["v2-origin" if origin else "distill-origin"] += 1
                        elif "distill_rows" in path:
                            rk = "distill-suffix"
                            stat["row_kinds"][rk] += 1
                        else:
                            stat["row_kinds"][rk] += 1
                        objs = sorted(blocks_in(s) | blocks_in(goal))
                        F = forms(objs, s, goal)
                        todo = (("full", F[0]), ("init", F[1]), ("goal", F[2]))
                    else:
                        goal, plan, rk = frozenset(), None, "forward"
                        objs = sorted(blocks_in(s))
                        todo = (("init", canon(objs, [("I", s)])),)
                    stat["block_counts"][len(objs)] += 1
                    match_row(ln, s, goal, objs, plan, outcome, rk, todo, per)
                    if stat["rows"] % 25000 == 0:
                        print(path, stat["rows"], round(time.time() - tf, 1), "s", flush=True)
        out = {}
        for (name, form), P in sorted(per.items()):
            d = {"matching_rows": P["rows"], "matched_instances": len(P["inst"]),
                 "all_matched_instances": sorted(P["inst"], key=ikey), "examples_first20": P["ex"]}
            if "pairs" in P:
                d["row_instance_pairs"] = P["pairs"]
                d["pairs_where_mapped_target_plan_solves_eval"] = P["solves"]
                d["pairs_by_rowkind"] = dict(P["by_rowkind"])
                d["instances_by_rowkind"] = {k: len(v) for k, v in P["inst_by_kind"].items()}
            out.setdefault(name, {})[form] = d
        stat["block_counts"] = dict(sorted(stat["block_counts"].items()))
        stat["row_kinds"] = dict(stat["row_kinds"])
        stat["wall_s"] = round(time.time() - tf, 1)
        res["files"][path] = {"stat": stat, "matches": out}
        print("DONE", path, json.dumps(stat), flush=True)
        for name in out:
            print("   ", name, {f: (v["matched_instances"], v["matching_rows"]) for f, v in out[name].items()}, flush=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump({**res, "canon_stats": STATS}, fh, indent=1, default=list)
    res["canon_stats"] = STATS
    res["wall_s"] = round(time.time() - t0, 1)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=list)
    print("ALL DONE", STATS, res["wall_s"])


if __name__ == "__main__":
    main()
