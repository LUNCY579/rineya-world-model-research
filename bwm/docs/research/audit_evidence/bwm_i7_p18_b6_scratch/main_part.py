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
    for path, kind, used in TRAIN:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            res["files"][path] = {"missing": True}
            continue
        tf = time.time()
        stat = {"used_by": used, "kind": kind, "rows": 0, "block_counts": defaultdict(int), "row_kinds": defaultdict(int),
                "skipped_unparsed": 0, "hand_holding_rows": 0}
        per = defaultdict(lambda: {"rows": 0, "inst": set(), "ex": []})
        prev = None
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
                    rk = "start(t=0)"
                    if prev is not None and prev[1] == goal and prev[2] and prev[2][1:] == plan and step(prev[0], prev[2][0]) == s:
                        rk = "suffix(t>0)"
                    elif grp0 is not None and grp0 == (s, goal):
                        rk = "noisy-" + str(outcome)
                    if rk == "start(t=0)":
                        grp0 = (s, goal)
                    prev = (s, goal, plan)
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
                            solves = replay(ei, eg, [(a[0],) + tuple(m[x] for x in a[1:]) for a in plan])
                            P["solves"] = P.get("solves", 0) + int(solves)
                            P.setdefault("by_rowkind", defaultdict(int))
                            P["by_rowkind"][(rk.split(":")[-1] if rk.startswith("v2line") else rk) + "|outcome=" + str(outcome)] += 1
                            if len(P["ex"]) < 20:
                                P["ex"].append({"eval": iid, "line": ln, "row_kind": rk, "outcome": outcome,
                                                "plan_len": len(plan), "target_plan_solves_eval": solves,
                                                "n_blocks": len(objs), "map_train_to_eval": dict(sorted(m.items()))})
                        elif new_inst and len(P["ex"]) < 20:
                            P["ex"].append({"eval": iid, "line": ln, "row_kind": rk})
                if stat["rows"] % 25000 == 0:
                    print(path, stat["rows"], round(time.time() - tf, 1), "s", flush=True)
        out = {}

        def ikey(x):
            a, b = x.rsplit("/", 1)
            d = re.sub(r"\D", "", b)
            return (a, int(d) if d else 0, b)
        for (name, form), P in sorted(per.items()):
            d = {"matching_rows": P["rows"], "matched_instances": len(P["inst"]),
                 "instances_first20": sorted(P["inst"], key=ikey)[:20], "examples_first20": P["ex"]}
            if "solves" in P:
                d["rows_whose_target_plan_solves_eval"] = P["solves"]
                d["by_rowkind"] = dict(P["by_rowkind"])
                d["all_matched_instances"] = sorted(P["inst"], key=ikey)
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


