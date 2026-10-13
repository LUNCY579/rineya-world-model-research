"""Read-only contamination audit (Method A). Streams every Blocksworld training-row file once; writes results only to
the scratchpad directory given as argv[1]."""
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
SP = Path(sys.argv[1])
sys.path.insert(0, str(SP))
from audit_common import B, ROOT, state_key, goal_key, find_iso, user_json, objs_of, valid_state  # noqa: E402

SIG_MAX = 7          # B.signature brute force used for n <= 7 (5040 perms)
SWEEP_MAX = 5        # independent full-signature sweep (no prefilter) for every row with n <= 5
SWEEP6_FRAC = 0.05   # ... and a 5% random sample of 6-block rows
FIRST = 20


# ----------------------------------------------------------------------------- eval sets (mirror blocks_eval.load_set)

def _json_set(path, prefix):
    return [{"id": f"{prefix}/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
             "goal": frozenset(map(tuple, r["goal"]))} for i, r in enumerate(json.loads(Path(path).read_text()))]


def load_eval_sets():
    S = {}
    pb = []
    for sub in ("generated_basic_3", "generated_basic"):
        for p in sorted((B.BENCH / "blocksworld" / sub).glob("*.pddl"), key=lambda p: int(p.stem.split("-")[1])):
            objs, init, goal = B.parse_problem(p)
            pb.append({"id": f"{sub}/{p.stem}", "objs": objs, "init": init, "goal": goal})
    S["planbench_PB602"] = pb
    hard = []
    for p in sorted((B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"), key=lambda p: int(p.stem.split("-")[1])):
        objs, init, goal = B.parse_problem(p)
        hard.append({"id": f"hard/{p.stem}", "objs": objs, "init": init, "goal": goal})
    S["hard"] = hard
    S["mirror"] = _json_set(ROOT / "runs/BW_val_large/instances_mirror_hard.json", "mirror")
    S["b3fresh"] = _json_set(ROOT / "runs/BW_val_large/instances_b3_fresh.json", "b3fresh")
    S["val_large"] = _json_set(ROOT / "runs/BW_val_large/instances_val_large.json", "val_large")
    S["seeds_fresh"] = _json_set(ROOT / "runs/BW_val_large/instances_seeds_fresh.json", "seeds_fresh")
    S["val_v2"] = _json_set(ROOT / "runs/BW_data_v2/instances_val.json", "val_v2")
    S["val_v1"] = _json_set(ROOT / "runs/BW_data/instances_val.json", "val")
    d = []
    for line in open(ROOT / "runs/BW_DISTILL_r1/solutions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        d.append({"id": r["id"], "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
                  "goal": frozenset(map(tuple, r["goal"])), "verified": r["verified"]})
    S["distill240"] = d
    return S


TRAIN = [
    # name, path, kind, used_by
    ("forward_train", "runs/BW_data/forward_train.jsonl", "forward", "BW_F_s0 (state-format F; BW_SELECT_v1 fwd-state)"),
    ("forward_delta_train", "runs/BW_data/forward_delta_train.jsonl", "forward", "BW_Fdelta_s0/s1/s2 (frozen B2 F = s0; item-7 R0-R2)"),
    ("forward_delta_ko_train", "runs/BW_data/forward_delta_ko_train.jsonl", "forward", "BW_Fdelta_ko_s0 (D-54 arm FK)"),
    ("v2_proposer_actions_train", "runs/BW_data_v2/proposer_actions_train.jsonl", "proposer",
     "BW_Pa_v2_s1/s2/s3 (B2 Pa = s1; init adapter of Pa_v2c/Pa_v3)"),
    ("distill_rows", "runs/BW_DISTILL_r1/distill_rows.jsonl", "proposer", "first half of mix_v3 (Pa_v3)"),
    ("mix_v3", "runs/BW_DISTILL_r1/mix_v3.jsonl", "proposer", "BW_Pa_v3_s1 (B3 / H1 / H2)"),
    ("mix_v2c", "runs/BW_DISTILL_r1/mix_v2c.jsonl", "proposer", "BW_Pa_v2c_s1 (B3 control)"),
    ("v1_proposer_actions_train", "runs/BW_data/proposer_actions_train.jsonl", "proposer",
     "v1 proposer data (bw-train-Pa per D-32/B-02); no v1 Pa adapter present in runs/"),
    ("superseded_v1_proposer_actions_train", "runs/_superseded/BW_data_v1_suffix_leak/proposer_actions_train.jsonl",
     "proposer", "superseded leaky v1 data; DECISIONS B-02: never trained on"),
]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    rng = random.Random(7)
    S = load_eval_sets()
    report = {"eval_sets": {}, "train_files": {}, "pairs": {}, "checks": Counter()}
    EV_K, EV_SIG = defaultdict(list), defaultdict(list)
    for sname, insts in S.items():
        bad = 0
        for e in insts:
            objs = e["objs"]
            factblocks = set(B.blocks_of(e["init"])) | {x for f in e["goal"] for x in f[1:]}
            e["sk"] = state_key(e["init"], objs)
            e["gk"] = goal_key(e["goal"], objs)
            if e["sk"] is None or e["gk"] is None or factblocks != set(objs):
                bad += 1
            e["sig"] = B.signature(objs, e["init"], e["goal"]) if len(objs) <= SIG_MAX else None
            EV_K[(e["sk"], e["gk"])].append(e)
            if e["sig"]:
                EV_SIG[e["sig"]].append(e)
        report["eval_sets"][sname] = {"n": len(insts), "invalid_or_nonstandard": bad,
                                      "blocks": dict(sorted(Counter(len(e["objs"]) for e in insts).items()))}
        print(sname, report["eval_sets"][sname], flush=True)

    # source maps so mix rows can be traced to v2 / distill lines (and inherit 'problem-initial' flags)
    src_hash = {}

    for fname, rel, kind, used_by in TRAIN:
        path = ROOT / rel
        t1 = time.time()
        nb = Counter()
        cnt_sk, cnt_gk, cnt_sk_p, cnt_gk_p = Counter(), Counter(), Counter(), Counter()
        ex_sk, ex_gk = defaultdict(list), defaultdict(list)
        invalid_state = invalid_goal = 0
        full = defaultdict(list)       # eval id -> list of match dicts
        sig_cache = {}
        prev_goal, first_state = None, None
        n_rows = n_problems = 0
        sweep_rows = sweep_hits = sweep_miss = 0
        disagree = 0
        for ln, line in enumerate(open(path, encoding="utf-8"), start=1):
            r = json.loads(line)
            d = user_json(r["prompt"])
            n_rows += 1
            state = B.state_from_json(d["state"])
            if kind == "forward":
                objs = B.blocks_of(state)
                goal = None
                sk = state_key(state, objs)
                nb[len(objs)] += 1
                if sk is None:
                    invalid_state += 1
                    continue
                cnt_sk[sk] += 1
                if len(ex_sk[sk]) < FIRST:
                    ex_sk[sk].append(ln)
                continue
            gj = json.dumps(d["goal"])
            goal = frozenset(("on", x, y) for x, y in d["goal"])
            objs = objs_of(state, goal)
            n = len(objs)
            nb[n] += 1
            hkey = hashlib.sha1((r["prompt"] + "\x00" + r["target"]).encode()).digest()
            if fname in ("v2_proposer_actions_train", "distill_rows"):
                new_group = gj != prev_goal
                if new_group:
                    prev_goal, first_state = gj, state
                    n_problems += 1
                init_row = state == first_state
                src_hash[hkey] = (fname, ln, init_row, new_group)
                src = None
            elif fname.startswith("mix_"):
                src = src_hash.get(hkey)
                init_row = src[2] if src else None
                new_group = src[3] if src else False
            else:
                new_group = gj != prev_goal
                if new_group:
                    prev_goal, first_state = gj, state
                    n_problems += 1
                init_row = state == first_state
                src = None
            sk = state_key(state, objs)
            gk = goal_key(goal, objs)
            if sk is None:
                invalid_state += 1
            else:
                cnt_sk[sk] += 1
                if len(ex_sk[sk]) < FIRST:
                    ex_sk[sk].append(ln)
            if gk is None:
                invalid_goal += 1
            else:
                cnt_gk[gk] += 1
                if len(ex_gk[gk]) < FIRST:
                    ex_gk[gk].append(ln)
            if new_group:
                if sk is not None:
                    cnt_sk_p[sk] += 1
                if gk is not None:
                    cnt_gk_p[gk] += 1
            ukey = json.dumps([d["state"], d["goal"]], sort_keys=True)

            def rowsig():
                if ukey not in sig_cache:
                    sig_cache[ukey] = B.signature(objs, state, goal)
                return sig_cache[ukey]

            hits = set()
            if sk is not None and gk is not None and (sk, gk) in EV_K:
                for e in EV_K[(sk, gk)]:
                    m = find_iso((e["objs"], e["init"], e["goal"]), (objs, state, goal))
                    if n <= SIG_MAX:
                        s_eq = rowsig() == e["sig"]
                        report["checks"]["sig_vs_iso_candidates"] += 1
                        if s_eq != (m is not None):
                            disagree += 1
                    if m is not None:
                        hits.add(e["id"])
                        if len(full[e["id"]]) < 1000:
                            try:
                                tl = len(json.loads(r["target"]).get("actions", []))
                            except Exception:
                                tl = None
                            full[e["id"]].append({"line": ln, "problem_initial_row": init_row, "outcome": d.get("outcome"),
                                                  "target_len": tl, "src": src[:2] if src else None,
                                                  "map_eval_to_row": m if len(full[e["id"]]) < 3 else None,
                                                  "sig_confirmed": (rowsig() == e["sig"]) if n <= SIG_MAX else "n>7: iso only"})
            # independent sweep without the prefilter: full signature for small rows
            if n <= SWEEP_MAX or (n == 6 and rng.random() < SWEEP6_FRAC):
                sweep_rows += 1
                for e in EV_SIG.get(rowsig(), []):
                    sweep_hits += 1
                    if e["id"] not in hits:
                        sweep_miss += 1
        # ------------------------------------------------------------------ per (eval set x file) summary
        finfo = {"path": rel, "kind": kind, "used_by": used_by, "rows": n_rows, "problems(groups)": n_problems,
                 "blocks": dict(sorted(nb.items())), "invalid_states": invalid_state, "invalid_goals": invalid_goal,
                 "sha256_prefix": sha(path)[:16], "sig_vs_iso_disagreements": disagree,
                 "sweep_rows_full_signature": sweep_rows, "sweep_sig_hits": sweep_hits,
                 "sweep_hits_missed_by_prefilter": sweep_miss, "wall_s": round(time.time() - t1, 1)}
        report["train_files"][fname] = finfo
        print(fname, json.dumps(finfo), flush=True)
        for sname, insts in S.items():
            P = {}
            if kind == "proposer":
                fl = [(e["id"], mm) for e in insts for mm in full.get(e["id"], [])]
                P["full_contaminated_instances"] = sum(1 for e in insts if full.get(e["id"]))
                P["full_matching_rows"] = len(fl)
                P["full_contaminated_instances_problem_level"] = sum(
                    1 for e in insts if any(mm["problem_initial_row"] for mm in full.get(e["id"], [])))
                P["full_first20"] = [dict(eval_id=i, **{k: v for k, v in mm.items() if v is not None}) for i, mm in fl[:FIRST]]
                P["full_ids"] = [e["id"] for e in insts if full.get(e["id"])][:200]
                P["goal_only_instances"] = sum(1 for e in insts if cnt_gk.get(e["gk"]))
                P["goal_only_matching_rows"] = sum(cnt_gk.get(e["gk"], 0) for e in insts)
                P["goal_only_instances_problem_level"] = sum(1 for e in insts if cnt_gk_p.get(e["gk"]))
                P["goal_only_first20"] = [(e["id"], l) for e in insts for l in ex_gk.get(e["gk"], [])[:1]][:FIRST]
                P["init_only_instances"] = sum(1 for e in insts if cnt_sk.get(e["sk"]))
                P["init_only_matching_rows"] = sum(cnt_sk.get(e["sk"], 0) for e in insts)
                P["init_only_instances_problem_level"] = sum(1 for e in insts if cnt_sk_p.get(e["sk"]))
                P["init_only_first20"] = [(e["id"], l) for e in insts for l in ex_sk.get(e["sk"], [])[:1]][:FIRST]
            else:
                P["F_state_overlap_instances"] = sum(1 for e in insts if cnt_sk.get(e["sk"]))
                P["F_state_overlap_rows"] = sum(cnt_sk.get(e["sk"], 0) for e in insts)
                P["F_state_overlap_first20"] = [(e["id"], l) for e in insts for l in ex_sk.get(e["sk"], [])[:1]][:FIRST]
            report["pairs"][f"{sname} x {fname}"] = P
            if kind == "proposer":
                print(f"  {sname:16s} full {P['full_contaminated_instances']}/{len(insts)} (rows {P['full_matching_rows']}, "
                      f"problem-level {P['full_contaminated_instances_problem_level']}) | init-only {P['init_only_instances']} "
                      f"| goal-only {P['goal_only_instances']}", flush=True)
            else:
                print(f"  {sname:16s} F-state overlap {P['F_state_overlap_instances']}/{len(insts)} rows {P['F_state_overlap_rows']}",
                      flush=True)
        del sig_cache
    report["checks"] = dict(report["checks"])
    report["wall_s"] = round(time.time() - t0, 1)
    (SP / "audit_results.json").write_text(json.dumps(report, indent=1, default=str))
    print("total wall", report["wall_s"])


if __name__ == "__main__":
    main()
