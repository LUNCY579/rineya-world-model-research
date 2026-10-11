"""RCX-E07b runner (PREREG_E07B.md, frozen; CX D-20): can the learned relational value keep its low-budget GBFS
advantage *and* close the high-budget coverage gap?

Tier B external specialist, exact dynamics, CPU mechanism tier; no Rineya-core claim. The R-04 run (runs/RCX-E07/main:
A0 models, train/val rows, E07 test states, E08 instances) is read strictly read-only; outputs go only to --out
(the full run must use runs/RCX-E07/e07b, as the prereg fixes).

Usage (from complementary/; CPU only, <= 4 threads, ThermalGuard on):
  python -B experiments/RCX-E07/run_e07b.py --smoke --threads 2   # plumbing check -> runs/RCX-E07/e07b_smoke
  python -B experiments/RCX-E07/run_e07b.py --threads 4           # full prereg run -> runs/RCX-E07/e07b
Phases (--phase, default all, in order): data_b, train_b, select, e08b, policy, report. Finished work is skipped
(phase outputs, models per seed, GBFS runs per (instance, heuristic)), so re-running the same command resumes. The
config is frozen in <out>/config.json and the sha256 of every R-04 input in <out>/inputs.json; a mismatch aborts. The
prereg's sha256 is verified at every start (abort if PREREG_E07B.md changed). R-04's runner is imported and reused
unchanged (run_e07.py is not modified).
See experiments/RCX-E07/README.md, section "RCX-E07b", for the design as implemented and the resolved ambiguities.
"""
import argparse
import contextlib
import copy
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported (never '-1')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def _import_run_e07():
    """R-04's runner as a module (its main() does not run; nothing in it is modified)."""
    spec = importlib.util.spec_from_file_location("rcx_run_e07", HERE / "run_e07.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _import_run_e07()  # imports torch CPU-only (CUDA hidden, torch.cuda patched) and rcx.*

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rcx import stats as ST  # noqa: E402
from rcx.bw import bwm_import as BI  # noqa: E402
from rcx.bw import data as D  # noqa: E402
from rcx.bw import encode as EN  # noqa: E402
from rcx.bw import excl as EX  # noqa: E402
from rcx.bw import heuristics as H  # noqa: E402
from rcx.bw import models as M  # noqa: E402
from rcx.bw import search as S  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

PREREG_PATH = HERE / "PREREG_E07B.md"
PREREG_SHA256 = "8a4cbde047f17eeb415ad9353459cc17f99e4f81c3a6008ba090db074af29cfb"  # DECISIONS CX D-20
EXCLUSIONS_SHA_PREFIX = "2f211de3"  # PREREG_E07B 'Instances': EXCLUSIONS.json at sha 2f211de3...
R04_DIR = ROOT / "runs" / "RCX-E07" / "main"
OUT_FULL = ROOT / "runs" / "RCX-E07" / "e07b"
OUT_SMOKE = ROOT / "runs" / "RCX-E07" / "e07b_smoke"
SEED_BLOCK = (20261290, 20261299)  # reserved for E07b (PREREG_E07B 'Seeds')
SEEDS_B = {"testb_instances": 20261290, "valb_instances": 20261291, "a1_models": [20261292, 20261293, 20261294],
           "a2_models": [20261295, 20261296, 20261297], "spare_a1_n8": 20261298, "bootstrap": 20261299}
A1_DISCOVERY_SEED = f"{SEEDS_B['spare_a1_n8']}:discovery"  # derived from the spare seed, as R-04 derived its
A1_STATES_SEED = f"{SEEDS_B['spare_a1_n8']}:states"        # bootstrap seed from the split seed
A0_SEEDS = [20261276, 20261277, 20261278]  # the R-04 GNN models
BASELINES = ("misplaced_blocks", "h_add", "unmet_goal_facts")
ARM_ORDER = ("A1", "A2", "A3")  # prereg listing order = last tie-break for BEST
PHASES = ("data_b", "train_b", "select", "e08b", "policy", "report")
CODE_FILES = ["experiments/RCX-E07/run_e07b.py"] + R.CODE_FILES
R04_FILES = ["config.json", "data/meta.json", "data/instances.json", "data/train.jsonl.gz", "data/val.jsonl.gz",
             "data/test.jsonl.gz", "e07_scores.jsonl.gz", "e08_instances.json"] + \
    [f"models/gnn_s{s}.pt" for s in A0_SEEDS]
TRAIN_KEYS = ("epochs", "batch_size", "lr", "time_cap_s")  # must equal R-04's (prereg: same epochs and time cap)

FULL = dict(
    testb_sizes=[10, 12, 14, 15], testb_per_size=25, valb_sizes=[12, 14], valb_per_size=15, primary_sizes=[12, 14, 15],
    a1_n=8, a1_classes=10, a1_states=10000, a1_discovery_draws=50000, a1_min_class_count=5, a1_max_draws=50000,
    r04_train_stride=1, a0_seeds=A0_SEEDS, a1_seeds=SEEDS_B["a1_models"], a2_seeds=SEEDS_B["a2_models"],
    epochs=12, batch_size=64, lr=2e-3, time_cap_s=3600.0, rank_margin=0.5, rank_weight=1.0,
    lambdas=[0.25, 0.5, 1.0], budgets=[50, 200, 500, 2000], b_primary=2000, b_low=200, opt_budget=300_000,
    policy_sizes=[8, 10, 12], policy_inst_per_size=None, tie_tol=1e-5,
    closed_margin=3, improved_delta=8, alpha=0.05, bootstrap_resamples=5000, mono_check=[3, 2],
)
SMOKE = dict(FULL, testb_per_size=2, valb_per_size=2, a1_classes=2, a1_states=300, a1_discovery_draws=3000,
             r04_train_stride=37, a0_seeds=A0_SEEDS[:2], a1_seeds=SEEDS_B["a1_models"][:2],
             a2_seeds=SEEDS_B["a2_models"][:2], epochs=1, time_cap_s=30.0, budgets=[20, 50, 100, 300],
             b_primary=300, b_low=50, opt_budget=20_000, policy_inst_per_size=2, bootstrap_resamples=200,
             mono_check=[2, 1])


# ----------------------------------------------------------------------------- small utilities

def verify_prereg():
    got = R.sha_file(PREREG_PATH)
    if got != PREREG_SHA256:
        raise SystemExit(f"PREREG_E07B.md sha256 {got} != frozen {PREREG_SHA256}: the prereg changed; refusing to run")
    return got


@contextlib.contextmanager
def e07b_seed_block():
    """data.fresh_instances guards R-04's seed range 20261270-89. E07b draws use their own reserved block
    20261290-99, so the guard is retargeted to that block for the duration of the draw (data.py is unchanged)."""
    old = D.SEED_RANGE
    D.SEED_RANGE = SEED_BLOCK
    try:
        yield
    finally:
        D.SEED_RANGE = old


def save_instances(path, inst):
    R.write_json(path, [dict({k: v for k, v in it.items() if k not in ("init", "goal")},
                             init=R.fs_json(it["init"]), goal=R.fs_json(it["goal"])) for it in inst])


def load_instance_list(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    for it in raw:
        it["init"], it["goal"] = R.fs_from(it["init"]), R.fs_from(it["goal"])
    return raw


def inst_key(it):
    return D.instance_key(it["objs"], it["init"], it["goal"])


def mean_sd(vals):
    vals = [v for v in vals if v is not None]
    return {"mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None, "per_seed": vals}


def load_gnn(path):
    model = R.new_model("gnn")
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def model_path(ctx, arm, seed):
    if arm == "A0":
        return R04_DIR / "models" / f"gnn_s{seed}.pt"
    return ctx.out / "models" / f"{arm}_s{seed}.pt"


def a3_name(lam):
    return f"A3_l{lam:g}"


def arm_names(cfg, arm):
    return [f"{arm}_s{s}" for s in cfg[f"{arm.lower()}_seeds"]]


class HybridH:
    """A3 key h(s) = GNN(s) + lam * misplaced_blocks(s) (PREREG_E07B H-b3); batched like models.LearnedH."""

    def __init__(self, model, lam):
        self.gnn, self.lam = M.LearnedH(model), float(lam)

    def batch(self, states, goal):
        v = self.gnn.batch(states, goal)
        return [x + self.lam * H.misplaced_blocks(s, goal) for x, s in zip(v, states)]

    def __call__(self, s, goal):
        return self.batch([s], goal)[0]


def heuristic_factory(ctx):
    def make(name):
        if name in H.BASELINES:
            return H.BASELINES[name]
        if name.startswith("A3_l"):
            return HybridH(load_gnn(model_path(ctx, "A0", ctx.cfg["a0_seeds"][0])), float(name[4:]))
        arm, seed = name.split("_s")
        return M.LearnedH(load_gnn(model_path(ctx, arm, int(seed))))
    return make


# ----------------------------------------------------------------------------- GBFS runs and budget derivation

RUN_KEYS = ("success", "found", "plan_len", "expansions", "generated", "h_calls", "reason", "seconds")


def gbfs_record(it, name, h, budget):
    r = S.gbfs(it["objs"], it["init"], it["goal"], h, max_expansions=budget)
    rec = {"inst": it["id"], "n": it["n"], "heur": name, "budget": budget, "opt": it.get("opt"),
           **{k: r[k] for k in RUN_KEYS}, "plan": [" ".join(a) for a in r["plan"]]}
    rec["subopt"] = (rec["plan_len"] / rec["opt"]) if rec["success"] and rec["opt"] else None
    return rec


def at_budget(rec, b):
    """GBFS outcome at budget b <= rec['budget'], derived from the larger-budget run.

    Valid because search.gbfs is budget-monotone: the budget is only compared (exp >= max_expansions) before each pop,
    the goal test is applied at generation, and neither the pop order nor the heuristic calls (one batch per expansion)
    depend on max_expansions. So the b-run is a prefix of the B-run: identical if the B-run ended ('goal' or
    'exhausted') after e <= b expansions, else it stops with reason 'budget' after exactly b expansions. Checked
    empirically on every e08b run (monotonicity_check)."""
    if b > rec["budget"]:
        raise ValueError(f"cannot derive budget {b} from a {rec['budget']}-budget run")
    if b == rec["budget"]:
        return rec
    if rec["reason"] in ("goal", "exhausted") and rec["expansions"] <= b:
        return dict(rec, budget=b, derived_from=rec["budget"])
    return {"inst": rec["inst"], "n": rec["n"], "heur": rec["heur"], "budget": b, "opt": rec["opt"], "success": False,
            "found": False, "plan_len": None, "expansions": b, "generated": None, "h_calls": None, "reason": "budget",
            "seconds": None, "plan": [], "subopt": None, "derived_from": rec["budget"]}


def read_runs(path, budget):
    """Resume a runs .jsonl; a torn last line (crash mid-write) is dropped and the file rewritten without it."""
    done = {}
    if not path.exists():
        return done
    lines = path.read_text(encoding="utf-8").splitlines()
    good = []
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if k == len(lines) - 1:
                path.write_text("".join(x + "\n" for x in good), encoding="utf-8", newline="\n")
                break
            raise
        if r["budget"] != budget:
            raise SystemExit(f"{path}: run with budget {r['budget']} != {budget}; use a new --out")
        good.append(line)
        done[(r["inst"], r["heur"])] = r
    return done


def run_set(ctx, inst, names, path, budget, factory):
    """GBFS for every (instance, heuristic) at `budget`, resumable per run (instance-major order)."""
    done = read_runs(path, budget)
    todo = [(it, nm) for it in inst for nm in names if (it["id"], nm) not in done]
    ctx.log(f"{path.name}: {len(todo)} GBFS runs to do, {len(done)} already done")
    cache, t0 = {}, time.perf_counter()
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for k, (it, nm) in enumerate(todo):
            if nm not in cache:
                cache[nm] = factory(nm)
            rec = gbfs_record(it, nm, cache[nm], budget)
            done[(it["id"], nm)] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ctx.courtesy()
            if (k + 1) % 100 == 0:
                ctx.log(f"{path.name}: {k + 1}/{len(todo)} runs in {time.perf_counter() - t0:.0f}s")
    return done


def succ_count(done, ids, name, b):
    return sum(bool(at_budget(done[(i, name)], b)["success"]) for i in ids)


def mcnemar(done, ids, base, cand, b):
    sb = {i: bool(at_budget(done[(i, base)], b)["success"]) for i in ids}
    sc = {i: bool(at_budget(done[(i, cand)], b)["success"]) for i in ids}
    bo = sum(sb[i] and not sc[i] for i in ids)
    co = sum(sc[i] and not sb[i] for i in ids)
    return {"base": base, "cand": cand, "budget": b, "n": len(ids), "base_success": sum(sb.values()),
            "cand_success": sum(sc.values()), "base_only": bo, "cand_only": co, "p_exact": ST.mcnemar_exact(bo, co)}


def cell(done, ids, name, b):
    recs = [at_budget(done[(i, name)], b) for i in ids]
    k = sum(bool(r["success"]) for r in recs)
    succ = [r for r in recs if r["success"]]
    sub = [r["subopt"] for r in succ if r["subopt"] is not None]
    exps = [r["expansions"] for r in recs]
    return {"n": len(recs), "success": k, "rate": k / len(recs) if recs else None, "wilson": ST.wilson(k, len(recs)),
            "expansions_median": float(np.median(exps)) if exps else None,
            "expansions_mean": float(np.mean(exps)) if exps else None,
            "plan_len_mean": float(np.mean([r["plan_len"] for r in succ])) if succ else None,
            "subopt_mean": float(np.mean(sub)) if sub else None, "subopt_n": len(sub),
            "invalid_found_plans": sum(1 for r in recs if r["found"] and not r["success"]),
            "seconds_total": float(sum(r["seconds"] for r in recs if r["seconds"] is not None and
                                       "derived_from" not in r))}


# ----------------------------------------------------------------------------- phase: data_b

def a1_discovery(ctx, B, VL, ex):
    """Goal-class availability at n = a1_n: discovery draws of B.generate(full_goal_frac=1.0) with the derived seed
    '20261298:discovery'; generate()=None and the BWM invariant filter applied, as R-04's allocate_classes does."""
    cfg = ctx.cfg
    rng = random.Random(A1_DISCOVERY_SEED)
    cnt, st = Counter(), Counter()
    for _ in range(cfg["a1_discovery_draws"]):
        g = B.generate(rng, cfg["a1_n"], full_goal_frac=1.0)
        if g is None:
            st["generate_none"] += 1
            continue
        if VL.invariant(*g) in ex["invariants"]:
            st["excluded_invariant"] += 1
            continue
        cnt[D.goal_key(g[0], g[2])] += 1
        ctx.courtesy()
    st["counted"] = sum(cnt.values())
    return dict(sorted(cnt.items())), dict(st)


def phase_data_b(ctx):
    cfg, out = ctx.cfg, ctx.out
    dd = out / "data"
    if (dd / "meta.json").exists():
        ctx.log("data_b: outputs exist, skipping")
        return
    dd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    B, VL = BI.load()
    ex = EX.with_selection_sets(EX.load_exclusions())  # the same exclusion machinery as run_e07's data phase
    excl_sha = R.sha_file(EX.EXCLUSIONS_PATH)
    r04_meta = json.loads((R04_DIR / "data" / "meta.json").read_text(encoding="utf-8"))
    if excl_sha != r04_meta["exclusions_sha256"] or not excl_sha.startswith(EXCLUSIONS_SHA_PREFIX):
        raise SystemExit(f"EXCLUSIONS.json sha {excl_sha} is not the prereg's {EXCLUSIONS_SHA_PREFIX}... / R-04's")
    inst04 = R.load_instances(R04_DIR)
    e08_04 = load_instance_list(R04_DIR / "e08_instances.json")
    e08_keys = {inst_key(it) for it in e08_04}
    e07_keys = {inst_key(it) for it in inst04.values() if it["split"] == "test" and it["n"] > 7}
    trval_classes = {it["goal_key"] for it in inst04.values() if it["split"] in ("train", "val")}

    # TEST-B then VAL-B (seed order); one `seen` set seeded with R-04's E08 and E07-OOD-test instance keys
    gen, seen = {}, set(e08_keys) | e07_keys
    with e07b_seed_block():
        testb, gen["testb"] = D.fresh_instances(
            SEEDS_B["testb_instances"], [n for n in cfg["testb_sizes"] for _ in range(cfg["testb_per_size"])],
            full_goal_frac=1.0, seen=seen, ex=ex, prefix="testb")
        valb, gen["valb"] = D.fresh_instances(
            SEEDS_B["valb_instances"], [n for n in cfg["valb_sizes"] for _ in range(cfg["valb_per_size"])],
            full_goal_frac=1.0, seen=seen, ex=ex, prefix="valb")
    kt, kv = {inst_key(it) for it in testb}, {inst_key(it) for it in valb}
    checks = {"testb_unique": len(kt) == len(testb), "valb_unique": len(kv) == len(valb),
              "testb_x_r04_e08": len(kt & e08_keys), "valb_x_r04_e08": len(kv & e08_keys),
              "testb_x_valb": len(kt & kv), "testb_valb_x_r04_e07_test": len((kt | kv) & e07_keys),
              "goal_classes_x_r04_train_val": len({it["goal_key"] for it in testb + valb} & trval_classes),
              "goals_all_on_facts": all(f[0] == "on" for it in testb + valb for f in it["goal"])}
    if not (checks["testb_unique"] and checks["valb_unique"] and checks["goals_all_on_facts"]) or any(
            checks[k] for k in ("testb_x_r04_e08", "valb_x_r04_e08", "testb_x_valb", "testb_valb_x_r04_e07_test",
                                "goal_classes_x_r04_train_val")):
        raise RuntimeError(f"TEST-B/VAL-B disjointness check failed: {checks}")
    ctx.log(f"data_b: TEST-B {len(testb)} VAL-B {len(valb)}; generation {gen}; checks {checks}")
    for it in testb:  # optimal length for plan suboptimality (as R-04's E08)
        t1 = time.perf_counter()
        h, exp = SP.astar_hstar(it["objs"], it["init"], it["goal"], cfg["opt_budget"])
        it.update(opt=h, opt_expansions=exp, opt_s=round(time.perf_counter() - t1, 4))
        ctx.courtesy()
    ctx.log(f"data_b: TEST-B optimal lengths in {sum(it['opt_s'] for it in testb):.1f}s "
            f"({sum(it['opt'] is None for it in testb)} over the A* budget)")

    # A1: n = 8, full goals, the a1_classes most available fresh goal classes, one instance per class
    n8 = cfg["a1_n"]
    disc, disc_st = a1_discovery(ctx, B, VL, ex)
    ref = sorted({it["goal_key"] for it in inst04.values() if it["n"] == n8})  # R-04 E07 test classes at n = 8
    e08_n8 = sorted({it["goal_key"] for it in e08_04 if it["n"] == n8})
    fresh = sorted((k for k, v in disc.items() if k not in ref and v >= cfg["a1_min_class_count"]),
                   key=lambda k: (-disc[k], k))
    if len(fresh) < cfg["a1_classes"]:
        raise SystemExit(f"A1: only {len(fresh)} fresh n={n8} goal classes available ({fresh}); the prereg asks for "
                         f"{cfg['a1_classes']} -- stopping instead of deviating")
    chosen = fresh[:cfg["a1_classes"]]
    with e07b_seed_block():
        a1, gen["a1"] = D.fresh_instances(SEEDS_B["spare_a1_n8"], [n8] * len(chosen), full_goal_frac=1.0, seen=set(),
                                          ex=ex, prefix="a1n8", unique="goal", accept=set(chosen).__contains__,
                                          max_draws_per_item=cfg["a1_max_draws"], strict=True)
    ka = {inst_key(it) for it in a1}
    a1_checks = {"classes_as_chosen": sorted(it["goal_key"] for it in a1) == sorted(chosen),
                 "x_r04_e08_instances": len(ka & e08_keys), "x_r04_e07_test_instances": len(ka & e07_keys),
                 "x_r04_e07_test_classes": len(set(chosen) & set(ref)),
                 "x_r04_train_val_classes": len(set(chosen) & trval_classes)}
    if not a1_checks["classes_as_chosen"] or any(v for k, v in a1_checks.items() if k != "classes_as_chosen"):
        raise RuntimeError(f"A1 instance checks failed: {a1_checks}")
    t1 = time.perf_counter()
    rows, a1_build = D.build_dataset(a1, cfg["a1_states"], seed=A1_STATES_SEED, with_opt=False, ex=ex)
    a1_rows = [{"i": x["inst"], "n": x["n"], "s": R.fs_json(x["state"]), "h": x["h"]} for x in rows]
    a1_label_s = time.perf_counter() - t1
    SP.get_space.cache_clear()  # release the n = 8 space (~180 MB)
    ctx.log(f"data_b: A1 classes {chosen}; {len(a1_rows)} n={n8} rows in {a1_label_s:.1f}s; build {a1_build}")

    save_instances(dd / "testb_instances.json", testb)
    save_instances(dd / "valb_instances.json", valb)
    save_instances(dd / "a1_instances.json", a1)
    R.write_jsonl_gz(dd / "a1_n8.jsonl.gz", a1_rows)
    by = lambda xs: dict(sorted(Counter(it["n"] for it in xs).items()))  # noqa: E731
    meta = {"instances": {"testb": by(testb), "valb": by(valb), "a1": by(a1)}, "generation": gen, "checks": checks,
            "opt_coverage": {str(n): f"{sum(1 for it in testb if it['n'] == n and it['opt'] is not None)}/"
                                      f"{sum(1 for it in testb if it['n'] == n)}" for n in cfg["testb_sizes"]},
            "opt_seconds_mean": {str(n): float(np.mean([it["opt_s"] for it in testb if it["n"] == n]))
                                 for n in cfg["testb_sizes"]},
            "a1": {"n": n8, "discovery_seed": A1_DISCOVERY_SEED, "discovery_draws": cfg["a1_discovery_draws"],
                   "discovery_filters": disc_st, "discovery_counts": disc,
                   "excluded_as_not_fresh_r04_e07_test_classes": ref, "fresh_ranked": [[k, disc[k]] for k in fresh],
                   "chosen": chosen, "r04_e08_n8_classes_context": e08_n8,
                   "strict_reading_fresh_classes": [k for k in fresh if k not in e08_n8],
                   "instance_seed": SEEDS_B["spare_a1_n8"], "states_seed": A1_STATES_SEED, "checks": a1_checks,
                   "build": a1_build, "rows": len(a1_rows),
                   "rows_per_instance": dict(Counter(r["i"] for r in a1_rows)), "label_s": a1_label_s},
            "exclusions_sha256": excl_sha, "selection_set_exclusions": ex["selection_sources"],
            "wall_s": time.perf_counter() - t0}
    R.write_json(dd / "meta.json", meta)
    ctx.log(f"data_b: done in {meta['wall_s']:.1f}s")


# ----------------------------------------------------------------------------- phase: train_b

def build_rank_data(ctx, rows, inst, ex):
    """Successor structure for the A2 ranking hinge, from the exact spaces (n <= 7) of the R-04 train instances.

    For every training row with h* > 0: its successors (BWM semantics, space.successors), each labelled optimal iff
    h*(succ) = h*(s) - 1. A successor whose (state, goal) falls under the row-level exclusion recipe is dropped (no
    reserved instance enters training, even as a ranking partner). Pairs = (optimal o, non-optimal q); a row with no
    pair gets no hinge. Successor graphs: a successor that is itself a training row reuses that row's graph; the others
    are encoded once per (instance, state) into `extra`. G = X + extra; succ_g[r] indexes G."""
    by_inst = defaultdict(list)
    for k, r in enumerate(rows):
        by_inst[r["i"]].append(k)
    n_rows = len(rows)
    extra, succ_g, succ_pairs, st = [], [()] * n_rows, [()] * n_rows, Counter()
    for iid in sorted(by_inst):
        it = inst[iid]
        objs, goal = it["objs"], it["goal"]
        L = SP.GoalLabels(objs, goal)
        sp = L.space
        sid = {k: L.sid(R.fs_from(rows[k]["s"])) for k in by_inst[iid]}
        row_of = {i: k for k, i in sid.items()}
        gid, excluded = {}, {}
        for k in by_inst[iid]:
            i = sid[k]
            d = int(L.dist[i])
            if d != rows[k]["h"]:
                raise RuntimeError(f"row {k}: stored h {rows[k]['h']} != exact {d}")
            if d <= 0:
                st["rows_h0"] += 1
                continue
            gl, opt = [], []
            for j in sp.fwd_dst[sp.fwd_ptr[i]:sp.fwd_ptr[i + 1]]:
                j = int(j)
                st["succ_refs"] += 1
                if j in row_of:
                    g = row_of[j]
                    st["succ_is_row"] += 1
                else:
                    if j not in excluded:
                        s = L.state(j)
                        why = EX.row_exclusion_reason(objs, s, goal, ex)
                        excluded[j] = why
                        if why is None:
                            gid[j] = n_rows + len(extra)
                            extra.append(EN.graph(s, goal, objs))
                    if excluded[j]:
                        st[f"succ_excluded_{excluded[j]}"] += 1
                        continue
                    g = gid[j]
                gl.append(g)
                opt.append(int(L.dist[j]) == d - 1)
            pairs = tuple((a, b) for a in range(len(gl)) if opt[a] for b in range(len(gl)) if not opt[b])
            if pairs:
                succ_g[k], succ_pairs[k] = tuple(gl), pairs
                st["rows_ranked"] += 1
                st["pairs"] += len(pairs)
                st["ranked_succ"] += len(gl)
                st["ranked_opt_succ"] += sum(opt)
            else:
                st[f"rows_no_pairs_{'no_optimal' if not any(opt) else 'all_optimal'}"] += 1
            ctx.courtesy()
    st["extra_graphs"] = len(extra)
    st = dict(st)
    st["pairs_per_ranked_row"] = st.get("pairs", 0) / max(1, st.get("rows_ranked", 0))
    st["succ_per_ranked_row"] = st.get("ranked_succ", 0) / max(1, st.get("rows_ranked", 0))
    return extra, succ_g, succ_pairs, st


def train_rank(ctx, seed, X, y, Xv, yv, G, succ_g, succ_pairs):
    """R-04's train_one (same init, per-epoch shuffle, batches, AdamW, epochs, time cap, val-MAE checkpoint) with the
    A2 ranking hinge added to the Huber loss.

    Batch construction: the R-04 minibatch of `batch_size` training rows (same permutation as R-04 for the seed); after
    the rows' graphs come the successor graphs of every row in the batch that has pairs, all collated into ONE graph
    batch and scored by ONE forward pass. Huber is computed on the rows' outputs exactly as in R-04. Hinge per pair
    (o optimal, q non-optimal successor of the same row) = relu(margin + f(o) - f(q)); per row the mean over its pairs;
    the batch hinge is the mean over rows with pairs. loss = huber + rank_weight * hinge (weight 1: 'added to Huber').
    """
    cfg = ctx.cfg
    torch.manual_seed(seed)
    model = R.new_model("gnn")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    rng = random.Random(seed)
    yt = torch.tensor(y, dtype=torch.float32)
    margin, weight = float(cfg["rank_margin"]), float(cfg["rank_weight"])
    best_mae, best_state, best_ep = math.inf, None, -1
    hist, step, cap_hit = [], 0, False
    t0, gw0 = time.perf_counter(), ctx.guard_wait
    for ep in range(cfg["epochs"]):
        perm = list(range(len(X)))
        rng.shuffle(perm)
        model.train()
        losses, hubers, hinges, te, gw_ep = [], [], [], time.perf_counter(), ctx.guard_wait
        n_pairs = n_margin_viol = n_misorder = n_ranked = n_graphs = 0
        for b0 in range(0, len(perm), cfg["batch_size"]):
            idx = perm[b0:b0 + cfg["batch_size"]]
            nb = len(idx)
            graphs = [X[i] for i in idx]
            po, pq, ps = [], [], []
            for j, i in enumerate(idx):
                gl = succ_g[i]
                if gl:
                    off = len(graphs) - nb
                    graphs.extend(G[g] for g in gl)
                    for a, b in succ_pairs[i]:
                        po.append(off + a)
                        pq.append(off + b)
                        ps.append(j)
            out = model(model.collate(graphs))
            huber = F.huber_loss(out[:nb], yt[idx], delta=1.0)
            if po:
                sv = out[nb:]
                diff = sv[torch.tensor(po)] - sv[torch.tensor(pq)]
                v = F.relu(margin + diff)
                pst = torch.tensor(ps)
                tot = torch.zeros(nb).index_add(0, pst, v)
                cnt = torch.zeros(nb).index_add(0, pst, torch.ones(len(ps)))
                has = cnt > 0
                hinge = (tot[has] / cnt[has]).mean()
                n_pairs += len(po)
                n_margin_viol += int((v > 0).sum())
                n_misorder += int((diff >= 0).sum())
                n_ranked += int(has.sum())
            else:
                hinge = out.new_zeros(())
            loss = huber + weight * hinge
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            n_graphs += len(graphs)
            losses.append(float(loss))
            hubers.append(float(huber))
            hinges.append(float(hinge))
            ctx.courtesy()
            if time.perf_counter() - t0 - (ctx.guard_wait - gw0) > cfg["time_cap_s"]:
                cap_hit = True
                break
        train_s = time.perf_counter() - te - (ctx.guard_wait - gw_ep)
        pv = M.predict(model, Xv)
        mae, rmse = float(np.mean(np.abs(pv - yv))), float(np.sqrt(np.mean((pv - yv) ** 2)))
        hist.append({"epoch": ep, "steps": step, "train_loss": float(np.mean(losses)), "huber": float(np.mean(hubers)),
                     "hinge": float(np.mean(hinges)), "pairs": n_pairs,
                     "margin_violation_rate": n_margin_viol / max(1, n_pairs),
                     "misorder_rate": n_misorder / max(1, n_pairs), "ranked_rows": n_ranked,
                     "graphs_per_step": n_graphs / max(1, len(losses)), "val_mae": mae, "val_rmse": rmse,
                     "train_s": round(train_s, 3), "sec_per_step": train_s / max(1, len(losses))})
        ctx.log(f"train A2 s{seed} ep{ep} steps {step} loss {np.mean(losses):.4f} huber {np.mean(hubers):.4f} "
                f"hinge {np.mean(hinges):.4f} misorder {n_misorder / max(1, n_pairs):.4f} val_mae {mae:.4f}")
        if mae < best_mae:
            best_mae, best_state, best_ep = mae, copy.deepcopy(model.state_dict()), ep
        ctx.courtesy(force=True)
        if cap_hit:
            ctx.log(f"train A2 s{seed}: TIME CAP {cfg['time_cap_s']}s hit at step {step} (budget mismatch!)")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    wall = time.perf_counter() - t0
    return model, {"arch": "gnn", "seed": seed, "config": M.config_dict(model), "n_params": M.n_params(model),
                   "epochs_run": len(hist), "steps": step, "best_epoch": best_ep, "best_val_mae": best_mae,
                   "cap_hit": cap_hit, "wall_s": wall, "guard_wait_s": ctx.guard_wait - gw0,
                   "sec_per_step": float(np.mean([h["sec_per_step"] for h in hist])), "history": hist,
                   "loss": f"huber(delta=1) + {weight:g} * pairwise hinge(margin {margin:g})"}


def save_model(ctx, arm, seed, model, info):
    md = ctx.out / "models"
    p = md / f"{arm}_s{seed}.pt"
    torch.save(model.state_dict(), p)
    info.update(arm=arm, weights_sha256=R.sha_file(p))
    R.write_json(md / f"{arm}_s{seed}.json", info)
    ctx.log(f"train_b: {arm} s{seed} best ep {info['best_epoch']} val_mae {info['best_val_mae']:.4f} "
            f"in {info['wall_s']:.1f}s ({info['steps']} steps, {info['sec_per_step'] * 1000:.1f} ms/step)")


def phase_train_b(ctx):
    cfg, out = ctx.cfg, ctx.out
    md = out / "models"
    md.mkdir(exist_ok=True)
    todo = [(arm, s) for arm in ("A1", "A2") for s in cfg[f"{arm.lower()}_seeds"]
            if not (md / f"{arm}_s{s}.json").exists()]
    if not todo:
        ctx.log("train_b: outputs exist, skipping")
        return
    torch.set_num_threads(ctx.threads)
    ctx.courtesy(force=True)
    inst = R.load_instances(R04_DIR)
    tr = R.read_jsonl_gz(R04_DIR / "data" / "train.jsonl.gz")[::cfg["r04_train_stride"]]
    va = R.read_jsonl_gz(R04_DIR / "data" / "val.jsonl.gz")
    y = np.array([r["h"] for r in tr], dtype=np.float32)
    yv = np.array([r["h"] for r in va], dtype=np.float32)
    t1 = time.perf_counter()
    X, Xv = R.encode_rows("gnn", tr, inst), R.encode_rows("gnn", va, inst)
    enc_s = time.perf_counter() - t1
    ctx.log(f"train_b: encoded {len(X)} R-04 train / {len(Xv)} val rows in {enc_s:.1f}s")
    seeds = [s for a, s in todo if a == "A1"]
    if seeds:
        a1_inst = {it["id"]: it for it in load_instance_list(out / "data" / "a1_instances.json")}
        a1 = R.read_jsonl_gz(out / "data" / "a1_n8.jsonl.gz")
        t1 = time.perf_counter()
        Xa = X + R.encode_rows("gnn", a1, a1_inst)
        ya = np.concatenate([y, np.array([r["h"] for r in a1], dtype=np.float32)])
        enc_a = time.perf_counter() - t1
        ctx.log(f"train_b: A1 rows {len(X)} R-04 + {len(a1)} n=8 = {len(Xa)}")
        for seed in seeds:
            model, info = R.train_one(ctx, "gnn", seed, Xa, ya, Xv, yv)
            info.update(encode_s=enc_s + enc_a, train_rows=len(Xa), train_rows_r04=len(X), train_rows_n8=len(a1),
                        val_rows=len(Xv), loss="huber(delta=1) (R-04)")
            save_model(ctx, "A1", seed, model, info)
        del Xa, ya
    seeds = [s for a, s in todo if a == "A2"]
    if seeds:
        t1 = time.perf_counter()
        ex = EX.with_selection_sets(EX.load_exclusions())
        extra, succ_g, succ_pairs, rst = build_rank_data(ctx, tr, inst, ex)
        rst["build_s"] = time.perf_counter() - t1
        G = X + extra
        ctx.log(f"train_b: A2 rank data {rst}")
        for seed in seeds:
            model, info = train_rank(ctx, seed, X, y, Xv, yv, G, succ_g, succ_pairs)
            info.update(encode_s=enc_s, train_rows=len(X), val_rows=len(Xv), rank_data=rst)
            save_model(ctx, "A2", seed, model, info)


# ----------------------------------------------------------------------------- phase: select (VAL-B)

def phase_select(ctx):
    cfg, out = ctx.cfg, ctx.out
    if (out / "select.json").exists():
        ctx.log("select: outputs exist, skipping")
        return
    torch.set_num_threads(ctx.threads)
    ctx.courtesy(force=True)
    t0 = time.perf_counter()
    valb = load_instance_list(out / "data" / "valb_instances.json")
    bp, bl = cfg["b_primary"], cfg["b_low"]
    lam_names = [a3_name(lam) for lam in cfg["lambdas"]]
    a1s1, a2s1, a0s1 = arm_names(cfg, "A1")[0], arm_names(cfg, "A2")[0], arm_names(cfg, "A0")[0]
    names = lam_names + [a1s1, a2s1] + [a0s1, "misplaced_blocks"]  # last two: context only, not candidates
    done = run_set(ctx, valb, names, out / "valb_runs.jsonl", max(cfg["budgets"]), heuristic_factory(ctx))
    ids = [it["id"] for it in valb]
    table = {nm: {str(b): succ_count(done, ids, nm, b) for b in cfg["budgets"]} for nm in names}
    per_size = {nm: {str(n): succ_count(done, [it["id"] for it in valb if it["n"] == n], nm, bp)
                     for n in cfg["valb_sizes"]} for nm in names}
    # lambda: success@b_primary, then success@b_low, then the smaller lambda (least departure from A0)
    lam = max(cfg["lambdas"], key=lambda x: (table[a3_name(x)][str(bp)], table[a3_name(x)][str(bl)], -x))
    cands = [("A1", a1s1), ("A2", a2s1), ("A3", a3_name(lam))]
    # BEST: success@b_primary, then success@b_low, then prereg listing order A1, A2, A3
    k = max(range(len(cands)), key=lambda j: (table[cands[j][1]][str(bp)], table[cands[j][1]][str(bl)], -j))
    sel = {"lambda": lam, "a3_heur": a3_name(lam), "best_arm": cands[k][0], "best_heur": cands[k][1],
           "candidates": [{"arm": a, "heur": h, f"success@{bp}": table[h][str(bp)], f"success@{bl}": table[h][str(bl)]}
                          for a, h in cands],
           "lambda_candidates": [{"lambda": x, f"success@{bp}": table[a3_name(x)][str(bp)],
                                  f"success@{bl}": table[a3_name(x)][str(bl)]} for x in cfg["lambdas"]],
           "valb_n": len(ids), "valb_success": table, f"valb_success_per_size@{bp}": per_size,
           "context_only": [a0s1, "misplaced_blocks"],
           "rules": {"lambda": f"max success@{bp}, then success@{bl}, then smaller lambda",
                     "best": f"max success@{bp}, then success@{bl}, then order A1, A2, A3; seed 1 of trained arms"},
           "wall_s": time.perf_counter() - t0}
    R.write_json(out / "select.json", sel)
    ctx.log(f"select: lambda* = {lam}, BEST = {sel['best_arm']} ({sel['best_heur']}); VAL-B {table}")


# ----------------------------------------------------------------------------- phase: e08b (TEST-B)

def monotonicity_check(ctx, inst, done, names):
    """Re-run GBFS directly at each smaller budget for a few instances (some cut by the smaller budgets, some solved
    within b_low) and compare with the derived records; any mismatch aborts."""
    cfg = ctx.cfg
    bmax = max(cfg["budgets"])
    n_cut, n_in = cfg["mono_check"]
    make, res = heuristic_factory(ctx), []
    for name in names:
        h = make(name)
        cut = [it for it in inst if done[(it["id"], name)]["expansions"] > cfg["b_low"]][:n_cut]
        within = [it for it in inst if done[(it["id"], name)]["expansions"] <= cfg["b_low"]][:n_in]
        for it in cut + within:
            for b in [b for b in cfg["budgets"] if b < bmax]:
                direct = gbfs_record(it, name, h, b)
                der = at_budget(done[(it["id"], name)], b)
                same = all(direct[k] == der[k] for k in ("success", "found", "plan_len", "expansions", "reason",
                                                         "plan"))
                res.append({"inst": it["id"], "heur": name, "budget": b, "same": same,
                            "full_run_expansions": done[(it["id"], name)]["expansions"]})
                ctx.courtesy()
    bad = [r for r in res if not r["same"]]
    if bad:
        raise RuntimeError(f"budget-monotonicity check failed: {bad[:5]}")
    return res


def phase_e08b(ctx):
    cfg, out = ctx.cfg, ctx.out
    if (out / "e08b_metrics.json").exists():
        ctx.log("e08b: outputs exist, skipping")
        return
    torch.set_num_threads(ctx.threads)
    ctx.courtesy(force=True)
    t0 = time.perf_counter()
    sel = json.loads((out / "select.json").read_text(encoding="utf-8"))
    testb = load_instance_list(out / "data" / "testb_instances.json")
    names = arm_names(cfg, "A0") + arm_names(cfg, "A1") + arm_names(cfg, "A2") + [sel["a3_heur"]] + list(BASELINES)
    done = run_set(ctx, testb, names, out / "testb_runs.jsonl", max(cfg["budgets"]), heuristic_factory(ctx))
    mono = monotonicity_check(ctx, testb, done, ["misplaced_blocks", arm_names(cfg, "A0")[0]])
    ctx.log(f"e08b: budget-monotonicity check {sum(r['same'] for r in mono)}/{len(mono)} identical")
    m = e08b_metrics(cfg, testb, done, names, sel)
    m["monotonicity_check"] = mono
    m["wall_s"] = time.perf_counter() - t0
    R.write_json(out / "e08b_metrics.json", m)
    p = m["primary"]
    ctx.log(f"e08b: PRIMARY outcome {p['outcome']} (BEST {p['best_heur']} {p['best_at_primary']}/{p['n']} vs "
            f"misplaced {p['misplaced_at_primary']}/{p['n']} @{cfg['b_primary']}); done in {m['wall_s']:.1f}s")


def e08b_metrics(cfg, inst, done, names, sel):
    bp, bl = cfg["b_primary"], cfg["b_low"]
    scopes = {str(n): [it["id"] for it in inst if it["n"] == n] for n in cfg["testb_sizes"]}
    scopes["primary"] = [it["id"] for it in inst if it["n"] in cfg["primary_sizes"]]
    scopes["all"] = [it["id"] for it in inst]
    res = {"names": names, "scopes": {k: len(v) for k, v in scopes.items()}, "cells": {}, "seed_agg": {},
           "mcnemar": [], "sec_per_expansion": {}}
    for nm in names:
        for b in cfg["budgets"]:
            for sc, ids in scopes.items():
                res["cells"][f"{nm}|{b}|{sc}"] = cell(done, ids, nm, b)
        real = [done[(i, nm)] for i in scopes["all"]]
        res["sec_per_expansion"][nm] = sum(r["seconds"] for r in real) / max(1, sum(r["expansions"] for r in real))
    for arm in ("A0", "A1", "A2"):
        for b in cfg["budgets"]:
            for sc in scopes:
                cs = [res["cells"][f"{nm}|{b}|{sc}"] for nm in arm_names(cfg, arm)]
                res["seed_agg"][f"{arm}|{b}|{sc}"] = {
                    "success": mean_sd([c["success"] for c in cs]),
                    "subopt_mean": mean_sd([c["subopt_mean"] for c in cs]),
                    "expansions_median": mean_sd([c["expansions_median"] for c in cs])}
    P = scopes["primary"]
    a0, best, mis = arm_names(cfg, "A0")[0], sel["best_heur"], "misplaced_blocks"
    # ---- PRIMARY decision, exactly per PREREG_E07B 'Decision rule (primary)' (paired tests use seed 1 models)
    kb, km, ka = succ_count(done, P, best, bp), succ_count(done, P, mis, bp), succ_count(done, P, a0, bp)
    kbl, kml = succ_count(done, P, best, bl), succ_count(done, P, mis, bl)
    mc_low = mcnemar(done, P, mis, best, bl)
    mc_hi = mcnemar(done, P, mis, best, bp)
    mc_a0 = mcnemar(done, P, a0, best, bp)
    c1 = kb >= km - cfg["closed_margin"]
    c2 = kbl > kml and mc_low["p_exact"] < cfg["alpha"]
    closed = c1 and c2
    improved = (kb - ka >= cfg["improved_delta"]) and mc_a0["p_exact"] < cfg["alpha"] and not closed
    res["primary"] = {
        "scope": f"TEST-B n in {cfg['primary_sizes']}", "n": len(P), "best_arm": sel["best_arm"], "best_heur": best,
        "lambda": sel["lambda"], "a0_heur": a0, "b_primary": bp, "b_low": bl,
        "best_at_primary": kb, "misplaced_at_primary": km, "a0_at_primary": ka,
        "best_at_low": kbl, "misplaced_at_low": kml,
        "closed_c1": c1, "closed_c1_rule": f"{kb} >= {km} - {cfg['closed_margin']}",
        "closed_c2": c2, "closed_c2_rule": f"{kbl} > {kml} and McNemar p {mc_low['p_exact']:.4g} < {cfg['alpha']}",
        "mcnemar_low_best_vs_misplaced": mc_low, "mcnemar_primary_best_vs_misplaced": mc_hi,
        "mcnemar_primary_best_vs_a0": mc_a0, "improved_delta": kb - ka,
        "improved_rule": f"{kb} - {ka} = {kb - ka} >= {cfg['improved_delta']} and McNemar p {mc_a0['p_exact']:.4g} < "
                         f"{cfg['alpha']} and not CLOSED",
        "closed": closed, "improved": improved,
        "outcome": "CLOSED" if closed else ("IMPROVED" if improved else "NOT IMPROVED")}
    # ---- secondary (descriptive)
    cands = [arm_names(cfg, "A1")[0], arm_names(cfg, "A2")[0], sel["a3_heur"]]
    for b in (bp, bl):
        for sc in ("primary", "all"):
            for c in cands:
                res["mcnemar"].append(dict(mcnemar(done, scopes[sc], a0, c, b), scope=sc, kind="A0_vs_arm"))
            for c in [a0] + cands:
                res["mcnemar"].append(dict(mcnemar(done, scopes[sc], mis, c, b), scope=sc, kind="misplaced_vs_arm"))
            for j, s0 in enumerate(arm_names(cfg, "A0")):
                for arm in ("A1", "A2"):
                    an = arm_names(cfg, arm)
                    if j < len(an):
                        res["mcnemar"].append(dict(mcnemar(done, scopes[sc], s0, an[j], b), scope=sc,
                                                   kind="seed_matched"))
    return res


# ----------------------------------------------------------------------------- phase: policy (R-04 E07 test states)

def phase_policy(ctx):
    cfg, out = ctx.cfg, ctx.out
    if (out / "policy_metrics.json").exists():
        ctx.log("policy: outputs exist, skipping")
        return
    torch.set_num_threads(ctx.threads)
    ctx.courtesy(force=True)
    t0 = time.perf_counter()
    sel = json.loads((out / "select.json").read_text(encoding="utf-8"))
    B, _ = BI.load()
    inst = R.load_instances(R04_DIR)
    test = R.read_jsonl_gz(R04_DIR / "data" / "test.jsonl.gz")
    old = R.read_jsonl_gz(R04_DIR / "e07_scores.jsonl.gz")  # R-04's per-state E07 scores, same order as test
    if len(old) != len(test) or any((a["i"], a["n"], a["h"]) != (b["i"], b["n"], b["h"]) for a, b in zip(test, old)):
        raise RuntimeError("R-04 e07_scores.jsonl.gz does not align with data/test.jsonl.gz")
    keep = [k for k, r in enumerate(test) if r["n"] in cfg["policy_sizes"]]
    if cfg["policy_inst_per_size"]:
        first = defaultdict(list)
        for k in keep:
            r = test[k]
            if r["i"] not in first[r["n"]] and len(first[r["n"]]) < cfg["policy_inst_per_size"]:
                first[r["n"]].append(r["i"])
        allowed = {i for v in first.values() for i in v}
        keep = [k for k in keep if test[k]["i"] in allowed]
    flat, rows = [], []
    for k in keep:  # as R-04's phase_e07: root + its BWM successors
        r = test[k]
        it = inst[r["i"]]
        s = R.fs_from(r["s"])
        succ = R.successors_bwm(B, it["objs"], s)
        base = len(flat)
        flat.append((r["i"], s))
        flat += [(r["i"], ns) for _, ns in succ]
        rows.append({"k": k, "i": r["i"], "n": r["n"], "h": r["h"], "miss": r["miss"], "idx": base,
                     "succ_idx": list(range(base + 1, base + 1 + len(succ))), "succ_act": [a for a, _ in succ],
                     "opt": None if r["opt"] is None else {tuple(a) for a in r["opt"]}})
    ctx.log(f"policy: {len(rows)} R-04 E07 test states (n in {cfg['policy_sizes']}), {len(flat)} states to score")
    enc = [EN.graph(s, inst[i]["goal"], inst[i]["objs"]) for i, s in flat]
    scores = {}
    for arm in ("A0", "A1", "A2"):
        for nm in arm_names(cfg, arm):
            scores[nm] = M.predict(load_gnn(model_path(ctx, arm, int(nm.split("_s")[1]))), enc).astype(float)
            ctx.courtesy()
    a0 = arm_names(cfg, "A0")[0]
    mis = np.array([H.misplaced_blocks(s, inst[i]["goal"]) for i, s in flat], float)
    scores[sel["a3_heur"]] = scores[a0] + float(sel["lambda"]) * mis  # descriptive extra (A3 is untrained)
    models, tol = list(scores), cfg["tie_tol"]
    for r in rows:  # identical to R-04's phase_e07 policy/tie computation
        r["pred"], r["pol"], r["tie"] = {}, {}, {}
        for m in models:
            v = scores[m]
            r["pred"][m] = float(v[r["idx"]])
            if r["h"] is None or r["h"] == 0 or r["opt"] is None or not r["succ_idx"]:
                r["pol"][m], r["tie"][m] = None, None
                continue
            sv = v[r["succ_idx"]]
            lo = float(np.min(sv))
            mins = [j for j, x in enumerate(sv) if x <= lo + tol * max(1.0, abs(lo))]
            r["pol"][m] = int(all(r["succ_act"][j] in r["opt"] for j in mins))
            r["tie"][m] = len(mins) > 1
    groups = {str(n): [n] for n in cfg["policy_sizes"]}
    groups["OOD"] = list(cfg["policy_sizes"])
    met = {"per_model": {}, "arm_mean": {}, "bootstrap": {}, "a0_reproduction": {}}
    for g, ns in groups.items():
        selr = [r for r in rows if r["n"] in ns]
        for m in models:
            met["per_model"].setdefault(m, {})[g] = R.metric_block(selr, m, not m.startswith("A3_"))
        for arm in ("A0", "A1", "A2"):
            met["arm_mean"].setdefault(arm, {})[g] = {
                key: mean_sd([met["per_model"][m][g][key] for m in arm_names(cfg, arm)])
                for key in ("policy_acc", "spearman", "tie_rate", "mae", "rmse", "exact")}

        def arms_of(names):
            out_ = []
            for m in names:
                d = defaultdict(list)
                for r in selr:
                    if r["pol"][m] is not None:
                        d[r["i"]].append(r["pol"][m])
                out_.append(d)
            return out_

        for arm in ("A1", "A2"):
            try:
                met["bootstrap"].setdefault(g, {})[f"{arm}_minus_A0"] = ST.seed_parent_bootstrap(
                    arms_of(arm_names(cfg, "A0")), arms_of(arm_names(cfg, arm)),
                    resamples=cfg["bootstrap_resamples"], seed=SEEDS_B["bootstrap"])
            except ValueError as e:
                met["bootstrap"].setdefault(g, {})[f"{arm}_minus_A0"] = {"error": str(e)}
    for s in cfg["a0_seeds"]:  # A0 must reproduce R-04's stored E07 predictions and policy decisions
        nm, old_nm = f"A0_s{s}", f"gnn_s{s}"
        dif = [abs(r["pred"][nm] - old[r["k"]]["pred"][old_nm]) for r in rows]
        met["a0_reproduction"][nm] = {"max_abs_pred_diff": float(max(dif)) if dif else None,
                                      "policy_disagreements": sum(r["pol"][nm] != old[r["k"]]["pol"][old_nm]
                                                                  for r in rows), "states": len(rows)}
    met.update(n_states=len(rows), n_scored=len(flat), models=models, wall_s=time.perf_counter() - t0)
    R.write_jsonl_gz(out / "policy_scores.jsonl.gz", [{k: r[k] for k in ("i", "n", "h", "miss", "pred", "pol", "tie")}
                                                      for r in rows])
    R.write_json(out / "policy_metrics.json", met)
    ctx.log(f"policy: A0 reproduction {met['a0_reproduction']}; done in {met['wall_s']:.1f}s")


# ----------------------------------------------------------------------------- phase: report

def ms(blk, nd=3):
    if blk is None or blk.get("mean") is None:
        return "–"
    return R.fmt(blk["mean"], nd) + (f" ± {blk['sd']:.{nd}f}" if blk.get("sd") is not None else "")


def estimate_full(cfg, trains, m08):
    """Indicative full-run CPU time from this smoke run's unit costs (same machine and --threads)."""
    full = FULL
    r04_meta = json.loads((R04_DIR / "data" / "meta.json").read_text(encoding="utf-8"))
    r04_rows = sum(r04_meta["stats"]["rows"]["train"].values())
    sps = {arm: float(np.mean([t["sec_per_step"] for t in trains if t["arm"] == arm])) for arm in ("A1", "A2")}
    a1_keep = np.mean([t["train_rows_n8"] for t in trains if t["arm"] == "A1"]) / (cfg["a1_classes"] *
                                                                                    cfg["a1_states"])
    a1_rows = full["a1_classes"] * full["a1_states"] * a1_keep
    steps = {"A1": full["epochs"] * math.ceil((r04_rows + a1_rows) / full["batch_size"]),
             "A2": full["epochs"] * math.ceil(r04_rows / full["batch_size"])}
    est = {"sec_per_step": sps, "steps_per_model": steps,
           "train_s": sum(sps[a] * steps[a] * len(full[f"{a.lower()}_seeds"]) for a in sps)}
    spe = m08["sec_per_expansion"]
    learned = float(np.mean([v for k, v in spe.items() if k.startswith(("A0", "A1", "A2", "A3"))]))
    n_learned = 3 * 3 + 1
    bmax = max(full["budgets"])
    upper = bmax * (full["testb_per_size"] * len(full["testb_sizes"]) *
                    (n_learned * learned + sum(spe[b] for b in BASELINES)) +
                    full["valb_per_size"] * len(full["valb_sizes"]) * (6 * learned + spe["misplaced_blocks"]))
    r04_m08 = json.loads((R04_DIR / "e08_metrics.json").read_text(encoding="utf-8"))
    est.update(gbfs_upper_s=upper, gbfs_expected_s=0.4 * upper,
               opt_astar_s=sum(full["testb_per_size"] * r04_m08["opt_seconds"].get(str(n), 2.0)
                               for n in full["testb_sizes"]),
               other_s=120.0)
    est["total_expected_s"] = est["train_s"] + est["gbfs_expected_s"] + est["opt_astar_s"] + est["other_s"]
    est["total_upper_s"] = est["train_s"] * 1.3 + upper + 3 * est["opt_astar_s"] + est["other_s"]
    est["assumptions"] = ("train: smoke sec/step x full steps (12 epochs, batch 64); GBFS: smoke sec/expansion at "
                          "TEST-B sizes, upper = every run exhausts the 2000 budget, expected = 0.4 x upper (R-04: "
                          "~35% of learned runs exhaust the budget, solved runs need ~50-400 expansions); A*: R-04's "
                          "mean seconds per size; other: data draws, n=8 labels, encodings, policy")
    return est


def phase_report(ctx, timings):
    cfg, out = ctx.cfg, ctx.out
    meta = json.loads((out / "data" / "meta.json").read_text(encoding="utf-8"))
    trains = [json.loads((out / "models" / f"{a}_s{s}.json").read_text(encoding="utf-8"))
              for a in ("A1", "A2") for s in cfg[f"{a.lower()}_seeds"]]
    sel = json.loads((out / "select.json").read_text(encoding="utf-8"))
    m08 = json.loads((out / "e08b_metrics.json").read_text(encoding="utf-8"))
    pol = json.loads((out / "policy_metrics.json").read_text(encoding="utf-8"))
    inputs = json.loads((out / "inputs.json").read_text(encoding="utf-8"))
    bp, bl, P = cfg["b_primary"], cfg["b_low"], m08["primary"]
    smoke = ctx.smoke
    L = [f"# RCX-E07b report ({'SMOKE — plumbing check only, numbers are not results' if smoke else 'full'})", "",
         "Preregistered in `experiments/RCX-E07/PREREG_E07B.md` (frozen, sha256 verified at start; CX D-20). CPU "
         "mechanism tier, exact dynamics; a learned heuristic is an external specialist (tier B), so nothing here "
         "supports a Rineya-core claim. One primary test; everything else is descriptive.", ""]
    L += ["## PRIMARY decision (computed by the prereg rule)", "",
          f"Scope: {P['scope']} ({P['n']} instances). `BEST` = **{P['best_arm']}** (`{P['best_heur']}`), chosen on "
          f"VAL-B; A3's λ = {P['lambda']}. Paired tests use seed 1 of each arm (A0 = `{P['a0_heur']}`).", "",
          "| quantity | value |", "|---|---|",
          f"| BEST success@{bp} | {P['best_at_primary']}/{P['n']} |",
          f"| misplaced_blocks success@{bp} | {P['misplaced_at_primary']}/{P['n']} |",
          f"| A0 success@{bp} | {P['a0_at_primary']}/{P['n']} |",
          f"| BEST success@{bl} vs misplaced_blocks@{bl} | {P['best_at_low']} vs {P['misplaced_at_low']} "
          f"(misplaced-only {P['mcnemar_low_best_vs_misplaced']['base_only']}, BEST-only "
          f"{P['mcnemar_low_best_vs_misplaced']['cand_only']}, exact McNemar p = "
          f"{P['mcnemar_low_best_vs_misplaced']['p_exact']:.4g}) |",
          f"| BEST vs misplaced_blocks @{bp} | misplaced-only {P['mcnemar_primary_best_vs_misplaced']['base_only']}, "
          f"BEST-only {P['mcnemar_primary_best_vs_misplaced']['cand_only']}, p = "
          f"{P['mcnemar_primary_best_vs_misplaced']['p_exact']:.4g} |",
          f"| BEST vs A0 @{bp} | A0-only {P['mcnemar_primary_best_vs_a0']['base_only']}, BEST-only "
          f"{P['mcnemar_primary_best_vs_a0']['cand_only']}, p = {P['mcnemar_primary_best_vs_a0']['p_exact']:.4g} |",
          f"| CLOSED (a): BEST@{bp} ≥ misplaced@{bp} − {cfg['closed_margin']} | {P['closed_c1_rule']} → "
          f"{P['closed_c1']} |",
          f"| CLOSED (b): BEST@{bl} > misplaced@{bl}, McNemar p < {cfg['alpha']} | {P['closed_c2_rule']} → "
          f"{P['closed_c2']} |",
          f"| IMPROVED: BEST@{bp} − A0@{bp} ≥ +{cfg['improved_delta']}, McNemar p < {cfg['alpha']}, not CLOSED | "
          f"{P['improved_rule']} → {P['improved']} |",
          f"| **Outcome** | **{P['outcome']}** |", ""]
    L += ["## Selection on VAL-B (λ and BEST only)", "",
          f"VAL-B: {sel['valb_n']} instances (n {cfg['valb_sizes']}). Rules: λ — {sel['rules']['lambda']}; BEST — "
          f"{sel['rules']['best']}. Rows marked context are not candidates.", "",
          "| heuristic | " + " | ".join(f"n={n} @{bp}" for n in cfg["valb_sizes"]) + " | " +
          " | ".join(f"all @{b}" for b in cfg["budgets"]) + " |",
          "|---|" + "---|" * (len(cfg["valb_sizes"]) + len(cfg["budgets"]))]
    for nm, row in sel["valb_success"].items():
        tag = " (context)" if nm in sel["context_only"] else ""
        L.append(f"| {nm}{tag} | " + " | ".join(str(sel[f'valb_success_per_size@{bp}'][nm][str(n)])
                                                for n in cfg["valb_sizes"]) +
                 " | " + " | ".join(str(row[str(b)]) for b in cfg["budgets"]) + " |")
    L += ["", f"Selected λ = **{sel['lambda']}**; BEST = **{sel['best_arm']}** (`{sel['best_heur']}`).", ""]
    sizes = [str(n) for n in cfg["testb_sizes"]]
    for b in sorted(cfg["budgets"], reverse=True):
        L += [f"## TEST-B — GBFS success, budget {b}" + (" (derived from the full-budget runs)" if b != max(
            cfg["budgets"]) else ""), "",
              "| heuristic | " + " | ".join(sizes) + " | n 12–15 | all | rate n 12–15 [Wilson 95%] | "
              "median exp (all) | mean plan len | mean subopt (n) |", "|---|" + "---|" * (len(sizes) + 7)]
        for nm in m08["names"]:
            c = {sc: m08["cells"][f"{nm}|{b}|{sc}"] for sc in sizes + ["primary", "all"]}
            pr = c["primary"]
            L.append(f"| {nm} | " + " | ".join(f"{c[sc]['success']}/{c[sc]['n']}" for sc in sizes) +
                     f" | {pr['success']}/{pr['n']} | {c['all']['success']}/{c['all']['n']} | "
                     f"{R.fmt(pr['rate'])} [{pr['wilson'][0]:.3f}, {pr['wilson'][1]:.3f}] | "
                     f"{c['all']['expansions_median']:.0f} | {R.fmt(c['all']['plan_len_mean'], 1)} | "
                     f"{R.fmt(c['all']['subopt_mean'])} ({c['all']['subopt_n']}) |")
        L.append("")
    L += ["## TEST-B — seed mean ± sd (success counts; A0 = R-04 seeds, A1/A2 = E07b seeds)", "",
          "| arm | budget | " + " | ".join(sizes) + " | n 12–15 | all | mean subopt (all) | median exp (all) |",
          "|---|---|" + "---|" * (len(sizes) + 4)]
    for arm in ("A0", "A1", "A2"):
        for b in sorted(cfg["budgets"], reverse=True):
            g = {sc: m08["seed_agg"][f"{arm}|{b}|{sc}"] for sc in sizes + ["primary", "all"]}
            L.append(f"| {arm} | {b} | " + " | ".join(ms(g[sc]["success"], 2) for sc in sizes + ["primary", "all"]) +
                     f" | {ms(g['all']['subopt_mean'])} | {ms(g['all']['expansions_median'], 1)} |")
    L += ["", "## TEST-B — paired exact McNemar (descriptive; seed 1 unless seed-matched)", "",
          "| kind | baseline | candidate | budget | scope | base succ | cand succ | base only | cand only | n | p |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in m08["mcnemar"]:
        L.append(f"| {v['kind']} | {v['base']} | {v['cand']} | {v['budget']} | {v['scope']} | {v['base_success']} | "
                 f"{v['cand_success']} | {v['base_only']} | {v['cand_only']} | {v['n']} | {v['p_exact']:.4f} |")
    mono = m08["monotonicity_check"]
    L += ["", "## Budget monotonicity", "",
          f"Each (instance, heuristic) runs once at budget {max(cfg['budgets'])}; success at a smaller budget b is "
          "derived as (found a valid plan and expansions ≤ b), as R-04 did. This is exact because `search.gbfs` "
          "checks the budget only before each pop, applies the goal test at generation, and its pop order and "
          "heuristic calls do not depend on the budget, so a b-budget run is a prefix of the larger run. Direct "
          "re-runs at every "
          f"smaller budget: {sum(r['same'] for r in mono)}/{len(mono)} identical (heuristics "
          f"{sorted({r['heur'] for r in mono})}).", ""]
    L += [f"## Policy accuracy on R-04's E07 test states (n {cfg['policy_sizes']}; E07 definition)", "",
          "| arm | " + " | ".join(str(n) for n in cfg["policy_sizes"]) + " | OOD (pooled states) |",
          "|---|" + "---|" * (len(cfg["policy_sizes"]) + 1)]
    gs = [str(n) for n in cfg["policy_sizes"]] + ["OOD"]
    for arm in ("A0", "A1", "A2"):
        L.append(f"| {arm} (mean ± sd over {len(cfg[arm.lower() + '_seeds'])} seeds) | " +
                 " | ".join(ms(pol["arm_mean"][arm][g]["policy_acc"]) for g in gs) + " |")
    L.append(f"| {sel['a3_heur']} (A0 s1 + λ·misplaced; descriptive extra) | " +
             " | ".join(R.fmt(pol["per_model"][sel["a3_heur"]][g]["policy_acc"]) for g in gs) + " |")
    L += ["", "| comparison | scope | mean A0 | mean arm | diff | 95% CI | instances |",
          "|---|---|---|---|---|---|---|"]
    for g, comp in pol["bootstrap"].items():
        for lab, c in comp.items():
            if "error" in c:
                L.append(f"| {lab} | {g} | – | – | – | {c['error']} | – |")
            else:
                L.append(f"| {lab} | {g} | {c['mean_a']:.3f} | {c['mean_b']:.3f} | {c['diff']:+.3f} | "
                         f"[{c['lo']:+.3f}, {c['hi']:+.3f}] | {c['n_parents']} |")
    L += ["", f"Instance-clustered bootstrap, seeds averaged within instance, {cfg['bootstrap_resamples']} resamples, "
          f"seed {SEEDS_B['bootstrap']}. A0 reproduction against R-04's stored E07 scores: "
          f"`{json.dumps(pol['a0_reproduction'])}`.", ""]
    L += ["## Training (A1, A2; R-04 recipe: same epochs, batch, AdamW lr, time cap, val-MAE checkpoint)", "",
          "| arm | seed | train rows | epochs | steps | best epoch | best val MAE | cap hit | wall s | ms/step | "
          "last-epoch hinge / misorder |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in trains:
        h = t["history"][-1]
        rk = f"{h['hinge']:.4f} / {h['misorder_rate']:.4f}" if "hinge" in h else "–"
        L.append(f"| {t['arm']} | {t['seed']} | {t['train_rows']} | {t['epochs_run']} | {t['steps']} | "
                 f"{t['best_epoch']} | {t['best_val_mae']:.4f} | {t['cap_hit']} | {t['wall_s']:.1f} | "
                 f"{t['sec_per_step'] * 1000:.1f} | {rk} |")
    a2 = next((t for t in trains if t["arm"] == "A2"), None)
    if a2:
        L += ["", f"A2 rank data: `{json.dumps(a2['rank_data'])}`"]
    L += ["", "## Data", "",
          f"TEST-B {meta['instances']['testb']}, VAL-B {meta['instances']['valb']} (full goals; seeds "
          f"{SEEDS_B['testb_instances']}, {SEEDS_B['valb_instances']}). Disjointness checks: "
          f"`{json.dumps(meta['checks'])}`.",
          f"Generation filters: `{json.dumps(meta['generation'])}`.",
          f"Optimal-length coverage (A* within {cfg['opt_budget']} expansions): {meta['opt_coverage']}.", "",
          f"A1 n = {meta['a1']['n']} goal classes. Availability = discovery count in {meta['a1']['discovery_draws']} "
          f"draws (seed `{meta['a1']['discovery_seed']}`, filters {meta['a1']['discovery_filters']}); fresh = not a "
          f"goal class of R-04's E07 test states at n = {meta['a1']['n']}; chosen = the {cfg['a1_classes']} most "
          f"frequent fresh classes with ≥ {cfg['a1_min_class_count']} draws.", "",
          "| goal class | discovery count | status |", "|---|---|---|"]
    a1m = meta["a1"]
    for k, v in sorted(a1m["discovery_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        st = ("chosen" if k in a1m["chosen"] else "R-04 E07 test class (not fresh)"
              if k in a1m["excluded_as_not_fresh_r04_e07_test_classes"] else "fresh, not chosen")
        st += " · in R-04 E08 n=8" if k in a1m["r04_e08_n8_classes_context"] else ""
        L.append(f"| {k} | {v} | {st} |")
    L += ["", f"A1 instances: seed {a1m['instance_seed']}; states: {cfg['a1_states']} sampled per instance (seed "
          f"`{a1m['states_seed']}`), exact h* from the n = 8 space; rows kept {a1m['rows']}; build "
          f"`{json.dumps(a1m['build'])}`; checks `{json.dumps(a1m['checks'])}`.", ""]
    L += ["## Resolutions of prereg ambiguities (see README, section RCX-E07b)", ""] + [f"- {x}" for x in RESOLUTIONS]
    shas = {f: R.sha_file(ROOT / f) for f in CODE_FILES}
    prov = {"prereg": {"path": "experiments/RCX-E07/PREREG_E07B.md", "sha256_verified": verify_prereg()},
            "code_sha256": shas, "exclusions_json_sha256": meta["exclusions_sha256"],
            "selection_set_exclusions": meta["selection_set_exclusions"], "r04_inputs": inputs, "bwm_pins": BI.PINS,
            "seeds": {**SEEDS_B, "a1_discovery": A1_DISCOVERY_SEED, "a1_states": A1_STATES_SEED,
                      "a0_models_r04": cfg["a0_seeds"], "r04_seeds": D.SEEDS},
            "config": cfg, "smoke": smoke, "threads": ctx.threads, "python": sys.version.split()[0],
            "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(), "wall_s": timings,
            "guard_wait_s": ctx.guard_wait, "bwm_bytecode_check": BI.check_no_bytecode()}
    L += ["", "## Provenance", "", "```json", json.dumps(prov, indent=1), "```", ""]
    est = None
    if smoke:
        est = estimate_full(cfg, trains, m08)
        L += ["## Estimated full-run CPU time (extrapolated from this smoke run; indicative)", "", "```json",
              json.dumps(est, indent=1), "```", ""]
    (out / "REPORT_E07B.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    R.write_json(out / "metrics_e07b.json", {"primary": P, "select": sel, "e08b": m08, "policy": pol,
                                             "train": trains, "data": meta, "provenance": prov,
                                             "estimate_full": est})
    ctx.log(f"report: wrote REPORT_E07B.md and metrics_e07b.json; PRIMARY outcome {P['outcome']}")
    return est


RESOLUTIONS = [
    "A1 'fresh goal classes' = n = 8 full-goal classes that are not goal classes of R-04's E07 test states at n = 8 "
    "(the only n = 8 states E07b evaluates). 11 such classes exist, so '10' is satisfiable. The stricter reading that "
    "also excludes the classes of R-04's E08 n = 8 instances (not used by E07b) leaves only 5 classes and would make "
    "the prereg impossible; it is recorded in data/meta.json as `strict_reading_fresh_classes`.",
    "A1 'largest-available' = highest frequency among discovery draws of the generator (B.generate, full goals) with "
    "seed '20261298:discovery' (50,000 draws; generate()=None and invariant filters as R-04's allocate_classes); a "
    "class must be drawn at least 5 times (R-04's min_class_count); ties by class key. One instance per class (R-04's "
    "unique='goal' scheme), drawn with the spare seed 20261298; 10,000 states per instance sampled with seed "
    "'20261298:states' (R-04's include_init and row-exclusion rules). A1's checkpoint selection uses R-04's val rows.",
    "A2 hinge: for every training row with h* > 0, every pair (optimal successor o, non-optimal successor q) with "
    "relu(0.5 + f(o) - f(q)); mean over a row's pairs, mean over rows with pairs, added to the Huber loss with weight "
    "1. Successors whose (state, goal) is a reserved instance (row-level exclusion) are dropped. One forward pass per "
    "batch over the rows plus their successors.",
    "Paired tests (primary and secondary) use seed 1 of each arm (A0 = 20261276, A1 = 20261292, A2 = 20261295); A3 is "
    "built on A0 seed 1. Seed mean ± sd is descriptive. 'A0 success@2000' in IMPROVED is A0 seed 1.",
    "Remaining ties: λ by smaller λ (least departure from A0); BEST by prereg listing order A1, A2, A3.",
    "VAL-B also runs A0 seed 1 and misplaced_blocks as context; they are never candidates.",
    "Smaller budgets are derived from one 2,000-expansion run per (instance, heuristic) (budget-monotone GBFS; "
    "verified by direct re-runs).",
    "TEST-B/VAL-B are also kept disjoint from R-04's E07 OOD test instances (n > 7), as R-04's E08 was.",
]


# ----------------------------------------------------------------------------- main

def r04_input_shas():
    missing = [f for f in R04_FILES if not (R04_DIR / f).exists()]
    if missing:
        raise SystemExit(f"R-04 inputs missing in {R04_DIR}: {missing}")
    return {f: R.sha_file(R04_DIR / f) for f in R04_FILES}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default runs/RCX-E07/e07b; smoke runs/RCX-E07/e07b_smoke)")
    ap.add_argument("--phase", choices=PHASES + ("all",), default="all")
    ap.add_argument("--smoke", action="store_true", help="tiny configuration for a plumbing check (a few minutes)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    prereg_sha = verify_prereg()
    if not 1 <= a.threads <= 4:
        raise SystemExit("PREREG_E07B: CPU, single lane, <= 4 threads")
    cfg = copy.deepcopy(SMOKE if a.smoke else FULL)
    if cfg["b_primary"] != max(cfg["budgets"]) or cfg["b_low"] not in cfg["budgets"]:
        raise SystemExit("config: b_primary must be the largest budget and b_low one of the budgets")
    out = Path(a.out) if a.out else (OUT_SMOKE if a.smoke else OUT_FULL)
    out = (ROOT / out) if not out.is_absolute() else out
    res = out.resolve()
    forbidden = [BI.BUNDLE_ROOT, ROOT.parent / "bwm", ROOT.parent / "rcwm", R04_DIR]
    if any(res == f.resolve() or f.resolve() in res.parents for f in forbidden):
        raise SystemExit(f"refusing to write to {res} (BWM bundle, ../bwm, ../rcwm or the R-04 run directory)")
    if not a.smoke and res != OUT_FULL.resolve():
        raise SystemExit(f"PREREG_E07B fixes the output directory {OUT_FULL}; use --smoke for other locations")
    if res.exists() and any(res.iterdir()) and not (res / "config.json").exists():
        raise SystemExit(f"{res} exists and is not an E07b run directory; never write into an old directory")
    r04 = json.loads((R04_DIR / "config.json").read_text(encoding="utf-8"))
    if r04["smoke"] or r04["cfg"]["seeds"] != A0_SEEDS:
        raise SystemExit("runs/RCX-E07/main is not the full R-04 run with GNN seeds 20261276-78")
    if not a.smoke and any(r04["cfg"][k] != cfg[k] for k in TRAIN_KEYS):
        raise SystemExit(f"training budget differs from R-04's: {[(k, r04['cfg'][k], cfg[k]) for k in TRAIN_KEYS]}")
    out.mkdir(parents=True, exist_ok=True)
    frozen = {"cfg": cfg, "smoke": a.smoke}
    cpath = out / "config.json"
    if cpath.exists():
        if json.loads(cpath.read_text(encoding="utf-8")) != json.loads(json.dumps(frozen)):
            raise SystemExit(f"{cpath} differs from this invocation's config; use a new --out")
    else:
        R.write_json(cpath, frozen)
    inputs = {"prereg_sha256": prereg_sha, "r04_dir": "runs/RCX-E07/main", "r04_files_sha256": r04_input_shas()}
    ipath = out / "inputs.json"
    if ipath.exists():
        if json.loads(ipath.read_text(encoding="utf-8")) != inputs:
            raise SystemExit(f"R-04 inputs or prereg changed since {ipath} was written; refusing to mix")
    else:
        R.write_json(ipath, inputs)
    ctx = R.Ctx(out, cfg, a.smoke)
    ctx.threads = a.threads
    torch.set_num_threads(a.threads)
    ctx.log(f"run_e07b start phase={a.phase} smoke={a.smoke} threads={a.threads} out={out} prereg OK")
    tpath = out / "timings.json"
    timings = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
    ctx.courtesy(force=True)
    for ph in (PHASES if a.phase == "all" else (a.phase,)):
        t0 = time.perf_counter()
        if ph == "report":
            est = phase_report(ctx, timings)
            if est:
                ctx.log(f"estimate (full design): expected ~{est['total_expected_s'] / 3600:.2f} h, upper "
                        f"~{est['total_upper_s'] / 3600:.2f} h CPU")
        else:
            {"data_b": phase_data_b, "train_b": phase_train_b, "select": phase_select, "e08b": phase_e08b,
             "policy": phase_policy}[ph](ctx)
        dt = time.perf_counter() - t0
        if dt > 1.0 or ph not in timings:
            timings[ph] = round(dt, 2)
        R.write_json(tpath, timings)
    diff = BI.check_no_bytecode()
    if diff["new_pycache_dirs"] or diff["ours"]:
        ctx.log(f"WARNING BWM bytecode check: {diff}")
    ctx.log(f"run_e07b end; guard wait {ctx.guard_wait:.1f}s; timings {timings}")


if __name__ == "__main__":
    main()
