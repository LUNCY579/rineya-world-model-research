"""RCX-E11 runner: reusable skills for Blocksworld (RSM-001 E11; CPU mechanism tier; tier B external specialist).

E11a  macro (option) discovery and reuse. Parameterised macros (length 2-4, goal-annotated) are mined from BWM's 211
      F-verified B3 plans (runs/BW_DISTILL_r1/solutions.jsonl, read-only, sha pinned), validated against exact dynamics
      (rcx/bw/macros.py), and GBFS / beam(4) over primitives (P) is compared with primitives + the mined library (M),
      a matched random sound library (R) and a library mined from shuffled plans (S; same length mix as M, sharing no
      abstract step signature with M; no internal no-op macros in M, R or S) on fresh n = 12/14/15 instances, with
      heuristics A3 (the B4 hook itself: rcx/bw/b4_hook LearnedUnmet, hybrid_lambda 1.0, R-04 GNN seed 20261276) and
      misplaced_blocks. Arm G (primitives + the goal-gated single-move library) is descriptive, not in the decision.
E11b  compositional subgoal policy (rcx/bw/subgoal.py). A GNN subgoal selector is trained on exact labels (cfg
      b_label_target = "achiever": achiever-level sets A_A(s); the optimal-plan sets A(s) are stored and evaluated
      descriptively, and the achiever-level optimum OPT_A is the descriptive ceiling) from fresh small problems
      (n <= 7, at most 2 goal towers of height <= 4) and executed with the same fixed achiever, without search, on
      fresh n = 10-15 goals with more and/or taller towers; compared with bottom-up (BU), random order (RND),
      ready-first (RF; enters only the E11b ADVANCE condition L-r0 vs RF <= 0) and goal-count GBFS.

Design: experiments/RCX-E11/DESIGN.md. Prereg: experiments/RCX-E11/PREREG_E11_DRAFT.md (DRAFT). A full run refuses
to start until the frozen PREREG_E11.md exists and its sha256 equals PREREG_SHA256 below (set at the freeze); --smoke
runs on the draft and writes only to runs/RCX-E11/smoke (or an --out strictly below runs/RCX-E11/; nothing is ever
written elsewhere). Every invocation, including a failed one, appends its BWM __pycache__ diff to
bwm_bytecode_checks.json, and the report states whether all checks were clean.

Usage (from complementary/; CPU only, single lane, <= 4 threads; ThermalGuard on):
  python -B experiments/RCX-E11/run_e11.py --smoke --threads 2     # plumbing check -> runs/RCX-E11/smoke
  python -B experiments/RCX-E11/run_e11.py --threads 4             # full run -> runs/RCX-E11/main (after the freeze)
  python -B experiments/RCX-E11/run_e11.py --phase a_search        # a single phase (resumes)
Phases (--phase, default all, in order): a_data, a_mine, a_search, b_data, b_train, b_eval, report. Finished work is
skipped (phase outputs; one .jsonl record per search/execution run, torn last lines repaired), so re-running the same
command resumes. config.json and inputs.json (sha256 of every data input) are frozen at first use and a mismatch
aborts; code sha256s are recorded at every start (code_shas.json) and a full run aborts if they changed mid-run.
"""
import argparse
import contextlib
import copy
import hashlib
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
    """R-04's runner as a module (utilities, Ctx with ThermalGuard; its main() does not run; nothing is modified)."""
    spec = importlib.util.spec_from_file_location("rcx_run_e07", ROOT / "experiments" / "RCX-E07" / "run_e07.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _import_run_e07()  # imports torch CPU-only (CUDA hidden, torch.cuda patched) and rcx.*

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rcx import stats as ST  # noqa: E402
from rcx.bw import b4_hook as HK  # noqa: E402
from rcx.bw import bwm_import as BI  # noqa: E402
from rcx.bw import data as D  # noqa: E402
from rcx.bw import excl as EX  # noqa: E402
from rcx.bw import heuristics as H  # noqa: E402
from rcx.bw import macros as MA  # noqa: E402
from rcx.bw import search as S  # noqa: E402
from rcx.bw import space as SP  # noqa: E402
from rcx.bw import subgoal as SG  # noqa: E402

# ----------------------------------------------------------------------------- custody constants

PREREG_DRAFT = HERE / "PREREG_E11_DRAFT.md"
PREREG_FROZEN = HERE / "PREREG_E11.md"
PREREG_SHA256 = "c67f036d530ed1f961f84caf2e9b2e598562190ccf1722069379807046b5f40c"  # frozen 2026-10-11 (DECISIONS CX D-31)
OUT_FULL = ROOT / "runs" / "RCX-E11" / "main"
OUT_SMOKE = ROOT / "runs" / "RCX-E11" / "smoke"
R04_DIR = ROOT / "runs" / "RCX-E07" / "main"
E07B_DIR = ROOT / "runs" / "RCX-E07" / "e07b"
DISTILL_DIR = BI.BWM_ROOT / "runs" / "BW_DISTILL_r1"
SOLUTIONS = DISTILL_DIR / "solutions.jsonl"
SOLUTIONS_SHA256 = "fdf80151a52d5728c27d2bb7136f4ea1caa17ba3241487af9553a77dac86244c"  # read 2026-10-11 (DESIGN.md)
DISTILL_META = DISTILL_DIR / "DISTILL_META.json"
DISTILL_OFF_LIMITS = ("distill_rows.jsonl", "mix_v2c.jsonl", "mix_v3.jsonl", "fcache.jsonl")  # never opened
TEACHER_EXPECTED = {"rows": 240, "verified": 211}  # DESIGN section 3; checked after the sha pin (macros.teacher_plans)
RCWM_BUNDLE = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05")
A3_MODEL = R04_DIR / "models" / "gnn_s20261276.pt"

SEED_BLOCK = (20261256, 20261259)  # the RCX E11 block; no other seed is used
SEEDS = {"a_instances": 20261256, "b_instances": 20261257, "b_exec_instances": 20261258, "b_models": 20261259}
DERIVED = {  # string seeds derived from the block (random.Random(str) is deterministic across processes)
    "a_random_macros": "20261256:random_macros", "a_shuffle": "20261256:shuffle",
    "a_mono": "20261256:mono_check", "a_bootstrap": "20261256:bootstrap",
    "b_split": "20261257:{split}", "b_states": "20261257:states:{split}", "b_exec": "20261258:exec",
    "b_random_order": "20261258:random_order:{inst}", "b_bootstrap": "20261258:bootstrap",
}


def derive_seed(s: str) -> int:
    """Integer seed derived from a block seed string (for torch.manual_seed, which needs an int)."""
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big") % (2 ** 31 - 1)


MODEL_SEEDS = [SEEDS["b_models"], derive_seed("20261259:r1"), derive_seed("20261259:r2")]  # r0, r1, r2
assert all(SEED_BLOCK[0] <= v <= SEED_BLOCK[1] for v in SEEDS.values())
assert MODEL_SEEDS[1:] == [816062654, 250692302], MODEL_SEEDS

PHASES = ("a_data", "a_mine", "a_search", "b_data", "b_train", "b_eval", "report")
CODE_FILES = ["experiments/RCX-E11/run_e11.py", "experiments/RCX-E07/run_e07.py", "rcx/stats.py", "rcx/guard.py"] + \
    [f"rcx/bw/{m}.py" for m in ("__init__", "bwm_import", "excl", "space", "data", "encode", "models", "heuristics",
                                "search", "b4_hook", "macros", "subgoal")]

FULL = dict(
    # ---- E11a
    a_sizes=[12, 14, 15], a_per_size=40, a_opt_budget=300_000,
    mine_lengths=[2, 3, 4], mine_max_vars=4, K=8, min_support=5, s_fill_min_support=1, n_check_max=6,
    exclude_noop_macros=True,  # drop candidates with an internal no-op pair from M, R and S (prereg section 2, gate G1)
    # G: descriptive goal-gated single-move control (macros.goal_move_library); not in the decision rule (prereg 2)
    arms=["P", "M", "R", "S", "G"], heuristics=["A3", "misplaced_blocks"], algos=["gbfs", "beam4"], beam_width=4,
    budgets=[50, 200, 500, 2000], primary_budget=200, primary_heur="A3", primary_algo="gbfs",
    a3_model_seed=20261276, a3_lambda=1.0, a3_cache=50_000,
    hcall_thresholds=[500, 1_000, 2_500, 5_000], prim_thresholds=[500, 1_000, 2_500, 5_000],
    compute_secondary_threshold=1_000,  # prereg 'E11a compute-matched secondary' (grades wording only)
    a_g_clause_margin=5,  # prereg E11a interpretation clause: ADVANCE with M - G < 5 at the primary cell
    mono_check_runs=8, mono_check_budget=200,
    # ---- E11b
    b_train_sizes=[4, 5, 6, 7], train_k=2, train_h=4,
    b_per_class={"train": 12, "val": 3, "test_id": 4}, b_ood_small_sizes=[5, 6, 7], b_ood_small_per_class=6,
    b_states={"train": 150, "val": 150, "test_id": 100, "ood_small": 100}, b_max_draws=2000,
    b_exec_sizes=[10, 12, 14, 15], b_exec_strata=["S", "T", "H", "TH"], b_exec_per_cell=8, b_opt_budget=300_000,
    b_calls_per_fact=6, b_steps_per_block=16, gc_budgets=[50, 200, 500, 2000],
    sel_hidden=48, sel_rounds=4, sel_head=64, epochs=10, batch_size=64, lr=2e-3, time_cap_s=1200.0,
    b_select_unseen_val=True,  # epoch selection on val rows whose (state, goal) class is not a training row
    # training target (prereg H-b1): "achiever" = achiever-level set A_A(s) (subgoal.achiever_label_set); the
    # optimal-plan set A(s) ("plan") is stored and evaluated descriptively. Both sets are always stored and evaluated.
    b_label_target="achiever", b_ach_budget=200_000,
    model_seeds=MODEL_SEEDS, b_advance_log_ratio=math.log(0.95), b_success_margin=2, b_stop_success_frac=0.8,
    # ---- statistics
    alpha=0.05, bootstrap_resamples=5000,
)
# SMOKE inherits the frozen settings (exclude_noop_macros, arms incl. G, the compute thresholds, b_label_target) so
# the smoke runs the configuration that will be frozen; at its 200-expansion cap the larger thresholds are censored.
SMOKE = dict(FULL, a_sizes=[12, 14], a_per_size=2, a_opt_budget=20_000, K=3, n_check_max=5,
             budgets=[20, 50, 100, 200], primary_budget=50,
             mono_check_runs=2, mono_check_budget=50,
             b_train_sizes=[4, 5], b_per_class={"train": 2, "val": 1, "test_id": 1}, b_ood_small_sizes=[5, 6],
             b_ood_small_per_class=1, b_states={"train": 30, "val": 30, "test_id": 20, "ood_small": 20},
             b_exec_sizes=[10], b_exec_per_cell=1, b_opt_budget=20_000, gc_budgets=[20, 50, 100, 200],
             epochs=1, time_cap_s=60.0, model_seeds=MODEL_SEEDS[:1], bootstrap_resamples=200)

RUN_KEYS = ("success", "found", "plan_len", "macro_edges", "macro_uses", "expansions", "generated", "h_calls",
            "prim_steps", "macro_children", "macro_bindings", "goal_mid_macro", "sym_mismatch", "reason", "seconds")


# ----------------------------------------------------------------------------- small utilities

@contextlib.contextmanager
def e11_seed_block():
    """data.fresh_instances guards R-04's seed range 20261270-89; E11 draws use the E11 block 20261256-59, so the
    guard is retargeted to that block for the duration of the draw (data.py is unchanged; the E07b pattern)."""
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


def read_records(path, key_fn, budget=None):
    """Resume a .jsonl of run records {key: record}; a torn last line (crash mid-write) is dropped and the file
    rewritten without it. If `budget` is given every record must carry that budget."""
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
        if budget is not None and r.get("budget") != budget:
            raise SystemExit(f"{path}: record with budget {r.get('budget')} != {budget}; use a new --out")
        good.append(line)
        done[key_fn(r)] = r
    return done


def at_budget(rec, b):
    """Outcome at budget b <= rec['budget'] derived from the larger-budget run (gbfs/beam with or without macros are
    budget-monotone: the budget is only compared before each expansion; see macros module doc / run_e07b.at_budget).
    Counters that are unknown at b for a run that went on beyond b are None."""
    if b > rec["budget"]:
        raise ValueError(f"cannot derive budget {b} from a {rec['budget']}-budget run")
    if b == rec["budget"]:
        return rec
    if rec["reason"] in ("goal", "exhausted") and rec["expansions"] <= b:
        return dict(rec, budget=b, derived_from=rec["budget"])
    out = {k: rec[k] for k in ("inst", "n", "arm", "heur", "algo", "method", "stratum", "opt") if k in rec}
    out.update(budget=b, success=False, found=False, plan_len=None, macro_edges=None, expansions=b, generated=None,
               h_calls=None, prim_steps=None, reason="budget", seconds=None, plan=[], subopt=None,
               derived_from=rec["budget"])
    return out


def mcnemar_maps(sb: dict, sc: dict, ids, base, cand, b) -> dict:
    bo = sum(sb[i] and not sc[i] for i in ids)
    co = sum(sc[i] and not sb[i] for i in ids)
    return {"base": base, "cand": cand, "budget": b, "n": len(ids), "base_success": sum(sb[i] for i in ids),
            "cand_success": sum(sc[i] for i in ids), "base_only": bo, "cand_only": co,
            "p_exact": ST.mcnemar_exact(bo, co)}


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def median_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


# ----------------------------------------------------------------------------- A3 heuristic

def make_heuristic(cfg, name, a3=None):
    """misplaced_blocks, or A3 = the B4 hook itself: b4_hook.LearnedUnmet(R-04 gnn_s20261276, hybrid_lambda=1.0),
    single-state forwards at 1 thread (bit-reproducible; exact parity with B4 arm (a); CX decision on E11 Q2).
    a3 = (model, provenance) from HK.load_gnn(A3_MODEL), loaded once per phase; without it the file is loaded here.
    Callers make a NEW hook (empty cache, zero counters) for every search run: values never depend on the cache, but
    a cache shared across the arms of one instance made later arms (M, R, S run after P) look faster in seconds and
    made their evaluations fewer than their h_calls. With a per-run hook, h_calls = hook calls = evaluations (each
    state is scored at most once per run; macros module doc), so h_calls is an exact A3 compute count per run."""
    if name == "misplaced_blocks":
        return H.misplaced_blocks
    if name != "A3":
        raise ValueError(name)
    if a3 is None:
        return HK.LearnedUnmet.from_file(A3_MODEL, hybrid_lambda=cfg["a3_lambda"], cache_size=cfg["a3_cache"])
    model, prov = a3
    return HK.LearnedUnmet(model, hybrid_lambda=cfg["a3_lambda"], cache_size=cfg["a3_cache"], _provenance=prov)


HOOK_COUNTERS = ("calls", "cache_hits", "evaluations", "model_evals", "invalid_states", "fallbacks", "failsafe")


# ----------------------------------------------------------------------------- phase a_data

def prior_instance_keys():
    """Instance keys (n >= 8) of earlier RCX test/selection sets that E11a instances must avoid: R-04 E08 and E07 OOD
    test, E07b TEST-B / VAL-B / A1 (read-only RCX files)."""
    keys, src = set(), {}
    files = [R04_DIR / "e08_instances.json", E07B_DIR / "data" / "testb_instances.json",
             E07B_DIR / "data" / "valb_instances.json", E07B_DIR / "data" / "a1_instances.json"]
    for p in files:
        if not p.exists():
            raise SystemExit(f"prior RCX instance file missing: {p}")
        lst = load_instance_list(p)
        keys |= {inst_key(it) for it in lst if it["n"] >= 8}
        src[str(p.relative_to(ROOT))] = {"sha256": R.sha_file(p), "instances": len(lst)}
    r04 = R.load_instances(R04_DIR)
    keys |= {inst_key(it) for it in r04.values() if it["n"] >= 8}
    src["runs/RCX-E07/main/data/instances.json"] = {"sha256": R.sha_file(R04_DIR / "data" / "instances.json"),
                                                    "instances": len(r04)}
    return keys, src


def a3_training_goal_classes():
    """Goal classes (data.goal_key) of R-04's train and val splits: the data behind the A3 GNN (gnn_s20261276)."""
    return {it["goal_key"] for it in R.load_instances(R04_DIR).values() if it["split"] in ("train", "val")}


def prior_and_e11a_keys(ctx):
    """(keys, sources) of every earlier RCX test/selection instance (n >= 8) plus this run's E11a instances (if
    a_data has run), for the E11b execution-set disjointness check."""
    keys, src = prior_instance_keys()
    ap = ctx.out / "a" / "instances.json"
    src = {"prior_rcx": src, "e11a_checked": ap.exists()}
    if ap.exists():
        keys = keys | {inst_key(it) for it in load_instance_list(ap)}
    return keys, src


def phase_a_data(ctx):
    cfg, ad = ctx.cfg, ctx.out / "a"
    if (ad / "meta.json").exists():
        ctx.log("a_data: outputs exist, skipping")
        return
    ad.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ex = EX.with_selection_sets(EX.load_exclusions())
    prior, prior_src = prior_instance_keys()
    sizes = [n for n in cfg["a_sizes"] for _ in range(cfg["a_per_size"])]
    with e11_seed_block():
        inst, gen = D.fresh_instances(SEEDS["a_instances"], sizes, full_goal_frac=1.0, seen=set(prior), ex=ex,
                                      prefix="e11a")
    keys = {inst_key(it) for it in inst}
    checks = {"unique": len(keys) == len(inst), "x_prior_rcx_sets": len(keys & prior),
              "x_a3_train_val_goal_classes": len({it["goal_key"] for it in inst} & a3_training_goal_classes()),
              "excluded_now": sum(EX.exclusion_reason(it["objs"], it["init"], it["goal"], ex) is not None
                                  for it in inst),
              "goals_all_on_facts": all(f[0] == "on" for it in inst for f in it["goal"]),
              "sizes": dict(sorted(Counter(it["n"] for it in inst).items()))}
    if (not checks["unique"] or checks["x_prior_rcx_sets"] or checks["x_a3_train_val_goal_classes"]
            or checks["excluded_now"] or not checks["goals_all_on_facts"]
            or checks["sizes"] != {n: cfg["a_per_size"] for n in sorted(cfg["a_sizes"])}):
        raise RuntimeError(f"E11a instance checks failed: {checks}")
    for it in inst:
        t1 = time.perf_counter()
        h, exp = SP.astar_hstar(it["objs"], it["init"], it["goal"], cfg["a_opt_budget"])
        it.update(opt=h, opt_expansions=exp, opt_s=round(time.perf_counter() - t1, 4))
        ctx.courtesy()
    save_instances(ad / "instances.json", inst)
    meta = {"instances": len(inst), "generation": gen, "checks": checks, "prior_sources": prior_src,
            "opt_coverage": {str(n): f"{sum(1 for it in inst if it['n'] == n and it['opt'] is not None)}/"
                                      f"{sum(1 for it in inst if it['n'] == n)}" for n in cfg["a_sizes"]},
            "exclusions_sha256": R.sha_file(EX.EXCLUSIONS_PATH), "selection_sources": ex["selection_sources"],
            "seed": SEEDS["a_instances"], "wall_s": round(time.perf_counter() - t0, 2)}
    R.write_json(ad / "meta.json", meta)
    ctx.log(f"a_data: {len(inst)} instances; A* coverage {meta['opt_coverage']}; {meta['wall_s']}s")


# ----------------------------------------------------------------------------- phase a_mine

def load_teacher_plans(ctx):
    """The 211 verified plans: sha-pinned read of solutions.jsonl only; every plan re-verified with B.execute and its
    F-predicted trajectory compared with the exact one (custody statistics; macros.teacher_plans). Fails closed on a
    sha mismatch or on custody counts other than TEACHER_EXPECTED."""
    B, _ = BI.load()
    if SOLUTIONS.name in DISTILL_OFF_LIMITS or SOLUTIONS.name != "solutions.jsonl":  # not an assert (python -O)
        raise SystemExit(f"teacher file {SOLUTIONS} is not the allowed solutions.jsonl")
    raw = SOLUTIONS.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != SOLUTIONS_SHA256:
        raise SystemExit(f"solutions.jsonl sha256 {sha} != pinned {SOLUTIONS_SHA256}; refusing to mine")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    try:
        plans, cust = MA.teacher_plans(rows, B, expected=TEACHER_EXPECTED)
    except MA.CustodyError as e:
        raise SystemExit(f"teacher custody: {e}; refusing to mine") from None
    return plans, {"sha256": sha, "path": str(SOLUTIONS), "counts": dict(cust),
                   "by_n": dict(sorted(Counter(p["n"] for p in plans).items())),
                   "distill_meta_sha256": R.sha_file(DISTILL_META)}


def _cand_json(c):
    return {k: c[k] for k in ("key", "L", "nvars", "occurrences", "support", "score")}


def describe_lib(lib):
    """Per macro: size, mining provenance, internal no-op pairs (macros.noop_pairs) and how its goal gating relates to
    its own effect (macros.annotation_alignment: established / kept / destroyed / pending / unrelated literals)."""
    keys = ("occurrences", "support", "tier", "rank")
    return [{"key": m.key, "L": m.L, "V": m.nvars, "G": len(m.goal_on), **{k: m.meta.get(k) for k in keys},
             "noop_pairs": MA.noop_pairs(m.steps), "alignment": MA.annotation_alignment(m)} for m in lib]


def alignment_summary(lib) -> dict:
    """Library totals of annotation_alignment classes and of macros with internal no-op pairs."""
    al = [MA.annotation_alignment(m) for m in lib]
    out = {c: sum(a[c] for a in al) for c in MA.ALIGN_CLASSES + ("aligned",)}
    out.update(literals=sum(len(m.goal_on) for m in lib), noop_macros=sum(bool(MA.noop_pairs(m.steps)) for m in lib))
    return out


def phase_a_mine(ctx):
    cfg, ad = ctx.cfg, ctx.out / "a"
    path = ad / "libraries.json"
    if path.exists():
        ctx.log("a_mine: outputs exist, skipping")
        return
    ad.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    plans, custody = load_teacher_plans(ctx)
    ctx.log(f"a_mine: teacher custody {custody['counts']}")
    vc = MA.ValidationCache(n_check_max=cfg["n_check_max"], courtesy=ctx.courtesy)
    K, ms = cfg["K"], cfg["min_support"]
    table, mstats = MA.mine(plans, tuple(cfg["mine_lengths"]), cfg["mine_max_vars"])
    ranked = MA.rank(table)
    xn = cfg["exclude_noop_macros"]  # drop candidates with an internal no-op pair from M, R and S (macros.noop_pairs)
    lib_m, log_m = MA.select_library(ranked, K, ms, vc, exclude_noop=xn)
    if len(lib_m) < K:
        raise SystemExit(f"gate G1: only {len(lib_m)} accepted mined macros (K = {K})")
    lib_r, log_r = MA.random_library(lib_m, random.Random(DERIVED["a_random_macros"]), vc,
                                     exclude_keys={m.key for m in lib_m},
                                     exclude_step_keys={m.step_key for m in lib_m}, max_vars=cfg["mine_max_vars"],
                                     exclude_noop=xn)
    splans = MA.shuffled_plans(plans, random.Random(DERIVED["a_shuffle"]))
    stable, sstats = MA.mine(splans, tuple(cfg["mine_lengths"]), cfg["mine_max_vars"])
    sranked = MA.rank(stable)
    lib_s, log_s = MA.select_matched_library(sranked, lib_m, ms, vc, exclude_step_keys={m.step_key for m in lib_m},
                                             fill_min_support=cfg["s_fill_min_support"], exclude_noop=xn)
    libs = {"M": lib_m, "R": lib_r, "S": lib_s}
    if "G" in cfg["arms"]:  # descriptive goal-gated single-move control (macros.goal_move_library); not mined
        libs["G"] = MA.goal_move_library()
    if len(lib_s) != K or {m.key for m in lib_m} & {m.key for m in lib_s} or \
            {m.step_key for m in lib_m} & {m.step_key for m in lib_s}:
        raise RuntimeError("gate G1: the S library must have K macros and share no macro or step signature with M")
    if xn and any(MA.noop_pairs(m.steps) for a in ("M", "R", "S") for m in libs[a]):
        raise RuntimeError("gate G1: a library macro has an internal no-op pair although exclude_noop_macros is set")
    validation = {m.step_key: vc(m) for lib in libs.values() for m in lib}
    if not all(v["accepted"] and "shortcut" not in v for v in validation.values()):
        raise RuntimeError("a library macro was not exhaustively accepted")
    out = {"K": K, "libraries": {a: MA.library_json(lib) for a, lib in libs.items()},
           "descriptors": {a: describe_lib(lib) for a, lib in libs.items()},
           "alignment": {a: alignment_summary(lib) for a, lib in libs.items()}, "exclude_noop_macros": xn,
           "selection_logs": {"M": log_m, "R": log_r, "S": log_s},
           "top_candidates": {"M": [_cand_json(c) for c in ranked[:40]], "S": [_cand_json(c) for c in sranked[:40]]},
           "mining_stats": {"M": mstats, "S": sstats}, "validation": validation,
           "validation_summary": vc.summary(), "custody": custody,
           "overlap": {"M_S_keys": sorted({m.key for m in lib_m} & {m.key for m in lib_s}),
                       "M_S_step_patterns": sorted({m.step_key for m in lib_m} & {m.step_key for m in lib_s}),
                       "M_R_step_patterns": sorted({m.step_key for m in lib_m} & {m.step_key for m in lib_r})},
           "seeds": {"random": DERIVED["a_random_macros"], "shuffle": DERIVED["a_shuffle"]},
           "wall_s": round(time.perf_counter() - t0, 2)}
    R.write_json(path, out)
    for a, lib in libs.items():
        ctx.log(f"a_mine: {a} ({len(lib)}): " + " || ".join(m.key for m in lib))
    ctx.log(f"a_mine: validation {out['validation_summary']}; {out['wall_s']}s")


def load_libraries(ctx):
    raw = json.loads((ctx.out / "a" / "libraries.json").read_text(encoding="utf-8"))
    libs = {a: MA.library_from_json(rows) for a, rows in raw["libraries"].items()}
    libs["P"] = []
    return libs, raw


# ----------------------------------------------------------------------------- phase a_search

def a_record(cfg, it, arm, heur, algo, lib, h, budget):
    if algo == "gbfs":
        r = MA.gbfs_macro(it["objs"], it["init"], it["goal"], h, lib, max_expansions=budget)
    elif algo == "beam4":
        r = MA.beam_macro(it["objs"], it["init"], it["goal"], h, lib, width=cfg["beam_width"], max_expansions=budget)
    else:
        raise ValueError(algo)
    rec = {"inst": it["id"], "n": it["n"], "arm": arm, "heur": heur, "algo": algo, "budget": budget,
           "opt": it.get("opt"), **{k: r[k] for k in RUN_KEYS}, "plan": [" ".join(a) for a in r["plan"]]}
    rec["subopt"] = (rec["plan_len"] / rec["opt"]) if rec["success"] and rec["opt"] else None
    if isinstance(h, HK.LearnedUnmet):  # per-run hook (make_heuristic): its counters are this run's A3 compute
        rec["hook"] = {k: h.counters.get(k, 0) for k in HOOK_COUNTERS}
    return rec


def a_key(r):
    return (r["inst"], r["arm"], r["heur"], r["algo"])


def phase_a_search(ctx):
    cfg, ad = ctx.cfg, ctx.out / "a"
    inst = load_instance_list(ad / "instances.json")
    by_id = {it["id"]: it for it in inst}
    libs, _ = load_libraries(ctx)
    if not (ad / "a3_hook.json").exists():
        R.write_json(ad / "a3_hook.json", make_heuristic(cfg, "A3").describe())
    bmax = max(cfg["budgets"])
    path = ad / "runs.jsonl"
    done = read_records(path, a_key, budget=bmax)
    todo = [(it, heur, arm, algo) for it in inst for heur in cfg["heuristics"] for arm in cfg["arms"]
            for algo in cfg["algos"] if (it["id"], arm, heur, algo) not in done]
    ctx.log(f"a_search: {len(todo)} runs to do, {len(done)} done (budget {bmax})")
    t0 = time.perf_counter()
    a3 = HK.load_gnn(A3_MODEL) if "A3" in cfg["heuristics"] and todo else None  # weights loaded once
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for k, (it, heur, arm, algo) in enumerate(todo):
            h = make_heuristic(cfg, heur, a3)  # a fresh hook (empty cache, zero counters) per run: fair seconds
            rec = a_record(cfg, it, arm, heur, algo, libs[arm], h, bmax)
            done[a_key(rec)] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ctx.courtesy()
            if (k + 1) % 50 == 0:
                ctx.log(f"a_search: {k + 1}/{len(todo)} runs in {time.perf_counter() - t0:.0f}s")
    mpath = ad / "mono_check.json"
    if not mpath.exists():
        rng = random.Random(DERIVED["a_mono"])
        keys = sorted(done)
        pick = rng.sample(keys, min(cfg["mono_check_runs"], len(keys)))
        b = cfg["mono_check_budget"]
        rows = []
        for key in pick:
            rec = done[key]
            r = a_record(cfg, by_id[key[0]], key[1], key[2], key[3], libs[key[1]], make_heuristic(cfg, key[2]), b)
            d = at_budget(rec, b)
            same = all(r[f] == d[f] for f in ("success", "found", "plan_len", "expansions", "reason"))
            rows.append({"key": list(key), "budget": b, "same": same, "direct": {f: r[f] for f in ("success",
                         "plan_len", "expansions", "reason")}, "derived": {f: d[f] for f in ("success", "plan_len",
                                                                                       "expansions", "reason")}})
        R.write_json(mpath, {"runs": rows, "all_same": all(x["same"] for x in rows)})
        ctx.log(f"a_search: monotonicity spot-check all_same={all(x['same'] for x in rows)} ({len(rows)} runs)")


# ----------------------------------------------------------------------------- phase b_data

def b_draw(ctx, rng, classes, per_class, seen, ex, split, max_draws):
    """Fresh E11b instances: per class (n, heights), per_class instances; BWM exclusion recipe; unique across all
    E11b sets (renaming signature for n <= 7, raw key above)."""
    B, _ = BI.load()
    out, st = [], Counter()
    for n, heights in classes:
        hs = ",".join(map(str, heights))
        for k in range(per_class):
            for _ in range(max_draws):
                g = SG.make_instance(rng, n, heights)
                st["draws"] += 1
                if g is None:
                    st["goal_met_at_init"] += 1
                    continue
                objs, init, goal = g
                sig = B.signature(objs, init, goal) if n <= 7 else None
                why = EX.exclusion_reason(objs, init, goal, ex, sig=sig)
                if why:
                    st[f"excluded_{why}"] += 1
                    continue
                key = D.instance_key(objs, init, goal, sig)
                if key in seen:
                    st["duplicate"] += 1
                    continue
                seen.add(key)
                out.append({"id": f"e11b-{split}/{n}/{hs}/{k}", "n": n, "objs": objs, "init": init, "goal": goal,
                            "heights": list(heights), "stratum": SG.stratum_of(heights, ctx.cfg["train_k"],
                                                                                ctx.cfg["train_h"]),
                            "split": split})
                break
            else:
                st["exhausted"] += 1
            ctx.courtesy()
    st["kept"] = len(out)
    return out, dict(st)


def b_rows(ctx, split, insts, ex, k_states, earlier_keys=frozenset()):
    """Sampled hand-empty rows with exact label sets. Every row carries `seen` = its renaming-canonical (state, goal)
    key (subgoal.canonical_row_key) occurs in an earlier split (`earlier_keys`; order train, val, test_id,
    ood_small). Instances of one goal class share their state space up to renaming, so instance-level uniqueness does
    not separate rows; selection and in-distribution accuracy use unseen rows. Every row stores both exact label
    sets (subgoal.row_label_sets): `labels_plan` = A(s), `labels_ach` = A_A(s); `labels` (the training target and
    the primary label-accuracy basis) is the one named by cfg['b_label_target']. Returns (rows, stats, keys)."""
    target, ach_budget = ctx.cfg["b_label_target"], ctx.cfg["b_ach_budget"]
    if target not in SG.LABEL_TARGETS:
        raise ValueError(f"b_label_target {target!r}")
    rng = random.Random(DERIVED["b_states"].format(split=split))
    rows, st, keys = [], Counter(), set()
    for it in insts:
        L = SP.GoalLabels(it["objs"], it["goal"])
        vcache = {}  # V_A per state for this instance's goal (achiever_q cache)
        ids = SG.handempty_ids(L)
        pick = list(ids) if k_states >= len(ids) else rng.sample(ids, k_states)
        i0 = L.sid(it["init"])
        if pick and i0 not in pick:
            pick[0] = i0
        for i in pick:
            s = L.state(i)
            why = EX.row_exclusion_reason(it["objs"], s, it["goal"], ex)
            if why:
                st[f"rows_excluded_{why}"] += 1
                continue
            cands = SG.unmet_facts(s, it["goal"])
            ls = SG.row_label_sets(L, i, it["objs"], it["goal"], ach_budget, vcache)
            labels = ls[target]
            ck = SG.canonical_row_key(s, it["goal"], it["objs"])
            seen = ck in earlier_keys
            keys.add(ck)
            rows.append({"i": it["id"], "s": R.fs_json(s), "cands": [list(f) for f in cands],
                         "labels": sorted(list(f) for f in labels),
                         "labels_plan": sorted(list(f) for f in ls["plan"]),
                         "labels_ach": sorted(list(f) for f in ls["achiever"]), "seen": seen})
            st["rows"] += 1
            st[f"rows_n{it['n']}"] += 1
            st["rows_seen"] += seen
            st[f"rows_seen_n{it['n']}"] += seen
            st["label_size_sum"] += len(labels)
            st["cands_sum"] += len(cands)
            st["single_label_rows"] += len(labels) == 1
            st["plan_eq_ach_rows"] += ls["plan"] == ls["achiever"]
            st["plan_ach_disjoint_rows"] += not (ls["plan"] & ls["achiever"])
        ctx.courtesy()
    return rows, dict(sorted(st.items())), keys


def phase_b_data(ctx):
    cfg, bd = ctx.cfg, ctx.out / "b" / "data"
    if (bd / "meta.json").exists():
        ctx.log("b_data: outputs exist, skipping")
        return
    bd.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ex = EX.with_selection_sets(EX.load_exclusions())
    tk, th = cfg["train_k"], cfg["train_h"]
    id_classes = [(n, p) for n in cfg["b_train_sizes"] for p in SG.partitions_in(n, "S", tk, th)]
    ood_classes = [(n, p) for n in cfg["b_ood_small_sizes"] for st in ("T", "H", "TH")
                   for p in SG.partitions_in(n, st, tk, th)]
    seen, splits, gen = set(), {}, {}
    for split in ("train", "val", "test_id"):
        rng = random.Random(DERIVED["b_split"].format(split=split))
        splits[split], gen[split] = b_draw(ctx, rng, id_classes, cfg["b_per_class"][split], seen, ex, split,
                                           cfg["b_max_draws"])
    rng = random.Random(DERIVED["b_split"].format(split="ood_small"))
    splits["ood_small"], gen["ood_small"] = b_draw(ctx, rng, ood_classes, cfg["b_ood_small_per_class"], seen, ex,
                                                   "ood_small", cfg["b_max_draws"])
    rows_meta, row_keys = {}, set()
    for split in ("train", "val", "test_id", "ood_small"):  # this order defines `seen` (earlier splits)
        insts = splits[split]
        rows, st, keys = b_rows(ctx, split, insts, ex, cfg["b_states"][split], frozenset(row_keys))
        row_keys |= keys
        R.write_jsonl_gz(bd / f"{split}.jsonl.gz", rows)
        rows_meta[split] = st
        ctx.log(f"b_data: {split}: {len(insts)} instances, {len(rows)} rows {st}")
    SP.get_space.cache_clear()
    # execution instances (n = 10-15), stratified by goal structure
    rng = random.Random(DERIVED["b_exec"])
    B, _ = BI.load()
    execs, est = [], Counter()
    for stratum in cfg["b_exec_strata"]:
        for n in cfg["b_exec_sizes"]:
            parts = SG.partitions_in(n, stratum, tk, th)
            for k in range(cfg["b_exec_per_cell"]):
                for _ in range(cfg["b_max_draws"]):
                    heights = parts[rng.randrange(len(parts))]
                    g = SG.make_instance(rng, n, heights)
                    est["draws"] += 1
                    if g is None:
                        est["goal_met_at_init"] += 1
                        continue
                    objs, init, goal = g
                    why = EX.exclusion_reason(objs, init, goal, ex)
                    if why:
                        est[f"excluded_{why}"] += 1
                        continue
                    key = D.instance_key(objs, init, goal)
                    if key in seen:
                        est["duplicate"] += 1
                        continue
                    seen.add(key)
                    execs.append({"id": f"e11b-exec/{stratum}/{n}/{k}", "n": n, "objs": objs, "init": init,
                                  "goal": goal, "heights": list(heights), "stratum": stratum, "split": "exec"})
                    break
                else:
                    raise RuntimeError(f"no exec instance for {stratum} n={n}")
    for it in execs:
        t1 = time.perf_counter()
        h, exp = SP.astar_hstar(it["objs"], it["init"], it["goal"], cfg["b_opt_budget"])
        it.update(opt=h, opt_expansions=exp, opt_s=round(time.perf_counter() - t1, 4))
        # exact achiever-level optimum OPT_A (descriptive ceiling of every selector under this achiever)
        t1 = time.perf_counter()
        oa = SG.achiever_search(it["objs"], it["init"], it["goal"], cfg["b_ach_budget"])
        if oa["value"] is not None and (len(oa["plan"]) != oa["value"] or not B.execute(
                it["init"], it["goal"], [tuple(a) for a in oa["plan"]])["success"]
                or (it["opt"] is not None and oa["value"] < it["opt"])):
            raise RuntimeError(f"OPT_A check failed for {it['id']}: value {oa['value']}, opt {it['opt']}")
        it.update(opt_ach=oa["value"], opt_ach_expansions=oa["expansions"],
                  opt_ach_s=round(time.perf_counter() - t1, 4))
        ctx.courtesy()
    allinst = [it for split in ("train", "val", "test_id", "ood_small") for it in splits[split]] + execs
    id_keys = {",".join(map(str, p)) + f"@{n}" for n, p in id_classes}
    ood_keys = {",".join(map(str, p)) + f"@{n}" for n, p in ood_classes}
    prior_keys, prior_src = prior_and_e11a_keys(ctx)
    checks = {"ood_small_x_train_classes": len(id_keys & ood_keys),
              # strata recomputed from each goal itself (not from the declared heights)
              "structure_matches_declared": all(SG.structure(it["goal"], it["objs"])["heights"] == list(it["heights"])
                                                for it in allinst),
              "exec_strata_ok": all(SG.stratum_of(SG.structure(it["goal"], it["objs"])["heights"], tk, th)
                                    == it["stratum"] for it in execs),
              "train_strata_all_S": all(SG.stratum_of(SG.structure(it["goal"], it["objs"])["heights"], tk, th) == "S"
                                        for s in ("train", "val", "test_id") for it in splits[s]),
              "ood_small_strata": dict(Counter(it["stratum"] for it in splits["ood_small"])),
              "ood_small_all_non_S": all(SG.stratum_of(SG.structure(it["goal"], it["objs"])["heights"], tk, th) != "S"
                                         for it in splits["ood_small"]),
              "ood_small_rows_seen_in_id_splits": rows_meta["ood_small"].get("rows_seen", 0),
              "exec_x_prior_rcx_and_e11a": len({D.instance_key(it["objs"], it["init"], it["goal"]) for it in execs}
                                               & prior_keys),
              "exec_prior_sources": prior_src,
              "unique_ids": len({it["id"] for it in allinst}) == len(allinst)}
    if checks["ood_small_x_train_classes"] or checks["ood_small_rows_seen_in_id_splits"] or \
            checks["exec_x_prior_rcx_and_e11a"] or not (
                checks["structure_matches_declared"] and checks["exec_strata_ok"] and checks["train_strata_all_S"]
                and checks["ood_small_all_non_S"] and checks["unique_ids"]):
        raise RuntimeError(f"E11b data checks failed: {checks}")
    save_instances(bd / "instances.json", allinst)
    cover = Counter((it["stratum"], it["n"]) for it in execs if it["opt"] is not None)
    cells = Counter((it["stratum"], it["n"]) for it in execs)
    meta = {"classes": {"id": [list(p) + [n] for n, p in id_classes], "ood_small": [list(p) + [n] for n, p in
                                                                                     ood_classes]},
            "instances": {s: len(v) for s, v in splits.items()} | {"exec": len(execs)}, "generation": gen,
            "exec_generation": dict(est), "rows": rows_meta, "checks": checks,
            "exec_opt_coverage": {f"{st}@{n}": f"{cover[(st, n)]}/{cells[(st, n)]}" for st in cfg["b_exec_strata"]
                                  for n in cfg["b_exec_sizes"]},
            "exclusions_sha256": R.sha_file(EX.EXCLUSIONS_PATH), "seeds": {k: v for k, v in DERIVED.items()
                                                                           if k.startswith("b_")},
            "wall_s": round(time.perf_counter() - t0, 2)}
    R.write_json(bd / "meta.json", meta)
    ctx.log(f"b_data: done in {meta['wall_s']}s; exec A* coverage {meta['exec_opt_coverage']}")


# ----------------------------------------------------------------------------- phase b_train

def b_instances(ctx):
    return {it["id"]: it for it in load_instance_list(ctx.out / "b" / "data" / "instances.json")}


def b_encode(rows, insts):
    out = []
    for r in rows:
        it = insts[r["i"]]
        out.append(SG.encode_row(R.fs_from(r["s"]), it["goal"], it["objs"], [tuple(f) for f in r["cands"]],
                                 [tuple(f) for f in r["labels"]]))
    return out


def sel_config(cfg):
    return SG.SubgoalNetConfig(hidden=cfg["sel_hidden"], rounds=cfg["sel_rounds"], head_hidden=cfg["sel_head"])


def phase_b_train(ctx):
    cfg = ctx.cfg
    md = ctx.out / "b" / "models"
    md.mkdir(parents=True, exist_ok=True)
    todo = [s for s in cfg["model_seeds"] if not (md / f"sel_s{s}.pt").exists()]
    if not todo:
        ctx.log("b_train: all selector models exist, skipping")
        return
    insts = b_instances(ctx)
    dd = ctx.out / "b" / "data"
    xtr = b_encode(R.read_jsonl_gz(dd / "train.jsonl.gz"), insts)
    val_rows = R.read_jsonl_gz(dd / "val.jsonl.gz")
    basis = "val_unseen_in_train" if cfg["b_select_unseen_val"] else "val_all"
    xva = b_encode([r for r in val_rows if not (cfg["b_select_unseen_val"] and r["seen"])], insts)
    if not xva:
        if not ctx.smoke:
            raise SystemExit("b_train: no validation rows unseen in training (selection would use training rows)")
        basis, xva = "val_all_smoke_fallback", b_encode(val_rows, insts)
    ctx.log(f"b_train: {len(xtr)} train rows, {len(xva)} of {len(val_rows)} val rows for selection ({basis}); "
            f"seeds {todo}")
    for seed in todo:
        model = SG.new_net(sel_config(cfg), seed=seed)
        info = SG.fit(model, xtr, xva, epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"], seed=seed,
                      time_cap_s=cfg["time_cap_s"], log=ctx.log, courtesy=ctx.courtesy)
        p = md / f"sel_s{seed}.pt"
        tmp = Path(str(p) + ".tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, p)
        R.write_json(md / f"sel_s{seed}.json", dict(info, threads=ctx.threads, rows_train=len(xtr),
                                                    rows_val=len(xva), rows_val_total=len(val_rows),
                                                    selection_basis=basis))


def load_selector(ctx, seed):
    model = SG.new_net(sel_config(ctx.cfg))
    model.load_state_dict(torch.load(ctx.out / "b" / "models" / f"sel_s{seed}.pt", weights_only=True))
    return model.eval()


# ----------------------------------------------------------------------------- phase b_eval

def b_methods(cfg):
    return [f"learned_s{s}" for s in cfg["model_seeds"]] + ["bottom_up", "ready_first", "random_order", "gc_gbfs"]


def b_key(r):
    return (r["inst"], r["method"])


def b_exec_record(cfg, it, method, sel):
    base = {"inst": it["id"], "n": it["n"], "stratum": it["stratum"], "method": method, "opt": it.get("opt")}
    if method == "gc_gbfs":
        b = max(cfg["gc_budgets"])
        r = S.gbfs(it["objs"], it["init"], it["goal"], H.unmet_goal_facts, max_expansions=b)
        rec = dict(base, budget=b, **{k: r[k] for k in ("success", "found", "plan_len", "expansions", "generated",
                                                          "h_calls", "reason", "seconds")},
                   plan=[" ".join(a) for a in r["plan"]])
    else:
        max_calls = cfg["b_calls_per_fact"] * len(it["goal"])
        max_steps = cfg["b_steps_per_block"] * it["n"]
        r = SG.execute(it["objs"], it["init"], it["goal"], sel, max_calls=max_calls, max_steps=max_steps)
        rec = dict(base, max_calls=max_calls, max_steps=max_steps, **r)
    rec["subopt"] = (rec["plan_len"] / rec["opt"]) if rec["success"] and rec["opt"] else None
    return rec


def phase_b_eval(ctx):
    cfg = ctx.cfg
    bdir = ctx.out / "b"
    insts = b_instances(ctx)
    models = {s: load_selector(ctx, s) for s in cfg["model_seeds"]}
    lpath = bdir / "label_eval.json"
    if not lpath.exists():
        res = {}
        for split in ("test_id", "ood_small"):
            rows = R.read_jsonl_gz(bdir / "data" / f"{split}.jsonl.gz")
            enc = b_encode(rows, insts)
            groups = defaultdict(list)
            for k, r in enumerate(rows):
                it = insts[r["i"]]
                gs = ["all", f"stratum:{it['stratum']}", f"n:{it['n']}",
                      "class:" + ",".join(map(str, it["heights"])) + f"@{it['n']}"]
                # `seen` (KeyError on rows without the flag: fail closed): the row's renaming class occurs in an
                # earlier split. "unseen" is the clean in-distribution accuracy (test_id rows not in train or val).
                if not r["seen"]:
                    gs += ["unseen", f"unseen_n:{it['n']}", f"unseen_stratum:{it['stratum']}"]
                for g in gs:
                    groups[g].append(k)
            picks = {}  # method -> chosen candidate fact (as a list) per row
            for s, model in models.items():
                ch = SG.predict_choice(model, enc)
                picks[f"learned_s{s}"] = [r["cands"][c] for c, r in zip(ch, rows)]
            bu, rf = SG.BottomUp(), SG.ReadyFirst()
            for name in ("bottom_up", "ready_first", "random_order"):
                cc = []
                for r in rows:
                    it = insts[r["i"]]
                    s_ = R.fs_from(r["s"])
                    cands = [tuple(f) for f in r["cands"]]
                    sel = {"bottom_up": bu, "ready_first": rf}.get(name) or SG.RandomOrder(
                        it["goal"], DERIVED["b_random_order"].format(inst=it["id"]))
                    cc.append(list(sel(s_, it["goal"], it["objs"], cands)))
                picks[name] = cc
                ctx.courtesy()
            # `m`: top-1 in the training-target set (cfg b_label_target); `m|vs_plan` / `m|vs_achiever`: in A(s) /
            # A_A(s) (KeyError on rows without both sets: fail closed)
            correct = {}
            for name, ch in picks.items():
                correct[name] = [c in r["labels"] for c, r in zip(ch, rows)]
                correct[f"{name}|vs_plan"] = [c in r["labels_plan"] for c, r in zip(ch, rows)]
                correct[f"{name}|vs_achiever"] = [c in r["labels_ach"] for c, r in zip(ch, rows)]
            res[split] = {g: {"rows": len(ix), **{m: float(np.mean([v[i] for i in ix])) for m, v in correct.items()}}
                          for g, ix in sorted(groups.items())}
        R.write_json(lpath, res)
        ctx.log(f"b_eval: label accuracy {json.dumps({s: {g: res[s].get(g) for g in ('all', 'unseen')} for s in res})}")
    execs = [it for it in insts.values() if it["split"] == "exec"]
    path = bdir / "exec_runs.jsonl"
    done = read_records(path, b_key)
    sels = {f"learned_s{s}": SG.LearnedSelector(m) for s, m in models.items()}
    sels.update(bottom_up=SG.BottomUp(), ready_first=SG.ReadyFirst())
    todo = [(it, m) for it in execs for m in b_methods(cfg) if (it["id"], m) not in done]
    ctx.log(f"b_eval: {len(todo)} execution runs to do, {len(done)} done")
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        for it, m in todo:
            sel = sels.get(m)
            if m == "random_order":
                sel = SG.RandomOrder(it["goal"], DERIVED["b_random_order"].format(inst=it["id"]))
            rec = b_exec_record(cfg, it, m, sel)
            done[b_key(rec)] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ctx.courtesy()


# ----------------------------------------------------------------------------- report: E11a metrics and decision

def a_success(done, ids, arm, heur, algo, b):
    return {i: bool(at_budget(done[(i, arm, heur, algo)], b)["success"]) for i in ids}


def a_cell(done, ids, arm, heur, algo, b):
    recs = [at_budget(done[(i, arm, heur, algo)], b) for i in ids]
    succ = [r for r in recs if r["success"]]
    k = len(succ)
    full = [r for r in recs if "derived_from" not in r or r["reason"] != "budget"]
    return {"n": len(recs), "success": k, "rate": k / len(recs) if recs else None, "wilson": ST.wilson(k, len(recs)),
            "expansions_median": median_or_none([r["expansions"] for r in recs]),
            "expansions_median_solved": median_or_none([r["expansions"] for r in succ]),
            "h_calls_mean_solved": mean_or_none([r["h_calls"] for r in succ]),
            "prim_steps_mean_solved": mean_or_none([r["prim_steps"] for r in succ]),
            "children_per_expansion": mean_or_none([r["generated"] / max(1, r["expansions"]) for r in full
                                                    if r.get("generated") is not None]),
            "plan_len_mean": mean_or_none([r["plan_len"] for r in succ]),
            "subopt_mean": mean_or_none([r["subopt"] for r in succ]), "subopt_n": sum(r["subopt"] is not None
                                                                                       for r in succ),
            "macro_edges_mean": mean_or_none([r.get("macro_edges") for r in succ]),
            "invalid_found_plans": sum(1 for r in recs if r["found"] and not r["success"]),
            "seconds_total": float(sum(r["seconds"] for r in recs if r.get("seconds") is not None and
                                       "derived_from" not in r))}


def a_metrics(cfg, inst, done, libs_raw):
    ids = [it["id"] for it in inst]
    missing = [(i, a, h, g) for i in ids for a in cfg["arms"] for h in cfg["heuristics"] for g in cfg["algos"]
               if (i, a, h, g) not in done]
    if missing:
        return {"complete": False, "missing_runs": len(missing)}
    cells, per_n = {}, {}
    for arm in cfg["arms"]:
        for heur in cfg["heuristics"]:
            for algo in cfg["algos"]:
                for b in cfg["budgets"]:
                    name = f"{arm}|{heur}|{algo}|{b}"
                    cells[name] = a_cell(done, ids, arm, heur, algo, b)
                    per_n[name] = {str(n): a_success_count(done, [it["id"] for it in inst if it["n"] == n], arm,
                                                           heur, algo, b) for n in cfg["a_sizes"]}
    pb, ph, pa = cfg["primary_budget"], cfg["primary_heur"], cfg["primary_algo"]
    sP, sM = a_success(done, ids, "P", ph, pa, pb), a_success(done, ids, "M", ph, pa, pb)
    primary = mcnemar_maps(sP, sM, ids, f"P|{ph}|{pa}", f"M|{ph}|{pa}", pb)
    attr = {f"M_vs_{c}": mcnemar_maps(a_success(done, ids, c, ph, pa, pb), sM, ids, f"{c}|{ph}|{pa}",
                                      f"M|{ph}|{pa}", pb) for c in ("R", "S")}
    holm = ST.holm({k: v["p_exact"] for k, v in attr.items()})
    for k in attr:
        attr[k]["p_holm"] = holm[k]
    secondary = {}
    for heur in cfg["heuristics"]:
        for algo in cfg["algos"]:
            for b in cfg["budgets"]:
                for c in [a for a in cfg["arms"] if a != "M"]:  # P, R, S (+ G when that arm is configured)
                    secondary[f"M_vs_{c}|{heur}|{algo}|{b}"] = mcnemar_maps(
                        a_success(done, ids, c, heur, algo, b), a_success(done, ids, "M", heur, algo, b), ids,
                        c, "M", b)
    bmax = max(cfg["budgets"])
    compute, compute_tests = compute_matched_view(cfg, ids, done, bmax)
    a3_recs = [r for r in done.values() if r["heur"] == "A3"]
    hook_acct = {"runs": len(a3_recs), "without_counters": sum("hook" not in r for r in a3_recs),
                 "calls_ne_h_calls": sum(1 for r in a3_recs if "hook" in r and r["hook"]["calls"] != r["h_calls"]),
                 **{k: sum(r["hook"][k] for r in a3_recs if "hook" in r)
                    for k in ("cache_hits", "invalid_states", "fallbacks", "failsafe")}}
    usage = {}
    for arm in [a for a in cfg["arms"] if a != "P"]:
        u = Counter()
        for (i, a, h_, g), r in done.items():
            if a == arm and r["success"]:
                u.update(r.get("macro_uses") or {})
        names = [d["key"] for d in libs_raw["libraries"][arm]]
        usage[arm] = {names[int(k)]: v for k, v in sorted(u.items(), key=lambda kv: int(kv[0]))}
    gates = {"G1_library_sizes": {a: len(libs_raw["libraries"][a]) for a in ("M", "R", "S")},
             "G1_ok": all(len(libs_raw["libraries"][a]) == cfg["K"] for a in ("M", "R", "S")) and not
             libs_raw["overlap"]["M_S_keys"] and not libs_raw["overlap"]["M_S_step_patterns"],
             "G2_invalid_found_plans": sum(1 for r in done.values() if r["found"] and not r["success"]),
             "G3_sym_mismatch": sum(r.get("sym_mismatch") or 0 for r in done.values())}
    gates["ok"] = gates["G1_ok"] and gates["G2_invalid_found_plans"] == 0 and gates["G3_sym_mismatch"] == 0
    stop_check = {f"{heur}|{b}": cells[f"M|{heur}|gbfs|{b}"]["success"] <= cells[f"P|{heur}|gbfs|{b}"]["success"]
                  for heur in cfg["heuristics"] for b in (pb, bmax)}
    return {"complete": True, "n_instances": len(ids), "cells": cells, "per_n_success": per_n, "primary": primary,
            "attribution": attr, "secondary_mcnemar": secondary, "compute_matched_success": compute,
            "compute_matched_mcnemar": compute_tests, "a3_hook_accounting": hook_acct,
            "macro_usage_in_solved_plans": usage, "gates": gates, "stop_check_M_not_better": stop_check}


def compute_matched_view(cfg, ids, done, bmax):
    """Descriptive compute-matched view (prereg secondary 'success within h-call thresholds'), for h_calls and
    prim_steps: per (arm, heur, algo, threshold) the success count derived from the bmax-expansion runs
    (macros.compute_matched) with its censored count, and exact McNemar M vs P / R / S on the derived outcomes. A
    threshold above what an arm's expansion-budget-stopped runs consumed is CENSORED for that arm ('exact' False):
    counting those runs as failures would bias the view toward the arm with more children per expansion (M)."""
    fields = (("h_calls", cfg["hcall_thresholds"]), ("prim_steps", cfg["prim_thresholds"]))
    counts, tests = {}, {}
    for field, ths in fields:
        for heur in cfg["heuristics"]:
            for algo in cfg["algos"]:
                outc = {arm: {t: {i: MA.compute_matched(done[(i, arm, heur, algo)], field, t) for i in ids}
                              for t in ths} for arm in cfg["arms"]}
                for arm in cfg["arms"]:
                    counts[f"{field}|{arm}|{heur}|{algo}"] = {
                        str(t): MA.compute_matched_counts([done[(i, arm, heur, algo)] for i in ids], field, t)
                        for t in ths}
                for t in ths:
                    sm = {i: v is True for i, v in outc["M"][t].items()}
                    for c in [a for a in cfg["arms"] if a != "M"]:
                        sc = {i: v is True for i, v in outc[c][t].items()}
                        test = mcnemar_maps(sc, sm, ids, c, "M", None)
                        del test["budget"]
                        test.update(unit=field, threshold=t, source_expansion_budget=bmax,
                                    exact=all(v is not None for v in outc["M"][t].values())
                                    and all(v is not None for v in outc[c][t].values()))
                        tests[f"M_vs_{c}|{field}|{heur}|{algo}|{t}"] = test
    return counts, tests


def a_success_count(done, ids, arm, heur, algo, b):
    return sum(a_success(done, ids, arm, heur, algo, b).values())


def a_decision(cfg, m, mono_ok):
    """E11a decision rule of the prereg ('E11a decision rule'): validity gates first, then the outcome rows top to
    bottom, first match wins. Also returned, without changing the outcome: the pre-declared compute-matched secondary
    (M vs P within cfg compute_secondary_threshold units at the primary heuristic/algorithm; it grades wording only:
    a saving claim needs an exact cell with p < alpha in M's favour) and the arm-G interpretation flag."""
    if not m.get("complete"):
        return {"outcome": "NOT_RUN"}
    if not (m["gates"]["ok"] and mono_ok):
        return {"outcome": "INVALID", "why": {"gates": m["gates"], "monotonicity_ok": mono_ok}}
    p, a = m["primary"], cfg["alpha"]
    sig, better = p["p_exact"] < a, p["cand_only"] > p["base_only"]
    attr_ok = all(v["p_holm"] < a and v["cand_only"] > v["base_only"] for v in m["attribution"].values())
    if sig and better and attr_ok:
        out = "ADVANCE"
    elif sig and better:
        out = "REVISE"
    elif (sig and not better) or all(m["stop_check_M_not_better"].values()):
        out = "STOP"
    else:
        out = "INCONCLUSIVE"
    ph, pa, t = cfg["primary_heur"], cfg["primary_algo"], cfg["compute_secondary_threshold"]
    cs = {}
    for field in ("h_calls", "prim_steps"):
        v = m["compute_matched_mcnemar"].get(f"M_vs_P|{field}|{ph}|{pa}|{t}")
        cs[field] = None if v is None else {
            "threshold": t, "exact": v["exact"], "p_exact": v["p_exact"] if v["exact"] else None,
            "M_better": v["cand_only"] > v["base_only"], "evaluable": bool(v["exact"])}
    h = cs["h_calls"]
    cs["saving_claim_allowed"] = bool(h and h["exact"] and h["p_exact"] < a and h["M_better"])
    g = m["secondary_mcnemar"].get(f"M_vs_G|{ph}|{pa}|{cfg['primary_budget']}")
    g_gap = None if g is None else g["cand_success"] - g["base_success"]
    return {"outcome": out, "primary_significant": sig, "M_better": better, "attribution_ok": attr_ok,
            "compute_matched_secondary": cs, "M_minus_G_primary_cell": g_gap,
            "g_clause": None if g is None else (out == "ADVANCE" and g_gap < cfg["a_g_clause_margin"])}


# ----------------------------------------------------------------------------- report: E11b metrics and decision

def headroom_closed(base, cand, ceiling):
    """Fraction of a baseline's excess over a ceiling that a candidate closes, on paired positive lengths:
    (mean log base - mean log cand) / (mean log base - mean log ceiling). None without pairs or without excess."""
    if not base or not len(base) == len(cand) == len(ceiling):
        return None
    lb, lc, lo = (sum(map(math.log, xs)) / len(xs) for xs in (base, cand, ceiling))
    return (lb - lc) / (lb - lo) if lb - lo > 1e-12 else None


def b_metrics(cfg, insts, done, labels):
    execs = [it for it in insts.values() if it["split"] == "exec"]
    methods = b_methods(cfg)
    if any((it["id"], m) not in done for it in execs for m in methods):
        return {"complete": False}
    r0 = f"learned_s{cfg['model_seeds'][0]}"

    def length(it, m):
        r = done[(it["id"], m)]
        return r["plan_len"] if r["success"] else r.get("max_steps", cfg["b_steps_per_block"] * it["n"])

    def logratio(ids_, cand, base):
        """Paired instance bootstrap of mean(log L_cand - log L_base); failures count as max_steps."""
        a = {i: [math.log(length(by[i], base))] for i in ids_}
        b = {i: [math.log(length(by[i], cand))] for i in ids_}
        return ST.cluster_bootstrap_diff(a, b, resamples=cfg["bootstrap_resamples"], seed=DERIVED["b_bootstrap"],
                                         alpha=cfg["alpha"])

    def mean_logratio(ids_, cand, base):
        return float(np.mean([math.log(length(by[i], cand)) - math.log(length(by[i], base)) for i in ids_]))

    by = {it["id"]: it for it in execs}
    ood = [it["id"] for it in execs if it["stratum"] in ("T", "H", "TH")]
    table = {}
    for st in ["pool_OOD"] + cfg["b_exec_strata"]:
        ids_ = ood if st == "pool_OOD" else [it["id"] for it in execs if it["stratum"] == st]
        table[st] = {}
        for m in methods:
            recs = [done[(i, m)] for i in ids_]
            succ = [r for r in recs if r["success"]]
            table[st][m] = {"n": len(recs), "success": len(succ), "plan_len_mean": mean_or_none(
                [r["plan_len"] for r in succ]), "subopt_mean": mean_or_none([r["subopt"] for r in succ]),
                "subopt_n": sum(r["subopt"] is not None for r in succ),
                "reasons": dict(Counter(r["reason"] for r in recs))}
            if m != "gc_gbfs" and ids_:
                table[st][m]["geo_ratio_vs_BU"] = math.exp(mean_logratio(ids_, m, "bottom_up"))
                oa = [i for i in ids_ if by[i].get("opt_ach")]  # OPT_A: achiever-level optimum (descriptive)
                table[st][m]["geo_ratio_vs_OPTA"] = math.exp(float(np.mean(
                    [math.log(length(by[i], m)) - math.log(by[i]["opt_ach"]) for i in oa]))) if oa else None
                table[st][m]["opta_n"] = len(oa)
        table[st]["gc_gbfs_success_by_budget"] = {
            str(b): sum(bool(at_budget(done[(i, "gc_gbfs")], b)["success"]) for i in ids_) for b in cfg["gc_budgets"]}
    primary = logratio(ood, r0, "bottom_up")
    vs_rnd = logratio(ood, r0, "random_order")
    vs_rf = logratio(ood, r0, "ready_first")
    reps = {f"learned_s{s}": mean_logratio(ood, f"learned_s{s}", "bottom_up") for s in cfg["model_seeds"]}
    comp_gap = {st: table[st][r0].get("geo_ratio_vs_BU") for st in cfg["b_exec_strata"]}
    succ = {m: table["pool_OOD"][m]["success"] for m in methods}
    ood_oa = [i for i in ood if by[i].get("opt_ach")]
    headroom = {"n": len(ood_oa), "n_pool": len(ood), **{
        m: headroom_closed([length(by[i], "bottom_up") for i in ood_oa], [length(by[i], m) for i in ood_oa],
                           [by[i]["opt_ach"] for i in ood_oa])
        for m in methods if m not in ("bottom_up", "gc_gbfs")}}
    return {"complete": True, "n_exec": len(execs), "n_ood_pool": len(ood), "table": table, "primary": primary,
            "vs_random_order": vs_rnd, "vs_ready_first": vs_rf, "replicates_log_ratio_vs_BU": reps,
            "composition_geo_ratio_vs_BU_by_stratum": comp_gap, "ood_success": succ,
            "ood_headroom_closed_vs_OPTA": headroom, "label_accuracy": labels}


def b_decision(cfg, m):
    """E11b decision rule of the prereg ('E11b decision rule'): rows top to bottom, first match wins. ADVANCE also
    needs the L-r0 vs RND point estimate < 0 and the L-r0 vs RF point estimate <= 0. `rf_clause` flags the
    interpretation clause: every other ADVANCE condition holds but L-r0 is longer than RF."""
    if not m.get("complete"):
        return {"outcome": "NOT_RUN"}
    p, rnd, rf = m["primary"], m["vs_random_order"], m["vs_ready_first"]
    r0 = f"learned_s{cfg['model_seeds'][0]}"
    sL, sBU = m["ood_success"][r0], m["ood_success"]["bottom_up"]
    reps_ok = all(v < 0 for v in m["replicates_log_ratio_vs_BU"].values())
    rnd_ok, rf_ok = rnd["diff"] < 0, rf["diff"] <= 0
    adv_but_rf = (p["hi"] < 0 and p["diff"] <= cfg["b_advance_log_ratio"] and sL >= sBU - cfg["b_success_margin"]
                  and reps_ok and rnd_ok)
    if adv_but_rf and rf_ok:
        out = "ADVANCE"
    elif rnd["diff"] >= 0 or sL < cfg["b_stop_success_frac"] * sBU:
        out = "STOP"
    elif rnd["hi"] < 0:
        out = "REVISE"
    else:
        out = "INCONCLUSIVE"
    return {"outcome": out, "learned_success": sL, "bu_success": sBU, "replicates_all_negative": reps_ok,
            "vs_rnd_point_negative": rnd_ok, "vs_rf_point_le_0": rf_ok, "rf_clause": adv_but_rf and not rf_ok}


# ----------------------------------------------------------------------------- report

def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_report(ctx, out):
    cfg = ctx.cfg
    cu = out.get("custody", {})
    L = [f"# RCX-E11 report ({'SMOKE: plumbing only, not a result' if ctx.smoke else 'full run'})", "",
         f"Prereg: {out['prereg']}", "",
         f"Custody: BWM bytecode checks of earlier invocations {cu.get('bwm_bytecode_checks')}, all clean: "
         f"{cu.get('bwm_bytecode_all_clean')}" + ("" if cu.get("bwm_bytecode_all_clean", True) else
                                                   " -- INVESTIGATE before using any result"), ""]
    a = out.get("e11a", {})
    if a.get("metrics", {}).get("complete"):
        m = a["metrics"]
        L += ["## E11a macro discovery and reuse", "", f"Decision: **{a['decision']['outcome']}** "
              f"({json.dumps(a['decision'])})", ""]
        for arm, desc in a["libraries"].items():
            L += [f"Library {arm}: " + "; ".join(f"`{d['key']}`" for d in desc), ""]
        L += ["| arm \\| heur \\| algo | " + " | ".join(f"@{b}" for b in cfg["budgets"]) +
              " | children/exp @max | subopt (solved) |", "|---|" + "---|" * (len(cfg["budgets"]) + 2)]
        bmax = max(cfg["budgets"])
        for heur in cfg["heuristics"]:
            for algo in cfg["algos"]:
                for arm in cfg["arms"]:
                    c = [m["cells"][f"{arm}|{heur}|{algo}|{b}"] for b in cfg["budgets"]]
                    last = m["cells"][f"{arm}|{heur}|{algo}|{bmax}"]
                    L.append(f"| {arm} \\| {heur} \\| {algo} | " + " | ".join(f"{x['success']}/{x['n']}" for x in c)
                             + f" | {fmt(last['children_per_expansion'], 1)} | {fmt(last['subopt_mean'])} |")
        p = m["primary"]
        L += ["", f"PRIMARY (M vs P, {p['cand']} @{p['budget']}): {p['cand_success']} vs {p['base_success']} of "
              f"{p['n']}; discordant M-only {p['cand_only']} / P-only {p['base_only']}; exact McNemar p = "
              f"{p['p_exact']:.3g}"]
        for k, v in m["attribution"].items():
            L.append(f"Attribution {k}: {v['cand_success']} vs {v['base_success']}; p = {v['p_exact']:.3g}, Holm "
                     f"{v['p_holm']:.3g}")
        ph, pa = cfg["primary_heur"], cfg["primary_algo"]
        L += ["", f"Compute-matched view ({ph}, {pa}; descriptive; derived from the {max(cfg['budgets'])}-expansion "
              "runs; `*` = censored for an arm, a lower bound, no test):", "",
              "| unit | threshold | " + " | ".join(cfg["arms"]) + " | M vs P exact McNemar p |",
              "|---|---|" + "---|" * len(cfg["arms"]) + "---|"]
        for field, ths in (("h_calls", cfg["hcall_thresholds"]), ("prim_steps", cfg["prim_thresholds"])):
            for t in ths:
                cc = {arm: m["compute_matched_success"][f"{field}|{arm}|{ph}|{pa}"][str(t)] for arm in cfg["arms"]}
                tp = m["compute_matched_mcnemar"][f"M_vs_P|{field}|{ph}|{pa}|{t}"]
                L.append(f"| {field} | {t} | " + " | ".join(f"{cc[x]['success']}{'' if cc[x]['exact'] else '*'}"
                                                         for x in cfg["arms"])
                         + f" | {format(tp['p_exact'], '.3g') if tp['exact'] else 'censored'} |")
        L += ["", f"A3 hook accounting (per-run hooks): {json.dumps(m['a3_hook_accounting'])}"]
        L += ["", f"Gates: {json.dumps(m['gates'])}; monotonicity ok: {a['monotonicity_ok']}; A3 hook: "
              f"{json.dumps(a.get('a3_hook'))}", ""]
    else:
        L += ["## E11a: not complete", ""]
    b = out.get("e11b", {})
    if b.get("metrics", {}).get("complete"):
        m = b["metrics"]
        L += ["## E11b compositional subgoal policy", "", f"Decision: **{b['decision']['outcome']}** "
              f"({json.dumps(b['decision'])})", "",
              "| stratum | method | success | mean len (solved) | subopt (A*-covered) | geo ratio vs BU |",
              "|---|---|---|---|---|---|"]
        for st, row in m["table"].items():
            for meth, v in row.items():
                if meth in b_methods(cfg):
                    L.append(f"| {st} | {meth} | {v['success']}/{v['n']} | {fmt(v['plan_len_mean'], 1)} | "
                             f"{fmt(v['subopt_mean'])} ({v['subopt_n']}) | {fmt(v.get('geo_ratio_vs_BU'))} |")
        L += ["", "Goal-count GBFS success by budget: " + json.dumps(
            {st: row["gc_gbfs_success_by_budget"] for st, row in m["table"].items()})]
        p = m["primary"]
        L += ["", f"PRIMARY (learned r0 vs BU, OOD pool n = {p['n_parents']}): mean log length ratio "
              f"{p['diff']:.4f} [{p['lo']:.4f}, {p['hi']:.4f}] (geo ratio {math.exp(p['diff']):.3f})",
              f"vs random order: {m['vs_random_order']['diff']:.4f} [{m['vs_random_order']['lo']:.4f}, "
              f"{m['vs_random_order']['hi']:.4f}]; vs ready-first: {m['vs_ready_first']['diff']:.4f}",
              f"Replicates (log ratio vs BU): {json.dumps(m['replicates_log_ratio_vs_BU'])}", "",
              "Achiever-level ceiling OPT_A (descriptive): OOD-pool geo ratio vs OPT_A " + json.dumps(
                  {k: fmt(v.get("geo_ratio_vs_OPTA")) for k, v in m["table"]["pool_OOD"].items()
                   if isinstance(v, dict) and "geo_ratio_vs_OPTA" in v}) + "; fraction of BU's excess over OPT_A "
              "closed: " + json.dumps({k: fmt(v) for k, v in m["ood_headroom_closed_vs_OPTA"].items()}), "",
              f"Label accuracy (top-1 in the training-target set `{cfg['b_label_target']}`; `|vs_plan` / "
              "`|vs_achiever` = in A(s) / A_A(s); `unseen` = rows whose renaming class is not in an earlier split, "
              "the in-distribution basis):", "```", json.dumps(
                  {s: {g: v.get(g) for g in ("all", "unseen")} for s, v in m["label_accuracy"].items()}, indent=1),
              "```", ""]
    else:
        L += ["## E11b: not complete", ""]
    (ctx.out / "REPORT_E11.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def estimate_full(ctx, done_a, done_b):
    """Rough full-design CPU estimate from a smoke run: per (arm, heur, algo) mean seconds of solved runs and of
    budget-exhausted runs (scaled by the full/smoke max-budget ratio), weighted by the smoke success rate."""
    full, cfg = FULL, ctx.cfg
    scale = max(full["budgets"]) / max(cfg["budgets"])
    n_full = len(full["a_sizes"]) * full["a_per_size"]
    by = defaultdict(list)
    for r in done_a.values():
        by[(r["arm"], r["heur"], r["algo"])].append(r)
    a_s = 0.0
    for recs in by.values():
        ok = [r["seconds"] for r in recs if r["success"]]
        bad = [r["seconds"] * scale for r in recs if not r["success"]]
        rate = len(ok) / len(recs)
        a_s += n_full * (rate * (np.mean(ok) if ok else 0) + (1 - rate) * (np.mean(bad) if bad else 0))
    b_exec = len(full["b_exec_strata"]) * len(full["b_exec_sizes"]) * full["b_exec_per_cell"]
    per = np.mean([r["seconds"] for r in done_b.values()]) if done_b else 0.0
    b_s = b_exec * len(b_methods(full)) * per
    return {"a_search_s": round(a_s), "b_exec_s": round(b_s), "note": "rough; excludes A*, mining and training"}


def phase_report(ctx, timings, prereg):
    cfg = ctx.cfg
    out = {"prereg": prereg, "smoke": ctx.smoke}
    done_a, done_b = {}, {}
    ad = ctx.out / "a"
    if (ad / "runs.jsonl").exists() and (ad / "libraries.json").exists():
        inst = load_instance_list(ad / "instances.json")
        done_a = read_records(ad / "runs.jsonl", a_key, budget=max(cfg["budgets"]))
        libs_raw = json.loads((ad / "libraries.json").read_text(encoding="utf-8"))
        m = a_metrics(cfg, inst, done_a, libs_raw)
        mono = json.loads((ad / "mono_check.json").read_text(encoding="utf-8")) if (ad / "mono_check.json").exists() \
            else {"all_same": False}
        hk = json.loads((ad / "a3_hook.json").read_text(encoding="utf-8")) if (ad / "a3_hook.json").exists() else {}
        out["e11a"] = {"metrics": m, "decision": a_decision(cfg, m, mono["all_same"]),
                       "monotonicity_ok": mono["all_same"], "libraries": libs_raw["descriptors"],
                       "a3_hook": {k: hk.get(k) for k in ("model_sha256", "weights_sha256", "hybrid_lambda",
                                                          "invalid_policy")},
                       "teacher_custody": libs_raw["custody"]}
    bd = ctx.out / "b"
    if (bd / "exec_runs.jsonl").exists() and (bd / "label_eval.json").exists():
        insts = b_instances(ctx)
        done_b = read_records(bd / "exec_runs.jsonl", b_key)
        labels = json.loads((bd / "label_eval.json").read_text(encoding="utf-8"))
        m = b_metrics(cfg, insts, done_b, labels)
        out["e11b"] = {"metrics": m, "decision": b_decision(cfg, m)}
    out["timings"] = timings
    bpath = ctx.out / "bwm_bytecode_checks.json"  # earlier invocations (this one is recorded when main ends)
    bhist = json.loads(bpath.read_text(encoding="utf-8")) if bpath.exists() else []
    out["custody"] = {"bwm_bytecode_checks": len(bhist), "bwm_bytecode_all_clean": all(r["clean"] for r in bhist),
                      "bwm_bytecode_dirty": [r for r in bhist if not r["clean"]]}
    if ctx.smoke and done_a:
        out["estimate_full_from_smoke"] = estimate_full(ctx, done_a, done_b)
    R.write_json(ctx.out / "metrics_e11.json", out)
    write_report(ctx, out)
    ctx.log("report: E11a " + json.dumps(out.get("e11a", {}).get("decision")) + "; E11b " +
            json.dumps(out.get("e11b", {}).get("decision")))
    if "estimate_full_from_smoke" in out:
        ctx.log(f"report: rough full-design estimate {out['estimate_full_from_smoke']}")


# ----------------------------------------------------------------------------- main

def check_prereg(smoke: bool) -> dict:
    info = {"draft": str(PREREG_DRAFT.relative_to(ROOT)),
            "draft_sha256": R.sha_file(PREREG_DRAFT) if PREREG_DRAFT.exists() else None,
            "frozen": str(PREREG_FROZEN.relative_to(ROOT)),
            "frozen_sha256": R.sha_file(PREREG_FROZEN) if PREREG_FROZEN.exists() else None,
            "pinned_sha256": PREREG_SHA256}
    info["status"] = "FROZEN" if PREREG_SHA256 and info["frozen_sha256"] == PREREG_SHA256 else "DRAFT"
    if not smoke and info["status"] != "FROZEN":
        raise SystemExit("full E11 run refused: PREREG_E11.md is not frozen (or its sha256 differs from "
                         "PREREG_SHA256 in run_e11.py). Use --smoke for plumbing checks.")
    return info


def input_shas() -> dict:
    if R.sha_file(SOLUTIONS) != SOLUTIONS_SHA256:
        raise SystemExit("solutions.jsonl changed since the E11 design pinned it")
    side = A3_MODEL.with_suffix(".json")
    return {"solutions_jsonl": SOLUTIONS_SHA256, "distill_meta_json": R.sha_file(DISTILL_META),
            "a3_model": {"path": str(A3_MODEL.relative_to(ROOT)), "sha256": R.sha_file(A3_MODEL),
                         "sidecar_sha256": R.sha_file(side) if side.exists() else None},
            "exclusions_json": R.sha_file(EX.EXCLUSIONS_PATH), "bwm_pins": dict(BI.PINS),
            "bwm_planner_pin": HK.BWM_PLANNER_PIN}


def code_shas() -> dict:
    return {f: R.sha_file(ROOT / f) for f in CODE_FILES}


def out_dir_problem(res, smoke: bool, allowed_root, forbidden, full_out):
    """Why the resolved path `res` may not be an E11 output directory, else None (pure; tested). Every E11 write goes
    to a strict subdirectory of `allowed_root` (runs/RCX-E11), never into a protected tree, and a full run writes only
    to `full_out` (runs/RCX-E11/main)."""
    if any(res == f or f in res.parents for f in forbidden):
        return f"refusing to write to {res} (protected tree)"
    if allowed_root not in res.parents:
        return f"refusing to write to {res}: E11 writes only below {allowed_root}"
    if not smoke and res != full_out:
        return f"the full run writes only to {full_out}; use --smoke for other locations"
    return None


def record_bwm_bytecode(out: Path, phase: str) -> dict:
    """Append this invocation's BWM __pycache__ diff (bwm_import.check_no_bytecode, against the snapshot anchored at
    the start of main) to out/bwm_bytecode_checks.json. RCX never imports BWM through the import system, so a dirty
    diff means either an RCX custody bug or BWM's own processes writing their caches; it is recorded and reported,
    and the report flags it."""
    diff = BI.check_no_bytecode()
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": phase,
           "clean": not (diff["new_pycache_dirs"] or diff["ours"]), "diff": diff}
    p = out / "bwm_bytecode_checks.json"
    hist = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    R.write_json(p, hist + [rec])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="output dir (default runs/RCX-E11/main; smoke runs/RCX-E11/smoke)")
    ap.add_argument("--phase", choices=PHASES + ("all",), default="all")
    ap.add_argument("--smoke", action="store_true", help="tiny configuration for a plumbing check (minutes)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if not 1 <= a.threads <= 4:
        raise SystemExit("CPU single lane: 1-4 torch threads")
    BI.check_no_bytecode()  # anchors the BWM __pycache__ snapshot before any phase (no-op if load() already took it)
    prereg = check_prereg(a.smoke)
    cfg = copy.deepcopy(SMOKE if a.smoke else FULL)
    if cfg["primary_budget"] not in cfg["budgets"] or cfg["mono_check_budget"] > max(cfg["budgets"]):
        raise SystemExit("config: primary_budget must be one of the budgets")
    if cfg["b_label_target"] not in SG.LABEL_TARGETS:
        raise SystemExit(f"config: b_label_target must be one of {SG.LABEL_TARGETS}")
    out = Path(a.out) if a.out else (OUT_SMOKE if a.smoke else OUT_FULL)
    out = (ROOT / out) if not out.is_absolute() else out
    res = out.resolve()
    forbidden = [p.resolve() for p in (BI.BUNDLE_ROOT, RCWM_BUNDLE, ROOT.parent / "bwm", ROOT.parent / "rcwm",
                                       ROOT / "runs" / "RCX-E07", ROOT / "runs" / "RCX-E01N", ROOT / "runs" / "RCX-E01",
                                       ROOT / "experiments")]
    why = out_dir_problem(res, a.smoke, (ROOT / "runs" / "RCX-E11").resolve(), forbidden, OUT_FULL.resolve())
    if why:
        raise SystemExit(why)
    if res.exists() and any(res.iterdir()) and not (res / "config.json").exists():
        raise SystemExit(f"{res} exists and is not an E11 run directory; never write into an old directory")
    out.mkdir(parents=True, exist_ok=True)
    frozen = {"cfg": cfg, "smoke": a.smoke, "seeds": SEEDS, "derived_seeds": DERIVED, "model_seeds": MODEL_SEEDS}
    cpath = out / "config.json"
    if cpath.exists():
        if json.loads(cpath.read_text(encoding="utf-8")) != json.loads(json.dumps(frozen)):
            raise SystemExit(f"{cpath} differs from this invocation's config; use a new --out")
    else:
        R.write_json(cpath, frozen)
    inputs = input_shas()
    ipath = out / "inputs.json"
    if ipath.exists():
        if json.loads(ipath.read_text(encoding="utf-8")) != json.loads(json.dumps(inputs)):
            raise SystemExit(f"inputs changed since {ipath} was written; refusing to mix")
    else:
        R.write_json(ipath, inputs)
    spath = out / "code_shas.json"
    hist = json.loads(spath.read_text(encoding="utf-8")) if spath.exists() else []
    cs = code_shas()
    if hist and hist[0]["code"] != cs:
        changed = sorted(f for f in cs if hist[0]["code"].get(f) != cs[f])
        if not a.smoke:
            raise SystemExit(f"code changed since this run started ({changed}); refusing to mix code versions")
        print(f"WARNING (smoke): code changed since the first start of this directory: {changed}", flush=True)
    hist.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "phase": a.phase, "prereg": prereg, "code": cs,
                 "python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform()})
    R.write_json(spath, hist)
    ctx = R.Ctx(out, cfg, a.smoke)
    ctx.threads = a.threads
    torch.set_num_threads(a.threads)
    ctx.log(f"run_e11 start phase={a.phase} smoke={a.smoke} threads={a.threads} out={out} prereg={prereg['status']}")
    tpath = out / "timings.json"
    timings = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}
    ctx.courtesy(force=True)
    try:
        for ph in (PHASES if a.phase == "all" else (a.phase,)):
            t0 = time.perf_counter()
            if ph == "report":
                phase_report(ctx, timings, prereg)
            else:
                {"a_data": phase_a_data, "a_mine": phase_a_mine, "a_search": phase_a_search,
                 "b_data": phase_b_data, "b_train": phase_b_train, "b_eval": phase_b_eval}[ph](ctx)
            dt = time.perf_counter() - t0
            if dt > 1.0 or ph not in timings:
                timings[ph] = round(dt, 2)
            R.write_json(tpath, timings)
    finally:  # also after a crash: every invocation's BWM bytecode state is recorded
        bc = record_bwm_bytecode(out, a.phase)
        if not bc["clean"]:
            ctx.log(f"WARNING BWM bytecode check (recorded in bwm_bytecode_checks.json): {bc['diff']}")
    ctx.log(f"run_e11 end; guard wait {ctx.guard_wait:.1f}s; timings {timings}")


if __name__ == "__main__":
    main()
