"""CPU tests for the ledger beam test (joint roadmap item 3): problems, proposers, LedgerF and BWM's plan_beam (imported
read-only from the BWM bundle). No GPU, no lease, no real model (tiny random Qwen3 stand-ins only), 2 threads,
below-normal priority. Each part runs in its own process (RAM < 1.5 GB):

  python -B scripts/lbeam_test.py --part oracle   # determinism; the two v2 dev sets (primary, descriptive oor);
                                                  # reachable-state closure of the primary (independent BFS over all
                                                  # reachable states, rule-blind dynamics too); the registered plan_beam
                                                  # max_depth; blind planner instances; proposers; the input-range
                                                  # guard (inert for an exact F, bites for a noisy F); harness; ORACLE F
                                                  # through plan_beam on every primary instance (success vs budget) and
                                                  # the oor set (descriptive); IGNORE_STATUS floor; answer-free floors;
                                                  # hard matched budgets; GPU-call estimates with simulated noisy F
  python -B scripts/lbeam_test.py --part tiny     # LedgerF with a tiny random Qwen3 + LoRA (PEFT, unmerged): the
                                                  # canonical frame only (enforced), decoding identity with
                                                  # rcwmt.generation.chain, one generate call per predict list with new
                                                  # keys, memoisation, model_queries, None on unparseable, OFF vs T2 in
                                                  # one process, cache, loader injection, SurfaceAuditF (D-35, report
                                                  # only), the guard on a model arm, plan_beam end to end
  python -B scripts/lbeam_test.py --part fresh    # freshness of both v2 dev sets against the CURRENT exclusion union
                                                  # (name-skipping harvest), multi-event rollouts, audit renderings,
                                                  # regeneration identity

Writes docs/ledger_beam/TEST_<PART>.json (with the code sha256s). Exit 0 iff every check passes.
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""            # never "-1" (DECISIONS I-21)
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["BWM_DEVICE"] = "cpu"
os.environ["RCWM_DUTY_FORCE"] = "1"
os.environ["RCWM_DUTY_RATIO"] = "0"
os.environ["RCWM_BREATHE_S"] = "0"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BWM_ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
DOCS = ROOT / "docs" / "ledger_beam"
SETS = {"primary": DOCS / "dev2_primary.jsonl", "oor": DOCS / "dev2_oor.jsonl"}
V1_DEV = DOCS / "dev_set.jsonl"                      # superseded (generator v1); read only, for the closure comparison
SCRATCH = Path(r"C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-Consequence-World-Model-Bundle-2026-10-05/4c2a936e-32bf-4e01-aa62-961d6b162be5/scratchpad")
CODE = ("scripts/lbeam_problems.py", "scripts/lbeam_proposer.py", "scripts/lbeam_f.py", "scripts/lbeam_test.py")
RESULTS: list = []
NUM: dict = {}
RATES = {"observed_3.8": 3.8, "planning_2.7": 2.7}       # gens/s: PREREG-2 observed ledger chain_one / planning rate


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "ok": bool(ok), "detail": str(detail)[:600]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail != "" else ""), flush=True)


def peak_mb() -> float:
    import psutil
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)


def nice():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def import_torch_cpu():
    import torch
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0
    torch.cuda.synchronize = lambda *a, **k: None
    torch.set_num_threads(2)
    return torch


def import_plan_beam():
    """BWM's planner, read-only (no bytecode is written into the BWM bundle). bwmn.lm imports torch at module level, so
    torch is imported CPU-patched first (BWM_DEVICE=cpu is set as well)."""
    import_torch_cpu()
    if str(BWM_ROOT) not in sys.path:
        sys.path.insert(0, str(BWM_ROOT))
    from bwmn import blocks_planner as BP
    if Path(BP.__file__).resolve() != (BWM_ROOT / "bwmn" / "blocks_planner.py").resolve():
        raise SystemExit(f"blocks_planner imported from {BP.__file__}")
    return BP


def load_sets(record_checks: bool = True) -> tuple:
    """(sets {primary, oor}, source {kind: 'file' | 'in-memory'}, build records {kind: SOURCE record or in-memory rec}).
    A file must equal its SOURCE record and its in-memory regeneration from (seed, recorded spec) + the recorded freshness
    rejections (no exclusion union needed)."""
    import lbeam_problems as P
    sets, src, recs = {}, {}, {}
    for kind, path in SETS.items():
        spec = P.primary_spec() if kind == "primary" else P.oor_spec()
        if path.exists():
            insts = P.load_set(path)
            srec = json.loads(path.with_name(path.name.replace(".jsonl", ".SOURCE.json")).read_text(encoding="utf-8"))
            gen, rec = P.make_set(P.spec_from_record(srec), replay_rejections=srec["fresh_rejections"])
            if record_checks:
                check(f"{kind} dev set file ({path.name}): sha256 == SOURCE record == regenerated in memory from (seed, "
                      f"recorded spec) + the recorded freshness rejections ({len(srec['fresh_rejections'])}), without "
                      f"loading the exclusion union; generator {P.GEN_VERSION}; spec == the registered "
                      f"{'primary' if kind == 'primary' else 'oor'}_spec()",
                      sha_file(path) == srec["file_sha256"] == rec["file_sha256"] == P.set_sha(insts)
                      and srec["generator"] == P.GEN_VERSION and P.spec_from_record(srec) == spec,
                      f"file {sha_file(path)[:16]} regen {rec['file_sha256'][:16]}")
            sets[kind], src[kind], recs[kind] = insts, "file", srec
        else:
            insts, rec = P.make_set(spec)
            sets[kind], src[kind], recs[kind] = insts, "in-memory (no file yet)", rec
    return sets, src, recs


def write_record(part: str, extra: dict) -> Path:
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / f"TEST_{part.upper()}.json"
    rec = {"part": part, "ok": all(r["ok"] for r in RESULTS), "n_checks": len(RESULTS),
           "n_failed": sum(not r["ok"] for r in RESULTS), "checks": RESULTS, "numbers": NUM,
           "code_sha256": {c: sha_file(ROOT / c) for c in CODE},
           "set_sha256": {k: (sha_file(p) if p.exists() else None) for k, p in SETS.items()},
           "bwm_blocks_planner_sha256": sha_file(BWM_ROOT / "bwmn" / "blocks_planner.py"),
           "peak_rss_mb": peak_mb(), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1, default=str) + "\n")
    return out


def _tmp_write(text: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix="lbeam_", dir=SCRATCH if SCRATCH.exists() else None))
    p = d / "set.jsonl"
    p.write_text(text, encoding="utf-8")
    return p


def _out_of_range_state(s) -> bool:
    return any(abs(v) > 98 for _, v in s)


# ================================================================================================ part: oracle
def part_oracle() -> dict:
    import lbeam_problems as P
    import lbeam_proposer as PR
    import lbeam_f as LF
    from rcwmt import env
    t0 = time.time()
    print("== determinism", flush=True)
    small = P.SetSpec(name="det", seed=777, n_in_per_depth=3, n_oor_per_depth=1)
    a1, r1 = P.make_set(small)
    a2, r2 = P.make_set(small)
    b1, _ = P.make_set(P.SetSpec(name="det", seed=778, n_in_per_depth=3, n_oor_per_depth=1))
    big, _ = P.make_set(P.SetSpec(name="det", seed=777, n_in_per_depth=5, n_oor_per_depth=2))
    check("generator: same seed -> identical sha256; different seed -> different sha256",
          r1["file_sha256"] == r2["file_sha256"] == P.set_sha(a2) and P.set_sha(b1) != r1["file_sha256"],
          f"{r1['file_sha256'][:16]} / {P.set_sha(b1)[:16]}")
    by_big = {i["id"]: i for i in big}
    check("generator: slots are independent (a larger spec reproduces every slot of the smaller one byte for byte, "
          "apart from cross-instance action-uniqueness redraws)",
          sum(json.dumps(P._jsonable(i), sort_keys=True) == json.dumps(P._jsonable(by_big[i["id"]]), sort_keys=True)
              for i in a1) >= len(a1) - 2, f"{len(a1)} slots")
    rt = P.load_set(_tmp_write(P.dumps_set(a1)))
    check("serialisation round trip (tuples restored; re-serialised sha identical)", P.set_sha(rt) == r1["file_sha256"]
          and rt[0]["init"] == a1[0]["init"] and rt[0]["vocab"] == a1[0]["vocab"])
    rng = random.Random(5)
    bad = 0
    pool = list(range(-120, 121))
    for _ in range(20000):
        s = tuple((x, rng.choice(pool)) for x in P.ACCOUNTS)
        acc = rng.choice(P.ACCOUNTS)
        a = (rng.choice(("add", "scale", "transfer")), acc, rng.choice([x for x in P.ACCOUNTS if x != acc]),
             rng.randint(-30, 30), rng.choice(("posted", "void", "pending")), rng.randint(1, 99))
        bad += P.from_vec(P.vstep(P.vec(s), P.compile_action(a))) != P.step_state(s, a)
    check("vstep == env.step on 20,000 random transitions (any status)", bad == 0, f"{bad} mismatches")
    for _ in range(300):
        s = tuple((x, rng.choice(pool)) for x in P.ACCOUNTS)
        acc = rng.choice(P.ACCOUNTS)
        a = (rng.choice(("add", "scale", "transfer")), acc, rng.choice([x for x in P.ACCOUNTS if x != acc]),
             rng.randint(-30, 30), rng.choice(("posted", "void")), rng.randint(1, 99))
        P.env.cross_check(P.LEDGER, P.state_to_dict(s), P.action_to_event(a))
    check("env.step == the validated FDS oracle on 300 random transitions (env.cross_check)", True)

    sets, src, brecs = load_sets()
    insts, OOR = sets["primary"], sets["oor"]
    NUM["sets"] = {k: {"source": src[k], "sha256": P.set_sha(v), "n": len(v)} for k, v in sets.items()}
    vp, vo = P.verify_instances(insts), P.verify_instances(OOR)
    per_d = Counter(i["opt_depth"] for i in insts)
    check("PRIMARY set: every instance re-verifies (BFS d*, plans, fixed candidates, oracle, in-range definition, "
          "reachable-state closure incl. the rule-blind dynamics, no void step on an optimal plan); only 'in' instances; "
          "n = 120 (24 per d* = 2..6)",
          vp["ok"] and all(i["range"] == "in" for i in insts) and len(insts) == 120 and set(per_d.values()) == {24},
          {"problems": vp["problems"], "per_depth": dict(sorted(per_d.items()))})
    check("DESCRIPTIVE oor set: re-verifies; only 'oor' instances, drawn as a separate set with its own seed (never "
          "pooled with the primary); n = 30 (6 per d*)",
          vo["ok"] and all(i["range"] == "oor" for i in OOR) and len(OOR) == 30
          and brecs["oor"]["spec"]["seed"] != brecs["primary"]["spec"]["seed"], vo["problems"])
    NUM["primary_describe"] = P.describe(insts)
    NUM["oor_describe"] = P.describe(OOR)

    print("== reachable-state closure of the primary set (BFS over ALL reachable states)", flush=True)
    t = time.time()
    tot = mx = n_bad = n_mism = 0
    depths = set()
    for i in insts:
        D = i["closure"]["depth"]
        depths.add(D)
        evs = [P.action_to_event(a) for a in i["vocab"]]
        seen, layer = {i["init"]}, [i["init"]]
        for _ in range(D):                               # independent of the generator: env.step on dicts, all actions
            nxt = []
            for s in layer:
                bd = dict(s)
                for ev in evs:
                    ns = tuple(sorted(env.step("ledger_replay", bd, ev).items()))
                    if ns not in seen:
                        seen.add(ns)
                        nxt.append(ns)
            layer = nxt
            if len(seen) > 300000:
                break
        m = max(abs(v) for s in seen for _, v in s)
        tot += len(seen)
        mx = max(mx, m)
        n_bad += m > P.IN_RANGE_MAX_ABS
        n_mism += len(seen) != i["closure"]["n_states"]
    NUM["closure_independent"] = {"depth": sorted(depths), "instances": len(insts), "reachable_states_total": tot,
                                  "max_abs_balance": mx, "instances_violating": n_bad, "count_mismatches": n_mism,
                                  "seconds": round(time.time() - t, 1)}
    check(f"closure, re-derived independently (env.step on dicts, not the generator's vstep): for EVERY primary instance, "
          f"every state reachable from init by any sequence of <= {P.D_CLOSE} vocabulary actions has |b| <= 98; the "
          f"reachable-state counts equal the generator's records",
          n_bad == 0 and n_mism == 0 and depths == {P.D_CLOSE},
          f"{tot} reachable states over {len(insts)} instances; max |b| {mx}; {time.time() - t:.0f}s")
    trained = P.trained_amounts()
    check("every vocabulary action of the primary set lies in T2's trained surface: amount one of the 21 trained values "
          "(-9..12, so |amount| <= 12), status posted / void, accounts A-D (sorted canonical state), kinds add / scale / "
          "transfer",
          all(a[3] in trained and a[4] in ("posted", "void") and a[1] in P.ACCOUNTS and a[2] in P.ACCOUNTS
              and a[0] in P.KIND_CODE for i in insts for a in i["vocab"])
          and all(P.is_state(i["init"]) for i in insts) and max(abs(a[3]) for i in insts for a in i["vocab"]) <= 12,
          NUM["primary_describe"]["posted_kind_amount"]["scale"])
    eq = sum(P.inst_closure(i, keep=True)["states"] == P.inst_closure(i, ignore_status=True, keep=True)["states"]
             for i in insts)
    check("the closure covers the rule-blind dynamics: applying every void twin as posted (IGNORE_STATUS; OFF's dominant "
          "error) reaches exactly the same state set on every primary instance (each twin repeats a posted vocabulary "
          "action's arithmetic)", eq == len(insts), f"{eq}/{len(insts)}")
    rej_closure = sum(v for d in brecs["primary"]["rejected"].values() for r, v in d.items() if "closure" in r)
    v1_fail = None
    if V1_DEV.exists():
        v1_in = [i for i in P.load_set(V1_DEV) if i["range"] == "in"]
        v1_fail = (sum(not P.inst_closure(i, depth=P.D_CLOSE)["ok"] for i in v1_in), len(v1_in))
    NUM["closure_enforcement"] = {"primary_draws_rejected_for_closure": rej_closure,
                                  "v1_dev_in_instances_failing_closure": v1_fail}
    check("the generator enforces the closure (draws rejected for it while building the primary set); the superseded v1 "
          "dev set (no closure) fails it on most of its 'in' instances",
          rej_closure > 0 and (v1_fail is None or v1_fail[0] > 0), NUM["closure_enforcement"])
    oo = [any(a[3] not in trained for a in i["vocab"]) or not P.inst_closure(i, depth=P.D_CLOSE)["ok"] for i in OOR]
    check("descriptive oor set: every instance leaves T2's verified range (an untrained amount in its vocabulary, or a "
          "state with |b| > 98 within 8 steps)", all(oo), f"{sum(oo)}/{len(OOR)}")

    print("== registered plan_beam max_depth (lbeam_proposer.beam_kwargs)", flush=True)
    bk = {"enum_d1": PR.beam_kwargs(PR.EnumerateProposer(), 2, 12)["max_depth"],
          "enum_d2": PR.beam_kwargs(PR.EnumerateProposer(depth=2), 2, 12)["max_depth"],
          "random_L6_R3": PR.beam_kwargs(PR.RandomProposer(K=8, length=6), 4, 8, 3)["max_depth"],
          "random_L6_R0": PR.beam_kwargs(PR.RandomProposer(K=8, length=6), 1, 1, 0)["max_depth"],
          "fixed": PR.beam_kwargs(PR.FixedCandidatesProposer(insts), 1, 1)["max_depth"],
          "oracle": PR.beam_kwargs(PR.OracleProposer(), 1, 1)["max_depth"]}
    kw = PR.beam_kwargs(PR.EnumerateProposer(), 2, 12)
    check("beam_kwargs: max_depth = D_CLOSE - simulated candidate length (enum depth 1: 7; enum depth 2: 6; random L6 "
          "R3: 5; random one-shot L6: 2; fixed / oracle plans <= 6: 2); the hooks are lbeam_problems'",
          bk == {"enum_d1": 7, "enum_d2": 6, "random_L6_R3": 5, "random_L6_R0": 2, "fixed": 2, "oracle": 2}
          and kw["goal_met"] is P.goal_met and kw["unmet"] is P.unmet, bk)
    NUM["registered_max_depth"] = bk

    floors = P.answer_free_floors(insts)
    NUM["answer_free_floors"] = floors
    check("answer-free floors computed (primary): fixed view uniform-over-void-free > 1/K; greedy true-unmet descent "
          "solves part of the set (reported; score() splits success by greedy / needs_lookahead)",
          floors["in"]["fixed_uniform_over_void_free"] > 0.125 and 0 < floors["in"]["greedy_true_unmet_solves"] < floors["in"]["n"],
          {k: (v["fixed_uniform_over_void_free"], f"{v['greedy_true_unmet_solves']}/{v['n']}") for k, v in floors.items()})
    PI = P.planner_instances(insts)                     # blind (the BWM-facing form)
    own = P.owner_of_action(insts)
    own_o = P.owner_of_action(OOR)
    GUARD = P.in_range_actions(insts)

    print("== blind planner instances and proposers", flush=True)
    leak_fields = ("rec", "opt_plan", "seed_plan", "vocab_roles", "fixed", "opt_states", "vocab_effect_at_init", "closure")
    check("planner_instances() is blind by default: exactly id / init / goal / objs, no answer key (opt plan, seed plan, "
          "vocab roles, fixed candidates) reachable from what BWM receives",
          all(set(d) == set(P.PLANNER_FIELDS) for d in PI)
          and not any(f in json.dumps([{k: v for k, v in d.items()} for d in PI], default=str) for f in leak_fields)
          and all("rec" in d for d in P.planner_instances(insts[:3], blind=False)))
    items = [{"id": it["id"], "objs": it["objs"], "goal": it["goal"], "state": it["init"], "round": "0.0"} for it in PI[:20]]
    rp = PR.RandomProposer(K=8, length=6)
    o1 = rp.propose(items, 3)
    o2 = rp.propose(items, 3)
    single = [rp.propose([x], 3)[0] for x in items]
    o4 = rp.propose(items, 4)
    o5 = rp.propose([{**x, "round": "1.2"} for x in items], 3)
    check("RandomProposer (no instance list needed): deterministic; batch-independent (CRN on (eval_seed, id, round, j)); "
          "eval_seed and round change the draws; n_tokens 0; plans from the item's objs",
          o1 == o2 == single and o1 != o4 and o1 != o5
          and all(c["n_tokens"] == 0 and len(c["plan"]) == 6 and all(a in set(it["objs"]) for a in c["plan"])
                  for it, cs in zip(items, o1) for c in cs))
    ep = PR.EnumerateProposer()
    e1 = ep.propose(items, 0)
    check("EnumerateProposer: the V = 8 single actions in vocabulary order per item; deterministic, batch- and "
          "seed-independent; n_tokens 0; depth 2 gives V^2 = 64 sequences",
          all([c["plan"] for c in cs] == [[a] for a in it["objs"]] and all(c["n_tokens"] == 0 for c in cs)
              for it, cs in zip(items, e1))
          and e1 == ep.propose(items, 99) == [ep.propose([x], 5)[0] for x in items]
          and len(PR.EnumerateProposer(depth=2).propose(items[:1], 0)[0]) == 64)
    fp = PR.FixedCandidatesProposer(insts)
    f1 = fp.propose(items, 0)
    try:
        PR.FixedCandidatesProposer(PI)
        refuses = False
    except ValueError:
        refuses = True
    try:
        PR.select_fixed(PI[:2], LF.LedgerF("ORACLE"))
        refuses2 = False
    except ValueError:
        refuses2 = True
    check("FixedCandidatesProposer / select_fixed take the RCWM records: the stored K candidates at round 0, nothing "
          "later; both refuse the blind planner dicts",
          all([c["plan"] for c in cs] == [list(x) for x in r["fixed"]["cands"]] for r, cs in zip(insts[:20], f1))
          and fp.propose([{**items[0], "round": "1.0"}], 0) == [[]] and refuses and refuses2)

    BP = import_plan_beam()
    NUM["bwm_blocks_planner_sha256"] = sha_file(BWM_ROOT / "bwmn" / "blocks_planner.py")

    print("== input-range guard (RangeGuardF) unit check", flush=True)
    gi = insts[0]
    a_in = gi["vocab"][0]
    s_out = tuple((a, 99 if a == "A" else v) for a, v in gi["init"])
    a_oor = OOR[0]["vocab"][0]
    inner = LF.LedgerF("ORACLE")
    g = LF.RangeGuardF(inner, applies_to=GUARD)
    r = g.predict([(gi["init"], a_in), (s_out, a_in), (s_out, a_oor), (gi["init"], a_in)])
    check("RangeGuardF: a primary key with an input balance |b| = 99 returns None without reaching the inner F; keys "
          "outside applies_to (the descriptive oor set) pass unguarded; admitted keys go to the inner F in one predict call",
          r[0] == (P.step_state(gi["init"], a_in), True) and r[1] is None and r[2] == (P.step_state(s_out, a_oor), True)
          and r[3] == r[0] and inner.model_queries == 2 and inner.n_predict == 1 and (s_out, a_in) not in inner.requested
          and g.refused == {(s_out, a_in)} and LF.guarded(LF.LedgerF("ORACLE"), insts).applies_to == GUARD
          and len(GUARD) == 8 * len(insts), g.stats())

    EXACT: list = []                                     # (config, arm, refused, inputs out of range, outputs out of range)

    def exposure_exact(tag, arm, f, inner_):
        ins_out = sum(not P.in_verified_range(k[0], k[1]) for k in inner_.requested)
        outs_out = sum(1 for v in inner_.memo.values() if v is not None and _out_of_range_state(v[0]))
        EXACT.append((tag, arm, len(f.refused), ins_out, outs_out, len(inner_.requested)))

    def make_f(arm, recs, guard=True, sim=None, budget=None):
        inner_ = LF.SimulatedF(sim) if sim else LF.LedgerF(arm)
        mid = LF.BudgetF(inner_, P.owner_of_action(recs), budget) if budget is not None else inner_
        return LF.RangeGuardF(mid, applies_to=P.in_range_actions(recs) if guard else ()), inner_, mid

    def run_beam(prop, W, E, R=0, arm="ORACLE", guard=True, sim=None, budget=None, recs=None, eval_seed=0,
                 max_depth=None, tag=None):
        recs = insts if recs is None else recs
        o = P.owner_of_action(recs)
        f, inner_, mid = make_f(arm, recs, guard, sim, budget)
        kw_ = PR.beam_kwargs(prop, W, E, R)
        if max_depth is not None:
            kw_["max_depth"] = max_depth
        t_ = time.time()
        eps = BP.plan_beam(P.planner_instances(recs), prop, f, eval_seed=eval_seed, **kw_)
        calls = mid.calls_by_owner() if budget is not None else inner_.calls_by_owner(o)
        summ, rows = P.score(recs, eps, calls)
        summ["model_queries_total"] = inner_.model_queries
        summ["refused_unique"] = len(f.refused)
        summ["max_depth"] = kw_["max_depth"]
        summ["wall_s"] = round(time.time() - t_, 2)
        if tag and recs is insts and sim is None and max_depth is None and arm in ("ORACLE", "IGNORE_STATUS"):
            exposure_exact(tag, arm, f, inner_)
        return summ, rows, eps, f, inner_

    def run_fixed(arm="ORACLE", guard=True, sim=None, recs=None, tag=None):
        recs = insts if recs is None else recs
        f, inner_, _ = make_f(arm, recs, guard, sim)
        eps = PR.select_fixed(recs, f, eval_seed=0)
        summ, rows = P.score(recs, eps, inner_.calls_by_owner(P.owner_of_action(recs)))
        summ["selected_kind"] = dict(sorted(Counter(e["selected_kind"] for e in eps).items()))
        summ["refused_unique"] = len(f.refused)
        if tag and recs is insts and sim is None:
            exposure_exact(tag, arm, f, inner_)
        return summ, rows, eps, f, inner_

    print("== harness (ORACLE F, guarded, registered max_depth; primary set)", flush=True)
    s, rows, eps, f, inner = run_beam(PR.OracleProposer(K=1), W=1, E=1, tag="oracle_proposer")
    NUM["harness_oracle_proposer"] = s
    check("plan_beam + OracleProposer + guarded ORACLE F (W1 E1): success 1.0, verified 1.0, precision 1.0, every plan "
          "of optimal length d*; nothing refused",
          s["success"] == 1.0 and s["verified_rate"] == 1.0 and s["verification_precision"] == 1.0
          and s["optimal_length_rate"] == 1.0 and not f.refused, f"success {s['success']}")
    check("F accounting: model_queries == distinct keys requested == sum of per-instance attributions (every action "
          "has one owner)", inner.model_queries == len(inner.requested) == sum(inner.calls_by_owner(own).values())
          and None not in inner.calls_by_owner(own), f"{inner.model_queries}")
    s, rows, eps, f, inner = run_beam(PR.FixedCandidatesProposer(insts), W=1, E=1, tag="fixed_view")
    NUM["fixed_view_through_plan_beam_oracle"] = s
    check("plan_beam + FixedCandidatesProposer + guarded ORACLE F (W1 E1): success 1.0 (the optimal candidate is found)",
          s["success"] == 1.0 and s["verification_precision"] == 1.0, s["f_calls_per_instance"])
    fixed_view_total = s["f_calls_per_instance"]["total"]
    s_or, _, eps_or, f_or, in_or = run_fixed("ORACLE", tag="select_fixed")
    NUM["fixed_select_oracle"] = s_or
    cb = in_or.calls_by_owner(own)
    by_d = {}
    for i in insts:
        by_d.setdefault(i["opt_depth"], []).append(cb.get(i["id"], 0))
    NUM["fixed_select_f_calls_per_instance_by_depth"] = {str(d): round(statistics.fmean(v), 2) for d, v in sorted(by_d.items())}
    check("select_fixed + guarded ORACLE F: success 1.0, regret 0; F calls/instance <= distinct candidate prefixes <= "
          "K x d* and == the fixed view through plan_beam",
          s_or["success"] == 1.0 and s_or["mean_nregret"] == 0.0
          and all(cb.get(i["id"], 0) <= len({tuple(c[:t + 1]) for c in i["fixed"]["cands"] for t in range(i["opt_depth"])})
                  <= i["fixed"]["K"] * i["opt_depth"] for i in insts)
          and s_or["f_calls_per_instance"]["total"] == fixed_view_total, s_or["f_calls_per_instance"])
    s_ig, _, eps_ig, _, _ = run_fixed("IGNORE_STATUS", tag="select_fixed")
    NUM["fixed_select_ignore_status"] = s_ig
    check("select_fixed + IGNORE_STATUS F (void treated as posted): success well below 1 - the problems punish "
          "ignoring the void rule", s_ig["success"] < 0.95, f"success {s_ig['success']:.3f}, selected {s_ig['selected_kind']}")

    print("== success vs budget (ORACLE F through plan_beam; primary, guarded, registered max_depth)", flush=True)
    sweep = []
    # E counts NODE expansions (plan_beam: one per beam node per round), so with width W a round costs up to W of E.
    grids = [("enum_W1", "enum", 1, None, 0, (1, 2, 4, 6, 8, 12, 16)),
             ("enum_W2", "enum", 2, None, 0, (2, 6, 12, 24, 48)),
             ("enum_W4", "enum", 4, None, 0, (4, 12, 24, 48, 96)),
             ("enum_W8", "enum", 8, None, 0, (8, 24, 48, 96)),
             ("random_oneshot_W1", "random", 1, None, 0, None),
             ("random_W4K8R3", "random", 4, 8, 3, (1, 4, 8, 16, 32, 64)),
             ("random_W4K32R3", "random", 4, 32, 3, (1, 4, 8, 16, 32)),
             ("random_W8K64R3", "random", 8, 64, 3, (8, 32, 64))]
    for tag, kind, W, K, R, Es in grids:
        runs = [(k, 1) for k in (8, 32, 128, 256)] if Es is None else [(K, E) for E in Es]
        for K_, E in runs:
            prop = PR.EnumerateProposer() if kind == "enum" else PR.RandomProposer(K=K_, length=6)
            s, rows, _, _, _ = run_beam(prop, W=W, E=E, R=R, tag=f"{tag}_K{K_}_E{E}")
            row = {"config": tag, "proposer": kind, "W": W, "K": K_ if kind == "random" else 8, "R": R, "E": E,
                   "max_depth": s["max_depth"], "success": round(s["success"], 4),
                   "success_by_lookahead": s["success_by_lookahead"],
                   "verified_rate": round(s["verified_rate"], 4), "precision": s["verification_precision"],
                   "optimal_length_rate": round(s["optimal_length_rate"], 4), "mean_nregret": round(s["mean_nregret"], 4),
                   "f_calls_mean": s["f_calls_per_instance"]["mean"], "f_calls_p90": s["f_calls_per_instance"]["p90"],
                   "f_calls_max": s["f_calls_per_instance"]["max"], "refused": s["refused_unique"],
                   "success_by_depth": s["success_by_depth"], "wall_s": s["wall_s"]}
            sweep.append(row)
            print(f"  {tag:18s} K={row['K']:<3} E={E:<3} md={row['max_depth']} success {row['success']} (greedy/lookahead "
                  f"{row['success_by_lookahead']}) precision {row['precision']} calls mean {row['f_calls_mean']} "
                  f"p90 {row['f_calls_p90']} refused {row['refused']}", flush=True)
    NUM["oracle_sweep"] = sweep
    mono = all([r["success"] for r in sweep if r["config"] == c] == sorted(r["success"] for r in sweep if r["config"] == c)
               for c in {r["config"] for r in sweep if r["config"] != "random_oneshot_W1"})
    top_r = max((r for r in sweep if r["config"] == "random_W8K64R3"), key=lambda r: r["E"])
    top_e = max((r for r in sweep if r["config"] == "enum_W8"), key=lambda r: r["E"])
    check("ORACLE F through plan_beam on all 120 primary instances: success never falls as E grows (CRN / deterministic "
          "enumeration replays the smaller run); precision 1.0 everywhere; success >= 0.95 at the most generous "
          "budgets of both proposers",
          mono and all(r["precision"] in (None, 1.0) for r in sweep) and top_r["success"] >= 0.95 and top_e["success"] >= 0.95,
          f"random W8K64R3E{top_r['E']}: {top_r['success']} at {top_r['f_calls_mean']} calls; enum W8E{top_e['E']}: "
          f"{top_e['success']} at {top_e['f_calls_mean']} calls")
    e2 = next(r for r in sweep if r["config"] == "enum_W2" and r["E"] == 12)
    r4 = next(r for r in sweep if r["config"] == "random_W4K8R3" and r["E"] == 8)
    check("the enumerating proposer dominates the random one at lower calls (ORACLE): enum W2 E12 beats random "
          "W4 K8 R3 E8 on success with fewer calls", e2["success"] > r4["success"] and e2["f_calls_mean"] < r4["f_calls_mean"],
          f"enum W2E12 {e2['success']} @ {e2['f_calls_mean']} vs random W4K8R3E8 {r4['success']} @ {r4['f_calls_mean']}")

    print("== IGNORE_STATUS F (rule-blind) in the beam", flush=True)
    for tag, prop, W, E, R in (("random_W4K8R3E32", PR.RandomProposer(K=8, length=6), 4, 32, 3),
                               ("random_W4K32R3E32", PR.RandomProposer(K=32, length=6), 4, 32, 3),
                               ("enum_W2E12", PR.EnumerateProposer(), 2, 12, 0),
                               ("enum_W4E24", PR.EnumerateProposer(), 4, 24, 0)):
        s, _, _, _, _ = run_beam(prop, W=W, E=E, R=R, arm="IGNORE_STATUS", tag=tag)
        NUM[f"beam_ignore_status_{tag}"] = {k: s[k] for k in ("success", "verified_rate", "verification_precision",
                                                              "success_by_lookahead")}
        check(f"IGNORE_STATUS F ({tag}): verification precision < 1 (a rule-blind F verifies plans that fail)",
              (s["verification_precision"] or 0) < 1.0,
              f"success {s['success']:.3f} verified {s['verified_rate']:.3f} precision {s['verification_precision']:.3f}")

    print("== range exposure: exact F inside the closure; plan_beam's default max_depth; noisy F", flush=True)
    NUM["exact_f_exposure"] = [{"config": c, "arm": a, "refused": r_, "inputs_out_of_range": i_, "outputs_out_of_range": o_,
                                "distinct_keys": n_} for c, a, r_, i_, o_, n_ in EXACT]
    check(f"with the registered max_depth an EXACT F never leaves T2's verified range: over all {len(EXACT)} guarded "
          f"ORACLE / IGNORE_STATUS runs above (harness, select_fixed, the whole sweep, the rule-blind beams) RangeGuardF "
          f"refused 0 keys, and no F input and no F output had |b| > 98",
          EXACT and all(r_ == 0 and i_ == 0 and o_ == 0 for _, _, r_, i_, o_, _n in EXACT),
          f"{len(EXACT)} runs; {sum(x[5] for x in EXACT)} distinct keys checked (summed over runs)")
    leak = {}
    for name, prop, W, E, R in (("enum_W1E16", PR.EnumerateProposer(), 1, 16, 0),
                                ("random_W4K8R3E32", PR.RandomProposer(K=8, length=6), 4, 32, 3),
                                ("random_W8K64R3E64", PR.RandomProposer(K=64, length=6), 8, 64, 3),
                                ("enum_W8E96", PR.EnumerateProposer(), 8, 96, 0)):
        su, _, _, fu, inu = run_beam(prop, W, E, R, guard=False, max_depth=80)
        sg, _, _, fg, _ = run_beam(prop, W, E, R, guard=True, max_depth=80)
        reg, _, _, _, _ = run_beam(prop, W, E, R)
        leak[name] = {"unguarded_keys_outside_range": sum(not P.in_verified_range(k[0], k[1]) for k in inu.requested),
                      "keys": len(inu.requested), "guarded_refused": len(fg.refused),
                      "success_default_unguarded": round(su["success"], 4), "success_default_guarded": round(sg["success"], 4),
                      "success_registered": round(reg["success"], 4), "calls_default_unguarded": su["f_calls_per_instance"]["mean"],
                      "calls_registered": reg["f_calls_per_instance"]["mean"]}
        print(f"  {name}: plan_beam default max_depth 80: {leak[name]['unguarded_keys_outside_range']} of "
              f"{leak[name]['keys']} keys outside T2's range (guard refuses {leak[name]['guarded_refused']}); success "
              f"{leak[name]['success_default_unguarded']} (default, unguarded) vs {leak[name]['success_registered']} "
              f"(registered max_depth {reg['max_depth']})", flush=True)
    NUM["default_max_depth_leak"] = leak
    check("plan_beam's own default max_depth (80) lets a long search leave the closure: at least one configuration "
          "queries F outside T2's verified range (which is why max_depth is part of the registered run; the ORACLE "
          "success the registered max_depth gives up is reported, not thresholded)",
          any(v["unguarded_keys_outside_range"] > 0 for v in leak.values()),
          {k: (v["unguarded_keys_outside_range"], v["success_default_unguarded"], v["success_registered"]) for k, v in leak.items()})
    noisy = {}
    for sim in ("SIM_T2", "SIM_T2_P3", "SIM_OFF"):
        for name, prop, W, E, R in (("enum_W2E12", PR.EnumerateProposer(), 2, 12, 0), ("enum_W4E24", PR.EnumerateProposer(), 4, 24, 0),
                                    ("random_W4K32R3E8", PR.RandomProposer(K=32, length=6), 4, 8, 3)):
            s, _, _, f, inn = run_beam(prop, W, E, R, sim=sim)
            noisy[f"{sim}/{name}"] = {"refused": len(f.refused), "refused_share": f.oor_key_share(),
                                      "inner_inputs_out_of_range": sum(not P.in_verified_range(k[0], k[1]) for k in inn.requested),
                                      "refused_reaching_inner": len(f.refused & inn.requested)}
    NUM["noisy_f_guard"] = noisy
    check("a NOISY F (SimulatedF) can imagine a wrong state beyond the bound; the guard refuses such inputs before the "
          "inner F (no refused key reaches it; every admitted input in range) - the guard's remaining role",
          all(v["inner_inputs_out_of_range"] == 0 and v["refused_reaching_inner"] == 0 for v in noisy.values()),
          {k: v["refused"] for k, v in noisy.items()})

    print("== descriptive out-of-range set (ORACLE F, unguarded; report only)", flush=True)
    oor_rows = {}
    for name, prop, W, E, R in (("enum_W2E12", PR.EnumerateProposer(), 2, 12, 0), ("enum_W4E24", PR.EnumerateProposer(), 4, 24, 0),
                                ("random_W4K8R3E8", PR.RandomProposer(K=8, length=6), 4, 8, 3)):
        s, _, _, f, inn = run_beam(prop, W, E, R, guard=False, recs=OOR)
        oor_rows[name] = {"success": round(s["success"], 4), "calls_mean": s["f_calls_per_instance"]["mean"],
                          "share_keys_outside_verified_range": round(sum(not P.in_verified_range(k[0], k[1]) for k in inn.requested)
                                                                    / max(1, len(inn.requested)), 4)}
    s, _, _, _, _ = run_fixed("ORACLE", guard=False, recs=OOR)
    oor_rows["select_fixed"] = {"success": s["success"], "calls_mean": s["f_calls_per_instance"]["mean"]}
    NUM["oor_descriptive_oracle"] = oor_rows
    print(f"  {oor_rows}", flush=True)

    print("== hard matched budgets (RangeGuardF(BudgetF(ORACLE)), primary)", flush=True)
    cb_in = {i["id"]: cb.get(i["id"], 0) for i in insts}
    prop8 = PR.RandomProposer(K=8, length=6)
    ref_eps = BP.plan_beam(PI, prop8, LF.RangeGuardF(LF.LedgerF("ORACLE"), applies_to=GUARD), eval_seed=0,
                           **PR.beam_kwargs(prop8, 4, 8, 3))
    fb, _, mid = make_f("ORACLE", insts, budget=10 ** 9)
    eps_big = BP.plan_beam(PI, prop8, fb, eval_seed=0, **PR.beam_kwargs(prop8, 4, 8, 3))
    check("BudgetF with an unbinding cap reproduces the unbudgeted (guarded) plan_beam episodes exactly",
          eps_big == ref_eps and mid.refused == 0)
    matched = {}
    for name, prop, W, E, R in (("enum_W1E64", PR.EnumerateProposer(), 1, 64, 0), ("enum_W2E64", PR.EnumerateProposer(), 2, 64, 0),
                                ("enum_W4E64", PR.EnumerateProposer(), 4, 64, 0),
                                ("random_W2K4R2E64", PR.RandomProposer(K=4, length=6), 2, 64, 2),
                                ("random_W4K8R3E64", PR.RandomProposer(K=8, length=6), 4, 64, 3),
                                ("random_oneshot_K8", PR.RandomProposer(K=8, length=6), 1, 1, 0)):
        for mult in (1.0, 1.5, 2.0):
            cap = {k: int(v * mult) for k, v in cb_in.items()}
            s, _, _, f, _ = run_beam(prop, W=W, E=E, R=R, budget=cap)
            charged = f.inner.calls_by_owner()
            matched[f"{name}@{mult}x"] = {"oracle_success": round(s["success"], 4),
                                          "success_by_lookahead": s["success_by_lookahead"],
                                          "calls_per_instance": s["f_calls_per_instance"]["mean"],
                                          "within_cap": all(charged.get(i["id"], 0) <= cap[i["id"]] for i in insts)}
            print(f"  {name} at {mult}x the fixed baseline's per-instance calls: ORACLE success {s['success']:.3f} "
                  f"({s['f_calls_per_instance']['mean']} calls; fixed-candidate selection: 1.000 at "
                  f"{round(statistics.fmean(cb_in.values()), 2)})", flush=True)
    NUM["matched_to_fixed_calls_oracle"] = {"fixed_calls_per_instance": round(statistics.fmean(cb_in.values()), 2),
                                            "fixed_select_success": s_or["success"], "runs": matched}
    check("BudgetF caps every instance at mult x its fixed-candidate call count (hard matched budget)",
          all(v["within_cap"] for v in matched.values()), {k: v["oracle_success"] for k, v in matched.items()})

    print("== GPU-call estimates (primary, guarded, registered max_depth; ORACLE and simulated noisy F)", flush=True)
    est_arms = {"ORACLE": None, "SIM_T2": "SIM_T2", "SIM_T2_P3": "SIM_T2_P3", "SIM_OFF": "SIM_OFF"}
    bundle = [("fixed_K8", None, 0, 0, 0, None), ("enum_W2E12", "enum", 2, 12, 0, 1), ("enum_W4E24", "enum", 4, 24, 0, 1),
              ("random_oneshot_K32", "random", 1, 1, 0, 32), ("random_W4K8R3E8", "random", 4, 8, 3, 8),
              ("random_W4K32R3E8", "random", 4, 8, 3, 32)]

    def run_cfg(f, recs, name, kind, W, E, R, K):
        if kind is None:
            eps = PR.select_fixed(recs, f, eval_seed=0)
            ub = statistics.fmean(i["fixed"]["K"] * i["opt_depth"] for i in recs)
        else:
            prop = PR.EnumerateProposer() if kind == "enum" else PR.RandomProposer(K=K, length=6)
            eps = BP.plan_beam(P.planner_instances(recs), prop, f, eval_seed=0, **PR.beam_kwargs(prop, W, E, R))
            ub = (E + W - 1) * (8 if kind == "enum" else K) * (1 if kind == "enum" else (R or 6))
        return eps, ub

    est = {"n_instances": len(insts), "rates_gens_per_s": RATES, "arms": {}}
    for arm, sim in est_arms.items():
        inner = LF.SimulatedF(sim) if sim else LF.LedgerF("ORACLE")
        f = LF.RangeGuardF(inner, applies_to=GUARD)
        cfgs = {}
        for name, kind, W, E, R, K in bundle:
            inner.reset_accounting()
            eps, ub = run_cfg(f, insts, name, kind, W, E, R, K)
            s, _ = P.score(insts, eps, inner.calls_by_owner(own))
            c = s["f_calls_per_instance"]
            cfgs[name] = {"success": round(s["success"], 4), "success_by_lookahead": s["success_by_lookahead"],
                          "precision": s["verification_precision"], "calls_mean": c["mean"], "calls_p90": c["p90"],
                          "calls_max": c["max"], "calls_upper_bound_per_instance": round(ub, 1),
                          "gpu_h_per_100_instances": {k: round(c["mean"] * 100 / r_ / 3600, 2) for k, r_ in RATES.items()},
                          "gpu_h_per_100_instances_at_upper_bound": {k: round(ub * 100 / r_ / 3600, 2) for k, r_ in RATES.items()}}
        union = round(inner.model_queries / len(insts), 2)
        est["arms"][arm] = {"configs": cfgs, "bundle_union_calls_per_instance": union, "refused_unique": len(f.refused),
                            "bundle_union_gpu_h_per_100_instances": {k: round(union * 100 / r_ / 3600, 2) for k, r_ in RATES.items()}}
        print(f"  {arm}: " + "; ".join(f"{n} {v['calls_mean']} (succ {v['success']})" for n, v in cfgs.items())
              + f"; bundle union {union} calls/inst; refused {len(f.refused)}", flush=True)

    def plan_h(arms, cfg_names, n_inst, recs=None, guard=True, rate="observed_3.8"):
        recs = insts if recs is None else recs
        tot = 0.0
        for a in arms:
            inner_ = LF.SimulatedF(est_arms[a]) if est_arms[a] else LF.LedgerF("ORACLE")
            f_ = LF.RangeGuardF(inner_, applies_to=P.in_range_actions(recs) if guard else ())
            for name, kind, W, E, R, K in bundle:
                if name in cfg_names:
                    run_cfg(f_, recs, name, kind, W, E, R, K)
            tot += inner_.model_queries / len(recs) * n_inst / RATES[rate] / 3600
        return round(tot, 2)
    est["plans"] = {
        "primary_fixed+enumW2E12_T2+OFF_per_100": plan_h(("SIM_T2_P3", "SIM_OFF"), ("fixed_K8", "enum_W2E12"), 100),
        "primary_fixed+enumW2E12+enumW4E24_T2+OFF_per_100": plan_h(("SIM_T2_P3", "SIM_OFF"), ("fixed_K8", "enum_W2E12", "enum_W4E24"), 100),
        "full_bundle_T2+OFF_per_100": plan_h(("SIM_T2_P3", "SIM_OFF"), tuple(n for n, *_ in bundle), 100),
        "one_surface_audit_arm_fixed+enumW2E12_per_100_(SIM_OFF_proxy)": plan_h(("SIM_OFF",), ("fixed_K8", "enum_W2E12"), 100),
        "oor_descriptive_T2_only_fixed+enumW2E12_per_30_(SIM_OFF_proxy,_unguarded)": plan_h(("SIM_OFF",), ("fixed_K8", "enum_W2E12"), 30, recs=OOR, guard=False),
        "rate": "observed_3.8 gens/s; T2 sized with SIM_T2_P3 (the noisier T2 preset), OFF / audit / oor arms with SIM_OFF",
        "anchor_and_load_h": 0.15}
    NUM["gpu_estimates"] = est
    print(f"  plans (GPU h): {est['plans']}", flush=True)
    t2_up = est["arms"]["SIM_T2_P3"]["configs"]["enum_W2E12"]["calls_mean"] >= est["arms"]["ORACLE"]["configs"]["enum_W2E12"]["calls_mean"]
    off_up = est["arms"]["SIM_OFF"]["configs"]["enum_W2E12"]["calls_mean"] > est["arms"]["SIM_T2_P3"]["configs"]["enum_W2E12"]["calls_mean"]
    check("estimates: a noisier F costs more calls (SIM_OFF > SIM_T2_P3 >= ORACLE on enum W2E12); every estimate is "
          "bounded by the per-instance upper bound; ORACLE refuses nothing", t2_up and off_up
          and est["arms"]["ORACLE"]["refused_unique"] == 0
          and all(v["calls_max"] <= v["calls_upper_bound_per_instance"] + 1e-9 for a in est["arms"].values()
                  for n, v in a["configs"].items() if n != "fixed_K8"),
          {a: est["arms"][a]["configs"]["enum_W2E12"]["calls_mean"] for a in est_arms})
    NUM["seconds"] = round(time.time() - t0, 1)
    return {"set_sha256_seen": {k: P.set_sha(v) for k, v in sets.items()}}


# ================================================================================================ part: tiny
CFG = dict(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
           num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, rms_norm_eps=1e-6, rope_theta=1e6,
           tie_word_embeddings=True)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def part_tiny() -> dict:
    torch = import_torch_cpu()
    torch.Tensor.cuda = lambda self, *a, **k: self        # generation.generate_texts moves inputs with .cuda()
    import transformers
    from transformers import LogitsProcessorList, Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model

    def _no_real_models(*a, **k):
        raise AssertionError("a real model load was attempted inside the tiny test")
    transformers.AutoModelForCausalLM.from_pretrained = _no_real_models
    from rcwmt import env, generation, runtime
    runtime.gpu_temp_c = lambda: None                    # never spawn nvidia-smi
    generation.breathe = lambda: None                    # pacing only
    generation.throttle_list = lambda: LogitsProcessorList([])
    import lbeam_problems as P
    import lbeam_proposer as PR
    import lbeam_f as LF
    import p3_common as P3C
    BP = import_plan_beam()
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="lbeam_tiny_", dir=SCRATCH if SCRATCH.exists() else None))

    @torch.no_grad()
    def build(seed):
        cfg = Qwen3Config(**CFG)
        cfg._attn_implementation = "sdpa"
        torch.manual_seed(seed)
        m = Qwen3ForCausalLM(cfg)
        g = torch.Generator().manual_seed(seed)
        for _, p in m.named_parameters():
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
        return m.eval()

    with torch.no_grad():
        m = build(2002)
        pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM"))
        g = torch.Generator().manual_seed(31)
        for n, p in pm.named_parameters():
            if "lora_B" in n or "lora_A" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
        pm.save_pretrained(str(tmp / "adapter"))
        del pm, m
    base = build(2002)
    model = PeftModel.from_pretrained(base, str(tmp / "adapter" / "default") if (tmp / "adapter" / "default").exists()
                                      else str(tmp / "adapter"), adapter_name="t2", is_trainable=False).eval()
    plain = build(2002)                                   # the base alone (= what OFF must equal)
    tok = runtime.tokenizer()
    print(f"tiny models ready ({time.time() - t0:.1f}s, peak {peak_mb()} MB)", flush=True)

    sets, _, _ = load_sets()
    insts, OOR = sets["primary"], sets["oor"]
    PI = P.planner_instances(insts)
    own = P.owner_of_action(insts)
    GUARD = P.in_range_actions(insts)
    keys = []
    for i in insts[:5]:
        keys += [(i["init"], a) for a in i["vocab"][:2]]
    MAXNEW = 12

    # 0 the canonical frame only
    sig = inspect.signature(LF.LedgerF)
    errs = []
    for fn in (lambda: LF.LedgerF("T2", model=model, tok=tok, surface="keys_reversed"),
               lambda: LF._ModelPathF("ORACLE")):
        try:
            fn()
            errs.append(False)
        except TypeError:
            errs.append(True)
    try:
        LF.SurfaceAuditF("T2", "train", model=model, tok=tok)
        errs.append(False)
    except ValueError:
        errs.append(True)
    vk = [(i["init"], a) for i in insts[:30] for a in i["vocab"]] + [(i["init"], a) for i in OOR[:5] for a in i["vocab"]]
    check("LedgerF renders ONLY the canonical frame: no surface parameter (passing one is a TypeError), the base class is "
          "not constructible, SurfaceAuditF refuses the canonical name and is not a LedgerF; LF.messages == "
          "env.one_step_messages and is_canonical_frame holds on every vocabulary key of 30 primary + 5 oor instances "
          "(sorted keys at every level, accounts A-D, posted / void)",
          "surface" not in sig.parameters and all(errs)
          and not issubclass(LF.SurfaceAuditF, LF.LedgerF)
          and all(LF.messages(s, a) == env.one_step_messages("ledger_replay", P.state_to_dict(s), P.action_to_event(a))
                  and LF.is_canonical_frame(LF.messages(s, a), s, a) for s, a in vk)
          and not LF.is_canonical_frame(LF.audit_messages(*vk[0], "keys_reversed"), *vk[0])
          and not LF.is_canonical_frame(LF.audit_messages(*vk[0], "names_letters"), *vk[0])
          and set(P.action_to_event(keys[0][1])) == {"account", "amount", "kind", "other", "status", "tick"},
          f"{len(vk)} keys; errs {errs}")
    called = []

    def spy_gen(m_, t_, msgs, n):
        called.append(len(msgs))
        return ["{}"] * len(msgs)
    real_messages = LF.messages
    LF.messages = lambda s, a: LF.audit_messages(s, a, "keys_reversed")
    try:
        fx = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", generate_fn=spy_gen)
        try:
            fx.predict(keys[:3])
            raised = False
        except RuntimeError as e:
            raised = "canonical" in str(e)
    finally:
        LF.messages = real_messages
    check("LedgerF enforces the canonical frame at generation time: if its renderer were swapped for a non-canonical "
          "one (reordered keys), predict raises BEFORE any generation call", raised and not called and not fx.memo)

    # 1 frame + decoding identity with the registered chain_one step
    f = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", max_new_tokens=MAXNEW)
    res = f.predict(keys)
    generation_max = generation.max_tokens
    generation.max_tokens = lambda fam: MAXNEW
    try:
        ref = generation.chain(model, tok, [{"family": "ledger_replay", "s0": P.state_to_dict(s), "events": [P.action_to_event(a)]}
                                            for s, a in keys], "chain_one")
    finally:
        generation.max_tokens = generation_max
    check("decoding identity: LedgerF(T2) texts == rcwmt.generation.chain('chain_one') texts on the same keys "
          "(same frame, greedy settings, batching, parser)",
          [f.texts[k] for k in keys] == [r["texts"][0] for r in ref]
          and [None if x is None else x[0] for x in res] == [None if r["states"][1] is None else P.state_from_dict(r["states"][1]) for r in ref])
    check("batching: ONE generation call and ONE model.generate call for a 10-key predict list; model_queries 10; "
          "shapes (list of len(keys); None or (canonical state, True))",
          f.n_generate_fn_calls == 1 and f.n_model_generate_calls == 1 and f.prompts_per_model_generate == [10]
          and f.model_queries == 10 and len(res) == 10
          and all(x is None or (P.is_state(x[0]) and x[1] is True) for x in res), f.stats())
    check("generated_tokens: exact count of new tokens before EOS (0 < tokens <= 10 x max_new_tokens)",
          0 < f.generated_tokens <= 10 * MAXNEW and f.decode_steps <= MAXNEW, f"{f.generated_tokens} tokens")
    n_none = sum(x is None for x in res)
    check("None on unparseable: the random tiny model's answers do not parse -> None (memoised as None)",
          n_none == 10 and all(f.memo[k] is None for k in keys), f"{n_none}/10 None; sample {f.texts[keys[0]][:40]!r}")
    res2 = f.predict(keys + keys[:3])
    check("memoisation: a repeated list (with duplicates) makes no model call; model_queries unchanged",
          res2 == res + res[:3] and f.n_generate_fn_calls == 1 and f.model_queries == 10 and f.n_predict_new == 1)
    more = [(i["init"], a) for i in insts[5:7] for a in i["vocab"][:2]]
    f.predict(keys[:4] + more + more[:1])
    check("memoisation: a mixed list sends only its NEW distinct keys, in one call (4 new prompts)",
          f.n_generate_fn_calls == 2 == f.n_predict_new and f.prompts_per_model_generate == [10, 4] and f.model_queries == 14
          and len(f.requested) == 14, f.prompts_per_model_generate)
    long_keys = [(i["init"], a) for i in insts[10:40] for a in i["vocab"][:2]]
    f3 = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", max_new_tokens=MAXNEW)
    f3.predict(long_keys)
    check("batching: a 60-key list is ONE generation call; generate_texts splits it into registered sub-batches "
          "(<= 32 prompts, batch x (prompt + max_new) <= 8000 tokens)",
          f3.n_generate_fn_calls == 1 and sum(f3.prompts_per_model_generate) == 60
          and max(f3.prompts_per_model_generate) <= 32, f3.prompts_per_model_generate)

    # 2 OFF vs T2 in one process
    f_off = LF.LedgerF("OFF", model=model, tok=tok, max_new_tokens=MAXNEW)
    f_off.predict(keys)
    generation.max_tokens = lambda fam: MAXNEW
    try:
        ref_plain = generation.chain(plain, tok, [{"family": "ledger_replay", "s0": P.state_to_dict(s), "events": [P.action_to_event(a)]}
                                                  for s, a in keys], "chain_one")
    finally:
        generation.max_tokens = generation_max
    check("OFF arm == the base model alone (disable_adapter in the same process); T2 texts differ (adapter active); "
          "the adapter is re-enabled after OFF",
          [f_off.texts[k] for k in keys] == [r["texts"][0] for r in ref_plain]
          and [f_off.texts[k] for k in keys] != [f.texts[k] for k in keys] and LF.peft_side(model) == "t2"
          and not LF.any_merged(model))

    # 3 parsing paths (stub generator: no model call)
    crafted = ['{"A":1,"B":2,"C":3,"D":4}', '{"A":1,"B":2,"C":3}', '{"A":1,"B":2,"C":3,"D":4,"E":5}', "garbage",
               '{"before":{"A":0,"B":0,"C":0,"D":0},"after":{"A":5,"B":6,"C":7,"D":8}}', '{"A":true,"B":2,"C":3,"D":4}',
               '{"A":1.5,"B":2,"C":3,"D":4}', '{"A": 3 + 4 = 7, "B": 1, "C": 2, "D": 3}', '{"D":4,"C":3,"B":2,"A":1}.']
    want = [((("A", 1), ("B", 2), ("C", 3), ("D", 4)), True), None, None, None,
            ((("A", 5), ("B", 6), ("C", 7), ("D", 8)), True), None, None, ((("A", 7), ("B", 1), ("C", 2), ("D", 3)), True),
            ((("A", 1), ("B", 2), ("C", 3), ("D", 4)), True)]
    seen_n = []

    def crafted_gen(m_, t_, msgs, n):
        seen_n.append(n)
        return crafted[:len(msgs)]
    fs = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", generate_fn=crafted_gen)
    ck = [(i["init"], i["vocab"][0]) for i in insts[:len(crafted)]]
    got = fs.predict(ck)
    check("parser: env.parse_state(lenient) paths; wrong account set / extra account / non-int / garbage -> None; "
          "legal=True for every parsed answer (the ledger has no illegal events)", got == want, got)
    check("default decode cap = rcwmt.generation.max_tokens('ledger_replay') = 160 (PREREG-2's)",
          seen_n == [160] == [generation_max("ledger_replay")], seen_n)

    # 4 SurfaceAuditF (D-35 audit renderings, report-only): rendering identity with PREREG-3's helpers, parse inversion
    s0, a0 = insts[0]["init"], next(a for a in insts[0]["vocab"] if a[4] == "void")
    ok_render, detail = True, {}
    for name, sf in LF.SURFACES.items():
        for s, a in ((s0, a0), keys[0]):
            msgs = LF.audit_messages(s, a, name)
            u = msgs[1]["content"]
            b, e, ko = LF.surface_payload(s, a, name)
            good = msgs == P3C.messages(b, e, ko) and msgs[0]["content"] == env.SYSTEM_ONE["ledger_replay"]
            d = json.loads(u)
            names = sf.get("names") or P.ACCOUNTS
            good = good and list(d["before"]) == (sorted(names, reverse=True) if sf.get("key_order") == "reversed" else sorted(names))
            good = good and (list(d)[0] == "event") == bool(sf.get("key_order"))
            good = good and d["event"]["status"] == (sf.get("void_as", "void") if a[4] == "void" else "posted")
            good = good and d["event"]["account"] == dict(zip(P.ACCOUNTS, names))[a[1]]
            stays_canonical = name in ("status_cancelled", "status_pending") and a[4] != "void"
            good = good and LF.is_canonical_frame(msgs, s, a) == stays_canonical
            ok_render &= good
            detail[name] = u[:90]
    check("SurfaceAuditF renderings (report-only): every audit surface == p3_common.messages(renamed payload, key order) "
          "with Titan's system text; names, key order and the void status word rendered as specified; each differs from "
          "the canonical frame wherever it changes the payload (a status surface leaves a posted event canonical)",
          ok_render, detail["heldout_combined"])
    words = LF.WORD_NAMES4
    pa = [LF.audit_parse('{"bank":1,"card":2,"cash":3,"fees":4}', s0, "names_words"),
          LF.audit_parse('{"fees":4,"cash":3,"card":2,"bank":1}.', s0, "heldout_combined"),
          LF.audit_parse('{"A":1,"B":2,"C":3,"D":4}', s0, "names_words"),
          LF.audit_parse('{"E":1,"F":2,"G":3,"H":4}', s0, "names_letters"),
          LF.audit_parse('{"after":{"bank":5,"card":6,"cash":7,"fees":8}}', s0, "names_words"),
          LF.audit_parse('{"A":1,"B":2,"C":3,"D":4}', s0, "keys_reversed")]
    st = lambda *v: tuple(zip(P.ACCOUNTS, v))  # noqa: E731
    check("audit parse: renamed answers map back to canonical A-D (any key order, {'after':...}); an A-D answer under a "
          "renamed surface -> None; key-order and status surfaces parse with env.parse_state",
          pa == [st(1, 2, 3, 4), st(1, 2, 3, 4), None, st(1, 2, 3, 4), st(5, 6, 7, 8), st(1, 2, 3, 4)] and words == tuple(sorted(words)), pa)

    # 5 rule-following stub generators through the full model path (and every prompt they receive is canonical)
    non_canonical_seen = []

    def stub(rule_level: bool, status_blind_when_off: bool = True, noise=None, canonical_expected=True):
        def gen(model_, tok_, msgs, n):
            side = LF.peft_side(model_)
            out = []
            for m_ in msgs:
                u = json.loads(m_[1]["content"])
                b, e = u["before"], dict(u["event"])
                if canonical_expected:
                    try:
                        kk = (P.state_from_dict(b), P.event_to_action(e))
                        non_canonical_seen.append(not LF.is_canonical_frame(m_, *kk))
                    except Exception:
                        non_canonical_seen.append(True)
                if side == "off" and status_blind_when_off:
                    e["status"] = "posted"
                elif not rule_level and e["status"] in ("void", "Void", "VOID", "voided"):
                    pass                                          # trigger-level: only 'void' forms are a no-op
                elif not rule_level:
                    e["status"] = "posted"
                ans = P3C.rule_gold(b, e)                         # the frame's stated rule, any names
                if noise is not None:
                    ans = noise(b, e, ans)
                out.append(json.dumps(ans))
            return out
        return gen
    sub = P.planner_instances(insts[:24])
    prop = PR.RandomProposer(K=8, length=6)

    def beam(fobj, p=prop, W=4, E=8, R=0):
        return BP.plan_beam(sub, p, fobj, eval_seed=0, **PR.beam_kwargs(p, W, E, R))
    ft = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", generate_fn=stub(True))
    eps_t = beam(ft)
    eps_o = beam(LF.LedgerF("ORACLE"))
    eps_off = beam(LF.LedgerF("OFF", model=model, tok=tok, generate_fn=stub(True)))
    eps_ig = beam(LF.LedgerF("IGNORE_STATUS"))
    check("text path end to end: LedgerF(T2) with a rule-following text generator reproduces ORACLE's plan_beam "
          "episodes exactly; LedgerF(OFF) with a status-blind generator reproduces IGNORE_STATUS's (24 instances, W4 K8 E8, "
          "registered max_depth); one generation call per predict call with new keys; model_queries == distinct keys "
          "requested; every prompt the generator received was the canonical frame",
          eps_t == eps_o and eps_off == eps_ig and eps_t != eps_ig and ft.n_generate_fn_calls == ft.n_predict_new > 1
          and ft.n_predict_new <= ft.n_predict and ft.model_queries == len(ft.requested) == len(ft.memo)
          and non_canonical_seen and not any(non_canonical_seen),
          {**ft.stats(), "prompts_checked": len(non_canonical_seen)})
    ok_sf, det_sf = True, {}
    for name in LF.SURFACES:
        e_rule = beam(LF.SurfaceAuditF("T2", name, model=model, tok=tok, adapter_name="t2",
                                       generate_fn=stub(True, canonical_expected=False)))
        e_trig = beam(LF.SurfaceAuditF("T2", name, model=model, tok=tok, adapter_name="t2",
                                       generate_fn=stub(False, canonical_expected=False)))
        void_heldout = bool(LF.SURFACES[name].get("void_as"))
        ok_sf &= e_rule == eps_o and (e_trig == (eps_ig if void_heldout else eps_o))
        det_sf[name] = (e_rule == eps_o, e_trig == eps_o)
    e_trig_canon = beam(LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", generate_fn=stub(False)))
    check("the D-35 audit discriminates (SurfaceAuditF, report-only): a RULE-level F reproduces ORACLE's episodes under "
          "every audit surface; a TRIGGER-level F ('void' forms only) reproduces ORACLE in the canonical frame and under "
          "the key-order and naming surfaces, and IGNORE_STATUS under the held-out status surfaces",
          ok_sf and e_trig_canon == eps_o, det_sf)

    # 6 the guard on a model arm: a noisy text generator imagines states beyond the bound; they never reach generation
    sent = []

    def noise(b, e, ans):
        h = int(hashlib.sha256(json.dumps([b, e], sort_keys=True).encode()).hexdigest()[:8], 16)
        if h % 4 == 0:
            k = sorted(ans)[h % 3]
            ans = {**ans, k: ans[k] + (70 if ans[k] >= 0 else -70)}
        return ans

    def rec_gen(model_, tok_, msgs, n):
        sent.extend(json.loads(m_[1]["content"])["before"] for m_ in msgs)
        return stub(True, noise=noise)(model_, tok_, msgs, n)
    fg_in = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", generate_fn=rec_gen)
    fg = LF.RangeGuardF(fg_in, applies_to=GUARD)
    BP.plan_beam(sub, PR.RandomProposer(K=32, length=6), fg, eval_seed=0,
                 **PR.beam_kwargs(PR.RandomProposer(K=32, length=6), 4, 8, 3))
    check("RangeGuardF on a model arm: a noisy generator imagines states beyond |b| = 98; no prompt with |balance| > 98 "
          "reaches the generation function; refused keys are not model calls; exposure logged",
          sent and all(abs(v) <= 98 for b in sent for v in b.values())
          and len(fg.refused) > 0 and fg_in.model_queries == len(sent) == len(fg.seen) - len(fg.refused), fg.stats())

    # 7 plan_beam and select_fixed with the real tiny model (everything unparseable: candidates stop, no crash)
    sub4 = PI[:4]
    ftt = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", max_new_tokens=MAXNEW)
    p4 = PR.RandomProposer(K=4, length=3)
    eps = BP.plan_beam(sub4, p4, ftt, eval_seed=0, **PR.beam_kwargs(p4, 2, 2, 3))
    summ, _ = P.score(insts[:4], eps, ftt.calls_by_owner(own))
    check("plan_beam end to end with the tiny T2 model: runs; one generation call per predict list with new keys; "
          "model_queries == distinct keys == per-instance attribution; unparseable answers end candidates (dead_end)",
          ftt.n_generate_fn_calls == ftt.n_predict_new == ftt.n_predict
          and ftt.model_queries == len(ftt.requested) == sum(ftt.calls_by_owner(own).values())
          and all(e["fail"] == "dead_end" for e in eps), {**ftt.stats(), "summary_success": summ["success"]})
    ff = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", max_new_tokens=MAXNEW)
    eps_f = PR.select_fixed(insts[:4], ff, eval_seed=0)
    check("select_fixed with the tiny T2 model: F calls == the K candidates' first steps (all unparseable), ties broken "
          "at random, nothing verified", ff.model_queries == sum(len({c[0] for c in r["fixed"]["cands"]}) for r in insts[:4])
          and not any(e["verified"] for e in eps_f), ff.stats())

    # 8 persistent cache and loader injection
    cpath = tmp / "fcache.jsonl"
    fc = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", max_new_tokens=MAXNEW, cache=cpath, identity={"adapter": "tiny"})
    r_a = fc.predict(keys)
    fc2 = LF.LedgerF("T2", model=model, tok=tok, adapter_name="t2", max_new_tokens=MAXNEW, cache=cpath, identity={"adapter": "tiny"})
    r_b = fc2.predict(keys)
    refused = []
    for mk in (lambda: LF.LedgerF("OFF", model=model, tok=tok, cache=cpath, identity={"adapter": "tiny"}),
               lambda: LF.SurfaceAuditF("T2", "keys_reversed", model=model, tok=tok, cache=cpath, identity={"adapter": "tiny"})):
        try:
            mk()
            refused.append(False)
        except RuntimeError:
            refused.append(True)
    check("cache: reloaded answers give identical results with 0 new model calls; another arm or an audit surface "
          "refuses the file", r_a == r_b and fc2.model_queries == 0 and fc2.cache_loaded == 10 and all(refused), fc2.stats())
    calls = []

    def loader_fn():
        calls.append(1)
        return model, tok
    sl = LF.SharedLoader(loader_fn)
    fa = LF.LedgerF("T2", loader=sl, adapter_name="t2", max_new_tokens=MAXNEW)
    fb = LF.LedgerF("OFF", loader=sl, max_new_tokens=MAXNEW)
    no_call_yet = not calls
    fa.predict(keys[:2])
    fb.predict(keys[:2])
    try:
        LF.load_registered_t2()
        guarded_load = False
    except RuntimeError as e:
        guarded_load = "GPU-only" in str(e)
    check("loader injection: lazy, called once for two arms sharing one SharedLoader; the registered loader refuses "
          "on CPU before touching any model file", no_call_yet and len(calls) == 1 and sl.calls == 1 and guarded_load)
    bad = []
    for fn in (lambda: f.predict([({"A": 1}, keys[0][1])]), lambda: fg.predict([({"A": 1}, keys[0][1])]),
               lambda: LF.SurfaceAuditF("ORACLE", "names_words"), lambda: LF.SurfaceAuditF("T2", "nope", model=model, tok=tok)):
        try:
            fn()
            bad.append(False)
        except ValueError:
            bad.append(True)
    check("malformed keys raise in LedgerF and RangeGuardF (programming errors, not 'unsupported'); an audit surface on "
          "a model-free arm or an unknown surface raises", all(bad), bad)

    # 9 SimulatedF: deterministic per key, rates near the preset
    sk = [(i["init"], a) for i in insts for a in i["vocab"]]
    s1, s2 = LF.SimulatedF("SIM_T2"), LF.SimulatedF("SIM_T2")
    r1 = s1.predict(sk)
    r2 = s2.predict(list(reversed(sk)))[::-1]
    posted = [(k, r) for k, r in zip(sk, r1) if k[1][4] == "posted"]
    wrong = sum(r[0] != P.step_state(*k) for k, r in posted) / len(posted)
    off = LF.SimulatedF("SIM_OFF").predict(sk)
    voids = [(k, r) for k, r in zip(sk, off) if k[1][4] == "void"]
    applied = sum(r[0] != k[0] for k, r in voids) / len(voids)        # a void step changed the state
    check("SimulatedF (diagnostic, sizing only): a deterministic function of the key (order-independent); SIM_T2 posted "
          "error rate and SIM_OFF void-applied rate near their presets", r1 == r2 and abs(wrong - 0.09) < 0.04
          and abs(applied - 0.90) < 0.08, f"SIM_T2 wrong {wrong:.3f} on {len(posted)}; SIM_OFF void applied {applied:.3f} on {len(voids)}")
    shutil.rmtree(tmp, ignore_errors=True)
    NUM["tiny_stats_T2"] = f.stats()
    NUM["seconds"] = round(time.time() - t0, 1)
    return {}


# ================================================================================================ part: fresh
def part_fresh() -> dict:
    import lbeam_problems as P
    import lbeam_f as LF
    t0 = time.time()
    sets, src, brecs = load_sets()
    NUM["sets"] = {k: {"source": src[k], "sha256": P.set_sha(v), "n": len(v)} for k, v in sets.items()}
    excl = P.load_exclusions(historical=True)
    NUM["exclusion_sources"] = excl["sources"]
    NUM["exclusion_union"] = {"payloads": len(excl["pay"]), "operative": len(excl["ops"])}
    for kind, insts in sets.items():
        srec = brecs[kind]
        fr = srec.get("freshness")
        built = (fr.get("sources") if isinstance(fr, dict) else None) or {}
        changed = {k: {"built": (built.get(k) or {}).get("sha256"), "now": v.get("sha256")}
                   for k, v in excl["sources"].items()
                   if v.get("sha256") and (k not in built or built[k].get("sha256") != v.get("sha256"))}
        NUM[f"{kind}_sources_changed_since_build"] = {"changed_or_new": changed,
                                                      "gone": [k for k in built if k not in excl["sources"]],
                                                      "union_at_build": srec.get("exclusion_union_size")}
        print(f"  {kind}: union now {len(excl['pay'])} payloads / {len(excl['ops'])} operative (at build: "
              f"{srec.get('exclusion_union_size')}); sources changed or new since the build: {sorted(changed)}", flush=True)
    sk = excl["sources"]["historical"]["skipped_by_name_never_opened"]
    check("historical harvest: only HSKILL_CONFIRM_V2* files skipped, by name (never opened); "
          "rcwmt.bank.load_exclusions() not called", sk and all(Path(p).name.startswith("HSKILL_CONFIRM_V2") for p in sk)
          and "rcwmt.bank" not in sys.modules, sk)
    L1 = P._import_builders().L1
    for kind, insts in sets.items():
        ov = P.overlap_counts(insts, excl)
        NUM[f"{kind}_binding_overlap"] = ov
        check(f"{kind} set vs the CURRENT union: zero payload and zero operative-content overlap on every binding "
              f"transition (init x vocab, optimal plan, fixed candidates)",
              ov["payload_overlap"] == 0 and ov["operative_overlap"] == 0, ov)
        NUM[f"{kind}_ball_overlap_depth2_descriptive"] = P.ball_overlap(insts, excl, 2)
    sov = {}
    for name in LF.SURFACES:
        n = p = 0
        for i in sets["primary"]:
            for b, e in P.binding_transitions(i):
                bb, ee, _ = LF.surface_payload(P.state_from_dict(b), P.event_to_action(e), name)
                n += 1
                p += L1.payload(bb, ee) in excl["pay"]
        sov[name] = {"transitions": n, "payload_overlap": p}
    NUM["primary_audit_rendering_payload_overlap"] = sov
    check("SurfaceAuditF renderings of every primary binding transition (renamed accounts, held-out status words; key "
          "order does not change a payload): zero payload overlap with the union (incl. PREREG-3's renamed / status rows)",
          all(v["payload_overlap"] == 0 for v in sov.values()), {k: v["payload_overlap"] for k, v in sov.items()})
    for kind, insts in sets.items():
        srec = brecs[kind]
        regen, rec = P.make_set(P.spec_from_record(srec) if "spec" in srec else (P.primary_spec() if kind == "primary" else P.oor_spec()), excl)
        NUM[f"{kind}_fresh_rejections"] = rec["fresh_rejections"]
        check(f"{kind}: regeneration WITH the current exclusion union reproduces the set sha256 and the recorded freshness "
              f"rejections exactly", rec["file_sha256"] == P.set_sha(insts)
              and rec["fresh_rejections"] == srec.get("fresh_rejections", rec["fresh_rejections"]),
              f"{rec['file_sha256'][:16]}, {len(rec['fresh_rejections'])} rejection(s)")
    MB = P._import_builders()
    allt = [bt for insts in sets.values() for i in insts for bt in P.binding_transitions(i)]
    pay = {L1.payload(b, e) for b, e in allt}
    ops = {L1.opkey("ledger_replay", b, e) for b, e in allt}
    del excl
    cen = MB.status_census(bank_pay=pay, bank_ops=ops)
    mer = cen["multi_event_rollouts"]
    NUM["multi_event_rollouts"] = mer
    check("zero overlap (both sets) with every multi-event ledger prompt rolled out in the records tree and the pinned "
          "sources (the same name-skipping walk)", mer["payload_overlap"] == 0 and mer["operative_overlap"] == 0
          and mer["transitions"] > 0 and all(Path(p).name.startswith("HSKILL_CONFIRM_V2") for p in cen["skipped_by_name_never_opened"]), mer)
    check("no torch import in the freshness path", "torch" not in sys.modules)
    NUM["seconds"] = round(time.time() - t0, 1)
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("oracle", "tiny", "fresh"), required=True)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    nice()
    print(f"== lbeam_test part {a.part}", flush=True)
    try:
        extra = {"oracle": part_oracle, "tiny": part_tiny, "fresh": part_fresh}[a.part]()
    except Exception as e:                       # a crash is a failed check, recorded
        import traceback
        traceback.print_exc()
        check(f"part {a.part} ran to completion", False, f"{type(e).__name__}: {e}")
        extra = {}
    ok = all(r["ok"] for r in RESULTS)
    if not a.no_record:
        print(f"wrote {write_record(a.part, extra)}", flush=True)
    print(f"== {a.part}: {sum(r['ok'] for r in RESULTS)}/{len(RESULTS)} checks pass; peak {peak_mb()} MB", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
