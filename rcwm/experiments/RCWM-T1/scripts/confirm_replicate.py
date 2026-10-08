"""EXPLORATORY replication, on the sealed confirmation bank, of three pilot-only findings.

PREREG-1 is already analysed (runs/bank_confirm/analysis/PREREG_1_RESULTS.json). Nothing in this script
is part of PREREG-1, and none of it is confirmatory in the registered sense: it asks whether three
exploratory pilot findings (RESULTS_PILOT.md R8 and R10, docs/s_roll/S_ROLL_PILOT.md) hold on 150 fresh
parents per family. Every definition, hyper-parameter, head, selection rule, seed and checkpoint rule
comes from the pilot work; nothing is selected, tuned or thresholded on confirmation data.

  (A) R8 per-operation one-step failure map of Titan-Sparse (chain_one), with the R8 rule (chain steps
      whose INPUT state equals the gold state; s_roll.onestep_stats(rule='input_correct')) and on first
      steps from the gold start state (rule='step1'); strata = s_roll.strata; plus the error type of
      every wrong output (the R8 "dominant error" column).
  (B) Titan's prerequisite rule is integer addition: applicable first steps whose wrong output equals
      state + add, split by whether the add bit was already set (unambiguous carry) and whether delete
      clears a set bit; plus s_roll.titan_error_anatomy (inapplicable / void anatomy).
  (C) S-roll (R10): s_roll.py's generic one-step MLP, retrained exactly as in the pilot (pilot TRAIN
      one-step transitions, pilot DEV for checkpointing, the saved pilot selections
      s_roll_select{,_regsel}_{L,P,R}.json, seeds 17-21, resample seeds 9000+r), chained 4x on the
      confirmation candidates (twins excluded, exactly as data.candidate_table / prereg1) and scored with
      decisions.g_only (test goals seed 1001, regret_table, fresh rng 17 per method, parent-mean fill).
      Titan chain_one - S-roll per family at k = 1, 2, 4: decide.paired_bootstrap over parents.
      Main arms: FULL and TITAN-MATCHED (MATCHED128) under both selection rules (spec = crit 'exact',
      DEV-regret = crit 'regret'). Supplementary (also pilot R10 claims): TITANROWS and the N512/N2000
      learning curve.

Sanity (the run aborts if either fails):
  * Titan chain_one k=4 regret through this path == the confirmatory means (L 0.539, R 0.356, P 0.899)
    to 3 decimals, and every Titan per-parent regret == PREREG_1_RESULTS_PER_PARENT.npz (|diff| < 1e-6);
  * an oracle one-step model through the identical rollout -> phi -> regret path gives regret 0 and
    exact state 1 for every family and k (pilot and confirmation).
Reproduction checks (recorded, not tuned on): pilot (A)/(B) numbers recomputed here == s_roll.json;
every retrained S-roll model's DEV exact / best step and its pilot-TEST per-parent regrets == the
stored pilot models (s_roll_{,regsel_}{L,P,R}.json, s_roll_per_model*.npz).

CPU only (CUDA hidden, DECISIONS I-21), 2 threads, below-normal priority. Read-only on everything
except runs/bank_pilot/analysis/confirm_replicate*.{json,log,npz}.

  python scripts/confirm_replicate.py              # full run (resumable: one part file per crit x family)
  python scripts/confirm_replicate.py --parts AB   # (A) and (B) only (seconds)
  python scripts/confirm_replicate.py --smoke      # tiny training schedule, *_smoke outputs only
"""
from __future__ import annotations

import os

# --- hard rules: no GPU, 2 threads (DECISIONS I-21: CUDA_VISIBLE_DEVICES must be "" not "-1") ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["RCWM_THREADS"] = "2"
import torch  # noqa: E402

torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_num_threads(2)

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import s_roll as SR  # noqa: E402  (import only; s_roll.py is read-only)
from rcwmt import data, decide, decisions, env  # noqa: E402

torch.set_num_threads(2)
assert not torch.cuda.is_available() and torch.cuda.device_count() == 0
try:
    import psutil  # noqa: E402
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
except Exception:                                   # priority is a courtesy, never a reason to fail
    psutil = None
warnings.filterwarnings("ignore", message="Mean of empty slice")

FAMS = env.FAMILIES
SHORT = env.SHORT
KS = SR.KS
H = SR.H
SEEDS = SR.SEEDS
CONFIRM = "bank_confirm"
CONFIRM_SHA = "8bc6e2f8a8c7cae5b7b66293ef12b0efc440c281f820eca9d384a659b26a6a16"   # confirm_analysis.CONFIRM_SHA
FP_SPARSE = "818be9c6b6a3"                                                         # confirm_analysis.FP["sparse"]
CONFIRM_TITAN_K4 = {"ledger_replay": 0.539, "register_machine": 0.356, "prerequisite_plan": 0.899}
OUT = ROOT / "runs" / "bank_pilot" / "analysis"
CONF_AN = ROOT / "runs" / CONFIRM / "analysis"
CRIT_NAME = {"exact": "spec", "regret": "DEV-regret"}
MAIN_ARMS = ("FULL", "MATCHED128")
SUPP_ARMS = ("TITANROWS", "N512", "N2000")
CPU_SOFT_BUDGET_SEC = 50 * 60      # supplementary arms are skipped (and recorded) beyond this process CPU time
RSS_LIMIT_GB = 1.9


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def cpu_sec() -> float:
    if psutil is None:
        return time.process_time()
    t = psutil.Process().cpu_times()
    return float(t.user + t.system)


def rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9 if psutil else float("nan")


def guard(log, where):
    r = rss_gb()
    log(f"  [resources] {where}: rss {r:.2f} GB, process CPU {cpu_sec() / 60:.1f} min")
    if r == r and r > RSS_LIMIT_GB:
        raise SystemExit(f"RSS {r:.2f} GB > {RSS_LIMIT_GB} GB limit at {where}; aborting (shared machine)")


def code_meta():
    files = [Path(__file__).resolve(), ROOT / "scripts" / "s_roll.py"] + sorted((ROOT / "rcwmt").glob("*.py"))
    return {"code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files},
            "torch": torch.__version__, "numpy": np.__version__, "threads": torch.get_num_threads()}


def jsonable(x):
    return json.loads(json.dumps(x, default=float))


# ============================================================================ confirmation inputs

def load_gen_dir(d: Path) -> dict:
    """scripts/analyze.load_gen, inlined (analyze.py is not needed otherwise)."""
    out = {"e2e": {r["item_id"]: r for r in map(json.loads, (d / "gen_e2e.jsonl").read_text(encoding="utf-8").splitlines())}}
    for mode in ("chain_one", "chain_m1"):
        out[mode] = {r["cand_id"]: r for r in map(json.loads, (d / f"gen_{mode}.jsonl").read_text(encoding="utf-8").splitlines())}
    return out


def load_confirm(log):
    p = ROOT / "data" / f"{CONFIRM}.json"
    assert sha(p) == CONFIRM_SHA, "confirmation bank hash mismatch"
    bank = data.load_bank(CONFIRM)
    rows = data.candidate_table(bank)                       # candidates only: twins are NOT in the decision set
    assert rows and all(r["split"] == "test" for r in rows)
    gold = {(r["cand_id"], r["k"]): r for r in rows}
    cands = {f: [] for f in FAMS}
    T = {f: [] for f in FAMS}
    twins = {f: [] for f in FAMS}
    n_check = 0
    for p_ in bank["parents"]:
        fam = p_["family"]
        for c in p_["candidates"]:
            states = [p_["s0"]]
            for t, e in enumerate(c["events"]):
                env.cross_check(fam, states[-1], e)
                n_check += 1
                s2 = env.step(fam, states[-1], e)
                T[fam].append({"parent_id": p_["parent_id"], "cand_id": c["cand_id"], "t": t, "s": states[-1], "e": e, "s2": s2})
                states.append(s2)
            assert len(states) == H + 1
            for k in KS:
                r = gold[(c["cand_id"], k)]
                assert env.dumps(states[k]) == r["gold"], (c["cand_id"], k)
                assert np.array_equal(data.phi_from_state(fam, p_["s0"], c["events"][:k], states[k]), r["phi"])
            cands[fam].append({"cand_id": c["cand_id"], "parent_id": p_["parent_id"], "s0": p_["s0"],
                               "events": c["events"], "gold": states})
        if "twin" in p_:
            twins[fam].append(p_["twin"]["cand_id"])
    tw = {c for v in twins.values() for c in v}
    assert not tw & {r["cand_id"] for r in rows}, "a twin entered the decision set"
    d = ROOT / "runs" / CONFIRM / "sparse"
    assert (d / "DONE.json").exists(), "sparse confirmation run incomplete"
    assert json.loads((d / "index.json").read_text(encoding="utf-8"))["bank_sha256"] == CONFIRM_SHA
    assert json.loads((d / "provenance.json").read_text(encoding="utf-8"))["param_fingerprint"].startswith(FP_SPARSE)
    gen = load_gen_dir(d)
    for mode in ("chain_one", "chain_m1"):
        assert {r["cand_id"] for r in rows} <= set(gen[mode]), f"{mode}: missing chains"
        assert all(len(v["states"]) == H + 1 for v in gen[mode].values())
    assert {r["item_id"] for r in rows} <= set(gen["e2e"])
    log(f"confirmation bank {CONFIRM} sha ok; {len(bank['parents'])} parents "
        f"({ {SHORT[f]: len({c['parent_id'] for c in cands[f]}) for f in FAMS} } per family), "
        f"{sum(len(v) for v in cands.values())} decision candidates (twins excluded: {len(tw)}); "
        f"{n_check} gold one-step transitions == FDS oracle; chained gold == candidate_table gold/phi; "
        f"sparse generations complete (fingerprint {FP_SPARSE}...)")
    return {"bank": bank, "rows": rows, "T": T, "cands": cands, "gen": gen, "twins": twins,
            "gen_sha256": {m: sha(d / f"gen_{m}.jsonl") for m in ("chain_one", "chain_m1", "e2e")}}


# ============================================================================ Titan regrets + sanity

def titan_regrets(rows, fam, gen, modes=("chain_one", "chain_m1", "e2e")):
    out = {}
    for mode in modes:
        out[mode] = {}
        for k in KS:
            r = decisions.g_only(rows, fam, k, gen, modes=(mode,))[mode]    # one mode per call: fresh rng 17
            out[mode][k] = (list(r["pids"]), np.asarray(r["nregret"], np.float64))
    return out


def sanity_titan_confirm(C, log):
    ref_pp = np.load(CONF_AN / "PREREG_1_RESULTS_PER_PARENT.npz")
    ref = json.loads((CONF_AN / "PREREG_1_RESULTS.json").read_text(encoding="utf-8"))["per_family"]
    titan, chk = {}, {}
    for fam in FAMS:
        titan[fam] = titan_regrets(C["rows"], fam, C["gen"])
        for mode in ("chain_one", "chain_m1", "e2e"):
            for k in KS:
                pids, v = titan[fam][mode][k]
                rp = np.asarray(ref_pp[f"{fam}|k{k}|Gtitan_{mode}"], np.float64)
                assert pids == [str(x) for x in ref_pp[f"{fam}|k{k}|pids"]], (fam, mode, k)
                assert np.array_equal(np.isnan(v), np.isnan(rp)), (fam, mode, k)
                d = float(np.nanmax(np.abs(v - rp))) if np.isfinite(v - rp).any() else 0.0
                chk[f"{fam}|{mode}|k{k}"] = {"ours": float(np.nanmean(v)),
                                            "PREREG_1_RESULTS": ref[f"{fam}|k{k}"]["means"][f"Gtitan_{mode}"],
                                            "max_abs_diff_per_parent": d, "n_parents": int((~np.isnan(v)).sum())}
    worst = max(x["max_abs_diff_per_parent"] for x in chk.values())
    k4 = {SHORT[f]: chk[f"{f}|chain_one|k4"]["ours"] for f in FAMS}
    rounded_ok = {SHORT[f]: f"{chk[f'{f}|chain_one|k4']['ours']:.3f}" == f"{CONFIRM_TITAN_K4[f]:.3f}" for f in FAMS}
    log(f"SANITY Titan through g_only on confirmation: chain_one k4 { {k: round(v, 4) for k, v in k4.items()} } "
        f"(confirmatory L 0.539, R 0.356, P 0.899; 3-decimal match {rounded_ok}); max |per-parent diff| vs "
        f"PREREG_1_RESULTS_PER_PARENT.npz over chain_one/chain_m1/e2e x k1/2/4 = {worst:.2e}")
    assert all(rounded_ok.values()) and worst < 1e-6, chk
    return titan, {"titan_vs_PREREG_1_RESULTS": chk, "max_abs_diff": worst, "chain_one_k4": k4,
                   "chain_one_k4_3dp_match": rounded_ok, "PASS": True}


def sanity_oracle(rows, cands, log, tag):
    res = {}
    for fam in FAMS:
        orc = SR.rollout(cands[fam], SR.oracle_step_fn(fam))
        rg = SR.chain_regrets(rows, fam, orc, mode="oracle_roll")
        ex = SR.exact_at_k(cands[fam], orc)
        res[SHORT[fam]] = {"regret_mean": {k: float(np.nanmean(rg[k][1])) for k in KS},
                           "regret_max": {k: float(np.nanmax(rg[k][1])) for k in KS},
                           "exact": ex, "n_parents_scored": {k: int((~np.isnan(rg[k][1])).sum()) for k in KS}}
        assert all(abs(res[SHORT[fam]]["regret_max"][k]) < 1e-9 for k in KS), res
        assert all(ex[k] == 1.0 for k in KS), ex
    log(f"SANITY oracle one-step model through the rollout -> phi -> g_only path ({tag}): regret max "
        f"{ {f: max(v['regret_max'].values()) for f, v in res.items()} }, exact "
        f"{ {f: min(v['exact'].values()) for f, v in res.items()} } -> PASS")
    return {"per_family": res, "PASS": True}


# ============================================================================ (A) R8 failure map

def err_type(fam, inp, e, out, want) -> str:
    """Mutually exclusive error type of one WRONG one-step output (priority order as listed).
    'group:subtype' types are pooled by group for the dominant-error column.
    R 'missing_mod97' = updated register outside 0..96, which is R8's definition: it reproduces R8's
    pilot counts exactly (add 7 of 15 errors, mix 41 of 125)."""
    if out is None:
        return "unparseable"
    if fam == "ledger_replay":
        if e["status"] != "posted" and out == env.step(fam, inp, {**e, "status": "posted"}):
            return "void_event_applied_as_posted"
        nw = sum(out[a] != want[a] for a in env.ACCOUNTS)
        return {1: "one_balance_wrong", 2: "two_balances_wrong"}.get(nw, f"{nw}_balances_wrong")
    if fam == "register_machine":
        k, r, o, v = e["kind"], e["register"], e["other"], e["value"]
        if k == "swap":
            return "swap_wrong"
        if any(out[x] != want[x] for x in env.REGISTERS if x != r):
            return "untouched_register_changed"
        if not 0 <= out[r] < env.MOD:
            return "missing_mod97"
        if k == "conditional":
            alt = inp[r] + (-v if inp[o] > e["threshold"] else v)
            if out[r] == alt % env.MOD:
                return "wrong_branch"
        return "wrong_value"
    s, add, dl = inp, e["add"], e["delete"]
    if not env.applicable(s, e):
        if out == ((s & ~dl) | add):
            return "applied_anyway:bitwise"
        if out == s + add:
            return "applied_anyway:int_sum"
        return "other_wrong_integer"
    if out == s + add:
        return "eq_state_plus_add"
    if out == (s | add):
        return "eq_state_or_add_delete_ignored"
    if out == s:
        return "unchanged_state"
    return "other_wrong_integer"


def step_records(fam, chains, cands, rule):
    """Per-step records under s_roll.onestep_stats' selection rules ('input_correct' = R8, 'step1')."""
    recs = []
    for c in cands:
        g = chains.get(c["cand_id"])
        if g is None:
            continue
        ps = [None if s is None else json.loads(s) for s in g["states"]]
        for t in range(H):
            if rule == "step1" and t > 0:
                break
            if ps[t] is None or ps[t] != c["gold"][t]:
                continue
            inp, want, e = c["gold"][t], c["gold"][t + 1], c["events"][t]
            out = ps[t + 1]
            ok = out is not None and out == want
            recs.append({"pid": c["parent_id"], "labels": SR.strata(fam, inp, e) + ["ALL"], "ok": bool(ok),
                         "unparsed": out is None, "err": None if ok else err_type(fam, inp, e, out, want)})
    return recs


def cluster_ci(recs, lab, n_boot=2000, seed=17):
    by = {}
    for r in recs:
        if lab in r["labels"]:
            a = by.setdefault(r["pid"], [0, 0])
            a[0] += 1
            a[1] += int(r["ok"])
    arr = np.array(list(by.values()), np.float64)
    if len(arr) < 2:
        return None, len(arr)
    idx = np.random.default_rng(seed).integers(0, len(arr), (n_boot, len(arr)))
    acc = arr[idx, 1].sum(1) / arr[idx, 0].sum(1)
    return [float(np.quantile(acc, 0.025)), float(np.quantile(acc, 0.975))], len(arr)


def failure_map(fam, gen, cands):
    out = {}
    for rule in ("input_correct", "step1"):
        recs = step_records(fam, gen["chain_one"], cands, rule)
        ref = SR.onestep_stats(fam, gen["chain_one"], cands, rule)     # the pilot's own function: must agree
        labs = sorted({lab for r in recs for lab in r["labels"]})
        tab = {}
        for lab in labs:
            sel = [r for r in recs if lab in r["labels"]]
            n, hit = len(sel), sum(r["ok"] for r in sel)
            errs = collections.Counter(r["err"] for r in sel if not r["ok"])
            groups = collections.Counter(r["err"].split(":")[0] for r in sel if not r["ok"])
            ci, npar = cluster_ci(recs, lab)
            dom = groups.most_common(1)[0] if groups else None
            tab[lab] = {"n": n, "acc": hit / n, "ci95_parent_cluster": ci, "n_parents": npar,
                        "n_unparsed": sum(r["unparsed"] for r in sel), "n_wrong": n - hit,
                        "errors": dict(errs.most_common()),
                        "dominant_error": None if dom is None else {"type": dom[0], "count": dom[1], "of_wrong": n - hit}}
            assert ref[lab]["n"] == n and abs(ref[lab]["acc"] - hit / n) < 1e-12, (fam, rule, lab)
        out[rule] = tab
    return out


# ============================================================================ (B) P anatomy

def p_anatomy(gen, cands):
    """Applicable first steps (input = gold s0) of chain_one: right / wrong, wrong == s + add, split by
    add bit already set (carry => unambiguous integer addition) x delete clears a set bit."""
    cells = {}
    tot = collections.Counter()
    per_parent = {}
    for c in cands:
        s, e = c["s0"], c["events"][0]
        if not env.applicable(s, e):
            continue
        g = gen["chain_one"].get(c["cand_id"])
        st = None if g is None else g["states"][1]
        out = None if st is None else json.loads(st)
        true = c["gold"][1]
        add_set, del_clears = bool(s & e["add"]), bool(s & e["delete"])
        key = f"add_bit_{'already_set' if add_set else 'clear'}|delete_{'clears_a_set_bit' if del_clears else 'clears_nothing'}"
        right = out == true
        wrong = not right
        eq_sum = wrong and out is not None and out == s + e["add"]
        cc = cells.setdefault(key, collections.Counter())
        for cnt in (cc, tot):
            cnt["n"] += 1
            cnt["right"] += int(right)
            cnt["wrong"] += int(wrong)
            cnt["wrong_unparsed"] += int(wrong and out is None)
            cnt["wrong_eq_state_plus_add"] += int(eq_sum)
        if wrong:
            tot["wrong_type:" + err_type("prerequisite_plan", s, e, out, true)] += 1
        tag = "sum_rule_gives_gold" if s + e["add"] == true else "sum_rule_gives_wrong_state"
        tot[f"{tag}|n"] += 1
        tot[f"{tag}|titan_right"] += int(right)
        pp = per_parent.setdefault(c["parent_id"], [0, 0])
        pp[0] += int(wrong)
        pp[1] += int(eq_sum)
    a_set = [k for k in cells if k.startswith("add_bit_already_set")]
    a_clr = [k for k in cells if k.startswith("add_bit_clear")]
    carry = sum(cells[k]["wrong_eq_state_plus_add"] for k in a_set)
    carry_nodel = cells.get("add_bit_already_set|delete_clears_nothing", collections.Counter())["wrong_eq_state_plus_add"]
    ign_del = sum(cells[k]["wrong_eq_state_plus_add"] for k in a_clr)
    assert cells.get("add_bit_clear|delete_clears_nothing", collections.Counter())["wrong_eq_state_plus_add"] == 0
    arr = np.array([v for v in per_parent.values() if v[0] > 0], np.float64)
    idx = np.random.default_rng(17).integers(0, len(arr), (2000, len(arr)))
    share = arr[idx, 1].sum(1) / arr[idx, 0].sum(1)
    return {"applicable_first_steps": tot["n"], "right": tot["right"], "wrong_total": tot["wrong"],
            "wrong_unparseable": tot["wrong_unparsed"], "wrong_parsed": tot["wrong"] - tot["wrong_unparsed"],
            "wrong_eq_state_plus_add": tot["wrong_eq_state_plus_add"],
            "share_of_wrong_eq_state_plus_add": tot["wrong_eq_state_plus_add"] / max(tot["wrong"], 1),
            "share_of_wrong_eq_state_plus_add_ci95_parent_cluster": [float(np.quantile(share, 0.025)), float(np.quantile(share, 0.975))],
            "unambiguous_carry_add_bit_already_set": carry,
            "unambiguous_carry_and_nothing_deleted_gold_is_unchanged_state": carry_nodel,
            "add_bit_clear_eq_sum_consistent_with_ignoring_delete": ign_del,
            "titan_right_where_state_plus_add_is_gold": [tot["sum_rule_gives_gold|titan_right"], tot["sum_rule_gives_gold|n"]],
            "titan_right_where_state_plus_add_is_not_gold": [tot["sum_rule_gives_wrong_state|titan_right"], tot["sum_rule_gives_wrong_state|n"]],
            "cells": {k: dict(v) for k, v in sorted(cells.items())},
            "wrong_types": {k.split(":", 1)[1]: v for k, v in sorted(tot.items()) if k.startswith("wrong_type:")}}


# ============================================================================ (C) S-roll

def overlap_stats(fam, tr, T_conf):
    trkeys = {SR.enc(fam, r["s"], r["e"]).tobytes() for r in tr}
    hit = [SR.enc(fam, r["s"], r["e"]).tobytes() in trkeys for r in T_conf[fam]]
    tr_states = {env.dumps(r["s"]) for r in tr}
    return {"n_conf_transitions": len(hit), "conf_transition_input_in_pilot_train_frac": float(np.mean(hit)),
            "conf_parents_with_such_a_transition": len({r["parent_id"] for r, h in zip(T_conf[fam], hit) if h}),
            "conf_input_state_seen_as_train_input_frac": float(np.mean([env.dumps(r["s"]) in tr_states for r in T_conf[fam]]))}


def arm_specs(fam, crit, cfg_full, cfg_128, tr, trows):
    specs = [("FULL", cfg_full, [(f"seed{s}", tr, s) for s in SEEDS]),
             ("MATCHED128", cfg_128, [(f"r{r}_seed{s}", SR.sample(tr, SR.MATCHED_N, SR.RESAMPLE_SEED0 + r), s)
                                      for r in range(SR.N_RESAMPLES) for s in SEEDS])]
    if trows is not None:
        specs.append(("TITANROWS", cfg_128, [(f"seed{s}", trows[fam], s) for s in SEEDS]))
    for n in SR.CURVE_NS:
        if n < len(tr):
            specs.append((f"N{n}", cfg_full, [(f"r{i}_seed{s}", SR.sample(tr, n, SR.RESAMPLE_SEED0 + i), s)
                                              for i, s in enumerate(SEEDS[:SR.N_CURVE_MODELS])]))
    return specs


def run_sroll(crit, fam, P, C, titan_c, titan_p, trows, smoke, log, force, sfx):
    f = SHORT[fam]
    tag = SR.CRIT_TAG[crit]
    part = OUT / f"confirm_replicate{sfx}_part_{crit}_{f}.json"
    part_npz = OUT / f"confirm_replicate{sfx}_part_{crit}_{f}.npz"
    meta = code_meta()
    if part.exists() and part_npz.exists() and not force:
        old = json.loads(part.read_text(encoding="utf-8"))
        if old.get("meta", {}).get("code_sha256") == meta["code_sha256"]:
            log(f"[C {crit} {f}] reusing {part.name} (same code)")
            return old
        log(f"[C {crit} {f}] {part.name} exists but code changed -> recomputing")
    t_start = time.time()
    sel_path = OUT / f"s_roll_select{tag}_{f}.json"
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    assert sel.get("crit", "exact") == crit
    cfg_full, cfg_128 = sel["selection"]["FULL"]["chosen"], sel["selection"]["N128"]["chosen"]
    pilot = json.loads((OUT / f"s_roll{tag}_{f}.json").read_text(encoding="utf-8"))
    pilot_npz = np.load(OUT / f"s_roll_per_model{tag}_{f}.npz", allow_pickle=False)
    assert pilot["fits"]["FULL"]["config"] == cfg_full and pilot["fits"]["MATCHED128"]["config"] == cfg_128
    tr, dv = SR.dedupe(fam, P["T"][fam]["train"]), P["T"][fam]["dev"]
    assert len(tr) == pilot["data"]["n_train_dedup"] and len(dv) == pilot["data"]["n_dev"]
    log(f"[C {crit} {f}] pilot selection ({sel_path.name}): FULL {cfg_full} | N128 {cfg_128}; "
        f"TRAIN {len(tr)} dedup, DEV {len(dv)}")
    res = {"family": fam, "crit": crit, "crit_name": CRIT_NAME[crit], "selection_file": sel_path.name,
           "selection_file_sha256": sha(sel_path), "config": {"FULL": cfg_full, "N128": cfg_128},
           "data": overlap_stats(fam, tr, C["T"]), "arms": {}, "skipped": [], "reproduction": {}}
    npz = {}
    pids_c = titan_c[fam]["chain_one"][4][0]
    for name, cfg, sets in arm_specs(fam, crit, cfg_full, cfg_128, tr, trows):
        if name in SUPP_ARMS and cpu_sec() > CPU_SOFT_BUDGET_SEC:
            res["skipped"].append({"arm": name, "reason": f"process CPU {cpu_sec() / 60:.1f} min > soft budget"})
            log(f"[C {crit} {f}] SKIP supplementary arm {name}: CPU soft budget reached")
            continue
        per_c = {k: [] for k in KS}
        per_p = {k: [] for k in KS}
        exs, s1, ic, tf, fits = [], [], [], [], []
        for i, (mtag, trs, seed) in enumerate(sets):
            t0 = time.time()
            m = SR.fit_one(fam, cfg, seed, trs, dv, smoke, crit)
            ch_p = SR.rollout(P["cands"][fam], m.step_fn())
            rg_p = SR.chain_regrets(P["rows"], fam, ch_p)
            ch_c = SR.rollout(C["cands"][fam], m.step_fn())
            rg_c = SR.chain_regrets(C["rows"], fam, ch_c)
            for k in KS:
                assert rg_c[k][0] == titan_c[fam]["chain_one"][k][0] and rg_p[k][0] == titan_p[fam]["chain_one"][k][0]
                per_c[k].append(rg_c[k][1])
                per_p[k].append(rg_p[k][1])
            exs.append(SR.exact_at_k(C["cands"][fam], ch_c))
            s1.append(SR.onestep_stats(fam, ch_c, C["cands"][fam], "step1"))
            ic.append(SR.onestep_stats(fam, ch_c, C["cands"][fam], "input_correct"))
            tf.append(SR.teacher_forced_stats(fam, m, C["T"][fam]))
            pm = pilot["fits"].get(name, {}).get("models", [])
            pm = pm[i] if i < len(pm) else None
            fits.append({"tag": mtag, "n_train": len(trs), "dev_exact": m.dev_acc, "best_step": m.best_step,
                         "steps_run": m.steps_run,
                         "pilot_dev_exact": None if pm is None else pm["dev_exact"],
                         "pilot_best_step": None if pm is None else pm["best_step"],
                         "same_as_pilot": None if pm is None else bool(abs(pm["dev_exact"] - m.dev_acc) < 1e-12
                                                                       and pm["best_step"] == m.best_step
                                                                       and pm["tag"] == mtag)})
            log(f"  fit[{crit}] {f} {name:<10} {mtag:<11} n={len(trs):5d} dev_exact={m.dev_acc:.4f} "
                f"(pilot {fits[-1]['pilot_dev_exact']}) step={m.best_step}/{m.steps_run} | conf k4 regret "
                f"{np.nanmean(rg_c[4][1]):.3f} ({time.time() - t0:.1f}s)")
        arm = {"n_models": len(sets), "fits": fits, "regret": {}, "contrasts": {}, "exact_state": {},
               "onestep_step1": SR.mean_stats(s1), "onestep_input_correct": SR.mean_stats(ic),
               "teacher_forced": SR.mean_stats(tf)}
        rep = {}
        for k in KS:
            st_c, st_p = np.stack(per_c[k]), np.stack(per_p[k])
            npz[f"{crit}|{f}|{name}|k{k}|per_model_nregret_confirm"] = st_c
            npz[f"{crit}|{f}|{name}|k{k}|per_model_nregret_pilot_test"] = st_p
            with np.errstate(all="ignore"):
                pp = np.nanmean(st_c, 0)
            npz[f"{crit}|{f}|{name}|k{k}|per_parent_confirm"] = pp
            pmeans = [float(np.nanmean(x)) for x in st_c]
            arm["regret"][k] = {**SR.summarize_pp(pp), "per_model_mean": pmeans, "per_model_sd": float(np.std(pmeans))}
            pk = f"{f}|{name}|k{k}|per_model_nregret"
            pr = pilot["regret"].get(name, {}).get(str(k), {})
            arm["regret"][k]["pilot"] = {"mean": pr.get("mean"), "ci95": pr.get("ci95"), "n": pr.get("n")}
            for mode in ("chain_one", "chain_m1"):
                cb = decide.paired_bootstrap(titan_c[fam][mode][k][1], pp)
                pc = pilot["contrasts"].get(f"Titan {mode} - S-roll {name}|k{k}")
                arm["contrasts"][f"Titan {mode} - S-roll|k{k}"] = {"confirm": cb, "pilot": pc}
            if pk in pilot_npz.files and pilot_npz[pk].shape == st_p.shape:
                ref = np.asarray(pilot_npz[pk], np.float64)
                same_nan = bool(np.array_equal(np.isnan(ref), np.isnan(st_p)))
                dd = np.abs(ref - st_p)
                rep[k] = {"max_abs_diff_pilot_test_per_model_per_parent": float(np.nanmax(dd)) if np.isfinite(dd).any() else 0.0,
                          "nan_pattern_equal": same_nan}
            else:
                rep[k] = {"note": f"pilot per-model array {pk} missing or shape differs"}
        arm["exact_state"] = {k: {"mean": float(np.mean([e[k] for e in exs])), "sd": float(np.std([e[k] for e in exs]))} for k in KS}
        arm["pilot_exact_state"] = {k: pilot.get("exact_state", {}).get(name, {}).get(str(k), {}).get("mean") for k in KS}
        arm["pilot_onestep_step1"] = pilot.get("onestep", {}).get(name, {}).get("chain_step1")
        arm["reproduces_pilot_models"] = {"per_k": rep,
                                          "all_fits_same_dev_exact_and_step": all(x["same_as_pilot"] for x in fits),
                                          "pilot_test_regret_max_abs_diff": max(v.get("max_abs_diff_pilot_test_per_model_per_parent", float("nan")) for v in rep.values())}
        res["arms"][name] = arm
        c4 = arm["contrasts"]["Titan chain_one - S-roll|k4"]
        log(f"[C {crit} {f}] {name:<10} conf regret k1/2/4 = " + " / ".join(f"{arm['regret'][k]['mean']:.3f}" for k in KS)
            + f" | Titan chain_one - S-roll k4 = {c4['confirm']['mean_diff']:+.3f} "
              f"[{c4['confirm']['ci95'][0]:+.3f}, {c4['confirm']['ci95'][1]:+.3f}] (pilot "
            + (f"{c4['pilot']['mean_diff']:+.3f} [{c4['pilot']['ci95'][0]:+.3f}, {c4['pilot']['ci95'][1]:+.3f}]" if c4["pilot"] else "n/a")
            + f") | pilot models reproduced: fits {arm['reproduces_pilot_models']['all_fits_same_dev_exact_and_step']}, "
              f"pilot-TEST regret max|diff| {arm['reproduces_pilot_models']['pilot_test_regret_max_abs_diff']:.2e}")
        guard(log, f"C {crit} {f} {name}")
    res["wall_sec"] = round(time.time() - t_start, 1)
    res["meta"] = meta
    res = jsonable(res)
    part.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(part_npz, **npz)
    log(f"[C {crit} {f}] done in {res['wall_sec']:.0f}s -> {part.name}")
    return res


# ============================================================================ report tables (log)

def fmt_ci(ci):
    return "n/a" if not ci else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def tables(R, log):
    lines = []
    A = R["A_failure_map"]
    lines.append("\n## (A) Titan one-step accuracy by operation (chain_one). Cells: acc (n) [conf 95% parent-cluster CI]; dominant error count/wrong")
    for rule in ("input_correct", "step1"):
        lines.append(f"\n### rule = {rule} ({'R8: input state correct' if rule == 'input_correct' else 'first step from gold s0'})")
        lines.append("| Family | Stratum | pilot acc (n) | confirm acc (n) [95% CI] | pilot dominant error | confirm dominant error |")
        lines.append("|---|---|---|---|---|---|")
        for fam in FAMS:
            pt, ct = A["pilot"][SHORT[fam]][rule], A["confirm"][SHORT[fam]][rule]
            for lab in sorted(set(pt) | set(ct), key=lambda x: (x == "ALL", x)):
                p_, c_ = pt.get(lab), ct.get(lab)
                pd_ = "—" if not p_ or not p_["dominant_error"] else f"{p_['dominant_error']['type']} {p_['dominant_error']['count']}/{p_['dominant_error']['of_wrong']}"
                cd_ = "—" if not c_ or not c_["dominant_error"] else f"{c_['dominant_error']['type']} {c_['dominant_error']['count']}/{c_['dominant_error']['of_wrong']}"
                cci = "" if not c_ or not c_["ci95_parent_cluster"] else f" [{c_['ci95_parent_cluster'][0]:.3f}, {c_['ci95_parent_cluster'][1]:.3f}]"
                lines.append(f"| {SHORT[fam]} | {lab} | " + ("—" if not p_ else f"{p_['acc']:.3f} ({p_['n']})") + " | "
                             + ("—" if not c_ else f"{c_['acc']:.3f} ({c_['n']}){cci}")
                             + f" | {pd_} | {cd_} |")
    B = R["B_prerequisite_anatomy"]
    lines.append("\n## (B) P applicable first steps (gold s0 input), Titan chain_one")
    lines.append("| Quantity | pilot | confirm |")
    lines.append("|---|---|---|")
    for key in ("applicable_first_steps", "right", "wrong_total", "wrong_unparseable", "wrong_parsed", "wrong_eq_state_plus_add",
                "share_of_wrong_eq_state_plus_add", "unambiguous_carry_add_bit_already_set",
                "unambiguous_carry_and_nothing_deleted_gold_is_unchanged_state", "add_bit_clear_eq_sum_consistent_with_ignoring_delete"):
        pv, cv = B["pilot"]["P"][key], B["confirm"]["P"][key]
        fm = (lambda v: f"{v:.3f}") if isinstance(pv, float) else str
        lines.append(f"| {key} | {fm(pv)} | {fm(cv)}" + (f" {fmt_ci(B['confirm']['P']['share_of_wrong_eq_state_plus_add_ci95_parent_cluster'])}" if key == "share_of_wrong_eq_state_plus_add" else "") + " |")
    for key in ("titan_right_where_state_plus_add_is_gold", "titan_right_where_state_plus_add_is_not_gold"):
        pv, cv = B["pilot"]["P"][key], B["confirm"]["P"][key]
        lines.append(f"| {key} | {pv[0]}/{pv[1]} | {cv[0]}/{cv[1]} |")
    acc = B["applicable_first_step_accuracy"]
    lines.append(f"| applicable first-step accuracy (confirm re-weighted to pilot cell mix) | {acc['pilot']:.3f} | "
                 f"{acc['confirm']:.3f} ({acc['confirm_reweighted_to_pilot_cell_mix']:.3f}) |")
    for key in sorted(set(B["pilot"]["P"]["wrong_types"]) | set(B["confirm"]["P"]["wrong_types"])):
        lines.append(f"| wrong type {key} | {B['pilot']['P']['wrong_types'].get(key, 0)} | {B['confirm']['P']['wrong_types'].get(key, 0)} |")
    for cell in sorted(set(B["pilot"]["P"]["cells"]) | set(B["confirm"]["P"]["cells"])):
        pc, cc = B["pilot"]["P"]["cells"].get(cell, {}), B["confirm"]["P"]["cells"].get(cell, {})
        g = lambda d: f"n {d.get('n', 0)}, right {d.get('right', 0)}, wrong {d.get('wrong', 0)} (=s+add {d.get('wrong_eq_state_plus_add', 0)}, unparsed {d.get('wrong_unparsed', 0)})"
        lines.append(f"| cell {cell} | {g(pc)} | {g(cc)} |")
    for fam in ("prerequisite_plan", "ledger_replay"):
        pa, ca = B["pilot"]["anatomy"][SHORT[fam]], B["confirm"]["anatomy"][SHORT[fam]]
        for key in sorted(set(pa) | set(ca)):
            lines.append(f"| {SHORT[fam]} anatomy {key} | {pa.get(key, '—')} | {ca.get(key, '—')} |")
    if R.get("C_sroll"):
        lines.append("\n## (C) S-roll vs Titan chain_one on the confirmation parents. Cells: S-roll regret; Titan - S-roll [95% CI] (negative = Titan better)")
        for k in KS:
            lines.append(f"\n### k = {k}")
            lines.append("| Family | Titan chain_one pilot / confirm | S-roll arm | rule | pilot: S-roll; Titan - S-roll [CI] | confirm: S-roll; Titan - S-roll [CI] (n) |")
            lines.append("|---|---|---|---|---|---|")
            for fam in FAMS:
                tp, tc = R["C_sroll"]["titan_pilot"][SHORT[fam]][str(k)], R["C_sroll"]["titan_confirm"][SHORT[fam]][str(k)]
                for crit in ("exact", "regret"):
                    part = R["C_sroll"]["parts"].get(f"{crit}|{SHORT[fam]}")
                    if not part:
                        continue
                    for name in MAIN_ARMS + SUPP_ARMS:
                        a = part["arms"].get(name)
                        if not a:
                            continue
                        cc = a["contrasts"][f"Titan chain_one - S-roll|k{k}"]
                        pv = a["regret"][str(k)]["pilot"]["mean"]
                        pc = cc["pilot"]
                        lines.append(f"| {SHORT[fam]} | {tp:.3f} / {tc:.3f} | {name} | {CRIT_NAME[crit]} | "
                                     + ("n/a" if pv is None or not pc else f"{pv:.3f}; {pc['mean_diff']:+.3f} {fmt_ci(pc['ci95'])}")
                                     + f" | {a['regret'][str(k)]['mean']:.3f}; {cc['confirm']['mean_diff']:+.3f} {fmt_ci(cc['confirm']['ci95'])} ({cc['confirm']['n']}) |")
        lines.append("\n### First-step accuracy by operation: Titan vs S-roll FULL (spec / DEV-regret)")
        lines.append("| Family | Stratum | pilot Titan | pilot S-roll FULL spec / DEV-regret | confirm Titan | confirm S-roll FULL spec / DEV-regret |")
        lines.append("|---|---|---|---|---|---|")
        for fam in FAMS:
            pt, ct = R["A_failure_map"]["pilot"][SHORT[fam]]["step1"], R["A_failure_map"]["confirm"][SHORT[fam]]["step1"]
            pe = R["C_sroll"]["parts"].get(f"exact|{SHORT[fam]}", {}).get("arms", {}).get("FULL", {})
            pr = R["C_sroll"]["parts"].get(f"regret|{SHORT[fam]}", {}).get("arms", {}).get("FULL", {})
            for lab in sorted(set(ct), key=lambda x: (x == "ALL", x)):
                g = lambda a, key: "—" if not a or not a.get(key) or lab not in a[key] else f"{a[key][lab]['acc_mean']:.3f}"
                lines.append(f"| {SHORT[fam]} | {lab} | {pt[lab]['acc']:.3f} | {g(pe, 'pilot_onestep_step1')} / {g(pr, 'pilot_onestep_step1')} | "
                             f"{ct[lab]['acc']:.3f} | {g(pe, 'onestep_step1')} / {g(pr, 'onestep_step1')} |")
    txt = "\n".join(lines)
    for ln in txt.splitlines():
        log.f.write(ln + "\n")
    log.f.flush()
    print(txt, flush=True)
    return txt


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="ABC", help="any of A, B, C")
    ap.add_argument("--smoke", action="store_true", help="tiny training schedule; *_smoke outputs only")
    ap.add_argument("--force", action="store_true", help="recompute (C) parts even if cached")
    ap.add_argument("--families", default="L,P,R")
    ap.add_argument("--crits", default="exact,regret")
    a = ap.parse_args()
    sfx = "_smoke" if a.smoke else ""
    log = Log(OUT / f"confirm_replicate{sfx}.log")
    log("=" * 100)
    log(f"confirm_replicate start {time.strftime('%Y-%m-%d %H:%M:%S')} parts={a.parts} smoke={a.smoke}; torch {torch.__version__} "
        f"threads={torch.get_num_threads()} cuda_visible={os.environ['CUDA_VISIBLE_DEVICES']!r} "
        f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    t0, cpu0 = time.time(), cpu_sec()
    deviations = [
        "Twins: excluded from every table, as from the decision set (data.candidate_table iterates p['candidates'] "
        "only; the pilot bank has no twins). Their chains exist in gen_chain_one.jsonl but enter nothing here.",
        "(A) step selection uses s_roll.onestep_stats (the S_ROLL_PILOT recomputation of R8). RESULTS_PILOT R8's own "
        "counts differ by 1-5 steps for L/P (all wrong outputs; the R8 script is not in the repository); the pilot "
        "column here is the s_roll recomputation, run by this script on the pilot generations.",
        "(A) the R8 'dominant error' column was free text; here it is an explicit mutually exclusive error typology "
        "(err_type), applied identically to the pilot and confirmation generations. It was written for this "
        "replication; it selects nothing and sets no threshold.",
        "Confidence intervals for accuracies are 95% parent-cluster bootstraps (2,000 draws, seed 17), added here "
        "(the pilot reported none).",
        "(C) S-roll models are retrained (the pilot did not save weights) with s_roll.fit_one and the stored "
        "selections; reproduction is checked against the stored pilot fits and per-model pilot-TEST regrets.",
    ]
    R = {"status": "EXPLORATORY replication on the sealed confirmation bank (not PREREG-1; nothing tuned on it)",
         "deviations": deviations, "sanity": {}, "A_failure_map": {}, "B_prerequisite_anatomy": {}, "C_sroll": {}}

    # ---------------- inputs
    bank_p, rows_p, T_p, cands_p, dcands_p = SR.load_inputs(log)
    gen_p = SR.load_gen()
    P = {"rows": rows_p, "T": T_p, "cands": cands_p, "gen": gen_p}
    C = load_confirm(log)
    guard(log, "inputs loaded")
    pilot_sroll = json.loads((OUT / "s_roll.json").read_text(encoding="utf-8"))

    # ---------------- sanity
    titan_c, R["sanity"]["titan_confirm"] = sanity_titan_confirm(C, log)
    R["sanity"]["oracle_confirm"] = sanity_oracle(C["rows"], C["cands"], log, "confirmation")
    R["sanity"]["oracle_pilot"] = sanity_oracle(P["rows"], P["cands"], log, "pilot TEST")
    titan_p, chk = {}, {}
    for fam in FAMS:
        titan_p[fam] = titan_regrets(P["rows"], fam, P["gen"], modes=("chain_one", "chain_m1"))
        ref = pilot_sroll["fam"][fam]["exact"]["per_parent"]
        for mode in ("chain_one", "chain_m1"):
            for k in KS:
                v, rr = titan_p[fam][mode][k][1], np.asarray(ref[f"Titan {mode}"][f"k{k}"], np.float64)
                d = np.abs(v - rr)
                chk[f"{SHORT[fam]}|{mode}|k{k}"] = float(np.nanmax(d)) if np.isfinite(d).any() else 0.0
                assert np.array_equal(np.isnan(v), np.isnan(rr))
    R["sanity"]["titan_pilot_vs_s_roll_json_max_abs_diff"] = max(chk.values())
    assert max(chk.values()) < 1e-6, chk
    log(f"SANITY Titan pilot-TEST regrets == s_roll.json per-parent (max |diff| {max(chk.values()):.2e})")
    R["titan_regret"] = {"pilot": {SHORT[f]: {m: {k: SR.summarize_pp(titan_p[f][m][k][1]) for k in KS} for m in ("chain_one", "chain_m1")} for f in FAMS},
                         "confirm": {SHORT[f]: {m: {k: SR.summarize_pp(titan_c[f][m][k][1]) for k in KS} for m in ("chain_one", "chain_m1", "e2e")} for f in FAMS}}

    # ---------------- (A) and (B)
    if "A" in a.parts or "B" in a.parts:
        A = {"pilot": {}, "confirm": {}}
        for nm, X in (("pilot", P), ("confirm", C)):
            for fam in FAMS:
                A[nm][SHORT[fam]] = failure_map(fam, X["gen"], X["cands"][fam])
        repro = {}
        for fam in FAMS:
            ref = pilot_sroll["fam"][fam]["exact"]["onestep"]["Titan chain_one"]
            for rule, key in (("input_correct", "chain_input_correct"), ("step1", "chain_step1")):
                for lab, v in ref[key].items():
                    mine = A["pilot"][SHORT[fam]][rule][lab]
                    repro[f"{SHORT[fam]}|{rule}|{lab}"] = bool(mine["n"] == v["n"] and abs(mine["acc"] - v["acc"]) < 1e-12)
        A["pilot_reproduces_s_roll_json"] = all(repro.values())
        assert A["pilot_reproduces_s_roll_json"], {k: v for k, v in repro.items() if not v}
        R["A_failure_map"] = A
        log(f"(A) pilot recomputation == s_roll.json Titan one-step tables: {A['pilot_reproduces_s_roll_json']} ({len(repro)} cells)")
        B = {"pilot": {"P": p_anatomy(P["gen"], P["cands"]["prerequisite_plan"]), "anatomy": {}},
             "confirm": {"P": p_anatomy(C["gen"], C["cands"]["prerequisite_plan"]), "anatomy": {}}}
        for nm, X in (("pilot", P), ("confirm", C)):
            for fam in ("prerequisite_plan", "ledger_replay"):
                B[nm]["anatomy"][SHORT[fam]] = SR.titan_error_anatomy(fam, X["gen"], X["cands"][fam])
        for fam in ("prerequisite_plan", "ledger_replay"):
            assert B["pilot"]["anatomy"][SHORT[fam]] == pilot_sroll["fam"][fam]["exact"]["titan"]["error_anatomy_step1"]
        # composition check: confirm applicable first-step accuracy re-weighted to the pilot's cell mix
        pc, cc = B["pilot"]["P"]["cells"], B["confirm"]["P"]["cells"]
        npil = sum(v["n"] for v in pc.values())
        B["applicable_first_step_accuracy"] = {
            "pilot": B["pilot"]["P"]["right"] / B["pilot"]["P"]["applicable_first_steps"],
            "confirm": B["confirm"]["P"]["right"] / B["confirm"]["P"]["applicable_first_steps"],
            "confirm_reweighted_to_pilot_cell_mix": sum(pc[c]["n"] / npil * cc[c]["right"] / cc[c]["n"] for c in pc if c in cc),
            "cell_share_pilot": {c: pc[c]["n"] / npil for c in pc},
            "cell_share_confirm": {c: cc[c]["n"] / B["confirm"]["P"]["applicable_first_steps"] for c in cc}}
        R["B_prerequisite_anatomy"] = B
        log(f"(B) pilot P anatomy: {json.dumps({k: v for k, v in B['pilot']['P'].items() if k not in ('cells', 'wrong_types')}, default=float)}")
        log(f"(B) confirm P anatomy: {json.dumps({k: v for k, v in B['confirm']['P'].items() if k not in ('cells', 'wrong_types')}, default=float)}")

    # ---------------- (C)
    if "C" in a.parts:
        try:
            trows = SR.load_titan_rows(log)
        except Exception as ex:                               # supplementary arm only
            trows = None
            deviations.append(f"TITANROWS skipped: Titan's training-row file unavailable ({type(ex).__name__}: {ex})")
            log(f"TITANROWS unavailable: {ex}")
        short2fam = {v: k for k, v in SHORT.items()}
        parts = {}
        for crit in a.crits.split(","):
            for f in a.families.split(","):
                parts[f"{crit}|{f}"] = run_sroll(crit, short2fam[f], P, C, titan_c, titan_p, trows, a.smoke, log, a.force, sfx)
        R["C_sroll"] = {"parts": parts,
                        "titan_pilot": {SHORT[f]: {str(k): float(np.nanmean(titan_p[f]["chain_one"][k][1])) for k in KS} for f in FAMS},
                        "titan_confirm": {SHORT[f]: {str(k): float(np.nanmean(titan_c[f]["chain_one"][k][1])) for k in KS} for f in FAMS},
                        "decision_pipeline": "decisions.g_only (test goals seed 1001, regret_table, fresh rng 17 per method, "
                                             "parent-mean fill); per-parent S-roll regret = nanmean over models; contrasts "
                                             "decide.paired_bootstrap(Titan, S-roll) over confirmation parents (10,000 draws, seed 17)"}
        nonrepro = [f"{p}|{n}" for p, v in parts.items() for n, arm in v["arms"].items()
                    if not arm["reproduces_pilot_models"]["all_fits_same_dev_exact_and_step"]
                    or not (arm["reproduces_pilot_models"]["pilot_test_regret_max_abs_diff"] < 1e-6)]
        R["C_sroll"]["all_models_reproduce_pilot"] = not nonrepro
        if nonrepro and not a.smoke:
            deviations.append(f"Retrained S-roll models do not bit-reproduce the stored pilot models for: {nonrepro} "
                              "(same code, data, seeds; see per-arm reproduces_pilot_models).")
        log(f"(C) retrained S-roll models reproduce the pilot models: {not nonrepro} {nonrepro if nonrepro else ''}")

    R["meta"] = {**code_meta(), "confirm_bank_sha256": CONFIRM_SHA, "confirm_gen_sha256": C["gen_sha256"],
                 "pilot_bank_sha256": sha(ROOT / "data" / "bank_pilot.json"),
                 "wall_sec": round(time.time() - t0, 1), "process_cpu_sec": round(cpu_sec() - cpu0, 1),
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "smoke": a.smoke}
    R = jsonable(R)
    if a.parts == "ABC" or "C" not in a.parts:
        R["tables_markdown"] = tables(R, log)
    out = OUT / f"confirm_replicate{sfx}.json"
    if a.parts != "ABC":
        out = OUT / f"confirm_replicate{sfx}_{a.parts}.json"
    out.write_text(json.dumps(R, indent=1) + "\n", encoding="utf-8")
    log(f"wrote {out} | wall {R['meta']['wall_sec']:.0f}s, process CPU {R['meta']['process_cpu_sec'] / 60:.1f} min")


if __name__ == "__main__":
    main()
