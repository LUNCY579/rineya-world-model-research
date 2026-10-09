"""T2b-register screen analysis (RCWM roadmap item 3; joint item 5b). CPU only: numpy / json; never imports torch,
never loads a model. Applies the PRE-STATED go rule (t2br_common.GO_RULE_TEXT, verbatim in docs/t2b/register/DESIGN.md).

  python -B scripts/t2br_analysis.py --selftest    # scorer self-test on the dev bank (golds 100%, negatives 0%, floors
                                                   # exact, metric and chain-pipeline identities); writes
                                                   # docs/t2b/register/SCORER_SELFTEST.json
  python -B scripts/t2br_analysis.py --dryrun      # synthetic scenarios with known verdicts (GO, decision-point-3 stops,
                                                   # each NO-GO condition, malformed traces, multi-seed); writes
                                                   # docs/t2b/register/DRYRUN.json + DRYRUN.md
  python -B scripts/t2br_analysis.py --run-dir runs/t2b_register/s0 [--run-dir ...s1 ...]
                                                   # the screen analysis (registered: freeze + provenance verified);
                                                   # writes runs/t2b_register/analysis/T2BR_RESULTS_<seeds>.{json,md}
                                                   # (create-only)

Scoring. Outcome per one-step row = the exact 'after' state, read by the decision parser of the side (t2br_common.
parse_after: plain / OFF = rcwmt.env.parse_state(lenient); trace / filler = the strict joint parsers, then parse_state on
'after'; malformed = wrong). Mechanism fields for the trace side: format_ok, cmp_ok (predicate inputs + ok, ordered
JSON), ok_ok (branch boolean), calc_ok, after_follows_calc, calc_follows_cmp. Every side's branch choice on conditional
rows is also read from its 'after' (correct branch / wrong branch / neither).
Metrics. 'conditional' (PRIMARY) = mean exact over cond_oor_val + cond_oor_thr (both branch-balanced pairs); 'mix'
(PRIMARY) = mix_oor; 'conditional_in' = branch-balanced mean over the three in-range triplet strata (0.5 x ok-false rows
+ 0.5 x ok-true rows); 'mix_in' = mix; controls swap / add_nowrap. 95% CIs: cluster bootstrap over pairs / triplets
(10,000 draws, seed 17), paired across sides; Wilson for single accuracies.
Chains (k = 4 self-rollouts, chain_one frame): exact state at k, normalised regret at k = 1, 2, 4 through the frozen
decision pipeline (rcwmt.data.candidate_table -> rcwmt.decisions.g_only: 256 Gaussian goals seed 1001, tie seed 17,
unparseable -> parent-mean utility), per stratum (CH_IN, CH_X, ALL), parent-paired bootstrap contrasts, the copy-s0
floor; and the per-step -> k4 product law: p(class) from each side's own gold-input chain steps (RESULTS_PILOT R8
convention), predicted k4 exact = mean over chains of the product of p over the chain's 4 gold step classes, against
the observed k4 exact (and the pooled p^4).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RCWM_THREADS"):
    os.environ[_v] = "2"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if "torch" in sys.modules:
    raise SystemExit("torch must not be imported by the T2b-register analysis")
if "rcwmt.heads" not in sys.modules:          # decisions / fidelity import heads (torch); g_only never uses it
    _m = types.ModuleType("rcwmt.heads")
    _m.__doc__ = "placeholder installed by t2br_analysis.py (as t2_analysis.py does): keeps torch out"
    sys.modules["rcwmt.heads"] = _m
import t2br_common as C  # noqa: E402
from rcwmt import data, decide, decisions, env, fidelity  # noqa: E402

if "torch" in sys.modules:
    raise SystemExit("torch was imported by a dependency of the T2b-register analysis")

FAM = C.FAM
N_BOOT = 10000
BOOT_SEED = 17
Z95 = 1.959963984540054
KS = (1, 2, 4)
DOCS = C.DOCS
CONTRASTS = (("trace", "plain"), ("trace", "filler"), ("filler", "plain"), ("trace", "off"), ("plain", "off"),
             ("filler", "off"))


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(f"[t2br-analysis] {msg}")


def wilson(k: int, n: int):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / d
    h = Z95 * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


# ---------------------------------------------------------------- metrics on one-step rows
# Outcome keys. The PRIMARY outcomes are named primary_conditional / primary_mix here, because the in-range dev
# stratum is itself called 'mix' (a key collision would silently replace the primary by the in-range stratum).
GROUPS = ("primary_conditional", "primary_mix", "conditional_in", "mix_in")
METRICS = {**{s: ("mean", (s,)) for s in C.DEV_STRATA},
           "primary_conditional": ("mean", C.PRIMARY["conditional"]), "primary_mix": ("mean", C.PRIMARY["mix"]),
           "conditional_in": ("branch_balanced", C.NONINF["conditional_in"]), "mix_in": ("mean", C.NONINF["mix_in"])}
assert len(METRICS) == len(C.DEV_STRATA) + len(GROUPS), "outcome key collision"


class DevIndex:
    def __init__(self, dev: list):
        self.dev = dev
        self.strata = np.array([r["stratum"] for r in dev])
        self.cls = np.array([1 if r["gold_ok"] else 0 for r in dev])
        self.clus = np.array([r["cluster"] for r in dev])
        self.idx = {m: np.where(np.isin(self.strata, list(ss)))[0] for m, (_, ss) in METRICS.items()}


def metric_point(x: np.ndarray, D: DevIndex, m: str) -> float:
    kind, _ = METRICS[m]
    i = D.idx[m]
    if kind == "mean":
        return float(x[i].mean())
    c = D.cls[i]
    return float(0.5 * x[i][c == 0].mean() + 0.5 * x[i][c == 1].mean())


def _cluster_sums(x: np.ndarray, D: DevIndex, m: str):
    kind, _ = METRICS[m]
    i = D.idx[m]
    uniq, inv = np.unique(D.clus[i], return_inverse=True)
    K = 2 if kind == "branch_balanced" else 1
    k = D.cls[i] if K == 2 else np.zeros(len(i), int)
    S = np.zeros((len(uniq), K))
    N = np.zeros((len(uniq), K))
    np.add.at(S, (inv, k), x[i])
    np.add.at(N, (inv, k), 1.0)
    return S, N


def metric_boot_diff(xa: np.ndarray, xb: np.ndarray, D: DevIndex, m: str, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
    """Cluster bootstrap (pairs / triplets resampled jointly for both sides) of metric(a) - metric(b)."""
    Sa, N = _cluster_sums(xa, D, m)
    Sb, _ = _cluster_sums(xb, D, m)
    J = len(N)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, J, (n_boot, J))
    cnt = np.zeros((n_boot, J))
    np.add.at(cnt, (np.arange(n_boot)[:, None], draws), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma, mb = (cnt @ Sa) / (cnt @ N), (cnt @ Sb) / (cnt @ N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmean(ma, axis=1) - np.nanmean(mb, axis=1)
    d = d[~np.isnan(d)]
    return [_r(np.quantile(d, 0.025)), _r(np.quantile(d, 0.975))]


def score_side(dev: list, texts: list, side: str) -> list:
    arm = C.SIDE_ARM[side]
    return [C.score_text(r, t, arm) for r, t in zip(dev, texts)]


def branch_of(row: dict, st) -> str:
    """Which conditional branch an answer applied (reduced or unreduced), read from its 'after' state."""
    if st is None:
        return "none"
    b, ev = row["before"], row["event"]
    r, v = ev["register"], ev["value"]
    if any(st[k] != b[k] for k in C.REGS if k != r):
        return "neither"
    y = st[r]
    if y in ((b[r] + v) % C.MOD, b[r] + v):
        return "add"
    if y in ((b[r] - v) % C.MOD, b[r] - v):
        return "sub"
    return "neither"


def side_block(dev: list, texts: list, side: str, D: DevIndex) -> dict:
    sc = score_side(dev, texts, side)
    x = np.array([s["after_ok"] for s in sc], float)
    arm = C.SIDE_ARM[side]
    out = {"x": x, "scores": sc, "strata": {}, "metrics": {}}
    for s in C.DEV_STRATA:
        i = D.idx[s]
        k = int(x[i].sum())
        out["strata"][s] = {"n": len(i), "acc": _r(k / len(i)), "wilson95": wilson(k, len(i)),
                            "parsed": _r(np.mean([sc[j]["parsed"] for j in i]))}
    for m in GROUPS:
        out["metrics"][m] = _r(metric_point(x, D, m))
    # branch choice read from 'after' (every side)
    br = {}
    for s in [s for s in C.DEV_STRATA if s.startswith("cond")]:
        cnt = Counter()
        for j in D.idx[s]:
            row = dev[j]
            got = branch_of(row, C.parse_after(texts[j], arm))
            want = "add" if row["gold_ok"] else "sub"
            cnt["correct" if got == want else ("wrong" if got in ("add", "sub") else "neither_or_unparsed")] += 1
        n = len(D.idx[s])
        chosen = cnt["correct"] + cnt["wrong"]
        br[s] = {"correct_branch": _r(cnt["correct"] / n), "wrong_branch": _r(cnt["wrong"] / n),
                 "neither_or_unparsed": _r(cnt["neither_or_unparsed"] / n),
                 "branch_choice_given_a_branch": _r(cnt["correct"] / chosen) if chosen else None}
    out["branch_from_after"] = br
    # unreduced-sum answers on mix / wrap strata (OFF's known failure)
    un = {}
    for s in ("mix", "wrap", "mix_oor", "wrap_oor"):
        hit = 0
        for j in D.idx[s]:
            st = C.parse_after(texts[j], arm)
            h = C.heuristic_after("unreduced", dev[j]["before"], dev[j]["event"])
            hit += st is not None and env.dumps(st) == env.dumps(h) and dev[j]["gold_after"] != env.dumps(h)
        un[s] = _r(hit / len(D.idx[s]))
    out["unreduced_answers"] = un
    if arm in ("trace", "filler"):
        fields = ("format_ok", "cmp_ok", "ok_ok", "calc_ok", "after_follows_calc", "calc_follows_cmp") if arm == "trace" \
            else ("format_ok", "fill_exact")
        tf = {}
        for s in C.DEV_STRATA:
            i = D.idx[s]
            tf[s] = {f: _r(np.mean([sc[j][f] for j in i])) for f in fields
                     if not (f in ("ok_ok", "calc_follows_cmp") and not s.startswith("cond"))}
        out["trace_fields"] = tf
    return out


# ---------------------------------------------------------------- chains
def chain_rows_for(parents: list) -> list:
    return data.candidate_table({"parents": parents})


def regret(rows: list, k: int, gen: dict):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        g = decisions.g_only(rows, FAM, k, {"m": gen}, modes=("m",), split="dev")["m"]
    return g["pids"], np.asarray(g["nregret"], np.float64)


def floor_copy_regret(rows: list, k: int):
    """F_copy_s0 (answer-free floor) through the same goals and tie rule as g_only."""
    gen = {}
    for r in rows:
        if r["k"] == k:
            gen.setdefault(r["cand_id"], {"states": [env.dumps(r["s0"])] * (C.H + 1)})
    return regret(rows, k, gen)


def parent_exact(rows: list, k: int, gen: dict, pids: list) -> np.ndarray:
    acc = defaultdict(list)
    for r in rows:
        if r["k"] == k:
            acc[r["parent_id"]].append(float(gen[r["cand_id"]]["states"][k] == r["gold"]))
    return np.array([np.mean(acc[p]) for p in pids], np.float64)


def product_law(parents: list, gen: dict) -> dict:
    n, c = Counter(), Counter()
    chains = C.flat_chains(parents)
    golds = {}
    for ch in chains:
        gold = env.rollout(FAM, ch["s0"], ch["events"])
        golds[ch["cand_id"]] = gold
        st = gen[ch["cand_id"]]["states"]
        for t, ev in enumerate(ch["events"]):
            if st[t] != env.dumps(gold[t]):
                continue
            cl = C.op_class(gold[t], ev)
            n[cl] += 1
            c[cl] += int(st[t + 1] == env.dumps(gold[t + 1]))
    pooled = sum(c.values()) / sum(n.values()) if sum(n.values()) else float("nan")
    p = {cl: (c[cl] / n[cl]) for cl in n}
    preds, obs = [], []
    for ch in chains:
        gold = golds[ch["cand_id"]]
        pr = 1.0
        for t, ev in enumerate(ch["events"]):
            pr *= p.get(C.op_class(gold[t], ev), pooled)
        preds.append(pr)
        obs.append(float(gen[ch["cand_id"]]["states"][C.H] == env.dumps(gold[C.H])))
    return {"per_class": {cl: {"n_gold_input_steps": n[cl], "acc": _r(p[cl])} for cl in sorted(n)},
            "pooled_step_acc": _r(pooled), "predicted_k4_exact_product": _r(np.mean(preds)),
            "predicted_k4_exact_pooled_pow4": _r(pooled ** C.H) if not math.isnan(pooled) else None,
            "observed_k4_exact": _r(np.mean(obs)), "observed_minus_predicted": _r(np.mean(obs) - np.mean(preds)),
            "n_chains": len(chains)}


def chain_block(parents: list, gens: dict) -> dict:
    out = {}
    strata = {**{s: [p for p in parents if p["stratum"] == s] for s in C.CHAIN_STRATA}, "ALL": parents}
    for sname, ps in strata.items():
        rows = chain_rows_for(ps)
        blk = {"n_parents": len(ps), "n_chains": sum(len(p["candidates"]) for p in ps), "k": {}}
        for k in KS:
            per = {}
            pids = None
            for side in C.SIDES:
                pids, nr = regret(rows, k, gens[side])
                per[side] = {"nregret": nr, "exact": parent_exact(rows, k, gens[side], pids)}
            fp, fr = floor_copy_regret(rows, k)
            need(fp == pids, "floor parents differ")
            kb = {"sides": {s: {"regret": decide.bootstrap_mean(per[s]["nregret"]),
                                "exact": decide.bootstrap_mean(per[s]["exact"])} for s in C.SIDES},
                  "floor_copy_s0_regret": decide.bootstrap_mean(fr), "contrasts": {}}
            for a, b in CONTRASTS:
                kb["contrasts"][f"{a}-{b}"] = {"regret": decide.paired_bootstrap(per[a]["nregret"], per[b]["nregret"]),
                                               "exact": decide.paired_bootstrap(per[a]["exact"], per[b]["exact"])}
            blk["k"][str(k)] = kb
        blk["product_law"] = {s: product_law(ps, gens[s]) for s in C.SIDES}
        out[sname] = blk
    return out


# ---------------------------------------------------------------- the pre-stated rule
def verdict(m: dict) -> dict:
    """m: {'acc': {side: {metric: value}}}. Returns the verdict with every condition (point estimates)."""
    a = m["acc"]
    d = lambda x, y, k: a[x][k] - a[y][k]          # noqa: E731
    cond = {
        "G1 trace-plain >= +0.25 on conditional (OOR)": d("trace", "plain", "primary_conditional") >= C.GAIN_CONDITIONAL,
        "G2 trace-plain >= +0.30 on mix (OOR)": d("trace", "plain", "primary_mix") >= C.GAIN_MIX,
        "G3 filler below trace on both primaries": d("trace", "filler", "primary_conditional") > 0
        and d("trace", "filler", "primary_mix") > 0,
        "G4 controls: trace ON-OFF >= -0.05 on swap and add_nowrap": all(d("trace", "off", s) >= -C.CONTROL_LOSS
                                                                           for s in C.CONTROLS),
        "G5 in-distribution non-inferiority: trace-plain >= -0.05 on conditional_in and mix_in": all(
            d("trace", "plain", k) >= -C.NONINF_MARGIN for k in C.NONINF)}
    dp3 = d("trace", "filler", "primary_conditional") < C.FILLER_STOP and d("trace", "filler", "primary_mix") < C.FILLER_STOP
    filler_gain = d("filler", "plain", "primary_conditional") >= C.FILLER_STOP or \
        d("filler", "plain", "primary_mix") >= C.FILLER_STOP
    if dp3:
        v = "STOP (decision point 3): extra decode compute, not supervision"
        if not filler_gain:
            v += " - neither format helped (trace ~ filler ~ plain)"
    elif all(cond.values()):
        v = "GO"
    else:
        v = "NO-GO (failed: " + "; ".join(k.split(" ", 1)[0] for k, ok in cond.items() if not ok) + ")"
    return {"verdict": v, "decision_point_3_stop": dp3, "conditions": cond,
            "values": {"trace-plain|conditional": _r(d("trace", "plain", "primary_conditional")),
                       "trace-plain|mix": _r(d("trace", "plain", "primary_mix")),
                       "trace-filler|conditional": _r(d("trace", "filler", "primary_conditional")),
                       "trace-filler|mix": _r(d("trace", "filler", "primary_mix")),
                       "filler-plain|conditional": _r(d("filler", "plain", "primary_conditional")),
                       "filler-plain|mix": _r(d("filler", "plain", "primary_mix")),
                       **{f"trace-off|{s}": _r(d("trace", "off", s)) for s in C.CONTROLS},
                       **{f"trace-plain|{k}": _r(d("trace", "plain", k)) for k in C.NONINF}}}


def analyse_seed(dev: list, parents: list, texts: dict, gens: dict, n_boot: int = N_BOOT) -> dict:
    D = DevIndex(dev)
    sides = {s: side_block(dev, texts[s], s, D) for s in C.SIDES}
    acc = {s: {m: metric_point(sides[s]["x"], D, m) for m in METRICS} for s in C.SIDES}
    contr = {}
    for a, b in CONTRASTS:
        contr[f"{a}-{b}"] = {m: {"diff": _r(acc[a][m] - acc[b][m]),
                                 "ci95": metric_boot_diff(sides[a]["x"], sides[b]["x"], D, m, n_boot)}
                             for m in METRICS}
    res = {"onestep": {s: {k: v for k, v in sides[s].items() if k not in ("x", "scores")} for s in C.SIDES},
           "contrasts": contr, "rule": verdict({"acc": acc})}
    res["chains"] = chain_block(parents, gens)
    return res


# ---------------------------------------------------------------- run-dir loading and validation
def load_run_outputs(run_dir: Path, dev: list, parents: list) -> tuple:
    run_dir = Path(run_dir)
    texts, gens = {}, {}
    chains = C.flat_chains(parents)
    for s in C.SIDES:
        rows = C.load_jsonl(run_dir / "onestep" / f"onestep_{s}.jsonl")
        need([r["id"] for r in rows] == [r["id"] for r in dev], f"onestep_{s}: ids differ from the dev bank")
        need(all(isinstance(r["text"], str) for r in rows), f"onestep_{s}: non-string text")
        texts[s] = [r["text"] for r in rows]
        crow = C.load_jsonl(run_dir / "chains" / f"chains_{s}.jsonl")
        need([r["cand_id"] for r in crow] == [c["cand_id"] for c in chains], f"chains_{s}: cand_ids differ")
        arm = C.SIDE_ARM[s]
        for r, c in zip(crow, chains):
            st, tx = r["states"], r["texts"]
            need(len(st) == C.H + 1 and st[0] == env.dumps(c["s0"]), f"chains_{s} {c['cand_id']}: bad states")
            broken = False
            for t in range(C.H):
                if broken:
                    need(st[t + 1] is None and len(tx) <= t, f"chains_{s} {c['cand_id']}: continues after a break")
                    continue
                need(t < len(tx), f"chains_{s} {c['cand_id']}: missing text at step {t}")
                p = C.parse_after(tx[t], arm)
                need((None if p is None else env.dumps(p)) == st[t + 1],
                     f"chains_{s} {c['cand_id']}: state {t + 1} is not its text parsed by the {arm} parser")
                broken = p is None
        gens[s] = {r["cand_id"]: r for r in crow}
    return texts, gens


def provenance_check(run_dir: Path, registered: bool) -> dict:
    run_dir = Path(run_dir)
    import t2br_run as R                      # torch-free at import
    done = C.read_json(run_dir / "DONE.json")
    need(done is not None, f"{run_dir}: DONE.json missing (the run is not complete)")
    prov = C.read_json(run_dir / "provenance.json")
    need(prov is not None, "provenance.json unreadable")
    need(C.sha256(run_dir / "provenance.json") == done["provenance_sha256"], "provenance.json changed after DONE.json")
    for st in R.STAGE_ORDER:
        for rel, h in done["files"][st].items():
            need(C.sha256(run_dir / rel) == h, f"{rel} changed after it was recorded")
    a0 = C.read_json(run_dir / "anchor" / "anchor.json") or {}
    need(bool(a0.get("pass")), "anchor did not pass")
    need(all((t.get("anchor_recheck") or {"pass": True}).get("pass") for t in prov["turns"]), "an anchor re-check failed")
    info = {"seed": prov.get("seed"), "registered": prov.get("registered"), "turns": len(prov["turns"]),
            "anchor_identical_states": a0.get("identical_states"), "data_sha256": prov.get("data_sha256"),
            "train": {st: {k: prov["stages"][st].get(k) for k in ("final_loss_avg50", "supervised_tokens", "seconds")}
                      for st in R.TRAIN_STAGES}}
    if registered:
        need(prov.get("registered") is True, "the run was not a registered run")
        ok, errs, fz = R.verify_freeze(ROOT)
        need(ok, f"the freeze does not verify: {errs[:3]}")
        need(prov.get("freeze_sha256") == C.sha256(ROOT / R.FREEZE_REL), "the run used a different freeze record")
        for t in prov["turns"]:
            for rel, h in t["code_sha256"].items():
                need(fz["inputs"].get(rel) == h, f"turn code {rel} differs from the frozen file")
        for k, rel in (("dev", f"data/{C.DEV.name}"), ("chains", f"data/{C.CHAINS.name}")):
            need(prov["data_sha256"][k] == fz["inputs"][rel], f"{k} differs from the freeze")
    return info


def analyse_runs(run_dirs: list, registered: bool = True, out_dir: Path | None = None, dev_path: Path = C.DEV,
                 chains_path: Path = C.CHAINS, n_boot: int = N_BOOT, write: bool = True) -> dict:
    dev = C.load_dev(dev_path)
    parents = C.load_chain_bank(chains_path)
    per_seed = {}
    for rd in run_dirs:
        info = provenance_check(rd, registered)
        texts, gens = load_run_outputs(rd, dev, parents)
        res = analyse_seed(dev, parents, texts, gens, n_boot)
        res["provenance"] = info
        per_seed[f"s{info['seed']}"] = res
    seeds = sorted(per_seed)
    vs = {s: per_seed[s]["rule"]["verdict"] for s in seeds}
    overall = ("GO in every analysed seed" if all(v == "GO" for v in vs.values()) else "NOT GO in every analysed seed")
    if seeds == ["s0"]:
        overall = f"SCREEN (seed 0): {vs['s0']}"
    res = {"experiment": "RCWM-T1 T2b-register screen", "go_rule": C.GO_RULE_TEXT, "registered": registered,
           "seeds": seeds, "verdicts": vs, "overall": overall,
           "confirmation_note": "the roadmap's confirmation applies the same rule in all 3 seeds, then a sealed R bank",
           "per_seed": per_seed, "dev_sha256": C.sha256(dev_path), "chains_sha256": C.sha256(chains_path),
           "code_sha256": {r: C.sha256(ROOT / r) for r in ("scripts/t2br_analysis.py", "scripts/t2br_common.py")},
           "torch_imported": "torch" in sys.modules, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if write:
        out_dir = Path(out_dir or (ROOT / "runs" / "t2b_register" / "analysis"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "T2BR_RESULTS_" + "_".join(seeds)
        js = json.dumps(clean(res), indent=1) + "\n"
        md = render(res)
        with (out_dir / f"{stem}.json").open("x", encoding="utf-8") as f:
            f.write(js)
        with (out_dir / f"{stem}.md").open("x", encoding="utf-8") as f:
            f.write(md)
        print(f"wrote {out_dir / (stem + '.json')} and .md", flush=True)
    return res


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        return None if math.isnan(float(o)) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _ci(c):
    return "[" + ", ".join("nan" if x is None else f"{x:+.3f}" for x in c) + "]"


def render(res: dict) -> str:
    L = [f"# T2b-register screen results ({', '.join(res['seeds'])})", "",
         f"**Overall:** {res['overall']}", "", "Pre-stated rule (verbatim):", "", "> " + res["go_rule"], ""]
    for s in res["seeds"]:
        r = res["per_seed"][s]
        L += [f"## Seed {s}: {r['rule']['verdict']}", "", "| condition | holds |", "|---|---|"]
        L += [f"| {k} | {'yes' if v else 'NO'} |" for k, v in r["rule"]["conditions"].items()]
        L += ["", "| outcome | OFF | plain | trace | filler | trace-plain [95% CI] | trace-filler [95% CI] |",
              "|---|---|---|---|---|---|---|"]
        for m in (*GROUPS, *C.DEV_STRATA):
            vals = []
            for side in C.SIDES:
                v = r["onestep"][side]["metrics"].get(m)
                if v is None:
                    v = r["onestep"][side]["strata"][m]["acc"]
                vals.append(f"{v:.3f}")
            tp, tf = r["contrasts"]["trace-plain"][m], r["contrasts"]["trace-filler"][m]
            L.append(f"| {m} | " + " | ".join(vals) + f" | {tp['diff']:+.3f} {_ci(tp['ci95'])} | "
                     f"{tf['diff']:+.3f} {_ci(tf['ci95'])} |")
        tf = r["onestep"]["trace"].get("trace_fields", {})
        L += ["", "Trace fields (trace side):", "", "| stratum | format | cmp_ok | ok_ok | calc_ok | after follows calc |",
              "|---|---|---|---|---|---|"]
        for st, f in tf.items():
            L.append(f"| {st} | {f.get('format_ok')} | {f.get('cmp_ok')} | {f.get('ok_ok', '-')} | {f.get('calc_ok')} | "
                     f"{f.get('after_follows_calc')} |")
        L += ["", "Branch choice read from 'after' (correct / wrong / neither), conditional strata:", ""]
        for side in C.SIDES:
            br = r["onestep"][side]["branch_from_after"]
            L.append(f"- {side}: " + "; ".join(f"{k} {v['correct_branch']}/{v['wrong_branch']}/{v['neither_or_unparsed']}"
                                               for k, v in br.items()))
        L += ["", "Chains (k=4; normalised regret / exact; mean [95% CI]):", "",
              "| stratum | side | regret k4 | exact k4 | product-law predicted k4 | observed |", "|---|---|---|---|---|---|"]
        for cs, blk in r["chains"].items():
            for side in C.SIDES:
                k4 = blk["k"]["4"]["sides"][side]
                pl = blk["product_law"][side]
                L.append(f"| {cs} | {side} | {k4['regret']['mean']:.3f} {_ci(k4['regret']['ci95'])} | "
                         f"{k4['exact']['mean']:.3f} | {pl['predicted_k4_exact_product']} | {pl['observed_k4_exact']} |")
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- scorer self-test
def _neg_variants(row: dict) -> dict:
    """Malformed / wrong answers built from a row's golds, with the field each must fail."""
    cmp, calc, after = row["gold_cmp"], row["gold_calc"], json.loads(row["gold_after"])
    tr = row["targets"]["trace"]
    out = {
        "trace sorted keys": (json.dumps(json.loads(tr), separators=(",", ":"), sort_keys=True), "format"),
        "trace order calc,cmp,after": ('{"calc":' + C._dump(calc) + ',"cmp":' + C._dump(cmp) + ',"after":' + env.dumps(after) + "}", "format"),
        "trace extra key": (tr[:-1] + ',"note":1}', "format"),
        "trace duplicate after": (tr[:-1] + ',"after":' + env.dumps(after) + "}", "format"),
        "trace missing after": ('{"cmp":' + C._dump(cmp) + ',"calc":' + C._dump(calc) + "}", "format"),
        "trace truncated": (tr[: int(len(tr) * 0.8)], "format"),
        "trace + trailing text": (tr + " done", "format"),
        "plain text to trace parser": (row["targets"]["plain"], "format"),
        "filler text to trace parser": (row["targets"]["filler"], "format"),
    }
    if env.dumps(row["before"]) != row["gold_after"]:
        out["trace with copied after"] = ('{"cmp":' + C._dump(cmp) + ',"calc":' + C._dump(calc) + ',"after":' +
                                          env.dumps(row["before"]) + "}", "after")
    if calc is not None:
        bad = dict(calc, mod=(calc["mod"] + 1) % C.MOD)
        out["trace wrong calc"] = ('{"cmp":' + C._dump(cmp) + ',"calc":' + C._dump(bad) + ',"after":' + env.dumps(after) + "}",
                                   "calc")
        out["trace calc keys reordered"] = ('{"cmp":' + C._dump(cmp) + ',"calc":' +
                                            C._dump({"mod": calc["mod"], "raw": calc["raw"]}) + ',"after":' + env.dumps(after) + "}",
                                            "calc")
    if cmp is not None:
        flip = dict(cmp, ok=not cmp["ok"])
        out["trace flipped ok"] = ('{"cmp":' + C._dump(flip) + ',"calc":' + C._dump(calc) + ',"after":' + env.dumps(after) + "}",
                                   "cmp")
        out["trace ok as 1/0"] = ('{"cmp":' + C._dump(dict(cmp, ok=int(cmp["ok"]))) + ',"calc":' + C._dump(calc) +
                                  ',"after":' + env.dumps(after) + "}", "cmp")
    return out


def _filler_negs(row: dict) -> dict:
    f = row["targets"]["filler"]
    d = json.loads(f)
    return {"filler keys reversed": ('{"after":' + env.dumps(json.loads(row["gold_after"])) + ',"fill":"' + d["fill"] + '"}', "format"),
            "filler extra key": (f[:-1] + ',"x":1}', "format"),
            "trace text to filler parser": (row["targets"]["trace"], "format")}


def scorer_selftest(dev_path: Path = C.DEV, chains_path: Path = C.CHAINS) -> dict:
    dev = C.load_dev(dev_path)
    parents = C.load_chain_bank(chains_path)
    D = DevIndex(dev)
    res = {"golds": {}, "negatives": {}, "floors": {}, "metrics": {}, "chains": {}}
    for side in C.SIDES:
        sc = score_side(dev, [r["targets"][C.SIDE_ARM[side]] for r in dev], side)
        fields = ["after_ok", "parsed"] + (["format_ok", "cmp_ok", "calc_ok", "after_follows_calc"] if side == "trace" else []) \
            + (["format_ok", "fill_exact"] if side == "filler" else [])
        res["golds"][side] = {f: _r(np.mean([s[f] for s in sc])) for f in fields}
        if side == "trace":
            cond = [i for i, r in enumerate(dev) if r["event"]["kind"] == "conditional"]
            res["golds"][side]["ok_ok|cond"] = _r(np.mean([sc[i]["ok_ok"] for i in cond]))
            res["golds"][side]["calc_follows_cmp|cond"] = _r(np.mean([sc[i]["calc_follows_cmp"] for i in cond]))
    neg = Counter()
    neg_n = Counter()
    neg_fail = []
    for r in dev:
        for name, (text, must), arm in [(k, v, "trace") for k, v in _neg_variants(r).items()] +                 [(k, v, "filler") for k, v in _filler_negs(r).items()]:
            s = C.score_text(r, text, arm)
            if must == "format":
                bad = s["format_ok"] or s["parsed"]
            elif must == "after":
                bad = s["after_ok"]
            elif must == "calc":
                bad = s["calc_ok"] or not s["after_ok"]          # the after-only parser still reads the right after
            else:                                            # cmp: wrong predicate field, after still read right
                bad = s["cmp_ok"] or (name == "trace flipped ok" and s["ok_ok"]) or not s["after_ok"]
            neg_n[name] += 1
            neg[name] += int(bool(bad))
            if bad and len(neg_fail) < 5:
                neg_fail.append((r["id"], name, text))
    res["negatives"] = {k: {"n": neg_n[k], "wrongly_accepted": neg[k]} for k in sorted(neg_n)}
    res["negative_failures"] = neg_fail
    # floors: copy (never correct: every dev row changes the state) and branch heuristics on conditional strata
    for h in ("copy", "always_add", "always_sub", "unreduced"):
        fl = {}
        for s in C.DEV_STRATA:
            rs = [r for r in dev if r["stratum"] == s]
            app = [r for r in rs if C.heuristic_after(h, r["before"], r["event"]) is not None]
            if not app:
                continue
            got = np.mean([C.score_text(r, env.dumps(C.heuristic_after(h, r["before"], r["event"])), "plain")["after_ok"]
                           for r in app])
            if h == "copy":
                exp = np.mean([r["gold_after"] == env.dumps(r["before"]) for r in app])
            elif h == "always_add":
                exp = np.mean([bool(r["gold_ok"]) for r in app])
            elif h == "always_sub":
                exp = np.mean([not r["gold_ok"] for r in app])
            else:
                exp = np.mean([not C.wraps(r["before"], r["event"]) for r in app])
            fl[s] = {"scored": _r(got), "expected": _r(exp), "exact": abs(got - exp) < 1e-12}
        res["floors"][h] = fl
    # metric identities
    ones = np.ones(len(dev))
    sub = np.array([C.score_text(r, env.dumps(C.heuristic_after("always_sub", r["before"], r["event"]) or r["before"]),
                                 "plain")["after_ok"] for r in dev], float)
    res["metrics"] = {"perfect": {m: _r(metric_point(ones, D, m)) for m in GROUPS},
                      "always_sub": {m: _r(metric_point(sub, D, m)) for m in ("primary_conditional", "conditional_in")},
                      "boot_identical_sides": metric_boot_diff(ones, ones, D, "primary_conditional", 2000)}
    # chain pipeline identities: gold chains -> regret 0, exact 1, product law 1; copy chains -> the copy floor
    gold_gen = {}
    copy_gen = {}
    for ch in C.flat_chains(parents):
        g = env.rollout(FAM, ch["s0"], ch["events"])
        gold_gen[ch["cand_id"]] = {"states": [env.dumps(s) for s in g], "texts": [env.dumps(s) for s in g[1:]]}
        copy_gen[ch["cand_id"]] = {"states": [env.dumps(ch["s0"])] * (C.H + 1), "texts": [env.dumps(ch["s0"])] * C.H}
    rows = chain_rows_for(parents)
    pids, gr = regret(rows, 4, gold_gen)
    _, cr = regret(rows, 4, copy_gen)
    _, fr = floor_copy_regret(rows, 4)
    pl = product_law(parents, gold_gen)
    res["chains"] = {"gold_regret_k4_mean": _r(np.nanmean(gr)), "gold_exact_k4": _r(parent_exact(rows, 4, gold_gen, pids).mean()),
                     "copy_regret_equals_floor": bool(np.array_equal(cr, fr, equal_nan=True)),
                     "gold_product_law": {k: pl[k] for k in ("predicted_k4_exact_product", "observed_k4_exact")},
                     "n_parents": len(pids)}
    ok = all(v == 1.0 for side in res["golds"].values() for v in side.values()) \
        and all(v["wrongly_accepted"] == 0 for v in res["negatives"].values()) \
        and all(c["exact"] for fl in res["floors"].values() for c in fl.values()) \
        and all(v == 1.0 for v in res["metrics"]["perfect"].values()) \
        and res["metrics"]["always_sub"] == {"primary_conditional": 0.5, "conditional_in": 0.5} \
        and res["metrics"]["boot_identical_sides"] == [0.0, 0.0] \
        and res["chains"]["gold_regret_k4_mean"] == 0.0 and res["chains"]["gold_exact_k4"] == 1.0 \
        and res["chains"]["copy_regret_equals_floor"] \
        and res["chains"]["gold_product_law"] == {"predicted_k4_exact_product": 1.0, "observed_k4_exact": 1.0}
    res["ok"] = bool(ok)
    res["n_rows"] = len(dev)
    res["n_negatives"] = int(sum(neg_n.values()))
    res["dev_sha256"] = C.sha256(dev_path)
    res["chains_sha256"] = C.sha256(chains_path)
    res["code_sha256"] = {r: C.sha256(ROOT / r) for r in ("scripts/t2br_analysis.py", "scripts/t2br_common.py")}
    return res


# ---------------------------------------------------------------- dry run (synthetic sides with known verdicts)
def _coin(row_key: str, salt: str) -> bool:
    return int(hashlib.sha256(f"{salt}|{row_key}".encode()).hexdigest()[:8], 16) % 2 == 0


def _answer(before: dict, ev: dict, how: str, salt: str) -> dict:
    """A synthetic model's next state. how: gold | offlike (unreduced arithmetic, coin-flip branch) | plainlike (mod
    right in range, unreduced out of range, coin-flip branch) | swapbad (gold except swap is copied) | worse_in (gold
    out of range, coin-flip branch and unreduced mix in range)."""
    key = env.dumps({"b": before, "e": ev})
    v = ev.get("value", 0)
    oor = abs(v) > C.IN_V[1] or v < C.IN_V[0] or not (C.THR_IN[0] <= ev.get("threshold", 0) <= C.THR_IN[1])
    if how == "gold" or (how == "worse_in" and oor):
        return env.step(FAM, before, ev)
    if how == "swapbad":
        return dict(before) if ev["kind"] == "swap" else env.step(FAM, before, ev)
    k, r = ev["kind"], ev["register"]
    b = dict(before)
    if k == "swap":
        return env.step(FAM, before, ev)
    if k == "conditional":
        add = _coin(key, salt)
        raw = before[r] + (v if add else -v)
    else:
        raw = C.raw_value(before, ev)
    reduce_ = how in ("plainlike",) and not oor or how == "worse_in" and k != "mix"
    b[r] = raw % C.MOD if reduce_ else raw
    return b


def _fmt(before: dict, ev: dict, st: dict, arm: str, count) -> str:
    """Render a synthetic next state in an arm's format (trace fields from the synthetic state itself)."""
    if arm == "plain":
        return env.dumps(st)
    cmp, calc, gold = C.parts(before, ev)
    if ev["kind"] != "swap" and env.dumps(st) != env.dumps(gold):    # a wrong synthetic answer: calc from its own value
        y = st[ev["register"]]
        calc = {"raw": y, "mod": y % C.MOD}
    tr = C.trace_text(cmp, calc, st)
    if arm == "trace":
        return tr
    if arm == "trace_sorted":
        return json.dumps(json.loads(tr), separators=(",", ":"), sort_keys=True)
    return C.matched_filler(tr, st, count)[0]


SCENARIOS = {
    # name: ({side: (how, answer format)}, (match, expected verdict)); one coin per scenario, shared by its sides
    "go": ({"off": ("offlike", "plain"), "plain": ("plainlike", "plain"), "trace": ("gold", "trace"),
            "filler": ("plainlike", "filler")}, ("exact", "GO")),
    "filler_equals_trace": ({"off": ("offlike", "plain"), "plain": ("plainlike", "plain"), "trace": ("gold", "trace"),
                             "filler": ("gold", "filler")},
                            ("exact", "STOP (decision point 3): extra decode compute, not supervision")),
    "nothing_helps": ({"off": ("offlike", "plain"), "plain": ("plainlike", "plain"), "trace": ("plainlike", "trace"),
                       "filler": ("plainlike", "filler")},
                      ("exact", "STOP (decision point 3): extra decode compute, not supervision - neither format helped "
                                "(trace ~ filler ~ plain)")),
    "controls_lost": ({"off": ("gold", "plain"), "plain": ("plainlike", "plain"), "trace": ("swapbad", "trace"),
                       "filler": ("plainlike", "filler")}, ("exact", "NO-GO (failed: G4)")),
    "in_dist_worse": ({"off": ("offlike", "plain"), "plain": ("gold", "plain"), "trace": ("worse_in", "trace"),
                       "filler": ("offlike", "filler")}, ("exact", "NO-GO (failed: G1; G2; G5)")),
    "malformed_traces": ({"off": ("offlike", "plain"), "plain": ("plainlike", "plain"), "trace": ("gold", "trace_sorted"),
                          "filler": ("plainlike", "filler")}, ("prefix", "STOP (decision point 3)")),
}


def synth_outputs(dev: list, parents: list, spec: dict, salt: str, count) -> tuple:
    texts, gens = {}, {}
    for side, (how, fmt) in spec.items():
        arm = "trace" if fmt == "trace_sorted" else fmt
        texts[side] = [_fmt(r["before"], r["event"], _answer(r["before"], r["event"], how, salt), fmt, count)
                       for r in dev]
        g = {}
        for ch in C.flat_chains(parents):
            st, tx = [ch["s0"]], []
            for ev in ch["events"]:
                if st[-1] is None:
                    st.append(None)
                    continue
                t = _fmt(st[-1], ev, _answer(st[-1], ev, how, salt), fmt, count)
                tx.append(t)
                st.append(C.parse_after(t, arm))
            g[ch["cand_id"]] = {"cand_id": ch["cand_id"], "states": [None if s is None else env.dumps(s) for s in st],
                                "texts": tx}
        gens[side] = g
    return texts, gens


def dryrun(n_boot: int = 2000) -> dict:
    dev = C.load_dev()
    parents = C.load_chain_bank()
    count = C.token_counter()
    out = {"scenarios": {}, "n_boot": n_boot}
    for name, (spec, want) in SCENARIOS.items():
        t0 = time.time()
        texts, gens = synth_outputs(dev, parents, spec, name, count)
        res = analyse_seed(dev, parents, texts, gens, n_boot)
        v = res["rule"]["verdict"]
        hit = v == want[1] if want[0] == "exact" else v.startswith(want[1])
        out["scenarios"][name] = {"expected": list(want), "verdict": v, "ok": hit,
                                  "values": res["rule"]["values"], "seconds": round(time.time() - t0, 1),
                                  "trace_format_ok_mean": _r(np.mean([f["format_ok"] for f in
                                                                      res["onestep"]["trace"]["trace_fields"].values()])),
                                  "product_law_ALL_trace": {k: res["chains"]["ALL"]["product_law"]["trace"][k]
                                                            for k in ("predicted_k4_exact_product", "observed_k4_exact")},
                                  "regret_k4_ALL": {s: res["chains"]["ALL"]["k"]["4"]["sides"][s]["regret"]["mean"]
                                                    for s in C.SIDES}}
        print(f"  [{'PASS' if out['scenarios'][name]['ok'] else 'FAIL'}] {name}: {v}", flush=True)
        if name == "go":
            go_res = res
    # extra properties
    go = out["scenarios"]["go"]
    out["checks"] = {
        "go: gold trace side product law predicted == observed == 1": go["product_law_ALL_trace"] ==
        {"predicted_k4_exact_product": 1.0, "observed_k4_exact": 1.0},
        "go: gold trace side regret k4 == 0": go["regret_k4_ALL"]["trace"] == 0.0,
        "malformed: every sorted-key trace rejected (format_ok 0)": out["scenarios"]["malformed_traces"]["trace_format_ok_mean"] == 0.0,
        "go: trace fields of a gold trace all 1.0": all(v == 1.0 for f in go_res["onestep"]["trace"]["trace_fields"].values()
                                                        for v in f.values()),
    }
    for k, v in out["checks"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", flush=True)
    # multi-seed overall label (pure function of verdicts)
    vs = {"s0": "GO", "s1": "NO-GO (failed: G1)"}
    out["checks"]["multi-seed: one NO-GO seed -> not GO in every seed"] = not all(v == "GO" for v in vs.values())
    out["ok"] = all(s["ok"] for s in out["scenarios"].values()) and all(out["checks"].values())
    out["code_sha256"] = {r: C.sha256(ROOT / r) for r in ("scripts/t2br_analysis.py", "scripts/t2br_common.py")}
    out["dev_sha256"] = C.sha256(C.DEV)
    out["example_report_md"] = render({"seeds": ["s0"], "overall": "SCREEN (seed 0): " + go_res["rule"]["verdict"] +
                                       " [SYNTHETIC DRY RUN]", "go_rule": C.GO_RULE_TEXT, "per_seed": {"s0": clean(go_res)}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--dryrun", action="store_true")
    g.add_argument("--run-dir", action="append")
    ap.add_argument("--exploratory", action="store_true", help="analyse an unregistered run (labelled as such)")
    a = ap.parse_args()
    frozen = (ROOT / "docs" / "t2b" / "register" / "SCREEN_FREEZE.json").exists()
    if a.selftest:
        res = scorer_selftest()
        print(json.dumps({k: res[k] for k in ("golds", "metrics", "chains", "ok", "n_rows", "n_negatives")}, indent=1))
        if frozen:
            print("SCREEN_FREEZE.json exists: SCORER_SELFTEST.json is a frozen input and is not rewritten")
        else:
            DOCS.mkdir(parents=True, exist_ok=True)
            (DOCS / "SCORER_SELFTEST.json").write_text(json.dumps(clean(res), indent=1) + "\n", encoding="utf-8")
        return 0 if res["ok"] else 1
    if a.dryrun:
        res = dryrun()
        md = res.pop("example_report_md")
        if frozen:
            print("SCREEN_FREEZE.json exists: DRYRUN.json is a frozen input and is not rewritten")
        else:
            DOCS.mkdir(parents=True, exist_ok=True)
            (DOCS / "DRYRUN.json").write_text(json.dumps(clean(res), indent=1) + "\n", encoding="utf-8")
            (DOCS / "DRYRUN.md").write_text("<!-- synthetic dry run: NOT model output -->\n" + md, encoding="utf-8")
        print(f"dry run {'PASS' if res['ok'] else 'FAIL'}")
        return 0 if res["ok"] else 1
    res = analyse_runs([ROOT / d if not Path(d).is_absolute() else Path(d) for d in a.run_dir], registered=not a.exploratory)
    print(res["overall"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
