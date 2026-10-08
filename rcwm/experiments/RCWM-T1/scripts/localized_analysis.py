"""Analysis of scripts/run_localized.py: off-task drift against the transition skill per block-localised
delta arm (exploratory; CPU; numpy/json only, never imports torch or loads a model).

  python scripts/localized_analysis.py [--run-dir runs/bank_pilot/localized] [--out docs/localized/LOCALIZED_RESULTS.md]

Writes LOCALIZED_RESULTS.md and LOCALIZED_RESULTS.json (same stem). Deterministic: fixed bootstrap seeds,
--boot resamples (10,000), no timestamps of its own. Arms: supa (identity row), loc8_15 (Supa + Titan's
blocks 8-15), sparse (full delta), up16_27 (Supa + Titan's blocks 16-27); every comparison is arm vs Supa
from the same run (one process, one Supa load, arms spliced in place).

OFF-TASK DRIFT (the 248 neutral prompts of data/offtask_prompts.jsonl)
  * first-token KL(arm||supa) and KL(supa||arm) per prompt: t2_analysis.lp_compare (stored fp16 log-probs,
    clamped at -60 by capture_offtask, renormalised by a float64 log-softmax); mean and median with a
    percentile bootstrap 95% CI over prompts; top-1 agreement.
  * 64-token greedy replies: exact agreement with Supa's reply, overall (bootstrap CI over prompts) and per
    reply_class; JSON-bleed rate exactly as t2_analysis.forgetting_block (replies to prompts outside
    NUMERIC_CLASSES that look like a JSON state/object or a bare integer) and the JSON-state rate over all.
  * answer-slot state shift per capture layer (hidden_states index; 28 = after the final norm): per prompt
    the RMS over dimensions of (h_arm - h_supa) / sd_supa, where sd_supa is Supa's per-dimension SD over the
    prompts, floored at 1e-3 x that layer's median SD (layers where Supa's state is constant over prompts,
    e.g. layer 0 = the shared last template token, are reported raw only); mean with a bootstrap CI. Also
    the relative L2 shift ||h_arm - h_supa|| / ||h_supa|| without Supa's top-5 massive dims (R5's
    convention), and the share of prompts whose stored states are bit-identical to Supa's.
    Build check from the states (independent of the weight hashes): blocks below an arm's first Titan block
    are Supa's, so loc8_15 and sparse must equal Supa exactly at layers <= 8, up16_27 at layers <= 16, and
    loc8_15 must equal sparse exactly at layers <= 16 (same batches in one process).
  * graded greedy measure: prefix divergence 1 - (common-prefix characters) / (characters of the longer
    reply) per prompt (0 = identical, 1 = differ at the first character), so a ratio is not pinned at 1
    when nearly every sparse reply differs from Supa's somewhere.
  * KEY RATIOS drift(arm) / drift(sparse) for arm in {loc8_15, up16_27}: mean KL(arm||supa), mean
    KL(supa||arm) (nats), greedy disagreement (1 - exact agreement), greedy prefix divergence, top-1
    disagreement, and the per-layer state shift; paired percentile bootstrap over prompts (the same
    resampled prompts in numerator and denominator; resamples with a zero denominator are dropped and
    counted). The paired difference arm - sparse is reported next to each ratio.
  * DENOMINATOR DIAGNOSTICS (whether a ratio can mean anything): prompts on which sparse flips Supa's top-1
    token, sparse's greedy disagreement count and rate, and the share of sparse's KL carried by the top 10%
    of prompts. Flags: top-1 ratio UNINTERPRETABLE if sparse flips < 10 prompts; exact-greedy ratio
    UNINTERPRETABLE if sparse's greedy disagreement rate > 0.9 (ceiling) or < 10 prompts disagree (floor);
    prefix-divergence ratio UNINTERPRETABLE if < 10 prompts disagree; KL ratios CAUTION if sparse flips
    < 10 prompts (weak denominator) or the top 10% of prompts carry > 50% of sparse's KL (concentrated).
    Flags are reported next to the ratios and the D-23 verdict; they never change the verdict.
  Noise floor (if runs/bank_pilot/sparse/offtask_logprobs.npy exists): KL(stored Titan-Sparse || this
  run's sparse arm) and top-1 agreement, a cross-session reproduction of the full-delta arm.

PRE-STATED READING (DECISIONS D-23, written before any localized data existed; docs/localized/DESIGN.md)
  "localisation reduces interference" iff the KL ratio loc8_15/sparse has an upper 95% CI < 0.8 AND
  regret(loc8_15) - regret(sparse) <= +0.05 in every family. "No reduction" if the ratio's lower CI > 0.9.
  Otherwise inconclusive.
  Operationalised (DESIGN.md, before the run): KL ratio = mean KL(arm||supa) of loc8_15 / of sparse with the
  paired percentile bootstrap CI over prompts above (PRIMARY direction; the same rule on KL(supa||arm) is
  reported alongside as secondary); regret = chain_one k=4 normalised regret, the difference = the paired
  per-parent mean difference loc8_15 - sparse (decide.paired_bootstrap mean_diff; its CI shown), required
  <= +0.05 (+1e-12 float slack) in each of L, P, R; a family without a finite difference fails the leg.
  Written to the JSON as "d23_verdict" and at the top of the MD. Families whose copy-chain floor is within
  0.10 of sparse's regret are marked low-headroom (the +0.05 leg can barely fail there); their one-step FR
  is shown beside the regret leg.

SKILL (the pilot TEST chains, chain_one)
  * k=4 normalised regret per family: rcwmt.decisions.g_only(data.candidate_table(bank), fam, 4,
    {"chain_one": rows}, modes=("chain_one",)). g_only seeds its tie-break rng (17) per call and chain_one is
    its first mode, so these are the chain_one numbers t2_analysis.decision_block computes. Mean with
    decide.bootstrap_mean CI; arm - sparse with decide.paired_bootstrap (parents paired); copy-chain floor.
  * one-step accuracy from the FIRST chain step: states[1] == gold s1. That prompt is the bank's one-frame
    item cand_id-one, i.e. run_patch's item, so this is R12's measure on the same 640 prompts (batched
    differently): accuracy on all and on state-changing steps with parent-cluster bootstrap CIs
    (t2_analysis.cluster_boot), arm - sparse paired by chain, and FR = (acc - A_supa) / (A_sparse - A_supa)
    on changed steps (R12's definition; undefined if the gap < 0.10) with a parent bootstrap CI.
  * exact-state chain accuracy at k = 1, 2, 4.
  Consistency (descriptive): first-step texts vs run_patch's texts of the same items (base:supa,
  base:sparse, HlowT@16, HupT@16); supa / sparse chains vs the stored pilot chains (runs/bank_pilot/<arm>).
  The stored pilot chains predate the 160-token cap and the D-14 batch budget, so identical_chains < 1 is
  expected there (mostly supa); it is not a splice check.

INPUT VALIDATION (any failure stops with SystemExit, nothing is written): DONE.json present; every stage
file's sha256 equals STAGES.json; the four ARM.json records show the intended composition; off-task metas
list the same prompt ids and classes in every arm, equal to data/offtask_prompts.jsonl (with --allow-subset:
a subsequence of it, for the tiny test); log-prob arrays share one shape; offtask_gen.jsonl rows follow
the meta ids; chain files pass t2_analysis.chain_errors against the pilot TEST candidates and cover all 640
(with --allow-subset: complete parents only).
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
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import t2_analysis as T  # noqa: E402  (installs the rcwmt.heads placeholder; asserts torch is not imported)
from rcwmt import data, decide, decisions, env  # noqa: E402

assert "torch" not in sys.modules, "the localized analysis must not import torch"

ARMS = ("supa", "loc8_15", "sparse", "up16_27")
CMP = ("loc8_15", "up16_27")
FIRST_TITAN = {"supa": None, "loc8_15": 8, "sparse": 8, "up16_27": 16}       # first block that differs from Supa
EXPECTED_DIFF = {"supa": [], "loc8_15": [f"block{i}" for i in range(8, 16)],
                 "sparse": [f"block{i}" for i in range(8, 28)], "up16_27": [f"block{i}" for i in range(16, 28)]}
REF_FILES = {"supa": "gen_base_supa.json", "sparse": "gen_base_sparse.json",
             "loc8_15": "gen_HlowT_16.json", "up16_27": "gen_HupT_16.json"}
STAGE_FILES = {"offtask": ("offtask_meta.json", "offtask_logprobs.npy", "offtask_last.npy"),
               "offtask_gen": ("offtask_gen.jsonl",), "chain_one": ("gen_chain_one.jsonl",)}
SHORT = {f: T.CODE[f] for f in env.FAMILIES}
BOOT = 10000
SEED = 20261008
FR_MIN_GAP = 0.10
SD_FLOOR = 1e-3
N_MASSIVE = 5
# denominator diagnostics
MIN_DEN_PROMPTS = 10            # sparse must flip / disagree on at least this many prompts
GREEDY_CEILING = 0.9            # sparse's exact-greedy disagreement rate above this pins the ratio near 1
KL_TOP_SHARE_MAX = 0.5          # share of sparse's KL carried by the top 10% of prompts
# D-23 pre-stated reading (DECISIONS.md D-23; docs/localized/DESIGN.md)
D23_RULE = ('"localisation reduces interference" iff the KL ratio loc8_15/sparse has an upper 95% CI < 0.8 AND '
            'regret(loc8_15) − regret(sparse) ≤ +0.05 in every family. "No reduction" if the ratio\'s lower CI > 0.9. '
            'Otherwise inconclusive.')
D23_KL_UPPER, D23_KL_LOWER, D23_REGRET_TOL = 0.8, 0.9, 0.05
D23_PRIMARY, D23_SECONDARY = "kl_arm_supa", "kl_supa_arm"
REDUCES, NO_REDUCTION, INCONCLUSIVE = "localisation reduces interference", "no reduction", "inconclusive"
LOW_HEADROOM = 2 * D23_REGRET_TOL   # copy floor - sparse regret below this: the regret leg can barely fail


def die(what: str, errs: list[str]):
    T.die("localized_analysis: " + what, errs)


def sha256(path: Path) -> str:
    return T.sha256(path)


# ------------------------------------------------------------------------------------------ loading
class Run:
    def __init__(self, run_dir: Path, allow_subset: bool):
        self.dir = run_dir
        errs = []
        if not (run_dir / "DONE.json").exists():
            die(f"{run_dir} has no DONE.json (run_localized.py writes it only when every arm and stage is done)", ["missing"])
        self.done = json.loads((run_dir / "DONE.json").read_text(encoding="utf-8"))
        self.run = json.loads((run_dir / "RUN.json").read_text(encoding="utf-8"))
        self.arm_rec, self.stages = {}, {}
        for arm in ARMS:
            d = run_dir / arm
            try:
                self.arm_rec[arm] = json.loads((d / "ARM.json").read_text(encoding="utf-8"))
                self.stages[arm] = json.loads((d / "STAGES.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                errs.append(f"{arm}: ARM.json / STAGES.json unreadable ({e})")
                continue
            if self.stages[arm].get("fingerprint") != self.arm_rec[arm].get("fingerprint"):
                errs.append(f"{arm}: STAGES.json fingerprint differs from ARM.json")
            if self.arm_rec[arm].get("differs_from_supa") != EXPECTED_DIFF[arm]:
                errs.append(f"{arm}: differs from Supa in {self.arm_rec[arm].get('differs_from_supa')}, expected {EXPECTED_DIFF[arm]}")
            for st, files in STAGE_FILES.items():
                rec = self.stages[arm].get("stages", {}).get(st)
                if rec is None:
                    errs.append(f"{arm}: stage {st} not recorded")
                    continue
                for f in files:
                    if not (d / f).exists():
                        errs.append(f"{arm}/{f} missing")
                    elif sha256(d / f) != rec.get("sha256", {}).get(f):
                        errs.append(f"{arm}/{f}: sha256 differs from STAGES.json (file changed after the stage finished)")
        if errs:
            die("run outputs fail validation", errs)
        # ---- off-task
        prompts = T.load_jsonl(T.OFFTASK_PROMPTS)
        file_ids = [r["id"] for r in prompts]
        self.meta = {arm: json.loads((run_dir / arm / "offtask_meta.json").read_text(encoding="utf-8")) for arm in ARMS}
        m0 = self.meta["supa"]
        self.ids, self.classes, self.layers = m0["ids"], m0["reply_class"], m0["layers"]
        for arm in ARMS:
            m = self.meta[arm]
            if m["ids"] != self.ids or m["reply_class"] != self.classes or m["layers"] != self.layers:
                errs.append(f"{arm}: offtask_meta ids/classes/layers differ from supa's")
        if allow_subset:
            pos = {i: k for k, i in enumerate(file_ids)}
            ks = [pos.get(i) for i in self.ids]
            if any(k is None for k in ks) or ks != sorted(ks) or len(set(ks)) != len(ks):
                errs.append("off-task ids are not a subsequence of data/offtask_prompts.jsonl")
            elif [prompts[k]["reply_class"] for k in ks] != self.classes:
                errs.append("off-task reply classes differ from data/offtask_prompts.jsonl")
        elif self.ids != file_ids or self.classes != [r["reply_class"] for r in prompts]:
            errs.append(f"off-task ids/classes differ from data/offtask_prompts.jsonl ({len(self.ids)} vs {len(file_ids)})")
        self.lp = {arm: np.load(run_dir / arm / "offtask_logprobs.npy", mmap_mode="r") for arm in ARMS}
        self.h = {arm: np.load(run_dir / arm / "offtask_last.npy", mmap_mode="r") for arm in ARMS}
        for arm in ARMS:
            if self.lp[arm].shape != self.lp["supa"].shape or self.lp[arm].shape[0] != len(self.ids):
                errs.append(f"{arm}: offtask_logprobs shape {self.lp[arm].shape}")
            if self.h[arm].shape != self.h["supa"].shape or self.h[arm].shape[:2] != (len(self.ids), len(self.layers)):
                errs.append(f"{arm}: offtask_last shape {self.h[arm].shape}")
        self.gen = {}
        for arm in ARMS:
            rows = T.load_jsonl_checked(run_dir / arm / "offtask_gen.jsonl")
            if [r.get("id") for r in rows] != self.ids or [r.get("reply_class") for r in rows] != self.classes:
                errs.append(f"{arm}: offtask_gen.jsonl rows do not follow the meta ids/classes")
            if not all(isinstance(r.get("text"), str) for r in rows):
                errs.append(f"{arm}: offtask_gen.jsonl rows without a string text")
            self.gen[arm] = [r.get("text") for r in rows]
        # ---- chains
        self.bank = json.loads((ROOT / "data" / f"{self.run.get('bank', 'bank_pilot')}.json").read_text(encoding="utf-8"))
        all_cands = T.test_cands(self.bank)
        self.chains = {}
        present = None
        for arm in ARMS:
            rows = T.load_jsonl_checked(run_dir / arm / "gen_chain_one.jsonl")
            ids = [r.get("cand_id") for r in rows]
            if present is None:
                present = ids
            elif ids != present:
                errs.append(f"{arm}: chain cand_ids differ from supa's")
            sub = {c: all_cands[c] for c in ids if c in all_cands}
            errs += T.chain_errors(rows, sub if allow_subset else all_cands, f"{arm}/gen_chain_one.jsonl")
            if len(sub) != len(ids):
                errs.append(f"{arm}: {len(ids) - len(sub)} chain cand_ids are not pilot TEST candidates")
            self.chains[arm] = {r["cand_id"]: r for r in rows}
        self.cands = {c: all_cands[c] for c in (present or []) if c in all_cands}
        by_parent = defaultdict(set)
        for c, (p, _) in all_cands.items():
            by_parent[p["parent_id"]].add(c)
        partial = [pid for pid in {p["parent_id"] for p, _ in self.cands.values()} if not by_parent[pid] <= set(self.cands)]
        if partial:
            errs.append(f"{len(partial)} parents have only some of their candidates (g_only needs complete parents)")
        if errs:
            die("run outputs fail validation", errs)
        self.subset = len(self.ids) != len(file_ids) or len(self.cands) != len(all_cands)


# ------------------------------------------------------------------------------------- statistics
def boot_idx(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (B, n))


def pct(x) -> list:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [float("nan"), float("nan")]


def summary(x: np.ndarray, idx: np.ndarray) -> dict:
    x = np.asarray(x, float)
    xb = x[idx]
    return {"n": int(len(x)), "mean": float(x.mean()), "mean_ci95": pct(xb.mean(1)), "median": float(np.median(x)),
            "median_ci95": pct(np.median(xb, axis=1)), "p90": float(np.quantile(x, 0.9)), "max": float(x.max())}


def ratio(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> dict:
    num, den = np.asarray(num, float), np.asarray(den, float)
    nb, db = num[idx].mean(1), den[idx].mean(1)
    ok = db > 0
    return {"ratio": float(num.mean() / den.mean()) if den.mean() > 0 else float("nan"),
            "ci95": pct(nb[ok] / db[ok]), "dropped_zero_denominator": int((~ok).sum()),
            "diff": float(num.mean() - den.mean()), "diff_ci95": pct(nb - db),
            "num_mean": float(num.mean()), "den_mean": float(den.mean())}


def prefix_div(a: str, b: str) -> float:
    """Graded greedy disagreement in characters: 1 - common-prefix length / length of the longer reply
    (0 for identical replies, including two empty ones; 1 if they differ at the first character)."""
    if a == b:
        return 0.0
    return 1.0 - len(os.path.commonprefix([a, b])) / max(len(a), len(b))


def top_share(x, q: float = 0.10) -> float:
    """Share of sum(x) carried by the largest ceil(q * n) values."""
    x = np.sort(np.asarray(x, float))[::-1]
    tot = float(x.sum())
    return float(x[:max(1, int(math.ceil(q * len(x))))].sum() / tot) if tot > 0 else float("nan")


def ratio_flags(key: str, den: dict) -> list[dict]:
    """Interpretability flags of a drift ratio from its sparse denominator (module doc)."""
    f = []
    flips, ndis, rate = den["sparse_top1_flips"], den["sparse_greedy_disagreements"], den["sparse_greedy_disagree_rate"]
    if key == "top1_dis" and flips < MIN_DEN_PROMPTS:
        f.append({"level": "UNINTERPRETABLE", "why": f"sparse flips Supa's top-1 token on only {flips} prompts (< {MIN_DEN_PROMPTS})"})
    if key == "greedy_dis":
        if rate > GREEDY_CEILING:
            f.append({"level": "UNINTERPRETABLE", "why": f"ceiling: sparse's greedy replies differ from Supa's on {rate:.3f} "
                                                         f"of prompts (> {GREEDY_CEILING}), the ratio is pinned near 1; "
                                                         f"use the prefix-divergence ratio"})
        if ndis < MIN_DEN_PROMPTS:
            f.append({"level": "UNINTERPRETABLE", "why": f"floor: sparse's greedy reply differs on only {ndis} prompts (< {MIN_DEN_PROMPTS})"})
    if key == "greedy_prefix_div" and ndis < MIN_DEN_PROMPTS:
        f.append({"level": "UNINTERPRETABLE", "why": f"sparse's greedy reply differs on only {ndis} prompts (< {MIN_DEN_PROMPTS})"})
    if key in ("kl_arm_supa", "kl_supa_arm"):
        if flips < MIN_DEN_PROMPTS:
            f.append({"level": "CAUTION", "why": f"weak denominator: sparse flips Supa's top-1 token on only {flips} prompts"})
        sh = den["sparse_kl_top10pct_share"][key]
        if math.isfinite(sh) and sh > KL_TOP_SHARE_MAX:
            f.append({"level": "CAUTION", "why": f"concentrated: the top 10% of prompts carry {sh:.2f} of sparse's mean KL"})
    return f


# ---------------------------------------------------------------------------------------- off-task
def offtask_block(R: Run, B: int, seed: int, stored_root: Path) -> dict:
    n = len(R.ids)
    idx = boot_idx(n, B, seed)
    out = {"n_prompts": n, "kl": {}, "greedy": {}, "json_bleed": {}, "state_shift": {}, "ratios": {}}
    per = {"kl_arm_supa": {}, "kl_supa_arm": {}, "top1_dis": {}, "greedy_dis": {}, "greedy_prefix_div": {}}
    for arm in ARMS:
        k_as, a_arm, a_supa = T.lp_compare(R.lp[arm], R.lp["supa"])
        k_sa, _, _ = T.lp_compare(R.lp["supa"], R.lp[arm])
        per["kl_arm_supa"][arm], per["kl_supa_arm"][arm] = k_as, k_sa
        per["top1_dis"][arm] = (a_arm != a_supa).astype(float)
        out["kl"][arm] = {"arm_supa": summary(k_as, idx), "supa_arm": summary(k_sa, idx),
                          "top1_agree": float((a_arm == a_supa).mean()),
                          "top1_agree_ci95": pct(1.0 - per["top1_dis"][arm][idx].mean(1)),
                          "kl_arm_supa_per_prompt": [float(x) for x in k_as]}
        if arm != "sparse":
            k_ap, _, _ = T.lp_compare(R.lp[arm], R.lp["sparse"])
            out["kl"][arm]["arm_sparse"] = summary(k_ap, idx)
        agree = np.array([a == b for a, b in zip(R.gen[arm], R.gen["supa"])], bool)
        per["greedy_dis"][arm] = (~agree).astype(float)
        pdv = np.array([prefix_div(a, b) for a, b in zip(R.gen[arm], R.gen["supa"])], float)
        per["greedy_prefix_div"][arm] = pdv
        lcp = [len(os.path.commonprefix([a, b])) for a, b, s in zip(R.gen[arm], R.gen["supa"], agree) if not s]
        by = defaultdict(list)
        for c, s in zip(R.classes, agree):
            by[c].append(bool(s))
        out["greedy"][arm] = {"agree": float(agree.mean()), "agree_ci95": pct(agree[idx].mean(1)),
                              "disagree": float(1 - agree.mean()), "n_disagree": int((~agree).sum()),
                              "prefix_div_mean": float(pdv.mean()), "prefix_div_ci95": pct(pdv[idx].mean(1)),
                              "common_prefix_chars_mean_on_disagreeing": float(np.mean(lcp)) if lcp else None,
                              "by_class": {c: {"n": len(v), "agree": float(np.mean(v))} for c, v in sorted(by.items())}}
        nonnum = [t for t, c in zip(R.gen[arm], R.classes) if c not in T.NUMERIC_CLASSES]
        b = [T.looks_state(t) or T.int_only(t) for t in nonnum]
        js = [T.looks_state(t) for t in R.gen[arm]]
        out["json_bleed"][arm] = {"bleed_rate_nonnumeric": float(np.mean(b)) if b else float("nan"), "n_nonnumeric": len(nonnum),
                                  "json_state_rate_all": float(np.mean(js)), "n_all": len(js)}
    names = {"kl_arm_supa": "mean KL(arm||supa), nats", "kl_supa_arm": "mean KL(supa||arm), nats",
             "greedy_dis": "greedy 64-token exact disagreement with Supa",
             "greedy_prefix_div": "greedy 64-token prefix divergence from Supa (1 - common-prefix chars / longer reply)",
             "top1_dis": "top-1 first-token disagreement"}
    den = {"n_prompts": n, "sparse_top1_flips": int(per["top1_dis"]["sparse"].sum()),
           "sparse_greedy_disagreements": int(per["greedy_dis"]["sparse"].sum()),
           "sparse_greedy_disagree_rate": float(per["greedy_dis"]["sparse"].mean()),
           "sparse_kl_top10pct_share": {k: top_share(per[k]["sparse"]) for k in ("kl_arm_supa", "kl_supa_arm")},
           "rules": {"min_prompts": MIN_DEN_PROMPTS, "greedy_ceiling": GREEDY_CEILING, "kl_top10pct_share_max": KL_TOP_SHARE_MAX}}
    out["denominator"] = den
    for key, label in names.items():
        fl = ratio_flags(key, den)
        out["ratios"][key] = {"label": label, **{arm: ratio(per[key][arm], per[key]["sparse"], idx) for arm in CMP},
                              "flags": fl, "interpretable": not any(x["level"] == "UNINTERPRETABLE" for x in fl)}
    # ---- answer-slot states
    Hs = {arm: np.asarray(R.h[arm], np.float64) for arm in ARMS}
    shift_ratio = {}
    exact = {}
    for li, L in enumerate(R.layers):
        hs = Hs["supa"][:, li]
        sd = hs.std(0)
        med = float(np.median(sd))
        mass = np.argsort(-np.abs(hs).mean(0))[:N_MASSIVE]
        keep = np.ones(hs.shape[1], bool)
        keep[mass] = False
        rec = {"supa_sd_median": med, "massive_dims_excluded": [int(x) for x in mass]}
        rms = {}
        for arm in ARMS:
            dlt = Hs[arm][:, li] - hs
            ident = np.all(dlt == 0, axis=1)
            rel = np.linalg.norm(dlt[:, keep], axis=1) / np.maximum(np.linalg.norm(hs[:, keep], axis=1), 1e-12)
            r = {"identical_share": float(ident.mean()), "rel_l2_no_massive": summary(rel, idx)}
            if med > 0:
                z = dlt / np.maximum(sd, SD_FLOOR * med)
                rms[arm] = np.sqrt((z ** 2).mean(1))
                r["rms_z"] = summary(rms[arm], idx)
            else:
                r["rms_z"] = None
            rec[arm] = r
            exact.setdefault(arm, {})[str(L)] = bool(ident.all())
        loc_sp = np.all(Hs["loc8_15"][:, li] == Hs["sparse"][:, li], axis=1)
        exact.setdefault("loc8_15_vs_sparse", {})[str(L)] = bool(loc_sp.all())
        rec["note"] = "Supa constant over prompts: raw only" if med == 0 else ""
        out["state_shift"][str(L)] = rec
        if med > 0 and rms["sparse"].mean() > 0:
            shift_ratio[str(L)] = {arm: ratio(rms[arm], rms["sparse"], idx) for arm in CMP}
    out["ratios"]["state_shift_rms_z"] = {"label": "answer-slot state shift (RMS z) per layer", "per_layer": shift_ratio}
    # build check from the states
    chk = {}
    for arm in ARMS:
        lo = FIRST_TITAN[arm]
        want_eq = [str(L) for L in R.layers if lo is None or L <= lo]
        chk[arm] = {"expected_identical_layers": want_eq,
                    "identical_where_expected": all(exact[arm][L] for L in want_eq),
                    "identical_layers": [L for L, v in exact[arm].items() if v]}
    want = [str(L) for L in R.layers if L <= 16]
    chk["loc8_15_vs_sparse"] = {"expected_identical_layers": want,
                                "identical_where_expected": all(exact["loc8_15_vs_sparse"][L] for L in want),
                                "identical_layers": [L for L, v in exact["loc8_15_vs_sparse"].items() if v]}
    out["state_build_check"] = chk
    # ---- cross-session noise floor (stored Titan-Sparse capture)
    sp = stored_root / "sparse"
    out["noise_floor_stored_sparse"] = "not available"
    if (sp / "offtask_logprobs.npy").exists() and (sp / "offtask_meta.json").exists():
        ms = json.loads((sp / "offtask_meta.json").read_text(encoding="utf-8"))
        pos = {i: k for k, i in enumerate(ms.get("ids", []))}
        ST = np.load(sp / "offtask_logprobs.npy", mmap_mode="r")
        if all(i in pos for i in R.ids) and ST.shape[1] == R.lp["sparse"].shape[1]:
            rows = [pos[i] for i in R.ids]
            kl, a_st, a_run = T.lp_compare(np.asarray(ST[rows]), R.lp["sparse"])
            out["noise_floor_stored_sparse"] = {"kl_stored_run": summary(kl, idx), "top1_agree": float((a_st == a_run).mean()),
                                                "source": str(sp)}
    return out


# ------------------------------------------------------------------------------------------- skill
def skill_block(R: Run, B: int, seed: int, patch_dir: Path, stored_root: Path) -> dict:
    ct = data.candidate_table(R.bank)
    present = set(R.cands)
    rows = [r for r in ct if r["split"] != "test" or r["cand_id"] in present]
    gold = {(r["cand_id"], r["k"]): r["gold"] for r in rows if r["split"] == "test"}
    copy = {"chain_copy": {c: {"states": [env.dumps(p["s0"])] * (len(cc["events"]) + 1)} for c, (p, cc) in R.cands.items()}}
    out = {"families": {}, "per_parent_nregret": {}, "consistency": {}}
    for fam in env.FAMILIES:
        f = SHORT[fam]
        cids = [c for c, (p, _) in R.cands.items() if p["family"] == fam]
        if not cids:
            continue
        with warnings.catch_warnings():     # all-unparseable parents: "Mean of empty slice" inside g_only
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            G = {arm: decisions.g_only(rows, fam, 4, {"chain_one": R.chains[arm]}, modes=("chain_one",))["chain_one"] for arm in ARMS}
            cp = decisions.g_only(rows, fam, 4, copy, modes=("chain_copy",))["chain_copy"]
        pids = G["supa"]["pids"]
        assert all(G[arm]["pids"] == pids for arm in ARMS) and cp["pids"] == pids
        nr = {arm: np.asarray(G[arm]["nregret"], float) for arm in ARMS}
        fr = {"n_parents": len(pids), "copy_floor": float(np.nanmean(np.asarray(cp["nregret"], float))), "arms": {}}
        out["per_parent_nregret"][f] = {"pids": pids, **{arm: [None if math.isnan(x) else float(x) for x in nr[arm]] for arm in ARMS}}
        # one-step from the first chain step; chain accuracy at k = 1, 2, 4
        par = [R.cands[c][0]["parent_id"] for c in cids]
        s0 = {c: env.dumps(R.cands[c][0]["s0"]) for c in cids}
        chg = np.array([gold[(c, 1)] != s0[c] for c in cids], bool)
        step1 = {arm: np.array([R.chains[arm][c]["states"][1] == gold[(c, 1)] for c in cids], float) for arm in ARMS}
        accK = {arm: {k: np.array([R.chains[arm][c]["states"][k] == gold[(c, k)] for c in cids], float) for k in (1, 2, 4)}
                for arm in ARMS}
        upar = sorted(set(par))
        pix = np.array([upar.index(p) for p in par])
        bidx = boot_idx(len(upar), B, seed + 1)

        def cl_mean(x, mask):
            S = np.bincount(pix[mask], weights=x[mask], minlength=len(upar))
            N = np.bincount(pix[mask], minlength=len(upar)).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                return S[bidx].sum(1) / N[bidx].sum(1)

        A = {arm: float(step1[arm][chg].mean()) if chg.any() else float("nan") for arm in ARMS}
        Ab = {arm: cl_mean(step1[arm], chg) for arm in ARMS}
        for arm in ARMS:
            a_nr = nr[arm]
            rec = {"nregret": decide.bootstrap_mean(a_nr), "n_nan": int(np.isnan(a_nr).sum())}
            if arm != "sparse":
                rec["nregret_minus_sparse"] = decide.paired_bootstrap(a_nr, nr["sparse"])
            rec["onestep_all"] = {"n": len(cids), "acc": float(step1[arm].mean()),
                                  "ci95": list(T.cluster_boot(step1[arm], par, B, seed))}
            rec["onestep_changed"] = {"n": int(chg.sum()), "acc": A[arm],
                                      "ci95": list(T.cluster_boot(step1[arm][chg], [p for p, m in zip(par, chg) if m], B, seed))
                                      if chg.any() else None}
            if arm != "sparse":
                d = step1[arm] - step1["sparse"]
                rec["onestep_all_minus_sparse"] = {"diff": float(d.mean()), "ci95": list(T.cluster_boot(d, par, B, seed))}
            gap = A["sparse"] - A["supa"]
            if arm in CMP:
                with np.errstate(invalid="ignore", divide="ignore"):
                    frb = np.where(Ab["sparse"] - Ab["supa"] >= FR_MIN_GAP,
                                   (Ab[arm] - Ab["supa"]) / (Ab["sparse"] - Ab["supa"]), np.nan)
                rec["FR_changed"] = {"FR": float((A[arm] - A["supa"]) / gap) if gap >= FR_MIN_GAP else None,
                                     "ci95": pct(frb), "A_supa": A["supa"], "A_sparse": A["sparse"],
                                     "rule": f"FR = (acc - A_supa)/(A_sparse - A_supa) on changed first steps; undefined if gap < {FR_MIN_GAP}"}
            rec["chain_acc"] = {str(k): {"acc": float(accK[arm][k].mean()), "ci95": list(T.cluster_boot(accK[arm][k], par, B, seed))}
                                for k in (1, 2, 4)}
            rec["chains_complete"] = float(np.mean([R.chains[arm][c]["states"][-1] is not None for c in cids]))
            rec["steps_attempted"] = int(sum(len(R.chains[arm][c]["texts"]) for c in cids))
            fr["arms"][arm] = rec
        out["families"][f] = fr
    # consistency with run_patch (same prompts as the first chain step) and with the stored pilot chains
    for arm in ARMS:
        c = {}
        p = patch_dir / REF_FILES[arm]
        if p.exists():
            ref = json.loads(p.read_text(encoding="utf-8"))
            pairs = [(R.chains[arm][cid]["texts"][0], ref[f"{cid}-one"]) for cid in R.cands if f"{cid}-one" in ref]
            c["first_step_text_equals_run_patch"] = {"reference": str(p), "n": len(pairs),
                                                     "rate": float(np.mean([a == b for a, b in pairs])) if pairs else None}
        sp = stored_root / arm / "gen_chain_one.jsonl"
        if arm in ("supa", "sparse") and sp.exists():
            st = {r["cand_id"]: r for r in T.load_jsonl(sp)}
            cc = [cid for cid in R.cands if cid in st]
            same_states = [R.chains[arm][cid]["states"] == st[cid]["states"] for cid in cc]
            steps = [(a == b) for cid in cc for a, b in zip(R.chains[arm][cid]["texts"], st[cid]["texts"])]
            c["stored_pilot_chains"] = {"source": str(sp), "n": len(cc),
                                        "identical_chains": float(np.mean(same_states)) if cc else None,
                                        "identical_step_texts": float(np.mean(steps)) if steps else None}
        out["consistency"][arm] = c
    return out


# ----------------------------------------------------------------------------- D-23 pre-stated reading
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def d23_rule(kl_ci, regret_diff: dict, families) -> dict:
    """DECISIONS D-23 as a pure function of the loc8_15/sparse KL-ratio 95% CI [lo, hi] and the per-family
    paired mean regret difference loc8_15 - sparse ({family: float or None}). A non-finite CI satisfies
    neither KL leg; a family without a finite difference fails the regret leg."""
    lo, hi = (list(kl_ci or []) + [None, None])[:2]
    kl_finite = _finite(lo) and _finite(hi)
    kl_reduces = bool(kl_finite and hi < D23_KL_UPPER)
    kl_none = bool(kl_finite and lo > D23_KL_LOWER)
    reg = {f: bool(_finite(regret_diff.get(f)) and regret_diff[f] <= D23_REGRET_TOL + 1e-12) for f in families}
    reg_all = bool(reg) and all(reg.values())
    if kl_reduces and reg_all:
        verdict = REDUCES
    elif kl_none:
        verdict = NO_REDUCTION
    else:
        verdict = INCONCLUSIVE
    return {"verdict": verdict, "kl_ratio_ci95": [lo, hi], "kl_ci_finite": kl_finite,
            "kl_upper_ci_below_0.8": kl_reduces, "kl_lower_ci_above_0.9": kl_none,
            "regret_leg_by_family": reg, "regret_leg_all_families": reg_all}


def d23_block(O: dict, S: dict) -> dict:
    fams = [SHORT[f] for f in env.FAMILIES]
    diffs, leg = {}, {}
    for f in fams:
        fr = S["families"].get(f)
        if fr is None:
            diffs[f], leg[f] = None, {"missing": True}
            continue
        a = fr["arms"]
        pb = a["loc8_15"]["nregret_minus_sparse"]
        diffs[f] = pb["mean_diff"]
        head = fr["copy_floor"] - a["sparse"]["nregret"]["mean"]
        frc = a["loc8_15"].get("FR_changed") or {}
        leg[f] = {"mean_diff": pb["mean_diff"], "ci95": pb["ci95"], "n_parents_paired": pb["n"],
                  "nregret_sparse": a["sparse"]["nregret"]["mean"], "nregret_loc8_15": a["loc8_15"]["nregret"]["mean"],
                  "copy_floor": fr["copy_floor"], "headroom_copy_minus_sparse": head,
                  "low_headroom": bool(_finite(head) and head < LOW_HEADROOM),
                  "FR_changed_loc8_15": frc.get("FR"), "FR_changed_loc8_15_ci95": frc.get("ci95"),
                  "onestep_changed_loc8_15": a["loc8_15"]["onestep_changed"]["acc"],
                  "onestep_changed_sparse": a["sparse"]["onestep_changed"]["acc"],
                  "onestep_changed_supa": a["supa"]["onestep_changed"]["acc"]}
    out = {"rule": D23_RULE, "source": "DECISIONS.md D-23 (stated before any localized data existed); "
                                       "operationalised in docs/localized/DESIGN.md before the run",
           "thresholds": {"kl_ratio_upper_ci_lt": D23_KL_UPPER, "kl_ratio_lower_ci_gt": D23_KL_LOWER,
                          "regret_diff_le": D23_REGRET_TOL},
           "primary_kl": "KL(arm||supa)", "secondary_kl": "KL(supa||arm)"}
    for name, key in (("primary", D23_PRIMARY), ("secondary", D23_SECONDARY)):
        r = O["ratios"][key]
        out[name] = {"kl": r["label"], "kl_ratio": r["loc8_15"]["ratio"], **d23_rule(r["loc8_15"]["ci95"], diffs, fams),
                     "kl_ratio_flags": r["flags"]}
    out["verdict"] = out["primary"]["verdict"]
    out["directions_agree"] = out["primary"]["verdict"] == out["secondary"]["verdict"]
    out["regret_leg"] = leg
    cav = [f"{O['ratios'][k]['label']}: {x['level']}: {x['why']}" for k in (D23_PRIMARY, D23_SECONDARY) for x in O["ratios"][k]["flags"]]
    cav += [f"{f}: low headroom (copy floor {leg[f]['copy_floor']:.3f} − sparse regret {leg[f]['nregret_sparse']:.3f} = "
            f"{leg[f]['headroom_copy_minus_sparse']:.3f} < {LOW_HEADROOM:.2f}): the +{D23_REGRET_TOL} regret leg can barely fail; "
            f"read its one-step FR" for f in fams if leg[f].get("low_headroom")]
    cav += [f"{f}: no chains in this run, the regret leg fails" for f in fams if leg[f].get("missing")]
    out["caveats"] = cav
    return out


# ------------------------------------------------------------------------------------------ render
def f3(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def ci(c, nd=3):
    if not c or c[0] is None or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in c):
        return ""
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


def table(headers, rows):
    return T.table(headers, rows)


def render(res: dict) -> str:
    m, O, S = res["meta"], res["offtask"], res["skill"]
    L = ["# Block-localised Titan delta: off-task drift vs the transition skill (exploratory)", ""]
    L.append(f"Run `{m['run_dir']}` (DONE {m['done_finished']}); analysis `scripts/localized_analysis.py`; "
             f"{O['n_prompts']} off-task prompts, {m['n_chains']} pilot TEST chains (chain_one); {m['boot']} bootstrap "
             f"resamples. Exploratory: pilot bank, TEST seen in development, one greedy decode per prompt."
             + (" **SUBSET RUN (test mode): not the real pilot sets.**" if m["subset"] else ""))
    L += ["", "Arms (all built in place on one Supa load; build gates G1-G5 passed, see run_localized.py):", ""]
    rows = []
    for arm in ARMS:
        b = res["build"][arm]
        rp = b["repro"] or {}
        rows.append([arm, b["titan_blocks"] or "–", f"{b['differs_from_supa'][0]}..{b['differs_from_supa'][-1]}" if b["differs_from_supa"] else "none",
                     b["fingerprint"][:12], f"{rp.get('status')} ({rp.get('gate_equal')}/{rp.get('gate_n')}; batch {rp.get('batch_items_equal')}/{rp.get('batch_items_compared')})"
                     if rp.get("status") not in (None, "skipped") else f"{rp.get('status')}",
                     "yes" if O["state_build_check"][arm]["identical_where_expected"] else "**NO**"])
    L.append(table(["arm", "Titan blocks", "differs from Supa", "fingerprint", "run_patch text reproduction (first 4/family)",
                    "states == Supa below first Titan block"], rows))
    lc = O["state_build_check"]["loc8_15_vs_sparse"]
    L += ["", f"loc8_15 states equal sparse's exactly at layers {', '.join(lc['expected_identical_layers'])}: "
              f"{'yes' if lc['identical_where_expected'] else '**NO**'}.", ""]
    # D-23 verdict
    V = res["d23_verdict"]
    yn = lambda b: "yes" if b else "no"  # noqa: E731
    L += ["## Pre-stated reading (DECISIONS D-23)", "",
          f"Rule (verbatim from D-23): {V['rule']}", "",
          "Operationalisation, fixed in docs/localized/DESIGN.md before the run: KL ratio = mean KL(loc8_15‖supa) / "
          "mean KL(sparse‖supa) over the off-task prompts with the paired percentile bootstrap CI (primary direction; "
          "the same rule on KL(supa‖arm) is shown as secondary); regret = chain_one k=4 normalised regret, difference = "
          "paired per-parent mean loc8_15 − sparse; families L, P, R.", "",
          f"**Verdict (primary, KL(arm‖supa)): {V['primary']['verdict']}.** Secondary, KL(supa‖arm): "
          f"{V['secondary']['verdict']} ({'agrees' if V['directions_agree'] else '**differs**'}).", ""]
    rows = []
    for name in ("primary", "secondary"):
        p = V[name]
        rows.append([f"KL ratio loc8_15/sparse ({p['kl']}; {name})", f3(p["kl_ratio"]), ci(p["kl_ratio_ci95"]),
                     f"upper CI < {D23_KL_UPPER}: {yn(p['kl_upper_ci_below_0.8'])}; lower CI > {D23_KL_LOWER}: "
                     f"{yn(p['kl_lower_ci_above_0.9'])}"])
    for f, g in V["regret_leg"].items():
        if g.get("missing"):
            rows.append([f"{f}: regret loc8_15 − sparse", "–", "", "no chains: fails"])
            continue
        rows.append([f"{f}: k=4 regret loc8_15 − sparse (n {g['n_parents_paired']} parents)", f3(g["mean_diff"]), ci(g["ci95"]),
                     f"≤ +{D23_REGRET_TOL}: {yn(V['primary']['regret_leg_by_family'][f])}"])
    L.append(table(["leg", "value", "95% CI", "criterion met"], rows))
    L += ["", "Regret leg in context (the +0.05 leg is informative only where the copy chain is well above sparse's "
              f"regret; a family with headroom < {LOW_HEADROOM:.2f} is marked low-headroom and its skill evidence is the "
              "one-step FR on changed first steps):", ""]
    rows = []
    for f, g in V["regret_leg"].items():
        if g.get("missing"):
            continue
        rows.append([f, f3(g["nregret_sparse"]), f3(g["nregret_loc8_15"]), f3(g["copy_floor"]),
                     f3(g["headroom_copy_minus_sparse"]) + (" **low**" if g["low_headroom"] else ""),
                     f"{f3(g['FR_changed_loc8_15'], 2)} {ci(g['FR_changed_loc8_15_ci95'], 2)}",
                     f"{f3(g['onestep_changed_supa'])} / {f3(g['onestep_changed_sparse'])} / {f3(g['onestep_changed_loc8_15'])}"])
    L.append(table(["family", "regret sparse", "regret loc8_15", "copy chain", "headroom (copy − sparse)",
                    "FR loc8_15 (changed first steps) [CI]", "one-step acc supa / sparse / loc8_15 (changed)"], rows))
    if V["caveats"]:
        L += ["", "Caveats (reported, they do not change the pre-stated verdict):", ""] + [f"- {c}" for c in V["caveats"]]
    L.append("")
    # key ratios
    dn = O["denominator"]
    L += ["## Key ratios: drift of the localised arms relative to the full delta", "",
          "Paired bootstrap over prompts; a ratio < 1 whose CI excludes 1 = less drift than the full delta. "
          "The two ratios need not add to 1 (KL and disagreement are not additive over blocks). KL in nats.", "",
          f"Denominator (sparse vs Supa, {dn['n_prompts']} prompts): top-1 token flipped on {dn['sparse_top1_flips']} prompts; "
          f"greedy reply differs on {dn['sparse_greedy_disagreements']} ({f3(dn['sparse_greedy_disagree_rate'])}); "
          f"top 10% of prompts carry {f3(dn['sparse_kl_top10pct_share']['kl_arm_supa'], 2)} of mean KL(sparse‖supa) and "
          f"{f3(dn['sparse_kl_top10pct_share']['kl_supa_arm'], 2)} of mean KL(supa‖sparse).", ""]
    rows = []
    for key in ("kl_arm_supa", "kl_supa_arm", "greedy_dis", "greedy_prefix_div", "top1_dis"):
        r = O["ratios"][key]
        rows.append([r["label"], f3(r["loc8_15"]["den_mean"], 4)]
                    + [f"{f3(r[a]['ratio'])} {ci(r[a]['ci95'])}" + (f" ({r[a]['dropped_zero_denominator']} resamples dropped)" if r[a]["dropped_zero_denominator"] else "")
                       for a in CMP]
                    + [f"{f3(r[a]['diff'], 4)} {ci(r[a]['diff_ci95'], 4)}" for a in CMP]
                    + ["; ".join(f"**{x['level']}**: {x['why']}" for x in r["flags"]) or "ok"])
    L.append(table(["drift measure", "sparse (full delta)", "loc8_15 / sparse [95% CI]", "up16_27 / sparse [95% CI]",
                    "loc8_15 − sparse", "up16_27 − sparse", "denominator flag"], rows))
    L += ["", "## Skill: k=4 chain_one normalised regret (lower is better; decisions.g_only, pilot TEST goals)", ""]
    rows = []
    for f, fr in S["families"].items():
        a = fr["arms"]
        rows.append([f, fr["n_parents"]] + [f"{f3(a[arm]['nregret']['mean'])} {ci(a[arm]['nregret']['ci95'])}" for arm in ARMS]
                    + [f3(fr["copy_floor"])]
                    + [f"{f3(a[arm]['nregret_minus_sparse']['mean_diff'])} {ci(a[arm]['nregret_minus_sparse']['ci95'])}" for arm in CMP])
    L.append(table(["family", "parents"] + list(ARMS) + ["copy chain", "loc8_15 − sparse [CI]", "up16_27 − sparse [CI]"], rows))
    L += ["", "## Skill: one-step accuracy from the first chain step (= run_patch's one-frame items)", ""]
    rows = []
    for f, fr in S["families"].items():
        a = fr["arms"]
        rows.append([f, a["supa"]["onestep_changed"]["n"]]
                    + [f"{f3(a[arm]['onestep_changed']['acc'])} {ci(a[arm]['onestep_changed']['ci95'])}" for arm in ARMS]
                    + [f"{f3(a[arm]['FR_changed']['FR'], 2)} {ci(a[arm]['FR_changed']['ci95'], 2)}" for arm in CMP])
    L.append(table(["family", "changed steps"] + [f"{arm} acc (changed)" for arm in ARMS] + ["FR loc8_15", "FR up16_27"], rows))
    L += ["", "All first steps and chain exact-state accuracy:", ""]
    rows = []
    for f, fr in S["families"].items():
        for arm in ARMS:
            a = fr["arms"][arm]
            rows.append([f, arm, f"{f3(a['onestep_all']['acc'])} {ci(a['onestep_all']['ci95'])}",
                         f"{f3(a['onestep_all_minus_sparse']['diff'])} {ci(a['onestep_all_minus_sparse']['ci95'])}" if "onestep_all_minus_sparse" in a else "–"]
                        + [f3(a["chain_acc"][k]["acc"]) for k in ("1", "2", "4")] + [f3(a["chains_complete"]), a["steps_attempted"]])
    L.append(table(["family", "arm", "first step acc (all)", "− sparse", "k=1", "k=2", "k=4", "chains complete", "steps"], rows))
    # off-task detail
    L += ["", "## Off-task drift per arm (vs Supa)", ""]
    rows = []
    for arm in ARMS:
        k, g, jb = O["kl"][arm], O["greedy"][arm], O["json_bleed"][arm]
        rows.append([arm, f"{f3(k['arm_supa']['mean'], 4)} {ci(k['arm_supa']['mean_ci95'], 4)}",
                     f"{f3(k['arm_supa']['median'], 4)} {ci(k['arm_supa']['median_ci95'], 4)}",
                     f"{f3(k['supa_arm']['mean'], 4)} {ci(k['supa_arm']['mean_ci95'], 4)}",
                     f"{f3(k['supa_arm']['median'], 4)} {ci(k['supa_arm']['median_ci95'], 4)}",
                     f3(k["top1_agree"]), f"{f3(g['agree'])} {ci(g['agree_ci95'])}",
                     f"{f3(g['prefix_div_mean'])} {ci(g['prefix_div_ci95'])}",
                     f"{f3(jb['bleed_rate_nonnumeric'])} (n {jb['n_nonnumeric']})", f3(jb["json_state_rate_all"])])
    L.append(table(["arm", "KL(arm‖supa) mean, nats", "median", "KL(supa‖arm) mean, nats", "median", "top-1 agree",
                    "greedy exact agree", "greedy prefix divergence", "JSON bleed (non-numeric)", "JSON-state (all)"], rows))
    L += ["", "KL of each arm to the full delta, KL(arm‖sparse) mean, nats: "
          + ", ".join(f"{arm} {f3(O['kl'][arm]['arm_sparse']['mean'], 4)}" for arm in ARMS if "arm_sparse" in O["kl"][arm]) + "."]
    nf = O["noise_floor_stored_sparse"]
    if isinstance(nf, dict):
        L.append(f"Noise floor: KL(stored Titan-Sparse ‖ this run's sparse arm) mean {f3(nf['kl_stored_run']['mean'], 6)}, "
                 f"top-1 agreement {f3(nf['top1_agree'])}.")
    L += ["", "### Answer-slot state shift per layer (RMS over dims of (h_arm − h_supa)/sd_supa; mean over prompts)", ""]
    rows = []
    for Ls, r in O["state_shift"].items():
        sr = O["ratios"]["state_shift_rms_z"]["per_layer"].get(Ls, {})
        rows.append([Ls] + [f3(r[arm]["rms_z"]["mean"]) if r[arm]["rms_z"] else "raw only" for arm in ARMS]
                    + [f3(r[arm]["rel_l2_no_massive"]["mean"]) for arm in ARMS]
                    + [f"{f3(sr[a]['ratio'], 2)} {ci(sr[a]['ci95'], 2)}" if a in sr else "–" for a in CMP])
    L.append(table(["layer"] + [f"{a} RMS z" for a in ARMS] + [f"{a} rel L2" for a in ARMS]
                   + ["loc8_15/sparse", "up16_27/sparse"], rows))
    L += ["", "### Greedy exact agreement with Supa by reply class", ""]
    cls = list(O["greedy"]["supa"]["by_class"])
    cls.sort(key=lambda c: O["greedy"]["sparse"]["by_class"][c]["agree"])
    L.append(table(["reply class", "n"] + [a for a in ARMS if a != "supa"],
                   [[c, O["greedy"]["supa"]["by_class"][c]["n"]] + [f3(O["greedy"][a]["by_class"][c]["agree"], 2) for a in ARMS if a != "supa"]
                    for c in cls]))
    L += ["", "## Consistency checks (descriptive)", ""]
    for arm in ARMS:
        c = S["consistency"][arm]
        parts = []
        if "first_step_text_equals_run_patch" in c:
            x = c["first_step_text_equals_run_patch"]
            parts.append(f"first-step text == run_patch `{Path(x['reference']).name}` on {f3(x['rate'])} of {x['n']} items")
        if "stored_pilot_chains" in c:
            x = c["stored_pilot_chains"]
            parts.append(f"identical to the stored pilot chains: {f3(x['identical_chains'])} of {x['n']} chains, "
                         f"{f3(x['identical_step_texts'])} of step texts")
        L.append(f"- {arm}: " + ("; ".join(parts) if parts else "no reference available"))
    L += ["", "The stored pilot chains predate the 160-token cap and the D-14 generation batch budget (Supa's include "
              "truncated texts), so identical_chains < 1 is expected there, mostly for supa. It is not a splice check: the "
              "build gates G1-G5 and the run_patch text reproduction are.",
          "", "## Reading guide and limits", "",
          f"- Pre-stated reading (DECISIONS D-23, verbatim): {D23_RULE} The KL ratio is mean KL(arm‖supa) (primary; "
          "fixed in docs/localized/DESIGN.md before the run), the regret difference the paired per-parent mean of chain_one "
          "k=4 normalised regret. The verdict above applies this rule mechanically; the denominator flags and "
          "low-headroom marks are caveats, not part of the rule.",
          "- up16_27 is descriptive: whether the upper delta (blocks 16-27, no measurable skill in R12) carries drift on "
          "its own. Both localised arms are measured against Supa, so loc8_15's remaining drift is the price of the "
          "skill-carrying blocks.",
          "- Off-task drift here is first-token and 64-token greedy behaviour on 248 neutral prompts plus answer-slot "
          "states; it is not a retention benchmark. One greedy decode per prompt; replies compared by exact text.",
          "- Skill is chain_one on the pilot TEST plans (one-step frame, self-rollout) and its first step; P's primary "
          "chain frame in T2 was chain_m1, which this run does not generate.",
          "- Pilot TEST was seen in development; exploratory: the D-23 thresholds were stated in advance but are not a "
          "preregistration. Bootstrap CIs resample prompts (off-task) or parents (skill) and ignore the selection of the "
          "8/16 cut, which R12 fixed beforehand."]
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------------------------------- main
def analyse(run_dir: Path, patch_dir: Path, stored_root: Path, allow_subset: bool, B: int, seed: int) -> dict:
    R = Run(run_dir, allow_subset)
    build = {}
    for arm in ARMS:
        r = R.arm_rec[arm]
        build[arm] = {"fingerprint": r["fingerprint"], "titan_blocks": r["titan_blocks"], "differs_from_supa": r["differs_from_supa"],
                      "fingerprint_expected": r.get("fingerprint_expected"),
                      "repro": {k: v for k, v in (r.get("repro") or {}).items() if k != "differ_texts"}}
    res = {"meta": {"run_dir": str(run_dir), "done_finished": R.done.get("finished"), "n_chains": len(R.cands),
                    "n_prompts": len(R.ids), "subset": R.subset, "boot": B, "seed": seed,
                    "inputs_sha256": {f"{arm}/{f}": h for arm in ARMS for f, h in R.done["arms"][arm]["files_sha256"].items()}},
           "build": build}
    res["offtask"] = offtask_block(R, B, seed, stored_root)
    res["skill"] = skill_block(R, B, seed, patch_dir, stored_root)
    res["d23_verdict"] = d23_block(res["offtask"], res["skill"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run-dir", default=str(ROOT / "runs" / "bank_pilot" / "localized"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "localized" / "LOCALIZED_RESULTS.md"))
    ap.add_argument("--patch-dir", default=str(ROOT / "runs" / "bank_pilot" / "patch"))
    ap.add_argument("--stored-root", default=str(ROOT / "runs" / "bank_pilot"), help="stored pilot arms (supa/, sparse/)")
    ap.add_argument("--allow-subset", action="store_true", help="tiny test only: prompt/chain subsets")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    res = analyse(Path(a.run_dir), Path(a.patch_dir), Path(a.stored_root), a.allow_subset, a.boot, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(T.clean(res), indent=1) + "\n", encoding="utf-8")
    r = res["offtask"]["ratios"]
    V = res["d23_verdict"]
    print(f"[localized_analysis] wrote {out} and {out.with_suffix('.json').name}")
    print(f"  D-23 verdict (primary, KL(arm||supa)): {V['verdict']}; secondary KL(supa||arm): {V['secondary']['verdict']}")
    for key in ("kl_arm_supa", "kl_supa_arm", "greedy_dis", "greedy_prefix_div"):
        print(f"  {r[key]['label']}: " + "; ".join(f"{arm}/sparse {f3(r[key][arm]['ratio'])} {ci(r[key][arm]['ci95'])}" for arm in CMP)
              + ("" if r[key]["interpretable"] else "  [UNINTERPRETABLE]"))
    for f, fr in res["skill"]["families"].items():
        print(f"  {f} k=4 nregret: " + ", ".join(f"{arm} {f3(fr['arms'][arm]['nregret']['mean'])}" for arm in ARMS)
              + f"; loc8_15 - sparse {f3(fr['arms']['loc8_15']['nregret_minus_sparse']['mean_diff'])} "
              f"{ci(fr['arms']['loc8_15']['nregret_minus_sparse']['ci95'])}")


if __name__ == "__main__":
    main()
