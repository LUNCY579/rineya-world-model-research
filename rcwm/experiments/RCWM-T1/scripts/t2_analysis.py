"""RCWM-T2 analysis (DECISIONS D-19). CPU only, numpy/json; never loads a model.

  python scripts/t2_analysis.py --tags t2_s0[,t2_s1,t2_s2] [--runs-root runs/t2] [--out docs/t2_analysis/T2_RESULTS.md]

Reads, per tag, the outputs of `scripts/run_t2.py eval` in <runs-root>/<tag>/ and writes a markdown
report plus a JSON with the same stem. Every comparison is adapter ON minus the same-session adapter
OFF (= Titan-Sparse in the same process). Deterministic: fixed bootstrap seeds, 10,000 resamples, no
timestamps in the outputs.

Blocks (per seed)
  A  one-step per-op, trained frame: t2_eval rows joined with t2_eval_substrata by index, plus
     t2_eval_supp. Per D-19 cell: copy floor, OFF, ON, delta, discordant b (OFF right, ON wrong) /
     c (OFF wrong, ON right), exact McNemar p, Wilson 95% CI for ON, Newcombe (method 10) 95% CI for
     the paired delta. Minimal pairs scored at pair level (both members correct) and per member.
     False-copy rate P(output == before | target != before) and parse-failure rate per family.
  B  pilot oracle-anchored transitions (DEV+TEST, unique payloads): family x stratum x step
     (t=0 vs t=1-3) x magnitude (ledger: every |balance| <= 99 vs any > 99, before or after);
     OFF, ON, delta with a parent-cluster bootstrap CI (parent = cand_id without '-c<k>').
  C  cross-frame re-render of t2_eval (xframe): the same cells, all TRANSFER; P rows are in the
     one-step frame there, whose prompt presumes applicability -> P inapplicable cells (and the P
     inapplicable stratum, and 'P all', which mixes them in) are descriptive only; the P transfer
     summary is 'P applicable' (app-changed + app-noop rows).
  D  chains on pilot TEST: exact-state accuracy at k=1,2,4 per family for chain_one and chain_m1,
     OFF/ON/delta with a parent-cluster bootstrap CI, copy-chain floor, predicted k=4 accuracy
     (product of per-step accuracies from B on TEST; and the gold-path product), P chain steps split
     into state-changing applicable / unchanged applicable / inapplicable on the chain's own
     correct-input steps (RESULTS_PILOT R8), and reproducibility of the same-session OFF chains
     against the stored Titan-Sparse pilot chains (runs/bank_pilot/sparse).
  E  decisions: G_chain k=4 normalised regret via rcwmt.decisions.g_only (pilot TEST, its goals),
     OFF vs ON per family and mode, paired per-parent bootstrap (rcwmt.decide.paired_bootstrap).
  F  forgetting: first-token KL(OFF||ON) on the 248 off-task prompts, noise floor
     KL(stored Sparse||same-session OFF), reference KL(Supa||Sparse) (pending until
     runs/bank_pilot/supa/offtask_logprobs.npy exists), top-1 agreement, 64-token greedy reply exact
     agreement overall and per reply_class, JSON-bleed rate.
Gates (D-19, per seed): G1 targeted >= 0.80 ON, G2 untouched drop <= 0.05, G3 P app-del delta lower
CI > 0, G4 false-copy does not rise, G5 pilot direction at steps 2-4 and |balance| > 99, G6 chains at
k=4, G7 decision regret drop with CI excluding 0, G8 forgetting KL <= KL(Supa||Sparse). "Rule learned"
= all eight PASS (any FAIL -> FAIL, otherwise PENDING). Across seeds: per-seed gate table and sign
agreement of every key delta (a direction counts only if all three seeds agree).

Operational choices where D-19 leaves room (fixed before any T2 output exists)
  * Targeted cells (G1, ON >= 0.80 each): L void-disc; P app-changed add-only; P app-del = t2_eval
    app-changed rows whose delete clears a set bit + supp app-del (pooled, n=50); P inapp-disc;
    P pre_off-only; R mix wraps=0, wraps=1, wraps>=2 (t2_eval wraps=2 + supp mix-2wrap); R conditional
    without the value=0 mod 97 (branch-free) rows; R add/wrap; the three minimal-pair sets (both correct).
    Parts of pooled cells are shown as 'detail'; copy- or rule-ignoring-passable rows as 'descriptive'.
  * Untouched (G2): L posted (all three kinds pooled, 120 rows, as D-19 names one 'L posted' cell),
    R swap, R add/nowrap; point estimates. The per-kind L posted deltas are printed in the G2 detail.
  * G3 uses the Newcombe CI of the paired delta (ON - OFF) on the pooled P app-del cell.
  * G4 "does not rise" = point estimate ON <= OFF in every family (literal D-19 reading, kept on
    purpose: conservative for the claim), over t2_eval + supp rows with target != before; an
    unparseable answer is not a copy. One extra copied row in any family fails G4; the report prints
    the counts, the spare margin and the exact McNemar p next to the status so a one-row flip is visible.
  * G5 "same direction" = the per-op gains carry over: every RULE stratum of t2_eval (strata holding a
    D-19 targeted rule: L add/void, scale/void, transfer/void; P applicable, inapplicable; R add/wrap,
    mix/nowrap, mix/wrap, conditional/gt, conditional/le) whose per-op change is a significant GAIN
    (exact McNemar p < 0.05, delta > 0) must also gain (pilot delta > 0; a zero delta fails) on pilot
    transitions at steps 2-4 (t >= 1) and, for the ledger, at |balance| > 99. Pilot cells without rows,
    or with OFF = ON = 1.0 (no room), are skipped. No rule stratum with a significant gain, or no
    checkable pilot cell, = FAIL. Untouched strata (L */posted, R swap, R add/nowrap) and significant
    per-op DROPS are not gated by G5 (G1/G2 cover them); their pilot direction is reported descriptively.
  * G6/G7 use the primary chain mode per family (chain_one L/R, chain_m1 P); G7 drop = OFF - ON with the
    decide.paired_bootstrap CI lower bound > 0.
  * JSON bleed: replies on prompts outside NUMERIC_CLASSES that are a JSON state/object or a bare integer.
  * KL is computed from the stored fp16 log-probs after renormalisation (log-softmax in float64).
  * Pilot block B prints a row-level McNemar p for reference only; it ignores the clustering of pilot
    rows within parents. Inference in B uses the parent-cluster bootstrap CI.

Input validation (any failure stops the run with SystemExit; no gate is computed on bad inputs)
  * static data: sha256 prefixes of D-16/D-19; t2_eval_substrata row i has idx == i and the family and
    stratum of t2_eval row i; t2_eval_xframe row i has the family/stratum/before/event/target of
    t2_eval row i; every t2_eval stratum is either a rule stratum or an untouched stratum; the stored
    Titan-Sparse pilot chains and off-task meta pass the checks below.
  * gen_<set>.jsonl: same length and order as the reference rows (all row fields equal, 'i' == index),
    base_text and t2_text present and strings.
  * eval.json, if present (run_t2 writes it last, so the run is complete): per-op summaries equal a
    recomputation from the gen files, n equals the row counts, provenance data sha256 equal the current
    data files, and every gen/chain/off-task file exists.
  * chain files: exactly the pilot TEST cand_ids, no duplicates; per row len(states) == len(events)+1,
    states[0] == s0, states are strings until the first null and null after it, texts are strings,
    len(texts) == number of attempted steps, and each text parses to the state stored for it.
  * off-task: metas list the 248 prompt ids/classes of data/offtask_prompts.jsonl in order, log-prob
    arrays have one row per prompt and the stored vocabulary size, offtask_gen.jsonl has one row per
    prompt in the same order with string replies.
  Missing files of a run without eval.json (in progress or crashed) only make the dependent gates PENDING.

Claim scope (printed in the report): D-19 limits; a 'rule learned' PASS does not separate rule learning
from frame adaptation until the D-19 self-distillation control has run; the sealed confirmation bank is
not used; with fewer than three seeds every result is provisional.

rcwmt.decisions/fidelity import rcwmt.heads, which imports torch at module level. g_only and
fidelity._gen_state never use heads, so a placeholder module is installed before the import and torch
is never loaded (CUDA_VISIBLE_DEVICES="" is set anyway, DECISIONS I-21).
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
import re  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _stub_heads():
    if "rcwmt.heads" not in sys.modules:
        m = types.ModuleType("rcwmt.heads")
        m.__doc__ = "placeholder installed by t2_analysis.py: g_only/_gen_state do not use heads; keeps torch out"
        sys.modules["rcwmt.heads"] = m


_stub_heads()
from rcwmt import data, decide, decisions, env  # noqa: E402

assert "torch" not in sys.modules, "torch must not be imported by the T2 analysis"

CODE = {"ledger_replay": "L", "prerequisite_plan": "P", "register_machine": "R"}
FULL = {v: k for k, v in CODE.items()}
FAMS = ("L", "P", "R")
BOOT = 10000
SEED = 20261009
Z95 = 1.959963984540054
EPS = 1e-12
PRIMARY_CHAIN = {"L": "chain_one", "R": "chain_one", "P": "chain_m1"}
MODES = ("chain_one", "chain_m1")
TH_TARGETED = 0.80
TH_UNTOUCHED_DROP = 0.05
TH_CHAIN_K4 = {"L": 0.25, "R": 0.25, "P": 0.41}
TH_REGRET_DROP = {"L": 0.15, "P": 0.15, "R": 0.08}
SETS = ("t2_eval", "xframe", "supp", "pilot")
DATA_FILES = {"t2_eval": "t2_eval.jsonl", "xframe": "t2_eval_xframe.jsonl", "supp": "t2_eval_supp.jsonl"}
EXPECTED_SHA = {"t2_train": "bf5a2276", "t2_eval": "78b5c022", "t2_eval_xframe": "ddc0d798",
                "t2_eval_supp": "2c1efbc2", "t2_eval_substrata": "240ffddc", "bank_pilot": "7c9b432c"}
STORED_SPARSE = ROOT / "runs" / "bank_pilot" / "sparse"
STORED_SUPA = ROOT / "runs" / "bank_pilot" / "supa"
OFFTASK_PROMPTS = ROOT / "data" / "offtask_prompts.jsonl"
# t2_eval strata holding a D-19 targeted rule (G5 checks only these) vs the untouched strata (G2).
RULE_STRATA = {"L": ("add/void", "scale/void", "transfer/void"), "P": ("applicable", "inapplicable"),
               "R": ("add/wrap", "mix/nowrap", "mix/wrap", "conditional/gt", "conditional/le")}
UNTOUCHED_STRATA = {"L": ("add/posted", "scale/posted", "transfer/posted"), "P": (), "R": ("swap", "add/nowrap")}
XF_DESC = "descriptive-only (one-step frame presumes applicability)"
CHAIN_FILES = tuple(f"gen_{m}{s}.jsonl" for m in ("chain_one", "chain_m1") for s in ("_base", ""))
OFFTASK_FILES = ("offtask_logprobs.npy", "offtask_meta.json", "off/offtask_logprobs.npy", "off/offtask_meta.json",
                 "offtask_gen.jsonl")
# Off-task reply classes whose reference answer may legitimately be a bare number.
NUMERIC_CLASSES = frozenset({"arithmetic_chain", "arithmetic_short_answer", "counting_sets", "sequence_pattern",
                             "unit_conversion", "percentage_reasoning", "ratio_proportion", "estimation_sanity",
                             "time_duration", "output_only", "context_recall"})
GATE_NAMES = {
    "G1_targeted": "Per-op targeted cells: ON >= 0.80 in every cell",
    "G2_untouched": "Untouched cells (L posted, R swap, R add/nowrap): ON - OFF >= -0.05",
    "G3_app_del": "P app-del (pooled t2_eval+supp): Newcombe 95% CI of ON - OFF has lower bound > 0",
    "G4_false_copy": "False-copy rate (t2_eval+supp) does not rise: ON <= OFF in every family",
    "G5_pilot_direction": "Pilot transitions: every rule stratum with a significant per-op gain (McNemar p<0.05) "
                          "also gains at chain steps 2-4 and (ledger) at |balance|>99",
    "G6_chains_k4": "Chains k=4 ON: L/R chain_one >= 0.25, P chain_m1 >= 0.41",
    "G7_decision": "G_chain k=4 normalised regret drop (OFF - ON): >= 0.15 L/P, >= 0.08 R, paired CI excludes 0",
    "G8_forgetting": "Mean first-token KL(OFF||ON) <= mean KL(Supa||Sparse)",
}
IMPROVEMENT_GATES = ("G1_targeted", "G3_app_del", "G5_pilot_direction", "G6_chains_k4", "G7_decision")
NONREGRESSION_GATES = ("G2_untouched", "G4_false_copy", "G8_forgetting")


# ---------------------------------------------------------------- io helpers


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def die(what: str, errs: list[str]):
    """Stop the analysis: an input does not match what the instrument should have written."""
    more = f"\n  ... and {len(errs) - 12} more" if len(errs) > 12 else ""
    raise SystemExit(f"t2_analysis: {what} ({len(errs)} problem(s)); no result is written:\n  " + "\n  ".join(errs[:12]) + more)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return None if math.isnan(x) else x
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    return o


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = Z95):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p: binomial(b+c, 1/2), doubled smaller tail, capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def newcombe_paired(on: np.ndarray, off: np.ndarray, z: float = Z95):
    """Newcombe (1998) method 10 CI for p_on - p_off from paired binary outcomes."""
    on, off = np.asarray(on, bool), np.asarray(off, bool)
    n = len(on)
    if n == 0:
        return (float("nan"), float("nan"))
    e = int((on & off).sum()); f = int((on & ~off).sum()); g = int((~on & off).sum()); h = n - e - f - g
    p1, p2 = (e + f) / n, (e + g) / n
    d = p1 - p2
    l1, u1 = wilson(e + f, n, z)
    l2, u2 = wilson(e + g, n, z)
    den = (e + f) * (g + h) * (e + g) * (f + h)
    phi = (e * h - f * g) / math.sqrt(den) if den > 0 else 0.0
    dl = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (max(-1.0, d - dl), min(1.0, d + du))


def cluster_boot(x: np.ndarray, clusters: list, B: int = BOOT, seed: int = SEED):
    """Percentile 95% CI of the pooled mean of x when whole clusters are resampled with replacement."""
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    _, inv = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    S = np.bincount(inv, weights=x)
    N = np.bincount(inv).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), (B, len(S)))
    stat = S[idx].sum(1) / N[idx].sum(1)
    return (float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975)))


def sign(x, tol=EPS):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    return 1 if x > tol else (-1 if x < -tol else 0)


# ---------------------------------------------------------------- scoring


def score(fam_full: str, text, target: str, before):
    """(parsed, correct, copy): lenient parse (D-12), exact env.dumps match."""
    st = env.parse_state(fam_full, text, lenient=True) if isinstance(text, str) else None
    if st is None:
        return False, False, False
    s = env.dumps(st)
    return True, s == target, s == env.dumps(before)


def t2_stratum(code, before, ev):
    """Replica of scripts/make_t2_data.stratum (that module imports torch via rcwmt.runtime)."""
    if code == "L":
        return f"{ev['kind']}/{'posted' if ev['status'] == 'posted' else 'void'}"
    if code == "R":
        k = ev["kind"]
        if k == "swap":
            return "swap"
        if k == "conditional":
            return "conditional/" + ("gt" if before[ev["other"]] > ev["threshold"] else "le")
        raw = before[ev["register"]] + (ev["value"] if k == "add" else 2 * before[ev["other"]] + ev["value"])
        return f"{k}/" + ("wrap" if not 0 <= raw < env.MOD else "nowrap")
    return "applicable" if env.applicable(before, ev) else "inapplicable"


def pilot_reference(bank) -> list[dict]:
    """The pilot item list of run_t2.pilot_items without prompts (DEV+TEST, unique per (split, payload))."""
    seen, items = set(), []
    for p in bank["parents"]:
        if p["split"] not in ("dev", "test"):
            continue
        fam = p["family"]
        for c in p["candidates"]:
            st = env.rollout(fam, p["s0"], c["events"])
            for t, ev in enumerate(c["events"]):
                key = (p["split"], env.dumps({"before": st[t], "event": ev}))
                if key in seen:
                    continue
                seen.add(key)
                items.append({"target": env.dumps(env.step(fam, st[t], ev)), "family": fam,
                              "stratum": t2_stratum(CODE[fam], st[t], ev),
                              "frame": "m1" if fam == "prerequisite_plan" else "one", "before": st[t], "event": ev,
                              "split": p["split"], "step": t, "cand_id": c["cand_id"]})
    return items


def per_op_summary(rows, texts) -> dict:
    """Replica of run_t2.per_op (eval.json consistency check)."""
    res = {}
    for r, t in zip(rows, texts):
        st = env.parse_state(r["family"], t, lenient=True) if isinstance(t, str) else None
        k = f"{r['family'][:6]}|{r['stratum']}"
        n, c = res.get(k, (0, 0))
        res[k] = (n + 1, c + int(st is not None and env.dumps(st) == r["target"]))
    return {k: {"n": n, "acc": c / n} for k, (n, c) in sorted(res.items())}


def make_units(rows: list[dict], set_name: str, subs: list[str] | None = None) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        fam = CODE[r["family"]]
        po, co, cpo = score(r["family"], r.get("base_text"), r["target"], r["before"])
        pn, cn, cpn = score(r["family"], r.get("t2_text"), r["target"], r["before"])
        u = {"set": set_name, "i": i, "fam": fam, "stratum": r["stratum"], "sub": subs[i] if subs else None,
             "off": co, "on": cn, "off_parsed": po, "on_parsed": pn, "off_copy": cpo, "on_copy": cpn,
             "floor": env.dumps(r["before"]) == r["target"], "pair_id": r.get("pair_id"), "pair_role": r.get("pair_role")}
        if fam == "P":
            ev, b = r["event"], r["before"]
            u["app_del"] = bool(env.applicable(b, ev) and (b & ev["delete"] & ~ev["add"]))
        if set_name == "pilot":
            u["split"], u["step"] = r["split"], r["step"]
            u["parent"] = r["cand_id"].rsplit("-c", 1)[0]
            u["payload"] = env.dumps({"before": r["before"], "event": r["event"]})
            if fam == "L":
                tgt = json.loads(r["target"])
                u["big"] = max(abs(v) for v in list(r["before"].values()) + list(tgt.values())) > 99
            else:
                u["big"] = False
        out.append(u)
    return out


def pair_units(units: list[dict]) -> list[dict]:
    by = defaultdict(dict)
    for u in units:
        if u["pair_id"]:
            by[u["pair_id"]][u["pair_role"]] = u
    out = []
    for pid in sorted(by):
        a, b = by[pid].get("a"), by[pid].get("b")
        assert a is not None and b is not None, f"incomplete pair {pid}"
        out.append({"set": "supp_pair", "fam": a["fam"], "pair_id": pid, "on": a["on"] and b["on"],
                    "off": a["off"] and b["off"], "floor": a["floor"] and b["floor"]})
    return out


def summarize(units: list[dict], cid: str, label: str, role: str) -> dict:
    n = len(units)
    out = {"id": cid, "label": label, "role": role, "n": n}
    if n == 0:
        return out
    on = np.array([u["on"] for u in units], bool)
    off = np.array([u["off"] for u in units], bool)
    fl = np.array([u["floor"] for u in units], bool)
    b, c = int((off & ~on).sum()), int((~off & on).sum())
    out.update(copy_floor=float(fl.mean()), off=float(off.mean()), on=float(on.mean()),
               delta=float(on.mean() - off.mean()), b=b, c=c, p_mcnemar=mcnemar_exact(b, c),
               on_ci95=list(wilson(int(on.sum()), n)), delta_ci95=list(newcombe_paired(on, off)))
    return out


# ---------------------------------------------------------------- block A / C cells


def _t2(u, base):
    return u["set"] == base


def _wraps(u):
    s = u["sub"] or ""
    return int(s.split("=")[1]) if s.startswith("mix/wraps=") else None


def cell_specs():
    """(id, label, role, predicate(u, base)). base = 't2_eval' (A) or 'xframe' (C). Pair cells use '__pairs__'."""
    S = lambda st: (lambda u, base: u["set"] == "supp" and u["stratum"] == st)  # noqa: E731
    return [
        ("L.all", "L all t2_eval rows", "descriptive", lambda u, base: _t2(u, base) and u["fam"] == "L"),
        ("L.void_disc", "L void-disc (applying would change the state)", "targeted",
         lambda u, base: _t2(u, base) and u["fam"] == "L" and u["sub"] == "void-disc"),
        ("L.void_nondisc", "L void-nondisc (apply passes)", "descriptive",
         lambda u, base: _t2(u, base) and u["fam"] == "L" and u["sub"] == "void-nondisc(apply passes)"),
        ("L.posted", "L posted (add+scale+transfer)", "untouched",
         lambda u, base: _t2(u, base) and u["fam"] == "L" and u["stratum"].endswith("/posted")),
        ("L.add_posted", "  L add/posted", "detail", lambda u, base: _t2(u, base) and u["fam"] == "L" and u["stratum"] == "add/posted"),
        ("L.scale_posted", "  L scale/posted", "detail", lambda u, base: _t2(u, base) and u["fam"] == "L" and u["stratum"] == "scale/posted"),
        ("L.transfer_posted", "  L transfer/posted", "detail", lambda u, base: _t2(u, base) and u["fam"] == "L" and u["stratum"] == "transfer/posted"),
        ("L.pairs", "L minimal pairs posted/void (both correct)", "targeted", "__pairs__"),
        ("L.pair_posted", "  L pair member: posted", "detail", S("pair-posted")),
        ("L.pair_void", "  L pair member: void", "detail", S("pair-void")),
        ("P.all", "P all t2_eval rows", "descriptive", lambda u, base: _t2(u, base) and u["fam"] == "P"),
        ("P.applicable", "P applicable t2_eval rows (app-changed + app-noop)", "descriptive",
         lambda u, base: _t2(u, base) and u["fam"] == "P" and u["stratum"] == "applicable"),
        ("P.app_changed", "P app-changed (add-only + app-del)", "detail",
         lambda u, base: _t2(u, base) and u["fam"] == "P" and u["sub"] == "app-changed"),
        ("P.app_addonly", "P app-changed, add-only", "targeted",
         lambda u, base: _t2(u, base) and u["fam"] == "P" and u["sub"] == "app-changed" and not u["app_del"]),
        ("P.app_del", "P app-del (t2_eval app-changed with a deletion + supp app-del)", "targeted",
         lambda u, base: (_t2(u, base) and u["fam"] == "P" and u["sub"] == "app-changed" and u["app_del"])
         or (u["set"] == "supp" and u["stratum"] == "app-del")),
        ("P.app_del_t2eval", "  P app-del, t2_eval part", "detail",
         lambda u, base: _t2(u, base) and u["fam"] == "P" and u["sub"] == "app-changed" and u["app_del"]),
        ("P.app_del_supp", "  P app-del, supp part", "detail", S("app-del")),
        ("P.app_noop", "P applicable no-op (copy passes)", "descriptive",
         lambda u, base: _t2(u, base) and u["fam"] == "P" and u["sub"] == "app-noop(copy passes)"),
        ("P.inapp_disc", "P inapp-disc (applying would change the state)", "targeted",
         lambda u, base: _t2(u, base) and u["fam"] == "P" and u["sub"] == "inapp-disc"),
        ("P.inapp_nondisc", "P inapp-nondisc (apply passes)", "descriptive",
         lambda u, base: _t2(u, base) and u["fam"] == "P" and u["sub"] == "inapp-nondisc(apply passes)"),
        ("P.pre_off_only", "P pre_off-only (supp)", "targeted", S("pre_off-only")),
        ("P.pairs", "P minimal pairs applicable/inapp-1bit (both correct)", "targeted", "__pairs__"),
        ("P.pair_applicable", "  P pair member: applicable", "detail", S("pair-applicable")),
        ("P.pair_inapp1bit", "  P pair member: one extra unmet pre_on bit", "detail", S("pair-inapp1bit")),
        ("R.all", "R all t2_eval rows", "descriptive", lambda u, base: _t2(u, base) and u["fam"] == "R"),
        ("R.mix_w0", "R mix, wraps=0", "targeted", lambda u, base: _t2(u, base) and u["fam"] == "R" and _wraps(u) == 0),
        ("R.mix_w1", "R mix, wraps=1", "targeted", lambda u, base: _t2(u, base) and u["fam"] == "R" and _wraps(u) == 1),
        ("R.mix_w2", "R mix, wraps>=2 (t2_eval wraps=2 + supp mix-2wrap)", "targeted",
         lambda u, base: (_t2(u, base) and u["fam"] == "R" and (_wraps(u) or 0) >= 2)
         or (u["set"] == "supp" and u["stratum"] == "mix-2wrap")),
        ("R.mix_w2_t2eval", "  R mix wraps>=2, t2_eval part", "detail",
         lambda u, base: _t2(u, base) and u["fam"] == "R" and (_wraps(u) or 0) >= 2),
        ("R.mix_w2_supp", "  R mix wraps>=2, supp part", "detail", S("mix-2wrap")),
        ("R.cond", "R conditional (branch-dependent rows)", "targeted",
         lambda u, base: _t2(u, base) and u["fam"] == "R" and u["stratum"].startswith("conditional/")
         and not (u["sub"] or "").endswith("(branch-free)")),
        ("R.cond_gt", "  R conditional/gt", "detail", lambda u, base: _t2(u, base) and u["fam"] == "R" and u["sub"] == "conditional/gt"),
        ("R.cond_le", "  R conditional/le", "detail", lambda u, base: _t2(u, base) and u["fam"] == "R" and u["sub"] == "conditional/le"),
        ("R.cond_v0", "R conditional, value = 0 mod 97 (branch-free)", "descriptive",
         lambda u, base: _t2(u, base) and u["fam"] == "R" and (u["sub"] or "").endswith("(branch-free)")),
        ("R.add_wrap", "R add/wrap", "targeted", lambda u, base: _t2(u, base) and u["fam"] == "R" and u["stratum"] == "add/wrap"),
        ("R.add_nowrap", "R add/nowrap", "untouched", lambda u, base: _t2(u, base) and u["fam"] == "R" and u["stratum"] == "add/nowrap"),
        ("R.swap", "R swap", "untouched", lambda u, base: _t2(u, base) and u["fam"] == "R" and u["stratum"] == "swap"),
        ("R.pairs", "R conditional boundary pairs thr=other-1 / thr=other (both correct)", "targeted", "__pairs__"),
        ("R.pair_gt", "  R pair member: threshold = other-1 (gt)", "detail", S("pair-gt-boundary")),
        ("R.pair_le", "  R pair member: threshold = other (le)", "detail", S("pair-le-boundary")),
    ]


def false_copy_table(units: list[dict]) -> dict:
    out = {}
    for f in FAMS:
        us = [u for u in units if u["fam"] == f]
        dn = [u for u in us if not u["floor"]]
        res = {"n_rows": len(us), "n_target_ne_before": len(dn)}
        if dn:
            on = np.array([u["on_copy"] for u in dn], bool)
            off = np.array([u["off_copy"] for u in dn], bool)
            b, c = int((on & ~off).sum()), int((~on & off).sum())   # b: ON newly copies, c: ON stops copying
            res.update(off=float(off.mean()), on=float(on.mean()), delta=float(on.mean() - off.mean()),
                       off_copies=int(off.sum()), on_copies=int(on.sum()),
                       on_new_copies=b, on_stopped_copies=c, p_mcnemar=mcnemar_exact(b, c),
                       delta_ci95=list(newcombe_paired(on, off)))
        if us:
            res["off_parse_fail"] = float(np.mean([not u["off_parsed"] for u in us]))
            res["on_parse_fail"] = float(np.mean([not u["on_parsed"] for u in us]))
        out[f] = res
    return out


def onestep_block(units: list[dict], base: str) -> dict:
    pairs = pair_units([u for u in units if u["set"] == "supp"]) if base == "t2_eval" else []
    cells = []
    for cid, label, role, pred in cell_specs():
        if pred == "__pairs__":
            us = [p for p in pairs if p["fam"] == cid[0]]
        else:
            us = [u for u in units if pred(u, base)]
        if base == "xframe":
            if not us:
                continue
            if cid in ("P.inapp_disc", "P.inapp_nondisc"):
                role = XF_DESC
            elif cid == "P.all":
                role = "descriptive (mixes in the descriptive-only inapplicable rows; see P applicable)"
            else:
                role = "TRANSFER"
            if cid in ("P.app_del", "R.mix_w2"):
                label += " [xframe: t2_eval rows only; the supplement has no cross-frame version]"
        cells.append(summarize(us, cid, label, role))
    fc_units = [u for u in units if u["set"] in (base, "supp")] if base == "t2_eval" else [u for u in units if u["set"] == base]
    by_stratum = {}
    for f in FAMS:
        for st in sorted({u["stratum"] for u in units if u["set"] == base and u["fam"] == f}):
            if base == "t2_eval":
                role = "rule stratum" if st in RULE_STRATA[f] else "untouched stratum"
            else:
                role = XF_DESC if (f, st) == ("P", "inapplicable") else "TRANSFER"
            by_stratum[f"{f}|{st}"] = summarize([u for u in units if u["set"] == base and u["fam"] == f and u["stratum"] == st],
                                                f"{f}|{st}", f"{f} {st}", role)
    return {"cells": cells, "false_copy": false_copy_table(fc_units), "by_stratum": by_stratum}


# ---------------------------------------------------------------- block B


def pilot_block(units: list[dict]) -> dict:
    rows = []

    def add(us, key, label):
        s = summarize(us, key, label, "pilot")
        if us:
            s["n_clusters"] = len({u["parent"] for u in us})
            s["delta_ci95_cluster"] = list(cluster_boot(np.array([float(u["on"]) - float(u["off"]) for u in us]),
                                                        [u["parent"] for u in us]))
            s.pop("delta_ci95", None)
        rows.append(s)

    steps = (("t=0", lambda u: u["step"] == 0), ("t=1-3", lambda u: u["step"] >= 1))
    mags = {"L": (("|bal|<=99", lambda u: not u["big"]), (">99", lambda u: u["big"])), "P": (("all", lambda u: True),),
            "R": (("all", lambda u: True),)}
    for f in FAMS:
        fu = [u for u in units if u["fam"] == f]
        for st in sorted({u["stratum"] for u in fu}):
            for sn, sp in steps:
                for mn, mp in mags[f]:
                    us = [u for u in fu if u["stratum"] == st and sp(u) and mp(u)]
                    if us:
                        add(us, f"{f}|{st}|{sn}|{mn}", f"{f} {st} {sn} {mn}")
    pooled = []
    for f in FAMS:
        fu = [u for u in units if u["fam"] == f]
        for sn, sp in steps:
            pooled.append((fu, f, sn, sp))
    rows_pooled = []
    for fu, f, sn, sp in pooled:
        us = [u for u in fu if sp(u)]
        s = summarize(us, f"{f}|ALL|{sn}", f"{f} all strata {sn}", "pilot-pooled")
        if us:
            s["n_clusters"] = len({u["parent"] for u in us})
            s["delta_ci95_cluster"] = list(cluster_boot(np.array([float(u["on"]) - float(u["off"]) for u in us]), [u["parent"] for u in us]))
            s.pop("delta_ci95", None)
        rows_pooled.append(s)
    # stratum-level cells used by G5: steps 2-4 pooled over magnitude; ledger |balance|>99 pooled over steps
    g5 = {}
    for f in FAMS:
        fu = [u for u in units if u["fam"] == f]
        for st in sorted({u["stratum"] for u in fu}):
            for nm, pr in (("steps2-4", lambda u: u["step"] >= 1), ("big", lambda u: u["big"])):
                if nm == "big" and f != "L":
                    continue
                us = [u for u in fu if u["stratum"] == st and pr(u)]
                s = summarize(us, f"{f}|{st}|{nm}", f"{f} {st} {nm}", "g5")
                if us:
                    s["n_clusters"] = len({u["parent"] for u in us})
                    s["delta_ci95_cluster"] = list(cluster_boot(np.array([float(u["on"]) - float(u["off"]) for u in us]), [u["parent"] for u in us]))
                    s.pop("delta_ci95", None)
                g5[f"{f}|{st}|{nm}"] = s
    # TEST per-step accuracy (chain prediction)
    per_step = {}
    for f in FAMS:
        for t in range(4):
            us = [u for u in units if u["fam"] == f and u["split"] == "test" and u["step"] == t]
            per_step[f"{f}|{t}"] = {"n": len(us), "off": float(np.mean([u["off"] for u in us])) if us else float("nan"),
                                    "on": float(np.mean([u["on"] for u in us])) if us else float("nan")}
    return {"cells": rows, "pooled": rows_pooled, "g5_cells": g5, "per_step_test": per_step,
            "false_copy": false_copy_table(units)}


# ---------------------------------------------------------------- block D / E


def load_jsonl_checked(path: Path) -> list[dict]:
    """load_jsonl for instrument outputs: an unreadable line (e.g. a write cut short by a crash) is fatal."""
    out = []
    for ln, l in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError as e:
            die(f"{path} is not valid JSONL", [f"line {ln + 1}: {e}"])
        if not isinstance(r, dict):
            die(f"{path} is not valid JSONL", [f"line {ln + 1}: not a JSON object"])
        out.append(r)
    return out


def test_cands(bank: dict) -> dict:
    return {c["cand_id"]: (p, c) for p in bank["parents"] if p["split"] == "test" for c in p["candidates"]}


def chain_errors(rows: list[dict], cands: dict, name: str) -> list[str]:
    """Every way a chain file can differ from what run_t2._write_chains(generation.chain(...)) writes."""
    errs = []
    ids = [r.get("cand_id") for r in rows]
    cnt = defaultdict(int)
    for c in ids:
        cnt[str(c)] += 1
    dup = sorted(c for c, k in cnt.items() if k > 1)
    if dup:
        errs.append(f"{name}: {len(dup)} duplicated cand_id(s), e.g. {dup[:3]}")
    got = {str(c) for c in ids}
    miss, extra = sorted(set(cands) - got), sorted(got - set(cands))
    if miss or extra:
        errs.append(f"{name}: cand_id set differs from pilot TEST ({len(got)} unique ids on {len(rows)} lines vs {len(cands)}; "
                    f"{len(miss)} missing e.g. {miss[:3]}; {len(extra)} unknown e.g. {extra[:3]})")
    for ln, r in enumerate(rows):
        cid = r.get("cand_id")
        if cid not in cands:
            continue
        p, c = cands[cid]
        st, tx = r.get("states"), r.get("texts")
        where = f"{name} line {ln + 1} ({cid})"
        if not isinstance(st, list) or not isinstance(tx, list):
            errs.append(f"{where}: states/texts missing")
            continue
        if len(st) != len(c["events"]) + 1:
            errs.append(f"{where}: {len(st)} states for {len(c['events'])} events")
            continue
        if st[0] != env.dumps(p["s0"]):
            errs.append(f"{where}: states[0] is not this candidate's s0")
            continue
        fn = next((i for i, s in enumerate(st) if s is None), len(st))
        if any(s is not None for s in st[fn:]) or not all(isinstance(s, str) for s in st[:fn]):
            errs.append(f"{where}: states must be strings up to the first null and null after it")
            continue
        n_try = len(c["events"]) if fn == len(st) else fn
        if len(tx) != n_try or not all(isinstance(t, str) for t in tx):
            errs.append(f"{where}: {len(tx)} texts for {n_try} attempted steps")
            continue
        for t, (txt, s) in enumerate(zip(tx, st[1:])):
            ps = env.parse_state(p["family"], txt, lenient=True)
            if (None if ps is None else env.dumps(ps)) != s:
                errs.append(f"{where}: step {t} text does not parse to the stored state")
                break
    return errs


def load_run_chains(run_dir: Path, cands: dict) -> dict:
    """{mode: {'off': {cand_id: row}, 'on': {...}}}, or {'missing': [...]} if a chain file is absent.
    A present but incomplete / duplicated / malformed file is fatal: a missing OFF row would otherwise be scored
    as wrong (and as a near-random decision), which inflates every ON - OFF chain and regret difference."""
    gen, missing, errs = {}, [], []
    for m in MODES:
        for k, suffix in (("off", "_base"), ("on", "")):
            p = run_dir / f"gen_{m}{suffix}.jsonl"
            if not p.exists():
                missing.append(p.name)
                continue
            rows = load_jsonl_checked(p)
            errs += chain_errors(rows, cands, p.name)
            gen.setdefault(m, {})[k] = {r.get("cand_id"): r for r in rows}
    if errs:
        die(f"chain files in {run_dir} are incomplete or malformed", errs)
    return {"missing": missing} if missing else gen


def chain_block(gen: dict, S, pilot_units: list[dict] | None, pilot_block_res: dict | None) -> dict:
    ct_rows, cands, stored = S.ct_rows, S.cands, S.stored_chains
    te = [r for r in ct_rows if r["split"] == "test"]
    acc = []
    for m in MODES:
        for f in FAMS:
            primary = PRIMARY_CHAIN[f] == m
            for k in (1, 2, 4):
                rs = [r for r in te if r["family"] == FULL[f] and r["k"] == k]
                on = np.array([gen[m]["on"][r["cand_id"]]["states"][k] == r["gold"] for r in rs], bool)
                off = np.array([gen[m]["off"][r["cand_id"]]["states"][k] == r["gold"] for r in rs], bool)
                cp = np.array([r["gold"] == env.dumps(r["s0"]) for r in rs], bool)
                par = [r["parent_id"] for r in rs]
                acc.append({"mode": m, "fam": f, "k": k, "role": "primary" if primary else "TRANSFER", "n": len(rs),
                            "n_parents": len(set(par)), "copy_floor": float(cp.mean()), "off": float(off.mean()),
                            "on": float(on.mean()), "delta": float(on.mean() - off.mean()),
                            "delta_ci95_parent": list(cluster_boot(on.astype(float) - off.astype(float), par)),
                            "on_ci95_parent": list(cluster_boot(on.astype(float), par))})
    # predicted k=4 for the primary frame (pilot items are in the chain's primary frame)
    pred = {}
    if pilot_units is not None and pilot_block_res is not None:
        look = {u["payload"]: u for u in pilot_units if u["split"] == "test"}
        for f in FAMS:
            ps = pilot_block_res["per_step_test"]
            prod = {s: float(np.prod([ps[f"{f}|{t}"][s] for t in range(4)])) for s in ("off", "on")}
            path = {"off": [], "on": []}
            miss = 0
            for cid, (p, c) in cands.items():
                if p["family"] != FULL[f]:
                    continue
                st = env.rollout(p["family"], p["s0"], c["events"])
                us = [look.get(env.dumps({"before": st[t], "event": ev})) for t, ev in enumerate(c["events"])]
                if any(u is None for u in us):
                    miss += 1
                    continue
                for s in ("off", "on"):
                    path[s].append(float(all(u[s] for u in us)))
            obs = next(a for a in acc if a["mode"] == PRIMARY_CHAIN[f] and a["fam"] == f and a["k"] == 4)
            pred[f] = {"mode": PRIMARY_CHAIN[f], "product_of_step_acc": prod,
                       "gold_path_product": {s: float(np.mean(v)) if v else float("nan") for s, v in path.items()},
                       "observed": {"off": obs["off"], "on": obs["on"]}, "chains_without_pilot_items": miss}
    # P chain steps on the chain's own correct-input steps (R8)
    psteps = {}
    for m in MODES:
        for s in ("off", "on"):
            cnt = defaultdict(lambda: [0, 0])
            for cid, (p, c) in cands.items():
                if p["family"] != "prerequisite_plan":
                    continue
                g = gen[m][s][cid]
                gold = env.rollout(p["family"], p["s0"], c["events"])
                for t, ev in enumerate(c["events"]):
                    if g["states"][t] != env.dumps(gold[t]):
                        continue
                    if not env.applicable(gold[t], ev):
                        kind = "inapplicable"
                    elif gold[t + 1] != gold[t]:
                        kind = "applicable, state-changing"
                    else:
                        kind = "applicable, unchanged"
                    cnt[kind][0] += 1
                    cnt[kind][1] += int(g["states"][t + 1] == env.dumps(gold[t + 1]))
            for kind, (n, k) in cnt.items():
                psteps.setdefault(m, {}).setdefault(kind, {})[s] = {"n": n, "acc": k / n if n else float("nan")}
    for m, d in psteps.items():
        for kind, x in d.items():
            # chain_one P runs in the one-step frame, whose prompt presumes the action is applicable (D-19)
            x["role"] = "primary" if m == PRIMARY_CHAIN["P"] else (XF_DESC if kind == "inapplicable" else "TRANSFER")
    # reproducibility: same-session OFF vs stored Titan-Sparse pilot chains
    repro = {}
    for m in MODES:
        for f in FAMS:
            cids = [cid for cid, (p, c) in cands.items() if p["family"] == FULL[f]]
            same_states = same_steps = steps = 0
            for cid in cids:
                a, b = gen[m]["off"][cid], stored[m][cid]
                same_states += int(a["states"] == b["states"])
                for ta, tb in zip(a["texts"], b["texts"]):
                    steps += 1
                    same_steps += int(ta == tb)
            rs = [r for r in te if r["family"] == FULL[f]]
            st_acc = {k: float(np.mean([stored[m][r["cand_id"]]["states"][k] == r["gold"] for r in rs if r["k"] == k])) for k in (1, 2, 4)}
            off_acc = {k: next(a["off"] for a in acc if a["mode"] == m and a["fam"] == f and a["k"] == k) for k in (1, 2, 4)}
            repro[f"{m}|{f}"] = {"n_chains": len(cids), "identical_chains": same_states / max(len(cids), 1),
                                 "identical_step_texts": same_steps / max(steps, 1), "n_steps_compared": steps,
                                 "stored_acc": st_acc, "off_acc": off_acc}
    return {"acc": acc, "predicted_k4": pred, "p_steps": psteps, "repro": repro}


def decision_block(ct_rows: list[dict], bank: dict, gen: dict, stored: dict) -> dict:
    copy = {"chain_copy": {c["cand_id"]: {"states": [env.dumps(p["s0"])] * (len(c["events"]) + 1)}
                           for p in bank["parents"] if p["split"] == "test" for c in p["candidates"]}}
    out = {}
    for f in FAMS:
        fam = FULL[f]
        cp = decisions.g_only(ct_rows, fam, 4, copy, modes=("chain_copy",))["chain_copy"]["nregret"]
        # One g_only call per side with both modes, as in the review scripts: the tie-break rng (seed 17)
        # then reaches chain_m1 in the same state for OFF, ON and stored, so the pairing is exact and
        # the stored-Sparse numbers reproduce docs/t2_review (L 0.507/0.509, P 0.879/0.878, R 0.386/0.392).
        with warnings.catch_warnings():       # all-unparseable parents: "Mean of empty slice" inside g_only
            warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
            G_off = decisions.g_only(ct_rows, fam, 4, {m: gen[m]["off"] for m in MODES}, modes=MODES)
            G_on = decisions.g_only(ct_rows, fam, 4, {m: gen[m]["on"] for m in MODES}, modes=MODES)
            G_st = decisions.g_only(ct_rows, fam, 4, {m: stored[m] for m in MODES}, modes=MODES)
        for m in MODES:
            off, on, st = G_off[m], G_on[m], G_st[m]
            assert off["pids"] == on["pids"] == st["pids"]
            a, b = np.asarray(off["nregret"], float), np.asarray(on["nregret"], float)
            pb = decide.paired_bootstrap(a, b)          # mean(OFF - ON) = regret drop
            out[f"{f}|{m}"] = {"fam": f, "mode": m, "role": "primary" if PRIMARY_CHAIN[f] == m else "TRANSFER",
                               "n_parents": len(off["pids"]), "n_nan": int(np.isnan(a).sum()),
                               "off": float(np.nanmean(a)), "on": float(np.nanmean(b)),
                               "stored_sparse": float(np.nanmean(np.asarray(st["nregret"], float))),
                               "copy_chain": float(np.nanmean(np.asarray(cp, float))),
                               "drop": pb["mean_diff"], "drop_ci95": pb["ci95"], "n_paired": pb["n"]}
    return out


# ---------------------------------------------------------------- block F


_STATE_RE = re.compile(r'\{\s*"(?:A|B|C|D|X|Y|Z|before|after|events?|event)"\s*:')


def looks_state(t: str) -> bool:
    s = (t or "").strip()
    if _STATE_RE.search(s):
        return True
    if s[:1] in ("{", "["):
        try:
            json.loads(s)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    return False


def int_only(t: str) -> bool:
    return re.fullmatch(r"-?\d+\.?", (t or "").strip()) is not None


def _lsm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    m = a.max(1, keepdims=True)
    return a - (m + np.log(np.exp(a - m).sum(1, keepdims=True)))


def lp_compare(P: np.ndarray, Q: np.ndarray, chunk: int = 8):
    """Per-row KL(P||Q) of renormalised stored log-probs, and argmax of each."""
    n = P.shape[0]
    kl, ap, aq = np.zeros(n), np.zeros(n, np.int64), np.zeros(n, np.int64)
    for s in range(0, n, chunk):
        a, b = _lsm(np.asarray(P[s:s + chunk])), _lsm(np.asarray(Q[s:s + chunk]))
        kl[s:s + chunk] = (np.exp(a) * (a - b)).sum(1)
        ap[s:s + chunk], aq[s:s + chunk] = a.argmax(1), b.argmax(1)
    return np.maximum(kl, 0.0), ap, aq


def _kl_summary(kl):
    return {"mean": float(kl.mean()), "median": float(np.median(kl)), "p90": float(np.quantile(kl, 0.9)),
            "max": float(kl.max()), "n": int(len(kl))}


def offtask_errors(lp_path: Path, meta_path: Path, S, name: str) -> list[str]:
    """A captured off-task log-prob set must cover the 248 prompts of data/offtask_prompts.jsonl in order."""
    errs = []
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        return [f"{name}: offtask_meta.json unreadable ({e})"]
    if m.get("ids") != S.offtask_ids:
        errs.append(f"{name}: meta ids differ from data/offtask_prompts.jsonl ({len(m.get('ids') or [])} vs {len(S.offtask_ids)})")
    if m.get("reply_class") != S.offtask_classes:
        errs.append(f"{name}: meta reply_class list differs from data/offtask_prompts.jsonl")
    try:
        shp = np.load(lp_path, mmap_mode="r").shape
    except (ValueError, OSError) as e:
        return errs + [f"{name}: offtask_logprobs.npy unreadable ({e})"]
    if shp != (len(S.offtask_ids), S.offtask_vocab):
        errs.append(f"{name}: offtask_logprobs.npy shape {shp} != ({len(S.offtask_ids)}, {S.offtask_vocab})")
    return errs


def offtask_gen_errors(rows: list[dict], S) -> list[str]:
    errs = []
    if [r.get("id") for r in rows] != S.offtask_ids:
        errs.append(f"offtask_gen.jsonl: {len(rows)} rows whose ids do not equal the {len(S.offtask_ids)} prompt ids in order")
    elif [r.get("reply_class") for r in rows] != S.offtask_classes:
        errs.append("offtask_gen.jsonl: reply_class differs from data/offtask_prompts.jsonl")
    bad = [i for i, r in enumerate(rows) if not (isinstance(r.get("base_text"), str) and isinstance(r.get("t2_text"), str))]
    if bad:
        errs.append(f"offtask_gen.jsonl: {len(bad)} rows without string base_text/t2_text, e.g. line {bad[0] + 1}")
    return errs


def validate_offtask(run_dir: Path, S) -> None:
    errs = []
    if (run_dir / "offtask_meta.json").exists():
        errs += offtask_errors(run_dir / "offtask_logprobs.npy", run_dir / "offtask_meta.json", S, "ON")
    if (run_dir / "off" / "offtask_meta.json").exists():
        errs += offtask_errors(run_dir / "off" / "offtask_logprobs.npy", run_dir / "off" / "offtask_meta.json", S, "OFF")
    gp = run_dir / "offtask_gen.jsonl"
    if gp.exists():
        errs += offtask_gen_errors(load_jsonl_checked(gp), S)
    if errs:
        die(f"off-task files in {run_dir} are incomplete or malformed", errs)


def forgetting_block(run_dir: Path, S) -> dict:
    """Assumes validate_offtask passed (metas list the prompts in order; arrays have one row per prompt)."""
    need = [run_dir / "offtask_logprobs.npy", run_dir / "offtask_meta.json", run_dir / "off" / "offtask_logprobs.npy",
            run_dir / "off" / "offtask_meta.json"]
    out = {}
    miss = [str(p.relative_to(run_dir)) for p in need if not p.exists()]
    if miss:
        out["logprobs_missing"] = miss
    else:
        m_on = json.loads(need[1].read_text(encoding="utf-8"))
        m_st = json.loads((STORED_SPARSE / "offtask_meta.json").read_text(encoding="utf-8"))
        ON = np.load(need[0], mmap_mode="r")
        OFF = np.load(need[2], mmap_mode="r")
        ST = np.load(STORED_SPARSE / "offtask_logprobs.npy", mmap_mode="r")
        if not (ON.shape == OFF.shape == ST.shape):
            die("off-task log-prob arrays differ in shape", [f"ON {ON.shape}, OFF {OFF.shape}, stored Sparse {ST.shape}"])
        kl, a_off, a_on = lp_compare(OFF, ON)
        kl_noise, a_st, _ = lp_compare(ST, OFF)
        out["kl_off_on"] = _kl_summary(kl)
        out["kl_noise_stored_off"] = _kl_summary(kl_noise)
        out["top1_agree_off_on"] = float((a_off == a_on).mean())
        out["top1_agree_stored_off"] = float((a_st == a_off).mean())
        cls = m_on["reply_class"]
        by = defaultdict(list)
        for c, x in zip(cls, kl):
            by[c].append(x)
        out["kl_off_on_by_class_top10"] = sorted(((c, float(np.mean(v))) for c, v in by.items()), key=lambda t: -t[1])[:10]
        supa = STORED_SUPA / "offtask_logprobs.npy"
        if supa.exists() and (STORED_SUPA / "offtask_meta.json").exists():   # capture_offtask writes the meta last
            errs = offtask_errors(supa, STORED_SUPA / "offtask_meta.json", S, "stored Supa reference")
            if errs:
                die("the stored Supa off-task reference does not match the prompts", errs)
            SU = np.load(supa, mmap_mode="r")
            kl_ref, a_su, _ = lp_compare(SU, ST)
            out["kl_ref_supa_sparse"] = _kl_summary(kl_ref)
            out["top1_agree_supa_sparse"] = float((a_su == a_st).mean())
        else:
            out["kl_ref_supa_sparse"] = "pending"
    gp = run_dir / "offtask_gen.jsonl"
    if not gp.exists():
        out["greedy_missing"] = True
        return out
    g = load_jsonl_checked(gp)
    same = np.array([r["base_text"] == r["t2_text"] for r in g], bool)
    out["greedy_exact_agree"] = {"n": len(g), "rate": float(same.mean())}
    pc = defaultdict(list)
    for r, s in zip(g, same):
        pc[r["reply_class"]].append(bool(s))
    out["greedy_exact_agree_by_class"] = {c: {"n": len(v), "rate": float(np.mean(v))} for c, v in sorted(pc.items())}
    nonnum = [r for r in g if r["reply_class"] not in NUMERIC_CLASSES]
    bleed = {}
    for side, key in (("off", "base_text"), ("on", "t2_text")):
        b = [looks_state(r[key]) or int_only(r[key]) for r in nonnum]
        js = [looks_state(r[key]) for r in g]
        bleed[side] = {"bleed_rate_nonnumeric": float(np.mean(b)) if b else float("nan"), "n_nonnumeric": len(nonnum),
                       "json_state_rate_all": float(np.mean(js)), "n_all": len(g)}
    out["json_bleed"] = bleed
    return out


# ---------------------------------------------------------------- instrument checks


def check_rows(gen_rows: list[dict], ref_rows: list[dict], name: str) -> list[str]:
    """gen rows must be the reference rows in order (all row fields equal, 'i' == index) with string texts."""
    if len(gen_rows) != len(ref_rows):
        return [f"{name}: {len(gen_rows)} rows vs {len(ref_rows)} reference rows"]
    errs = []
    keys = ("target", "family", "stratum", "frame", "before", "event", "pair_id", "pair_role", "split", "step", "cand_id")
    for i, (g, r) in enumerate(zip(gen_rows, ref_rows)):
        bad = [k for k in keys if k in r and g.get(k) != r.get(k)]
        if g.get("i") != i:
            bad.append("i")
        bad += [f"{k} (not a string)" for k in ("base_text", "t2_text") if not isinstance(g.get(k), str)]
        if bad:
            errs.append(f"{name} row {i}: mismatch in {bad}")
    return errs


def eval_json_check(run_dir: Path, gens: dict, refs: dict) -> dict:
    """run_t2 writes eval.json last, after every stage: if it exists the run is complete and must be consistent."""
    p = run_dir / "eval.json"
    if not p.exists():
        return {"present": False}
    try:
        ej = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        die(f"{p} is not valid JSON", [str(e)])
    errs = []
    for name in SETS:
        rows = gens.get(name)
        if rows is None:
            errs.append(f"gen_{name}.jsonl missing although eval.json exists")
            continue
        if (ej.get("n") or {}).get(name) != len(refs[name]):
            errs.append(f"eval.json n[{name}] = {(ej.get('n') or {}).get(name)} != {len(refs[name])} reference rows")
        for side, key in (("base", "base_text"), ("t2", "t2_text")):
            ref = ej.get(f"per_op_{name}_{side}")
            if ref is None:
                errs.append(f"eval.json per_op_{name}_{side} absent")
                continue
            got = per_op_summary(rows, [r.get(key) for r in rows])
            if set(got) != set(ref) or any(got[k]["n"] != ref[k]["n"] or abs(got[k]["acc"] - ref[k]["acc"]) > 1e-9 for k in got):
                errs.append(f"eval.json per_op_{name}_{side} differs from a recomputation from gen_{name}.jsonl")
    missing = [f for f in CHAIN_FILES + OFFTASK_FILES if not (run_dir / f).exists()]
    if missing:
        errs.append(f"files missing although eval.json exists: {missing}")
    prov = ej.get("provenance") or {}
    shas = prov.get("data_sha256") or {}
    if not shas:
        errs.append("eval.json provenance has no data_sha256")
    for k, v in shas.items():
        f = ROOT / "data" / f"{k}.jsonl"
        if not (f.exists() and sha256(f) == v):
            errs.append(f"eval.json provenance data_sha256[{k}] does not match the current data/{k}.jsonl")
    if errs:
        die(f"{p} does not match the run's files", errs)
    return {"present": True, "n": ej.get("n"), "consistent": True, "mismatches": [],
            "base_fingerprint": prov.get("base_fingerprint"), "adapter": prov.get("adapter"),
            "data_sha_match": {k: True for k in shas}}


# ---------------------------------------------------------------- gates


def _status(ok, pending=False):
    return "PENDING" if pending else ("PASS" if ok else "FAIL")


def g5_gate(A: dict, B: dict) -> dict:
    """G5: per-op gains in rule strata must carry over to pilot chain steps 2-4 and (ledger) |balance| > 99.
    Significant changes elsewhere (untouched strata, per-op drops) are listed as descriptive, not gated."""
    checks, skipped, desc, gained = [], [], [], []
    for key, c in A["by_stratum"].items():
        if not c["n"] or c["p_mcnemar"] >= 0.05 or sign(c["delta"]) == 0:
            continue
        f, st = key.split("|", 1)
        rule = st in RULE_STRATA[f]
        gated = rule and c["delta"] > 0
        if gated:
            gained.append(f"{f} {st}")
        for nm in (("steps2-4", "big") if f == "L" else ("steps2-4",)):
            pc = B["g5_cells"].get(f"{f}|{st}|{nm}")
            has = pc is not None and pc["n"] > 0
            row = {"cell": f"{f} {st} {nm}", "perop_delta": c["delta"], "perop_p": c["p_mcnemar"],
                   "pilot_n": pc["n"] if has else 0, "pilot_off": pc["off"] if has else None,
                   "pilot_on": pc["on"] if has else None, "pilot_delta": pc["delta"] if has else None,
                   "pilot_delta_ci95_cluster": pc.get("delta_ci95_cluster") if has else None}
            if not gated:
                row["why_not_gated"] = "untouched stratum (G2)" if not rule else "per-op drop (G1/G2)"
                desc.append(row)
                continue
            if not has:
                skipped.append(f"{f} {st} {nm} (no pilot rows)")
                continue
            if pc["off"] >= 1 - EPS and pc["on"] >= 1 - EPS:
                skipped.append(f"{f} {st} {nm} (OFF = ON = 1.0, no room)")
                continue
            row["ok"] = pc["delta"] > EPS
            checks.append(row)
    base = {"checks": checks, "skipped": skipped, "descriptive": desc, "rule_strata_gained": gained}
    if not gained:
        return {"status": "FAIL", "detail": "no rule stratum gained significantly per-op (nothing to carry over)", **base}
    if not checks:
        return {"status": "FAIL", "detail": f"rule strata gained ({', '.join(gained)}) but no pilot cell could be checked", **base}
    bad = [c["cell"] for c in checks if not c["ok"]]
    return {"status": _status(not bad),
            "detail": (f"{len(checks)} pilot cells from {len(gained)} gained rule strata, all gain" if not bad
                       else f"no pilot gain in {len(bad)} of {len(checks)} cells: " + ", ".join(bad)), **base}


def gates(res: dict) -> dict:
    G = {}
    A = res.get("A")
    if A is None:
        for g in ("G1_targeted", "G2_untouched", "G3_app_del", "G4_false_copy", "G5_pilot_direction"):
            G[g] = {"status": "PENDING", "detail": "one-step files missing"}
    else:
        cells = {c["id"]: c for c in A["cells"]}
        tg = [c for c in A["cells"] if c["role"] == "targeted"]
        fail = [f"{c['id']} {c.get('on', float('nan')):.3f}" for c in tg if not (c["n"] and c["on"] >= TH_TARGETED - EPS)]
        G["G1_targeted"] = {"status": _status(not fail), "detail": ("all >= 0.80: " if not fail else "below 0.80: ")
                            + ", ".join(fail if fail else [f"{c['id']} {c['on']:.3f}" for c in tg]),
                            "values": {c["id"]: c.get("on") for c in tg}}
        un = [c for c in A["cells"] if c["role"] == "untouched"]
        fail = [f"{c['id']} {c['delta']:+.3f}" for c in un if not (c["n"] and c["delta"] >= -TH_UNTOUCHED_DROP - EPS)]
        kinds = ", ".join(f"{k.split('.')[1].split('_')[0]} {f3(cells[k].get('delta'), signed=True)}"
                          for k in ("L.add_posted", "L.scale_posted", "L.transfer_posted"))
        G["G2_untouched"] = {"status": _status(not fail),
                             "detail": ", ".join(f"{c['id']} {f3(c.get('delta'), signed=True)}" + (f" (per kind, not gated: {kinds})" if c["id"] == "L.posted" else "")
                                                 for c in un),
                             "values": {c["id"]: c.get("delta") for c in un}}
        ad = cells["P.app_del"]
        lo, hi = ad["delta_ci95"] if ad["n"] else (float("nan"), float("nan"))
        G["G3_app_del"] = {"status": _status(ad["n"] > 0 and lo > 0),
                           "detail": f"n={ad['n']} OFF {f3(ad.get('off'))} ON {f3(ad.get('on'))} "
                                     f"delta {f3(ad.get('delta'), signed=True)} CI [{f3(lo, signed=True)}, {f3(hi, signed=True)}]",
                           "values": {"delta": ad.get("delta"), "ci_lo": lo}}
        fc = A["false_copy"]
        rise = [f for f in FAMS if "on" in fc[f] and fc[f]["on"] > fc[f]["off"] + EPS]
        g4 = []
        for f in FAMS:
            x = fc[f]
            if "on" not in x:
                continue
            spare = x["off_copies"] - x["on_copies"]
            g4.append(f"{f} {x['off']:.3f}->{x['on']:.3f} ({x['off_copies']}->{x['on_copies']} of {x['n_target_ne_before']}; "
                      f"new {x['on_new_copies']}/stopped {x['on_stopped_copies']}, McNemar p {pfmt(x['p_mcnemar'])}; "
                      + (f"{spare} spare row(s) before FAIL)" if spare >= 0 else f"over by {-spare} row(s))"))
        G["G4_false_copy"] = {"status": _status(not rise),
                              "detail": "point estimate, one extra copied row fails: " + "; ".join(g4),
                              "values": {f: fc[f].get("delta") for f in FAMS},
                              "spare_rows": {f: fc[f]["off_copies"] - fc[f]["on_copies"] for f in FAMS if "on" in fc[f]}}
        B = res.get("B")
        if B is None:
            G["G5_pilot_direction"] = {"status": "PENDING", "detail": "gen_pilot.jsonl missing"}
        else:
            G["G5_pilot_direction"] = g5_gate(A, B)
    D = res.get("D")
    if D is None or "missing" in D:
        G["G6_chains_k4"] = {"status": "PENDING", "detail": "chain files missing"}
    else:
        vals = {f: next(a for a in D["acc"] if a["mode"] == PRIMARY_CHAIN[f] and a["fam"] == f and a["k"] == 4) for f in FAMS}
        fail = [f for f in FAMS if vals[f]["on"] < TH_CHAIN_K4[f] - EPS]
        G["G6_chains_k4"] = {"status": _status(not fail),
                             "detail": ", ".join(f"{f} {PRIMARY_CHAIN[f]} {vals[f]['off']:.3f}->{vals[f]['on']:.3f} (>= {TH_CHAIN_K4[f]})" for f in FAMS),
                             "values": {f: vals[f]["on"] for f in FAMS}}
    E = res.get("E")
    if E is None:
        G["G7_decision"] = {"status": "PENDING", "detail": "chain files missing"}
    else:
        vals = {f: E[f"{f}|{PRIMARY_CHAIN[f]}"] for f in FAMS}
        fail = [f for f in FAMS if not (vals[f]["drop"] >= TH_REGRET_DROP[f] - EPS and vals[f]["drop_ci95"][0] > 0)]
        G["G7_decision"] = {"status": _status(not fail),
                            "detail": ", ".join(f"{f} drop {vals[f]['drop']:+.3f} [{vals[f]['drop_ci95'][0]:+.3f}, {vals[f]['drop_ci95'][1]:+.3f}] (>= {TH_REGRET_DROP[f]})" for f in FAMS),
                            "values": {f: vals[f]["drop"] for f in FAMS}}
    F = res.get("F") or {}
    if "kl_off_on" not in F:
        G["G8_forgetting"] = {"status": "PENDING", "detail": "off-task log-probs missing"}
    elif F.get("kl_ref_supa_sparse") == "pending":
        G["G8_forgetting"] = {"status": "PENDING", "detail": f"KL(OFF||ON) mean {F['kl_off_on']['mean']:.4f}; reference KL(Supa||Sparse) pending "
                                                             f"(runs/bank_pilot/supa/offtask_logprobs.npy absent)",
                              "values": {"kl": F["kl_off_on"]["mean"], "ref": None}}
    else:
        k, r = F["kl_off_on"]["mean"], F["kl_ref_supa_sparse"]["mean"]
        G["G8_forgetting"] = {"status": _status(k <= r + EPS), "detail": f"KL(OFF||ON) mean {k:.4f} vs KL(Supa||Sparse) mean {r:.4f}",
                              "values": {"kl": k, "ref": r}}
    st = [G[g]["status"] for g in GATE_NAMES]
    G["rule_learned"] = {"status": "FAIL" if "FAIL" in st else ("PENDING" if "PENDING" in st else "PASS"),
                         "detail": "all of G1-G8; " + ", ".join(f"{g.split('_')[0]} {G[g]['status']}" for g in GATE_NAMES)}
    return G


# ---------------------------------------------------------------- per-seed driver


def static_errors(t2_eval: list[dict], xframe: list[dict], subs: list[dict]) -> list[str]:
    """Block A joins t2_eval with the substrata by index, block C reuses the same labels for xframe."""
    errs = []
    if len(subs) != len(t2_eval) or len(xframe) != len(t2_eval):
        return [f"lengths: t2_eval {len(t2_eval)}, substrata {len(subs)}, xframe {len(xframe)}"]
    for i, (r, s, x) in enumerate(zip(t2_eval, subs, xframe)):
        if s.get("idx") != i or s.get("family") != r["family"] or s.get("stratum") != r["stratum"] or not isinstance(s.get("sub"), str):
            errs.append(f"t2_eval_substrata row {i} does not describe t2_eval row {i}")
        if any(x.get(k) != r.get(k) for k in ("family", "stratum", "before", "event", "target")):
            errs.append(f"t2_eval_xframe row {i} is not t2_eval row {i} re-rendered")
        f = CODE[r["family"]]
        if r["stratum"] not in RULE_STRATA[f] + UNTOUCHED_STRATA[f]:
            errs.append(f"t2_eval row {i}: stratum {f} {r['stratum']} is neither a rule nor an untouched stratum")
    return errs


class Static:
    """Data shared by every seed (loaded and validated once)."""

    def __init__(self):
        self.sha = {k: sha256(ROOT / "data" / (f"{k}.jsonl" if k != "bank_pilot" else "bank_pilot.json")) for k in EXPECTED_SHA}
        bad = [f"{k}: {v[:8]} != expected {EXPECTED_SHA[k]}" for k, v in self.sha.items() if not v.startswith(EXPECTED_SHA[k])]
        if bad:
            die("data files differ from the hashes recorded in D-16/D-19", bad)
        self.data = {k: load_jsonl(ROOT / "data" / v) for k, v in DATA_FILES.items()}
        sub_rows = load_jsonl(ROOT / "data" / "t2_eval_substrata.jsonl")
        errs = static_errors(self.data["t2_eval"], self.data["xframe"], sub_rows)
        if errs:
            die("static data files are not aligned", errs)
        self.subs = [r["sub"] for r in sub_rows]
        self.bank = json.loads((ROOT / "data" / "bank_pilot.json").read_text(encoding="utf-8"))
        self.pilot_ref = pilot_reference(self.bank)
        self.ct_rows = data.candidate_table(self.bank)
        self.cands = test_cands(self.bank)
        self.refs = {"t2_eval": self.data["t2_eval"], "xframe": self.data["xframe"], "supp": self.data["supp"], "pilot": self.pilot_ref}
        prompts = load_jsonl(OFFTASK_PROMPTS)
        self.offtask_ids = [r["id"] for r in prompts]
        self.offtask_classes = [r["reply_class"] for r in prompts]
        self.offtask_vocab = int(np.load(STORED_SPARSE / "offtask_logprobs.npy", mmap_mode="r").shape[1])
        errs = offtask_errors(STORED_SPARSE / "offtask_logprobs.npy", STORED_SPARSE / "offtask_meta.json", self, "stored Sparse")
        self.stored_chains = {}
        for m in MODES:
            p = STORED_SPARSE / f"gen_{m}.jsonl"
            rows = load_jsonl_checked(p)
            errs += chain_errors(rows, self.cands, f"stored {p.name}")
            self.stored_chains[m] = {r.get("cand_id"): r for r in rows}
        if errs:
            die("stored Titan-Sparse pilot artefacts (runs/bank_pilot/sparse) fail validation", errs)


def analyse_seed(run_dir: Path, S: Static) -> dict:
    res = {"run_dir": str(run_dir), "checks": []}
    # ---- validate every input before computing anything (SystemExit on any inconsistency)
    gens = {}
    for name in SETS:
        p = run_dir / f"gen_{name}.jsonl"
        gens[name] = load_jsonl_checked(p) if p.exists() else None
        if gens[name] is None:
            res["checks"].append(f"missing gen_{name}.jsonl")
    errs = []
    for name, rows in gens.items():
        if rows is not None:
            errs += check_rows(rows, S.refs[name], name)
    if errs:
        die(f"gen files in {run_dir} do not align with the data files", errs)
    res["eval_json"] = eval_json_check(run_dir, gens, S.refs)
    chains = load_run_chains(run_dir, S.cands)
    validate_offtask(run_dir, S)
    # ---- blocks
    units = {}
    if gens["t2_eval"] is not None and gens["supp"] is not None:
        units["A"] = make_units(gens["t2_eval"], "t2_eval", S.subs) + make_units(gens["supp"], "supp")
        res["A"] = onestep_block(units["A"], "t2_eval")
    if gens["xframe"] is not None:
        units["C"] = make_units(gens["xframe"], "xframe", S.subs)
        res["C"] = onestep_block(units["C"], "xframe")
    if gens["pilot"] is not None:
        units["B"] = make_units(gens["pilot"], "pilot")
        res["B"] = pilot_block(units["B"])
    if "missing" in chains:
        res["D"] = {"missing": chains["missing"]}
    else:
        res["D"] = chain_block(chains, S, units.get("B"), res.get("B"))
        res["E"] = decision_block(S.ct_rows, S.bank, chains, S.stored_chains)
    res["F"] = forgetting_block(run_dir, S)
    res["gates"] = gates(res)
    return res


# ---------------------------------------------------------------- across seeds


def direction_metrics(r: dict) -> dict:
    m = {}
    if "A" in r:
        for c in r["A"]["cells"]:
            if c["role"] in ("targeted", "untouched") and c["n"]:
                m[f"A {c['id']} delta"] = c["delta"]
        for f in FAMS:
            if "delta" in r["A"]["false_copy"][f]:
                m[f"A false-copy {f} (ON-OFF)"] = r["A"]["false_copy"][f]["delta"]
    if "B" in r:
        for c in r["B"]["pooled"]:
            if c["n"]:
                m[f"B {c['id']} delta"] = c["delta"]
    if "C" in r:
        for f, cid in (("L", "L.all"), ("P", "P.applicable"), ("R", "R.all")):   # P inapplicable rows are descriptive only
            c = next((x for x in r["C"]["cells"] if x["id"] == cid), None)
            if c and c["n"]:
                m[f"C xframe {cid} delta"] = c["delta"]
    D = r.get("D", {})
    for a in D.get("acc", []):
        if a["k"] == 4:
            m[f"D {a['mode']} {a['fam']} k4 delta ({a['role']})"] = a["delta"]
    for key, e in (r.get("E") or {}).items():
        m[f"E regret drop {key} ({e['role']})"] = e["drop"]
    F = r.get("F", {})
    if "kl_off_on" in F:
        m["F KL(OFF||ON) mean"] = F["kl_off_on"]["mean"]
    return m


def across(results: dict) -> dict:
    tags = list(results)
    mets = {t: direction_metrics(r) for t, r in results.items()}
    keys = sorted(set().union(*[set(v) for v in mets.values()]))
    rows = []
    for k in keys:
        vals = [mets[t].get(k) for t in tags]
        sg = [sign(v) if v is not None else None for v in vals]
        signs = " ".join("?" if s is None else {1: "+", -1: "-", 0: "0"}[s] for s in sg)
        if k.startswith("F KL"):
            agree = "n/a (KL >= 0; see G8)"
        elif any(s is None for s in sg):
            agree = "incomplete"
        elif all(s == 1 for s in sg):
            agree = "yes (+)"
        elif all(s == -1 for s in sg):
            agree = "yes (-)"
        elif all(s == 0 for s in sg):
            agree = "no change"
        else:
            agree = "no"
        good = "-" if "false-copy" in k else ("lower" if k.startswith("F KL") else "+")
        rows.append({"metric": k, "values": dict(zip(tags, vals)), "signs": signs, "agree": agree, "good_direction": good})
    gate_rows = {g: {t: results[t]["gates"][g]["status"] for t in tags} for g in list(GATE_NAMES) + ["rule_learned"]}
    n_pass = sum(results[t]["gates"]["rule_learned"]["status"] == "PASS" for t in tags)
    return {"tags": tags, "n_seeds": len(tags), "three_seeds": len(tags) >= 3, "metrics": rows, "gates": gate_rows,
            "rule_learned_pass_seeds": n_pass,
            "rule_learned_all_seeds": len(tags) >= 3 and n_pass == len(tags)}


# ---------------------------------------------------------------- markdown


def f3(x, nd=3, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


def ci(c, nd=3, signed=True):
    if not c or c[0] is None or (isinstance(c[0], float) and math.isnan(c[0])):
        return "n/a"
    return f"[{f3(c[0], nd, signed)}, {f3(c[1], nd, signed)}]"


def pfmt(p):
    return "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def table(headers, rows):
    esc = lambda x: str(x).replace("|", "\\|")  # noqa: E731
    out = ["| " + " | ".join(esc(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def cell_rows(cells):
    rows = []
    for c in cells:
        if not c["n"]:
            rows.append([c["label"], c["role"], 0] + ["n/a"] * 8)
            continue
        rows.append([c["label"], c["role"], c["n"], f3(c["copy_floor"]), f3(c["off"]), f3(c["on"]), f3(c["delta"], signed=True),
                     f"{c['b']}/{c['c']}", pfmt(c["p_mcnemar"]), ci(c["on_ci95"], signed=False), ci(c["delta_ci95"])])
    return rows


CELL_HDR = ["cell", "role", "n", "copy", "OFF", "ON", "delta", "b/c", "McNemar p", "ON 95% CI (Wilson)", "delta 95% CI (Newcombe)"]


def fc_rows(fc):
    rows = []
    for f in FAMS:
        x = fc[f]
        rows.append([f, x["n_rows"], x["n_target_ne_before"], f3(x.get("off")), f3(x.get("on")), f3(x.get("delta"), signed=True),
                     f"{x.get('on_new_copies', 'n/a')}/{x.get('on_stopped_copies', 'n/a')}", pfmt(x.get("p_mcnemar")),
                     ci(x.get("delta_ci95")), f3(x.get("off_parse_fail")), f3(x.get("on_parse_fail"))])
    return rows


FC_HDR = ["family", "rows", "target != before", "false-copy OFF", "false-copy ON", "delta", "new/stopped copies",
          "McNemar p", "delta 95% CI", "parse-fail OFF", "parse-fail ON"]


def render_seed(tag: str, r: dict) -> str:
    L = [f"## Seed `{tag}`", "", f"Run dir: `{r['run_dir']}`", ""]
    ej = r["eval_json"]
    L.append("### Instrument checks")
    L.append("")
    L.append("- Inputs validated (any failure would have stopped the analysis): gen rows equal the data rows in order with string "
             "texts (t2_eval, xframe, supp; pilot rebuilt from bank_pilot); chain files hold exactly the pilot TEST candidates "
             "with well-formed states that match their texts; off-task files cover the 248 prompts in order.")
    if r["checks"]:
        L.append(f"- Missing stages (run in progress or crashed; dependent gates PENDING): {', '.join(r['checks'])}.")
    D = r.get("D", {})
    if "missing" in D:
        L.append(f"- Missing chain files (G6/G7 PENDING): {', '.join(D['missing'])}.")
    if ej.get("present"):
        L.append("- eval.json present (run complete): per-op summaries and n match a recomputation from the gen files; "
                 "provenance data sha256 match the current data files; every stage file exists.")
        L.append(f"- Base fingerprint: {ej.get('base_fingerprint')}; adapter: {ej.get('adapter')}.")
    else:
        L.append("- eval.json absent (run not complete).")
    L.append("")
    G = r["gates"]
    L.append("### Gates")
    L.append("")
    L.append(table(["gate", "criterion", "status", "numbers"], [[g, GATE_NAMES[g], f"**{G[g]['status']}**", G[g]["detail"]] for g in GATE_NAMES]
                   + [["rule learned", "all of G1-G8", f"**{G['rule_learned']['status']}**", ""]]))
    L.append("")
    L.append("G4 is the literal D-19 point estimate ('does not rise'): one extra copied row in any family fails it. "
             "The spare margin and the exact McNemar p are printed so that a one-row flip can be told apart from a "
             "systematic rise. G2 gates the pooled L posted cell named in D-19; its per-kind deltas are shown for visibility.")
    g5 = G["G5_pilot_direction"]
    if g5.get("checks"):
        L.append("")
        L.append("G5 checks (rule stratum with a significant per-op gain -> pilot cell; delta CI = parent bootstrap, not gated):")
        L.append("")
        L.append(table(["pilot cell", "per-op delta", "per-op p", "pilot n", "pilot OFF", "pilot ON", "pilot delta", "pilot delta 95% CI", "gains"],
                       [[c["cell"], f3(c["perop_delta"], signed=True), pfmt(c["perop_p"]), c["pilot_n"], f3(c["pilot_off"]), f3(c["pilot_on"]),
                         f3(c["pilot_delta"], signed=True), ci(c["pilot_delta_ci95_cluster"]), "yes" if c["ok"] else "NO"] for c in g5["checks"]]))
    if g5.get("skipped"):
        L.append("")
        L.append(f"G5 skipped: {', '.join(g5['skipped'])}.")
    if g5.get("descriptive"):
        L.append("")
        L.append("Significant per-op changes not gated by G5 (untouched strata or per-op drops; pilot direction descriptive only):")
        L.append("")
        L.append(table(["pilot cell", "why not gated", "per-op delta", "per-op p", "pilot n", "pilot delta", "pilot delta 95% CI"],
                       [[c["cell"], c["why_not_gated"], f3(c["perop_delta"], signed=True), pfmt(c["perop_p"]), c["pilot_n"],
                         f3(c["pilot_delta"], signed=True), ci(c["pilot_delta_ci95_cluster"])] for c in g5["descriptive"]]))
    L.append("")
    if "A" in r:
        L.append("### A. One-step per-op, trained frame (t2_eval + supp)")
        L.append("")
        L.append("Roles: targeted = D-19 rule-isolating primary cell (G1); untouched = G2; detail = part of a targeted cell; "
                 "descriptive = copy or rule-ignoring answer passes. Pair cells count a pair right only if both members are right. "
                 "b = OFF right & ON wrong, c = OFF wrong & ON right.")
        L.append("")
        L.append(table(CELL_HDR, cell_rows(r["A"]["cells"])))
        L.append("")
        L.append("False-copy rate P(output == before | target != before) and parse failures, t2_eval + supp rows:")
        L.append("")
        L.append(table(FC_HDR, fc_rows(r["A"]["false_copy"])))
        L.append("")
        L.append("Per t2_eval stratum (G5 starts from the rule strata with a significant gain; untouched strata are not gated by G5):")
        L.append("")
        L.append(table(CELL_HDR, cell_rows(list(r["A"]["by_stratum"].values()))))
        L.append("")
    if "B" in r:
        B = r["B"]
        L.append("### B. Pilot oracle-anchored transitions (DEV+TEST, chain primary frame)")
        L.append("")
        L.append("Delta CI: percentile bootstrap over parents (10,000 resamples); this is the inference for B. The McNemar p is "
                 "row-level and ignores the clustering of pilot rows within parents (anti-conservative; reference only). "
                 "Ledger magnitude: >99 if any |balance| before or after exceeds 99.")
        L.append("")
        hdr = ["cell", "n", "parents", "copy", "OFF", "ON", "delta", "b/c", "McNemar p (row-level, ignores clustering)",
               "delta 95% CI (parent bootstrap)"]
        rows = [[c["label"], c["n"], c.get("n_clusters", 0), f3(c.get("copy_floor")), f3(c.get("off")), f3(c.get("on")),
                 f3(c.get("delta"), signed=True), f"{c.get('b')}/{c.get('c')}", pfmt(c.get("p_mcnemar")),
                 ci(c.get("delta_ci95_cluster"))] for c in B["pooled"] + B["cells"]]
        L.append(table(hdr, rows))
        L.append("")
        L.append("False-copy on pilot transitions:")
        L.append("")
        L.append(table(FC_HDR, fc_rows(B["false_copy"])))
        L.append("")
    if "C" in r:
        L.append("### C. Cross-frame re-render of t2_eval (TRANSFER)")
        L.append("")
        L.append("L/R rows in the multi-step frame with one event; P rows in the one-step frame, whose prompt presumes the action "
                 "is applicable, so P inapplicable cells and the P inapplicable stratum are descriptive only, 'P all' mixes them in "
                 "(descriptive), and 'P applicable' is the P transfer summary. Supplement cells have no cross-frame version.")
        L.append("")
        L.append(table(CELL_HDR, cell_rows(r["C"]["cells"])))
        L.append("")
        L.append(table(FC_HDR, fc_rows(r["C"]["false_copy"])))
        L.append("")
    if D and "missing" not in D:
        L.append("### D. Chains on pilot TEST (exact state)")
        L.append("")
        L.append("Primary: chain_one L/R, chain_m1 P; the rest TRANSFER. Delta CI: percentile bootstrap over parents.")
        L.append("")
        hdr = ["mode", "fam", "role", "k", "n", "parents", "copy-chain", "OFF", "ON", "delta", "delta 95% CI", "ON 95% CI"]
        L.append(table(hdr, [[a["mode"], a["fam"], a["role"], a["k"], a["n"], a["n_parents"], f3(a["copy_floor"]), f3(a["off"]),
                              f3(a["on"]), f3(a["delta"], signed=True), ci(a["delta_ci95_parent"]), ci(a["on_ci95_parent"], signed=False)]
                             for a in D["acc"]]))
        L.append("")
        if D.get("predicted_k4"):
            L.append("Predicted k=4 accuracy (primary frame) next to observed. Product = product of TEST per-step accuracies "
                     "t=0..3 from B; gold path = share of chains whose four oracle-anchored transitions are all right in B.")
            L.append("")
            L.append(table(["fam", "mode", "OFF product", "OFF gold path", "OFF observed", "ON product", "ON gold path", "ON observed"],
                           [[f, p["mode"], f3(p["product_of_step_acc"]["off"]), f3(p["gold_path_product"]["off"]), f3(p["observed"]["off"]),
                             f3(p["product_of_step_acc"]["on"]), f3(p["gold_path_product"]["on"]), f3(p["observed"]["on"])]
                            for f, p in D["predicted_k4"].items()]))
            L.append("")
        L.append("P chain steps on the chain's own correct-input steps (R8 convention):")
        L.append("")
        rows = []
        for m, d in D["p_steps"].items():
            for kind in ("applicable, state-changing", "applicable, unchanged", "inapplicable"):
                x = d.get(kind, {})
                o, n_ = x.get("off", {}), x.get("on", {})
                ao, an = o.get("acc", float("nan")), n_.get("acc", float("nan"))
                role = x.get("role") or ("primary" if m == PRIMARY_CHAIN["P"] else (XF_DESC if kind == "inapplicable" else "TRANSFER"))
                rows.append([m, role, kind, o.get("n", 0), f3(ao), n_.get("n", 0), f3(an), f3(an - ao, signed=True)])
        L.append(table(["mode", "role", "step kind", "n OFF", "acc OFF", "n ON", "acc ON", "delta"], rows))
        L.append("")
        L.append("Reproducibility: same-session OFF chains vs stored Titan-Sparse pilot chains (runs/bank_pilot/sparse):")
        L.append("")
        L.append(table(["mode|fam", "chains", "identical chains", "identical step texts", "stored k1/k2/k4", "OFF k1/k2/k4"],
                       [[k, v["n_chains"], f3(v["identical_chains"]), f3(v["identical_step_texts"]),
                         "/".join(f3(v["stored_acc"][kk]) for kk in (1, 2, 4)), "/".join(f3(v["off_acc"][kk]) for kk in (1, 2, 4))]
                        for k, v in D["repro"].items()]))
        L.append("")
    if r.get("E"):
        L.append("### E. Decisions: G_chain k=4 normalised regret (decisions.g_only, pilot TEST goals)")
        L.append("")
        L.append("Drop = OFF - ON (positive = better decisions), paired per-parent bootstrap (decide.paired_bootstrap, 10,000, seed 17). "
                 "Random selection = 1.0; oracle = 0.")
        L.append("")
        L.append(table(["fam", "mode", "role", "parents (nan)", "copy-chain", "stored Sparse", "OFF", "ON", "drop", "drop 95% CI"],
                       [[e["fam"], e["mode"], e["role"], f"{e['n_parents']} ({e['n_nan']})", f3(e["copy_chain"]), f3(e["stored_sparse"]),
                         f3(e["off"]), f3(e["on"]), f3(e["drop"], signed=True), ci(e["drop_ci95"])] for e in r["E"].values()]))
        L.append("")
    F = r.get("F", {})
    L.append("### F. Forgetting (248 off-task prompts)")
    L.append("")
    if "kl_off_on" in F:
        ref = F["kl_ref_supa_sparse"]
        rows = [["KL(OFF||ON), adapter effect", f3(F["kl_off_on"]["mean"], 4), f3(F["kl_off_on"]["median"], 4), f3(F["kl_off_on"]["p90"], 4),
                 f3(F["top1_agree_off_on"])],
                ["KL(stored Sparse||OFF), noise floor", f3(F["kl_noise_stored_off"]["mean"], 4), f3(F["kl_noise_stored_off"]["median"], 4),
                 f3(F["kl_noise_stored_off"]["p90"], 4), f3(F["top1_agree_stored_off"])]]
        if ref == "pending":
            rows.append(["KL(Supa||Sparse), reference", "pending", "pending", "pending", "pending"])
        else:
            rows.append(["KL(Supa||Sparse), reference", f3(ref["mean"], 4), f3(ref["median"], 4), f3(ref["p90"], 4), f3(F["top1_agree_supa_sparse"])])
        L.append(table(["first-token comparison", "mean KL", "median KL", "p90 KL", "top-1 agreement"], rows))
        L.append("")
        L.append("Largest mean KL(OFF||ON) by reply class: " + ", ".join(f"{c} {v:.4f}" for c, v in F["kl_off_on_by_class_top10"]) + ".")
        L.append("")
    else:
        L.append(f"Off-task log-probs missing: {F.get('logprobs_missing')}.")
        L.append("")
    if "greedy_exact_agree" in F:
        jb = F["json_bleed"]
        L.append(f"64-token greedy replies, exact OFF/ON agreement: {f3(F['greedy_exact_agree']['rate'])} (n={F['greedy_exact_agree']['n']}).")
        L.append("")
        L.append(table(["side", "JSON-bleed rate (non-numeric classes)", "n", "JSON-state-like (all prompts)", "n"],
                       [[s, f3(jb[s]["bleed_rate_nonnumeric"]), jb[s]["n_nonnumeric"], f3(jb[s]["json_state_rate_all"]), jb[s]["n_all"]]
                        for s in ("off", "on")]))
        L.append("")
        bc = F["greedy_exact_agree_by_class"]
        low = sorted(bc.items(), key=lambda kv: (kv[1]["rate"], kv[0]))
        L.append("Greedy agreement by reply class (ascending): " + ", ".join(f"{c} {v['rate']:.2f}" for c, v in low) + ".")
        L.append("")
    return "\n".join(L)


def render(results: dict, meta: dict) -> str:
    L = ["# RCWM-T2 results (D-19 analysis)", "",
         f"Generated by `scripts/t2_analysis.py` (sha256 `{meta['script_sha256']}`). Tags: {', '.join(results)}. Runs root: `{meta['runs_root']}`.", "",
         "Every number is adapter ON minus the same-session adapter OFF (Titan-Sparse). Exploratory (D-15); pilot TEST was used to "
         "diagnose the failures T2 targets, so chain and regret results on it are hypothesis-generating only (D-19).", "",
         "Data sha256 (expected prefix from D-16/D-19): " + ", ".join(
             f"{k} {v[:8]} ({'ok' if v.startswith(EXPECTED_SHA[k]) else 'MISMATCH, expected ' + EXPECTED_SHA[k]})" for k, v in meta["data_sha256"].items()) + ".", "",
         "Claim scope (D-19): ledger claims cover 'void -> no-op' only ('pending' and 'draft' are never shown); register "
         "conditional-boundary claims come from the supplement boundary pairs only, never from t2_eval; seed 0 is a screen, and a "
         "direction counts only if all three seeds agree. A 'rule learned' PASS does not by itself separate rule learning from "
         "frame adaptation: that needs the D-19 self-distillation control (a LoRA on Sparse's own greedy answers, queued only if "
         "the gates pass). The sealed confirmation bank is not used.", ""]
    tags = list(results)
    X = meta["across"]
    L.append("## Gates per seed")
    L.append("")
    if not X["three_seeds"]:
        L.append(f"**Seeds present: {X['n_seeds']} of 3.** Seed 0 alone is a screen (D-19); a direction counts only when all three "
                 "seeds agree, so every gate status and direction in this report is provisional.")
        L.append("")
    L.append(table(["gate", "criterion"] + tags, [[g, GATE_NAMES[g]] + [results[t]["gates"][g]["status"] for t in tags] for g in GATE_NAMES]
                   + [["rule learned", "all of G1-G8"] + [f"**{results[t]['gates']['rule_learned']['status']}**" for t in tags]]))
    L.append("")
    if len(tags) > 1:
        L.append("## Across seeds: direction agreement")
        L.append("")
        L.append(f"Seeds present: {X['n_seeds']} of 3." + ("" if X["three_seeds"] else " A direction counts only when all three seeds agree, so every "
                                                            "agreement below is provisional."))
        L.append(f"Rule learned: PASS in {X['rule_learned_pass_seeds']} of {X['n_seeds']} seeds; "
                 f"{'all three seeds pass' if X['rule_learned_all_seeds'] else 'not established across seeds'}.")
        L.append("")
        L.append(table(["metric", "good direction"] + tags + ["signs", "all seeds agree"],
                       [[m["metric"], m["good_direction"]] + [f3(m["values"][t], 4 if "KL" in m["metric"] else 3, signed="KL" not in m["metric"])
                                                              for t in tags] + [m["signs"], m["agree"]] for m in X["metrics"]]))
        L.append("")
    for t in tags:
        L.append(render_seed(t, results[t]))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main


def run(tags: list[str], runs_root: Path, out_md: Path) -> dict:
    S = Static()
    results = {}
    for t in tags:
        d = runs_root / t
        if not d.is_dir():
            raise SystemExit(f"run dir not found: {d}")
        results[t] = analyse_seed(d, S)
    meta = {"script_sha256": sha256(Path(__file__)), "runs_root": str(runs_root), "tags": tags, "data_sha256": S.sha,
            "thresholds": {"targeted_on": TH_TARGETED, "untouched_drop": TH_UNTOUCHED_DROP, "chain_k4": TH_CHAIN_K4,
                           "regret_drop": TH_REGRET_DROP, "g4": "point estimate ON <= OFF per family",
                           "g5_rule_strata": RULE_STRATA, "untouched_strata": UNTOUCHED_STRATA},
            "bootstrap": {"B": BOOT, "seed": SEED},
            "numeric_reply_classes": sorted(NUMERIC_CLASSES), "torch_imported": "torch" in sys.modules}
    meta["across"] = across(results)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(results, meta), encoding="utf-8")
    out_json = out_md.with_suffix(".json")
    out_json.write_text(json.dumps(clean({"meta": meta, "results": results}), indent=1) + "\n", encoding="utf-8")
    return {"meta": meta, "results": results, "md": out_md, "json": out_json}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tags", required=True, help="comma-separated run tags, e.g. t2_s0,t2_s1,t2_s2")
    ap.add_argument("--runs-root", default=str(ROOT / "runs" / "t2"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "t2_analysis" / "T2_RESULTS.md"))
    a = ap.parse_args()
    tags = [t.strip() for t in a.tags.split(",") if t.strip()]
    out = run(tags, Path(a.runs_root).resolve(), Path(a.out).resolve())
    for t, r in out["results"].items():
        print(f"[t2_analysis] {t}: " + ", ".join(f"{g.split('_')[0]} {r['gates'][g]['status']}" for g in GATE_NAMES)
              + f" | rule learned {r['gates']['rule_learned']['status']}")
    print(f"[t2_analysis] wrote {out['md']} and {out['json']}")


if __name__ == "__main__":
    main()
