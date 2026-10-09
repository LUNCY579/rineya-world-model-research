"""Retention suite analysis (roadmap item 4 / joint 5c). CPU only, numpy only; never imports torch.

  python -B scripts/ret_analysis.py dev-select --run runs/retention/dev_<seed>
        Supa's dev accuracies -> the dev selection of the run's surface-diversity setting (DEV_SELECTION.json or
        DEV_SELECTION_SD.json, create-only). Refused once a confirmation draw of that setting exists.
  python -B scripts/ret_analysis.py analyze --run runs/retention/confirm_<seed>
        The registered analysis of one sealed confirmation run -> <run>/analysis/RET_RESULTS.json + .md (create-only;
        run it once). Refuses unless the kit freeze, the draw seal, the dev selection and every recorded output hash
        verify.
  python -B scripts/ret_analysis.py oc
        Prints the operating characteristics of the go rule (seeded simulation; the table in DESIGN.md section 7).
  python -B scripts/ret_analysis.py dryrun [--record]
        Synthetic fixtures (a test-root draw, generated answers with planted losses, planted format gains, planted KL):
        every pre-stated verdict, the Holm step, the two graders, the pair rule, the binding count, the dev selection
        rule, the AUROC and the operating characteristics are checked against hand-computed expectations. --record
        writes docs/retention/DRYRUN.json (refused once frozen).

The pre-stated rules are RULE_DEV and RULE_CONFIRM below (verbatim in docs/retention/DESIGN.md section 7).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
if "torch" in sys.modules:
    raise SystemExit("ret_analysis.py must run without torch (CPU-only analysis)")
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import ret_common as RC  # noqa: E402
import ret_gen as RGEN  # noqa: E402
import ret_grade as RG  # noqa: E402

RULE_DEV = (
    "For each category c and level l in {1, 2, 3}, a(c, l) is Supa 1.7's primary-grader accuracy on the 75 items of "
    "that cell of the dev draw of a surface-diversity setting (dev seed 20261121 with surface diversity off, 20261125 "
    "with it on; an unknown_twin item counts as correct only if both twins are). The selected level of c is the "
    "highest l with a(c, l) >= 0.70; if no level reaches 0.70, it is the highest l with a(c, l) >= 0.40. A category "
    "with no level at 0.40 or above is EXCLUDED from every confirmation draw of that setting (reported, never gated). "
    "The selection is written once per setting (DEV_SELECTION.json for off, DEV_SELECTION_SD.json for on; "
    "create-only) before any confirmation draw of that setting is minted, and every confirmation draw uses the "
    "selection of its own setting.")
RULE_CONFIRM = (
    "For each gating comparison of an arm A with its reference R (T2 s0 ON vs adapter OFF = Titan-Sparse; each extra "
    "Titan-Sparse adapter vs OFF; each extra Supa adapter vs Supa 1.7) on a sealed confirmation draw minted after the "
    "dev selection of its surface-diversity setting, for each of the two graders g (the primary typed grader and the "
    "lenient extraction grader) and each selected category c (75 items): lost = items R answers correctly and A does "
    "not; gained = items A answers correctly and R does not; net = gained - lost (an unknown_twin item is correct only "
    "if both twins are). p = two-sided exact McNemar p-value (binomial on the lost + gained discordant items, 1/2); "
    "p_Holm = Holm-Bonferroni adjustment of p over the selected categories of that comparison and grader. Category c "
    "has a DETECTED LOSS under g iff lost > gained and p_Holm < 0.05, and EXCEEDS ITS MARGIN under g iff net < -11 "
    "(more than 11 of its 75 items lost net). The pooled net under g is the sum of net over the selected categories; "
    "it EXCEEDS ITS MARGIN iff it is below -0.04 x the number of selected items (-24 for 600 items). Comparison "
    "verdict: LOSS DETECTED iff some category has a detected loss under either grader; otherwise MARGIN EXCEEDED iff "
    "some category or the pooled net exceeds its margin under either grader; otherwise RETAINED. GO (the retention "
    "guard passes for arm A) iff RETAINED, reported as GO (k/m binding), where k of the m selected categories are "
    "binding, i.e. R answers at least the fewest items whose loss with no gain would fail the category alone (9 when "
    "8 categories are selected); LOSS DETECTED and MARGIN EXCEEDED are NO-GO. Report-only, never gating: the "
    "per-category flag net < -5 of 75 (a point estimate, not a non-inferiority test), Titan-Sparse (OFF) vs Supa 1.7 "
    "and T2 s0 vs Supa "
    "1.7 (core and cumulative drift, verdicts computed the same way), excluded categories, accuracies with Wilson 95% "
    "intervals, EOS-termination rates, abstention components, the operating characteristics, and the AUROC of the "
    "per-item first-token KL(R || A) for predicting correctness flips. The suite is templated and narrow: a RETAINED "
    "verdict is a floor, not a proof of general retention.")
GRADERS = ("primary", "lenient")
BOOT, BOOT_SEED = 2000, 20261122
DRY_SEED = 20261123                  # dry-run draws live in a temporary root only (confirm: DRY_SEED + 1)
OC_SEED, OC_REPS = 20261126, 40000


# ------------------------------------------------------------------------------------------------ statistics
def mcnemar_p(lost: int, gained: int) -> float:
    n = lost + gained
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(lost, gained) + 1)) / 2 ** n)


def mcnemar(ref: dict, arm: dict) -> dict:
    keys = sorted(ref)
    if set(keys) != set(arm):
        raise ValueError("item ids differ between reference and arm")
    lost = sum(1 for k in keys if ref[k] and not arm[k])
    gained = sum(1 for k in keys if not ref[k] and arm[k])
    return {"n": len(keys), "ref_correct": sum(bool(ref[k]) for k in keys), "arm_correct": sum(bool(arm[k]) for k in keys),
            "lost": lost, "gained": gained, "net": gained - lost, "discordant": lost + gained,
            "p": mcnemar_p(lost, gained)}


def holm(ps: dict) -> dict:
    """Holm-Bonferroni step-down adjusted p-values for {name: p}."""
    items = sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def min_lost_to_detect(m: int) -> int:
    """Fewest losses with no gain that the detection clause can flag among m categories (the first Holm step:
    m x 2^(1-k) < ALPHA). 9 for m = 8."""
    k = 1
    while not (m * 2.0 ** (1 - k) < RC.ALPHA):
        k += 1
    return k


def min_lost_to_fail(m: int) -> int:
    """Fewest losses with no gain that fail a category alone: by detection or by the category margin."""
    return min(min_lost_to_detect(m), RC.CATEGORY_MARGIN + 1)


def pooled_margin(n_items: int) -> float:
    return -RC.POOLED_MARGIN_RATE * n_items


def category_verdict(rec: dict, p_holm: float, m: int) -> dict:
    k = min_lost_to_fail(m)
    det = rec["lost"] > rec["gained"] and p_holm < RC.ALPHA
    over = rec["net"] < -RC.CATEGORY_MARGIN
    return {**rec, "p_holm": p_holm, "detected_loss": det, "margin_exceeded": over, "fails": det or over,
            "flag_net_below_minus5": rec["net"] < -RC.MARGIN,
            "binding": {"min_lost_to_fail": k, "binding": rec["ref_correct"] >= k}}


def grader_verdict(per_cat: dict) -> dict:
    """One grader: Holm over the selected categories, detected losses, category margins, the pooled margin."""
    m = len(per_cat)
    adj = holm({c: r["p"] for c, r in per_cat.items()})
    cats = {c: category_verdict(r, adj[c], m) for c, r in per_cat.items()}
    n_items = sum(r["n"] for r in per_cat.values())
    pooled = {"lost": sum(c["lost"] for c in cats.values()), "gained": sum(c["gained"] for c in cats.values()),
              "net": sum(c["net"] for c in cats.values()), "n_items": n_items, "margin": pooled_margin(n_items)}
    pooled["margin_exceeded"] = pooled["net"] < pooled["margin"]
    return {"categories": cats, "pooled": pooled, "detected": any(c["detected_loss"] for c in cats.values()),
            "category_margin_exceeded": sorted(c for c, r in cats.items() if r["margin_exceeded"]),
            "margin_exceeded": pooled["margin_exceeded"] or any(c["margin_exceeded"] for c in cats.values())}


def decide(per_grader: dict) -> dict:
    """{grader: {category: mcnemar record}} (both graders over the same categories) -> RULE_CONFIRM's verdict."""
    gv = {g: grader_verdict(pc) for g, pc in per_grader.items()}
    if any(v["detected"] for v in gv.values()):
        verdict = "LOSS DETECTED"
    elif any(v["margin_exceeded"] for v in gv.values()):
        verdict = "MARGIN EXCEEDED"
    else:
        verdict = "RETAINED"
    prim = gv["primary"] if "primary" in gv else next(iter(gv.values()))
    binding = [c for c, r in prim["categories"].items() if r["binding"]["binding"]]
    m = len(prim["categories"])
    go = verdict == "RETAINED"
    return {"graders": gv, "verdict": verdict, "go": go,
            "binding": {"k": len(binding), "m": m, "categories": binding, "min_lost_to_fail": min_lost_to_fail(m)},
            "go_label": f"GO ({len(binding)}/{m} binding)" if go else "NO-GO",
            "flags_net_below_minus5": {g: sorted(c for c, r in v["categories"].items() if r["flag_net_below_minus5"])
                                       for g, v in gv.items()}}


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(score, label) -> float:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _rank(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auroc_ci(score, label, B: int = BOOT, seed: int = BOOT_SEED) -> dict:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    a = auroc(s, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, len(s), len(s))
        if y[idx].all() or (~y[idx]).all():
            continue
        bs.append(auroc(s[idx], y[idx]))
    ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if bs else [float("nan"), float("nan")]
    return {"auroc": a, "ci95": ci, "n": len(s), "n_pos": int(y.sum()), "boot_used": len(bs)}


def _lsm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def kl_rows(P, Q, chunk: int = 8) -> np.ndarray:
    """Per-row KL(P || Q) of renormalised stored first-token log-probs (t2_analysis.lp_compare's arithmetic)."""
    n = P.shape[0]
    kl = np.zeros(n)
    for s in range(0, n, chunk):
        a, b = _lsm(np.asarray(P[s:s + chunk])), _lsm(np.asarray(Q[s:s + chunk]))
        kl[s:s + chunk] = (np.exp(a) * (a - b)).sum(1)
    return np.maximum(kl, 0.0)


# ------------------------------------------------------------------------------------------------ operating characteristics
def _p_table(nmax: int = RC.N_PER_CELL) -> np.ndarray:
    P = np.ones((nmax + 1, nmax + 1))
    for lo in range(nmax + 1):
        for g in range(nmax + 1):
            P[lo, g] = mcnemar_p(lo, g)
    return P


def _holm_vec(p: np.ndarray) -> np.ndarray:
    m = p.shape[1]
    order = np.argsort(p, axis=1, kind="stable")
    ps = np.take_along_axis(p, order, 1)
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(ps * (m - np.arange(m)), axis=1))
    adj = np.empty_like(p)
    np.put_along_axis(adj, order, adj_sorted, 1)
    return adj


def rules_vec(lost: np.ndarray, gained: np.ndarray, P: np.ndarray, n: int = RC.N_PER_CELL) -> dict:
    """One grader, many replicates (rows) x m categories: the go rule's NO-GO and its clauses, and the superseded
    draft rule (category retained iff net >= -5 and no detected loss) for comparison."""
    p = P[lost, gained]
    adj = _holm_vec(p)
    det_cat = (lost > gained) & (adj < RC.ALPHA)
    net = gained - lost
    det = det_cat.any(1)
    cat = (net < -RC.CATEGORY_MARGIN).any(1)
    pooled = net.sum(1) < pooled_margin(n * lost.shape[1])
    return {"nogo": det | cat | pooled, "detected": det, "category_margin": cat, "pooled": pooled,
            "flag_any": (net < -RC.MARGIN).any(1), "old_nogo": ((net < -RC.MARGIN) | det_cat).any(1)}


def simulate(acc: float, D: float, extra_one: int = 0, extra_all: int = 0, m: int = 8, n: int = RC.N_PER_CELL,
             reps: int = OC_REPS, seed: int = OC_SEED, P: np.ndarray | None = None) -> dict:
    """Reference accuracy acc in every category; background churn D expected discordant items per category, balanced
    (E lost = E gained = D/2: no real effect); plus a real loss of extra_one items in one category and extra_all in
    every category (those items are lost for certain)."""
    P = _p_table(n) if P is None else P
    rng = np.random.default_rng(seed)
    rc = int(round(acc * n))
    ql, qg = (D / 2) / rc, (D / 2) / (n - rc)
    extra = np.full(m, extra_all)
    extra[0] += extra_one
    lost = rng.binomial(rc - extra, ql, size=(reps, m)) + extra
    gained = rng.binomial(n - rc, qg, size=(reps, m))
    r = rules_vec(lost, gained, P, n)
    return {k: round(float(v.mean()), 4) for k, v in r.items()}


OC_D = (3.75, 7.5, 11.25, 15.0, 22.5)          # 5, 10, 15, 20, 30 % of 75 items discordant per category
OC_ACC = (0.30, 0.45, 0.80)


def operating_characteristics(reps: int = OC_REPS) -> dict:
    """P(NO-GO) of ONE grader with 8 selected categories x 75 items (seeded; DESIGN.md section 7). Two graders: the
    false NO-GO is at most twice the single-grader value (union bound) and equals it when they agree item by item."""
    P = _p_table()
    null = {f"acc{a}_D{D}": simulate(a, D, reps=reps, P=P) for a in OC_ACC for D in OC_D}
    conc = {f"D{D}_k{k}": simulate(0.80, D, extra_one=k, reps=reps, P=P) for D in (7.5, 15.0) for k in (6, 9, 12, 15)}
    diff = {f"D{D}_j{j}": simulate(0.80, D, extra_all=j, reps=reps, P=P) for D in (7.5, 15.0) for j in (2, 3, 4, 5)}
    return {"reps": reps, "seed": OC_SEED, "m": 8, "n": RC.N_PER_CELL, "null": null,
            "null_worst_single_grader": max(v["nogo"] for v in null.values()),
            "null_worst_two_graders_union_bound": round(min(1.0, 2 * max(v["nogo"] for v in null.values())), 4),
            "concentrated_loss_one_category": conc, "diffuse_loss_every_category": diff,
            "note": "single grader; keys: nogo (the go rule), detected / category_margin / pooled (its clauses), "
                    "flag_any (some category net < -5, report-only), old_nogo (the superseded draft rule)"}


def oc_markdown(oc: dict) -> str:
    L = ["| background discordance per category | ref acc | P(NO-GO) go rule | via detection | via category margin | "
         "via pooled margin | superseded draft rule |", "|---|---|---|---|---|---|---|"]
    for a in OC_ACC:
        for D in OC_D:
            r = oc["null"][f"acc{a}_D{D}"]
            L.append(f"| {D:g} ({D / 75:.0%}) | {a:.2f} | {r['nogo']:.3f} | {r['detected']:.3f} | "
                     f"{r['category_margin']:.3f} | {r['pooled']:.3f} | {r['old_nogo']:.3f} |")
    L += ["", "| real loss (ref acc 0.80) | background 7.5 | background 15 |", "|---|---|---|"]
    for k in (6, 9, 12, 15):
        a, b = oc["concentrated_loss_one_category"][f"D7.5_k{k}"], oc["concentrated_loss_one_category"][f"D15.0_k{k}"]
        L.append(f"| {k} items in ONE category | {a['nogo']:.3f} (draft {a['old_nogo']:.3f}) | "
                 f"{b['nogo']:.3f} (draft {b['old_nogo']:.3f}) |")
    for j in (2, 3, 4, 5):
        a, b = oc["diffuse_loss_every_category"][f"D7.5_j{j}"], oc["diffuse_loss_every_category"][f"D15.0_j{j}"]
        L.append(f"| {j} items in EVERY category ({8 * j} pooled) | {a['nogo']:.3f} (draft {a['old_nogo']:.3f}) | "
                 f"{b['nogo']:.3f} (draft {b['old_nogo']:.3f}) |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ grading a run
def grade_side(rows: list, gens: list) -> dict:
    """{pid: {primary, lenient, terminated}} for one side; gens must cover exactly the draw's prompts."""
    by = {g["pid"]: g for g in gens}
    if set(by) != {r["pid"] for r in rows} or len(by) != len(gens):
        raise ValueError("generations do not cover exactly the draw's prompts")
    out = {}
    for r in rows:
        g = by[r["pid"]]
        out[r["pid"]] = {"primary": RG.grade(r["gold"], g["text"], bool(g["terminated"]))["correct"],
                         "lenient": RG.grade_lenient(r["gold"], g["text"]), "terminated": bool(g["terminated"]),
                         "text_is_unknown": RG.grade({"kind": "unknown"}, g["text"], True)["correct"]}
    return out


def item_correct(rows: list, graded: dict, key: str = "primary") -> dict:
    """{category: {item: bool}}: an item is correct iff every prompt of it is (the unknown_twin pair rule)."""
    acc = defaultdict(dict)
    for r in rows:
        ok = graded[r["pid"]][key]
        acc[r["category"]][r["item"]] = acc[r["category"]].get(r["item"], True) and ok
    return dict(acc)


def side_summary(rows: list, graded: dict) -> dict:
    prim, len_ = item_correct(rows, graded), item_correct(rows, graded, "lenient")
    out = {}
    for c in prim:
        k, n = sum(prim[c].values()), len(prim[c])
        out[c] = {"correct": k, "n": n, "acc": round(k / n, 4), "wilson95": wilson(k, n),
                  "lenient_correct": sum(len_[c].values())}
    out["_termination_rate"] = round(sum(graded[r["pid"]]["terminated"] for r in rows) / len(rows), 4)
    unk_u = [r for r in rows if r["category"] == "unknown_twin" and r["twin"] == "u"]
    unk_a = [r for r in rows if r["category"] == "unknown_twin" and r["twin"] == "a"]
    if unk_u:
        out["_abstention"] = {
            "unanswerable_literal_UNKNOWN": round(sum(graded[r["pid"]]["primary"] for r in unk_u) / len(unk_u), 4),
            "unanswerable_abstains_lenient": round(sum(graded[r["pid"]]["lenient"] for r in unk_u) / len(unk_u), 4),
            "answerable_correct": round(sum(graded[r["pid"]]["primary"] for r in unk_a) / len(unk_a), 4),
            "answerable_over_abstention": round(sum(graded[r["pid"]]["text_is_unknown"] for r in unk_a) / len(unk_a), 4)}
    return out


# ------------------------------------------------------------------------------------------------ loading a run
def load_run(run_dir: Path, root: Path, registered: bool) -> dict:
    run_dir, root = Path(run_dir), Path(root)
    prov = RC.read_json(run_dir / "provenance.json")
    done = RC.read_json(run_dir / "DONE.json")
    RC.need(prov is not None and done is not None, f"{run_dir}: provenance.json / DONE.json missing (run incomplete)")
    RC.need(RC.sha256_file(run_dir / "provenance.json") == done.get("provenance_sha256"), "provenance.json changed after DONE")
    for st, files in (done.get("files") or {}).items():
        for rel, h in files.items():
            RC.need((run_dir / rel).is_file() and RC.sha256_file(run_dir / rel) == h, f"{rel} differs from DONE.json")
            RC.need((prov["stages"][st]["files"] or {}).get(rel) == h, f"{rel} differs from provenance")
    draw_p = root / "docs" / "retention" / "draws" / f"{prov['draw_id']}.jsonl"
    rows, seal = RGEN.load_draw(draw_p)
    RC.need(seal["file_sha256"] == prov["draw_sha256"], "the run's draw hash differs from the sealed draw")
    if registered:
        ok, errs, _ = RC.verify_freeze(root)
        RC.need(ok, "kit freeze does not verify: " + "; ".join(errs[:3]))
        fz = RC.sha256_file(root / "docs" / "retention" / "RET_FREEZE.json")
        RC.need(prov.get("registered") is True and prov.get("freeze_sha256") == fz and seal.get("freeze_sha256") == fz,
                "the run or its draw is not bound to the current kit freeze")
        RC.need((seal.get("freshness") or {}).get("ok") is True, "the draw's freshness scan was skipped or failed")
        RC.need((prov.get("tokenizer") or {}).get("files") == RC.TOKENIZER_PINS,
                "the run did not record the pinned tokenizer files")
    gens, lps = {}, {}
    for st, sides in prov["sides"].items():
        d = run_dir / {"SPARSE": "sparse", "SUPA": "supa"}[st]
        for s in sides:
            gens[s] = RC.read_jsonl(d / f"gen_{s}.jsonl")
            if prov.get("lp"):
                lps[s] = np.load(d / f"lp_{s}.npy", mmap_mode="r")
    return {"prov": prov, "seal": seal, "rows": rows, "gens": gens, "lps": lps}


def comparisons_for(sides: list) -> list:
    """(name, arm, ref, gating). Titan-Sparse vs Supa and T2 s0 vs Supa are report-only (core / cumulative drift)."""
    out = []
    if "t2s0" in sides and "off" in sides:
        out.append(("t2s0_vs_off", "t2s0", "off", True))
    for s in sides:
        if s.startswith("x_"):
            out.append((f"{s}_vs_off", s, "off", True))
        if s.startswith("xs_"):
            out.append((f"{s}_vs_supa", s, "supa", True))
    if "off" in sides and "supa" in sides:
        out.append(("sparse_vs_supa", "off", "supa", False))
    if "t2s0" in sides and "supa" in sides:
        out.append(("t2s0_vs_supa", "t2s0", "supa", False))
    return out


def kl_block(rows: list, lps: dict, arm: str, ref: str, ref_items: dict, arm_items: dict) -> dict:
    """Report-only: per-item KL(R || A) (mean over the item's prompts) and its AUROC for flips / losses."""
    if arm not in lps or ref not in lps:
        return {"available": False}
    kl = kl_rows(lps[ref], lps[arm])
    per_item = defaultdict(list)
    for i, r in enumerate(rows):
        per_item[r["item"]].append(kl[i])
    items = [it for it in per_item if it in ref_items]
    score = np.array([float(np.mean(per_item[it])) for it in items])
    flip = np.array([ref_items[it] != arm_items[it] for it in items])
    ref_ok = np.array([ref_items[it] for it in items])
    loss = np.array([ref_items[it] and not arm_items[it] for it in items])
    return {"available": True, "n_items": len(items), "kl_mean": float(score.mean()), "kl_median": float(np.median(score)),
            "kl_mean_flipped": float(score[flip].mean()) if flip.any() else None,
            "kl_mean_unflipped": float(score[~flip].mean()) if (~flip).any() else None,
            "auroc_flip": auroc_ci(score, flip),
            "auroc_loss_among_ref_correct": auroc_ci(score[ref_ok], loss[ref_ok]) if ref_ok.any() else None}


def analyze_run(run_dir: Path, root: Path, registered: bool) -> dict:
    R = load_run(run_dir, root, registered)
    rows, prov = R["rows"], R["prov"]
    RC.need(prov["mode"] == "confirm", "analyze needs a confirmation run (dev runs feed dev-select)")
    sides = [s for v in prov["sides"].values() for s in v]
    graded = {s: grade_side(rows, R["gens"][s]) for s in sides}
    items = {s: {g: item_correct(rows, graded[s], g) for g in GRADERS} for s in sides}
    cats = sorted({r["category"] for r in rows}, key=RC.CATEGORIES.index)
    res = {"rule": RULE_CONFIRM, "draw_id": prov["draw_id"], "draw_sha256": prov["draw_sha256"],
           "selection_sha256": R["seal"].get("selection_sha256"), "surface_diversity": R["seal"].get("surface_diversity"),
           "registered": registered, "categories": cats, "levels": {c: R["seal"]["plan"][c] for c in cats},
           "accuracy": {s: side_summary(rows, graded[s]) for s in sides}, "comparisons": {}}
    for name, arm, ref, gating in comparisons_for(sides):
        d = decide({g: {c: mcnemar(items[ref][g][c], items[arm][g][c]) for c in cats} for g in GRADERS})
        d.update(arm=arm, reference=ref, gating=gating,
                 kl=kl_block(rows, R["lps"], arm, ref, {k: v for c in cats for k, v in items[ref]["primary"][c].items()},
                             {k: v for c in cats for k, v in items[arm]["primary"][c].items()}))
        res["comparisons"][name] = d
    res["go"] = {n: c["go"] for n, c in res["comparisons"].items() if c["gating"]}
    res["go_label"] = {n: c["go_label"] for n, c in res["comparisons"].items() if c["gating"]}
    return res


def render_md(res: dict) -> str:
    L = [f"# Retention suite results: {res['draw_id']}", "",
         f"Draw sha256 `{res['draw_sha256']}`; selection sha256 `{res['selection_sha256']}`; surface diversity "
         f"{res.get('surface_diversity')}; registered: {res['registered']}.", "", "## Pre-stated rule", "",
         "> " + res["rule"], "", "## Verdicts", "",
         "| comparison | gating | verdict | GO | pooled net primary / lenient (margin) | categories over -11 | "
         "flags net < -5 (report-only) |", "|---|---|---|---|---|---|---|"]
    for n, c in res["comparisons"].items():
        pp, pl = c["graders"]["primary"]["pooled"], c["graders"]["lenient"]["pooled"]
        flags = sorted(set(c["flags_net_below_minus5"]["primary"]) | set(c["flags_net_below_minus5"]["lenient"]))
        over = sorted(set(c["graders"]["primary"]["category_margin_exceeded"])
                      | set(c["graders"]["lenient"]["category_margin_exceeded"]))
        L.append(f"| {n} ({c['arm']} vs {c['reference']}) | {'yes' if c['gating'] else 'report-only'} | {c['verdict']} | "
                 f"{c['go_label']} | {pp['net']:+d} / {pl['net']:+d} ({pp['margin']:g}) | {', '.join(over) or '-'} | "
                 f"{', '.join(flags) or '-'} |")
    for n, c in res["comparisons"].items():
        for g in GRADERS:
            gv = c["graders"][g]
            L += ["", f"### {n}: {g} grader", "",
                  "| category | level | ref | arm | lost | gained | net | p | p_Holm | detected loss | net < -11 | "
                  "binding |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for cat, r in gv["categories"].items():
                L.append(f"| {cat} | {res['levels'][cat]} | {r['ref_correct']}/{r['n']} | {r['arm_correct']}/{r['n']} | "
                         f"{r['lost']} | {r['gained']} | {r['net']:+d} | {r['p']:.4g} | {r['p_holm']:.4g} | "
                         f"{'YES' if r['detected_loss'] else 'no'} | {'YES' if r['margin_exceeded'] else 'no'} | "
                         f"{'yes' if r['binding']['binding'] else 'no'} |")
        k = c["kl"]
        if k.get("available"):
            L.append("")
            L.append(f"Report-only: first-token KL({c['reference']} || {c['arm']}) mean {k['kl_mean']:.4f}; AUROC for "
                     f"correctness flips {k['auroc_flip']['auroc']:.3f} {k['auroc_flip']['ci95']} "
                     f"(n {k['auroc_flip']['n']}, flips {k['auroc_flip']['n_pos']}).")
    L += ["", "## Accuracy (primary grader; report-only)", "", "| side | " + " | ".join(res["categories"]) +
          " | EOS rate |", "|---" * (len(res["categories"]) + 2) + "|"]
    for s, a in res["accuracy"].items():
        L.append(f"| {s} | " + " | ".join(f"{a[c]['acc']:.2f} ({a[c]['lenient_correct']})" for c in res["categories"])
                 + f" | {a['_termination_rate']:.3f} |")
    L += ["", "Cells: primary accuracy (lenient-grader correct count).", ""]
    return "\n".join(L) + "\n"


def cmd_analyze(run_dir: Path, root: Path = ROOT) -> dict:
    out = Path(run_dir) / "analysis"
    RC.need(not (out / "RET_RESULTS.json").exists(), "RET_RESULTS.json exists: the registered analysis runs once")
    res = analyze_run(run_dir, root, registered=True)
    sel = root / RC.selection_rel(bool(res["surface_diversity"]))
    RC.need(sel.is_file() and RC.sha256_file(sel) == res["selection_sha256"], f"{sel.name} differs from the draw's")
    res["code_sha256"] = RC.code_sha(root)
    res["analysed_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    RC.write_json_create(out / "RET_RESULTS.json", res)
    RC.write_create(out / "RET_RESULTS.md", render_md(res))
    return res


# ------------------------------------------------------------------------------------------------ dev selection
def select_levels(acc: dict) -> tuple[dict, list, dict]:
    """RULE_DEV on {category: {level: {"correct", "n"}}} -> (selected, excluded, basis)."""
    selected, excluded, basis = {}, [], {}
    for c in RC.CATEGORIES:
        lv = acc.get(c, {})
        hi = [l for l in sorted(lv) if lv[l]["correct"] >= RC.DEV_TARGET_ACC * lv[l]["n"] - 1e-9]
        lo = [l for l in sorted(lv) if lv[l]["correct"] >= RC.DEV_MIN_ACC * lv[l]["n"] - 1e-9]
        if hi:
            selected[c], basis[c] = max(hi), f">= {RC.DEV_TARGET_ACC}"
        elif lo:
            selected[c], basis[c] = max(lo), f">= {RC.DEV_MIN_ACC} (no level reaches {RC.DEV_TARGET_ACC})"
        else:
            excluded.append(c)
    return selected, excluded, basis


def dev_select(run_dir: Path, root: Path = ROOT, registered: bool = True, write: bool = True) -> dict:
    root = Path(root)
    draws = root / "docs" / "retention" / "draws"
    prov0 = RC.read_json(Path(run_dir) / "provenance.json") or {}
    sd = bool((RC.read_json(draws / f"{prov0.get('draw_id')}.SEAL.json") or {}).get("surface_diversity"))
    confirm = sorted(p.name for p in draws.glob("confirm_*.SEAL.json")
                     if bool((RC.read_json(p) or {}).get("surface_diversity")) == sd)
    RC.need(not confirm, f"confirmation draws of this setting already exist ({confirm}): the dev selection must "
                         "precede them")
    R = load_run(run_dir, root, registered)
    RC.need(R["prov"]["mode"] == "dev" and R["seal"]["purpose"] == "dev", "dev-select needs a dev run")
    RC.need(bool(R["seal"]["surface_diversity"]) == sd, "the run's draw changed while selecting")
    if registered:
        RC.need(R["seal"]["seed"] == RC.DEV_SEEDS[sd], "the dev run must use the pre-registered dev seed of its setting")
    rows = R["rows"]
    graded = grade_side(rows, R["gens"]["supa"])
    acc = defaultdict(dict)
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["category"], r["level"])].append(r)
    for (c, l), rr in sorted(by_cell.items()):
        ic = item_correct(rr, graded)[c]
        il = item_correct(rr, graded, "lenient")[c]
        k, n = sum(ic.values()), len(ic)
        acc[c][l] = {"correct": k, "n": n, "acc": round(k / n, 4), "wilson95": wilson(k, n), "lenient_correct": sum(il.values())}
    selected, excluded, basis = select_levels(acc)
    res = {"rule": RULE_DEV, "dev_draw": R["seal"]["draw_id"], "dev_draw_sha256": R["seal"]["file_sha256"],
           "dev_run_done_sha256": RC.sha256_file(Path(run_dir) / "DONE.json"), "surface_diversity": sd,
           "accuracy": acc, "selected": selected, "basis": basis, "excluded": excluded,
           "freeze_sha256": R["seal"].get("freeze_sha256"), "registered": registered,
           "abstention": side_summary(rows, graded).get("_abstention"), "selected_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    if write:
        RC.write_json_create(root / RC.selection_rel(sd), res)
    return res


# ------------------------------------------------------------------------------------------------ dry run
def _fake_run(root: Path, draw_p: Path, sides: dict, texts: dict, lps: dict | None, mode: str) -> Path:
    """A DONE run directory in the layout ret_run.py writes, from given per-side texts (and log-prob arrays)."""
    rows, seal = RGEN.load_draw(draw_p)
    run = root / "runs" / "retention" / seal["draw_id"]
    prov = {"draw_id": seal["draw_id"], "draw_sha256": seal["file_sha256"], "mode": mode, "registered": False,
            "freeze_sha256": None, "sides": sides, "lp": lps is not None, "stages": {}, "turns": [],
            "tokenizer": {"files": dict(RC.TOKENIZER_PINS)}}
    for st, ss in sides.items():
        d = run / {"SPARSE": "sparse", "SUPA": "supa"}[st]
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        for s in ss:
            p = d / f"gen_{s}.jsonl"
            p.write_text("".join(json.dumps({"pid": r["pid"], "text": t, "terminated": True, "n_new": 3,
                                             "first_token": 0, "first_token_prefill": 0, "lp_row": i}) + "\n"
                                 for i, (r, t) in enumerate(zip(rows, texts[s]))), encoding="utf-8")
            files[str(p.relative_to(run)).replace("\\", "/")] = RC.sha256_file(p)
            if lps is not None:
                q = d / f"lp_{s}.npy"
                np.save(q, lps[s])
                files[str(q.relative_to(run)).replace("\\", "/")] = RC.sha256_file(q)
        prov["stages"][st] = {"files": files}
    (run / "provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    done = {"files": {st: prov["stages"][st]["files"] for st in sides},
            "provenance_sha256": RC.sha256_file(run / "provenance.json")}
    (run / "DONE.json").write_text(json.dumps(done), encoding="utf-8")
    return run


def _wrong(g: dict) -> str:
    return "UNKNOWN" if g["kind"] != "unknown" else str((g.get("passage_values") or ["x"])[0])


def _verbose(g: dict) -> str:
    """Right value, wrong format: the primary grader rejects it, the lenient grader accepts it."""
    return f"I think the answer is {RG.canonical_answer(g)}, as asked."


def dryrun(oc_reps: int = OC_REPS) -> dict:
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)[:300]})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""), flush=True)

    design = RC.DESIGN.read_text(encoding="utf-8") if RC.DESIGN.is_file() else ""
    check("DESIGN.md states RULE_DEV and RULE_CONFIRM verbatim", RULE_DEV in design and RULE_CONFIRM in design)
    # 1 statistics on hand-computed inputs
    check("McNemar 6 lost / 0 gained: p = 2/64", abs(mcnemar({i: True for i in range(6)}, {i: False for i in range(6)})["p"]
                                                     - 2 / 64) < 1e-12)
    check("McNemar 9 lost / 4 gained: p = 2 * P(X<=4 | 13)", abs(
        mcnemar({**{i: True for i in range(9)}, **{i: False for i in range(9, 13)}},
                {**{i: False for i in range(9)}, **{i: True for i in range(9, 13)}})["p"]
        - 2 * sum(math.comb(13, i) for i in range(5)) / 2 ** 13) < 1e-12)
    h = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    check("Holm step-down", h == {"a": 0.04, "c": 0.09, "b": 0.09, "d": 0.5}, h)
    check("fewest losses a Holm step can detect: 9 of 8 categories, 8 of 6, 6 of 1",
          (min_lost_to_detect(8), min_lost_to_detect(6), min_lost_to_detect(1)) == (9, 8, 6),
          (min_lost_to_detect(8), min_lost_to_detect(6), min_lost_to_detect(1)))
    check("pooled margin: -24 for 600 items, -21 for 525", pooled_margin(600) == -24 and pooled_margin(525) == -21)
    check("fewest losses that fail a category alone: 9 of 8 categories (detection), 12 is the category margin",
          min_lost_to_fail(8) == 9 and min_lost_to_fail(1) == 6 and RC.CATEGORY_MARGIN + 1 == 12)
    check("Wilson 30/75 = [0.2966, 0.5131] (hand-computed)", wilson(30, 75) == [0.2966, 0.5131], wilson(30, 75))
    check("AUROC perfect / inverted / ties", auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0 and
          auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0 and auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5)
    P = _p_table()
    old_dead = [(lo, g) for lo in range(76) for g in range(76) if lo > g and g - lo >= -RC.MARGIN and P[lo, g] < RC.ALPHA]
    check("why the draft rule was restated: no (lost, gained) with net >= -5 has raw p < 0.05, so its McNemar/Holm "
          "clause could never change GO (exhaustive over 0..75)", not old_dead, old_dead[:3])

    def rec(lost, gained, ref_correct=40, n=75):
        return {"n": n, "ref_correct": ref_correct, "arm_correct": ref_correct - lost + gained, "lost": lost,
                "gained": gained, "net": gained - lost, "discordant": lost + gained, "p": mcnemar_p(lost, gained)}

    def both(pc, pl=None):
        return {"primary": pc, "lenient": pl if pl is not None else pc}

    base = {c: rec(0, 0) for c in RC.CATEGORIES}
    scen = {
        "identical arms": (both(base), "RETAINED"),
        "one category net -6 (flag only; raw p 0.031, Holm 0.25)": (both({**base, "arith_word": rec(6, 0)}), "RETAINED"),
        "one category 8 lost (Holm 8 x 2^-7 = 0.0625): not detected, net -8 >= -11, pooled -8":
            (both({**base, "arith_word": rec(8, 0)}), "RETAINED"),
        "one category 9 lost (Holm 8 x 2^-8 = 0.031): detected": (both({**base, "kv_recall": rec(9, 0)}), "LOSS DETECTED"),
        "one category 14 lost / 2 gained (Holm 0.033): detected": (both({**base, "kv_recall": rec(14, 2)}), "LOSS DETECTED"),
        "one category 18 lost / 7 gained (net -11 = margin; Holm 0.35)": (both({**base, "date_time": rec(18, 7)}),
                                                                          "RETAINED"),
        "one category 19 lost / 7 gained (net -12 < -11; Holm 0.23)": (both({**base, "date_time": rec(19, 7)}),
                                                                       "MARGIN EXCEEDED"),
        "net -3 in every category (pooled -24 = margin)": (both({c: rec(3, 0) for c in RC.CATEGORIES}), "RETAINED"),
        "net -3 in seven categories, -4 in one (pooled -25 < -24)":
            (both({c: rec(4, 0) if i == 0 else rec(3, 0) for i, c in enumerate(RC.CATEGORIES)}), "MARGIN EXCEEDED"),
        "gains only (+10)": (both({**base, "pct_ratio": rec(0, 10)}), "RETAINED"),
        "format gains hide a skill loss: primary 12 lost / 12 gained, lenient 12 lost / 0 gained":
            (both({**base, "arith_word": rec(12, 12)}, {**base, "arith_word": rec(12, 0)}), "LOSS DETECTED"),
        "lenient pooled net -25, primary 0": (both(base, {c: rec(4, 0) if i == 0 else rec(3, 0)
                                                          for i, c in enumerate(RC.CATEGORIES)}), "MARGIN EXCEEDED"),
    }
    for name, (pg, want) in scen.items():
        got = decide(pg)["verdict"]
        check(f"rule: {name} -> {want}", got == want, got)
    d9 = decide(both({**base, "kv_recall": rec(9, 0)}))
    d12 = decide(both({**base, "date_time": rec(19, 7)}))
    dp = decide(both({c: rec(4, 0) if i == 0 else rec(3, 0) for i, c in enumerate(RC.CATEGORIES)}))
    check("each clause can change GO alone: detection (net -9), the category margin (net -12, no detection), the "
          "pooled margin (no detection, no category over -11)",
          not d9["go"] and not d9["graders"]["primary"]["margin_exceeded"]
          and not d12["go"] and not d12["graders"]["primary"]["detected"] and not d12["graders"]["primary"]["pooled"]["margin_exceeded"]
          and not dp["go"] and not dp["graders"]["primary"]["detected"]
          and not dp["graders"]["primary"]["category_margin_exceeded"])
    sub = {c: rec(0, 0) for c in ("arith_word", "kv_recall", "date_time")}
    sub["kv_recall"] = rec(7, 0)
    d = decide(both(sub))
    check("Holm over the SELECTED categories only (3): 7 lost -> p 0.0156 x 3 = 0.047 -> LOSS DETECTED",
          d["verdict"] == "LOSS DETECTED" and abs(d["graders"]["primary"]["categories"]["kv_recall"]["p_holm"]
                                                  - 3 * 2 / 128) < 1e-12)
    weak = decide(both({**base, "date_time": rec(0, 0, ref_correct=8)}))
    check("binding: a reference with 8 correct items cannot fail alone -> GO (7/8 binding)",
          weak["go_label"] == "GO (7/8 binding)" and weak["binding"]["categories"] == [c for c in RC.CATEGORIES
                                                                                     if c != "date_time"], weak["go_label"])
    # operating characteristics: vectorised rule == decide() on random replicates; then the table
    rng = np.random.default_rng(OC_SEED + 1)
    lost = rng.integers(0, 14, size=(400, 8))
    gained = rng.integers(0, 10, size=(400, 8))
    vec = rules_vec(lost, gained, P)
    agree = all(bool(vec["nogo"][i]) == (not decide({"primary": {c: rec(int(lost[i, j]), int(gained[i, j]))
                                                                for j, c in enumerate(RC.CATEGORIES)}})["go"])
                for i in range(400))
    check("operating characteristics: the vectorised rule equals decide() on 400 random replicates", agree)
    t0 = time.time()
    oc = operating_characteristics(oc_reps)
    worst = oc["null_worst_single_grader"]
    check("operating characteristics: single-grader null false NO-GO <= 0.03 at every simulated churn (5-30%) and "
          "reference accuracy (0.30-0.80) (Monte Carlo SE <= 0.0008)",
          worst <= 0.03, f"worst {worst}; {time.time() - t0:.1f}s")
    check("operating characteristics: the superseded draft rule's null false NO-GO exceeds 0.10 at 10% churn and 0.30 "
          "at 20% churn (the review's finding)",
          min(oc["null"][f"acc{a}_D7.5"]["old_nogo"] for a in OC_ACC) > 0.10
          and min(oc["null"][f"acc{a}_D15.0"]["old_nogo"] for a in OC_ACC) > 0.30,
          {a: (oc["null"][f"acc{a}_D7.5"]["old_nogo"], oc["null"][f"acc{a}_D15.0"]["old_nogo"]) for a in OC_ACC})

    # 2 end to end on synthetic draws (test root; no freeze, no scan)
    tmp = Path(tempfile.mkdtemp(prefix="ret_dry_"))
    try:
        (tmp / "docs" / "retention" / "draws").mkdir(parents=True)
        seal = RGEN.mint("dev", seed=DRY_SEED, scan=False, root=tmp, require_freeze=False, n=20)
        dp = tmp / seal["file"]
        rows, _ = RGEN.load_draw(dp)
        # planted per-cell accuracy (20 items): cat0 L1 1.0, L2 0.75, L3 0.45 -> L2 (>= 0.70); cat1 0.35 everywhere ->
        # excluded; cat2 L1 0.60, L2 0.45, L3 0.35 -> L2 (fallback >= 0.40); every other category: L3 exactly 0.70 -> L3
        plant = {}
        cells = defaultdict(list)
        for r in rows:
            cells[(r["category"], r["level"])].append(r["item"])
        for (c, l), its in cells.items():
            its = sorted(set(its))
            if c == RC.CATEGORIES[0]:
                k = {1: 20, 2: 15, 3: 9}[l]
            elif c == RC.CATEGORIES[1]:
                k = 7
            elif c == RC.CATEGORIES[2]:
                k = {1: 12, 2: 9, 3: 7}[l]
            else:
                k = 14
            plant.update({it: (i < k) for i, it in enumerate(its)})
        texts = {"supa": [RG.canonical_answer(r["gold"]) if plant[r["item"]] else _wrong(r["gold"]) for r in rows]}
        dev_run = _fake_run(tmp, dp, {"SUPA": ["supa"]}, texts, None, "dev")
        sel = dev_select(dev_run, tmp, registered=False, write=True)
        want_sel = {RC.CATEGORIES[0]: 2, RC.CATEGORIES[2]: 2, **{c: 3 for c in RC.CATEGORIES[3:]}}
        check("dev rule: highest level >= 0.70, else highest >= 0.40; below 0.40 everywhere -> excluded",
              sel["selected"] == want_sel and sel["excluded"] == [RC.CATEGORIES[1]], sel["selected"])
        check("dev selection written to the OFF setting's file", (tmp / RC.selection_rel(False)).is_file()
              and not (tmp / RC.selection_rel(True)).exists())
        # confirm draw from that selection; plant losses
        cseal = RGEN.mint("confirm", seed=DRY_SEED + 1, scan=False, root=tmp, require_freeze=False, n=75)
        cp = tmp / cseal["file"]
        crows, _ = RGEN.load_draw(cp)
        check("confirm draw = selected categories at selected levels, 75 items each",
              {(r["category"], r["level"]) for r in crows} == {(c, l) for c, l in want_sel.items()} and
              all(sum(1 for r in crows if r["category"] == c and r["twin"] != "u") == 75 for c in want_sel))
        items = defaultdict(list)
        for r in crows:
            items[r["category"]].append(r["item"])
        items = {c: sorted(set(v)) for c, v in items.items()}
        cats = list(items)
        fmt_cat = "arith_word"
        # supa: items 0..59 right. off = supa except 12 lost in cats[1] (detected, report-only comparison).
        # t2s0 = off except: one unknown_twin pair broken on its answerable twin (a loss); in fmt_cat off answers items
        # 0..11 in the wrong format (lenient-only) and t2s0 canonically (12 primary 'gains'), while t2s0 loses items
        # 12..23 outright (12 skill losses): primary net 0, lenient 12 lost -> LOSS DETECTED under the lenient grader.
        right = {"supa": {it for c in cats for it in items[c][:60]}}
        right["off"] = right["supa"] - set(items[cats[1]][:12])
        verbose_off = set(items[fmt_cat][:12])
        right["t2s0"] = right["off"] - set(items[fmt_cat][12:24])

        def side_texts(side):
            out = []
            for r in crows:
                ok = r["item"] in right[side]
                if side == "t2s0" and r["category"] == "unknown_twin" and r["item"] == items["unknown_twin"][59]:
                    ok = r["twin"] == "u"                    # the answerable twin answered UNKNOWN: pair wrong
                if side == "off" and r["item"] in verbose_off:
                    out.append(_verbose(r["gold"]))
                    continue
                out.append(RG.canonical_answer(r["gold"]) if ok else _wrong(r["gold"]))
            return out

        texts = {s: side_texts(s) for s in ("off", "t2s0", "supa")}
        V = 16
        rng = np.random.default_rng(DRY_SEED)
        base_lp = np.log(rng.dirichlet(np.ones(V), size=len(crows))).astype(np.float16)
        lps = {"supa": base_lp, "off": base_lp.copy(), "t2s0": base_lp.copy()}
        g_off = grade_side(crows, [{"pid": r["pid"], "text": t, "terminated": True} for r, t in zip(crows, texts["off"])])
        g_t2 = grade_side(crows, [{"pid": r["pid"], "text": t, "terminated": True} for r, t in zip(crows, texts["t2s0"])])
        i_off, i_t2 = item_correct(crows, g_off), item_correct(crows, g_t2)
        flips = {it for c in i_off for it in i_off[c] if i_off[c][it] != i_t2[c][it]}
        for i, r in enumerate(crows):                    # planted: KL large exactly on the primary t2s0-vs-off flips
            if r["item"] in flips:
                lps["t2s0"][i] = np.log(np.roll(np.exp(base_lp[i].astype(np.float64)), 3)).astype(np.float16)
        run = _fake_run(tmp, cp, {"SPARSE": ["off", "t2s0"], "SUPA": ["supa"]}, texts, lps, "confirm")
        res = analyze_run(run, tmp, registered=False)
        cs = res["comparisons"]
        check("sparse_vs_supa is report-only; 12 lost in one category -> LOSS DETECTED (not in go)",
              cs["sparse_vs_supa"]["gating"] is False and cs["sparse_vs_supa"]["verdict"] == "LOSS DETECTED"
              and "sparse_vs_supa" not in res["go"], cs["sparse_vs_supa"]["verdict"])
        t2p = cs["t2s0_vs_off"]["graders"]["primary"]["categories"]
        t2l = cs["t2s0_vs_off"]["graders"]["lenient"]["categories"]
        check("t2s0_vs_off: format gains hide 12 skill losses under the primary grader (12 / 12, net 0); the lenient "
              "grader detects them (12 lost / 0 gained) -> LOSS DETECTED, NO-GO",
              (t2p[fmt_cat]["lost"], t2p[fmt_cat]["gained"]) == (12, 12) and not t2p[fmt_cat]["detected_loss"]
              and (t2l[fmt_cat]["lost"], t2l[fmt_cat]["gained"]) == (12, 0) and t2l[fmt_cat]["detected_loss"]
              and cs["t2s0_vs_off"]["verdict"] == "LOSS DETECTED" and res["go"]["t2s0_vs_off"] is False
              and res["go_label"]["t2s0_vs_off"] == "NO-GO",
              {g: (cs["t2s0_vs_off"]["graders"][g]["categories"][fmt_cat]["lost"],
                   cs["t2s0_vs_off"]["graders"][g]["categories"][fmt_cat]["gained"]) for g in GRADERS})
        check("a pair broken on its answerable twin counts as a loss (unknown_twin lost 1)",
              t2p["unknown_twin"]["lost"] == 1, t2p["unknown_twin"])
        check("t2s0_vs_supa is report-only", cs["t2s0_vs_supa"]["gating"] is False and "t2s0_vs_supa" not in res["go"])
        k = cs["t2s0_vs_off"]["kl"]
        check("planted KL: AUROC for flips = 1.0", k["available"] and abs(k["auroc_flip"]["auroc"] - 1.0) < 1e-12,
              k["auroc_flip"])
        k0 = cs["sparse_vs_supa"]["kl"]
        check("no KL difference: every item KL 0 -> AUROC 0.5 (ties)", abs(k0["auroc_flip"]["auroc"] - 0.5) < 1e-12)
        ab = res["accuracy"]["t2s0"]["_abstention"]
        check("abstention components: 15 wrong items + 1 planted answerable twin answered UNKNOWN -> over-abstention 16/75",
              abs(ab["answerable_over_abstention"] - round((15 + 1) / 75, 4)) < 1e-9, ab)
        md = render_md(res)
        check("markdown renders every comparison and both graders",
              all(n in md for n in cs) and all(f"{n}: {g} grader" in md for n in cs for g in GRADERS))
        g_all = grade_side(crows, [{"pid": r["pid"], "text": RG.canonical_answer(r["gold"]), "terminated": True} for r in crows])
        check("canonical answers grade 100% under both graders", all(v["primary"] and v["lenient"] for v in g_all.values()))
        g_unk = grade_side(crows, [{"pid": r["pid"], "text": "UNKNOWN", "terminated": True} for r in crows])
        ic = item_correct(crows, g_unk)
        check("all-UNKNOWN answers: unknown_twin pair accuracy 0 (over-abstention fails the pair)",
              sum(ic["unknown_twin"].values()) == 0)
        g_trunc = grade_side(crows, [{"pid": r["pid"], "text": RG.canonical_answer(r["gold"]), "terminated": False} for r in crows])
        check("unterminated answers are wrong (primary) but count for the lenient grader",
              not any(v["primary"] for v in g_trunc.values()) and all(v["lenient"] for v in g_trunc.values()))
        try:
            analyze_run(run, tmp, registered=True)
            check("the registered path refuses an unfrozen test root", False)
        except RC.RetStop as e:
            check("the registered path refuses an unfrozen test root", True, str(e)[:80])
        (run / "sparse" / "gen_off.jsonl").write_text("tampered\n", encoding="utf-8")
        try:
            analyze_run(run, tmp, registered=False)
            check("a changed output is refused", False)
        except RC.RetStop as e:
            check("a changed output is refused", True, str(e)[:80])
        try:
            dev_select(dev_run, tmp, registered=False, write=False)
            check("dev-select refuses once a confirmation draw of its setting exists", False)
        except RC.RetStop as e:
            check("dev-select refuses once a confirmation draw of its setting exists", "confirmation draws" in str(e),
                  str(e)[:80])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "n_checks": len(checks), "n_pass": sum(c["ok"] for c in checks), "checks": checks,
            "operating_characteristics": oc, "oc_markdown": oc_markdown(oc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("dev-select", "analyze"):
        s = sp.add_parser(n)
        s.add_argument("--run", required=True)
    d = sp.add_parser("dryrun")
    d.add_argument("--record", action="store_true")
    sp.add_parser("oc")
    a = ap.parse_args()
    try:
        if a.cmd == "dev-select":
            res = dev_select(Path(a.run))
            print(json.dumps({"selected": res["selected"], "basis": res["basis"], "excluded": res["excluded"]}, indent=1))
            return 0
        if a.cmd == "analyze":
            res = cmd_analyze(Path(a.run))
            print(json.dumps({n: [c["verdict"], c["go_label"], "gating" if c["gating"] else "report-only"]
                              for n, c in res["comparisons"].items()}, indent=1))
            return 0
        if a.cmd == "oc":
            print(oc_markdown(operating_characteristics()))
            return 0
        res = dryrun()
        print(f"dry run: {res['n_pass']}/{res['n_checks']} checks pass")
        if a.record:
            RC.need(not RC.FREEZE.exists(), "the kit is frozen: DRYRUN.json is a frozen record and is not rewritten")
            res["code_sha256"] = {r: RC.sha256_file(ROOT / r) for r in ("scripts/ret_analysis.py", "scripts/ret_grade.py",
                                                                         "scripts/ret_gen.py", "scripts/ret_common.py",
                                                                         "docs/retention/DESIGN.md")}
            res["recorded_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
            (RC.DOCS / "DRYRUN.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        return 0 if res["ok"] else 1
    except RC.RetStop as e:
        print(str(e), flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
