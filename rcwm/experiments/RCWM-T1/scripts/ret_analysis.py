"""Retention suite analysis (roadmap item 4 / joint 5c). CPU only, numpy only; never imports torch.

  python -B scripts/ret_analysis.py dev-select --run runs/retention/dev_20261121
        Supa's dev accuracies -> docs/retention/DEV_SELECTION.json (create-only). Refused once any confirmation draw
        exists: the selection must precede every confirmation draw.
  python -B scripts/ret_analysis.py analyze --run runs/retention/confirm_<seed>
        The registered analysis of one sealed confirmation run -> <run>/analysis/RET_RESULTS.json + .md (create-only;
        run it once). Refuses unless the kit freeze, the draw seal, DEV_SELECTION.json and every recorded output hash
        verify.
  python -B scripts/ret_analysis.py dryrun [--record]
        Synthetic fixtures (a test-root draw, generated answers with planted losses, planted KL): every pre-stated
        verdict, the Holm step, the pair rule, the binding note, the dev selection rule and the AUROC are checked
        against hand-computed expectations. --record writes docs/retention/DRYRUN.json (refused once frozen).

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
    "that cell of the dev draw (dev seed 20261121; an unknown_twin item counts as correct only if both twins are). "
    "The selected level of c is the highest l with a(c, l) >= 0.40. A category with no such level is EXCLUDED from "
    "every confirmation draw (reported, never gated). The selection is written once (DEV_SELECTION.json, "
    "create-only) and before any confirmation draw is minted.")
RULE_CONFIRM = (
    "For each comparison of an arm A with its reference R (Titan-Sparse = adapter OFF vs Supa 1.7; T2 s0 ON vs OFF; "
    "each extra Titan-Sparse adapter vs OFF; each extra Supa adapter vs Supa 1.7) on a sealed confirmation draw "
    "minted after DEV_SELECTION.json, and for each selected category c (75 items): lost = items R answers correctly "
    "and A does not; gained = items A answers correctly and R does not; net = gained - lost (primary grader; an "
    "unknown_twin item is correct only if both twins are). p = two-sided exact McNemar p-value (binomial on the "
    "lost + gained discordant items, 1/2); p_Holm = Holm-Bonferroni adjustment of p over the selected categories of "
    "that comparison. Category c is RETAINED iff net >= -5 and NOT (lost > gained and p_Holm < 0.05). Comparison "
    "verdict: RETAINED iff every selected category is RETAINED; LOSS DETECTED iff some category has lost > gained "
    "with p_Holm < 0.05; otherwise MARGIN EXCEEDED (some category has net < -5 without a detected loss). GO (the "
    "retention guard passes for arm A) iff RETAINED; LOSS DETECTED and MARGIN EXCEEDED are NO-GO. Report-only, never "
    "gating: T2 s0 vs Supa 1.7 (cumulative drift), excluded categories, accuracies with Wilson 95% intervals, the "
    "lenient grader, EOS-termination rates, abstention components, the can-fail binding note, and the AUROC of the "
    "per-item first-token KL(R || A) for predicting correctness flips. The suite is templated and narrow: a "
    "RETAINED verdict is a floor, not a proof of general retention.")
GATING = ("sparse_vs_supa", "t2s0_vs_off")
BOOT, BOOT_SEED = 2000, 20261122
DRY_SEED = 20261123


# ------------------------------------------------------------------------------------------------ statistics
def mcnemar(ref: dict, arm: dict) -> dict:
    keys = sorted(ref)
    if set(keys) != set(arm):
        raise ValueError("item ids differ between reference and arm")
    lost = sum(1 for k in keys if ref[k] and not arm[k])
    gained = sum(1 for k in keys if not ref[k] and arm[k])
    n = lost + gained
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(lost, gained) + 1)) / 2 ** n)
    return {"n": len(keys), "ref_correct": sum(bool(ref[k]) for k in keys), "arm_correct": sum(bool(arm[k]) for k in keys),
            "lost": lost, "gained": gained, "net": gained - lost, "discordant": n, "p": p}


def holm(ps: dict) -> dict:
    """Holm-Bonferroni step-down adjusted p-values for {name: p}."""
    items = sorted(ps.items(), key=lambda kv: (kv[1], kv[0]))
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def min_lost_to_fail(n: int = RC.N_PER_CELL, m: int = 8) -> int:
    """Fewest losses (no gains) that fail a category: by the margin (net < -5) or by the detection clause
    (p_Holm < 0.05), whichever binds first; the detection clause is evaluated as the first Holm step, m x 2/2^k."""
    k = 1
    while not (k > RC.MARGIN or m * 2.0 ** (1 - k) < RC.ALPHA):
        k += 1
    return k


def category_verdict(rec: dict, p_holm: float) -> dict:
    detected = rec["lost"] > rec["gained"] and p_holm < RC.ALPHA
    within = rec["net"] >= -RC.MARGIN
    return {**rec, "p_holm": p_holm, "within_margin": within, "detected_loss": detected,
            "retained": within and not detected,
            "binding": {"min_lost_to_fail": min_lost_to_fail(rec["n"]),
                        "can_fail": rec["ref_correct"] >= min_lost_to_fail(rec["n"])}}


def comparison_verdict(cats: dict) -> str:
    if all(c["retained"] for c in cats.values()):
        return "RETAINED"
    if any(c["detected_loss"] for c in cats.values()):
        return "LOSS DETECTED"
    return "MARGIN EXCEEDED"


def decide(per_cat: dict) -> dict:
    """per_cat {category: mcnemar record} -> category verdicts + comparison verdict (RULE_CONFIRM)."""
    adj = holm({c: r["p"] for c, r in per_cat.items()})
    cats = {c: category_verdict(r, adj[c]) for c, r in per_cat.items()}
    v = comparison_verdict(cats)
    return {"categories": cats, "verdict": v, "go": v == "RETAINED",
            "pooled": {"lost": sum(c["lost"] for c in cats.values()), "gained": sum(c["gained"] for c in cats.values()),
                       "net": sum(c["net"] for c in cats.values())}}


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
    pids = [r for r in rows]
    out["_termination_rate"] = round(sum(graded[r["pid"]]["terminated"] for r in pids) / len(pids), 4)
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
def _load_gens(path: Path) -> list:
    return RC.read_jsonl(path)


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
    gens, lps = {}, {}
    for st, sides in prov["sides"].items():
        d = run_dir / {"SPARSE": "sparse", "SUPA": "supa"}[st]
        for s in sides:
            gens[s] = _load_gens(d / f"gen_{s}.jsonl")
            if prov.get("lp"):
                lps[s] = np.load(d / f"lp_{s}.npy", mmap_mode="r")
    return {"prov": prov, "seal": seal, "rows": rows, "gens": gens, "lps": lps}


def comparisons_for(sides: list) -> list:
    """(name, arm, ref, gating)."""
    out = []
    if "off" in sides and "supa" in sides:
        out.append(("sparse_vs_supa", "off", "supa", True))
    if "t2s0" in sides and "off" in sides:
        out.append(("t2s0_vs_off", "t2s0", "off", True))
    for s in sides:
        if s.startswith("x_"):
            out.append((f"{s}_vs_off", s, "off", True))
        if s.startswith("xs_"):
            out.append((f"{s}_vs_supa", s, "supa", True))
    if "t2s0" in sides and "supa" in sides:
        out.append(("t2s0_vs_supa", "t2s0", "supa", False))
    return out


def kl_block(rows: list, lps: dict, arm: str, ref: str, ref_items: dict, arm_items: dict) -> dict:
    """Report-only: per-item KL(R || A) (mean over the item's prompts) and its AUROC for flips / losses."""
    if arm not in lps or ref not in lps:
        return {"available": False}
    kl = kl_rows(lps[ref], lps[arm])
    per_item = defaultdict(list)
    cat_of = {}
    for i, r in enumerate(rows):
        per_item[r["item"]].append(kl[i])
        cat_of[r["item"]] = r["category"]
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
    items = {s: item_correct(rows, graded[s]) for s in sides}
    cats = sorted({r["category"] for r in rows}, key=RC.CATEGORIES.index)
    res = {"rule": RULE_CONFIRM, "draw_id": prov["draw_id"], "draw_sha256": prov["draw_sha256"],
           "selection_sha256": R["seal"].get("selection_sha256"), "registered": registered,
           "categories": cats, "levels": {c: R["seal"]["plan"][c] for c in cats},
           "accuracy": {s: side_summary(rows, graded[s]) for s in sides}, "comparisons": {}}
    for name, arm, ref, gating in comparisons_for(sides):
        per_cat = {c: mcnemar(items[ref][c], items[arm][c]) for c in cats}
        d = decide(per_cat)
        d.update(arm=arm, reference=ref, gating=gating,
                 kl=kl_block(rows, R["lps"], arm, ref, {k: v for c in cats for k, v in items[ref][c].items()},
                             {k: v for c in cats for k, v in items[arm][c].items()}))
        res["comparisons"][name] = d
    res["go"] = {n: c["go"] for n, c in res["comparisons"].items() if c["gating"]}
    return res


def render_md(res: dict) -> str:
    L = [f"# Retention suite results: {res['draw_id']}", "",
         f"Draw sha256 `{res['draw_sha256']}`; selection sha256 `{res['selection_sha256']}`; registered: "
         f"{res['registered']}.", "", "## Pre-stated rule", "", "> " + res["rule"], "", "## Verdicts", "",
         "| comparison | gating | verdict | GO | pooled lost / gained / net |", "|---|---|---|---|---|"]
    for n, c in res["comparisons"].items():
        p = c["pooled"]
        L.append(f"| {n} ({c['arm']} vs {c['reference']}) | {'yes' if c['gating'] else 'report-only'} | {c['verdict']} | "
                 f"{'GO' if c['go'] else 'NO-GO'} | {p['lost']} / {p['gained']} / {p['net']:+d} |")
    for n, c in res["comparisons"].items():
        L += ["", f"### {n}", "", "| category | level | ref | arm | lost | gained | net | p | p_Holm | retained | can fail |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for cat, r in c["categories"].items():
            L.append(f"| {cat} | {res['levels'][cat]} | {r['ref_correct']}/{r['n']} | {r['arm_correct']}/{r['n']} | "
                     f"{r['lost']} | {r['gained']} | {r['net']:+d} | {r['p']:.4g} | {r['p_holm']:.4g} | "
                     f"{'yes' if r['retained'] else 'NO'} | {'yes' if r['binding']['can_fail'] else 'no'} |")
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
    L += ["", "Cells: primary accuracy (lenient-grader correct count, report-only).", ""]
    return "\n".join(L) + "\n"


def cmd_analyze(run_dir: Path, root: Path = ROOT) -> dict:
    out = Path(run_dir) / "analysis"
    RC.need(not (out / "RET_RESULTS.json").exists(), "RET_RESULTS.json exists: the registered analysis runs once")
    sel = root / "docs" / "retention" / "DEV_SELECTION.json"
    res = analyze_run(run_dir, root, registered=True)
    RC.need(sel.is_file() and RC.sha256_file(sel) == res["selection_sha256"], "DEV_SELECTION.json differs from the draw's")
    res["code_sha256"] = RC.code_sha(root)
    res["analysed_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    RC.write_json_create(out / "RET_RESULTS.json", res)
    RC.write_create(out / "RET_RESULTS.md", render_md(res))
    return res


# ------------------------------------------------------------------------------------------------ dev selection
def dev_select(run_dir: Path, root: Path = ROOT, registered: bool = True, write: bool = True) -> dict:
    root = Path(root)
    draws = root / "docs" / "retention" / "draws"
    confirm = sorted(p.name for p in draws.glob("confirm_*.jsonl"))
    RC.need(not confirm, f"confirmation draws already exist ({confirm}): the dev selection must precede them")
    R = load_run(run_dir, root, registered)
    RC.need(R["prov"]["mode"] == "dev" and R["seal"]["purpose"] == "dev", "dev-select needs the dev run")
    if registered:
        RC.need(R["seal"]["seed"] == RC.DEV_SEED, "the dev run must use the fixed dev seed")
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
    selected, excluded = {}, []
    for c in RC.CATEGORIES:
        ok = [l for l in sorted(acc.get(c, {})) if acc[c][l]["correct"] >= RC.DEV_MIN_ACC * acc[c][l]["n"] - 1e-9]
        if ok:
            selected[c] = max(ok)
        else:
            excluded.append(c)
    res = {"rule": RULE_DEV, "dev_draw": R["seal"]["draw_id"], "dev_draw_sha256": R["seal"]["file_sha256"],
           "dev_run_done_sha256": RC.sha256_file(Path(run_dir) / "DONE.json"), "surface_diversity": R["seal"]["surface_diversity"],
           "accuracy": acc, "selected": selected, "excluded": excluded,
           "freeze_sha256": R["seal"].get("freeze_sha256"), "registered": registered,
           "abstention": side_summary(rows, graded).get("_abstention"), "selected_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    if write:
        RC.write_json_create(root / "docs" / "retention" / "DEV_SELECTION.json", res)
    return res


# ------------------------------------------------------------------------------------------------ dry run
def _fake_run(root: Path, draw_p: Path, sides: dict, texts: dict, lps: dict | None, mode: str) -> Path:
    """A DONE run directory in the layout ret_run.py writes, from given per-side texts (and log-prob arrays)."""
    rows, seal = RGEN.load_draw(draw_p)
    run = root / "runs" / "retention" / seal["draw_id"]
    prov = {"draw_id": seal["draw_id"], "draw_sha256": seal["file_sha256"], "mode": mode, "registered": False,
            "freeze_sha256": None, "sides": sides, "lp": lps is not None, "stages": {}, "turns": []}
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


def dryrun() -> dict:
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
    check("min losses to fail a 75-item category (8 categories) = 6", min_lost_to_fail(75, 8) == 6, min_lost_to_fail(75, 8))
    check("Wilson 30/75 = [0.2966, 0.5131] (hand-computed)", wilson(30, 75) == [0.2966, 0.5131], wilson(30, 75))
    check("AUROC perfect / inverted / ties", auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0 and
          auroc([0, 1, 2, 3], [1, 1, 0, 0]) == 0.0 and auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5)

    def rec(lost, gained, ref_correct=40, n=75):
        p = 1.0 if lost + gained == 0 else min(1.0, 2 * sum(math.comb(lost + gained, i) for i in range(min(lost, gained) + 1))
                                                 / 2 ** (lost + gained))
        return {"n": n, "ref_correct": ref_correct, "arm_correct": ref_correct - lost + gained, "lost": lost, "gained": gained,
                "net": gained - lost, "discordant": lost + gained, "p": p}

    base = {c: rec(0, 0) for c in RC.CATEGORIES}
    scen = {
        "identical arms": (base, "RETAINED"),
        "one category net -5 (5 lost, 0 gained)": ({**base, "arith_word": rec(5, 0)}, "RETAINED"),
        "one category 9 lost / 4 gained (net -5)": ({**base, "arith_word": rec(9, 4)}, "RETAINED"),
        "one category net -6 (raw p 0.031, Holm 0.25)": ({**base, "arith_word": rec(6, 0)}, "MARGIN EXCEEDED"),
        "one category 12 lost (Holm p < 0.05)": ({**base, "kv_recall": rec(12, 0)}, "LOSS DETECTED"),
        "net -4 in every category (pooled -32)": ({c: rec(4, 0) for c in RC.CATEGORIES}, "RETAINED"),
        "gains only (+10)": ({**base, "pct_ratio": rec(0, 10)}, "RETAINED"),
        "net -6 everywhere: detection via Holm (6 x p 0.031 -> first step 0.25)": ({c: rec(6, 0) for c in RC.CATEGORIES},
                                                                                   "MARGIN EXCEEDED"),
        "net -8 everywhere: every p 0.0078, Holm 0.0625 -> no detection": ({c: rec(8, 0) for c in RC.CATEGORIES},
                                                                            "MARGIN EXCEEDED"),
        "net -9 everywhere: Holm 8 x 0.0039 = 0.031 -> detected": ({c: rec(9, 0) for c in RC.CATEGORIES}, "LOSS DETECTED"),
    }
    for name, (pc, want) in scen.items():
        got = decide(pc)["verdict"]
        check(f"rule: {name} -> {want}", got == want, got)
    sub = {c: rec(0, 0) for c in ("arith_word", "kv_recall", "date_time")}
    sub["kv_recall"] = rec(7, 0)
    d = decide(sub)
    check("Holm over the SELECTED categories only (3): 7 lost -> p 0.0156 x 3 = 0.047 -> LOSS DETECTED",
          d["verdict"] == "LOSS DETECTED" and abs(d["categories"]["kv_recall"]["p_holm"] - 3 * 2 / 128) < 1e-12)
    check("binding note: a reference with 5 correct items cannot fail", not category_verdict(rec(0, 0, 5), 1.0)["binding"]["can_fail"])

    # 2 end to end on a synthetic draw (test root; no freeze, no scan)
    tmp = Path(tempfile.mkdtemp(prefix="ret_dry_"))
    try:
        (tmp / "docs" / "retention" / "draws").mkdir(parents=True)
        seal = RGEN.mint("dev", seed=DRY_SEED, scan=False, root=tmp, require_freeze=False, n=20)
        dp = tmp / seal["file"]
        rows, _ = RGEN.load_draw(dp)
        # dev: plant accuracies per cell: cat0 L1 1.0, L2 0.40 (8/20), L3 0.35 (7/20) -> L2; cat1 all 0.35 -> excluded;
        # every other category: L3 exactly 0.40 -> L3
        plant = {}
        cells = defaultdict(list)
        for r in rows:
            cells[(r["category"], r["level"])].append(r["item"])
        for (c, l), its in cells.items():
            its = sorted(set(its))
            if c == RC.CATEGORIES[0]:
                k = {1: 20, 2: 8, 3: 7}[l]
            elif c == RC.CATEGORIES[1]:
                k = 7
            else:
                k = 8
            plant.update({it: (i < k) for i, it in enumerate(its)})
        texts = {"supa": [RG.canonical_answer(r["gold"]) if plant[r["item"]] else _wrong(r["gold"]) for r in rows]}
        run = _fake_run(tmp, dp, {"SUPA": ["supa"]}, texts, None, "dev")
        sel = dev_select(run, tmp, registered=False, write=True)
        want_sel = {RC.CATEGORIES[0]: 2, **{c: 3 for c in RC.CATEGORIES[2:]}}
        check("dev rule: highest level >= 0.40; a category below 0.40 at every level is excluded",
              sel["selected"] == want_sel and sel["excluded"] == [RC.CATEGORIES[1]], sel["selected"])
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
        # supa: items 0..59 right; off: = supa except 12 lost in cats[0]; t2s0: = off except 6 lost in cats[1] and
        # one unknown_twin pair broken on the answerable twin only (counts as a loss)
        right = {"supa": {it for c in cats for it in items[c][:60]}}
        right["off"] = right["supa"] - set(items[cats[0]][:12])
        lost_t2 = set(items[cats[1]][:6])
        right["t2s0"] = right["off"] - lost_t2

        def side_texts(side):
            out = []
            for r in crows:
                ok = r["item"] in right[side]
                if side == "t2s0" and r["category"] == "unknown_twin" and r["item"] == items["unknown_twin"][59]:
                    ok = r["twin"] == "u"                    # the answerable twin answered UNKNOWN: pair wrong
                out.append(RG.canonical_answer(r["gold"]) if ok else _wrong(r["gold"]))
            return out

        texts = {s: side_texts(s) for s in ("off", "t2s0", "supa")}
        V = 16
        rng = np.random.default_rng(DRY_SEED)
        base_lp = np.log(rng.dirichlet(np.ones(V), size=len(crows))).astype(np.float16)
        lps = {"supa": base_lp, "off": base_lp.copy(), "t2s0": base_lp.copy()}
        flipped_rows = [i for i, r in enumerate(crows) if (r["item"] in right["off"]) != (r["item"] in right["t2s0"])
                        or r["item"] == items["unknown_twin"][59]]
        for i in flipped_rows:                          # planted: KL large exactly on the t2s0-vs-off flips
            lps["t2s0"][i] = np.log(np.roll(np.exp(base_lp[i].astype(np.float64)), 3)).astype(np.float16)
        run = _fake_run(tmp, cp, {"SPARSE": ["off", "t2s0"], "SUPA": ["supa"]}, texts, lps, "confirm")
        res = analyze_run(run, tmp, registered=False)
        cs = res["comparisons"]
        check("sparse_vs_supa: 12 lost in one category -> LOSS DETECTED, NO-GO",
              cs["sparse_vs_supa"]["verdict"] == "LOSS DETECTED" and not cs["sparse_vs_supa"]["go"],
              cs["sparse_vs_supa"]["categories"][cats[0]])
        t2 = cs["t2s0_vs_off"]["categories"]
        check("t2s0_vs_off: 6 lost (margin) in one category + 1 pair broken on its answerable twin -> MARGIN EXCEEDED",
              cs["t2s0_vs_off"]["verdict"] == "MARGIN EXCEEDED" and t2[cats[1]]["lost"] == 6 and
              t2["unknown_twin"]["lost"] == 1, {c: (v["lost"], v["gained"]) for c, v in t2.items()})
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
        check("markdown renders every comparison", all(n in md for n in cs))
        # grading: all-canonical side is 100%, all-UNKNOWN side is 0 on unknown_twin pairs
        g_all = grade_side(crows, [{"pid": r["pid"], "text": RG.canonical_answer(r["gold"]), "terminated": True} for r in crows])
        check("canonical answers grade 100% (primary)", all(v["primary"] for v in g_all.values()))
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
            dev_select(run, tmp, registered=False, write=False)
            check("dev-select refuses once a confirmation draw exists", False)
        except RC.RetStop as e:
            check("dev-select refuses once a confirmation draw exists", True, str(e)[:80])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "n_checks": len(checks), "n_pass": sum(c["ok"] for c in checks), "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n in ("dev-select", "analyze"):
        s = sp.add_parser(n)
        s.add_argument("--run", required=True)
    d = sp.add_parser("dryrun")
    d.add_argument("--record", action="store_true")
    a = ap.parse_args()
    try:
        if a.cmd == "dev-select":
            res = dev_select(Path(a.run))
            print(json.dumps({"selected": res["selected"], "excluded": res["excluded"]}, indent=1))
            return 0
        if a.cmd == "analyze":
            res = cmd_analyze(Path(a.run))
            print(json.dumps({n: c["verdict"] for n, c in res["comparisons"].items()}, indent=1))
            return 0
        res = dryrun()
        print(f"dry run: {res['n_pass']}/{res['n_checks']} checks pass")
        if a.record:
            RC.need(not RC.FREEZE.exists(), "the kit is frozen: DRYRUN.json is a frozen record and is not rewritten")
            res["code_sha256"] = {r: RC.sha256_file(ROOT / r) for r in ("scripts/ret_analysis.py", "scripts/ret_grade.py",
                                                                         "scripts/ret_gen.py", "scripts/ret_common.py")}
            res["recorded_local"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
            (RC.DOCS / "DRYRUN.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        return 0 if res["ok"] else 1
    except RC.RetStop as e:
        print(str(e), flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
