"""PREREG_E01R section 2: the mechanical arm choice from the step A DEV2 diagnostic. Fixed before any step A output.

    python -B experiments/RCX-E01R/decide_branch.py          (writes runs/RCX-E01R/dev2/BRANCH_STEP_A.json, create-only)

Inputs: the DEV2 battery (run_e01r.load_battery; sha checked) and the raw predictions of N-CFSD s0, s1, s2
(preds_<tag>.jsonl.gz; every eval_<tag>.json must name the battery sha). Quantities are computed from the raw predictions,
not from DIAG_DEV2.json (which is cross-checked when present).

Required-no-op items under positive rules (S+) on surface `p2`, pooled over the three seeds:
  p_f = k/n for form f in {LF, CF, IM}; pbar = pooled over forms; E = pooled over LF and CF; I = p_IM; S = max - min.
  q_f, qbar: the same for negative rules (reported only). a_in: no-op accuracy on `in`, both polarities (instrument
  anchor). E - I lower bound: template-then-parent cluster bootstrap (resample P2 templates within each group with
  replacement, then parents within each sampled template; statistic = count ratio), 10,000 resamples, seed 20261267.
Branches, first match wins:
  B0  instrument fault: a seed missing, a battery sha mismatch, any n = 0, or a_in < 0.85     -> no training
  B1  H-alt: min_f p_f >= 0.80                                                               -> P3 coverage check
  B3  form-specific (EXPLICIT > IMPLICIT): E - I >= 0.15 and bootstrap 95% lower bound > 0    -> R1 and R3
  B2  near ceiling: pbar >= 0.65                                                             -> no training
  B4  otherwise (uniform deficit)                                                            -> R2 and R3
"""
import gzip
import hashlib
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

TAGS = ("N-CFSD_s0", "N-CFSD_s1", "N-CFSD_s2")
FORMS = {"LF": "EXPLICIT_LISTED_FIRST", "CF": "EXPLICIT_COMPLEMENT_FIRST", "IM": "IMPLICIT"}
BOOT_SEED, RESAMPLES, TOL = 20261267, 10_000, 1e-9


def _run_e01r():
    spec = importlib.util.spec_from_file_location("run_e01r", HERE / "run_e01r.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_inputs(out_dir=None):
    R = _run_e01r()
    out = Path(out_dir) if out_dir else R.OUT
    battery, meta = R.load_battery()
    preds, problems = {}, []
    for tag in TAGS:
        ev, pp = out / f"eval_{tag}.json", out / f"preds_{tag}.jsonl.gz"
        if not (ev.exists() and pp.exists()):
            problems.append(f"{tag} missing")
            continue
        if json.loads(ev.read_text(encoding="utf-8"))["battery_sha256"] != meta["battery_sha256"]:
            problems.append(f"{tag} battery sha mismatch")
            continue
        with gzip.open(pp, "rt", encoding="utf-8") as f:
            preds[tag] = [json.loads(x) for x in f]
        if len(preds[tag]) != len(battery):
            problems.append(f"{tag} has {len(preds[tag])} predictions for {len(battery)} records")
    return battery, meta, preds, problems


def items(battery, preds, surface, pol, forms=None):
    """{template: {parent: [k, n]}} over required-no-op records of `surface` with rule polarity `pol`."""
    op = "S+" if pol == "POS" else "S-"
    acc = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for i, r in enumerate(battery):
        if r["surface"] != surface or r["rule_ast"]["op"] != op or r["gold"]["decision"] != "noop":
            continue
        if forms is not None and r.get("p2_form") not in forms:
            continue
        for p in preds.values():
            c = acc[r.get("p2_template", "-")][r["parent_id"]]
            c[0] += p[i]["decision_ok"]
            c[1] += 1
    return acc


def ratio(acc):
    k = sum(c[0] for t in acc.values() for c in t.values())
    n = sum(c[1] for t in acc.values() for c in t.values())
    return (k / n if n else None), k, n


def boot_diff(acc_e, acc_i, resamples=RESAMPLES, seed=BOOT_SEED):
    """Template-then-parent cluster bootstrap of ratio(E) - ratio(I); returns (lo, hi) percentile 95%."""
    rng = random.Random(seed)
    groups = [[list(t.values()) for t in acc.values()] for acc in (acc_e, acc_i)]
    diffs = []
    for _ in range(resamples):
        vals = []
        for tmpl in groups:
            k = n = 0
            for _t in range(len(tmpl)):
                parents = tmpl[rng.randrange(len(tmpl))]
                for _p in range(len(parents)):
                    c = parents[rng.randrange(len(parents))]
                    k += c[0]
                    n += c[1]
            vals.append(k / n if n else 0.0)
        diffs.append(vals[0] - vals[1])
    diffs.sort()
    return diffs[int(0.025 * resamples)], diffs[int(0.975 * resamples) - 1]


def decide(battery, preds, problems, meta=None):
    q = {"problems": problems}
    if problems:
        q["branch"] = "B0"
        return q
    p = {f: ratio(items(battery, preds, "p2", "POS", {name})) for f, name in FORMS.items()}
    qn = {f: ratio(items(battery, preds, "p2", "NEG", {name})) for f, name in FORMS.items()}
    pbar = ratio(items(battery, preds, "p2", "POS"))
    acc_e = items(battery, preds, "p2", "POS", {FORMS["LF"], FORMS["CF"]})
    acc_i = items(battery, preds, "p2", "POS", {FORMS["IM"]})
    E, I = ratio(acc_e)[0], ratio(acc_i)[0]
    ain_acc = {**{("POS", k): v for k, v in items(battery, preds, "in", "POS").items()},
               **{("NEG", k): v for k, v in items(battery, preds, "in", "NEG").items()}}
    a_in = ratio(ain_acc)
    ns = [v[2] for v in p.values()] + [v[2] for v in qn.values()] + [pbar[2], a_in[2]]
    q.update({"p": {f: v[0] for f, v in p.items()}, "p_counts": {f: v[1:] for f, v in p.items()},
              "q": {f: v[0] for f, v in qn.items()}, "q_counts": {f: v[1:] for f, v in qn.items()},
              "pbar": pbar[0], "qbar": ratio(items(battery, preds, "p2", "NEG"))[0], "E": E, "I": I,
              "a_in": a_in[0], "templates": {"E": len(acc_e), "I": len(acc_i)}})
    if any(n == 0 for n in ns):
        q["branch"] = "B0"
        q["problems"] = ["a cell has n = 0"]
        return q
    q["S"] = max(q["p"].values()) - min(q["p"].values())
    q["E_minus_I"] = E - I
    q["E_minus_I_ci95"] = boot_diff(acc_e, acc_i)
    if q["a_in"] < 0.85 - TOL:
        q["branch"] = "B0"
        q["problems"] = [f"a_in {q['a_in']:.3f} < 0.85"]
    elif min(q["p"].values()) >= 0.80 - TOL:
        q["branch"] = "B1"
    elif q["E_minus_I"] >= 0.15 - TOL and q["E_minus_I_ci95"][0] > 0:
        q["branch"] = "B3"
    elif q["pbar"] >= 0.65 - TOL:
        q["branch"] = "B2"
    else:
        q["branch"] = "B4"
    q["arms"] = {"B0": [], "B1": [], "B2": [], "B3": ["R1", "R3"], "B4": ["R2", "R3"]}[q["branch"]]
    return q


def main():
    battery, meta, preds, problems = load_inputs()
    out = _run_e01r().OUT / "BRANCH_STEP_A.json"
    if out.exists():
        raise SystemExit(f"{out} exists (create-only)")
    q = decide(battery, preds, problems, meta)
    q["battery_sha256"] = meta["battery_sha256"]
    q["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with open(out, "x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(q, indent=1) + "\n")
    print(json.dumps(q, indent=1))


if __name__ == "__main__":
    main()
