"""T2b-ledger dev gate: scorer self-test, the dev analysis and the PRE-STATED gate rule (CPU, numpy only, never torch).

  python -B scripts/t2bl_gate.py --selftest [--record]     # scorer self-test on the dev bank (golds, floors, negatives,
                                                           # parser identity) -> docs/t2b/ledger/SCORER_SELFTEST.json
  python -B scripts/t2bl_gate.py --dryrun [--record]       # every gate branch on synthetic outcomes through the real
                                                           # scorer, metrics, decide() and wording() -> GATE_DRYRUN.json
  python -B scripts/t2bl_gate.py --write-rule-md           # docs/t2b/ledger/GATE_RULE.md from RULE_TEXT (refused once a
                                                           # runs/t2bl output exists)
  python -B scripts/t2bl_gate.py --freeze-rule --approval "<who, when, DECISIONS entry>"
                                                           # GATE_RULE.json: sha256 of every frozen input (create-only;
                                                           # refused if any runs/t2bl output exists). NOT run by the kit
                                                           # build: freezing is the operator's step.
  python -B scripts/t2bl_gate.py [--run runs/t2bl/dev]     # the registered analysis (once; create-only):
                                                           # <run>/analysis/GATE.json + GATE.md

Before any output is read: the frozen rule record verifies (registered mode) and the scorer self-test passes.
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
if "torch" in sys.modules:
    raise SystemExit("t2bl_gate.py must run without torch")
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "2"

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import t2bl_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

RULE_VERSION = 1
SEEDS = (0, 1, 2)
BAR_HS = 0.80
NI_MARGIN = 0.05
COPY_CAP = 0.10
TRIGGER_TRAINED_MIN = 0.80
TRIGGER_HELDOUT_MAX = 0.30
RULE_TEXT = (
    "T2b-ledger dev gate, revision 1 (pre-stated on 2026-10-10 before any T2b-ledger adapter or output exists). Inputs: "
    "the dev bank data/t2bl_dev.jsonl (seed 20261162; 100 bases; 17 strata) generated in one run of scripts/t2bl_run.py "
    "(tag dev) with sides OFF (Titan-Sparse, adapter disabled), T2s0 (the PREREG-2 T2 seed-0 adapter) and ON (each "
    "T2b-ledger seed s = 0, 1, 2, unmerged), scored by p3_common.score_onestep (exact balances, any key order). For each "
    "seed s: G1 (held-out words on the combined held-out surface): the ON accuracy on HS_words (100 rows: a held-out "
    "non-posted status word in the combined held-out surface: rule paraphrase 4, held-out account names, every object "
    "in a held-out key order; gold = balances unchanged) is >= 0.80; G2 (in-distribution non-inferiority to T2 s0): the "
    "ON accuracy on ID (ID_posted and ID_void, 200 canonical-surface in-range rows) minus T2s0's accuracy on the same "
    "rows is >= -0.05 (point estimate; the paired base-bootstrap 95% lower bound is reported, not gated); G3 (copy "
    "guard on the held-out surface): on HS_posted (the posted twins of HS_words' bases in the same surface) the share of "
    "ON answers equal to the input balances is <= 0.10. PROCEED to the sealed confirmation (PREREG-5) iff G1, G2 and G3 "
    "hold in all three seeds, the run is complete (DONE.json, every stage recorded and unchanged) and the A0 anchor "
    "passed in every turn. Otherwise STOP, worded as: TRIGGER WIDENED iff in all three seeds the ON accuracy on S_words "
    "(training non-posted words, canonical surface) is >= 0.80 and on HS_words is <= 0.30; PARTIAL otherwise, with the "
    "per-stratum, per-variant, per-seed table and the named patterns (heldout_below_bar, in_distribution_cost, "
    "copy_on_heldout_surface, trained_words_below_bar, seed_disagreement). NO VERDICT (instrument) iff the run is "
    "incomplete or an anchor failed. Revision path after a STOP (pre-stated): no confirmation is drafted from this kit; "
    "a revised kit may change only the training data (composition, vocabulary sizes, magnitude bands, paraphrases), "
    "must draw a fresh dev bank under a new seed (this dev bank and every adapter trained on the old data are then "
    "recorded as spent), and keeps these thresholds. T2b-ledger seed 0 is the registered artifact for PREREG-5's planning "
    "and retention parts; seeds 1 and 2 replicate; no seed is ever chosen by its outcome.")
PATTERNS = ("heldout_below_bar", "in_distribution_cost", "copy_on_heldout_surface", "trained_words_below_bar",
            "seed_disagreement")
VARIANT_KEYS = {"S_words": "word_kind", "H_words": "word_kind", "HS_words": "word_kind", "S_surface_words": "word_kind",
                "H_names": "names_kind", "HS_posted": "names_kind", "H_order": "event_order", "H_para": "status",
                "P3_words": "word", "S_surface_posted": "names_kind"}
ANALYSIS_CODE = ("scripts/t2bl_gate.py", "scripts/t2bl_common.py", "scripts/p3_common.py", "rcwmt/env.py")


class GateError(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise GateError(msg)


# ============================================================================================== scorer self-test
def renderings(gold: dict, row: dict) -> list:
    """Correct answers in several surface forms: compact sorted, the row's own key order, spaced, trailing period,
    pretty, {"before","after"}-wrapped, reversed key order."""
    keys = list(gold)
    rev = OrderedDict((k, gold[k]) for k in reversed(sorted(keys)))
    own = OrderedDict((k, gold[k]) for k in (row.get("surface") or {}).get("before", sorted(keys)))
    return [env.dumps(gold), json.dumps(own, separators=(",", ":")), json.dumps(gold), env.dumps(gold) + ".",
            json.dumps(gold, indent=2), json.dumps({"before": row["before"], "after": gold}),
            json.dumps(rev, separators=(",", ":"))]


def negatives(row: dict) -> list:
    g = json.loads(row["gold"])
    k0 = sorted(g)[0]
    out = [({**g, k0: g[k0] + 1}, "value +1"), ({**g, k0: g[k0] - 1}, "value -1"),
           ({k: v for k, v in g.items() if k != k0}, "missing key"), ({**g, "ZZ": 0}, "extra key"),
           ({**g, k0: str(g[k0])}, "string value"), ({**g, k0: True}, "bool value")]
    texts = [(json.dumps(x, separators=(",", ":")), why) for x, why in out]
    texts += [("", "empty"), ("The balances are unchanged.", "prose")]
    if row["gold_rule"] == "noop":
        texts.append((row["posted_gold"], "applied on a no-op row"))
    else:
        texts.append((env.dumps(row["before"]), "copied on an apply row"))
    return texts


def selftest(rows: list) -> dict:
    n_gold = n_gold_ok = n_neg = n_neg_wrong = n_cls_ok = 0
    copy_ok = True
    parser_same = parser_n = 0
    fails = []
    for r in rows:
        g = json.loads(r["gold"])
        for t in renderings(g, r):
            n_gold += 1
            sc = C.score(r, t)
            n_gold_ok += sc["correct"]
            if not sc["correct"] and len(fails) < 5:
                fails.append({"id": r["id"], "text": t[:120], "why": "gold rendering scored wrong"})
        sc = C.score(r, env.dumps(r["before"]))
        if sc["correct"] != (r["gold_rule"] == "noop"):
            copy_ok = False
        for t, why in negatives(r):
            n_neg += 1
            sc = C.score(r, t)
            n_neg_wrong += not sc["correct"]
            want = {"applied on a no-op row": "applied_as_posted", "copied on an apply row": "copied",
                    "empty": "unparsed", "prose": "unparsed"}.get(why)
            n_cls_ok += want is None or sc["cls"] == want
            if sc["correct"] and len(fails) < 10:
                fails.append({"id": r["id"], "text": t[:120], "why": f"negative ({why}) scored correct"})
        if set(r["before"]) == set(env.ACCOUNTS):
            for t in renderings(g, r) + [x for x, _ in negatives(r)]:
                parser_n += 1
                parser_same += C.P3.parse_general(t, list(env.ACCOUNTS)) == env.parse_state(C.LEDGER, t, lenient=True)
    ok = n_gold_ok == n_gold and n_neg_wrong == n_neg and n_cls_ok == n_neg and copy_ok and parser_same == parser_n
    return {"ok": ok, "rows": len(rows), "gold_renderings": n_gold, "gold_correct": n_gold_ok, "negatives": n_neg,
            "negatives_wrong": n_neg_wrong, "error_classes_right": n_cls_ok, "copy_scores_copy_floor": copy_ok,
            "parse_general_equals_env_parse_state_on_A_D": f"{parser_same}/{parser_n}", "failures": fails,
            "bank_sha256": C.sha256(C.DEV) if C.DEV.exists() else None}


# ============================================================================================== metrics
def score_side(rows: list, texts: list) -> list:
    return [C.score(r, t) for r, t in zip(rows, texts)]


def copied(r: dict, text) -> bool:
    st = None if text is None else C.parse_ledger(text, list(r["before"]))
    return st is not None and st == r["before"]


def stratum_metrics(rows: list, sc: list, texts: list) -> dict:
    out = {}
    for s in C.DEV_STRATA:
        ix = [i for i, r in enumerate(rows) if r["stratum"] == s]
        if not ix:
            continue
        k = sum(sc[i]["correct"] for i in ix)
        lo, hi = C.wilson(k, len(ix))
        d = {"n": len(ix), "correct": k, "acc": k / len(ix), "wilson": [lo, hi],
             "classes": dict(Counter(sc[i]["cls"] for i in ix)),
             "copied_rate": sum(copied(rows[i], texts[i]) for i in ix) / len(ix)}
        vk = VARIANT_KEYS.get(s)
        if vk:
            var: dict = {}
            for i in ix:
                v = str(rows[i]["meta"].get(vk))
                var.setdefault(v, [0, 0])
                var[v][0] += 1
                var[v][1] += sc[i]["correct"]
            d["variants"] = {v: {"n": n, "acc": c / n} for v, (n, c) in sorted(var.items())}
        out[s] = d
    return out


def id_diff(rows: list, sc_on: list, sc_ref: list) -> dict:
    ix = [i for i, r in enumerate(rows) if r["stratum"] in C.ID_STRATA]
    a = sum(sc_on[i]["correct"] for i in ix) / len(ix)
    b = sum(sc_ref[i]["correct"] for i in ix) / len(ix)
    by_base: dict = {}
    for i in ix:
        d = by_base.setdefault(rows[i]["base"], [0, 0])
        d[0] += int(sc_on[i]["correct"]) - int(sc_ref[i]["correct"])
        d[1] += 1
    bl = C.boot_lower([tuple(v) for _, v in sorted(by_base.items())])
    return {"n": len(ix), "acc_on": a, "acc_t2s0": b, "diff": a - b, "boot_lower_95_one_sided": bl["lo"]}


def metrics(rows: list, texts: dict) -> dict:
    """texts: {'off': [...], 't2s0': [...], 's0': [...], 's1': [...], 's2': [...]} in row order."""
    sc = {k: score_side(rows, v) for k, v in texts.items()}
    res = {"strata": {k: stratum_metrics(rows, sc[k], texts[k]) for k in texts}, "id": {}}
    for s in SEEDS:
        k = f"s{s}"
        if k in sc:
            res["id"][k] = id_diff(rows, sc[k], sc["t2s0"])
    res["id"]["off_vs_t2s0"] = id_diff(rows, sc["off"], sc["t2s0"])
    return res


# ============================================================================================== the rule
def decide(M: dict, instrument: dict) -> dict:
    """The pre-stated rule (RULE_TEXT) applied mechanically."""
    if not instrument.get("complete") or not instrument.get("anchors_passed"):
        return {"verdict": "NO_VERDICT_INSTRUMENT", "reason": instrument.get("reason") or "incomplete run or failed anchor",
                "per_seed": {}, "patterns": []}
    per = {}
    for s in SEEDS:
        k = f"s{s}"
        st = M["strata"][k]
        g1 = st["HS_words"]["acc"] >= BAR_HS
        g2 = M["id"][k]["diff"] >= -NI_MARGIN - 1e-12
        g3 = st["HS_posted"]["copied_rate"] <= COPY_CAP + 1e-12
        per[k] = {"G1": g1, "G2": g2, "G3": g3, "HS_words": st["HS_words"]["acc"], "id_diff": M["id"][k]["diff"],
                  "HS_posted_copied": st["HS_posted"]["copied_rate"], "S_words": st["S_words"]["acc"]}
    if all(v["G1"] and v["G2"] and v["G3"] for v in per.values()):
        return {"verdict": "PROCEED", "per_seed": per, "patterns": []}
    if all(v["S_words"] >= TRIGGER_TRAINED_MIN and v["HS_words"] <= TRIGGER_HELDOUT_MAX for v in per.values()):
        return {"verdict": "TRIGGER_WIDENED", "per_seed": per, "patterns": []}
    pats = []
    if any(not v["G1"] for v in per.values()):
        pats.append("heldout_below_bar")
    if any(not v["G2"] for v in per.values()):
        pats.append("in_distribution_cost")
    if any(not v["G3"] for v in per.values()):
        pats.append("copy_on_heldout_surface")
    if any(v["S_words"] < TRIGGER_TRAINED_MIN for v in per.values()):
        pats.append("trained_words_below_bar")
    if len({v["G1"] for v in per.values()}) > 1:
        pats.append("seed_disagreement")
    return {"verdict": "PARTIAL", "per_seed": per, "patterns": pats}


def _rng(xs) -> str:
    xs = [round(x, 2) for x in xs]
    return f"{min(xs):.2f}" if min(xs) == max(xs) else f"{min(xs):.2f}-{max(xs):.2f}"


def wording(D: dict, M: dict | None) -> dict:
    v = D["verdict"]
    if v == "NO_VERDICT_INSTRUMENT":
        return {"headline": "DEV GATE: NO VERDICT (instrument)", "sentence": f"No gate verdict: {D['reason']}.",
                "next": "Investigate, log in DECISIONS.md; the dev bank is not re-read for any decision."}
    per = D["per_seed"]
    hs = [per[f"s{s}"]["HS_words"] for s in SEEDS]
    idd = [per[f"s{s}"]["id_diff"] for s in SEEDS]
    cp = [per[f"s{s}"]["HS_posted_copied"] for s in SEEDS]
    sw = [per[f"s{s}"]["S_words"] for s in SEEDS]
    t2 = M["strata"]["t2s0"]["HS_words"]["acc"]
    off = M["strata"]["off"]["HS_words"]["acc"]
    seedlist = lambda key: ", ".join(f"s{s} {per[f's{s}'][key]:.2f}" for s in SEEDS)  # noqa: E731
    if v == "PROCEED":
        return {"headline": "DEV GATE: PROCEED",
                "sentence": (f"In all three T2b-ledger seeds, on the combined held-out surface (rule paraphrase 4, held-out "
                             f"account names, held-out key orders) the adapter left held-out non-posted statuses unchanged "
                             f"on {_rng(hs)} of rows (each >= {BAR_HS:.2f}; T2 s0 {t2:.2f}, OFF {off:.2f}), copied the "
                             f"posted twins on {_rng(cp)} (each <= {COPY_CAP:.2f}), and its in-distribution accuracy was "
                             f"within the margin of T2 s0's (ON - T2 s0: {seedlist('id_diff')}; each >= -{NI_MARGIN:.2f})."),
                "next": "The sealed confirmation PREREG-5 may be frozen (operator step). This is a dev screen, not a claim."}
    if v == "TRIGGER_WIDENED":
        return {"headline": "DEV GATE: STOP (TRIGGER WIDENED)",
                "sentence": (f"In all three seeds the adapter treated the TRAINED non-posted words as no-ops ({_rng(sw)}, "
                             f"each >= {TRIGGER_TRAINED_MIN:.2f}) but HELD-OUT words on the combined held-out surface only "
                             f"on {_rng(hs)} of rows (each <= {TRIGGER_HELDOUT_MAX:.2f}; T2 s0 {t2:.2f}): the surface "
                             "augmentation widened the trigger set; it did not teach the stated rule."),
                "next": "No confirmation is drafted from this kit (revision path in the rule)."}
    parts = []
    if "heldout_below_bar" in D["patterns"]:
        parts.append(f"held-out words on the combined held-out surface below {BAR_HS:.2f} ({seedlist('HS_words')})")
    if "in_distribution_cost" in D["patterns"]:
        parts.append(f"in-distribution accuracy more than {NI_MARGIN:.2f} below T2 s0's (ON - T2 s0: {seedlist('id_diff')})")
    if "copy_on_heldout_surface" in D["patterns"]:
        parts.append(f"posted twins on the held-out surface copied above {COPY_CAP:.2f} ({seedlist('HS_posted_copied')})")
    if "trained_words_below_bar" in D["patterns"]:
        parts.append(f"even the trained non-posted words below {TRIGGER_TRAINED_MIN:.2f} ({seedlist('S_words')})")
    if "seed_disagreement" in D["patterns"]:
        parts.append("the seeds disagree on G1")
    return {"headline": "DEV GATE: STOP (PARTIAL)",
            "sentence": "Neither PROCEED nor TRIGGER WIDENED: " + "; ".join(parts) + ". The per-stratum, per-variant table "
                        "follows.", "next": "No confirmation is drafted from this kit (revision path in the rule)."}


def render_md(res: dict) -> str:
    L = [f"# T2b-ledger dev gate: {res['wording']['headline']}", "", res["wording"]["sentence"], "",
         res["wording"]["next"], "", "## Pre-stated rule (verbatim)", "", "> " + RULE_TEXT, "", "## Gate per seed", "",
         "| seed | G1 HS_words | G2 ID diff vs T2 s0 (boot LB) | G3 HS_posted copied | S_words |", "|---|---:|---:|---:|---:|"]
    M = res.get("metrics")
    for k, v in (res["decision"].get("per_seed") or {}).items():
        lb = M["id"][k]["boot_lower_95_one_sided"] if M else float("nan")
        L.append(f"| {k} | {v['HS_words']:.3f} ({'pass' if v['G1'] else 'FAIL'}) | {v['id_diff']:+.3f} ({lb:+.3f}) "
                 f"({'pass' if v['G2'] else 'FAIL'}) | {v['HS_posted_copied']:.3f} ({'pass' if v['G3'] else 'FAIL'}) | "
                 f"{v['S_words']:.3f} |")
    if M:
        sides = [k for k in ("off", "t2s0", "s0", "s1", "s2") if k in M["strata"]]
        L += ["", "## Accuracy per stratum and side (descriptive)", "", "| stratum | " + " | ".join(sides) + " |",
              "|---|" + "---:|" * len(sides)]
        for s in C.DEV_STRATA:
            if s in M["strata"]["off"]:
                L.append(f"| {s} | " + " | ".join(f"{M['strata'][k][s]['acc']:.2f}" for k in sides) + " |")
        L += ["", "## Variants (descriptive)", ""]
        for s, vk in VARIANT_KEYS.items():
            if s not in M["strata"]["off"]:
                continue
            vs = sorted(M["strata"]["off"][s].get("variants", {}))
            L.append(f"- **{s}** by {vk}: " + "; ".join(
                f"{v}: " + ", ".join(f"{k} {M['strata'][k][s]['variants'][v]['acc']:.2f}" for k in sides) for v in vs))
    return "\n".join(L) + "\n"


# ============================================================================================== registered analysis
def load_run(run: Path, rows: list, registered: bool) -> tuple:
    import t2bl_run as R                             # torch-free at import
    run = Path(run)
    need((run / "DONE.json").exists(), f"{run}/DONE.json missing: the run is not complete")
    prov = C.read_json(run / "provenance.json")
    need(prov is not None, "provenance.json unreadable")
    done = C.read_json(run / "DONE.json")
    need(done.get("provenance_sha256") == C.sha256(run / "provenance.json"), "provenance.json changed after DONE.json")
    cfg = R.Cfg(out=run, seeds=tuple(prov["seeds"]), rule_record=None)
    st = R.stage_states(prov, run, cfg)
    need(all(v == "done" for u in st.values() for v in u.values()), "not every stage is recorded and unchanged")
    need(prov.get("dev_subset") in (None, 0) or not registered, "a registered run uses the whole dev bank")
    need(prov.get("dev_ids_sha256") == C.sha256_text("\n".join(r["id"] for r in rows)), "the run's dev ids differ from the bank")
    texts = {}
    for side in R.SIDES_E0:
        recs = C.load_jsonl(run / "ref" / f"dev_{side}.jsonl")
        need([x["id"] for x in recs] == [r["id"] for r in rows], f"ref/{side}: ids not in bank order")
        texts[side] = [x["text"] for x in recs]
    for s in prov["seeds"]:
        recs = C.load_jsonl(run / f"s{s}" / "dev_on.jsonl")
        need([x["id"] for x in recs] == [r["id"] for r in rows], f"s{s}: ids not in bank order")
        texts[f"s{s}"] = [x["text"] for x in recs]
    turns = prov.get("turns") or []
    eval_turns = {rec["turn"] for u in prov["stages"].values() for k, rec in u.items() if k in ("E0", "E")}
    anchors = all((turns[t].get("anchor") or {}).get("pass") for t in eval_turns)
    inst = {"complete": True, "anchors_passed": anchors, "eval_turns": sorted(eval_turns),
            "anchor_records": [turns[t].get("anchor") for t in sorted(eval_turns)],
            "reason": None if anchors else "an evaluation turn has no passing anchor"}
    if registered:
        code_now = {r: C.sha256(ROOT / r) for r in prov.get("code_sha256", {})}
        need(code_now == prov.get("code_sha256"), "runner code differs from the run's recorded code")
    return texts, inst, prov


def verify_rule_record(path: Path = C.DOCS / "GATE_RULE.json") -> list:
    import t2bl_run as R
    rec = C.read_json(path)
    probs = R.frozen_problems(rec, R.frozen_files(R.Cfg(rule_record=path)))
    if rec and rec.get("rule_text") != RULE_TEXT:
        probs.append("the rule text differs from the frozen record")
    md = C.DOCS / "GATE_RULE.md"
    if not md.exists() or RULE_TEXT not in md.read_text(encoding="utf-8"):
        probs.append("GATE_RULE.md does not contain the rule text verbatim")
    return probs


def analyze(run: Path, registered: bool = True, rows: list | None = None, out_dir: Path | None = None) -> dict:
    if registered:
        probs = verify_rule_record()
        need(not probs, "the frozen gate rule does not verify: " + "; ".join(probs[:5]))
    rows = rows if rows is not None else C.load_jsonl(C.DEV)
    st = selftest(rows)
    need(st["ok"], f"scorer self-test failed: {st['failures'][:3]}")
    out_dir = Path(out_dir or Path(run) / "analysis")
    need(not (out_dir / "GATE.json").exists(), f"{out_dir / 'GATE.json'} exists (create-only; the analysis runs once)")
    texts, inst, prov = load_run(Path(run), rows, registered)
    M = metrics(rows, texts)
    D = decide(M, inst)
    W = wording(D, M)
    res = {"record": "T2BL_GATE", "rule_version": RULE_VERSION, "rule": RULE_TEXT, "decision": D, "wording": W,
           "metrics": M, "instrument": inst, "selftest": {k: v for k, v in st.items() if k != "failures"},
           "registered": registered, "code_sha256": C.code_hashes(ANALYSIS_CODE), "run": str(run),
           "analysed_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    out_dir.mkdir(parents=True, exist_ok=True)
    C.write_create(out_dir / "GATE.json", json.dumps(res, indent=1) + "\n")
    C.write_create(out_dir / "GATE.md", render_md(res))
    return res


# ============================================================================================== dry run (synthetic)
def synth_texts(rows: list, acc: dict, seed: int, copy_posted: float = 0.0) -> list:
    """Synthetic answers: per stratum, the first round(acc * n) rows (in a seeded order) get the gold, the rest the
    stratum's typical error (no-op rows: the event applied; apply rows: a wrong value), except that on HS_posted a
    share copy_posted of the rows copy the input."""
    rng = random.Random(seed)
    out = [None] * len(rows)
    by: dict = {}
    for i, r in enumerate(rows):
        by.setdefault(r["stratum"], []).append(i)
    for s, ix in by.items():
        ix = list(ix)
        rng.shuffle(ix)
        a = acc.get(s, acc.get("*", 1.0))
        k = round(a * len(ix))
        n_copy = round(copy_posted * len(ix)) if s == "HS_posted" else 0
        for j, i in enumerate(ix):
            r = rows[i]
            if j < k:
                out[i] = r["gold"]
            elif j < k + n_copy and r["gold_rule"] == "apply":
                out[i] = env.dumps(r["before"])
            elif r["gold_rule"] == "noop":
                out[i] = r["posted_gold"]
            else:
                g = json.loads(r["gold"])
                k0 = sorted(g)[0]
                out[i] = env.dumps({**g, k0: g[k0] + 3})
    return out


def scenario(rows: list, spec: dict) -> dict:
    texts = {"off": synth_texts(rows, spec["off"], 1), "t2s0": synth_texts(rows, spec["t2s0"], 2)}
    for s in SEEDS:
        sp = spec["seeds"][s]
        texts[f"s{s}"] = synth_texts(rows, sp, 10 + s, sp.get("_copy", 0.0))
    M = metrics(rows, texts)
    D = decide(M, spec.get("inst", {"complete": True, "anchors_passed": True}))
    return {"M": M, "D": D, "W": wording(D, M)}


def dryrun(rows: list) -> dict:
    base_t2 = {"*": 0.9, "HS_words": 0.03, "H_words": 0.05, "S_words": 0.05, "S_surface_words": 0.03, "P3_words": 0.05,
               "LB_void": 0.03, "ID_posted": 0.87, "ID_void": 0.99}
    off = {"*": 0.3, "ID_void": 0.05}
    good = {"*": 0.95, "ID_posted": 0.90, "ID_void": 0.99}
    cases = []

    def run_case(name, seeds, want, patterns=None, inst=None, check=None):
        spec = {"off": off, "t2s0": base_t2, "seeds": seeds}
        if inst:
            spec["inst"] = inst
        r = scenario(rows, spec)
        ok = r["D"]["verdict"] == want and (patterns is None or sorted(r["D"]["patterns"]) == sorted(patterns))
        if check:
            ok = ok and check(r)
        txt = r["W"]["headline"] + " " + r["W"]["sentence"]
        ok = ok and {"PROCEED": "PROCEED", "TRIGGER_WIDENED": "TRIGGER WIDENED", "PARTIAL": "PARTIAL",
                     "NO_VERDICT_INSTRUMENT": "NO VERDICT"}[want] in r["W"]["headline"]
        cases.append({"case": name, "want": want, "got": r["D"]["verdict"], "patterns": r["D"]["patterns"], "ok": ok,
                      "headline": r["W"]["headline"], "sentence": r["W"]["sentence"][:400]})

    run_case("all seeds good -> PROCEED", {s: good for s in SEEDS}, "PROCEED")
    run_case("HS_words exactly 0.80 in every seed -> PROCEED (boundary)",
             {s: {**good, "HS_words": 0.80} for s in SEEDS}, "PROCEED")
    run_case("HS_words 0.79 in s1 -> PARTIAL heldout_below_bar + seed_disagreement",
             {0: good, 1: {**good, "HS_words": 0.79}, 2: good}, "PARTIAL", ["heldout_below_bar", "seed_disagreement"])
    run_case("ID 0.05 below T2 s0 exactly -> PROCEED (boundary)",
             {s: {**good, "ID_posted": 0.77, "ID_void": 0.99} for s in SEEDS}, "PROCEED",
             check=lambda r: all(abs(r["D"]["per_seed"][f"s{s}"]["id_diff"] + 0.05) < 1e-9 for s in SEEDS))
    run_case("ID 0.06 below T2 s0 in s2 -> PARTIAL in_distribution_cost",
             {0: good, 1: good, 2: {**good, "ID_posted": 0.75, "ID_void": 0.99}}, "PARTIAL", ["in_distribution_cost"])
    run_case("held-out posted twins copied 0.10 -> PROCEED (boundary)",
             {s: {**good, "HS_posted": 0.90, "_copy": 0.10} for s in SEEDS}, "PROCEED")
    run_case("held-out posted twins copied 0.12 in s0 -> PARTIAL copy_on_heldout_surface",
             {0: {**good, "HS_posted": 0.85, "_copy": 0.12}, 1: good, 2: good}, "PARTIAL", ["copy_on_heldout_surface"])
    run_case("trained words learned, held-out words base-like -> TRIGGER WIDENED",
             {s: {**good, "HS_words": 0.10, "H_words": 0.12} for s in SEEDS}, "TRIGGER_WIDENED")
    run_case("held-out words exactly 0.30 -> TRIGGER WIDENED (boundary)",
             {s: {**good, "HS_words": 0.30} for s in SEEDS}, "TRIGGER_WIDENED")
    run_case("held-out words 0.31 in one seed -> PARTIAL (not trigger)",
             {0: {**good, "HS_words": 0.31}, 1: {**good, "HS_words": 0.2}, 2: {**good, "HS_words": 0.2}}, "PARTIAL",
             ["heldout_below_bar"])
    run_case("trained words below bar and held-out low -> PARTIAL trained_words_below_bar",
             {s: {**good, "HS_words": 0.10, "S_words": 0.6} for s in SEEDS}, "PARTIAL",
             ["heldout_below_bar", "trained_words_below_bar"])
    run_case("everything fails -> PARTIAL with every pattern but seed disagreement",
             {s: {**good, "HS_words": 0.5, "S_words": 0.5, "ID_posted": 0.6, "HS_posted": 0.7, "_copy": 0.3} for s in SEEDS},
             "PARTIAL", ["heldout_below_bar", "in_distribution_cost", "copy_on_heldout_surface", "trained_words_below_bar"])
    run_case("failed anchor -> NO VERDICT", {s: good for s in SEEDS}, "NO_VERDICT_INSTRUMENT",
             inst={"complete": True, "anchors_passed": False, "reason": "an evaluation turn has no passing anchor"})
    run_case("incomplete run -> NO VERDICT", {s: good for s in SEEDS}, "NO_VERDICT_INSTRUMENT",
             inst={"complete": False, "anchors_passed": True, "reason": "DONE.json missing"})
    # a pure copier passes G1 (no-op) but not G3
    run_case("a copier on the held-out surface -> PARTIAL copy_on_heldout_surface",
             {s: {**good, "HS_posted": 0.0, "_copy": 1.0} for s in SEEDS}, "PARTIAL", ["copy_on_heldout_surface"])
    return {"ok": all(c["ok"] for c in cases), "n_cases": len(cases), "cases": cases,
            "patterns_covered": sorted({p for c in cases for p in c["patterns"]}),
            "verdicts_covered": sorted({c["got"] for c in cases}), "code_sha256": C.code_hashes(ANALYSIS_CODE)}


# ============================================================================================== rule records
def rule_md() -> str:
    return ("# T2b-ledger dev gate rule (revision 1; pre-stated before any T2b-ledger output)\n\n"
            "Status: PRE-STATED, NOT FROZEN. Freezing writes `GATE_RULE.json` (sha256 of every frozen input) via "
            "`python -B scripts/t2bl_gate.py --freeze-rule --approval \"...\"`, an operator step logged in DECISIONS.md; "
            "the runner refuses to start without it. The same record pins the sha256 of PREREG-5's decision rules, wording "
            "templates, limits and thresholds (`t2bl_confirm_analysis.rules_digest()`), so PREREG-5's freeze can show "
            "whether they changed after the dev screen's outputs existed.\n\n## Rule (verbatim; `t2bl_gate.RULE_TEXT`)\n\n> "
            + RULE_TEXT +
            "\n\n## Thresholds\n\n| clause | quantity | threshold |\n|---|---|---|\n"
            f"| G1 | ON accuracy, HS_words (held-out words, combined held-out surface) | >= {BAR_HS:.2f} |\n"
            f"| G2 | ON - T2 s0 accuracy on ID (ID_posted + ID_void), point estimate | >= -{NI_MARGIN:.2f} |\n"
            f"| G3 | ON copied rate on HS_posted | <= {COPY_CAP:.2f} |\n"
            f"| TRIGGER WIDENED | S_words >= {TRIGGER_TRAINED_MIN:.2f} and HS_words <= {TRIGGER_HELDOUT_MAX:.2f}, every seed | |\n"
            "\n## Why these choices\n\n"
            "- **G1 at 0.80** is the programme's bar (PREREG-3). HS_words is the hardest dev stratum: every surface factor "
            "is held out at once, and the status word is held out too. T2 s0 is expected near 0 there (PREREG-3: unseen "
            "statuses 0.01-0.18 on the canonical surface; PREREG-LB: 5/180 plans on a held-out surface).\n"
            "- **G2 at -0.05** is the task's margin against T2 s0 on T2's own surface. It is a point estimate, because this "
            "is a screen; PREREG-5 tests non-inferiority with an interval.\n"
            "- **G3 (added to the task's two clauses)**: a no-op accuracy alone is met by a model that copies every "
            "unfamiliar input. The copy guard on the posted twins in the same surface rules that out, as PREREG-3's twin "
            "guard did (copied <= 0.10).\n"
            "- **TRIGGER WIDENED** uses PREREG-3's trigger threshold (0.30) for 'stays near T2 s0'.\n"
            "- **All three seeds.** The question is about the recipe, not one artifact.\n")


def freeze_rule(approval: str) -> dict:
    import t2bl_run as R
    need(len(approval.strip()) >= 12, "--approval must name who approved, when, and the DECISIONS entry")
    need(not C.run_outputs(), "a runs/t2bl output exists: the rule can no longer be frozen")
    p = C.DOCS / "GATE_RULE.json"
    need(not p.exists(), "GATE_RULE.json exists (create-only)")
    md = C.DOCS / "GATE_RULE.md"
    need(md.exists() and RULE_TEXT in md.read_text(encoding="utf-8"), "GATE_RULE.md missing or not the rule text")
    st = selftest(C.load_jsonl(C.DEV))
    need(st["ok"], "scorer self-test fails")
    files = R.frozen_files(R.Cfg(rule_record=p))
    import t2bl_confirm_analysis as CA               # numpy only: PREREG-5's rules are pinned before any dev output
    rec = {"record": "T2BL_GATE_RULE", "rule_version": RULE_VERSION, "rule_text": RULE_TEXT, "approval": approval,
           "frozen_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), "inputs": {rel: C.sha256(f) for rel, f in files.items()},
           "prereg5_rules": {**CA.rules_digest(), "note": "sha256 of PREREG-5's registered rules, wording templates, "
                                                          "limits and thresholds (t2bl_confirm_analysis.rules_digest) at "
                                                          "this freeze; PREREG-5's freeze records whether they changed"},
           "prereg5_draft_sha256": C.sha256(C.ROOT / "PREREG_5_DRAFT.md") if (C.ROOT / "PREREG_5_DRAFT.md").is_file() else None}
    C.write_create(p, json.dumps(rec, indent=1) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--write-rule-md", action="store_true")
    ap.add_argument("--freeze-rule", action="store_true")
    ap.add_argument("--approval", default="")
    ap.add_argument("--run", default=str(C.OUT_ROOT / "dev"))
    a = ap.parse_args()
    try:
        if a.selftest:
            res = selftest(C.load_jsonl(C.DEV))
            print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=1))
            if a.record:
                print("wrote", C.write_doc("SCORER_SELFTEST.json", json.dumps({**res, "code_sha256": C.code_hashes(ANALYSIS_CODE)}, indent=1) + "\n"))
            return 0 if res["ok"] else 1
        if a.dryrun:
            res = dryrun(C.load_jsonl(C.DEV))
            for c in res["cases"]:
                print(("PASS " if c["ok"] else "FAIL ") + c["case"] + " -> " + c["got"] + " " + str(c["patterns"]))
            if a.record:
                print("wrote", C.write_doc("GATE_DRYRUN.json", json.dumps(res, indent=1) + "\n"))
            return 0 if res["ok"] else 1
        if a.write_rule_md:
            need(not C.run_outputs(), "a runs/t2bl output exists: the rule document is not rewritten")
            print("wrote", C.write_doc("GATE_RULE.md", rule_md()))
            return 0
        if a.freeze_rule:
            rec = freeze_rule(a.approval)
            print(json.dumps({k: v for k, v in rec.items() if k != "rule_text"}, indent=1))
            return 0
        res = analyze(Path(a.run))
        print(res["wording"]["headline"])
        print(res["wording"]["sentence"])
        return 0
    except GateError as e:
        print(f"[t2bl-gate] {e}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
