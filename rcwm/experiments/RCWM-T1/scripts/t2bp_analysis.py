"""RCWM-T2b prerequisite representation screen: analysis (CPU only; numpy/json; torch is never imported).

  python -B scripts/t2bp_analysis.py --dry-run       # synthetic outcomes only (no run output read): DRYRUN.json and
                                                     # SCORER_SELFTEST.json in docs/t2b/prereq (--no-record: neither)
  python -B scripts/t2bp_analysis.py --freeze-rule   # writes SCREEN_RULE.md + SCREEN_RULE.json (every frozen input's
                                                     # sha256); refuses once any runs/t2bp output exists, or unless the
                                                     # dry run, the tiny test and the audit records match the code
  python -B scripts/t2bp_analysis.py                 # runs/t2bp/s0 -> runs/t2bp/s0/analysis/T2BP_SCREEN.{json,md}
  python -B scripts/t2bp_analysis.py --filler-run runs/t2bp/s0_ffill
                                                     # R5 addendum (only after an S-trace win) ->
                                                     # runs/t2bp/s0_ffill/analysis/T2BP_FILLER.{json,md}
Every output is create-only. Before any run output is read: the frozen inputs (SCREEN_RULE.json: this file, the common
module, the runner, the data builder, the rcwmt modules the runner and the scorer use, the vendored trainer, the dev
bank and the training files) must be unchanged (else exit 3), and the scorer self-test must pass (golds 100% in every
arm's format, every heuristic and shortcut policy scored at its exact floor, malformed answers rejected; else exit 3).
A run is scored only if DONE.json exists, every output's sha256 equals its provenance record, one-step ids are the dev
bank's ids in order, chain rows are the dev chains in order, and every recorded chain state equals the arm parser
applied to the recorded text.

PRE-STATED SCREEN RULE, revision 2 (T2b prerequisite representation screen, seed 0). Revision 1 was written 2026-10-09
before any runs/t2bp output existed. Revision 2 replaces it after review (a mirror false-apply gate, IAW >= 0.20 for the
named arm, an instrument-validity gate, false copy on the 8-bit rows, the filler control as its own run), also before
any runs/t2bp output existed and without any model output being read. Its text and the sha256 of every frozen input
are recorded in docs/t2b/prereq/SCREEN_RULE.json; the runner and this analysis refuse when a frozen input differs.
Verbatim in docs/t2b/prereq/SCREEN_RULE.md.

  Inputs. Adapter ON, seed 0, each screen arm A in {D-int, S-trace, B-bits}, on the dev screen bank data/t2bp_dev.jsonl
  (1,050 one-step rows in 9 strata; 240 k=4 chains). All three arms must be complete (DONE.json; every output
  hash-verified; the frozen inputs unchanged; the scorer self-test passes before any output is read); otherwise no
  verdict is issued. Answers are scored by the arm's after-only parser (t2bp_common.parse: D-int lenient exactly as
  rcwmt.env.parse_state; the others strict); an unparseable or malformed answer is wrong.
  Metrics (point estimates, adapter ON; "8-bit rows" = the 750 one-step rows outside D_ood10 and D_ood12):
    IAW(A)  accuracy on D_int_add_wrong (150 rows: applicable, the add bit is already set and the delete clears a set
            bit, so the bitwise answer differs from before+add, from (before&~delete)+add and from before);
    OOR(A)  mean of the accuracies on D_ood10 and D_ood12 (10- and 12-bit masks; out of range; primary per D-52 (5));
    K4(A)   k=4 exact final-state accuracy over the 240 dev chains (self-rollout in A's own frame; a broken chain is
            wrong);
    FC(A)   false copy (T2's definition): the share of answers equal to the input state among the 350 8-bit rows whose
            gold differs from the input (D_app_add, D_app_delete, D_int_add_wrong);
    FA(A)   false apply (the mirror of FC): the LARGEST, over the inapplicable strata D_inapp_pre_on (100),
            D_inapp_pre_off (100) and D_inapp_addset (100; the add bit is already set and the delete clears a set bit:
            the inapplicable twin of D_int_add_wrong), of the share of answers that are not the input state (on every
            such row the gold is the input and applying the action would change it; a malformed answer counts);
    WF(A)   the share of answers in the arm's format on the 750 8-bit rows (t2bp_common.format_ok: D-int, the lenient
            parser finds an integer; S-trace, the literal keys and an integer after; B-bits, the literal keys, a
            boolean flag and a binary string; the value is not checked here, so an out-of-range state such as
            200 + 64 = 264 is a wrong answer, not a format failure);
    ID(A)   the mean of the accuracies of the seven 8-bit strata (D_app_add, D_app_unchanged, D_app_delete,
            D_int_add_wrong, D_inapp_pre_on, D_inapp_pre_off, D_inapp_addset), equal weight.
  R0 (instrument validity). If WF(A) < 0.90 for any screen arm, or a turn that ran an evaluation stage has no passed
     anchor record: verdict NO_VERDICT_INSTRUMENT. This is an instrument or format failure, not a result: no programme
     decision follows; it is investigated and logged in DECISIONS.md.
  R1 (decision point 2). If IAW(A) < 0.20 for every screen arm: verdict STOP_P_TRAINING. Stop training the
     prerequisite family and move P to a code-side bit interface (code converts the integer state and the masks and
     applies the bitwise rule), reported as an interface, not as model skill. Every statement of it quotes the dose:
     150 of the 1,500 training rows were int-add-wrong (A4) rows. No arm is named.
  R2. Otherwise an arm is ELIGIBLE iff IAW(A) >= 0.20 (decision point 2 holds for the named arm itself), FC(A) <= 0.25,
     FA(A) <= 0.25 and ID(A) >= ID(D-int) - 0.10 (non-inferiority against the data control; D-int meets this last
     condition by definition).
  R3. If no arm is eligible: verdict NO_ELIGIBLE_ARM (in every arm, int-add-wrong below 0.20, the copy shortcut, a
     false-apply shortcut or an in-distribution collapse); no arm is named for seeds 1-2.
  R4. Otherwise rank the eligible arms by R(A) = (K4(A) + IAW(A) + OOR(A)) / 3. The winner is the arm with the largest
     R; if the top two differ by less than 0.02 in R, the one with the larger IAW wins; if those differ by less than
     0.02, the one with the larger OOR; if still tied, the first in the order D-int, B-bits, S-trace (fewest decode
     tokens). Verdict WINNER=<arm>: that arm (its rendering, the same rows, the T2 recipe) runs seeds 1 and 2. If the
     winner is D-int, the result is worded "the representation does not beat the rebalanced-data control". If another
     eligible arm's R is within 0.05 of the winner's (about two standard errors at these n), the result is worded as a
     close call: a screen choice, not a finding.
  R5 (decode-compute control). F-filler is a control and is never eligible. It runs only if S-trace wins, as its own
     run (tag s0_ffill: seed 0, the same rows, the T2 recipe; analysed by --filler-run), unless it was part of the s0
     run. If R(F-filler) >= R(S-trace) - 0.05, the S-trace result carries the flag
     TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE (the winner is unchanged; the flag is quoted with every statement of
     the result). If WF(F-filler) < 0.90 or a filler turn lacks a passed anchor, the control is void and the result
     carries DECODE_COMPUTE_CONTROL_VOID. Until the control is analysed, an S-trace win is stated with "decode-compute
     control pending".
  Everything else (OFF baselines, the other strata and subtypes, out-of-range false copy, union cells, shortcut
  agreement, cmp_ok / calc_ok / flag_ok, decision regret, paired contrasts, the held-out-surface stage E3 of a
  surface-diversity run) is descriptive and cannot change the verdict. This is a screen on a dev bank: no confirmatory
  claim follows from it; a confirmatory claim needs a sealed bank and a preregistration (roadmap item 2, "Confirm").
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import warnings  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _stub_heads():
    """rcwmt.decisions/fidelity import rcwmt.heads (torch). g_only and heuristic_states never use heads (as t2_analysis)."""
    if "rcwmt.heads" not in sys.modules:
        m = types.ModuleType("rcwmt.heads")
        m.__doc__ = "placeholder installed by t2bp_analysis.py: g_only does not use heads; keeps torch out"
        sys.modules["rcwmt.heads"] = m


_stub_heads()
import t2bp_common as C  # noqa: E402
from rcwmt import data, decide, decisions, env, fidelity  # noqa: E402

RUN = C.OUT_ROOT / "s0"
FILLER_RUN = C.OUT_ROOT / C.FILLER_TAG
IAW_STOP = 0.20
FC_MAX = 0.25
FA_MAX = 0.25
WF_MIN = 0.90
ID_MARGIN = 0.10
TIE_R, TIE_IAW = 0.02, 0.02
CLOSE_CALL = 0.05
FILLER_FLAG_MARGIN = 0.05
TIE_ORDER = ("dint", "bbits", "strace")
FLAG_FILLER = "TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE"
FLAG_VOID = "DECODE_COMPUTE_CONTROL_VOID"
EVAL_STAGES = ("E1", "E2", "E3")
UNIONS = {"applicable state-changing (T2 definition)": ("D_app_add", "D_app_delete", "D_int_add_wrong"),
          "app-del (L1/T2 definition)": ("D_app_delete", "D_int_add_wrong"),
          "int-add-wrong (R11 definition: applicable, gold != before+add)": ("D_app_unchanged", "D_app_delete", "D_int_add_wrong"),
          "inapplicable (8-bit)": C.INAPP_STRATA,
          "add bit already set and delete hits (A4 + its inapplicable twin)": ("D_int_add_wrong", "D_inapp_addset"),
          "out-of-range (10/12-bit)": ("D_ood10", "D_ood12")}
BOOT = 2000
DOSE = f"{C.N_A4_TRAIN} of the {C.N_TRAIN:,} training rows were int-add-wrong (A4) rows"
RULE_HEAD = "PRE-STATED SCREEN RULE"


class Refuse(Exception):
    pass


def need(cond: bool, msg: str) -> None:
    if not cond:
        raise Refuse(f"[t2bp-analysis] {msg}")


# ---------------------------------------------------------------- scorer self-test (before any output is read)
def scorer_selftest(dev_rows: list, chains: list) -> dict:
    count = None
    try:
        import tokenizers
        tk = tokenizers.Tokenizer.from_file(str(C.TOKENIZER_JSON))
        count = lambda s: len(tk.encode(s, add_special_tokens=False).ids)   # noqa: E731
    except Exception:                                                          # noqa: BLE001
        count = None
    res = {"golds": {}, "heuristics": {}, "shortcuts": {}, "negatives": {}, "dint_parser_equals_env": None}
    arms = C.ARMS if count is not None else C.SCREEN_ARMS
    for a in arms:
        ok = 0
        mech = Counter()
        for r in dev_rows:
            sc = C.score_row(a, C.target_text(a, r["before"], r["event"], r["width"], count), r)
            ok += int(sc["correct"] and sc["well_formed"])
            for k in ("cmp_ok", "calc_ok", "flag_ok"):
                if k in sc:
                    mech[k] += int(sc[k])
        res["golds"][a] = {"n": len(dev_rows), "correct": ok, **dict(mech)}
        for kind, fns in (("heuristics", {h: (lambda b, e, h=h: C.heuristics(b, e)[h])
                                          for h in list(C.AUDITED) + list(C.DESCRIPTIVE_HEURISTICS)}),
                          ("shortcuts", C.SHORTCUT_POLICIES)):
            hh = {}
            for h, f in fns.items():
                direct = scored = n_in = 0
                for r in dev_rows:
                    v = f(r["before"], r["event"])
                    direct += int(v == r["gold"])
                    if 0 <= v < 2 ** r["width"]:
                        n_in += 1
                        scored += int(C.score_row(a, C.render_prediction(a, v, r["before"], r["event"], r["width"]), r)["correct"])
                hh[h] = {"direct_floor": direct, "scored_in_format": scored, "out_of_range_predictions": len(dev_rows) - n_in,
                         "match": direct == scored}
            res[kind][a] = hh
    r0 = dev_rows[0]
    w, b, ev, g = r0["width"], r0["before"], r0["event"], r0["gold"]
    tr = C.trace_text(b, ev)
    neg = {
        "strace: keys reordered": '{"calc":' + str(g) + ',"cmp":' + json.dumps(C.gold_cmp(b, ev), separators=(",", ":")) + ',"after":' + str(g) + "}",
        "strace: extra key": tr[:-1] + ',"note":1}',
        "strace: duplicate after": tr[:-1] + ',"after":' + str(g) + "}",
        "strace: after as string": tr.replace(',"after":' + str(g) + "}", ',"after":"' + str(g) + '"}'),
        "strace: after out of range": tr.replace(',"after":' + str(g) + "}", ',"after":' + str(2 ** w) + "}"),
        "strace: bare integer": str(g),
        "bbits: keys reordered": '{"after":"' + C.bitstr(g, w) + '","applicable":true}',
        "bbits: wrong width": '{"applicable":true,"after":"' + C.bitstr(g, w)[1:] + '"}',
        "bbits: integer after": '{"applicable":true,"after":' + str(g) + "}",
        "bbits: flag as string": '{"applicable":"true","after":"' + C.bitstr(g, w) + '"}',
        "bbits: non-binary digit": '{"applicable":true,"after":"' + "2" + C.bitstr(g, w)[1:] + '"}',
        "ffill: keys reordered": '{"after":' + str(g) + ',"fill":" . ."}',
        "dint: out of range": str(2 ** w),
        "dint: boolean": "true",
        "dint: text": "the state is unchanged",
    }
    value_only = {"strace: after out of range", "bbits: wrong width", "dint: out of range"}   # right format, wrong value
    for name, text in neg.items():
        arm = name.split(":")[0]
        res["negatives"][name] = C.parse(arm, text, w)["after"] is None and C.format_ok(arm, text) is (name in value_only)
    res["format_ok_on_golds"] = {a: sum(C.format_ok(a, C.target_text(a, r["before"], r["event"], r["width"], count))
                                        for r in dev_rows) for a in arms}
    # D-int parser == env.parse_state(P, lenient=True) on 8-bit answers (incl. the lenient paths)
    battery = ["37", " 37 ", "37.", '{"after": 37}', '{"before": 5, "after": 37}', "0", "255", "256", "-1", "true",
               '{"state": 37}', 'x "after": 37 y', '"after": {"a": 1}', "[37]", "37 38", '{"after":37', "3.5", "",
               '{"after": 300}', '{"after": true}', "  12.  "]
    res["dint_parser_equals_env"] = all(C.parse_dint(t, 8) == env.parse_state(C.FAM, t, lenient=True) for t in battery)
    ok = all(v["correct"] == v["n"] and all(v.get(k, v["n"]) == v["n"] for k in ("cmp_ok", "calc_ok", "flag_ok"))
             for v in res["golds"].values()) and \
        all(h["match"] for kind in ("heuristics", "shortcuts") for hh in res[kind].values() for h in hh.values()) and \
        all(res["negatives"].values()) and res["dint_parser_equals_env"] and \
        all(v == len(dev_rows) for v in res["format_ok_on_golds"].values())
    res["ok"] = bool(ok)
    res["filler_checked"] = count is not None
    return res


# ---------------------------------------------------------------- scoring
def stratum_rows(dev_rows: list) -> dict:
    out = defaultdict(list)
    for i, r in enumerate(dev_rows):
        out[r["stratum"]].append(i)
    return out


def onestep_scores(arm: str, texts: list, dev_rows: list) -> list:
    sc = [C.score_row(arm, t, r) for t, r in zip(texts, dev_rows)]
    for s, r in zip(sc, dev_rows):
        sts = C.shortcut_states(r["before"], r["event"])
        s["agrees_shortcut"] = {k: s["after"] is not None and s["after"] == v for k, v in sts.items()}
    return sc


def cell(sc: list, idx: list, key: str = "correct") -> dict:
    k = sum(int(sc[i][key]) for i in idx)
    n = len(idx)
    return {"k": k, "n": n, "acc": round(k / n, 4) if n else None, "wilson95": C.wilson(k, n)}


def summarize_onestep(arm: str, sc: list, dev_rows: list) -> dict:
    by = stratum_rows(dev_rows)
    out = {"strata": {}, "unions": {}, "subtypes": {}}
    for s in C.DEV_STRATA:
        idx = by.get(s, [])
        e = cell(sc, idx)
        e["well_formed"] = cell(sc, idx, "well_formed")["acc"]
        e["copy"] = cell(sc, idx, "copy")["acc"]
        e["agrees"] = {h: round(sum(sc[i]["agrees"][h] for i in idx) / len(idx), 4)
                       for h in list(C.AUDITED) + list(C.DESCRIPTIVE_HEURISTICS)} if idx else {}
        e["agrees_shortcut"] = {h: round(sum(sc[i]["agrees_shortcut"][h] for i in idx) / len(idx), 4)
                                for h in C.SHORTCUT_POLICIES} if idx else {}
        for k in ("cmp_ok", "calc_ok", "flag_ok"):
            if idx and k in sc[idx[0]]:
                e[k] = cell(sc, idx, k)["acc"]
        out["strata"][s] = e
        if s in C.OOD_WIDTH or s == "D_inapp_addset":
            for sub in (C.OOD_SUBTYPES if s in C.OOD_WIDTH else C.ADDSET_SUBTYPES):
                ii = [i for i in idx if dev_rows[i]["subtype"] == sub]
                out["subtypes"][f"{s}/{sub}"] = cell(sc, ii)
    for name, ss in UNIONS.items():
        out["unions"][name] = cell(sc, [i for s in ss for i in by.get(s, [])])
    fc_idx = [i for s in C.FC_STRATA for i in by.get(s, [])]
    out["false_copy"] = cell(sc, fc_idx, "copy")
    ood_ch = [i for s in C.OOD_WIDTH for i in by.get(s, []) if dev_rows[i]["gold"] != dev_rows[i]["before"]]
    out["false_copy_out_of_range_descriptive"] = cell(sc, ood_ch, "copy")
    fa = {}
    for s in C.INAPP_STRATA:
        idx = by.get(s, [])
        fa[s] = round(1 - cell(sc, idx, "copy")["acc"], 4) if idx else None
    out["false_apply_by_stratum"] = fa
    w8 = [i for i, r in enumerate(dev_rows) if r["width"] == C.WIDTH_ID]
    out["format_ok_8bit"] = cell(sc, w8, "format_ok")
    out["well_formed_8bit_descriptive"] = cell(sc, w8, "well_formed")
    out["well_formed_all"] = cell(sc, list(range(len(dev_rows))), "well_formed")["acc"]
    out["ID"] = round(float(np.mean([out["strata"][s]["acc"] for s in C.ID_STRATA])), 4)
    out["IAW"] = out["strata"]["D_int_add_wrong"]["acc"]
    out["OOR"] = round((out["strata"]["D_ood10"]["acc"] + out["strata"]["D_ood12"]["acc"]) / 2, 4)
    out["FC"] = out["false_copy"]["acc"]
    out["FA"] = max(v for v in fa.values() if v is not None)
    out["WF"] = out["format_ok_8bit"]["acc"]
    return out


def decision_rows(chains: list) -> list:
    rows = []
    for c in chains:
        st = C.chain_gold_states(c)
        rows.append({"item_id": f"{c['cand_id']}-k{C.CHAIN_H}", "parent_id": c["parent_id"], "cand_id": c["cand_id"],
                     "family": C.FAM, "split": "test", "k": C.CHAIN_H, "s0": c["s0"], "events": c["events"],
                     "gold": env.dumps(st[-1]), "phi": data.phi_from_features(C.FAM, env.features(C.FAM, st, c["events"])).tolist()})
    return rows


def summarize_chains(chain_out: list, chains: list) -> dict:
    gm = {r["cand_id"]: r for r in chain_out}
    out = {}
    for k in (1, 2, 4):
        ok = [gm[c["cand_id"]]["states"][k] == env.dumps(C.chain_gold_states(c)[k]) for c in chains]
        out[f"k{k}_exact"] = {"k": int(sum(ok)), "n": len(ok), "acc": round(sum(ok) / len(ok), 4), "wilson95": C.wilson(int(sum(ok)), len(ok))}
    out["broken_by_k4"] = round(sum(gm[c["cand_id"]]["states"][C.CHAIN_H] is None for c in chains) / len(chains), 4)
    steps = Counter()
    for c in chains:
        sts = gm[c["cand_id"]]["states"]
        for t, e in enumerate(c["events"]):
            if sts[t] is None:
                break
            prev, nxt = json.loads(sts[t]), sts[t + 1]
            key = "applicable" if C.applicable(prev, e) else "inapplicable"
            steps[f"{key}_n"] += 1
            steps[f"{key}_ok"] += int(nxt is not None and json.loads(nxt) == C.gold(prev, e))
    out["step_accuracy_from_own_state"] = {k: round(steps[f"{k}_ok"] / steps[f"{k}_n"], 4) if steps[f"{k}_n"] else None
                                           for k in ("applicable", "inapplicable")}
    rows = decision_rows(chains)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        g = decisions.g_only(rows, C.FAM, C.CHAIN_H, {"m": gm}, modes=("m",))["m"]
    out["regret_G_chain"] = decide.bootstrap_mean(np.asarray(g["nregret"], np.float64), n_boot=BOOT)
    out["_per_parent_regret"] = np.asarray(g["nregret"], np.float64)
    out["_pids"] = g["pids"]
    return out


def floors(chains: list) -> dict:
    rows = decision_rows(chains)
    te = rows
    dim = len(te[0]["phi"])
    G = decide.goals(dim, decisions.N_TEST_GOALS, 1001)
    pids = sorted({r["parent_id"] for r in te})
    pos = {r["item_id"]: i for i, r in enumerate(te)}
    U_true = decisions._cube(pids, C.CHAIN_K, C.CHAIN_H, pos, np.stack([r["phi"] for r in te]) @ G.T)
    phis = defaultdict(list)
    for r in te:
        for name, st in fidelity.heuristic_states(C.FAM, r["s0"], r["events"]).items():
            phis[name].append(data.phi_from_state(C.FAM, r["s0"], r["events"], st))
    out = {}
    for name, v in phis.items():
        U = decisions._cube(pids, C.CHAIN_K, C.CHAIN_H, pos, np.stack(v) @ G.T)
        out[f"F_{name}"] = decide.bootstrap_mean(np.asarray(decide.regret_table(U_true, U, np.random.default_rng(17))["nregret"],
                                                            np.float64), n_boot=BOOT)
    k4 = {name: round(sum(env.dumps(fidelity.heuristic_states(C.FAM, r["s0"], r["events"])[name]) == r["gold"] for r in te) / len(te), 4)
          for name in ("copy_s0", "ignore_pre")}
    return {"regret": out, "k4_exact": k4}


# ---------------------------------------------------------------- the pre-stated rule
def _r(x: float) -> float:
    """Comparisons at 6 decimals, so that exact boundary values (e.g. FC = 0.25) are not decided by float error."""
    return round(x, 6)


def r_of(mm: dict) -> float:
    return round((mm["K4"] + mm["IAW"] + mm["OOR"]) / 3, 4)


def filler_control(r_strace: float, mf: dict, anchors_ok: bool = True) -> dict:
    """R5 on the filler arm's ON metrics: flag, void or separated."""
    rf = r_of(mf)
    if _r(mf["WF"] - WF_MIN) < 0 or not anchors_ok:
        return {"status": "void", "flag": FLAG_VOID, "R_ffill": rf, "R_strace": r_strace,
                "reason": f"WF(F-filler) {mf['WF']:.3f} < {WF_MIN}" if mf["WF"] < WF_MIN else "a filler turn lacks a passed anchor"}
    if _r(rf - (r_strace - FILLER_FLAG_MARGIN)) >= 0:
        return {"status": "not_separable", "flag": FLAG_FILLER, "R_ffill": rf, "R_strace": r_strace}
    return {"status": "separated", "flag": None, "R_ffill": rf, "R_strace": r_strace}


def apply_rule(m: dict, anchors_ok: bool = True, filler_anchors_ok: bool = True) -> dict:
    """m: {arm: {"IAW", "OOR", "K4", "FC", "FA", "WF", "ID"}} for the arms that ran (ON). Returns the verdict (R0-R5)."""
    for a in C.SCREEN_ARMS:
        need(a in m, f"screen arm {a} missing: no verdict")
    R = {a: r_of(m[a]) for a in m}
    out = {"R": R, "metrics": m, "anchors_ok": anchors_ok, "flags": [], "notes": [], "winner": None, "eligible": []}
    bad_wf = [a for a in C.SCREEN_ARMS if _r(m[a]["WF"] - WF_MIN) < 0]
    if bad_wf or not anchors_ok:
        why = [f"WF({C.ARM_LABEL[a]}) {m[a]['WF']:.3f} < {WF_MIN}" for a in bad_wf] + ([] if anchors_ok else ["a turn lacks a passed anchor"])
        out.update(verdict="NO_VERDICT_INSTRUMENT", instrument_problems=why,
                   wording=("No verdict: instrument or format failure (" + "; ".join(why) + "). This is not a result; no "
                            "programme decision follows; investigate and log it in DECISIONS.md."))
        return out
    if all(_r(m[a]["IAW"] - IAW_STOP) < 0 for a in C.SCREEN_ARMS):
        out.update(verdict="STOP_P_TRAINING",
                   wording=("Every arm scores below 0.20 on int-add-wrong (decision point 2): stop prerequisite training "
                            "and move P to a code-side bit interface, reported as an interface, not as model skill "
                            f"({DOSE})."))
        return out
    base_id = m["dint"]["ID"]
    excluded = {}
    for a in C.SCREEN_ARMS:
        why = []
        if _r(m[a]["IAW"] - IAW_STOP) < 0:
            why.append(f"IAW {m[a]['IAW']:.3f} < {IAW_STOP}")
        if _r(m[a]["FC"] - FC_MAX) > 0:
            why.append(f"false copy {m[a]['FC']:.3f} > {FC_MAX}")
        if _r(m[a]["FA"] - FA_MAX) > 0:
            why.append(f"false apply {m[a]['FA']:.3f} > {FA_MAX}")
        if _r(m[a]["ID"] - (base_id - ID_MARGIN)) < 0:
            why.append(f"ID {m[a]['ID']:.3f} < ID(D-int) - {ID_MARGIN}")
        if why:
            excluded[a] = "; ".join(why)
    eligible = [a for a in C.SCREEN_ARMS if a not in excluded]
    out["eligible"], out["excluded"] = eligible, excluded
    if not eligible:
        out.update(verdict="NO_ELIGIBLE_ARM",
                   wording=("No arm is eligible (int-add-wrong < 0.20, false copy > 0.25, false apply > 0.25 or an "
                            "in-distribution collapse in every arm); no arm is named for seeds 1-2."))
        return out
    order = sorted(eligible, key=lambda a: (-R[a], TIE_ORDER.index(a)))
    best, tie = order[0], None
    if len(order) > 1 and _r(R[order[0]] - R[order[1]] - TIE_R) < 0:          # the top two differ by < 0.02 in R
        x, y = order[0], order[1]
        if _r(abs(m[x]["IAW"] - m[y]["IAW"]) - TIE_IAW) >= 0:
            best, tie = (x if m[x]["IAW"] > m[y]["IAW"] else y), "IAW"
        elif _r(abs(m[x]["OOR"] - m[y]["OOR"]) - TIE_IAW) >= 0:
            best, tie = (x if m[x]["OOR"] > m[y]["OOR"] else y), "OOR"
        else:
            best, tie = min((x, y), key=TIE_ORDER.index), "fixed order D-int, B-bits, S-trace"
    out["tie_break"] = tie
    others = [a for a in eligible if a != best]
    margin = round(R[best] - max(R[a] for a in others), 4) if others else None
    out["margin_R_over_best_other_eligible"] = margin
    out["close_call"] = margin is not None and _r(margin - CLOSE_CALL) < 0
    wording = f"WINNER={C.ARM_LABEL[best]}: seeds 1 and 2 run {C.ARM_LABEL[best]} (same rows, T2 recipe)."
    if best == "dint":
        wording += " The representation does not beat the rebalanced-data control."
    if out["close_call"]:
        wording += (f" Close call: the winner's R exceeds another eligible arm's by {margin:.3f} < {CLOSE_CALL} (about two "
                    "standard errors); a screen choice, not a finding.")
    if best == "strace":
        if "ffill" in m:
            fc = filler_control(R["strace"], m["ffill"], filler_anchors_ok)
            out["filler_control"] = fc
            if fc["flag"]:
                out["flags"].append(fc["flag"])
        else:
            out["filler_control"] = {"status": "pending", "flag": None}
            out["notes"].append("decode-compute control pending")
            wording += (f" Decode-compute control pending (R5: run tag {C.FILLER_TAG}, then t2bp_analysis.py "
                        f"--filler-run runs/t2bp/{C.FILLER_TAG}).")
    if out["flags"]:
        wording += " Flag: " + ", ".join(out["flags"]) + "."
    out.update(verdict=f"WINNER={best}", winner=best, ranking=order, wording=wording)
    return out


def apply_filler(screen_rule: dict, mf: dict, anchors_ok: bool = True) -> dict:
    """R5 addendum: the screen named S-trace and the filler control ran as its own run."""
    need(screen_rule.get("verdict") == "WINNER=strace", f"R5 applies only after an S-trace win (screen verdict "
         f"{screen_rule.get('verdict')})")
    need("ffill" not in (screen_rule.get("metrics") or {}), "the filler arm was part of the screen run: R5 was applied there")
    fc = filler_control(screen_rule["R"]["strace"], mf, anchors_ok)
    w = {"separated": f"R5: R(F-filler) {fc['R_ffill']:.3f} < R(S-trace) {fc['R_strace']:.3f} - {FILLER_FLAG_MARGIN}: "
                      "the S-trace gain is separated from extra decode compute.",
         "not_separable": f"R5: R(F-filler) {fc['R_ffill']:.3f} >= R(S-trace) {fc['R_strace']:.3f} - {FILLER_FLAG_MARGIN}: "
                          f"flag {FLAG_FILLER} (the winner is unchanged; quote the flag with every statement).",
         "void": f"R5: the filler control is void ({fc.get('reason')}): flag {FLAG_VOID}."}[fc["status"]]
    return {"screen_verdict": screen_rule["verdict"], "filler_control": fc, "flags": [fc["flag"]] if fc["flag"] else [],
            "wording": w}


# ---------------------------------------------------------------- paired contrasts (descriptive)
def paired_rows(a_sc: list, b_sc: list, idx: list, seed: int = 17) -> dict:
    d = np.array([int(a_sc[i]["correct"]) - int(b_sc[i]["correct"]) for i in idx], np.float64)
    if not len(d):
        return {}
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (BOOT, len(d)))].mean(1)
    return {"n": int(len(d)), "diff": round(float(d.mean()), 4), "ci95": [round(float(np.quantile(bs, 0.025)), 4),
                                                                       round(float(np.quantile(bs, 0.975)), 4)]}


# ---------------------------------------------------------------- inputs
def eval_files(run: Path, a: str, side: str, data_variant: str) -> dict:
    fs = {"E1": run / a / f"onestep_{side}.jsonl", "E2": run / a / f"chains_{side}.jsonl"}
    if data_variant == "sd":
        fs["E3"] = run / a / f"onestep_heldout_{side}.jsonl"
    return fs


def anchors_ok(prov: dict) -> bool:
    """Every turn that ran an evaluation stage carries a passed anchor record."""
    for t in prov.get("turns") or []:
        if any(s.split(":")[-1] in EVAL_STAGES for s in t.get("stages_run") or []):
            if not (t.get("anchor") or {}).get("pass") is True:
                return False
    return True


def validate_run(run: Path, dev_rows: list, chains: list, arms: list) -> dict:
    """Loads and validates every output file; returns {arm: {side: {"onestep", "chains"[, "heldout"]}}}."""
    prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    need((run / "DONE.json").is_file(), "DONE.json missing: the run is incomplete")
    dv = prov.get("data_variant") or ""
    out = {}
    for a in arms:
        rec = (prov.get("stages") or {}).get(a) or {}
        out[a] = {}
        for side in ("off", "on"):
            fs = eval_files(run, a, side, dv)
            for st, f in fs.items():
                rel = str(f.relative_to(run)).replace("\\", "/")
                need(f.is_file() and C.sha256(f) == ((rec.get(st) or {}).get("files") or {}).get(rel), f"{rel}: missing or hash differs")
            o1 = C.load_jsonl(fs["E1"])
            need([x["id"] for x in o1] == [r["id"] for r in dev_rows], f"{a}/{side}: one-step ids differ from the dev bank")
            o2 = C.load_jsonl(fs["E2"])
            need([x["cand_id"] for x in o2] == [c["cand_id"] for c in chains], f"{a}/{side}: chain ids differ")
            for x, c in zip(o2, chains):
                re_states = [env.dumps(c["s0"])]
                for t, txt in enumerate(x["texts"]):
                    v = C.parse_after(a, txt, c["width"])
                    re_states.append(None if v is None else env.dumps(v))
                    if v is None:
                        break
                re_states += [None] * (C.CHAIN_H + 1 - len(re_states))
                need(re_states == x["states"], f"{a}/{side}: chain {c['cand_id']} states != parse(texts)")
            out[a][side] = {"onestep": [x["text"] for x in o1], "chains": o2}
            if "E3" in fs:
                o3 = C.load_jsonl(fs["E3"])
                need([x["id"] for x in o3] == [r["id"] for r in dev_rows], f"{a}/{side}: held-out-surface ids differ")
                out[a][side]["heldout"] = [x["text"] for x in o3]
    return out


def analyse(outputs: dict, dev_rows: list, chains: list, selftest: dict | None = None, anchors: bool = True) -> dict:
    t0 = time.time()
    res = {"selftest_ok": None if selftest is None else selftest["ok"], "arms": {}, "floors": floors(chains)}
    sc_all = {}
    metrics = {}
    for a, sides in outputs.items():
        res["arms"][a] = {}
        for side, o in sides.items():
            sc = onestep_scores(a, o["onestep"], dev_rows)
            sc_all[(a, side)] = sc
            one = summarize_onestep(a, sc, dev_rows)
            ch = summarize_chains(o["chains"], chains)
            res["arms"][a][side] = {"onestep": one, "chains": {k: v for k, v in ch.items() if not k.startswith("_")}}
            sc_all[(a, side, "pp")] = ch["_per_parent_regret"]
            if "heldout" in o:
                h = summarize_onestep(a, onestep_scores(a, o["heldout"], dev_rows), dev_rows)
                res["arms"][a][side]["heldout_surface_descriptive"] = {
                    "surface": C.HELDOUT_SURFACE, "strata_acc": {s: h["strata"][s]["acc"] for s in C.DEV_STRATA},
                    **{k: h[k] for k in ("IAW", "OOR", "FC", "FA", "WF", "ID")}}
            if side == "on":
                metrics[a] = {"IAW": one["IAW"], "OOR": one["OOR"], "K4": ch["k4_exact"]["acc"], "FC": one["FC"],
                              "FA": one["FA"], "WF": one["WF"], "ID": one["ID"]}
    if all(a in metrics for a in C.SCREEN_ARMS):
        res["rule"] = apply_rule(metrics, anchors_ok=anchors, filler_anchors_ok=anchors)
    else:                                     # a seed-1/2 run of the named arm, or the filler control: descriptive only
        res["rule"] = {"verdict": "NO_VERDICT", "winner": None, "flags": [], "notes": [], "metrics": metrics,
                       "anchors_ok": anchors, "R": {a: r_of(v) for a, v in metrics.items()},
                       "wording": f"No screen verdict: the run holds arms {sorted(metrics)}, not all of "
                                  f"{list(C.SCREEN_ARMS)} (the rule needs all three); descriptive only."}
    res["verdict"] = res["rule"]["verdict"]
    con = {}
    idx = stratum_rows(dev_rows)
    for a in outputs:
        if ("on" in outputs[a]) and ("off" in outputs[a]):
            con[f"{a}: ON - OFF"] = {s: paired_rows(sc_all[(a, "on")], sc_all[(a, "off")], idx[s]) for s in C.DEV_STRATA}
        if a != "dint" and "dint" in outputs:
            con[f"{a} ON - D-int ON"] = {s: paired_rows(sc_all[(a, "on")], sc_all[("dint", "on")], idx[s]) for s in C.DEV_STRATA}
            d = sc_all[(a, "on", "pp")] - sc_all[("dint", "on", "pp")]
            con[f"{a} ON - D-int ON"]["regret_G_chain"] = decide.paired_bootstrap(sc_all[(a, "on", "pp")], sc_all[("dint", "on", "pp")], n_boot=BOOT) if len(d) else {}
    res["contrasts_descriptive"] = con
    res["seconds"] = round(time.time() - t0, 1)
    res["torch_imported"] = "torch" in sys.modules
    return res


def render_md(res: dict) -> str:
    r = res["rule"]
    L = [f"# T2b prerequisite representation screen: {res.get('run', '(in memory)')}", "",
         f"**Verdict (pre-stated rule revision 2, scripts/t2bp_analysis.py):** {r['verdict']}"
         + (f" (flags: {', '.join(r['flags'])})" if r.get("flags") else ""), "", r["wording"], "",
         f"Frozen inputs unchanged: {res.get('frozen_inputs_unchanged', '(not checked: in memory)')}. "
         f"Anchors passed in every evaluating turn: {r.get('anchors_ok')}.", "",
         "| Arm | IAW | OOR | K4 | FC | FA | WF | ID | R |", "|---|---|---|---|---|---|---|---|---|"]
    for a, mm in r["metrics"].items():
        L.append(f"| {C.ARM_LABEL[a]} | {mm['IAW']:.3f} | {mm['OOR']:.3f} | {mm['K4']:.3f} | {mm['FC']:.3f} | "
                 f"{mm['FA']:.3f} | {mm['WF']:.3f} | {mm['ID']:.3f} | {r['R'][a]:.3f} |")
    if r.get("excluded"):
        L += ["", "Excluded: " + "; ".join(f"{C.ARM_LABEL[a]}: {w}" for a, w in r["excluded"].items())]
    L += ["", "Per-stratum accuracy (OFF -> ON):", "", "| Stratum | " + " | ".join(C.ARM_LABEL[a] for a in res["arms"]) + " |",
          "|---|" + "---|" * len(res["arms"])]
    for s in C.DEV_STRATA:
        L.append(f"| {s} | " + " | ".join(f"{res['arms'][a]['off']['onestep']['strata'][s]['acc']:.2f} -> "
                                          f"{res['arms'][a]['on']['onestep']['strata'][s]['acc']:.2f}" for a in res["arms"]) + " |")
    L += ["", f"Training dose: {DOSE}.",
          "", "Descriptive only below this line; nothing here changes the verdict. This is a dev-bank screen, not a confirmation."]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- dry run on synthetic outcomes (no run output read)
def policy_text(policy: str, arm: str, before: int, ev: dict, w: int, rng: random.Random) -> str:
    g = C.gold(before, ev)
    h = C.heuristics(before, ev)
    if policy == "perfect":
        st = g
    elif policy == "copy":
        st = before
    elif policy == "intadd":
        st = h["int_add"]
    elif policy == "ignore_pre":
        st = h["apply_regardless"]
    elif policy == "perfect_id_ignore_ood":       # the rule in range; out of range it ignores preconditions
        st = g if w == C.WIDTH_ID else h["apply_regardless"]
    elif policy == "iaw_only":                    # right on int-add-wrong-type rows only, copies elsewhere
        st = g if C.rtype(before, ev) == "A4" else before
    elif policy == "noisy80":                     # the rule 80% of the time, else a copy
        st = g if rng.random() < 0.8 else before
    elif policy == "t2like":                      # T2's copy shortcut: copies 60% of state-changing answers
        st = before if (g != before and rng.random() < 0.6) else g
    elif policy == "malformed":
        return "I cannot tell."
    elif policy == "malformed15":                 # the rule, but 15% of answers are malformed
        if rng.random() < 0.15:
            return "I cannot tell."
        st = g
    elif policy in C.SHORTCUT_POLICIES:
        st = C.SHORTCUT_POLICIES[policy](before, ev)
    else:
        raise ValueError(policy)
    if not 0 <= st < 2 ** w:                      # out of range: the arm's format, a wrong value (never a gold)
        if arm == "bbits":
            return '{"applicable":true,"after":"' + format(st, "b") + '"}'
        return C.render_prediction(arm, st, before, ev, w)
    if arm == "strace" and st == g:
        return C.trace_text(before, ev)
    return C.render_prediction(arm, st, before, ev, w)


def synth_outputs(policies: dict, dev_rows: list, chains: list, seed: int = 5) -> dict:
    """policies: {arm: {side: policy}} -> outputs in the analysis' input format (chains rolled out under the policy)."""
    out = {}
    for a, sides in policies.items():
        out[a] = {}
        for side, pol in sides.items():
            rng = random.Random(seed)
            one = [policy_text(pol, a, r["before"], r["event"], r["width"], rng) for r in dev_rows]
            rows = []
            for c in chains:
                st, sts, txs = c["s0"], [env.dumps(c["s0"])], []
                for e in c["events"]:
                    if st is None:
                        sts.append(None)
                        continue
                    t = policy_text(pol, a, st, e, c["width"], rng)
                    txs.append(t)
                    st = C.parse_after(a, t, c["width"])
                    sts.append(None if st is None else env.dumps(st))
                rows.append({"cand_id": c["cand_id"], "states": sts, "texts": txs})
            out[a][side] = {"onestep": one, "chains": rows}
    return out


def verdict_string(r: dict) -> str:
    return r["verdict"] + "".join("+" + f for f in r.get("flags") or [])


W_STRACE_FLAG = "WINNER=strace+" + FLAG_FILLER
SCENARIOS = {   # end to end: synthetic answers -> parsers -> metrics -> rule
    "stop_all_below_0.20": ({"dint": "copy", "strace": "intadd", "bbits": "copy"}, "STOP_P_TRAINING"),
    "stop_titan_reading_everywhere (R11 pa_int_add / pa_or_add)": (
        {"dint": "pa_int_add", "strace": "pa_int_add", "bbits": "pa_or_add"}, "STOP_P_TRAINING"),
    "trace_wins (filler control pending)": ({"dint": "copy", "strace": "perfect", "bbits": "noisy80"}, "WINNER=strace"),
    "bbits_wins_trace_excluded_by_copy": ({"dint": "copy", "strace": "iaw_only", "bbits": "perfect_id_ignore_ood"}, "WINNER=bbits"),
    "no_eligible_arm": ({"dint": "iaw_only", "strace": "iaw_only", "bbits": "iaw_only"}, "NO_ELIGIBLE_ARM"),
    "tie_goes_to_dint": ({"dint": "perfect", "strace": "perfect", "bbits": "perfect"}, "WINNER=dint"),
    "ignore_pre_arm_excluded_by_false_apply": ({"dint": "perfect_id_ignore_ood", "strace": "ignore_pre", "bbits": "copy"}, "WINNER=dint"),
    "review case 1: pre_on-only arm (ignores pre_off) loses to an 80% rule arm": (
        {"dint": "t2like", "strace": "pre_on_only", "bbits": "noisy80"}, "WINNER=bbits"),
    "review case 1b: pre_off-only arm loses to an 80% rule arm": (
        {"dint": "t2like", "strace": "pre_off_only", "bbits": "noisy80"}, "WINNER=bbits"),
    "review case 2: ignore-preconditions arm vs a T2-like control": (
        {"dint": "t2like", "strace": "ignore_pre", "bbits": "copy"}, "NO_ELIGIBLE_ARM"),
    "review case 3: precondition-count cue arm": ({"dint": "t2like", "strace": "precount_cue", "bbits": "copy"}, "NO_ELIGIBLE_ARM"),
    "add-bit-set -> apply cue arm (caught by D_inapp_addset)": (
        {"dint": "t2like", "strace": "addset_apply", "bbits": "noisy80"}, "WINNER=bbits"),
    "delete-only arm (IAW 1.0, fails D_app_add) vs a perfect D-int": (
        {"dint": "perfect", "strace": "pa_delete_only", "bbits": "copy"}, "WINNER=dint"),
    "malformed_everywhere": ({"dint": "malformed", "strace": "malformed", "bbits": "malformed"}, "NO_VERDICT_INSTRUMENT"),
    "one arm 15% malformed": ({"dint": "perfect", "strace": "malformed15", "bbits": "perfect"}, "NO_VERDICT_INSTRUMENT"),
    "filler in the s0 run flags": ({"dint": "copy", "strace": "perfect", "bbits": "copy", "ffill": "perfect"}, W_STRACE_FLAG),
}


def _m(iaw, oor, k4, fc, idd, fa=0.0, wf=1.0):
    return {"IAW": iaw, "OOR": oor, "K4": k4, "FC": fc, "ID": idd, "FA": fa, "WF": wf}


RULE_UNITS = {   # the rule's branches and boundaries on metric tables
    "ID non-inferiority excludes the best-R arm": (
        {"dint": _m(.5, .5, .5, .1, .9), "strace": _m(1, 1, .9, .1, .79), "bbits": _m(.3, .3, .3, .1, .9)}, "WINNER=dint"),
    "ID exactly ID(D-int) - 0.10 is eligible": (
        {"dint": _m(.5, .5, .5, .1, .9), "strace": _m(1, 1, .9, .1, .8), "bbits": _m(.3, .3, .3, .1, .9)}, "WINNER=strace"),
    "FC exactly 0.25 is eligible": (
        {"dint": _m(.5, .5, .5, .9, .9), "strace": _m(.6, .6, .6, .25, .9), "bbits": _m(.3, .3, .3, .9, .9)}, "WINNER=strace"),
    "FC 0.26 is excluded": (
        {"dint": _m(.5, .5, .5, .9, .9), "strace": _m(.6, .6, .6, .26, .9), "bbits": _m(.3, .3, .3, .9, .9)}, "NO_ELIGIBLE_ARM"),
    "FA exactly 0.25 is eligible": (
        {"dint": _m(.5, .5, .5, .1, .9, fa=.9), "strace": _m(.6, .6, .6, .1, .9, fa=.25), "bbits": _m(.3, .3, .3, .1, .9, fa=.9)},
        "WINNER=strace"),
    "FA 0.26 excludes the best-R arm (a false-apply shortcut cannot win)": (
        {"dint": _m(.5, .5, .5, .1, .9), "strace": _m(1, 1, 1, 0, .9, fa=.26), "bbits": _m(.3, .3, .3, .1, .9)}, "WINNER=dint"),
    "IAW exactly 0.20 everywhere is not a stop and is eligible": (
        {"dint": _m(.2, .5, .5, .1, .9), "strace": _m(.2, .4, .4, .1, .9), "bbits": _m(.2, .3, .3, .1, .9)}, "WINNER=dint"),
    "one arm at 0.20 prevents the stop and is the only eligible arm": (
        {"dint": _m(.19, .5, .5, .1, .9), "strace": _m(.2, .4, .4, .1, .9), "bbits": _m(.1, .3, .3, .1, .9)}, "WINNER=strace"),
    "R1/R2 gap: the only arm with IAW >= 0.20 is ineligible -> no winner below 0.20": (
        {"dint": _m(.25, .5, .3, .6, .8), "strace": _m(.05, .5, .4, .1, .75), "bbits": _m(.10, .4, .3, .5, .7)}, "NO_ELIGIBLE_ARM"),
    "WF 0.89 in one arm: NO_VERDICT_INSTRUMENT (checked before the stop)": (
        {"dint": _m(0, .5, .5, .1, .9), "strace": _m(0, .4, .4, .1, .9, wf=.89), "bbits": _m(0, .3, .3, .1, .9)}, "NO_VERDICT_INSTRUMENT"),
    "WF exactly 0.90 is valid": (
        {"dint": _m(0, .5, .5, .1, .9, wf=.9), "strace": _m(0, .4, .4, .1, .9, wf=.9), "bbits": _m(0, .3, .3, .1, .9, wf=.9)},
        "STOP_P_TRAINING"),
    "R gap >= 0.02 decides regardless of IAW": (
        {"dint": _m(.30, .9, .9, .1, .9), "strace": _m(.9, .3, .64, .1, .9), "bbits": _m(.1, .1, .1, .1, .9)}, "WINNER=dint"),
    "R tie (< 0.02): larger IAW wins": (
        {"dint": _m(.30, .9, .9, .1, .9), "strace": _m(.9, .3, .87, .1, .9), "bbits": _m(.1, .1, .1, .1, .9)}, "WINNER=strace"),
    "R and IAW tie: larger OOR wins": (
        {"dint": _m(.5, .60, .80, .1, .9), "strace": _m(.5, .65, .74, .1, .9), "bbits": _m(.1, .1, .1, .1, .9)}, "WINNER=strace"),
    "all tie: fixed order D-int, B-bits, S-trace": (
        {"dint": _m(.1, .1, .1, .1, .9), "strace": _m(.6, .6, .6, .1, .9), "bbits": _m(.6, .6, .6, .1, .9)}, "WINNER=bbits"),
    "filler at R(S) - 0.05 flags": (
        {"dint": _m(.3, .3, .3, .1, .9), "strace": _m(.9, .9, .9, .1, .9), "bbits": _m(.3, .3, .3, .1, .9),
         "ffill": _m(.85, .85, .85, .1, .9)}, W_STRACE_FLAG),
    "filler below R(S) - 0.05 does not flag": (
        {"dint": _m(.3, .3, .3, .1, .9), "strace": _m(.9, .9, .9, .1, .9), "bbits": _m(.3, .3, .3, .1, .9),
         "ffill": _m(.84, .84, .84, .1, .9)}, "WINNER=strace"),
    "filler is never eligible (best R, yet only the R5 flag)": (
        {"dint": _m(.3, .3, .3, .1, .9), "strace": _m(.4, .4, .4, .1, .9), "bbits": _m(.3, .3, .3, .1, .9),
         "ffill": _m(1, 1, 1, 0, 1)}, W_STRACE_FLAG),
    "filler never eligible: D-int wins over a perfect filler": (
        {"dint": _m(.5, .5, .5, .1, .9), "strace": _m(.4, .4, .4, .1, .9), "bbits": _m(.3, .3, .3, .1, .9),
         "ffill": _m(1, 1, 1, 0, 1)}, "WINNER=dint"),
    "a malformed filler voids the control (no flag of separation)": (
        {"dint": _m(.3, .3, .3, .1, .9), "strace": _m(.9, .9, .9, .1, .9), "bbits": _m(.3, .3, .3, .1, .9),
         "ffill": _m(0, 0, 0, 0, 0, wf=.5)}, "WINNER=strace+" + FLAG_VOID),
}
CLOSE_UNITS = {   # (metrics, expected close_call)
    "margin 0.049 is a close call": ({"dint": _m(.5, .5, .5, .1, .9), "strace": _m(.647, .647, .647, .1, .9),
                                      "bbits": _m(.598, .598, .598, .1, .9)}, True),
    "margin 0.05 is not a close call": ({"dint": _m(.5, .5, .5, .1, .9), "strace": _m(.65, .65, .65, .1, .9),
                                         "bbits": _m(.6, .6, .6, .1, .9)}, False),
    "an ineligible runner-up does not make a close call": ({"dint": _m(.5, .5, .5, .1, .9), "strace": _m(.8, .8, .8, .1, .9),
                                                            "bbits": _m(.79, .79, .79, .3, .9)}, False),
}
FILLER_UNITS = {   # R5 addendum: (R(S-trace) of the screen, filler metrics, anchors ok, expected status)
    "addendum: filler R at R(S) - 0.05 flags": (0.9, _m(.85, .85, .85, .1, .9), True, "not_separable"),
    "addendum: filler R below R(S) - 0.05 separates": (0.9, _m(.84, .84, .84, .1, .9), True, "separated"),
    "addendum: filler WF 0.89 voids the control": (0.9, _m(.1, .1, .1, .1, .9, wf=.89), True, "void"),
    "addendum: a filler turn without a passed anchor voids the control": (0.9, _m(.1, .1, .1, .1, .9), False, "void"),
}


def dry_run(dev_rows: list, chains: list) -> dict:
    out = {"scenarios": {}, "ok": True}
    for name, (pols, expect) in SCENARIOS.items():
        policies = {a: {"off": "copy", "on": p} for a, p in pols.items()}
        res = analyse(synth_outputs(policies, dev_rows, chains), dev_rows, chains)
        v = verdict_string(res["rule"])
        ok = v == expect
        out["scenarios"][name] = {"expected": expect, "got": v, "ok": ok, "R": res["rule"]["R"], "policies_on": pols,
                                  "metrics": res["rule"]["metrics"], "excluded": res["rule"].get("excluded"),
                                  "wording": res["rule"]["wording"]}
        out["ok"] &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: expected {expect}, got {v}", flush=True)
    out["rule_units"] = {}
    for name, (m, expect) in RULE_UNITS.items():
        r = apply_rule(m)
        v = verdict_string(r)
        out["rule_units"][name] = {"expected": expect, "got": v, "ok": v == expect, "R": r["R"], "tie_break": r.get("tie_break")}
        out["ok"] &= v == expect
        print(f"  [{'PASS' if v == expect else 'FAIL'}] rule unit: {name}: expected {expect}, got {v}", flush=True)
    r = apply_rule(RULE_UNITS["IAW exactly 0.20 everywhere is not a stop and is eligible"][0], anchors_ok=False)
    ok = r["verdict"] == "NO_VERDICT_INSTRUMENT"
    out["rule_units"]["a turn without a passed anchor: NO_VERDICT_INSTRUMENT"] = {"expected": "NO_VERDICT_INSTRUMENT",
                                                                                  "got": r["verdict"], "ok": ok}
    out["ok"] &= ok
    print(f"  [{'PASS' if ok else 'FAIL'}] rule unit: a turn without a passed anchor: got {r['verdict']}", flush=True)
    out["close_call_units"] = {}
    for name, (m, expect) in CLOSE_UNITS.items():
        r = apply_rule(m)
        ok = r.get("close_call") is expect and (("Close call" in r["wording"]) is expect)
        out["close_call_units"][name] = {"expected": expect, "got": r.get("close_call"), "margin": r.get("margin_R_over_best_other_eligible"),
                                         "verdict": r["verdict"], "ok": ok}
        out["ok"] &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] close call: {name}: got {r.get('close_call')} (margin {r.get('margin_R_over_best_other_eligible')})", flush=True)
    out["filler_units"] = {}
    for name, (rs, mf, anc, expect) in FILLER_UNITS.items():
        add = apply_filler({"verdict": "WINNER=strace", "R": {"strace": rs}, "metrics": {}}, mf, anc)
        ok = add["filler_control"]["status"] == expect
        out["filler_units"][name] = {"expected": expect, "got": add["filler_control"]["status"], "ok": ok, "flags": add["flags"]}
        out["ok"] &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {add['filler_control']['status']}", flush=True)
    refusals = {}
    for name, fn in {"an incomplete screen run gives no verdict": lambda: apply_rule({"dint": _m(1, 1, 1, 0, 1)}),
                     "the R5 addendum refuses after a non-S-trace verdict": lambda: apply_filler(
                         {"verdict": "WINNER=dint", "R": {"strace": .5}, "metrics": {}}, _m(1, 1, 1, 0, 1)),
                     "the R5 addendum refuses when the filler was in the screen run": lambda: apply_filler(
                         {"verdict": "WINNER=strace", "R": {"strace": .5}, "metrics": {"ffill": {}}}, _m(1, 1, 1, 0, 1))}.items():
        try:
            fn()
            refusals[name] = False
        except Refuse:
            refusals[name] = True
        print(f"  [{'PASS' if refusals[name] else 'FAIL'}] {name}", flush=True)
    out["refusals"] = refusals
    out["ok"] &= all(refusals.values())
    out["counts"] = {"scenarios": len(SCENARIOS), "rule_units": len(out["rule_units"]), "close_call_units": len(CLOSE_UNITS),
                     "filler_units": len(FILLER_UNITS), "refusals": len(refusals),
                     "passed": sum(v["ok"] for k in ("scenarios", "rule_units", "close_call_units", "filler_units")
                                   for v in out[k].values()) + sum(refusals.values())}
    return out


# ---------------------------------------------------------------- the frozen rule (SCREEN_RULE.md / .json)
def rule_text() -> str:
    doc = __doc__
    return doc[doc.index(RULE_HEAD):].rstrip() + "\n"


def rule_md() -> str:
    return ("# Pre-stated screen rule (revision 2): T2b prerequisite representation screen (seed 0)\n\n"
            "Copied verbatim from the docstring of `scripts/t2bp_analysis.py`, which implements it (`apply_rule`, "
            "`apply_filler`). Revision 2 was written on 2026-10-09 after review, before any `runs/t2bp/` output existed. "
            "Revision 1 is kept as `SCREEN_RULE_rev1.md` / `SCREEN_RULE_rev1.json`. Hashes of every frozen input: "
            "`SCREEN_RULE.json`.\n\n```\n" + rule_text() + "```\n")


def _sha_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def prefreeze_checks(docs: Path) -> dict:
    """The dry run, the tiny test and the audit must be recorded against the current code and data."""
    cur = {r: C.sha256(ROOT / r) for r in ("scripts/t2bp_analysis.py", "scripts/t2bp_common.py", "scripts/t2bp_run.py",
                                           "scripts/t2bp_make_data.py")}
    dr = C.read_rule_record(docs / "DRYRUN.json") or {}
    tt = C.read_rule_record(docs / "TINY_TEST.json") or {}
    au = C.read_rule_record(docs / "AUDIT.json") or {}
    st = C.read_rule_record(docs / "SCORER_SELFTEST.json") or {}
    chk = {
        "dry run ok and recorded against the current analysis/common code": bool(dr.get("ok")) and all(
            (dr.get("code_sha256") or {}).get(r) == cur[r] for r in ("scripts/t2bp_analysis.py", "scripts/t2bp_common.py")),
        "scorer self-test ok on the current dev bank": bool(st.get("ok")) and st.get("dev_sha256") == C.sha256(C.DEV),
        "tiny test ok and recorded against the current code": bool(tt.get("ok")) and all(
            (tt.get("code_sha256") or {}).get(r) == cur[r] for r in cur),
        "audit ok on the current training files": bool(au.get("ok")) and all(
            (au.get("files") or {}).get(C.TRAIN_FILES[a].name) == C.sha256(C.TRAIN_FILES[a]) for a in C.ARMS),
        "no runs/t2bp output exists": not C.run_outputs(),
    }
    return chk


def freeze_rule(docs: Path | None = None) -> dict:
    docs = Path(docs or C.DOCS)
    outs = C.run_outputs()
    need(not outs, f"runs/t2bp output exists ({outs[:3]}): the rule can no longer be (re)frozen")
    chk = prefreeze_checks(docs)
    need(all(chk.values()), f"pre-freeze checks failed: {[k for k, v in chk.items() if not v]}")
    mdp, jp = docs / C.SCREEN_RULE_MD, docs / C.SCREEN_RULE_JSON
    prev = C.read_rule_record(jp)
    supersedes = None
    if prev is not None:
        rev = int(prev.get("revision", 1))
        keep_md, keep_js = docs / f"SCREEN_RULE_rev{rev}.md", docs / f"SCREEN_RULE_rev{rev}.json"
        need(not keep_md.exists() and not keep_js.exists(), f"{keep_md.name} exists: resolve by hand")
        supersedes = {"revision": rev, "record_sha256": C.sha256(jp), "rule_md_sha256": prev.get("rule_md_sha256"),
                      "rule_text_sha256": prev.get("rule_text_sha256"), "written_local": prev.get("written_local"),
                      "kept_as": [keep_md.name, keep_js.name]}
        os.replace(mdp, keep_md)
        os.replace(jp, keep_js)
    md = rule_md()
    tmp_md, tmp_js = mdp.with_name(mdp.name + ".partial"), jp.with_name(jp.name + ".partial")
    tmp_md.write_text(md, encoding="utf-8", newline="\n")
    rec = {"record": "T2BP_SCREEN_RULE", "revision": (supersedes["revision"] + 1) if supersedes else 1,
           "written_local": time.strftime("%Y-%m-%d %H:%M:%S %z"), "runs_t2bp_exists": False,
           "statement": ("written before any runs/t2bp output existed; no model was loaded and no model output was read "
                         "to choose it"),
           "rule_md_sha256": C.sha256(tmp_md), "rule_text_sha256": _sha_text(rule_text()),
           "inputs": C.frozen_inputs(), "prefreeze_checks": chk, "supersedes": supersedes,
           "reason": ("review of revision 1 (2026-10-09): a mirror false-apply gate (FA) with the D_inapp_addset stratum, "
                      "IAW >= 0.20 for the named arm, an instrument-validity gate (WF, anchors), false copy on the 8-bit "
                      "rows (T2's definition), the filler control as its own conditional run, close-call wording; "
                      "training A4 rows 65 -> 150") if supersedes else None}
    tmp_js.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp_md, mdp)
    os.replace(tmp_js, jp)
    probs = check_frozen(jp)
    need(not probs, f"the frozen rule was written but does not verify: {probs}")
    return rec


def check_frozen(rec_path: Path | None = None) -> list:
    """[] if SCREEN_RULE.json/.md and every frozen input are unchanged, else the problems (the analysis refuses)."""
    rp = Path(rec_path or C.DOCS / C.SCREEN_RULE_JSON)
    rec = C.read_rule_record(rp)
    probs = C.frozen_problems(rec, {rel: ROOT / rel for rel in C.FROZEN_CODE + C.FROZEN_DATA})
    if isinstance(rec, dict) and rec.get("record") == "T2BP_SCREEN_RULE":
        mdp = rp.with_name(C.SCREEN_RULE_MD)
        if not mdp.is_file() or C.sha256(mdp) != rec.get("rule_md_sha256"):
            probs.append("SCREEN_RULE.md differs from its record")
        if rec.get("rule_text_sha256") != _sha_text(rule_text()):
            probs.append("the rule text in scripts/t2bp_analysis.py differs from the frozen one")
    return probs


# ---------------------------------------------------------------- main
def _write_create_only(outd: Path, stem: str, obj: dict, md: str) -> None:
    outd.mkdir(parents=True, exist_ok=True)
    with (outd / f"{stem}.json").open("x", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, default=str) + "\n")
    with (outd / f"{stem}.md").open("x", encoding="utf-8") as f:
        f.write(md)


def screen_main(run: Path, dev_rows: list, chains: list, st: dict) -> int:
    outd = run / "analysis"
    if (outd / "T2BP_SCREEN.json").exists():
        print(f"{outd / 'T2BP_SCREEN.json'} exists (create-only)")
        return 3
    try:
        prov = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
        arms = [x for x in prov["arms"] if x in C.ARMS]
        need(prov.get("dev_subset") in (None, []), "a subset run cannot be scored")
        need(prov["data_sha256"].get(C.DEV.name) == C.sha256(C.DEV), "dev bank differs from the run's")
        outputs = validate_run(run, dev_rows, chains, arms)
        res = analyse(outputs, dev_rows, chains, st, anchors=anchors_ok(prov))
    except Refuse as e:
        print(str(e))
        return 3
    res["run"] = str(run)
    res["frozen_inputs_unchanged"] = True
    res["screen_rule_record_sha256"] = C.sha256(C.DOCS / C.SCREEN_RULE_JSON)
    res["code_sha256"] = {r: C.sha256(ROOT / r) for r in ("scripts/t2bp_analysis.py", "scripts/t2bp_common.py")}
    _write_create_only(outd, "T2BP_SCREEN", res, render_md(res))
    print(res["rule"]["wording"])
    return 0


def filler_main(frun: Path, screen_run: Path, dev_rows: list, chains: list, st: dict) -> int:
    outd = frun / "analysis"
    if (outd / "T2BP_FILLER.json").exists():
        print(f"{outd / 'T2BP_FILLER.json'} exists (create-only)")
        return 3
    try:
        sp = screen_run / "analysis" / "T2BP_SCREEN.json"
        need(sp.is_file(), f"{sp} missing: the screen is analysed first")
        screen = json.loads(sp.read_text(encoding="utf-8"))
        prov = json.loads((frun / "provenance.json").read_text(encoding="utf-8"))
        sprov = json.loads((screen_run / "provenance.json").read_text(encoding="utf-8"))
        need(prov.get("arms") == ["ffill"] and prov.get("seed") == sprov.get("seed") == 0 and prov.get("steps") == sprov.get("steps")
             and (prov.get("data_variant") or "") == (sprov.get("data_variant") or "") and prov.get("dev_subset") in (None, []),
             "the filler run must be arms [ffill], seed 0, the screen's steps and data variant, the whole dev bank")
        need(prov["data_sha256"].get(C.DEV.name) == C.sha256(C.DEV) == sprov["data_sha256"].get(C.DEV.name),
             "dev bank differs between the screen and the filler run")
        outputs = validate_run(frun, dev_rows, chains, ["ffill"])
        res = analyse(outputs, dev_rows, chains, st, anchors=anchors_ok(prov))
        add = apply_filler(screen["rule"], res["rule"]["metrics"]["ffill"], anchors_ok(prov))
    except Refuse as e:
        print(str(e))
        return 3
    obj = {"record": "T2BP_FILLER_ADDENDUM", "run": str(frun), "screen": str(sp), "screen_sha256": C.sha256(sp),
           "addendum": add, "filler_metrics_on": res["rule"]["metrics"]["ffill"], "descriptive": res["arms"]["ffill"],
           "frozen_inputs_unchanged": True}
    md = (f"# T2b prerequisite screen: R5 decode-compute control ({frun})\n\n**Screen verdict:** {add['screen_verdict']}\n\n"
          f"{add['wording']}\n")
    _write_create_only(outd, "T2BP_FILLER", obj, md)
    print(add["wording"])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--filler-run", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--freeze-rule", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    dev_rows, parents = C.load_dev()
    chains = C.dev_chains(parents)
    st = scorer_selftest(dev_rows, chains)
    golds = ", ".join("%s: %d/%d" % (k, v["correct"], v["n"]) for k, v in st["golds"].items())
    print(f"scorer self-test: ok={st['ok']} golds {golds}", flush=True)
    if not st["ok"]:
        print(json.dumps(st, indent=1)[:4000])
        return 3
    if a.dry_run:
        t0 = time.time()
        dr = dry_run(dev_rows, chains)
        dr["selftest_ok"] = st["ok"]
        dr["code_sha256"] = {r: C.sha256(ROOT / r) for r in ("scripts/t2bp_analysis.py", "scripts/t2bp_common.py")}
        dr["dev_sha256"] = C.sha256(C.DEV)
        dr["seconds"] = round(time.time() - t0, 1)
        dr["torch_imported"] = "torch" in sys.modules
        if not a.no_record:
            C.write_doc("DRYRUN.json", json.dumps(dr, indent=1, default=str) + "\n")
            C.write_doc("SCORER_SELFTEST.json", json.dumps({**st, "dev_sha256": C.sha256(C.DEV),
                                                            "code_sha256": dr["code_sha256"]}, indent=1) + "\n")
        print(f"dry run ok={dr['ok']} {dr['counts']} ({dr['seconds']} s, torch imported: {dr['torch_imported']})")
        return 0 if dr["ok"] else 1
    if a.freeze_rule:
        try:
            rec = freeze_rule()
        except Refuse as e:
            print(str(e))
            return 3
        print(f"frozen: revision {rec['revision']}; SCREEN_RULE.md sha {rec['rule_md_sha256']}; {len(rec['inputs'])} inputs")
        return 0
    probs = check_frozen()
    if probs:
        print("[t2bp-analysis] refusing: the frozen screen rule does not verify: " + "; ".join(probs))
        return 3
    if a.filler_run:
        return filler_main(Path(a.filler_run), Path(a.run), dev_rows, chains, st)
    return screen_main(Path(a.run), dev_rows, chains, st)


if __name__ == "__main__":
    sys.exit(main())
