"""RCWM-T2b prerequisite representation screen: analysis (CPU only; numpy/json; torch is never imported).

  python -B scripts/t2bp_analysis.py --dry-run       # synthetic outcomes only (no run output read): DRYRUN.json and
                                                     # SCORER_SELFTEST.json in docs/t2b/prereq (--no-record: neither)
  python -B scripts/t2bp_analysis.py --freeze-rule   # writes SCREEN_RULE.md + SCREEN_RULE.json (every frozen input's
                                                     # sha256; the previous revision is kept as SCREEN_RULE_rev<N>.*);
                                                     # refuses once any runs/t2bp output exists, or unless the dry run,
                                                     # the tiny test and the audit records match the code and data
  python -B scripts/t2bp_analysis.py                 # runs/t2bp/sd_s0 -> runs/t2bp/sd_s0/analysis/T2BP_SCREEN.{json,md}
  python -B scripts/t2bp_analysis.py --filler-run runs/t2bp/sd_s0_ffill
                                                     # R5 addendum (only after an S-trace win) ->
                                                     # runs/t2bp/sd_s0_ffill/analysis/T2BP_FILLER.{json,md}
Every output is create-only. Before any run output is read: the frozen inputs (SCREEN_RULE.json: this file, the common
module, the runner, the data builder, the rcwmt modules the runner and the scorer use, the vendored trainer, the dev
bank and the training files) must be unchanged (else exit 3), and the scorer self-test must pass (golds 100% in every
arm's format, every heuristic and shortcut policy scored at its exact floor, malformed answers rejected; else exit 3).
A run is scored only if it is the surface-diversity build, DONE.json exists, every output's sha256 equals its
provenance record, one-step ids (in-surface and held-out surface) are the dev bank's ids in order, chain rows are the
dev chains in order, and every recorded chain state equals the arm parser applied to the recorded text.

PRE-STATED SCREEN RULE, revision 3 (T2b prerequisite representation screen, seed 0, surface-diversity build).
What changed and why. Revisions 1 (2026-10-09 03:28) and 2 (07:34, after review) were written before any runs/t2bp
output existed and are kept (docs/t2b/prereq/SCREEN_RULE_rev1.*, SCREEN_RULE_rev2.*). PREREG-3 then returned STATUS:
TRIGGER (DECISIONS R-PREREG-3, D-35, R16): the T2 ledger LoRA took the literal token 'void' as its no-op cue instead of
the stated rule and failed on reordered JSON keys, and BWM's delta world model is likewise locked to key order. A LoRA
can learn a surface cue in place of a rule. Revision 3 applies the operator's decisions (DECISIONS D-36: C1-C6 and 5a
(a)-(c)), again before any runs/t2bp output existed and without any model output being read:
  (1) decision point 1 has fired: the screen trains the surface-diversity build (the same canonical rows, each with a
      seeded rule-text paraphrase, a field-name variant and input key orders drawn from training subsets only); the
      canonical-only build is not run;
  (2) the ranking adds HS, accuracy on a combined held-out surface that no training prompt uses (R = the mean of K4,
      IAW, OOR and HS), so an arm that learned the rule only on its training surfaces cannot win on in-surface scores;
  (3) the F-filler control uses a constant fill length per action kind (amendment 4b), because a fill length matched
      to the trace row by row reveals the answer (on these training rows the trace length alone predicts
      applicability perfectly), which would give the control a hidden scratch channel.
Eligibility, R0 (on the in-surface format rate), R1-R3, the tie-breaks, the close-call wording and R5's thresholds are
unchanged. The rule text and the sha256 of every frozen input are recorded in docs/t2b/prereq/SCREEN_RULE.json; the
runner and this analysis refuse when a frozen input differs. Verbatim in docs/t2b/prereq/SCREEN_RULE.md.

  Inputs. Adapter ON, seed 0, each screen arm A in {D-int, S-trace, B-bits} trained on its surface-diversity file
  data/t2bp_train_sd_<arm>.jsonl (run runs/t2bp/sd_s0), on the dev screen bank data/t2bp_dev.jsonl (1,050 one-step rows
  in 9 strata; 240 k=4 chains). The one-step rows and the chains are evaluated in the in-surface rendering (the
  canonical surface: stages E1, E2); the one-step rows also in the COMBINED held-out-surface rendering (stage E3:
  paraphrase 4 with its state sentence, field names 4, the reversed top-level key order and one of the 3 held-out event
  key orders, rotating by the row's index within its stratum; no training prompt uses any of these; the same rows and
  golds as E1, paired by row). All three arms must be complete with every stage (DONE.json; every output
  hash-verified; the frozen inputs unchanged; the scorer self-test passes before any output is read); a run that is not
  the surface-diversity build, or lacks stage E3, gets no verdict. Answers are scored by the arm's after-only parser
  (t2bp_common.parse: D-int lenient exactly as rcwmt.env.parse_state; the others strict); an unparseable or malformed
  answer is wrong.
  Metrics (point estimates, adapter ON; "8-bit rows" = the 750 one-step rows outside D_ood10 and D_ood12; every metric
  except HS is in-surface):
    IAW(A)  accuracy on D_int_add_wrong (150 rows: applicable, the add bit is already set and the delete clears a set
            bit, so the bitwise answer differs from before+add, from (before&~delete)+add and from before);
    OOR(A)  mean of the accuracies on D_ood10 and D_ood12 (10- and 12-bit masks; out of range; primary per D-52 (5));
    K4(A)   k=4 exact final-state accuracy over the 240 dev chains (self-rollout in A's own frame; a broken chain is
            wrong);
    HS(A)   the mean of the accuracies of the seven 8-bit strata (the ID strata below) in the combined held-out surface
            (stage E3), equal weight;
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
  R0 (instrument validity). If WF(A) < 0.90 for any screen arm (in-surface; a malformed held-out-surface answer is a
     wrong answer that lowers HS, not an instrument failure), or a turn that ran an evaluation stage has no passed
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
  R4. Otherwise rank the eligible arms by R(A) = (K4(A) + IAW(A) + OOR(A) + HS(A)) / 4. The winner is the arm with the
     largest R; if the top two differ by less than 0.02 in R, the one with the larger IAW wins; if those differ by less
     than 0.02, the one with the larger OOR; if still tied, the first in the order D-int, B-bits, S-trace (fewest
     decode tokens). Verdict WINNER=<arm>: that arm (its rendering, the same rows and surfaces, the T2 recipe) runs
     seeds 1 and 2. If the winner is D-int, the result is worded "the representation does not beat the
     rebalanced-data control". If another eligible arm's R is within 0.05 of the winner's, the result is worded as a
     close call: a screen choice, not a finding.
  R5 (decode-compute control). F-filler (the amendment-4b constant fill) is a control and is never eligible. It runs
     only if S-trace wins, as its own run (tag sd_s0_ffill: seed 0, the same rows and surfaces, the T2 recipe, every
     stage including E3; analysed by --filler-run), unless it was part of the sd_s0 run. If R(F-filler) >=
     R(S-trace) - 0.05 (R as in R4), the S-trace result carries the flag TRACE_GAIN_NOT_SEPARABLE_FROM_DECODE_COMPUTE
     (the winner is unchanged; the flag is quoted with every statement of the result). If WF(F-filler) < 0.90 or a
     filler turn lacks a passed anchor, the control is void and the result carries DECODE_COMPUTE_CONTROL_VOID. Until
     the control is analysed, an S-trace win is stated with "decode-compute control pending".
  Everything else (OFF baselines, the other strata and subtypes, out-of-range false copy, the held-out-surface
  out-of-range strata, FC/FA/WF and ID in the held-out surface, the paired held-out minus in-surface contrasts,
  accuracy by held-out key order, the filler's residual length (filler minus trace tokens) as a covariate, union
  cells, shortcut agreement, cmp_ok / calc_ok / flag_ok, decision regret, paired contrasts) is descriptive and cannot
  change the verdict. This is a screen on a dev bank: no confirmatory claim follows from it; a confirmatory claim needs
  a sealed bank and a preregistration (roadmap item 2, "Confirm"), with held-out-surface primaries (D-35).
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

RUN = C.OUT_ROOT / C.SCREEN_TAG
FILLER_RUN = C.OUT_ROOT / C.FILLER_TAG
RULE_REVISION = 3
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
    arms = C.ARMS                     # amendment 4b: the filler target is constant-length, no tokenizer needed
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
        "ffill: keys reordered": '{"after":' + str(g) + ',"fill":"' + C.FILL_UNIT * C.fill_units(ev) + '"}',
        "ffill: after as string": C.filler_text(g).replace(',"after":' + str(g) + "}", ',"after":"' + str(g) + '"}'),
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
    fills = {C.target_text("ffill", r["before"], r["event"], r["width"])[:-len(',"after":' + str(r["gold"]) + "}")]
             for r in dev_rows}
    res["filler_constant_on_dev_golds"] = len(fills) == 1
    res["filler_tokens_ge_trace_tokens"] = None if count is None else all(
        count(C.target_text("ffill", r["before"], r["event"], r["width"])) >= count(C.trace_text(r["before"], r["event"]))
        for r in dev_rows)
    ok = ok and res["filler_constant_on_dev_golds"] and res["filler_tokens_ge_trace_tokens"] is not False
    res["ok"] = bool(ok)
    res["filler_checked"] = True
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
    """R4 of revision 3: the equal-weight mean of K4, IAW, OOR and HS (the held-out-surface accuracy)."""
    return round((mm["K4"] + mm["IAW"] + mm["OOR"] + mm["HS"]) / 4, 4)


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
    """m: {arm: {"IAW", "OOR", "K4", "HS", "FC", "FA", "WF", "ID"}} for the arms that ran (ON). Returns the verdict
    (R0-R5 of revision 3)."""
    for a in C.SCREEN_ARMS:
        need(a in m, f"screen arm {a} missing: no verdict")
        need(m[a].get("HS") is not None, f"screen arm {a} has no held-out-surface score HS (stage E3): no verdict")
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


def heldout_order_cells(hsc: list, dev_rows: list) -> dict:
    """Held-out-surface accuracy by the held-out event key order the row was rendered in (descriptive)."""
    out = {}
    for k, order in enumerate(C.HELDOUT_ORDERS["event"]):
        for part, keep in (("8bit", lambda r: r["width"] == C.WIDTH_ID), ("oor", lambda r: r["width"] != C.WIDTH_ID)):
            idx = [i for i, r in enumerate(dev_rows) if C.stratum_index(r) % 3 == k and keep(r)]
            out[f"order{k}_{part}"] = {**cell(hsc, idx), "event_order": list(order)}
    return out


_COUNT: list = []


def _token_count():
    if not _COUNT:
        try:
            import tokenizers
            tk = tokenizers.Tokenizer.from_file(str(C.TOKENIZER_JSON))
            _COUNT.append(lambda t: len(tk.encode(t, add_special_tokens=False).ids))
        except Exception:                                                      # noqa: BLE001
            _COUNT.append(None)
    return _COUNT[0]


def filler_residual(sc: list, dev_rows: list) -> dict:
    """Amendment 4b's residual as a descriptive covariate: filler gold tokens minus trace gold tokens per row, and the
    filler arm's accuracy by residual value and by stratum (the residual is a property of the row, not of the answer)."""
    count = _token_count()
    if count is None:
        return {"available": False}
    res = [count(C.target_text("ffill", r["before"], r["event"], r["width"])) - count(C.trace_text(r["before"], r["event"]))
           for r in dev_rows]
    by = defaultdict(list)
    for x, s_ in zip(res, sc):
        by[x].append(int(s_["correct"]))
    by_s = defaultdict(list)
    for x, r in zip(res, dev_rows):
        by_s[r["stratum"]].append(x)
    return {"available": True, "residual_tokens": {"min": min(res), "max": max(res), "mean": round(float(np.mean(res)), 3)},
            "accuracy_by_residual": {str(k): {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in sorted(by.items())},
            "mean_residual_by_stratum": {k: round(float(np.mean(v)), 3) for k, v in by_s.items()}}


# ---------------------------------------------------------------- inputs
E3_SIDES = ("on",)                      # == t2bp_run.E3_SIDES (the held-out-surface stage generates ON only)


def eval_files(run: Path, a: str, side: str, data_variant: str) -> dict:
    fs = {"E1": run / a / f"onestep_{side}.jsonl", "E2": run / a / f"chains_{side}.jsonl"}
    if data_variant == "sd" and side in E3_SIDES:
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
    need(dv == C.DATA_VARIANT, f"data variant {dv!r}: revision 3 scores only the surface-diversity build (C1)")
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
            hs = None
            if "heldout" in o:
                hsc = onestep_scores(a, o["heldout"], dev_rows)
                sc_all[(a, side, "hs")] = hsc
                h = summarize_onestep(a, hsc, dev_rows)
                hs = h["ID"]
                res["arms"][a][side]["heldout_surface"] = {
                    "surface": C.HELDOUT_SURFACE, "HS": hs, "strata": h["strata"], "subtypes": h["subtypes"],
                    **{k: h[k] for k in ("IAW", "OOR", "FC", "FA", "WF", "ID")},
                    "by_heldout_event_order": heldout_order_cells(hsc, dev_rows),
                    "paired_heldout_minus_in_surface": {s: paired_rows(hsc, sc, idx_s) for s, idx_s in stratum_rows(dev_rows).items()}}
            if side == "on":
                metrics[a] = {"IAW": one["IAW"], "OOR": one["OOR"], "K4": ch["k4_exact"]["acc"], "HS": hs, "FC": one["FC"],
                              "FA": one["FA"], "WF": one["WF"], "ID": one["ID"]}
                if a == "ffill":
                    res["arms"][a]["filler_residual_covariate_descriptive"] = filler_residual(sc, dev_rows)
    no_hs = sorted(a for a in metrics if metrics[a]["HS"] is None)
    if all(a in metrics for a in C.SCREEN_ARMS) and not no_hs:
        res["rule"] = apply_rule(metrics, anchors_ok=anchors, filler_anchors_ok=anchors)
    else:                                     # a seed-1/2 run of the named arm, or the filler control: descriptive only
        why = (f"the run holds arms {sorted(metrics)}, not all of {list(C.SCREEN_ARMS)} (the rule needs all three)"
               if not all(a in metrics for a in C.SCREEN_ARMS) else
               f"arms {no_hs} have no held-out-surface stage E3 (revision 3 needs HS)")
        res["rule"] = {"verdict": "NO_VERDICT", "winner": None, "flags": [], "notes": [], "metrics": metrics,
                       "anchors_ok": anchors, "R": {a: (r_of(v) if v["HS"] is not None else None) for a, v in metrics.items()},
                       "wording": f"No screen verdict: {why}; descriptive only."}
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
         f"**Verdict (pre-stated rule revision {RULE_REVISION}, scripts/t2bp_analysis.py):** {r['verdict']}"
         + (f" (flags: {', '.join(r['flags'])})" if r.get("flags") else ""), "", r["wording"], "",
         f"Frozen inputs unchanged: {res.get('frozen_inputs_unchanged', '(not checked: in memory)')}. "
         f"Anchors passed in every evaluating turn: {r.get('anchors_ok')}.", "",
         "| Arm | IAW | OOR | K4 | HS | FC | FA | WF | ID | R |", "|---|---|---|---|---|---|---|---|---|---|"]
    f3 = lambda x: "-" if x is None else f"{x:.3f}"   # noqa: E731
    for a, mm in r["metrics"].items():
        L.append(f"| {C.ARM_LABEL[a]} | {f3(mm['IAW'])} | {f3(mm['OOR'])} | {f3(mm['K4'])} | {f3(mm['HS'])} | {f3(mm['FC'])} | "
                 f"{f3(mm['FA'])} | {f3(mm['WF'])} | {f3(mm['ID'])} | {f3(r['R'].get(a))} |")
    if r.get("excluded"):
        L += ["", "Excluded: " + "; ".join(f"{C.ARM_LABEL[a]}: {w}" for a, w in r["excluded"].items())]
    L += ["", "Per-stratum accuracy (OFF -> ON):", "", "| Stratum | " + " | ".join(C.ARM_LABEL[a] for a in res["arms"]) + " |",
          "|---|" + "---|" * len(res["arms"])]
    for s in C.DEV_STRATA:
        L.append(f"| {s} | " + " | ".join(f"{res['arms'][a]['off']['onestep']['strata'][s]['acc']:.2f} -> "
                                          f"{res['arms'][a]['on']['onestep']['strata'][s]['acc']:.2f}" for a in res["arms"]) + " |")
    L += ["", "Held-out surface (stage E3, ON; paired by row with the in-surface ON answers):", "",
          "| Stratum | " + " | ".join(C.ARM_LABEL[a] for a in res["arms"]) + " |", "|---|" + "---|" * len(res["arms"])]
    for s in C.DEV_STRATA:
        L.append(f"| {s} | " + " | ".join(
            (f"{res['arms'][a]['on']['onestep']['strata'][s]['acc']:.2f} -> "
             f"{res['arms'][a]['on']['heldout_surface']['strata'][s]['acc']:.2f}") if "heldout_surface" in res["arms"][a]["on"]
            else "-" for a in res["arms"]) + " |")
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


def synth_outputs(policies: dict, dev_rows: list, chains: list, seed: int = 5, heldout: bool = True) -> dict:
    """policies: {arm: {"off": policy, "on": policy[, "hs": policy]}} -> outputs in the analysis' input format (chains
    rolled out under the policy; the held-out-surface stage E3 (ON) answers under policy "hs", default = "on")."""
    out = {}
    for a, sides in policies.items():
        out[a] = {}
        for side, pol in sides.items():
            if side == "hs":
                continue
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
            if side == "on" and heldout:
                rng_h = random.Random(seed + 1)
                out[a][side]["heldout"] = [policy_text(sides.get("hs", pol), a, r["before"], r["event"], r["width"], rng_h)
                                           for r in dev_rows]
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
    # revision 3: the held-out surface (an entry (in-surface policy, held-out-surface policy) sets E3's answers)
    "rev3: surface-locked arm (the rule in-surface, malformed on the held-out surface) loses to a surface-robust arm": (
        {"dint": "t2like", "strace": ("perfect", "malformed"), "bbits": "perfect_id_ignore_ood"}, "WINNER=bbits"),
    "rev3: surface-locked arm (the rule in-surface, copies on the held-out surface) loses (a close call)": (
        {"dint": "t2like", "strace": ("perfect", "copy"), "bbits": "perfect_id_ignore_ood"}, "WINNER=bbits"),
    "rev3: HS separates two arms tied in-surface (D-int loses the rule on the held-out surface)": (
        {"dint": ("perfect", "copy"), "strace": "perfect", "bbits": "copy"}, "WINNER=strace"),
    "rev3: malformed held-out answers everywhere are wrong answers, not an instrument failure (R0 is in-surface)": (
        {"dint": ("perfect", "malformed"), "strace": ("perfect", "malformed"), "bbits": ("perfect", "malformed")}, "WINNER=dint"),
    "rev3: an ignore-preconditions arm stays excluded by FA however well it does on the held-out surface": (
        {"dint": ("t2like", "copy"), "strace": "ignore_pre", "bbits": ("copy", "perfect")}, "NO_ELIGIBLE_ARM"),
}


def _m(iaw, oor, k4, fc, idd, fa=0.0, wf=1.0, hs=None):
    """A metric table; HS defaults to the mean of IAW, OOR and K4, so R(revision 3) == R(revision 2) on the units that
    were written for revision 2 (their verdicts are unchanged); the revision-3 units set HS explicitly."""
    return {"IAW": iaw, "OOR": oor, "K4": k4, "FC": fc, "ID": idd, "FA": fa, "WF": wf,
            "HS": (iaw + oor + k4) / 3 if hs is None else hs}


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
    # revision 3: R = (K4 + IAW + OOR + HS) / 4
    "rev3: equal in-surface scores, the larger HS wins": (
        {"dint": _m(.6, .6, .6, .1, .9, hs=.5), "strace": _m(.6, .6, .6, .1, .9, hs=.7), "bbits": _m(.3, .3, .3, .1, .9)},
        "WINNER=strace"),
    "rev3: HS weighs 1/4 (a 0.70 HS gain beats a 0.20 in-surface gain on each of K4, IAW, OOR by 0.025)": (
        {"dint": _m(.8, .8, .8, .1, .9, hs=.2), "strace": _m(.1, .1, .1, .1, .9), "bbits": _m(.6, .6, .6, .1, .9, hs=.9)},
        "WINNER=bbits"),
    "rev3: an R gap < 0.02 created by HS still goes to the IAW tie-break": (
        {"dint": _m(.5, .8, .8, .1, .9, hs=.8), "strace": _m(.9, .8, .8, .1, .9, hs=.33), "bbits": _m(.1, .1, .1, .1, .9)},
        "WINNER=strace"),
    "rev3: a perfect HS does not rescue an arm excluded by false apply": (
        {"dint": _m(.5, .5, .5, .1, .9, hs=.5), "strace": _m(1, 1, 1, 0, .9, fa=.3, hs=1), "bbits": _m(.3, .3, .3, .1, .9)},
        "WINNER=dint"),
    "rev3: HS 0 in every arm is no instrument failure (R0 reads the in-surface format rate)": (
        {"dint": _m(.5, .5, .5, .1, .9, hs=0), "strace": _m(.7, .7, .7, .1, .9, hs=0), "bbits": _m(.3, .3, .3, .1, .9, hs=0)},
        "WINNER=strace"),
    "rev3: the filler's R includes its HS (an S-trace win flagged by a filler with the same four scores)": (
        {"dint": _m(.3, .3, .3, .1, .9), "strace": _m(.9, .9, .9, .1, .9, hs=.5), "bbits": _m(.3, .3, .3, .1, .9),
         "ffill": _m(.9, .9, .9, .1, .9, hs=.5)}, W_STRACE_FLAG),
    "rev3: a filler that loses only on HS by more than 0.20 does not flag": (
        {"dint": _m(.3, .3, .3, .1, .9), "strace": _m(.9, .9, .9, .1, .9, hs=.9), "bbits": _m(.3, .3, .3, .1, .9),
         "ffill": _m(.9, .9, .9, .1, .9, hs=.69)}, "WINNER=strace"),
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


def _no_verdict_without_e3(dev_rows: list, chains: list) -> None:
    """Raises Refuse (the expected outcome) if analyse() gives NO_VERDICT to a 3-arm run whose outputs lack stage E3."""
    pols = {a: {"off": "copy", "on": "perfect"} for a in C.SCREEN_ARMS}
    res = analyse(synth_outputs(pols, dev_rows, chains, heldout=False), dev_rows, chains)
    need(res["rule"]["verdict"] != "NO_VERDICT", "NO_VERDICT without stage E3 (expected)")


def dry_run(dev_rows: list, chains: list) -> dict:
    out = {"scenarios": {}, "ok": True}
    for name, (pols, expect) in SCENARIOS.items():
        policies = {a: ({"off": "copy", "on": p[0], "hs": p[1]} if isinstance(p, tuple) else {"off": "copy", "on": p})
                    for a, p in pols.items()}
        res = analyse(synth_outputs(policies, dev_rows, chains), dev_rows, chains)
        v = verdict_string(res["rule"])
        ok = v == expect
        out["scenarios"][name] = {"expected": expect, "got": v, "ok": ok, "R": res["rule"]["R"], "policies_on": pols,
                                  "R_without_HS_revision2_formula": {a: round((mm["K4"] + mm["IAW"] + mm["OOR"]) / 3, 4)
                                                                     for a, mm in res["rule"]["metrics"].items()},
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
                         {"verdict": "WINNER=strace", "R": {"strace": .5}, "metrics": {"ffill": {}}}, _m(1, 1, 1, 0, 1)),
                     "rev3: a screen arm without HS (no held-out stage E3) gives no verdict": lambda: apply_rule(
                         {"dint": _m(1, 1, 1, 0, 1), "strace": {**_m(1, 1, 1, 0, 1), "HS": None}, "bbits": _m(1, 1, 1, 0, 1)}),
                     "rev3: a run without the held-out stage is analysed with NO_VERDICT": lambda: _no_verdict_without_e3(
                         dev_rows, chains)}.items():
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


RULE_PREFACE = """## What changed and why (revision 3, 2026-10-09)

Revision 3 was written before any `runs/t2bp/` output existed; no model was loaded and no model output was read to
choose it. Revisions 1 and 2 are kept (`SCREEN_RULE_rev1.*`, `SCREEN_RULE_rev2.*`).

- **Why.** PREREG-3 returned STATUS: TRIGGER (DECISIONS R-PREREG-3, D-35, R16). The T2 ledger LoRA took the literal
  token 'void' as its no-op cue instead of the stated rule, and it failed on reordered JSON keys (0.37-0.47 vs 0.87-0.91
  in format). BWM's delta world model is likewise key-order-locked (reversed key order 0.458 vs 1.000). A small LoRA can
  learn a surface cue in place of a rule, so a screen that only tests the training surface can pick a cue-learner.
- **C1 (decision point 1 fired).** The screen trains the surface-diversity build: the same 1,500 canonical rows (never
  re-drawn), each with a seeded rule-text paraphrase (with its own state sentence, which carries the width word), a
  field-name variant and input key orders. The canonical-only build is not run.
- **C2 (input key-order augmentation).** Every JSON object of a training prompt is serialized in a key order drawn from
  a training subset of permutations. Held out per object type: the reversed canonical order plus 2 fixed random
  permutations (the 2-key top-level object has only the reversed order to hold out). Targets keep the literal
  cmp -> calc -> after order.
- **C3 / 5a (a) (held-out surface in the ranking).** A new stage E3 evaluates the 1,050 one-step rows in a combined
  held-out surface (paraphrase 4 + field names 4 + a held-out key order, rotating by row), adapter ON, paired by row with
  the in-surface stage. HS = the mean accuracy of the seven 8-bit strata there. R = (K4 + IAW + OOR + HS) / 4.
  Eligibility (IAW >= 0.20, FC <= 0.25, FA <= 0.25, ID non-inferiority), R0 (in-surface format rate), the tie-breaks and
  the close-call wording are unchanged. Per-factor held-out renderings are not run (they do not fit the planning budget).
- **C4 / 5a (c) (amendment 4b).** The F-filler's fill length is a constant per action kind (26 units for the single
  prerequisite kind), at least the longest trace over the training and dev rows. Revision 2 matched the length to the
  trace row by row, and on these training rows that length alone predicts applicability perfectly.
- **5a (b) (revision 2's open questions, answered by the operator; nothing changes).** Accepted as they were: the
  add=0 -> copy cue at 0.62 (descriptive; FC and FA guard it); strict A4 IAW for decision point 2 (the R11 union is
  reported); S-trace's cmp lists bit positions; B-bits writes the most significant bit first; out-of-range prompts change
  only the width word, and since each paraphrase's state sentence words the width its own way, the held-out surface
  words it in held-out form too; ID margin 0.10, FA <= 0.25, WF >= 0.90, close call < 0.05.
- **Unchanged.** The rows, the dev bank, the arms' answer formats and parsers, the recipe, R0-R3, R5's thresholds.
"""


def rule_md() -> str:
    return ("# Pre-stated screen rule (revision 3): T2b prerequisite representation screen (seed 0, surface-diversity "
            "build)\n\n" + RULE_PREFACE + "\n## The rule (verbatim)\n\n"
            "Copied verbatim from the docstring of `scripts/t2bp_analysis.py`, which implements it (`apply_rule`, "
            "`apply_filler`). Hashes of every frozen input: `SCREEN_RULE.json`.\n\n```\n" + rule_text() + "```\n")


def _sha_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def prefreeze_checks(docs: Path) -> dict:
    """The dry run, the tiny test and the audit must be recorded against the current code and data."""
    cur = {r: C.sha256(ROOT / r) for r in ("scripts/t2bp_analysis.py", "scripts/t2bp_common.py", "scripts/t2bp_run.py",
                                           "scripts/t2bp_make_data.py")}
    dr = C.read_rule_record(docs / "DRYRUN.json") or {}
    tt = C.read_rule_record(docs / "TINY_TEST.json") or {}
    au = C.read_rule_record(docs / "AUDIT.json") or {}
    auc = C.read_rule_record(docs / "AUDIT_canonical.json") or {}
    st = C.read_rule_record(docs / "SCORER_SELFTEST.json") or {}
    chk = {
        "dry run ok and recorded against the current analysis/common code": bool(dr.get("ok")) and all(
            (dr.get("code_sha256") or {}).get(r) == cur[r] for r in ("scripts/t2bp_analysis.py", "scripts/t2bp_common.py")),
        "scorer self-test ok on the current dev bank": bool(st.get("ok")) and st.get("dev_sha256") == C.sha256(C.DEV),
        "tiny test ok and recorded against the current code": bool(tt.get("ok")) and all(
            (tt.get("code_sha256") or {}).get(r) == cur[r] for r in cur),
        "audit ok on the current training files (surface-diversity build, AUDIT.json)": bool(au.get("ok")) and
            au.get("variant") == "sd" and all(
            (au.get("files") or {}).get(C.TRAIN_FILES_SD[a].name) == C.sha256(C.TRAIN_FILES_SD[a]) for a in C.ARMS),
        "audit ok on the current canonical files (AUDIT_canonical.json)": bool(auc.get("ok")) and all(
            (auc.get("files") or {}).get(C.TRAIN_FILES[a].name) == C.sha256(C.TRAIN_FILES[a]) for a in C.ARMS),
        "no runs/t2bp output exists": not C.run_outputs(),
    }
    return chk


REASONS = {
    2: ("review of revision 1 (2026-10-09): a mirror false-apply gate (FA) with the D_inapp_addset stratum, IAW >= 0.20 "
        "for the named arm, an instrument-validity gate (WF, anchors), false copy on the 8-bit rows (T2's definition), "
        "the filler control as its own conditional run, close-call wording; training A4 rows 65 -> 150"),
    3: ("operator decisions after PREREG-3 STATUS: TRIGGER (DECISIONS D-35, R16, D-36; C1-C6, 5a (a)-(c)): the screen "
        "trains the surface-diversity build only (paraphrases, field-name variants, input key-order augmentation from "
        "training subsets; held out: paraphrase 4, field names 4, the reversed key order and 2 fixed random event key "
        "orders); a held-out-surface stage E3 (ON, paired by row) and HS in the ranking R = (K4 + IAW + OOR + HS) / 4; "
        "the F-filler's fill length is a constant per action kind (amendment 4b); eligibility, R0-R3, tie-breaks, "
        "close-call wording and R5's thresholds unchanged"),
}


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
           "reason": REASONS.get((supersedes["revision"] + 1) if supersedes else 1)}
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
    with (outd / f"{stem}.json").open("x", encoding="utf-8", newline="\n") as f:      # LF only (DECISIONS I-30)
        f.write(json.dumps(obj, indent=1, default=str) + "\n")
    with (outd / f"{stem}.md").open("x", encoding="utf-8", newline="\n") as f:
        f.write(md)


def screen_main(run: Path, dev_rows: list, chains: list, st: dict) -> int:
    outd = run / "analysis"
    if (outd / "T2BP_SCREEN.json").exists():
        print(f"{outd / 'T2BP_SCREEN.json'} exists (create-only)")
        return 3
    try:
        need((run / "provenance.json").is_file() and (run / "DONE.json").is_file(), f"{run}: no complete run (DONE.json)")
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
    res["rule_revision"] = RULE_REVISION
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
        need((frun / "provenance.json").is_file() and (frun / "DONE.json").is_file(), f"{frun}: no complete run (DONE.json)")
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
