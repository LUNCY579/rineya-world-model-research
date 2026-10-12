"""PREREG-5 analysis: the registered decision rules of PREREG_5_DRAFT.md sections 5-7, applied mechanically (CPU, numpy
only, never torch). DRAFT KIT: nothing here is registered until PREREG-5 is frozen.

  python -B scripts/t2bl_confirm_analysis.py --dryrun [--record]   # every wording branch on synthetic outcomes through
                                                                   # the real metrics / decide / wording code
                                                                   # -> docs/t2b/ledger/P5_DRYRUN.json + P5_DRYRUN.md
  python -B scripts/t2bl_confirm_analysis.py --power [--record]    # power from dev-like assumptions (scenario-based now;
                                                                   # --from-gate <GATE.json> after the dev screen)
                                                                   # -> docs/t2b/ledger/P5_POWER.json
  python -B scripts/t2bl_confirm_analysis.py                       # the registered analysis (once; create-only):
                                                                   # runs/prereg5/analysis/PREREG_5_RESULTS.{json,md}

Parts: (a) the one-step map on the sealed map bank (RULE_MAP, secondary RULE_P3); (b) planning on PREREG-LB's held-out
combined surface (RULE_PLAN); (c) the 5c retention guard (RULE_RET). Limits: LIMITS (always rendered).
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
if "torch" in sys.modules:
    raise SystemExit("t2bl_confirm_analysis.py must run without torch")
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "2"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from collections import Counter, OrderedDict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import t2bl_common as C  # noqa: E402
from rcwmt import env  # noqa: E402

# ================================================================================ registered constants
MAP_SIDES = ("off", "t2s0", "t2bl0", "t2bl1", "t2bl2")
T2BL = ("t2bl0", "t2bl1", "t2bl2")
BAR, BAR_LB = 0.80, 0.70
NI_OWN = 0.10                 # non-inferiority to the adapter's own canonical twins (bootstrap one-sided 95% LB)
ID_NI = 0.05                  # in-distribution non-inferiority to T2 s0 (bootstrap one-sided 95% LB)
ALPHA = 0.05                  # Holm over the 27 (cell, seed) gain tests
NEAR_T2 = 0.10                # 'stays near T2 s0': within this much of it, above or below
CTRL_GAIN = 0.10              # 'improves on trained surface variants': the control at least this much above T2 s0
P_PLAN, DELTA_PLAN = 0.01, 0.20
RATIO_BAR = 0.70              # planning absolute bar: success(T2BL@HS) / success(ORACLE)
NEAR_ORACLE, NEAR_ORACLE_LB = 0.80, 0.70
NI_PLAN = 0.10
CANON_COST = 0.05
P3_TRIGGER_MAX, P3_COPY_CAP = 0.30, 0.10
SURFACE_CTRL = ("S_surface_posted", "S_surface_words")
FACTORS = OrderedDict([
    ("words", {"label": "unseen status words", "cells": (("H_words", "all"),), "control": ("S_words",)}),
    ("order", {"label": "unseen key orders", "cells": (("H_order", "all"),), "control": SURFACE_CTRL}),
    ("names", {"label": "unseen account names", "cells": (("H_names", "all"),), "control": SURFACE_CTRL}),
    ("count", {"label": "an unseen account count (8 accounts)", "cells": (("H_count", "all"),), "control": SURFACE_CTRL}),
    ("para", {"label": "an unseen rule paraphrase", "cells": (("H_para", "all"),), "control": SURFACE_CTRL}),
    ("magnitude", {"label": "unseen magnitudes (posted add / transfer events: 5-digit balances, 4-digit amounts)",
                   "cells": (("H_bal", "at"), ("H_amt", "at")), "control": ("S_bal", "S_amt")}),
    ("combined", {"label": "the combined unseen surface", "cells": (("HS_words", "all"), ("HS_posted", "all")),
                  "control": SURFACE_CTRL}),
])
REPORT_CELLS = (("H_bal", "sc"), ("H_order_top", "all"), ("H_order_inner", "all"), ("LB_void", "all"),
                ("LB_posted", "all"), ("S_words", "all"), ("S_surface_posted", "all"), ("S_surface_words", "all"),
                ("S_bal", "all"), ("S_amt", "all"), ("ID_posted", "all"), ("ID_void", "all"))
WORD_CLUSTER_CELLS = ("H_words", "HS_words")   # report-only: accuracy with a word-cluster bootstrap interval
P3_STRATA = tuple(C.P3.UNSEEN_STATUS_STRATA)
P3_TRIGGER = tuple(C.P3.TRIGGER_STRATA)
VARIANT_KEYS = {"H_words": "word_kind", "HS_words": "word_kind", "S_words": "word_kind", "S_surface_words": "word_kind",
                "H_names": "names_kind", "HS_posted": "names_kind", "H_order": "event_order", "H_count": "status_kind",
                "H_para": "status_kind", "H_order_top": "status_kind", "H_order_inner": "event_order",
                "S_surface_posted": "names_kind"}
PINNED_CODE = ("scripts/t2bl_common.py", "scripts/t2bl_make_data.py", "scripts/t2bl_audit.py", "scripts/t2bl_confirm_bank.py",
               "scripts/t2bl_confirm_run.py", "scripts/t2bl_confirm_analysis.py")

RULE_MAP = (
    "PREREG-5 part (a), the one-step map. Bank: the sealed map bank minted at the freeze (t2bl_common.CONFIRM_STRATA: 28 "
    "strata on 100 fresh bases, rendered with the CONFIRM half of every split held-out partition, which the dev screen "
    "never used; S_bal, S_amt and H_amt on the 67 add / transfer bases). Sides, all generated in one process: OFF "
    "(Titan-Sparse, adapters disabled), T2 s0 (PREREG-2's adapter) and T2b-ledger seeds 0, 1, 2, unmerged. Scoring: "
    "p3_common.score_onestep (exact balances, any key order). For every primary cell c (words: H_words; key order: "
    "H_order; names: H_names; count: H_count; paraphrase: H_para; magnitude: H_bal on the add / transfer bases and "
    "H_amt; combined held-out surface: HS_words and HS_posted; H_order, H_names, H_count and H_para carry posted events "
    "on even bases and the base's training non-posted word on odd bases, the magnitude cells posted events only) and "
    "every T2b-ledger seed s: BAR(c, s) iff accuracy >= 0.80 and Wilson 95% lower bound >= 0.70 (on the 67-row cells "
    "that is at least 55 of 67); GAIN(c, s) iff the exact two-sided McNemar test of seed s against T2 s0 on c's rows is "
    "significant at 0.05 after Holm adjustment over the 27 (cell, seed) tests, with seed s ahead (b > c); NI(c) iff "
    "every seed's difference between its accuracy on c and its accuracy on the same bases' canonical twins (ID_posted "
    "for posted rows, ID_void for the held-out-word rows, S_words for the training-word rows) is >= -0.10 and the "
    "paired base-bootstrap one-sided 95% lower bound of that difference averaged over the three seeds (bases resampled "
    "with all their rows and seeds; 10,000 resamples, seed 17) is >= -0.10. Per factor X: GENERALISES TO UNSEEN X iff "
    "BAR and GAIN hold for every cell of X in all three seeds and NI holds for every cell of X; TRIGGER WIDENED iff the "
    "factor's trained-variant control (words: S_words; magnitude: S_bal and S_amt; every other factor: S_surface_posted "
    "and S_surface_words; all on in-range values) has, in all three seeds, accuracy >= 0.80 and at least 0.10 above T2 "
    "s0's on the same rows, while, on every cell of X and in every seed, the seed's accuracy is within 0.10 of T2 s0's "
    "(above or below) and at least one cell of X misses BAR in every seed; PARTIAL otherwise, reported with the "
    "per-cell, per-variant, per-seed table and its named patterns (below_bar, not_noninferior, no_gain, no_headroom "
    "where T2 s0 itself meets BAR on a cell without GAIN, below_t2s0 where a seed is more than 0.10 below T2 s0 on a "
    "cell, seed_disagreement). In-distribution control: ID_NI iff the paired base-bootstrap one-sided 95% lower bound "
    "of T2b-ledger's accuracy minus T2 s0's on ID_posted and ID_void (200 rows), averaged over the three seeds, is >= "
    "-0.05 (each seed's difference is reported); if ID_NI fails, every part (a) headline carries 'at an "
    "in-distribution cost relative to T2 s0'.")
RULE_P3 = (
    "PREREG-5 part (a), secondary (comparable with PREREG-3's V1; never changes a factor verdict): PREREG-3's V1 rule "
    "applied to each T2b-ledger seed on this bank's eight PREREG-3 strata (U_pending, U_draft, U_cancelled, U_reversed, "
    "U_held, U_neutral, N_voidcase, N_postedlike: PREREG-3's 16 non-posted strings, none of them in T2b-ledger's "
    "training), with ID_posted as the posted twins and ID_void as C_void: RULE iff in every seed every one of the eight "
    "strata meets BAR and the twin guard holds (ID_posted accuracy >= 0.80 and copied <= 0.10); TRIGGER iff in every "
    "seed the seven strata without a form of 'void' are each <= 0.30, ID_void >= 0.80 and more than half of the errors "
    "on those seven strata apply the event as if posted; PARTIAL otherwise. The same rule is applied to T2 s0 in this "
    "run as the comparator. Gains over T2 s0 per stratum and seed: exact two-sided McNemar, Holm over the 24 at 0.05.")
RULE_PLAN = (
    "PREREG-5 part (b), planning. Instances: the sealed planning draw minted at the freeze (lbeam_problems generator "
    "v2, 36 per optimal depth 2-6, N = 180, in range and closed under the vocabulary). Planner: BWM's plan_beam "
    "(native_v1/bwmn/blocks_planner.py, sha256 18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960) with "
    "lbeam_proposer.EnumerateProposer at width 2 and 12 expansions (lbeam_proposer.beam_kwargs), eval_seed 0, every F "
    "wrapped in lbeam_f.guarded (the registered range guard). Arms (one GPU stage each; every turn loads one "
    "PeftModel holding both adapters unmerged and re-runs A0 before any stage): T2s0@HS = "
    "lbeam_f.SurfaceAuditF('T2', 'heldout_combined') with adapter 't2' (PREREG-LB's held-out combined rendering: keys "
    "reversed, accounts bank / card / cash / fees, 'void' rendered 'cancelled'); T2BL@HS = the same with T2b-ledger "
    "seed 0's adapter; T2s0@canon and T2BL@canon = lbeam_f.LedgerF('T2') with each adapter; ORACLE = exact env.step "
    "(CPU only). Success: the one submitted plan reaches the goal exactly under env.step. PRIMARY: T2BL@HS vs T2s0@HS "
    "paired over all 180 instances, exact two-sided McNemar on the discordant pairs (b = T2BL only, c = T2s0 only), "
    "delta = (b - c) / 180: PASS iff p < 0.01 and delta >= +0.20; SHORT iff p < 0.01, b > c and delta < +0.20; "
    "REVERSED iff p < 0.01 and c > b; NS otherwise. Reading: GENERALISES (T2b-ledger plans on the unseen surface) iff "
    "PASS, the absolute bar success(T2BL@HS) / success(ORACLE) >= 0.70 holds, and T2BL@HS is non-inferior to "
    "T2BL@canon (paired instance-bootstrap one-sided 95% lower bound of success(T2BL@HS) - success(T2BL@canon) >= "
    "-0.10; 10,000 resamples, seed 17); TRIGGER WIDENED iff success(T2BL@canon) - success(T2s0@canon) >= -0.05 and the "
    "primary is NS with delta within 0.10 of zero; PARTIAL otherwise, with its named reason (short, below_bar, "
    "not_noninferior, reversed, canonical_cost, ns_with_gain). Report-only: NEAR-ORACLE on the unseen surface iff the "
    "T2BL@HS / ORACLE ratio is >= 0.80 with paired bootstrap 95% lower bound >= 0.70; T2BL@canon vs T2s0@canon (exact "
    "McNemar, the difference with its interval); success by optimal depth and on the needs-lookahead stratum, "
    "verified-plan precision, refused keys and void-twin exposure for every arm; IGNORE_STATUS (CPU).")
RULE_RET = (
    "PREREG-5 part (c), the retention guard: the frozen 5c retention suite (docs/retention/RET_FREEZE.json) on a "
    "confirmation draw minted at run time (ret_gen.py mint --purpose confirm: a fresh OS-entropy seed, the registered "
    "dev selection), run by ret_run.py --mode confirm with T2b-ledger seed 0 as the extra Titan-Sparse adapter "
    "'x_t2bl_s0' and scored once by ret_analysis.py analyze under its registered RULE_CONFIRM. The guard is the "
    "comparison 'x_t2bl_s0_vs_off': GO (k/m binding) iff RETAINED; NO-GO iff LOSS DETECTED or MARGIN EXCEEDED. It is "
    "reported with every headline and never changes a part (a) or (b) verdict; a NO-GO adds 'T2b-ledger s0 loses "
    "retention against Titan-Sparse on the 5c guard; it is not a packaging candidate without retention repair'. If the "
    "retention run is not completed, the guard is 'not evaluable' and every headline says so.")
LIMITS = (
    "Ledger only: Titan's one-step ledger frame family; T2b-ledger seeds 0-2 on frozen FP32 Titan-Sparse, unmerged; T2 s0 "
    "(PREREG-2's artifact) is the unaugmented comparator.",
    "T2 s0 differs from T2b-ledger in more than surface diversity: it trained on 501 ledger rows inside 1,501 rows of "
    "three families; T2b-ledger trains on 1,500 ledger rows with widened magnitudes, twin pairs and 600 rows of "
    "non-posted words. A gain over T2 s0 is a gain of the T2b-ledger recipe, not of surface augmentation alone.",
    "'Unseen' means outside T2b-ledger's training partitions, fixed at build time, and outside the dev screen's half of "
    "them: the confirmation uses 20 held-out status words (four per kind, each token-novel: no 'void' or 'posted' token "
    "and a token no trained status has), letters E-H and 8 held-out word names, held-out orders of the event and before "
    "objects (the reversal and two random orders each) and rule paraphrase 5, none of which the dev screen used. A pass "
    "is transfer to these items, not to every surface.",
    "Shared with the dev screen: the 8-account count and the magnitude bands (the dev gate read neither; the "
    "confirmation's values are freshly drawn), and PREREG-LB's rendering (names bank / card / cash / fees, keys "
    "reversed, 'cancelled'), which both one-step banks carry only in report-only strata. The reversed top-level order is "
    "a TRAINED level since kit revision 3 (both orders of the 2-key top object are trained), so H_order and the "
    "combined-surface cells test held-out event and before orders under a trained top-level reversal; H_order_top is a "
    "trained-variant report and H_order_inner reports the held-out inner orders alone.",
    "The top-level key order is held out nowhere since kit revision 3: both orders of the 2-key top-level object (before "
    "and event) are trained, and the dev bank, this map bank and PREREG-LB's rendering use only those two orders. Every key-order "
    "claim of PREREG-5 ('unseen key orders', H_order, H_order_inner and the combined-surface cells) therefore covers the "
    "inner orders of the event and before objects only; nothing here tests an unseen top-level order.",
    "Kit revision 3: the training data were revised after the revision-2 dev screen stopped (DECISIONS R29: the posted "
    "twins on the combined held-out surface degraded to copying), with knowledge of that screen's aggregate results, "
    "under the dev gate rule's pre-stated revision path (training data only, a fresh dev bank under a new seed, the "
    "same thresholds). The revision-2 dev bank and adapters are spent; neither dev screen used the confirm half.",
    "Every held-out event or before order is a few adjacent swaps from some trained order (every other permutation is "
    "trained; the random held-out orders are at least two swaps from the sorted order, the most trained one).",
    "Field names (before, event, account, amount, kind, other, status, tick) and the kind words are never varied. "
    "'Unseen magnitude' means a digit length that no training prompt or target contains (training values have at most 4 "
    "digits). Scale factors stay in the recorded marginal everywhere and training scale rows keep balances of at most 3 "
    "digits, so magnitude claims cover posted add / transfer events (arithmetic at unseen lengths, not status reading); "
    "scale at the held-out band is report-only.",
    "Status reading under each surface factor is tested with TRAINING non-posted words (the odd rows of H_order, H_names, "
    "H_count and H_para); unseen words under unseen surfaces are the combined cell's job (HS_words).",
    "Planning: in-range instances only, one planner configuration (enum W2 E12), one adapter seed (s0), a model-free "
    "proposer, and PREREG-LB's single held-out combined rendering.",
    "Retention: the five categories the 5c dev selection kept (R17); losses of 6-12 items in one category are often missed.",
    "Thresholds were fixed before any T2b-ledger output, with knowledge of PREREG-3's and PREREG-LB's aggregate results "
    "(T2 s0's per-stratum accuracies, its planning success 0.706 and 5/180, its 0.794 ratio to ORACLE), which informed "
    "the bars and the power scenarios: bar 0.80 with Wilson 0.70, own-canonical margin 0.10, in-distribution margin "
    "0.05, Holm at 0.05 over 27 tests, control gain 0.10, 'near T2 s0' within 0.10; planning +0.20 at p < 0.01, ratio "
    "0.70, margin 0.10.",
    "Rows are treated as independent: each held-out word kind has four words (five rows per word) and the paraphrase "
    "cell one paraphrase, so BAR's Wilson bound and the McNemar tests understate item-level uncertainty; word-cluster "
    "bootstrap intervals for H_words and HS_words are reported (report-only).",
    "Fixed rule: every prompt states the same rule (paraphrased); a pass shows that T2b-ledger follows this fixed rule "
    "on unseen surfaces, not that it reads a rule supplied in the prompt (RCX's replication, R19: neither OFF nor T2 s0 "
    "reads a supplied rule; that needs rule-varied training).",
    "PREREG-3's 16 strings are untrained for T2b-ledger, but two are token-near-trained: voided = [void, ed] (both tokens "
    "trained) and unposted = [un, posted] (the trained affix + 'posted' pattern of xposted, zposted, qposted); Void and "
    "VOID are single tokens never seen in training (the trained case variants are VOId = [VO, Id] and vOid = [v, O, "
    "id]). The secondary PREREG-3 reading counts them as PREREG-3 did; the seven strata without a form of 'void' carry "
    "its TRIGGER clause.",
    "The dev gate conditioned only on the dev half of the held-out partitions and on in-distribution rows; the "
    "confirmation's held-out items are the other half, and its bases, values, planning instances and retention items "
    "are fresh and sealed, minted after the dev gate's PROCEED.",
)
NOT_EVALUABLE_PART = {
    "a": "PREREG-5 (a): not evaluable ({reason}).",
    "b": "PREREG-5 (b): not evaluable ({reason}).",
}
NOT_EVALUABLE_REASONS = {"stopped": "the run ended by a deterministic stop before this part's stages were recorded",
                         "missing": "this part's stages are not recorded"}
RULES_SENTENCE = {
    True: ("The decision rules of parts (a)-(c), the wording templates, the limits and the decision code are "
           "byte-identical to those recorded when the dev gate rule was frozen ({when}), before any adapter trained on "
           "this kit revision's data existed."),
    False: ("The decision rules, wording templates, limits or decision code differ from those recorded when the dev "
            "gate rule was frozen ({when}); the changes were made after the dev screen's outputs existed and are "
            "listed, with their reasons, in PREREG_5.md Appendix A."),
}
THRESHOLDS = {"BAR": 0.80, "BAR_LB": 0.70, "NI_OWN": 0.10, "ID_NI": 0.05, "ALPHA": 0.05, "NEAR_T2": 0.10,
              "CTRL_GAIN": 0.10, "P_PLAN": 0.01, "DELTA_PLAN": 0.20, "RATIO_BAR": 0.70, "NEAR_ORACLE": 0.80,
              "NEAR_ORACLE_LB": 0.70, "NI_PLAN": 0.10, "CANON_COST": 0.05, "P3_TRIGGER_MAX": 0.30, "P3_COPY_CAP": 0.10,
              "BOOT_B": 10000, "BOOT_SEED": 17}


class AnalysisError(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise AnalysisError(msg)


# ================================================================================ part (a)
def cell_index(rows: list, stratum: str, part: str) -> list:
    out = []
    for i, r in enumerate(rows):
        if r["stratum"] != stratum:
            continue
        kc = "sc" if r["event"]["kind"] == "scale" else "at"
        if part == "all" or part == kc:
            out.append(i)
    return out


def score_all(rows: list, texts: dict) -> dict:
    return {k: np.array([C.score(r, t)["correct"] for r, t in zip(rows, v)], dtype=bool) for k, v in texts.items()}


def classes_all(rows: list, texts: dict) -> dict:
    return {k: [C.score(r, t)["cls"] for r, t in zip(rows, v)] for k, v in texts.items()}


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """a vs b on paired rows: b_ = a only, c_ = b only."""
    bb, cc = int((a & ~b).sum()), int((~a & b).sum())
    return {"b": bb, "c": cc, "p": C.mcnemar_p(bb, cc)}


def own_ni(rows: list, sc: dict, ix: list, by_id: dict) -> dict:
    """NI(c): per seed, the point difference (accuracy on c minus accuracy on the same bases' canonical twins); pooled,
    the paired base-bootstrap one-sided 95% lower bound of the seed-averaged difference (bases resampled; each base's
    rows over the three seeds kept together)."""
    per: dict = {}
    seeds = {}
    for s in T2BL:
        d = [int(sc[s][i]) - int(sc[s][by_id[rows[i]["twin"]]]) for i in ix]
        seeds[s] = float(np.mean(d)) if d else float("nan")
        for i, x in zip(ix, d):
            v = per.setdefault(rows[i]["base"], [0, 0])
            v[0] += x
            v[1] += 1
    bl = C.boot_lower([tuple(v) for _, v in sorted(per.items())])
    ok = bl["lo"] >= -NI_OWN - 1e-12 and all(v >= -NI_OWN - 1e-12 for v in seeds.values())
    return {"per_seed": seeds, "diff": bl["diff"], "lower_95": bl["lo"], "pass": ok}


def id_ni(rows: list, sc: dict) -> dict:
    """ID_NI: T2b-ledger minus T2 s0 on ID_posted + ID_void, per seed (point) and pooled over seeds (paired
    base-bootstrap one-sided 95% lower bound of the seed-averaged difference)."""
    ix = [i for i, r in enumerate(rows) if r["stratum"] in C.ID_STRATA]
    per: dict = {}
    seeds = {}
    for s in T2BL:
        seeds[s] = {"acc": float(sc[s][ix].mean()), "diff": float(sc[s][ix].mean() - sc["t2s0"][ix].mean())}
        for i in ix:
            v = per.setdefault(rows[i]["base"], [0, 0])
            v[0] += int(sc[s][i]) - int(sc["t2s0"][i])
            v[1] += 1
    bl = C.boot_lower([tuple(v) for _, v in sorted(per.items())])
    return {"n": len(ix), "acc_t2s0": float(sc["t2s0"][ix].mean()), "per_seed": seeds, "diff": bl["diff"],
            "lower_95": bl["lo"], "pass": bl["lo"] >= -ID_NI - 1e-12}


def acc_block(sc: np.ndarray, ix: list) -> dict:
    k, n = int(sc[ix].sum()), len(ix)
    lo, hi = C.wilson(k, n)
    return {"n": n, "k": k, "acc": k / n if n else float("nan"), "wilson": [lo, hi]}


def decide_map(rows: list, sc: dict, cls: dict | None = None) -> dict:
    by_id = {r["id"]: i for i, r in enumerate(rows)}
    cells = OrderedDict()
    for f, spec in FACTORS.items():
        for st, part in spec["cells"]:
            cells[f"{st}|{part}"] = (f, cell_index(rows, st, part))
    tests = {}
    for ck, (f, ix) in cells.items():
        for s in T2BL:
            tests[f"{ck}@{s}"] = mcnemar(sc[s][ix], sc["t2s0"][ix])
    padj = C.holm({k: v["p"] for k, v in tests.items()})
    cellres = OrderedDict()
    for ck, (f, ix) in cells.items():
        d = {"factor": f, "n": len(ix), "t2s0": acc_block(sc["t2s0"], ix), "off": acc_block(sc["off"], ix), "seeds": {}}
        d["t2s0_meets_bar"] = d["t2s0"]["acc"] >= BAR and d["t2s0"]["wilson"][0] >= BAR_LB
        ni = own_ni(rows, sc, ix, by_id)
        d["ni"] = ni
        for s in T2BL:
            a = acc_block(sc[s], ix)
            t = tests[f"{ck}@{s}"]
            d["seeds"][s] = {**a, "bar": a["acc"] >= BAR and a["wilson"][0] >= BAR_LB, "ni_diff": ni["per_seed"][s],
                             "mcnemar": t, "p_holm": padj[f"{ck}@{s}"], "gain": padj[f"{ck}@{s}"] < ALPHA and t["b"] > t["c"],
                             "minus_t2s0": a["acc"] - d["t2s0"]["acc"]}
        cellres[ck] = d
    ctrl = {}
    for st in sorted({c for spec in FACTORS.values() for c in spec["control"]}):
        ix = cell_index(rows, st, "all")
        ctrl[st] = {s: acc_block(sc[s], ix) for s in T2BL + ("t2s0", "off")}
        for s in T2BL:
            ctrl[st][s]["minus_t2s0"] = ctrl[st][s]["acc"] - ctrl[st]["t2s0"]["acc"]
    factors = OrderedDict()
    for f, spec in FACTORS.items():
        cks = [f"{st}|{part}" for st, part in spec["cells"]]
        per_seed_ok = [all(cellres[c]["seeds"][s]["bar"] and cellres[c]["seeds"][s]["gain"] for c in cks) for s in T2BL]
        ni_ok = all(cellres[c]["ni"]["pass"] for c in cks)
        ctrl_ok = all(ctrl[st][s]["acc"] >= BAR - 1e-12 and ctrl[st][s]["minus_t2s0"] >= CTRL_GAIN - 1e-12
                      for st in spec["control"] for s in T2BL)
        near = all(abs(cellres[c]["seeds"][s]["minus_t2s0"]) <= NEAR_T2 + 1e-12 for c in cks for s in T2BL)
        below = [(c, s) for c in cks for s in T2BL if cellres[c]["seeds"][s]["minus_t2s0"] < -NEAR_T2 - 1e-12]
        misses = any(all(not cellres[c]["seeds"][s]["bar"] for s in T2BL) for c in cks)
        if all(per_seed_ok) and ni_ok:
            v = "GENERALISES"
        elif ctrl_ok and near and misses:
            v = "TRIGGER_WIDENED"
        else:
            v = "PARTIAL"
        pats = []
        if v == "PARTIAL":
            if any(not cellres[c]["seeds"][s]["bar"] for c in cks for s in T2BL):
                pats.append("below_bar")
            if not ni_ok:
                pats.append("not_noninferior")
            ng = [(c, s) for c in cks for s in T2BL if not cellres[c]["seeds"][s]["gain"]]
            if any(not cellres[c]["t2s0_meets_bar"] for c, _ in ng):
                pats.append("no_gain")
            if any(cellres[c]["t2s0_meets_bar"] for c, _ in ng):
                pats.append("no_headroom")
            if below:
                pats.append("below_t2s0")
            if any(per_seed_ok) and not all(per_seed_ok):
                pats.append("seed_disagreement")
        factors[f] = {"verdict": v, "patterns": pats, "cells": cks, "control_ok": ctrl_ok, "near_t2s0": near,
                      "below_t2s0": [f"{c}@{s}" for c, s in below],
                      "a_cell_misses_bar_in_every_seed": misses, "ni_ok": ni_ok, "per_seed_bar_and_gain": per_seed_ok}
    idni = id_ni(rows, sc)
    report = OrderedDict()
    for st, part in REPORT_CELLS:
        ix = cell_index(rows, st, part)
        if ix:
            report[f"{st}|{part}"] = {s: acc_block(sc[s], ix) for s in MAP_SIDES}
    variants = {}
    for st, vk in VARIANT_KEYS.items():
        ix = cell_index(rows, st, "all")
        if not ix:
            continue
        g: dict = {}
        for i in ix:
            g.setdefault(str(rows[i]["meta"].get(vk)), []).append(i)
        variants[st] = {v: {s: acc_block(sc[s], ii) for s in MAP_SIDES} for v, ii in sorted(g.items())}
    clusters = {}
    for st in WORD_CLUSTER_CELLS:
        ix = cell_index(rows, st, "all")
        words = sorted({str(rows[i]["meta"].get("word")) for i in ix})
        clusters[st] = {s: C.cluster_ci([(int(sum(sc[s][i] for i in ix if str(rows[i]["meta"].get("word")) == w)),
                                          sum(1 for i in ix if str(rows[i]["meta"].get("word")) == w)) for w in words])
                        for s in MAP_SIDES}
    out = {"cells": cellres, "factors": factors, "id_ni": idni, "id_cost": not idni["pass"],
           "controls": ctrl, "holm_family": len(tests), "report_only": report, "variants": variants,
           "word_clusters_report_only": clusters}
    out["p3"] = decide_p3(rows, sc, cls) if cls is not None else None
    return out


def _p3_side(rows: list, sc: np.ndarray, cl: list, texts_copied: np.ndarray) -> dict:
    per = {}
    for st in P3_STRATA:
        ix = cell_index(rows, st, "all")
        per[st] = acc_block(sc, ix)
    tp = cell_index(rows, "ID_posted", "all")
    cv = cell_index(rows, "ID_void", "all")
    twin_acc = float(sc[tp].mean())
    twin_copy = float(texts_copied[tp].mean())
    rule = all(per[s]["acc"] >= BAR and per[s]["wilson"][0] >= BAR_LB for s in P3_STRATA) and twin_acc >= BAR and \
        twin_copy <= P3_COPY_CAP + 1e-12
    trig_ix = [i for s in P3_TRIGGER for i in cell_index(rows, s, "all")]
    errs = [cl[i] for i in trig_ix if cl[i] != "correct"]
    applied = sum(e == "applied_as_posted" for e in errs)
    trig = all(per[s]["acc"] <= P3_TRIGGER_MAX + 1e-12 for s in P3_TRIGGER) and float(sc[cv].mean()) >= BAR and \
        applied > len(errs) / 2
    return {"strata": per, "twin_acc": twin_acc, "twin_copied": twin_copy, "c_void": float(sc[cv].mean()),
            "rule": rule, "trigger": trig, "errors_applied": applied, "errors": len(errs)}


def decide_p3(rows: list, sc: dict, cls: dict) -> dict:
    cop = {k: np.array([c == "copied" or (rows[i]["gold_rule"] == "noop" and c == "correct") for i, c in enumerate(v)])
           for k, v in cls.items()}
    sides = {k: _p3_side(rows, sc[k], cls[k], cop[k]) for k in ("t2s0",) + T2BL}
    if all(sides[s]["rule"] for s in T2BL):
        v = "RULE"
    elif all(sides[s]["trigger"] for s in T2BL):
        v = "TRIGGER"
    else:
        v = "PARTIAL"
    tests = {f"{st}@{s}": mcnemar(sc[s][cell_index(rows, st, "all")], sc["t2s0"][cell_index(rows, st, "all")])
             for st in P3_STRATA for s in T2BL}
    padj = C.holm({k: t["p"] for k, t in tests.items()})
    gains = {k: {**t, "p_holm": padj[k], "gain": padj[k] < ALPHA and t["b"] > t["c"]} for k, t in tests.items()}
    t2v = "RULE" if sides["t2s0"]["rule"] else "TRIGGER" if sides["t2s0"]["trigger"] else "PARTIAL"
    return {"verdict": v, "t2s0_verdict": t2v, "sides": sides, "gains": gains,
            "n_gain": sum(g["gain"] for g in gains.values())}


# ================================================================================ part (b)
def boot_mean_diff(a: np.ndarray, b: np.ndarray, B: int = C.BOOT_B, seed: int = C.BOOT_SEED) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = a.size
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    done = 0
    while done < B:
        k = min(2000, B - done)
        idx = rng.integers(0, n, size=(k, n))
        vals[done:done + k] = a[idx].mean(1) - b[idx].mean(1)
        done += k
    return {"diff": float(a.mean() - b.mean()), "lower_95_one_sided": float(np.percentile(vals, 5)),
            "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def boot_ratio(a: np.ndarray, o: np.ndarray, B: int = C.BOOT_B, seed: int = C.BOOT_SEED) -> dict:
    a, o = np.asarray(a, float), np.asarray(o, float)
    n = a.size
    rng = np.random.default_rng(seed)
    vals, dropped, done = [], 0, 0
    while done < B:
        k = min(2000, B - done)
        idx = rng.integers(0, n, size=(k, n))
        x, y = a[idx].mean(1), o[idx].mean(1)
        ok = y > 0
        dropped += int((~ok).sum())
        vals.append(x[ok] / y[ok])
        done += k
    v = np.concatenate(vals)
    return {"ratio": float(a.mean() / o.mean()) if o.mean() > 0 else float("nan"),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [float("nan")] * 2,
            "dropped_zero_oracle": dropped}


def plan_status(b: int, c: int, n: int) -> tuple:
    p = C.mcnemar_p(b, c)
    delta = (b - c) / n
    if p < P_PLAN and c > b:
        return "REVERSED", p, delta
    if p < P_PLAN and delta >= DELTA_PLAN - 1e-12:
        return "PASS", p, delta
    if p < P_PLAN and b > c:
        return "SHORT", p, delta
    return "NS", p, delta


def decide_plan(succ: dict, B: int = C.BOOT_B) -> dict:
    """succ: {'T2BL@HS', 'T2s0@HS', 'T2BL@canon', 'T2s0@canon', 'ORACLE'} -> bool arrays over the same instances."""
    hs, t2hs = np.asarray(succ["T2BL@HS"], bool), np.asarray(succ["T2s0@HS"], bool)
    can, t2can, orc = (np.asarray(succ[k], bool) for k in ("T2BL@canon", "T2s0@canon", "ORACLE"))
    n = hs.size
    b, c = int((hs & ~t2hs).sum()), int((~hs & t2hs).sum())
    status, p, delta = plan_status(b, c, n)
    ratio = float(hs.mean() / orc.mean()) if orc.mean() > 0 else float("nan")
    ni = boot_mean_diff(hs, can, B=B)
    canon_diff = float(can.mean() - t2can.mean())
    bar = ratio >= RATIO_BAR - 1e-12 if not math.isnan(ratio) else False
    ni_ok = ni["lower_95_one_sided"] >= -NI_PLAN - 1e-12
    if status == "PASS" and bar and ni_ok:
        v, reasons = "GENERALISES", []
    elif canon_diff >= -CANON_COST - 1e-12 and status == "NS" and abs(delta) < NEAR_T2 - 1e-12:
        v, reasons = "TRIGGER_WIDENED", []
    else:
        v = "PARTIAL"
        reasons = []
        if status == "SHORT":
            reasons.append("short")
        if status == "REVERSED":
            reasons.append("reversed")
        if status == "PASS" and not bar:
            reasons.append("below_bar")
        if status == "PASS" and not ni_ok:
            reasons.append("not_noninferior")
        if canon_diff < -CANON_COST - 1e-12:
            reasons.append("canonical_cost")
        if status == "NS" and not reasons:
            reasons.append("ns_with_gain")
    rr = boot_ratio(hs, orc, B=B) if orc.mean() > 0 else {"ratio": float("nan"), "ci95": [float("nan")] * 2}
    near = (not math.isnan(rr["ratio"])) and rr["ratio"] >= NEAR_ORACLE - 1e-12 and rr["ci95"][0] >= NEAR_ORACLE_LB - 1e-12
    cc = {"b": int((can & ~t2can).sum()), "c": int((~can & t2can).sum())}
    cc["p"] = C.mcnemar_p(cc["b"], cc["c"])
    return {"verdict": v, "reasons": reasons, "n": n, "primary": {"b": b, "c": c, "p": p, "delta": delta, "status": status},
            "success": {k: float(np.asarray(x, bool).mean()) for k, x in succ.items()},
            "counts": {k: int(np.asarray(x, bool).sum()) for k, x in succ.items()},
            "ratio_hs_oracle": ratio, "bar": bar, "ni_hs_vs_canon": ni, "ni_ok": ni_ok, "canon_diff_vs_t2s0": canon_diff,
            "canon_mcnemar": cc, "canon_diff_ci": boot_mean_diff(can, t2can, B=B), "near_oracle_hs": near,
            "ratio_bootstrap": rr}


def plan_report(insts: list, eps: dict, caches: dict) -> dict:
    """Report-only: per arm success by optimal depth, on the needs-lookahead stratum (lbeam_problems.greedy_solvable
    false), verified-plan precision, optimal-length rate and void-twin exposure (PREREG-LB's definitions, computed by
    lbeam_analysis's own functions, imported read-only)."""
    import lbeam_analysis as LA
    import lbeam_problems as P
    la = np.array([not P.greedy_solvable(i) for i in insts])
    out = {"needs_lookahead_n": int(la.sum()), "arms": {}}
    for arm, rows in eps.items():
        rows = [{**r, "plan": [tuple(a) for a in r.get("plan") or []]} for r in rows]
        summ, s = LA.summarise_eps(insts, rows, la)
        if arm in caches:
            import lbeam_f as LF
            f = LA.f_answer_from_cache(caches[arm], "heldout_combined" if arm.endswith("@HS") else LF.CANONICAL)
        else:
            f = LA.f_answer_exact(arm)
        summ["void_twin_exposure"] = LA.void_twin_exposure(insts, rows, s, f)
        out["arms"][arm] = summ
    return out


# ================================================================================ part (c)
def read_retention(res: dict | None, reason: str | None = None) -> dict:
    if not res:
        return {"status": "NOT_EVALUABLE", "label": "not evaluable", "reason": reason or "no registered retention result"}
    cmp = (res.get("comparisons") or {}).get("x_t2bl_s0_vs_off")
    if not cmp:
        return {"status": "NOT_EVALUABLE", "label": "not evaluable", "reason": "no x_t2bl_s0_vs_off comparison"}
    go = bool(cmp.get("go"))
    return {"status": "GO" if go else "NO_GO", "label": cmp.get("go_label") or ("GO" if go else "NO-GO"),
            "verdict": cmp.get("verdict"), "draw_id": res.get("draw_id")}


# ================================================================================ wording (templates: verbatim in the draft)
T = OrderedDict([
    ("factor_generalises", "T2b-ledger GENERALISES TO {LABEL}{qual}: in all three seeds, accuracy on {cells} {accs} (each "
                           ">= 0.80, Wilson lower bound >= 0.70) and above T2 s0 ({t2}) in every Holm-adjusted McNemar "
                           "test, within the margin of its own canonical accuracy (difference {nis} per seed, seed-mean "
                           "lower bound {lbs}, each >= -0.10)."),
    ("factor_trigger", "TRIGGER WIDENED for {label}{qual}: on trained variants ({controls}) the adapter reached {ctrl} "
                       "(each >= 0.80 and at least 0.10 above T2 s0's {ctrl_t2}), but on {cells} it stayed near T2 s0 "
                       "({accs} vs T2 s0 {t2}; within 0.10 of it in every seed)."),
    ("factor_partial", "PARTIAL for {label}{qual}: accuracy {accs} (T2 s0 {t2}); {bits}. Per-cell and per-variant table "
                       "below."),
    ("bit_below_bar", "below the bar in {list}"),
    ("bit_not_noninferior", "not non-inferior to its own canonical accuracy in {list}"),
    ("bit_no_gain", "no Holm-significant gain over T2 s0 in {list}"),
    ("bit_no_headroom", "no shown gain where T2 s0 already meets the bar ({list})"),
    ("bit_below_t2s0", "more than 0.10 below T2 s0 in {list}"),
    ("bit_seed_disagreement", "the seeds disagree"),
    ("id_cost_qual", " (at an in-distribution cost relative to T2 s0)"),
    ("headline_a", "PREREG-5 (a): GENERALISES TO {g}; TRIGGER WIDENED for {t}; PARTIAL for {p}{cost}"),
    ("headline_a_cost", " (every part (a) result at an in-distribution cost relative to T2 s0)"),
    ("none", "none"),
    ("id_sentence", "In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: {per_seed}; seed mean {mean} "
                    "(one-sided 95% lower bound {lo}){tail}"),
    ("id_seed", "{s} {diff}"),
    ("id_tail_ok", ": non-inferior within 0.05."),
    ("id_tail_cost", ": not shown non-inferior within 0.05."),
    ("p3_sentence", "Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS {status} for "
                    "T2b-ledger (T2 s0 in this run: {t2status}). Accuracy on the eight PREREG-3 strata {allp} across seeds "
                    "(the seven without a form of 'void': {trig}); T2 s0 {t2}; Holm-significant gains over T2 s0 in "
                    "{n_gain} of 24 stratum x seed tests."),
    ("plan_base", "{hs}/{n} vs T2 s0's {t2hs}/{n} on PREREG-LB's held-out combined rendering (delta {delta}, discordant "
                  "{b}/{c}, exact McNemar p {p}; status {status})"),
    ("plan_generalises", "T2b-ledger PLANS ON THE UNSEEN SURFACE: {base}; {ratio} of ORACLE's success (>= 0.70) and within "
                         "the margin of its own canonical success ({can}/{n}; difference {nid}, lower bound {nilb})."),
    ("plan_trigger", "TRIGGER WIDENED (planning): T2b-ledger planned on the canonical surface ({can}/{n} vs T2 s0 "
                     "{t2can}/{n}) but stayed near T2 s0 on the held-out surface: {base}."),
    ("plan_partial", "PARTIAL (planning): {base}; {why}."),
    ("why_short", "the gain is significant but below +0.20"),
    ("why_reversed", "T2 s0 planned significantly better on the held-out surface"),
    ("why_below_bar", "its held-out success is below 0.70 of ORACLE's ({ratio})"),
    ("why_not_noninferior", "its held-out success is not non-inferior to its own canonical success (lower bound {nilb} "
                            "< -0.10)"),
    ("why_canonical_cost", "it plans worse than T2 s0 on the canonical surface ({can}/{n} vs {t2can}/{n})"),
    ("why_ns_with_gain", "the held-out difference is not significant but not near T2 s0 either"),
    ("plan_near", " Report-only: near-oracle on the held-out surface (ratio {ratio}, 95% interval {lo}-{hi})."),
    ("plan_canon_line", "Report-only: on the canonical surface T2b-ledger {can}/{n} vs T2 s0 {t2can}/{n} (difference "
                        "{d}, 95% paired bootstrap CI [{lo}, {hi}], exact McNemar p {p}); ORACLE {orc}/{n}; IGNORE_STATUS "
                        "{ign}/{n}."),
    ("headline_b_GENERALISES", "PREREG-5 (b): PLANS ON THE UNSEEN SURFACE"),
    ("headline_b_TRIGGER_WIDENED", "PREREG-5 (b): TRIGGER WIDENED"),
    ("headline_b_PARTIAL", "PREREG-5 (b): PARTIAL"),
    ("ret_go", "Retention guard (5c, draw {draw}): {label}: T2b-ledger s0 retains against Titan-Sparse."),
    ("ret_nogo", "Retention guard (5c, draw {draw}): NO-GO ({verdict}): T2b-ledger s0 loses retention against "
                 "Titan-Sparse on the 5c guard; it is not a packaging candidate without retention repair."),
    ("ret_ne", "Retention guard (5c): not evaluable ({reason}); every headline is read without it."),
    ("headline_c", "PREREG-5 (c): retention guard {label}"),
])


def _r(xs) -> str:
    """A value or a range over seeds / cells, two decimals ('0.91' or '0.88 to 0.95')."""
    xs = [round(float(x), 2) for x in xs]
    return f"{min(xs):.2f}" if min(xs) == max(xs) else f"{min(xs):.2f} to {max(xs):.2f}"


def _s(x: float) -> str:
    return f"{x:+.3f}"


def _p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def factor_sentence(f: str, A: dict) -> str:
    F = A["factors"][f]
    spec = FACTORS[f]
    cks = F["cells"]
    accs = [A["cells"][c]["seeds"][s]["acc"] for c in cks for s in T2BL]
    t2 = [A["cells"][c]["t2s0"]["acc"] for c in cks]
    nis = [A["cells"][c]["seeds"][s]["ni_diff"] for c in cks for s in T2BL]
    lbs = [A["cells"][c]["ni"]["lower_95"] for c in cks]
    cells = " and ".join(c.replace("|all", "").replace("|at", " (add / transfer)") for c in cks)
    qual = T["id_cost_qual"] if A["id_cost"] else ""
    if F["verdict"] == "GENERALISES":
        return T["factor_generalises"].format(LABEL=spec["label"].upper(), qual=qual, cells=cells, accs=_r(accs),
                                              nis=_r(nis), lbs=_r(lbs), t2=_r(t2))
    if F["verdict"] == "TRIGGER_WIDENED":
        ctrl = [A["controls"][st][s]["acc"] for st in spec["control"] for s in T2BL]
        ctrl_t2 = [A["controls"][st]["t2s0"]["acc"] for st in spec["control"]]
        return T["factor_trigger"].format(label=spec["label"], qual=qual, controls=", ".join(spec["control"]),
                                          ctrl=_r(ctrl), ctrl_t2=_r(ctrl_t2), cells=cells, accs=_r(accs), t2=_r(t2))
    fails = lambda key: [f"{c.split('|')[0]} {s.replace('t2bl', 's')}" for c in cks for s in T2BL  # noqa: E731
                         if not A["cells"][c]["seeds"][s][key]]
    bits = []
    if "below_bar" in F["patterns"]:
        bits.append(T["bit_below_bar"].format(list=", ".join(fails("bar"))))
    if "not_noninferior" in F["patterns"]:
        bits.append(T["bit_not_noninferior"].format(list=", ".join(
            f"{c.split('|')[0]} (seed-mean lower bound {A['cells'][c]['ni']['lower_95']:+.3f})" for c in cks
            if not A["cells"][c]["ni"]["pass"])))
    if "no_gain" in F["patterns"]:
        bits.append(T["bit_no_gain"].format(list=", ".join(
            f"{c.split('|')[0]} {s.replace('t2bl', 's')}" for c in cks for s in T2BL
            if not A["cells"][c]["seeds"][s]["gain"] and not A["cells"][c]["t2s0_meets_bar"])))
    if "no_headroom" in F["patterns"]:
        bits.append(T["bit_no_headroom"].format(list=", ".join(
            f"{c.split('|')[0]} T2 s0 {A['cells'][c]['t2s0']['acc']:.2f}" for c in cks if A["cells"][c]["t2s0_meets_bar"])))
    if "below_t2s0" in F["patterns"]:
        bits.append(T["bit_below_t2s0"].format(list=", ".join(x.split("|")[0] + " " + x.split("@")[1].replace("t2bl", "s")
                                                               for x in F["below_t2s0"])))
    if "seed_disagreement" in F["patterns"]:
        bits.append(T["bit_seed_disagreement"])
    return T["factor_partial"].format(label=spec["label"], qual=qual, accs=_r(accs), t2=_r(t2), bits="; ".join(bits))


def p3_sentence(Pp: dict) -> str:
    s = Pp["sides"]
    trig = [s[x]["strata"][st]["acc"] for x in T2BL for st in P3_TRIGGER]
    allp = [s[x]["strata"][st]["acc"] for x in T2BL for st in P3_STRATA]
    t2 = [s["t2s0"]["strata"][st]["acc"] for st in P3_STRATA]
    return T["p3_sentence"].format(status=Pp["verdict"], t2status=Pp["t2s0_verdict"], allp=_r(allp), trig=_r(trig),
                                   t2=_r(t2), n_gain=Pp["n_gain"])


def plan_sentence(D: dict) -> str:
    c, pr = D["counts"], D["primary"]
    n = D["n"]
    base = T["plan_base"].format(hs=c["T2BL@HS"], n=n, t2hs=c["T2s0@HS"], delta=_s(pr["delta"]), b=pr["b"], c=pr["c"],
                                 p=_p(pr["p"]), status=pr["status"])
    ni = D["ni_hs_vs_canon"]
    if D["verdict"] == "GENERALISES":
        s = T["plan_generalises"].format(base=base, ratio=f"{D['ratio_hs_oracle']:.3f}", can=c["T2BL@canon"], n=n,
                                         nid=_s(ni["diff"]), nilb=_s(ni["lower_95_one_sided"]))
    elif D["verdict"] == "TRIGGER_WIDENED":
        s = T["plan_trigger"].format(can=c["T2BL@canon"], n=n, t2can=c["T2s0@canon"], base=base)
    else:
        why = {"short": T["why_short"], "reversed": T["why_reversed"],
               "below_bar": T["why_below_bar"].format(ratio=f"{D['ratio_hs_oracle']:.3f}"),
               "not_noninferior": T["why_not_noninferior"].format(nilb=_s(ni["lower_95_one_sided"])),
               "canonical_cost": T["why_canonical_cost"].format(can=c["T2BL@canon"], n=n, t2can=c["T2s0@canon"]),
               "ns_with_gain": T["why_ns_with_gain"]}
        s = T["plan_partial"].format(base=base, why="; ".join(why[r] for r in D["reasons"]))
    if D["near_oracle_hs"]:
        rb = D["ratio_bootstrap"]
        s += T["plan_near"].format(ratio=f"{rb['ratio']:.3f}", lo=f"{rb['ci95'][0]:.3f}", hi=f"{rb['ci95'][1]:.3f}")
    return s


def plan_canon_line(D: dict) -> str:
    c, cc, ci = D["counts"], D["canon_mcnemar"], D["canon_diff_ci"]["ci95"]
    return T["plan_canon_line"].format(can=c["T2BL@canon"], n=D["n"], t2can=c["T2s0@canon"], d=_s(D["canon_diff_vs_t2s0"]),
                                       lo=_s(ci[0]), hi=_s(ci[1]), p=_p(cc["p"]), orc=c["ORACLE"],
                                       ign=c.get("IGNORE_STATUS", 0))


def ret_sentence(R: dict) -> str:
    if R["status"] == "GO":
        return T["ret_go"].format(draw=R.get("draw_id"), label=R["label"])
    if R["status"] == "NO_GO":
        return T["ret_nogo"].format(draw=R.get("draw_id"), verdict=R.get("verdict"))
    return T["ret_ne"].format(reason=R.get("reason"))


def wording(A: dict | None, D: dict | None, R: dict, extra: dict | None = None) -> dict:
    extra = extra or {}
    ne = extra.get("not_evaluable") or {}
    out = {"factor_sentences": {}, "limits": list(LIMITS)}
    if A is not None:
        g = [f for f, v in A["factors"].items() if v["verdict"] == "GENERALISES"]
        t = [f for f, v in A["factors"].items() if v["verdict"] == "TRIGGER_WIDENED"]
        p = [f for f, v in A["factors"].items() if v["verdict"] == "PARTIAL"]
        lab = lambda fs: ", ".join(FACTORS[f]["label"] for f in fs) if fs else T["none"]  # noqa: E731
        out["headline_a"] = T["headline_a"].format(g=lab(g), t=lab(t), p=lab(p),
                                                   cost=T["headline_a_cost"] if A["id_cost"] else "")
        out["factor_sentences"] = {f: factor_sentence(f, A) for f in FACTORS}
        out["id_sentence"] = T["id_sentence"].format(
            per_seed=", ".join(T["id_seed"].format(s=s.replace("t2bl", "s"), diff=_s(A["id_ni"]["per_seed"][s]["diff"]))
                               for s in T2BL),
            mean=_s(A["id_ni"]["diff"]), lo=_s(A["id_ni"]["lower_95"]),
            tail=T["id_tail_cost"] if A["id_cost"] else T["id_tail_ok"])
        if A.get("p3"):
            out["p3_sentence"] = p3_sentence(A["p3"])
    else:
        out["headline_a"] = NOT_EVALUABLE_PART["a"].format(reason=NOT_EVALUABLE_REASONS[ne.get("a", "missing")])
    if D is not None:
        out["headline_b"] = T[f"headline_b_{D['verdict']}"]
        out["plan_sentence"] = plan_sentence(D)
        out["plan_canon_line"] = plan_canon_line(D)
    else:
        out["headline_b"] = NOT_EVALUABLE_PART["b"].format(reason=NOT_EVALUABLE_REASONS[ne.get("b", "missing")])
    out["headline_c"] = T["headline_c"].format(label={"GO": R.get("label", "GO"), "NO_GO": "NO-GO",
                                                      "NOT_EVALUABLE": "not evaluable"}[R["status"]])
    out["ret_sentence"] = ret_sentence(R)
    rules = extra.get("rules")
    if rules is not None:
        out["rules_sentence"] = RULES_SENTENCE[bool(rules.get("unchanged"))].format(when=rules.get("when") or "unknown")
    return out


def registered_strings() -> list:
    return [RULE_MAP, RULE_P3, RULE_PLAN, RULE_RET, *LIMITS, *T.values(), *NOT_EVALUABLE_PART.values(),
            *NOT_EVALUABLE_REASONS.values(), *RULES_SENTENCE.values(), *(v["label"] for v in FACTORS.values())]


def registered_strings_missing(draft_text: str) -> list:
    """Every registered string must appear verbatim in the draft (whitespace-normalised: the draft wraps lines; inline
    code backticks are ignored)."""
    norm = lambda s: " ".join(str(s).replace("`", "").split())  # noqa: E731
    d = norm(draft_text)
    return [s for s in registered_strings() if norm(s) not in d]


def rules_digest() -> dict:
    """sha256 over every registered string and the thresholds: recorded when the dev gate rule is frozen (before any
    T2b-ledger output) and compared at PREREG-5's freeze."""
    text = "\n".join(registered_strings()) + "\n" + json.dumps(THRESHOLDS, sort_keys=True)
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "n_strings": len(registered_strings())}


def code_digest(root: Path | None = None) -> dict:
    """sha256 of every file of PREREG-5's decision code (PINNED_CODE): recorded when the dev gate rule is frozen and
    compared at PREREG-5's freeze (a change must be listed there)."""
    root = Path(root or ROOT)
    files = {r: C.sha256(root / r) for r in PINNED_CODE if (root / r).is_file()}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode("utf-8")).hexdigest()}


# ================================================================================ synthetic outcomes (dry run / power)
def synth_correct(rows: list, acc: dict, seed: int) -> np.ndarray:
    """A synthetic correctness vector: per stratum (or 'stratum|part'), round(acc * n) rows correct in a seeded order."""
    rng = random.Random(seed)
    out = np.zeros(len(rows), dtype=bool)
    groups: dict = {}
    for i, r in enumerate(rows):
        part = "sc" if r["event"]["kind"] == "scale" else "at"
        key = f"{r['stratum']}|{part}" if f"{r['stratum']}|{part}" in acc else r["stratum"]
        groups.setdefault(key, []).append(i)
    for key, ix in sorted(groups.items()):
        a = acc.get(key, acc.get("*", 1.0))
        ix = list(ix)
        rng.shuffle(ix)
        for j in ix[: round(a * len(ix))]:
            out[j] = True
    return out


def synth_texts_from(rows: list, correct: np.ndarray, copy_share: float = 0.0, seed: int = 0) -> list:
    """Answer texts realising a correctness vector: correct -> the gold; wrong no-op rows -> applied as posted; wrong
    apply rows -> a wrong value (or, for a share copy_share of them, the copied input)."""
    rng = random.Random(seed)
    out = []
    for r, ok in zip(rows, correct):
        if ok:
            out.append(r["gold"])
        elif r["gold_rule"] == "noop":
            out.append(r["posted_gold"])
        elif rng.random() < copy_share:
            out.append(env.dumps(r["before"]))
        else:
            g = json.loads(r["gold"])
            k0 = sorted(g)[0]
            out.append(env.dumps({**g, k0: g[k0] + 7}))
    return out


def test_bank() -> list:
    """The PREREG-5 map bank's strata on the TEST seed (20261164; tests only, never registered, no union)."""
    import t2bl_make_data as MD
    rows, _ = MD.build_bank(C.TEST_SEED, C.CONFIRM_STRATA, MD.empty_union(), "p5test", "p5-test", split="confirm")
    return rows


# T2 s0-like accuracies (synthetic outcomes and power scenarios). Set with knowledge of PREREG-3's and PREREG-LB's
# aggregate results (stated in LIMITS): unseen statuses 0.01-0.18 (the confirm half has no form of 'void'), reordered keys
# 0.37-0.47 and word names 0.60-0.66 on posted rows, 8 accounts 0.84-0.90, magnitudes 0.14-0.60; the mixed cells pair
# those posted rows with training words, which T2 s0 (it knows only 'void') mostly applies.
T2S0_LIKE = {"*": 0.85, "ID_posted": 0.87, "ID_void": 0.99, "H_words": 0.10, "HS_words": 0.05, "HS_posted": 0.20,
             "S_words": 0.10, "S_surface_words": 0.05, "S_surface_posted": 0.40, "S_bal": 0.50, "S_amt": 0.15,
             "LB_void": 0.02, "LB_posted": 0.25, "H_order": 0.25, "H_order_top": 0.40, "H_order_inner": 0.30,
             "H_names": 0.35, "H_count": 0.47, "H_para": 0.35, "H_bal": 0.50, "H_amt": 0.15,
             **{s: 0.05 for s in P3_STRATA}, "N_voidcase": 0.95}
OFF_LIKE = {"*": 0.30, "ID_void": 0.05, **{s: 0.03 for s in P3_STRATA}}
GOOD = {"*": 0.95, "ID_posted": 0.90, "ID_void": 0.99}


def map_case(rows: list, seeds: dict, t2=None, off=None, copy=0.0) -> tuple:
    texts = {"off": synth_texts_from(rows, synth_correct(rows, off or OFF_LIKE, 1)),
             "t2s0": synth_texts_from(rows, synth_correct(rows, t2 or T2S0_LIKE, 2))}
    for j, s in enumerate(T2BL):
        texts[s] = synth_texts_from(rows, synth_correct(rows, seeds[j], 10 + j), copy, seed=j)
    sc = score_all(rows, texts)
    return decide_map(rows, sc, classes_all(rows, texts)), texts


def dryrun() -> dict:
    rows = test_bank()
    cases = []

    def add(name, A=None, D=None, R=None, check=None, extra=None):
        R = R or {"status": "GO", "label": "GO (5/5 binding)", "draw_id": "confirm_test"}
        W = wording(A, D, R, extra)
        ok = bool(check(A, D, W)) if check else True

        def strings(o):
            if isinstance(o, dict):
                return [x for v in o.values() for x in strings(v)]
            if isinstance(o, (list, tuple)):
                return [x for v in o for x in strings(v)]
            return [str(o)]
        txt = " ".join(strings(W))
        bad = [s for s in ("nan", "None", "{", "}") if s in txt]
        ok = ok and not bad
        cases.append({"case": name, "ok": ok, "headline_a": W.get("headline_a"), "headline_b": W.get("headline_b"),
                      "headline_c": W["headline_c"], "sentences": {k: v for k, v in W.items() if k.endswith("sentence")
                                                                   or k in ("factor_sentences", "plan_canon_line")},
                      "bad_tokens": bad})

    good3 = [GOOD] * 3
    A, _ = map_case(rows, good3)
    add("(a) all seven factors generalise (T2 s0 far below the bar on every primary cell)", A,
        check=lambda A, D, W: all(v["verdict"] == "GENERALISES" for v in A["factors"].values())
        and W["headline_a"].startswith("PREREG-5 (a): GENERALISES TO unseen status words, unseen key orders")
        and "GENERALISES TO UNSEEN STATUS WORDS" in W["factor_sentences"]["words"] and not A["id_cost"])
    A, _ = map_case(rows, good3, t2={**T2S0_LIKE, "H_count": 0.90})
    add("(a) count: T2 s0 already meets the bar -> PARTIAL no_headroom", A,
        check=lambda A, D, W: A["factors"]["count"]["verdict"] == "PARTIAL"
        and "no_headroom" in A["factors"]["count"]["patterns"]
        and all(A["factors"][f]["verdict"] == "GENERALISES" for f in FACTORS if f != "count"))
    tw = {**GOOD, "H_words": 0.10, "HS_words": 0.08}
    A, _ = map_case(rows, [tw] * 3)
    add("(a) words: trained words learned (well above T2 s0), held-out words near T2 s0 -> TRIGGER WIDENED; combined "
        "PARTIAL", A,
        check=lambda A, D, W: A["factors"]["words"]["verdict"] == "TRIGGER_WIDENED"
        and A["factors"]["combined"]["verdict"] == "PARTIAL"
        and "TRIGGER WIDENED for unseen status words" in W["factor_sentences"]["words"])
    A, _ = map_case(rows, [tw] * 3, t2={**T2S0_LIKE, "S_words": 0.90})
    add("(a) words: held-out words near T2 s0 but the trained-word control is not 0.10 above T2 s0 -> PARTIAL", A,
        check=lambda A, D, W: A["factors"]["words"]["verdict"] == "PARTIAL" and not A["factors"]["words"]["control_ok"])
    A, _ = map_case(rows, [{**GOOD, "H_count": 0.50}] * 3, t2={**T2S0_LIKE, "H_count": 0.90})
    add("(a) count: far BELOW T2 s0 (not 'near' it) -> PARTIAL below_t2s0, not TRIGGER WIDENED", A,
        check=lambda A, D, W: A["factors"]["count"]["verdict"] == "PARTIAL"
        and "below_t2s0" in A["factors"]["count"]["patterns"] and "more than 0.10 below T2 s0" in W["factor_sentences"]["count"])
    A, _ = map_case(rows, [{**GOOD, "H_bal|at": 0.52, "H_amt": 0.16}] * 3)
    add("(a) magnitude: trained magnitude bands learned, held-out bands near T2 s0 -> TRIGGER WIDENED", A,
        check=lambda A, D, W: A["factors"]["magnitude"]["verdict"] == "TRIGGER_WIDENED"
        and "S_bal, S_amt" in W["factor_sentences"]["magnitude"])
    A, _ = map_case(rows, [{**GOOD, "H_order": 0.75}] * 3)
    add("(a) order below the bar but gaining -> PARTIAL below_bar", A,
        check=lambda A, D, W: A["factors"]["order"]["verdict"] == "PARTIAL" and "below_bar" in A["factors"]["order"]["patterns"])
    A, _ = map_case(rows, [{**GOOD, "ID_posted": 0.99, "S_words": 0.99, "H_names": 0.82}] * 3)
    add("(a) names at the bar but far below own canonical -> PARTIAL not_noninferior", A,
        check=lambda A, D, W: A["factors"]["names"]["verdict"] == "PARTIAL" and "not_noninferior" in A["factors"]["names"]["patterns"])
    A, _ = map_case(rows, [{**GOOD, "H_para": 0.86}] * 3, t2={**T2S0_LIKE, "H_para": 0.79})
    add("(a) paraphrase: bar met, T2 s0 just below the bar, gain not significant -> PARTIAL no_gain", A,
        check=lambda A, D, W: A["factors"]["para"]["verdict"] == "PARTIAL" and "no_gain" in A["factors"]["para"]["patterns"])
    A, _ = map_case(rows, [GOOD, GOOD, {**GOOD, "H_bal|at": 0.60}])
    add("(a) magnitude: one seed fails -> PARTIAL seed_disagreement + below_bar", A,
        check=lambda A, D, W: A["factors"]["magnitude"]["verdict"] == "PARTIAL"
        and {"seed_disagreement", "below_bar"} <= set(A["factors"]["magnitude"]["patterns"]))
    A, _ = map_case(rows, [{**GOOD, "ID_posted": 0.70, "ID_void": 0.90}] * 3)
    add("(a) in-distribution cost qualifies every headline", A,
        check=lambda A, D, W: A["id_cost"] and "in-distribution cost" in W["headline_a"]
        and all("in-distribution cost" in s for s in W["factor_sentences"].values()))
    A, _ = map_case(rows, [{**GOOD, "S_surface_posted": 0.60, "H_order": 0.30}] * 3)
    add("(a) order near T2 s0 but the trained-surface control also low -> PARTIAL, not TRIGGER WIDENED", A,
        check=lambda A, D, W: A["factors"]["order"]["verdict"] == "PARTIAL")
    A, _ = map_case(rows, good3)
    add("(a) report-only: word-cluster intervals for H_words and HS_words on every side", A,
        check=lambda A, D, W: set(A["word_clusters_report_only"]) == set(WORD_CLUSTER_CELLS)
        and all(A["word_clusters_report_only"][c]["t2bl0"]["n_clusters"] == 20 for c in WORD_CLUSTER_CELLS))
    # secondary P3
    add("(a) secondary: T2b-ledger RULE vs T2 s0 TRIGGER", A,
        check=lambda A, D, W: A["p3"]["verdict"] == "RULE" and A["p3"]["t2s0_verdict"] == "TRIGGER" and "STATUS RULE" in W["p3_sentence"])
    A, _ = map_case(rows, [{**GOOD, **{s: 0.05 for s in P3_STRATA}, "N_voidcase": 0.9}] * 3)
    add("(a) secondary: T2b-ledger TRIGGER on PREREG-3's strings", A,
        check=lambda A, D, W: A["p3"]["verdict"] == "TRIGGER")
    A, _ = map_case(rows, [{**GOOD, "U_neutral": 0.5}] * 3)
    add("(a) secondary: PARTIAL", A, check=lambda A, D, W: A["p3"]["verdict"] == "PARTIAL")
    # part (b)
    n = 180

    def arr(k):
        v = np.zeros(n, bool)
        v[:k] = True
        return v

    def plan(hs, t2hs, can, t2can, orc, shift=0):
        x = lambda k: np.roll(arr(k), shift)  # noqa: E731
        return decide_plan({"T2BL@HS": arr(hs), "T2s0@HS": x(t2hs), "T2BL@canon": arr(can), "T2s0@canon": arr(t2can),
                            "ORACLE": arr(orc)})

    add("(b) GENERALISES", D=plan(120, 5, 125, 127, 160), check=lambda A, D, W: D["verdict"] == "GENERALISES"
        and "PLANS ON THE UNSEEN SURFACE" in W["headline_b"])
    add("(b) GENERALISES + near-oracle qualifier", D=plan(150, 5, 152, 127, 160),
        check=lambda A, D, W: D["verdict"] == "GENERALISES" and D["near_oracle_hs"] and "near-oracle" in W["plan_sentence"])
    add("(b) TRIGGER WIDENED", D=plan(10, 5, 125, 127, 160), check=lambda A, D, W: D["verdict"] == "TRIGGER_WIDENED")
    add("(b) PARTIAL short", D=plan(35, 5, 125, 127, 160),
        check=lambda A, D, W: D["verdict"] == "PARTIAL" and D["reasons"] == ["short"])
    add("(b) PARTIAL below_bar", D=plan(100, 5, 125, 127, 160),
        check=lambda A, D, W: D["verdict"] == "PARTIAL" and "below_bar" in D["reasons"])
    add("(b) PARTIAL not_noninferior", D=plan(115, 5, 160, 127, 160),
        check=lambda A, D, W: D["verdict"] == "PARTIAL" and "not_noninferior" in D["reasons"])
    add("(b) PARTIAL canonical cost + near T2 s0 on held-out", D=plan(8, 5, 90, 127, 160),
        check=lambda A, D, W: D["verdict"] == "PARTIAL" and "canonical_cost" in D["reasons"])
    add("(b) PARTIAL reversed", D=plan(0, 40, 125, 127, 160, shift=100),
        check=lambda A, D, W: D["primary"]["status"] == "REVERSED" and D["verdict"] == "PARTIAL"
        and D["reasons"] == ["reversed"])
    add("(b) PARTIAL reversed with canonical cost", D=plan(0, 40, 60, 127, 160, shift=100),
        check=lambda A, D, W: D["verdict"] == "PARTIAL" and {"reversed", "canonical_cost"} <= set(D["reasons"]))
    hs, t2hs = np.zeros(n, bool), np.zeros(n, bool)
    hs[:100] = True
    t2hs[60:140] = True
    D_ns = decide_plan({"T2BL@HS": hs, "T2s0@HS": t2hs, "T2BL@canon": arr(125), "T2s0@canon": arr(127), "ORACLE": arr(160)})
    add("(b) NS but not near T2 s0 -> PARTIAL ns_with_gain", D=D_ns,
        check=lambda A, D, W: D["primary"]["status"] == "NS" and D["verdict"] == "PARTIAL" and D["reasons"] == ["ns_with_gain"])
    # part (c)
    add("(c) NO-GO", D=plan(120, 5, 125, 127, 160), R={"status": "NO_GO", "label": "NO-GO", "verdict": "LOSS DETECTED",
                                                       "draw_id": "confirm_test"},
        check=lambda A, D, W: "NO-GO" in W["headline_c"] and "not a packaging candidate" in W["ret_sentence"])
    add("(c) not evaluable", D=plan(120, 5, 125, 127, 160), R=read_retention(None),
        check=lambda A, D, W: "not evaluable" in W["headline_c"])
    add("(c) GO from a ret_analysis result", R=read_retention({"draw_id": "confirm_x", "comparisons": {
        "x_t2bl_s0_vs_off": {"go": True, "go_label": "GO (5/5 binding)", "verdict": "RETAINED"}}}),
        check=lambda A, D, W: W["headline_c"].endswith("GO (5/5 binding)"))
    # parts not evaluable, the rules sentence
    add("(a) and (b) not evaluable after an early stop", R=read_retention(None, "the retention run was not completed and bound"),
        extra={"not_evaluable": {"a": "stopped", "b": "stopped"}},
        check=lambda A, D, W: W["headline_a"].startswith("PREREG-5 (a): not evaluable (the run ended by a deterministic stop")
        and W["headline_b"].startswith("PREREG-5 (b): not evaluable") and "not evaluable" in W["headline_c"])
    A_ok, _ = map_case(rows, good3)
    add("(b) not evaluable, (a) rendered (a stop after M1)", A=A_ok, extra={"not_evaluable": {"b": "stopped"}},
        check=lambda A, D, W: W["headline_a"].startswith("PREREG-5 (a): GENERALISES TO") and "not evaluable" in W["headline_b"])
    add("rules and code unchanged since the dev gate", D=plan(120, 5, 125, 127, 160),
        extra={"rules": {"unchanged": True, "when": "2026-10-11 09:00:00 -0500"}},
        check=lambda A, D, W: W["rules_sentence"].startswith("The decision rules of parts (a)-(c), the wording templates")
        and "2026-10-11 09:00:00" in W["rules_sentence"])
    add("rules or code changed after the dev gate", D=plan(120, 5, 125, 127, 160),
        extra={"rules": {"unchanged": False, "when": "2026-10-11 09:00:00 -0500"}},
        check=lambda A, D, W: "differ from those recorded" in W["rules_sentence"])
    # every template is used by some realised case or is a fragment of one (no orphan template)
    allw = json.dumps([c for c in cases])
    unused = [k for k in T if k not in ("none",) and T[k].split("{")[0].strip()[:24] not in allw]
    cases.append({"case": "every wording template is realised by at least one case", "ok": not unused, "detail": unused})
    # statistics units
    st = stats_selftest()
    cases.append({"case": "statistics units (McNemar, Holm, bootstrap, Wilson) vs reference implementations",
                  "ok": st["ok"], "detail": st})
    return {"ok": all(c["ok"] for c in cases), "n_cases": len(cases), "cases": cases,
            "map_verdicts_covered": sorted({v for c in cases for v in [c.get("headline_a")] if v}),
            "code_sha256": C.code_hashes(("scripts/t2bl_confirm_analysis.py", "scripts/t2bl_common.py"))}


def stats_selftest() -> dict:
    import p3_common as P3
    ok = []
    try:
        import lbeam_analysis as LA                 # numpy-only analysis of PREREG-LB (read-only reference)
        for b, c in ((0, 0), (5, 0), (20, 3), (113, 0), (7, 9), (50, 61)):
            ok.append(abs(C.mcnemar_p(b, c) - LA.mcnemar_p(b, c)) < 1e-12)
        ps = {"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20}
        ref = LA.holm(ps)
        ok.append(all(abs(C.holm(ps)[k] - ref[k]) < 1e-12 for k in ps))
    except ImportError:
        ok.append(False)
    ok.append(abs(C.mcnemar_p(10, 0) - 2 * 0.5 ** 10) < 1e-15)
    ok.append(C.holm({"x": 0.01, "y": 0.02}) == {"x": 0.02, "y": 0.02})
    ok.append(C.wilson(80, 100) == P3.wilson(80, 100))
    ok.append(C.wilson(55, 67)[0] >= BAR_LB > C.wilson(54, 67)[0])          # the effective bar on 67 rows is 55 / 67
    bl = C.boot_lower([(1, 1)] * 50 + [(0, 1)] * 50)
    ok.append(abs(bl["diff"] - 0.5) < 1e-12 and 0.38 < bl["lo"] < 0.45)
    cc = C.cluster_ci([(5, 5)] * 10 + [(0, 5)] * 10)
    ok.append(abs(cc["acc"] - 0.5) < 1e-12 and cc["ci95"][0] < 0.5 < cc["ci95"][1] and cc["n_clusters"] == 20)
    a = np.array([True] * 90 + [False] * 90)
    d = boot_mean_diff(a, a)
    ok.append(d["diff"] == 0.0 and d["lower_95_one_sided"] == 0.0)
    consts = {"BAR": BAR, "BAR_LB": BAR_LB, "NI_OWN": NI_OWN, "ID_NI": ID_NI, "ALPHA": ALPHA, "NEAR_T2": NEAR_T2,
              "CTRL_GAIN": CTRL_GAIN, "P_PLAN": P_PLAN, "DELTA_PLAN": DELTA_PLAN, "RATIO_BAR": RATIO_BAR,
              "NEAR_ORACLE": NEAR_ORACLE, "NEAR_ORACLE_LB": NEAR_ORACLE_LB, "NI_PLAN": NI_PLAN, "CANON_COST": CANON_COST,
              "P3_TRIGGER_MAX": P3_TRIGGER_MAX, "P3_COPY_CAP": P3_COPY_CAP, "BOOT_B": C.BOOT_B, "BOOT_SEED": C.BOOT_SEED}
    ok.append(consts == THRESHOLDS)
    return {"ok": all(ok), "checks": len(ok), "passed": sum(ok)}


# ================================================================================ power (dev-like assumptions)
TWIN_P = {"noop": 0.99, "apply": 0.93, "mixed": 0.95}   # the canonical twins' accuracy in the scenarios
SEED_REGIMES = (("seeds independent", 0.0), ("seeds 80% shared", 0.8), ("seeds identical", 1.0))


def _boot_lo_fast(d_by_base: np.ndarray, cnt: np.ndarray, rng, B: int = 400) -> float:
    idx = rng.integers(0, d_by_base.size, size=(B, d_by_base.size))
    m = d_by_base[idx].sum(1) / cnt[idx].sum(1)
    return float(np.percentile(m, 5))


def power_map(p_on: float, p_twin: float, p_t2: float, n: int, reps: int = 400, seed: int = 5, m_holm: int = 27,
              shared: float = 0.0) -> dict:
    """P(GENERALISES on one cell) = P(BAR and GAIN in all three seeds, and NI) for a cell of n rows (one row per base),
    under two correlations of the cell with its canonical twins (nested: the lower success set inside the higher one;
    independent) and a seed correlation `shared` (each row's latent draws are shared by the three seeds with that
    probability: 0 independent seeds, 1 identical seeds); T2 s0 independent of the seeds. GAIN uses the Holm threshold
    bounds alpha / m (worst case) and alpha (best case). NI: every seed's point difference >= -0.10 and the
    base-bootstrap (B = 400 here) one-sided 95% lower bound of the seed-mean difference >= -0.10 (with the analysis'
    1e-12 tolerance)."""
    out = {"p_on": p_on, "p_twin": p_twin, "p_t2s0": p_t2, "n": n, "reps": reps, "shared": shared}
    for regime in ("nested", "independent"):
        rng = np.random.default_rng(seed)
        hit_lo = hit_hi = bar_all = ni_all = 0
        for _ in range(reps):
            t2 = rng.random(n) < p_t2
            uc0, ut0 = rng.random(n), rng.random(n)
            ok_lo = ok_hi = okbar = True
            diffs = np.zeros(n)
            seed_ok = True
            for _s in range(3):
                m = rng.random(n) < shared
                uc, ut = np.where(m, uc0, rng.random(n)), np.where(m, ut0, rng.random(n))
                if regime == "nested":
                    if p_on <= p_twin:
                        tw = ut < p_twin
                        on = tw & (uc < (p_on / p_twin if p_twin > 0 else 0.0))
                    else:
                        on = uc < p_on
                        tw = on & (ut < p_twin / p_on)
                else:
                    on, tw = uc < p_on, ut < p_twin
                k = int(on.sum())
                lo, _ = C.wilson(k, n)
                bar = k / n >= BAR - 1e-12 and lo >= BAR_LB - 1e-12
                d = on.astype(float) - tw.astype(float)
                diffs += d
                seed_ok &= d.mean() >= -NI_OWN - 1e-12
                b, c = int((on & ~t2).sum()), int((~on & t2).sum())
                p = C.mcnemar_p(b, c)
                ok_lo &= bar and b > c and p < ALPHA / m_holm
                ok_hi &= bar and b > c and p < ALPHA
                okbar &= bar
            ni = seed_ok and _boot_lo_fast(diffs, np.full(n, 3.0), rng) >= -NI_OWN - 1e-12
            hit_lo += ok_lo and ni
            hit_hi += ok_hi and ni
            bar_all += okbar
            ni_all += ni
        out[regime] = {"P_generalises_worst_case_holm": hit_lo / reps, "P_generalises_best_case_holm": hit_hi / reps,
                       "P_bar_all_seeds": bar_all / reps, "P_ni": ni_all / reps}
    return out


def power_plan(p_hs: float, p_can: float, p_t2: float = 5 / 180, p_orc: float = 0.889, n: int = 180, reps: int = 300,
               seed: int = 11) -> dict:
    """P(part (b) reads GENERALISES) by simulation through the registered decide_plan (bootstrap B reduced to 2,000 for
    the simulation) under two regimes: 'nested' (T2BL@HS successes inside T2BL@canon's, those inside ORACLE's: the best
    case for the held-out-vs-canonical non-inferiority) and 'independent within ORACLE' (both inside ORACLE's, otherwise
    independent); T2s0@HS independent at p_t2 (PREREG-LB's 5/180); T2s0@canon independent at 0.706 (PREREG-LB). Also
    the exact P(primary PASS) under independence (lbeam_analysis.exact_pass_prob)."""
    import lbeam_analysis as LA
    out = {"p_t2bl_hs": p_hs, "p_t2bl_canon": p_can, "p_t2s0_hs": round(p_t2, 4), "p_oracle": p_orc, "n": n, "reps": reps,
           "P_primary_PASS_exact_independent": LA.exact_pass_prob(n, p_hs, p_t2, P_PLAN, DELTA_PLAN)}
    for regime in ("nested", "independent_within_oracle"):
        rng = np.random.default_rng(seed)
        cnt = Counter()
        for _ in range(reps):
            orc = rng.random(n) < p_orc
            can = orc & (rng.random(n) < min(1.0, p_can / p_orc))
            if regime == "nested":
                hs = can & (rng.random(n) < (p_hs / p_can if p_can > 0 else 0.0))
            else:
                hs = orc & (rng.random(n) < min(1.0, p_hs / p_orc))
            t2hs = rng.random(n) < p_t2
            t2can = rng.random(n) < 0.706
            D = decide_plan({"T2BL@HS": hs, "T2s0@HS": t2hs, "T2BL@canon": can, "T2s0@canon": t2can, "ORACLE": orc}, B=2000)
            cnt[D["verdict"]] += 1
            cnt["PASS"] += D["primary"]["status"] == "PASS"
        out[regime] = {"P_GENERALISES": cnt["GENERALISES"] / reps, "P_PARTIAL": cnt["PARTIAL"] / reps,
                       "P_TRIGGER_WIDENED": cnt["TRIGGER_WIDENED"] / reps, "P_primary_PASS_sim": cnt["PASS"] / reps}
    return out


def gate_alignment(ps=(0.85, 0.90, 0.95), reps_gate: int = 2000, reps_map: int = 400) -> list:
    """P(the dev gate PROCEEDs) against P(PREREG-5 reads GENERALISES on the held-out-word cells), when every seed's true
    held-out-word accuracy is p in the dev screen and in the confirmation (ID_void 0.99, ID_posted 0.90, held-out posted
    twins 0.95, T2 s0 ID 0.93; T2 s0 0.10 on H_words and 0.05 on HS_words), with independent and 80%-shared seeds."""
    import t2bl_gate as G
    out = []
    for p in ps:
        for label, sh in SEED_REGIMES[:2]:
            hw = power_map(p, TWIN_P["noop"], T2S0_LIKE["H_words"], 100, reps=reps_map, shared=sh)
            hs = power_map(p, TWIN_P["noop"], T2S0_LIKE["HS_words"], 100, reps=reps_map, shared=sh)
            out.append({"p_true": p, "seeds": label, "P_PROCEED": G.gate_power(p, shared=sh, reps=reps_gate),
                        "P_GENERALISES_H_words_nested": hw["nested"]["P_generalises_worst_case_holm"],
                        "P_GENERALISES_HS_words_nested": hs["nested"]["P_generalises_worst_case_holm"]})
    return out


def power(from_gate: Path | None = None) -> dict:
    res = {"method": ("simulation through the registered rules (map: per cell, P(BAR and GAIN in all three seeds and NI) "
                      "under nested and independent cell-vs-twin correlation and three seed regimes (independent, 80% "
                      "shared, identical), Holm bounded by alpha / 27 and alpha; in-distribution control: P(ID_NI) under "
                      "nested and independent T2b-ledger-vs-T2 s0 correlation and the same seed regimes; planning: "
                      "P(GENERALISES) through decide_plan with nested success sets and with success sets independent "
                      "within ORACLE's, and the exact P(primary PASS) under independence; the dev gate: P(PROCEED) "
                      "against P(GENERALISES) on the held-out-word cells). The strata share bases, so joint figures "
                      "over factors are not products of these."),
           "assumption_source": "scenarios" if from_gate is None else str(from_gate),
           "twin_accuracy_assumed": TWIN_P}
    scen = []
    if from_gate is not None:
        G = C.read_json(from_gate) or {}
        st = (G.get("metrics") or {}).get("strata") or {}
        for f, spec in FACTORS.items():
            for cs, part in spec["cells"]:
                if cs not in (st.get("s0") or {}):
                    continue
                rule = C.STRATA[cs][1]
                tw = {"noop": ["ID_void"], "apply": ["ID_posted"], "mixed": ["ID_posted", "S_words"]}[rule]
                p_tw = min(sum(st[f"s{s}"][t]["acc"] for t in tw) / len(tw) for s in (0, 1, 2))
                scen.append((f, cs, min(st[f"s{s}"][cs]["acc"] for s in (0, 1, 2)), p_tw, st["t2s0"][cs]["acc"]))
        id_on = min(st[f"s{s}"]["ID_posted"]["acc"] + st[f"s{s}"]["ID_void"]["acc"] for s in (0, 1, 2)) / 2 if st else 0.93
        id_t2 = (st["t2s0"]["ID_posted"]["acc"] + st["t2s0"]["ID_void"]["acc"]) / 2 if st else 0.93
        res["id_ni"] = id_power([(id_on, id_t2)])
        hs = min(st[f"s{s}"]["LB_void"]["acc"] * st[f"s{s}"]["LB_posted"]["acc"] for s in (0,)) if st else 0.5
        ph = min(0.85, 0.889 * hs)
        res["plan"] = [power_plan(ph, max(0.75, ph))]
        res["plan_note"] = ("planning scenario from the dev screen: T2BL@HS success = ORACLE x the product of s0's LB_void "
                            "and LB_posted one-step accuracies (a crude per-instance proxy), T2BL@canon 0.75")
    else:
        for f, spec in FACTORS.items():
            for cs, part in spec["cells"]:
                t2 = T2S0_LIKE.get(cs, 0.5)
                for p_on in (0.85, 0.90, 0.95):
                    scen.append((f, cs, p_on, TWIN_P[C.STRATA[cs][1]], t2))
        res["id_ni"] = id_power([(0.90, 0.93), (0.93, 0.93), (0.96, 0.93)])
        res["plan"] = [power_plan(ph, pc) for ph, pc in ((0.30, 0.70), (0.50, 0.70), (0.60, 0.70), (0.65, 0.70),
                                                          (0.70, 0.75))]
        res["plan_note"] = ("T2 s0's held-out-surface planning success is taken as 5/180 (PREREG-LB's registered D-35 "
                            "audit, REPORT 3d) and its canonical success as 0.706 (PREREG-LB); ORACLE 0.889 (PREREG-LB); "
                            "N = 180 is PREREG-LB's and was not chosen from it.")
    res["map"] = []
    for f, cs, p_on, tw, t2 in scen:
        n = 67 if cs in ("H_amt", "H_bal") else 100
        res["map"].append({"factor": f, "cell": cs, "n": n, "p_on": p_on, "p_twin": tw, "p_t2s0": t2,
                           "regimes": {label: {k: v for k, v in power_map(p_on, tw, t2, n, shared=sh).items()
                                               if k in ("nested", "independent")} for label, sh in SEED_REGIMES}})
    res["gate_alignment"] = gate_alignment()
    return res


def id_power(pairs, reps: int = 400, seed: int = 9) -> dict:
    """P(ID_NI): 200 rows (100 bases x 2), T2 s0 at p_t2, each seed at p, nested or independent T2b-ledger-vs-T2 s0
    correlation, three seed regimes (each row's latent shared by the seeds with probability 0, 0.8, 1); the
    base-bootstrap lower bound of the seed-mean difference (B = 400 here)."""
    out = {}
    for p, p_t2 in pairs:
        rec = {}
        for label, sh in SEED_REGIMES:
            for regime in ("nested", "independent"):
                rng = np.random.default_rng(seed)
                hit = 0
                for _ in range(reps):
                    u_t2 = rng.random(200)
                    t2 = u_t2 < p_t2
                    u0 = rng.random(200)
                    diffs = np.zeros(100)
                    for _s in range(3):
                        u = np.where(rng.random(200) < sh, u0, rng.random(200))
                        if regime == "nested":
                            if p <= p_t2:
                                on = t2 & (u < p / p_t2)
                            else:
                                on = t2 | (u < (p - p_t2) / (1 - p_t2))
                        else:
                            on = u < p
                        d = on.astype(float) - t2.astype(float)
                        diffs += d[0::2] + d[1::2]
                    hit += _boot_lo_fast(diffs, np.full(100, 6.0), rng) >= -ID_NI - 1e-12
                rec[f"{label}, {regime}"] = hit / reps
        out[f"seed_acc_{p:.2f}_vs_t2s0_{p_t2:.2f}"] = rec
    return out


# ================================================================================ the draft (rendered from its template)
DRAFT_TEMPLATE = C.DOCS / "PREREG_5_DRAFT.template.md"
DRAFT_PATH = C.ROOT / "PREREG_5_DRAFT.md"


def _power_table(pw: dict) -> str:
    regs = [label for label, _ in SEED_REGIMES]
    L = ["P(GENERALISES) per cell, Holm worst case (alpha / 27), shown as nested / independent cell-vs-twin correlation:",
         "", "| factor | cell | n | true accuracy (every seed) | twin | T2 s0 | " + " | ".join(regs) + " |",
         "|---|---|---:|---:|---:|---:|" + "---:|" * len(regs)]
    for m in pw.get("map") or []:
        vals = [f"{float(m['regimes'][r]['nested']['P_generalises_worst_case_holm']):.2f} / "
                f"{float(m['regimes'][r]['independent']['P_generalises_worst_case_holm']):.2f}" for r in regs]
        L.append(f"| {m['factor']} | {m['cell']} | {m['n']} | {m['p_on']:.2f} | {m['p_twin']:.2f} | {m['p_t2s0']:.2f} | "
                 + " | ".join(vals) + " |")
    L += ["", "| in-distribution control (each seed vs T2 s0) | " + " | ".join(f"P(ID_NI) {r}, nested / independent"
                                                                           for r in regs) + " |",
          "|---|" + "---:|" * len(regs)]
    for k, v in (pw.get("id_ni") or {}).items():
        L.append(f"| {k} | " + " | ".join(f"{float(v[f'{r}, nested']):.2f} / {float(v[f'{r}, independent']):.2f}"
                                           for r in regs) + " |")
    L += ["", "| planning: T2BL@HS / T2BL@canon success | P(GENERALISES) nested | P(GENERALISES) independent within "
              "ORACLE | P(primary PASS), simulated / exact under independence |", "|---|---:|---:|---:|"]
    for r in pw.get("plan") or []:
        L.append(f"| {r['p_t2bl_hs']:.2f} / {r['p_t2bl_canon']:.2f} | {float(r['nested']['P_GENERALISES']):.2f} | "
                 f"{float(r['independent_within_oracle']['P_GENERALISES']):.2f} | "
                 f"{float(r['nested']['P_primary_PASS_sim']):.2f} / {float(r['P_primary_PASS_exact_independent']):.3f} |")
    ga = pw.get("gate_alignment") or []
    if ga:
        L += ["", "| true held-out-word accuracy (dev and confirmation) | seeds | P(dev gate PROCEED) | P(GENERALISES) H_words "
                  "| P(GENERALISES) HS_words |", "|---:|---|---:|---:|---:|"]
        for g in ga:
            L.append(f"| {g['p_true']:.2f} | {g['seeds']} | {float(g['P_PROCEED']):.2f} | "
                     f"{float(g['P_GENERALISES_H_words_nested']):.2f} | {float(g['P_GENERALISES_HS_words_nested']):.2f} |")
    L += ["", f"Assumption source: {pw.get('assumption_source')}. Twins assumed at {pw.get('twin_accuracy_assumed')} "
              f"(no-op rows' twin ID_void, posted rows' ID_posted, mixed cells' ID_posted / S_words). "
              f"{pw.get('plan_note', '')}"]
    return "\n".join(L)


def _plan_table(gp: dict) -> str:
    L = ["| stage | what | generations expected (ceiling) | h planning rate | h observed rate | h ceiling, planning rate |",
         "|---|---|---:|---:|---:|---:|"]
    for r in gp.get("stages") or []:
        L.append(f"| {r['stage']} | {r['what']} | {r['generations_expected']:,} ({r['generations_ceiling']:,}) | "
                 f"{r['h_planning']:.2f} | {r['h_observed']:.2f} | {r['h_ceiling_planning']:.2f} |")
    L += ["", "   Schedules simulated with the runner's own start rule (t2bl_confirm_run.start_estimate_s):"]
    for sc in gp.get("schedules") or []:
        if isinstance(sc.get("turns"), list):
            L.append(f"   - {sc['name']}: " + "; ".join(f"turn {i + 1}: {', '.join(t['stages'])} ({t['projected_h']} h)"
                                                        for i, t in enumerate(sc["turns"]))
                     + f"; {sc['gpu_h']} GPU-h; longest turn {sc['max_turn_h']} h")
        else:
            L.append(f"   - {sc['name']}: {sc.get('turns')}")
    L.append("   - part (c): one more RCWM turn under its own lease (about 0.5-1.1 GPU-h).")
    return "\n".join(L)


def render_draft(template: Path | None = None, power_rec: dict | None = None, plan_rec: dict | None = None) -> str:
    t = Path(template or DRAFT_TEMPLATE).read_text(encoding="utf-8")
    src_t, src_d = C.read_json(C.TRAIN_SOURCE) or {}, C.read_json(C.DEV_SOURCE) or {}
    fill = {"TRAIN_SHA": src_t.get("file_sha256", "?"), "DEV_SHA": src_d.get("file_sha256", "?"),
            "RULE_MAP": RULE_MAP, "RULE_P3": RULE_P3, "RULE_PLAN": RULE_PLAN, "RULE_RET": RULE_RET,
            "FACTOR_LABELS": "; ".join(f"{f}: '{v['label']}'" for f, v in FACTORS.items()),
            "POWER_TABLE": _power_table(power_rec or {}), "PLAN_TABLE": _plan_table(plan_rec or {}),
            "TEMPLATES": "\n".join(f"- `{k}`: {v}" for k, v in T.items()),
            "NOT_EVALUABLE": "\n".join([f"- {v}" for v in NOT_EVALUABLE_PART.values()] +
                                       [f"- reason `{k}`: {v}" for k, v in NOT_EVALUABLE_REASONS.items()]),
            "RULES_SENTENCES": "\n".join(f"- {'unchanged' if k else 'changed'}: {v}" for k, v in RULES_SENTENCE.items()),
            "LIMITS": "\n".join(f"- {x}" for x in LIMITS)}
    for k, v in fill.items():
        t = t.replace(f"@@{k}@@", v)
    if "@@" in t:
        raise AnalysisError("unfilled placeholder in the draft template")
    miss = registered_strings_missing(t)
    if miss:
        raise AnalysisError(f"{len(miss)} registered strings missing from the rendered draft")
    return t


# ================================================================================ registered analysis
def terminal_state(run: Path) -> dict:
    """DONE.json (every part) or a STOPPED.json naming its verdict-eligible parts; never both."""
    done, stop = C.read_json(run / "DONE.json"), C.read_json(run / "STOPPED.json")
    need(not (done and stop), "both DONE.json and STOPPED.json exist")
    need(bool(done) or bool(stop), "the PREREG-5 job has no terminal record (DONE.json / STOPPED.json)")
    if done:
        return {"kind": "done", "parts": ["a", "b"], "provenance_sha256": done.get("provenance_sha256")}
    return {"kind": "stopped", "parts": list(stop.get("eligible_parts") or []), "reason": stop.get("reason"),
            "provenance_sha256": stop.get("provenance_sha256")}


def instrument_errors(prov: dict, run: Path, fz: dict, root: Path, parts: list, registered: bool = True) -> list:
    """Every provenance check the verdict rests on (the runner enforced them; the analysis re-checks independently)."""
    import t2bl_confirm_run as CR
    errs = []
    st = CR.stage_states(prov, run)
    for p in parts:
        for s in CR.PART_STAGES[p]:
            if st.get(s) != "done":
                errs.append(f"part ({p}) needs stage {s}, which is {st.get(s)}")
    for name in ("anchor", "ident", "bi"):
        if (C.read_json(run / CR.FILES["A0"][name]) or {}).get("pass") is not True:
            errs.append(f"A0 {name} did not pass")
    a0_turn = (prov.get("stages") or {}).get("A0", {}).get("turn")
    allowed = CR.allowed_hashes(fz, root)
    want_ad = {"t2s0": {f: fz["inputs"][f"{CR.T2S0_REL}/{f}"] for f in CR.ADAPTER_FILES},
               **{f"t2bl{s}": {f: fz["inputs"][f"{CR.T2BL_RELS[s]}/{f}"] for f in CR.ADAPTER_FILES} for s in CR.SEEDS}}
    for i, t in enumerate(prov.get("turns") or []):
        ran = [s for s in t.get("stages_run") or [] if s != "A0"]
        if ran and i != a0_turn and not (t.get("a0_recheck") and all((t["a0_recheck"].get(k) or {}).get("pass")
                                                                     for k in ("anchor", "ident", "bi"))):
            errs.append(f"turn {i} ran {ran} without a passing A0 re-check")
        if t.get("stages_run"):
            rt = t.get("runtime") or {}
            if CR.runtime_problem(rt, t.get("versions") or {}):
                errs.append(f"turn {i}: runtime differs from the registered one")
            if not str(t.get("fingerprint", "")).startswith(CR.FP["sparse"]):
                errs.append(f"turn {i}: Titan-Sparse fingerprint differs")
            if t.get("adapter_sha256_at_load") != want_ad:
                errs.append(f"turn {i}: adapter hashes at load differ from the freeze")
            if t.get("bwm_planner_sha256") != CR.BWM_PLANNER_SHA:
                errs.append(f"turn {i}: BWM planner sha differs")
            for rel, h in (t.get("code_sha256") or {}).items():
                if rel in allowed and h not in allowed[rel]:
                    errs.append(f"turn {i}: {rel} differs from the freeze")
            if registered and t.get("device") != "cuda":
                errs.append(f"turn {i}: device {t.get('device')!r} is not cuda")
    if "b" in parts:
        cpu = prov.get("cpu_runs") or []
        if not cpu or cpu[-1].get("lease_held") or cpu[-1].get("cuda_initialized_before") or \
                cpu[-1].get("cuda_initialized_after"):
            errs.append("C1 was not recorded as a CUDA-free CPU run outside the lease")
        c1_turns = (prov.get("stages") or {}).get("C1", {})
        if c1_turns.get("after_a0") is not True:
            errs.append("C1 is not recorded after A0")
    return errs


def run_registered(root: Path = ROOT, run: Path | None = None, registered: bool = True) -> dict:
    import lbeam_problems as P
    import t2bl_confirm_run as CR
    root = Path(root)
    run = Path(run or (root / CR.OUT_REL))
    ok, errs, fz = CR.verify_freeze(root)
    need(ok, "the PREREG-5 freeze does not verify: " + "; ".join(errs[:5]))
    out_dir = run / "analysis"
    need(not (out_dir / "PREREG_5_RESULTS.json").exists(), "PREREG_5_RESULTS.json exists (create-only; run once)")
    term = terminal_state(run)
    prov = C.read_json(run / "provenance.json")
    need(prov is not None and term["provenance_sha256"] == C.sha256(run / "provenance.json"),
         "provenance.json missing or changed after the terminal record")
    parts = term["parts"]
    ierr = instrument_errors(prov, run, fz, root, parts, registered)
    need(not ierr, "instrument checks failed: " + "; ".join(ierr[:6]))
    ne = {p: ("stopped" if term["kind"] == "stopped" else "missing") for p in ("a", "b") if p not in parts}
    A = D = None
    if "a" in parts:
        rows = C.load_jsonl(root / CR.MAP_BANK_REL)
        need(C.sha256(root / CR.MAP_BANK_REL) == fz["sealed"]["map_bank"]["file_sha256"], "map bank differs from its seal")
        texts = {}
        for side in MAP_SIDES:
            recs = C.load_jsonl(run / CR.FILES["M1"][f"map:{side}"])
            need([x["id"] for x in recs] == [r["id"] for r in rows], f"map {side}: ids not in bank order")
            texts[side] = [x["text"] for x in recs]
        sc = score_all(rows, texts)
        A = decide_map(rows, sc, classes_all(rows, texts))
    if "b" in parts:
        insts = P.load_set(root / CR.PLAN_DRAW_REL)
        need(P.set_sha(insts) == fz["sealed"]["plan_draw"]["file_sha256"], "planning draw differs from its seal")
        succ = {}
        for key, rel in {**CR.EPISODE_FILES, **CR.REPORT_EPISODE_FILES}.items():
            eps = C.load_jsonl(run / rel)
            need([e["id"] for e in eps] == [i["id"] for i in insts], f"{key}: episode ids not in draw order")
            succ[key] = np.array([P.execute(i, [tuple(a) for a in e.get("plan") or []])["success"]
                                  for i, e in zip(insts, eps)])
        D = decide_plan({k: succ[k] for k in CR.EPISODE_FILES})
        D["counts"]["IGNORE_STATUS"] = int(succ["IGNORE_STATUS"].sum())
        D["success"]["IGNORE_STATUS"] = float(succ["IGNORE_STATUS"].mean())
        eps = {k: C.load_jsonl(run / rel) for k, rel in {**CR.EPISODE_FILES, **CR.REPORT_EPISODE_FILES}.items()}
        caches, accts = {}, {}
        for st, arms in CR.STAGE_ARMS.items():
            for arm in arms:
                lines = (run / CR.FILES[st][f"cache:{arm}"]).read_text(encoding="utf-8").splitlines()
                caches[arm] = [json.loads(x) for x in lines[1:] if x.strip()]
                accts[arm] = ((C.read_json(run / CR.FILES[st]["acct"]) or {}).get("configs") or {}).get(arm, {}).get("guard")
        D["report"] = plan_report(insts, eps, caches)
        D["report"]["guard"] = accts
    ptr = C.read_json(root / CR.RETENTION_RESULT_POINTER)
    if ptr and ptr.get("status") == "analysed":
        need(C.sha256(root / ptr["results"]) == ptr.get("results_sha256"), "the retention result changed after binding")
        R = read_retention(C.read_json(root / ptr["results"]))
    elif ptr and ptr.get("status") == "stopped":
        R = read_retention(None, f"the retention run stopped: {ptr.get('reason', '')[:120]}")
    else:
        R = read_retention(None, "the retention run was not completed and bound")
    rules = fz.get("rules") or {}
    W = wording(A, D, R, {"not_evaluable": ne, "rules": {"unchanged": rules.get("unchanged"),
                                                          "when": rules.get("gate_frozen_local")}})
    res = {"record": "PREREG_5_RESULTS", "rules": {"map": RULE_MAP, "p3": RULE_P3, "plan": RULE_PLAN, "ret": RULE_RET},
           "terminal": term, "parts_evaluated": parts, "map": A, "plan": D, "retention": R, "wording": W,
           "limits": list(LIMITS), "freeze_prereg_sha256": fz.get("prereg_sha256"), "registered": registered,
           "code_sha256": C.code_hashes(("scripts/t2bl_confirm_analysis.py", "scripts/t2bl_common.py",
                                         "scripts/t2bl_confirm_run.py")),
           "analysed_local": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    js, md = json.dumps(clean(res), indent=1) + "\n", render_md(res)   # both rendered before anything is written
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "PREREG_5_RESULTS.md").exists():      # an orphan from a crash between the two writes (no JSON yet)
        os.replace(out_dir / "PREREG_5_RESULTS.md",
                   out_dir / f"PREREG_5_RESULTS.md.orphan.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    C.write_create(out_dir / "PREREG_5_RESULTS.md", md, stale_partial="aside")
    C.write_create(out_dir / "PREREG_5_RESULTS.json", js, stale_partial="aside")   # last: its presence means 'done'
    return res


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def render_md(res: dict) -> str:
    W = res["wording"]
    L = ["# PREREG-5 results", "", f"**{W.get('headline_a')}**", "", f"**{W.get('headline_b')}**", "",
         f"**{W['headline_c']}**", ""]
    if W.get("rules_sentence"):
        L += [W["rules_sentence"], ""]
    L += ["## Part (a): factor verdicts", ""]
    for f, s in W["factor_sentences"].items():
        L.append(f"- {s}")
    L += ["", W.get("id_sentence", ""), "", W.get("p3_sentence", ""), "", "## Part (b)", "", W.get("plan_sentence", ""),
          "", W.get("plan_canon_line", ""), "", "## Part (c)", "", W["ret_sentence"], "", "## Limits (always stated)", ""]
    L += [f"- {x}" for x in res["limits"]]
    A = res.get("map")
    if A:
        L += ["", "## Cells (part (a))", "",
              "| cell | OFF | T2 s0 | s0 | s1 | s2 | bar / gain (s0, s1, s2) | NI: seed-mean difference (lower bound) |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
        for ck, d in A["cells"].items():
            L.append(f"| {ck.replace(chr(124), chr(32))} | {d['off']['acc']:.2f} | {d['t2s0']['acc']:.2f} | " + " | ".join(
                f"{d['seeds'][s]['acc']:.2f}" for s in T2BL) + " | " + ", ".join(
                "/".join("y" if d["seeds"][s][k] else "n" for k in ("bar", "gain")) for s in T2BL) +
                f" | {d['ni']['diff']:+.3f} ({d['ni']['lower_95']:+.3f}) {'pass' if d['ni']['pass'] else 'FAIL'} |")
        L += ["", "## Variants (descriptive)", ""]
        for st, vs in A["variants"].items():
            L.append(f"- **{st}**: " + "; ".join(f"{v}: " + ", ".join(f"{s} {x[s]['acc']:.2f}" for s in MAP_SIDES)
                                                 for v, x in vs.items()))
        L += ["", "## Trained-variant controls (accuracy; T2b-ledger minus T2 s0)", ""]
        for st, d in A["controls"].items():
            L.append(f"- {st}: T2 s0 {d['t2s0']['acc']:.2f}; " + ", ".join(
                f"{s.replace('t2bl', 's')} {d[s]['acc']:.2f} ({d[s]['minus_t2s0']:+.2f})" for s in T2BL))
        L += ["", "## Report-only cells", ""]
        for ck, d in A["report_only"].items():
            L.append(f"- {ck}: " + ", ".join(f"{s} {d[s]['acc']:.2f}" for s in MAP_SIDES))
        L += ["", "## Word-cluster bootstrap intervals (report-only; 20 held-out words, 5 rows each)", ""]
        for st, d in (A.get("word_clusters_report_only") or {}).items():
            L.append(f"- {st}: " + "; ".join(f"{s} {d[s]['acc']:.2f} [{d[s]['ci95'][0]:.2f}, {d[s]['ci95'][1]:.2f}]"
                                             for s in MAP_SIDES))
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--from-gate", default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--render-draft", action="store_true", help="PREREG_5_DRAFT.md from its template (refused once frozen)")
    a = ap.parse_args()
    try:
        if a.render_draft:
            need(not (C.ROOT / "PREREG_5_FREEZE.json").exists(), "PREREG-5 is frozen: the draft is not re-rendered")
            pw = C.read_json(C.DOCS / "P5_POWER_DEV.json") or C.read_json(C.DOCS / "P5_POWER.json") or {}
            text = render_draft(power_rec=pw, plan_rec=C.read_json(C.DOCS / "P5_GEN_PLAN.json") or {})
            with open(DRAFT_PATH, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print("wrote", DRAFT_PATH, C.sha256(DRAFT_PATH))
            return 0
        if a.dryrun:
            res = dryrun()
            for c in res["cases"]:
                print(("PASS " if c["ok"] else "FAIL ") + c["case"])
            if a.record:
                print("wrote", C.write_doc("P5_DRYRUN.json", json.dumps(clean(res), indent=1) + "\n"))
                md = ["# PREREG-5 dry run (synthetic outcomes; every wording branch)", ""]
                for c in res["cases"]:
                    md.append(f"## {'PASS' if c['ok'] else 'FAIL'}: {c['case']}")
                    for k in ("headline_a", "headline_b", "headline_c"):
                        if c.get(k):
                            md.append(f"- {c[k]}")
                    for k, v in (c.get("sentences") or {}).items():
                        if isinstance(v, dict):
                            md += [f"  - {x}" for x in v.values()]
                        else:
                            md.append(f"  - {v}")
                    md.append("")
                print("wrote", C.write_doc("P5_DRYRUN.md", "\n".join(md) + "\n"))
            return 0 if res["ok"] else 1
        if a.power:
            res = power(Path(a.from_gate) if a.from_gate else None)
            print(json.dumps(clean(res), indent=1)[:4000])
            if a.record:
                name = "P5_POWER.json" if not a.from_gate else "P5_POWER_DEV.json"
                print("wrote", C.write_doc(name, json.dumps(clean(res), indent=1) + "\n"))
            return 0
        res = run_registered()
        print(res["wording"]["headline_a"])
        print(res["wording"]["headline_b"])
        print(res["wording"]["headline_c"])
        return 0
    except AnalysisError as e:
        print(f"[prereg5-analysis] {e}", flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
