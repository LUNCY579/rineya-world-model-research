"""T2b-register screen (RCWM roadmap item 3; joint roadmap item 5b with BWM-001): shared definitions. Torch-free.

Question. Do intermediates in the answer ("trace": compare, then compute, then answer) teach frozen Titan-Sparse the
register rules that bare-answer rows did not? T2's evidence: mix 0.022 -> 0.022, conditional branch choice at chance
(0.51) while the in-branch arithmetic is right, OFF outputs the unreduced sum. Decision point 3 asks whether any gain
is supervision or only extra decode positions; the answer-free "filler" arm answers that.

Environment semantics (rcwmt/env.py, frozen; quoted here, never changed). State {"X","Y","Z"}: integers modulo 97.
One instruction per step (env.step):
  swap         reg[r], reg[o] = reg[o], reg[r]
  add          reg[r] = (reg[r] + v) % 97
  mix          reg[r] = (reg[r] + 2 * reg[o] + v) % 97
  conditional  reg[r] = (reg[r] + (v if reg[o] > threshold else -v)) % 97
The conditional predicate is STRICT: reg[o] == threshold subtracts (the frame text: "adds value if other is greater
than threshold, otherwise subtracts value"). Python's % is non-negative, so an underflow wraps up (3 - 5 -> 95).
'wrap' (as scripts/make_t2_data.stratum): the raw value of the updated register (r + v; r + 2 o + v; r +/- v on the
chosen branch) lies outside [0, 96].

Joint trace format (BWM D-52 + the binding amendments 1-5 and 4b; reference implementation, read-only:
Rineya_BWM_001_Research_Bundle/native_v1/bwmn/traces.py). One JSON object, keys in LITERAL order cmp -> calc -> after,
serialized with an explicit order (never sort_keys, never a sorting canonicaliser):
  cmp    the branch predicate's INPUTS and its boolean. conditional: {"x": reg[other], "thr": threshold,
         "ok": reg[other] > threshold} (env's strict '>'; the joint proposal's '>=' is replaced by the env semantics).
         swap / add / mix have no branch predicate: null (as BWM's 'finish').
  calc   the chosen branch's arithmetic: {"raw": the unreduced value of the updated register, "mod": raw % 97}
         (add: r + v; mix: r + 2 o + v; conditional: r + v if ok else r - v). swap: null (no arithmetic).
  after  the family's EXISTING answer: the register state JSON exactly as env.dumps writes it ({"X":..,"Y":..,"Z":..}).
Arms on the SAME rows and the SAME prompt:
  plain  the existing answer only (= after)
  trace  {"cmp":..,"calc":..,"after":..}
  filler {"fill":" . . .","after":..}: AMENDMENT 4b (agreed with BWM, their D-55; definition made exact in revision 4).
         The fill length is a PER-KIND CONSTANT (FILL_UNITS): the smallest run of the one-token unit " ." such that the
         filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind over train + dev
         (one-step rows and the chains' gold steps), whatever its 'after' (min_constant_fill: the shortest filler
         target of the kind is compared with the longest trace target of the kind; targets do not depend on the prompt
         surface, so this covers every surface variant). Every filler target of a kind is therefore identical up to
         'after', and the fill length carries NO information about the answer (the kind is in the prompt). The
         residual (filler tokens minus that row's trace tokens, >= 0) is a descriptive covariate. Revision 3's code
         compared each row's filler with THAT row's trace (a per-row maximum, swap 5 / add 14 / mix 14 / conditional
         26: FILL_UNITS_REV3); revision 4 implements the constant definition stated in the rule. The per-row token
         match of revisions 1-2 (amendment 4) leaked the answer through the fill length (wrap 0.957 / 0.970, branch
         0.57-0.85; docs/t2b/register/DESIGN_rev2.md) and is withdrawn.
Parsing. parse_trace rejects any other key order, extra or missing keys and duplicate keys; parse_filler likewise
(["fill","after"]). The decision parser reads ONLY 'after': after_text() re-serializes it (sorted, compact; BWM's
convention) for rcwmt.env.parse_state(lenient=True), the parser of every existing chain/regret pipeline. A malformed
trace or filler answer has no 'after' (= a wrong answer). cmp_ok / calc_ok are scored separately (mechanism).

Surface diversity (revision 3: decision point 1 FIRED, PREREG-3 STATUS: TRIGGER; operator decisions C1-C3). The
screen's PRIMARY (and only) training build is 'sd': the SAME canonical rows (never re-drawn) whose prompts are rendered
with a seeded per-row surface drawn from TRAIN subsets: a rule-text paraphrase (SD_PARAPHRASES[0..3]), a field-name
variant per field (SD_FIELDS[f][0..3]) and a JSON key order per object type (top level, state, event) drawn from
TRAIN_ORDERS. HELD OUT (never in training): paraphrase 4, field-name variant 4 of every field, and per object type the
fully reversed canonical order plus 2 fixed seeded permutations (HELDOUT_ORDERS; the 2-key top-level object has only the
reversed order to hold out). The register one-step prompt contains no lists (no list-order factor). Targets are
unchanged (literal cmp -> calc -> after). The held-out-surface (HS) rendering of the primary strata and of the
identification strata (cond_sametens, cond_oor_edge, cond_adv) combines paraphrase 4 + names 4 + a held-out key order per object type, rotating
over the held-out orders by row (heldout_surface). The held-out STATE orders are exactly the three odd permutations of
X/Y/Z (training sees only the three cyclic rotations): a harder-than-random hold-out (see DESIGN.md).

Identification (revision 4). 'cond_sametens' (G1, decided alone; not pooled into the primary 'conditional'): branch-
balanced minimal pairs with tens(other) == tens(threshold), |other - threshold| 2..9, out-of-range |value| 12..40 and no
wrap on either branch. Every primary conditional row of this bank is decided correctly by the tens-digit shortcut 'add
iff tens(other) > tens(threshold) or other == threshold + 1' and by its mirror 'add iff tens(other) >= tens(threshold)
and other != threshold'; on cond_sametens both score 1/2.
Identification (revision 5). (1) An absolute, arithmetic-independent clause: the trace side's PREDICATE ACCURACY
(score_text 'pred_ok': the strict joint parser accepts the answer, cmp is an object with exactly the keys x, thr, ok in
that literal order, and ok is a JSON boolean equal to the env's predicate; x and thr are not scored) must be >= 0.80
(PRED_MIN) on each identification stratum, in-surface and on the held-out surface. It does not depend on the plain arm,
so a plain arm that is weak on the out-of-range arithmetic cannot let a branch shortcut through. (2) A second
identification stratum 'cond_oor_edge': branch-balanced minimal QUADRUPLES at other - threshold = -1, 0 (ok false) /
+1, +2 (ok true), out-of-range |value| 12..40, no wrap on either branch. With cond_oor_val's (0, +1) pairs, G1 then sees
rows at other = threshold - 1, threshold and threshold + 1, and every off-by-one variant of the comparison (adding or
removing other == threshold - 1 / threshold / threshold + 1: '>=', '> threshold + 1', '>= threshold - 1', 'or other ==
threshold - 1', ...) is contradicted on at least 1/4 of cond_oor_edge, so its predicate accuracy there is at most 3/4.
Identification (revision 6). An explicit EXTENDED SHORTCUT FAMILY H (t2br_shortcuts: the enumerated digit / off-by-one
family + lexicographic, digit-sum, units-only, tens-only and magnitude-only comparisons with sign / length handling and
the isolated single-distance exceptions at |other - threshold| 2..4) and a third identification stratum 'cond_adv' (100
rows drawn by seeded rejection sampling to fixed quotas of 15 design types, 13 signature classes of H, 50 ok true / 50
ok false), gated
by the predicate clause alone (in-surface and held out). G1 excludes the members of H documented by the enumeration
over H (all but the isolated exception at other = threshold - 4). A report-only shortcut-agreement analysis compares
the trace's stated cmp.ok (stated_ok) with every member of H.
Revision 7 (wording only, DECISIONS D-43: revision 6's design approved as is). No gate, threshold, stratum, data row,
stage or computation changed. The LIMITS (ALWAYS_QUOTED_LIMITS and the rule's LIMITS sentence) now say that G1
excludes H except the exception at -4, that heuristics OUTSIDE H are NOT excluded (the natural ones the revision-6
independent check found passing G1 are named) and that cond_adv catches its targets only at narrow margins; the
analysis' OVERALL headline carries the report-only SHORTCUT_CONSISTENT flag and a pointer to the LIMITS whenever the
flag is set.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
from collections import OrderedDict  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rcwmt import env  # noqa: E402

FAM = "register_machine"
MOD = env.MOD
REGS = env.REGISTERS
ARMS = ("plain", "trace", "filler")
SIDES = ("off", "plain", "trace", "filler")          # off = adapter disabled (Titan-Sparse); scored with the plain parser
SIDE_ARM = {"off": "plain", "plain": "plain", "trace": "trace", "filler": "filler"}
FILL_UNIT = " ."
H = 4                                                # chain horizon
K = 8                                                # candidates per chain parent (the pilot / PREREG-2 decision setup)
KINDS = ("swap", "add", "mix", "conditional")
EVENT_KEYS = ("kind", "other", "register", "threshold", "tick", "value")   # every register event (env.gen_event)

DATA = ROOT / "data"
DOCS = ROOT / "docs" / "t2b" / "register"
TRAIN_STEM = "t2br_train"
TRAIN_VARIANT = "sd"                                 # C1: the primary (and only run) build
DEV = DATA / "t2br_dev.jsonl"
DEV_HS = DATA / "t2br_dev_hs.jsonl"                  # the held-out-surface rendering of the primary dev rows (C3)
CHAINS = DATA / "t2br_chains.jsonl"
DEV_SOURCE = DATA / "t2br_dev.SOURCE.json"
TOKENIZER_JSON = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                      r"Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")   # == rcwmt.runtime.TOKENIZER_DIR


def train_path(arm: str, variant: str = TRAIN_VARIANT) -> Path:
    """'base' = the canonical rendering (the rows' source of truth; not trained, C1); 'sd' = the surface-diversity
    rendering of the same rows (the screen's training data)."""
    return DATA / (f"{TRAIN_STEM}_{arm}.jsonl" if variant == "base" else f"{TRAIN_STEM}_{variant}_{arm}.jsonl")


def train_source(variant: str = "base") -> Path:
    return DATA / (f"{TRAIN_STEM}.SOURCE.json" if variant == "base" else f"{TRAIN_STEM}_{variant}.SOURCE.json")


# ---------------------------------------------------------------------------------------------- value ranges
IN_V = (-8, 11)                  # recorded CANDIDATE_VIEWS register 'value' range (asserted against env.load_marginals)
OOR_V_ABS = (12, 40)             # out-of-range values: |v| in 12..40, both signs
THR_IN = (0, 96)                 # recorded threshold range (= the register range)
OOR_THR = ((-40, -1), (97, 136))  # out-of-range thresholds, outside the register range: ok always true / always false
# distance |other - threshold| of the conditional rows. Training: the 150 core rows at -1 / 0 / +1 and 150 minimal-pair
# twins at 2..15 on the opposite side of the threshold (TWIN_DIST). Dev: cond_mid at 2..15 (in distribution), cond_far
# at 16..96 (beyond every training row's distance: out of range in distance, joint amendment 5).
TWIN_DIST = (2, 15)
FAR_DIST = (16, 96)
SAMETENS_DIST = (2, 9)           # cond_sametens: |other - threshold| with tens(other) == tens(threshold) (revision 4)
EDGE_OFFSETS = (-1, 0, 1, 2)     # cond_oor_edge (revision 5): other - threshold of the 4 members of a quadruple

# ---------------------------------------------------------------------------------------------- dev screen strata
# name -> (group, role, description). group 'in' = inside the training ranges (value, threshold, distance), 'oor' = out.
DEV_STRATA = OrderedDict([
    ("mix", ("in", "noninferiority", "mix, in-range value, 50 wrap / 50 no-wrap")),
    ("wrap", ("in", "reported", "add whose raw sum leaves [0, 96] (40 over / 40 under), in-range value")),
    ("cond_xm1", ("in", "noninferiority", "conditional with other = threshold - 1 (ok false: subtract); triplet member")),
    ("cond_x0", ("in", "noninferiority", "conditional with other = threshold (ok false under the STRICT '>'); triplet")),
    ("cond_xp1", ("in", "noninferiority", "conditional with other = threshold + 1 (ok true: add); triplet member")),
    ("cond_mid", ("in", "noninferiority", "conditional pairs with |other - threshold| 2..15 on both members (the "
                                          "training twins' band), one ok true / one ok false; 25 pairs move the "
                                          "threshold, 25 move the other register; in-range value")),
    ("swap", ("in", "control", "swap of two unequal registers (untouched skill: not in training)")),
    ("add_nowrap", ("in", "control", "add, in-range value != 0, no wrap (untouched skill: not in training)")),
    ("mix_oor", ("oor", "primary", "mix with |value| in 12..40 (outside the trained -8..11), 50 wrap / 50 no-wrap")),
    ("wrap_oor", ("oor", "reported", "add with |value| in 12..40 whose raw sum leaves [0, 96] (40 over / 40 under)")),
    ("cond_oor_val", ("oor", "primary", "conditional boundary pairs sharing the row: threshold = other (ok false) / "
                                        "threshold = other - 1 (ok true), |value| in 12..40")),
    ("cond_oor_thr", ("oor", "primary", "conditional pairs with thresholds outside the register range: -40..-1 (ok "
                                        "true) / 97..136 (ok false); value, registers in range")),
    ("cond_far", ("oor", "primary", "conditional pairs FAR from the threshold: |other - threshold| 16..96 on both "
                                    "members (beyond the training rows' 0..15); in-range value, threshold and "
                                    "registers; each pair shares the event and moves the other register")),
    # revision 4 (appended; drawn after the chains with its own seed, so every earlier row is unchanged)
    ("cond_sametens", ("oor", "identification", "conditional pairs with tens(other) == tens(threshold): the branch is "
                                                "decided by the units digit; |other - threshold| in 2..9 on both "
                                                "members (2..7 realized: a pair lies in one decade), one ok true / one "
                                                "ok false; |value| in 12..40 (outside the trained "
                                                "-8..11), neither branch wraps; in-range registers and thresholds; 25 "
                                                "pairs move the threshold, 25 move the other register")),
    # revision 5 (appended after cond_sametens; drawn after it with its own seed, so every earlier row is unchanged)
    ("cond_oor_edge", ("oor", "identification", "conditional minimal quadruples at the boundary of the strict "
                                                "predicate: other - threshold = -1, 0 (ok false) / +1, +2 (ok true); "
                                                "|value| in 12..40 (outside the trained -8..11), neither branch wraps; "
                                                "in-range registers and thresholds; 10 quadruples move the threshold, "
                                                "10 move the other register")),
    # revision 6 (appended after cond_oor_edge; drawn after it with its own seed, so every earlier row is unchanged)
    ("cond_adv", ("oor", "identification", "conditional rows drawn by seeded rejection sampling over valid conditional "
                                           "rows (other 0..96, threshold -40..136) to the quotas of 15 design types (13 "
                                           "signature classes of the extended shortcut family H, t2br_shortcuts."
                                           "ADV_TYPES: sign-crossing, cross-length and cross-decade cells, cells at "
                                           "other - threshold = -3, -2, +3, +4); 50 ok true / 50 ok false; |value| in "
                                           "12..40 (outside the trained -8..11), neither branch wraps; registers in "
                                           "range")),
])
DEV_N = {"mix": 100, "wrap": 80, "cond_xm1": 60, "cond_x0": 60, "cond_xp1": 60, "cond_mid": 100, "swap": 80,
         "add_nowrap": 80, "mix_oor": 100, "wrap_oor": 80, "cond_oor_val": 100, "cond_oor_thr": 100, "cond_far": 100,
         "cond_sametens": 100, "cond_oor_edge": 80, "cond_adv": 100}
# outcome groups of the pre-stated rule
PRIMARY = OrderedDict([("conditional", ("cond_oor_val", "cond_oor_thr", "cond_far")), ("mix", ("mix_oor",))])
NONINF = OrderedDict([("conditional_in", ("cond_xm1", "cond_x0", "cond_xp1", "cond_mid")), ("mix_in", ("mix",))])
CONTROLS = ("swap", "add_nowrap")
# G1 also decides these strata ALONE on trace - plain (each on both surfaces): the boundary pairs (revision 3), the
# boundary quadruples (revision 5) and the same-tens identification pairs (revision 4). No identification stratum is
# pooled into the primary 'conditional'; cond_adv (revision 6) has no trace - plain clause.
G1_ALONE = ("cond_oor_val", "cond_oor_edge", "cond_sametens")
IDENT_STRATA = ("cond_sametens", "cond_oor_edge", "cond_adv")         # in dev-bank order
# Revision 5: G1's absolute, arithmetic-independent clause: the TRACE side's predicate accuracy (score_text 'pred_ok')
# >= PRED_MIN on each of these strata, in-surface and on the held-out surface (no plain-arm term); revision 6 adds the
# adversarial stratum cond_adv (absolute clause only)
PRED_STRATA = ("cond_oor_edge", "cond_sametens", "cond_adv")
PRED_MIN = Fraction(4, 5)
# C3: the held-out-surface stage renders exactly the strata the held-out part of the rule needs (G1 / G2 = the
# primaries + the identification strata; no in-distribution stratum enters a held-out-surface condition)
HS_STRATA = tuple(s for ss in PRIMARY.values() for s in ss) + IDENT_STRATA
HS_STRATA_REV3 = tuple(s for ss in PRIMARY.values() for s in ss)      # revision 3's held-out-surface bank (400 rows)
HS_STRATA_REV4 = HS_STRATA_REV3 + ("cond_sametens",)                  # revision 4's held-out-surface bank (500 rows)
HS_STRATA_REV5 = HS_STRATA_REV4 + ("cond_oor_edge",)                  # revision 5's held-out-surface bank (580 rows)
CHAIN_STRATA = OrderedDict([("CH_IN", 32), ("CH_X", 16)])   # parents x K=8 candidates x H=4 steps

# ---------------------------------------------------------------------------------------------- amendment 4b filler
# Constant fill units per action kind (revision 4, the definition stated in the rule): the SMALLEST run of " ." such
# that the filler target is at least as long in Titan tokens as the LONGEST trace target of that kind over train + dev
# (one-step rows and the chains' gold steps), whatever its 'after' (i.e. the SHORTEST filler target of the kind is
# compared with the LONGEST trace target of the kind: min_constant_fill). Fixed here (pre-stated in the rule) and
# re-derived from the data by t2br_make_data (build and --verify): the build fails if the data imply any other value.
FILL_UNITS = {"swap": 7, "add": 17, "mix": 17, "conditional": 29}
# Revision 3's constants (SUPERSEDED): its code compared each row's filler target with THAT row's trace target (the
# smallest n with filler(n, after_i) >= trace_i on every row i: a per-row maximum), not with the kind's longest trace.
# Kept only to reconstruct the revision-3 data files byte for byte (t2br_make_data.rev3_reconstruction).
FILL_UNITS_REV3 = {"swap": 5, "add": 14, "mix": 14, "conditional": 26}

# ---------------------------------------------------------------------------------------------- pre-stated go rule
# Bars are exact rationals: every condition is decided on integer counts (t2br_analysis.metric_exact), never on floats.
GAIN_CONDITIONAL = Fraction(1, 4)   # G1: trace - plain on 'conditional', on cond_oor_val alone, on cond_oor_edge alone
#                                     (revision 5) AND on cond_sametens alone (revision 4), each in-surface AND HS; G1
#                                     also needs the trace predicate accuracy >= PRED_MIN on each PRED_STRATA (revision 5;
#                                     cond_adv added in revision 6)
GAIN_MIX = Fraction(3, 10)          # G2: trace - plain on 'mix', in-surface AND HS
FILLER_STOP = Fraction(1, 10)       # G3 bar per in-surface primary; decision point 3: trace - filler < 0.10 on BOTH
CONTROL_LOSS = Fraction(1, 20)      # G4: trace ON - OFF >= -0.05 on swap and add_nowrap
NONINF_MARGIN = Fraction(1, 20)     # G5: trace - plain >= -0.05 (conditional_in, mix_in)
FORMAT_VALID = Fraction(19, 20)     # validity: in-surface format_ok >= 0.95 on each primary (trace side; filler side)
GO_RULE_REVISION = 7
DP3_LABEL = ("STOP (decision point 3): trace ~ filler, not supervision of the intermediates (an answer-free filler of "
             "constant length per action kind, amendment 4b, does as well: the gain is attributable to extra decode "
             "positions or the answer format)")
DP3_SUB = {"trace_format": "trace format failed (trace format_ok < 0.95 on a primary)",
           "filler_invalid": "filler control invalid (filler format_ok < 0.95 on a primary)",
           "neither": "neither format helped (trace ~ filler ~ plain)"}
INCONCLUSIVE_FILLER = "INCONCLUSIVE: filler control invalid (G1-G5 hold, but filler format_ok < 0.95 on a primary)"
GO_RULE_TEXT = (
    "PRE-STATED SCREEN GO RULE, REVISION 7 (T2b-register; supersedes revisions 1-6, stated before any output exists; "
    "seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no "
    "floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% "
    "cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar "
    "is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 "
    "TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names "
    "(variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top "
    "level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully "
    "reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its "
    "reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp "
    "-> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed "
    "trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim "
    "one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object "
    "type, rotating over the held-out orders by row, on the primary strata and on the three identification strata. "
    "PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far "
    "pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond "
    "every training row's 0..15); 'cond_oor_val' = the boundary pairs alone (other = threshold: ok false; other = "
    "threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION strata (G1 only; none is pooled into 'conditional'): "
    "'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so the branch is decided "
    "by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie in one decade), "
    "|value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it separates the "
    "comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add iff tens(other) "
    "> tens(threshold) or other == threshold + 1' scores 1/2 on it); 'cond_oor_edge' (revision 5) = 20 "
    "branch-balanced minimal quadruples at the boundary of the env's strict 'other > threshold': other - threshold = "
    "-1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 (out of range), in-range registers and thresholds, "
    "neither branch wraps, 10 quadruples moving the threshold and 10 the other register; with cond_oor_val it gives "
    "G1 rows at other = threshold - 1, threshold and threshold + 1, so every off-by-one variant of the comparison "
    "(one that differs from it only at other = threshold - 1, threshold or threshold + 1, e.g. '>=', '> threshold + "
    "1', '>= threshold - 1', 'or other == threshold - 1') is contradicted on at least 1/4 of its rows; 'cond_adv' "
    "(revision 6) = 100 rows, 50 ok true and 50 ok false, drawn by seeded rejection sampling over valid conditional "
    "rows (other 0..96, threshold -40..136, |value| 12..40, neither branch wraps) to fixed quotas of 15 design types: "
    "13 signature classes of the extended shortcut family H (sets of (other, threshold) cells on which exactly the "
    "same members of H are wrong), one of them split by digit lengths, covering sign-crossing cells, cross-length "
    "cells (1 vs 2, 2 vs 1, 1 vs 3 and 2 vs 3 digits), cross-decade cells whose units comparison disagrees with the "
    "tens comparison and cells at other - threshold = -3, -2, +3, +4; H (scripts/t2br_shortcuts.py, DESIGN.md section "
    "15; 3,519 predicates) = the enumerated digit and off-by-one family with its modifiers at other = threshold - 1, "
    "threshold and threshold + 1 (3,483 predicates, the branch-blind constants included) + lexicographic string "
    "comparison with and without sign handling, digit-sum, units-only and tens-only comparisons with sign- and "
    "length-aware variants, magnitude-only |other| > |threshold| and the isolated single-distance exceptions at "
    "|other - threshold| = 2..4; on cond_adv every member of H that differs from the comparison somewhere on that "
    "domain and passes revision 5's G1 against a plain arm at 0 is wrong on at least 33 rows, except the isolated "
    "distance exceptions, of which those at other - threshold = -3, -2, +3 and +4 are wrong on exactly 21 (the "
    "operator's target, every member of H wrong on at least 40 rows, is infeasible on any 100-row branch-balanced "
    "stratum: five members of H have pairwise-disjoint error sets). PREDICATE ACCURACY (trace side; "
    "arithmetic-independent): the share of a stratum's rows whose trace answer the strict joint parser accepts, whose "
    "cmp is an object with exactly the keys x, thr, ok in that literal order, and whose ok is a JSON boolean equal to "
    "the env's predicate (other > threshold) for that row; x and thr are not scored; a malformed trace or a wrong key "
    "order counts as wrong. FILLER (amendment 4b): the fill length is a per-kind constant: the smallest run of ' .' "
    "units such that the filler target is at least as long (Titan tokens) as the LONGEST trace target of that kind "
    "over train + dev (one-step rows and the chains' gold steps), whatever its 'after' (swap 7, add 17, mix 17, "
    "conditional 29 units); every filler target of a kind is identical up to 'after', so its length carries no "
    "information about the answer. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace "
    "(filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): "
    "if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not supervision of the "
    "intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as well: the gain is "
    "attributable to extra decode positions or the answer format)', with the sub-label 'trace format failed' if the "
    "trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format "
    "helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace - plain >= "
    "+0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on cond_sametens alone, AND the trace "
    "side's predicate accuracy >= 0.80 on cond_oor_edge alone, on cond_sametens alone AND on cond_adv alone (cond_adv "
    "has no trace - plain clause), each both in-surface and on the held-out surface (the predicate clauses have no "
    "plain-arm term, so a plain arm that is weak on the out-of-range arithmetic cannot let through a member of H that "
    "they catch; G1 excludes the members of the enumerated family H that differ from the comparison, except the "
    "isolated exception 'other > threshold or other == threshold - 4', which passes every G1 part, as documented by "
    "the enumeration over H against three plain baselines (DRYRUN.json shortcut_enumeration); it does NOT exclude "
    "heuristics outside H, and natural ones pass every G1 part (LIMITS)); (G2) trace - plain >= +0.30 on mix, both "
    "in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface "
    "primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) "
    "in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, "
    "cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the "
    "filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition "
    "and its failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the "
    "surface audit (held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler "
    "residual covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split "
    "at its median; the shortcut-agreement analysis (the trace side's cmp.ok against the env's predicate and against "
    "every member of H over all conditional rows of the dev bank, in each rendering: 'SHORTCUT_CONSISTENT' if a "
    "member of H that differs from the predicate on those rows agrees with the trace's cmp.ok at least as well as the "
    "predicate does; it never changes the verdict; it is quoted with every statement of the verdict, so with any GO, "
    "and whenever it is set the OVERALL headline carries it beside the verdict, with a pointer to the LIMITS); "
    "per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget "
    "choice, not a capacity limit). LIMITS, quoted with every statement of the verdict "
    "(t2br_common.ALWAYS_QUOTED_LIMITS; DESIGN.md section 14): (1) G1 excludes the members of the enumerated family H "
    "that differ from the comparison EXCEPT the isolated exception 'other > threshold or other == threshold - 4', "
    "which passes every G1 part (on any 100-row branch-balanced stratum one of the exceptions at other - threshold = "
    "-4, -3, -2 passes the 0.80 bar); (2) heuristics OUTSIDE H are NOT excluded, and natural ones pass every G1 part "
    "(the revision-6 independent check, confirmed by verdict() on exact counts): isolated single-distance exceptions "
    "at |other - threshold| >= 5 (H has them only at 2..4) and flipped distance bands (e.g. 5..9); "
    "three-digit-threshold slips (the hundreds digit dropped, only the first two digits read, a threshold above 96 "
    "reduced modulo 97, string comparison only against three-digit thresholds, always adding when the threshold is "
    "above 96); a register of 0 never counted above a negative threshold; the sign of a threshold at or below -10 "
    "dropped; a circular comparison ((other - threshold) mod 97 in 1..48); string comparison only for one-digit "
    "thresholds or one-digit registers; guessing on the far rows (|other - threshold| > 15) by string comparison or a "
    "prior (other > 48, threshold < 48), and, against a plain arm well below chance on the pooled conditional (the "
    "dry run's coin-flip plain at 5/12, or a weak plain arm), also by always adding, always subtracting or the units "
    "digit; any rule that departs from the comparison only on cells that no G1 stratum contains (e.g. in-range "
    "cross-decade cells at |other - threshold| 5..15, decided by the units digit or with the branch flipped); the "
    "report-only shortcut-agreement flag compares the members of H only, so its absence does not rule these out; (3) "
    "cond_adv catches its targets only at narrow margins: the exceptions at other - threshold = -3, -2, +3, +4 are "
    "wrong on exactly 21 of its 100 rows (79/100, one row below the bar) and the signed string, magnitude, and "
    "sign+length-aware digit-sum and units comparisons on exactly 33 (67/100), so a trace that follows a caught "
    "member on only part of its rows can pass (following it everywhere except on any 1 of those 21 rows, or any 13 of "
    "those 33, passes G1), and a trace that departs from the comparison on at most 1/5 of each predicate-gated "
    "stratum's rows passes the predicate clauses; (4) per-factor held-out renderings are omitted; (5) a GO is a "
    "seed-0 screen result. The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds "
    "before a sealed R bank is built.")
# The limits quoted with every statement of the verdict (the rule's LIMITS sentence; t2br_analysis renders them
# beside the verdict in every report, with the report-only shortcut-agreement flag). Revision 7 (wording only):
# (1) the H scope with its one exception, (2) heuristics outside H that pass G1 (the revision-6 independent
# check), (3) cond_adv's narrow margins; (4) and (5) unchanged.
ALWAYS_QUOTED_LIMITS = (
    "G1 excludes the members of the enumerated extended shortcut family H (3,519 predicates of other and threshold; "
    "scripts/t2br_shortcuts.py, DESIGN.md section 15) that differ from the comparison, as the enumeration over H "
    "against three plain baselines documents (DRYRUN.json shortcut_enumeration), EXCEPT the isolated exception 'other "
    "> threshold or other == threshold - 4', which passes every G1 part (on any 100-row branch-balanced stratum one "
    "of the exceptions at other - threshold = -4, -3, -2 passes the 0.80 predicate bar; cond_adv catches -3 and -2).",
    "Heuristics OUTSIDE H are NOT excluded, and natural ones pass every G1 part (the revision-6 independent check, "
    "confirmed with verdict() on exact counts, both surfaces; DESIGN.md section 14): isolated single-distance "
    "exceptions at |other - threshold| >= 5 (H has them only at 2..4; e.g. 'other > threshold except at other - "
    "threshold = +5') and flipped distance bands (e.g. 5..9); three-digit-threshold slips (the hundreds digit "
    "dropped, only the first two digits read, a threshold above 96 reduced modulo 97, string comparison only against "
    "three-digit thresholds, always adding when the threshold is above 96); a register of 0 never counted above a "
    "negative threshold; the sign of a threshold at or below -10 dropped; a circular comparison ((other - threshold) "
    "mod 97 in 1..48); string comparison only for one-digit thresholds or one-digit registers; guessing on the far "
    "rows (|other - threshold| > 15) by string comparison or a prior (other > 48, threshold < 48), and, against a "
    "plain arm well below chance on the pooled conditional (the dry run's coin-flip plain at 5/12, or a weak plain "
    "arm), also by always adding, always subtracting or the units digit; and any rule that departs from the "
    "comparison only on cells that no G1 stratum contains (e.g. in-range cross-decade cells at |other - threshold| "
    "5..15, decided by the units digit or with the branch flipped). The report-only shortcut-agreement flag compares "
    "the members of H only, so its absence does not rule these out.",
    "cond_adv catches its targets only at narrow margins: the exceptions at other - threshold = -3, -2, +3 and +4 are "
    "wrong on exactly 21 of its 100 rows (79/100, one row below the 0.80 bar), and the signed string comparison, the "
    "magnitude comparison and the sign+length-aware digit-sum and units comparisons on exactly 33 (67/100). A trace "
    "that follows a caught member on only part of its rows can pass: following it everywhere except on any 1 of those "
    "21 rows, or any 13 of those 33, passes G1; and a trace that departs from the comparison on at most 1/5 of each "
    "predicate-gated stratum's rows passes the predicate clauses.",
    "Per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice: a "
    "held-out-surface loss cannot be attributed to one factor.",
    "A GO is a seed-0 screen result: the roadmap's confirmation requires the same rule in all 3 seeds before a sealed "
    "R bank is built.")
# Revision 6 (2026-10-10 about 03:55, the operator's decisions D-42 (1)-(5); its DESIGN approved as is by D-43 and
# SUPERSEDED by the wording-only revision 7 before any output existed). Kept verbatim (docs/t2b/register/
# GO_RULE_rev6.md, DESIGN_rev6.md section 7); never applied.
GO_RULE_TEXT_REV6 = (
    "PRE-STATED SCREEN GO RULE, REVISION 6 (T2b-register; supersedes revisions 1-5, stated before any output "
    "exists; seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as "
    "rationals (no floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not "
    "meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one "
    "bootstrap SE of its bar is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision "
    "point 1 fired: PREREG-3 TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase "
    "(0-3), field names (variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per "
    "object type (top level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per "
    "object type the fully reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key "
    "top level has only its reversed order, so it trains in the canonical order alone) never occur in training; "
    "targets keep the literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads "
    "only 'after'; a malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: "
    "IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a "
    "held-out key order per object type, rotating over the held-out orders by row, on the primary strata and on "
    "the three identification strata. PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = "
    "cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and "
    "thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the boundary "
    "pairs alone (other = threshold: ok false; other = threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION "
    "strata (G1 only; none is pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with "
    "tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 "
    "(2..7 realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers "
    "and thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on "
    "every primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores "
    "1/2 on it); 'cond_oor_edge' (revision 5) = 20 branch-balanced minimal quadruples at the boundary of the env's "
    "strict 'other > threshold': other - threshold = -1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 "
    "(out of range), in-range registers and thresholds, neither branch wraps, 10 quadruples moving the threshold "
    "and 10 the other register; with cond_oor_val it gives G1 rows at other = threshold - 1, threshold and "
    "threshold + 1, so every off-by-one variant of the comparison (one that differs from it only at other = "
    "threshold - 1, threshold or threshold + 1, e.g. '>=', '> threshold + 1', '>= threshold - 1', 'or other == "
    "threshold - 1') is contradicted on at least 1/4 of its rows; 'cond_adv' (revision 6) = 100 rows, 50 ok true "
    "and 50 ok false, drawn by seeded rejection sampling over valid conditional rows (other 0..96, threshold "
    "-40..136, |value| 12..40, neither branch wraps) to fixed quotas of 15 design types: 13 signature classes of "
    "the extended shortcut family H (sets of (other, threshold) cells on which exactly the same members of H are "
    "wrong), one of them split by digit lengths, covering sign-crossing cells, cross-length cells (1 vs 2, 2 vs 1, "
    "1 vs 3 and 2 vs 3 digits), cross-decade cells whose units comparison disagrees with the tens comparison and "
    "cells at other - threshold = -3, -2, +3, +4; H (scripts/t2br_shortcuts.py, DESIGN.md section 15; 3,519 "
    "predicates) = the enumerated digit and off-by-one family with its modifiers at other = threshold - 1, "
    "threshold and threshold + 1 (3,483 predicates, the branch-blind constants included) + lexicographic string "
    "comparison with and without sign handling, digit-sum, units-only and tens-only comparisons with sign- and "
    "length-aware variants, magnitude-only |other| > |threshold| and the isolated single-distance exceptions at "
    "|other - threshold| = 2..4; on cond_adv every member of H that differs from the comparison somewhere on that "
    "domain and passes revision 5's G1 against a plain arm at 0 is wrong on at least 33 rows, except the isolated "
    "distance exceptions, of which those at other - threshold = -3, -2, +3 and +4 are wrong on exactly 21 (the "
    "operator's target, every member of H wrong on at least 40 rows, is infeasible on any 100-row branch-balanced "
    "stratum: five members of H have pairwise-disjoint error sets). PREDICATE ACCURACY (trace side; "
    "arithmetic-independent): the share of a stratum's rows whose trace answer the strict joint parser accepts, "
    "whose cmp is an object with exactly the keys x, thr, ok in that literal order, and whose ok is a JSON boolean "
    "equal to the env's predicate (other > threshold) for that row; x and thr are not scored; a malformed trace or "
    "a wrong key order counts as wrong. FILLER (amendment 4b): the fill length is a per-kind constant: the "
    "smallest run of ' .' units such that the filler target is at least as long (Titan tokens) as the LONGEST "
    "trace target of that kind over train + dev (one-step rows and the chains' gold steps), whatever its 'after' "
    "(swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up to 'after', so "
    "its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser accepts the "
    "answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION POINT "
    "3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not "
    "supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, "
    "does as well: the gain is attributable to extra decode positions or the answer format)', with the sub-label "
    "'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not "
    "valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff "
    "ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on "
    "cond_sametens alone, AND the trace side's predicate accuracy >= 0.80 on cond_oor_edge alone, on cond_sametens "
    "alone AND on cond_adv alone (cond_adv has no trace - plain clause), each both in-surface and on the held-out "
    "surface (the predicate clauses have no plain-arm term, so a plain arm that is weak on the out-of-range "
    "arithmetic cannot let through a member of H that they catch; G1 excludes the members of H, documented by the "
    "enumeration over H against three plain baselines (DRYRUN.json shortcut_enumeration: every member of H that "
    "differs from the comparison, except the isolated exception 'other > threshold or other == threshold - 4', "
    "which passes every G1 part); it cannot exclude every conceivable heuristic); (G2) trace - plain >= +0.30 on "
    "mix, both in-surface and on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH "
    "in-surface primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; "
    "in-surface); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced "
    "mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If "
    "G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, "
    "naming every failed condition and its failed parts. Reported, not gating: the held-out-surface results of the "
    "OFF and filler sides; the surface audit (held-out minus in-surface, paired by row, every side and "
    "held-out-surface stratum); the filler residual covariate (constant filler tokens minus that row's trace "
    "tokens) with the filler side's accuracy split at its median; the shortcut-agreement analysis (the trace "
    "side's cmp.ok against the env's predicate and against every member of H over all conditional rows of the dev "
    "bank, in each rendering: 'SHORTCUT_CONSISTENT' if a member of H that differs from the predicate on those rows "
    "agrees with the trace's cmp.ok at least as well as the predicate does; it never changes the verdict and is "
    "quoted with any GO); per-factor held-out renderings (paraphrase only, names only, key order only) are omitted "
    "by choice (a GPU-budget choice, not a capacity limit). LIMITS, quoted with every statement of the verdict: G1 "
    "identifies the comparison only against H (it excludes the members of H that the enumeration shows failing it, "
    "not every conceivable heuristic); the isolated exception 'other > threshold or other == threshold - 4' passes "
    "G1 (on any 100-row branch-balanced stratum one of the exceptions at -4, -3, -2 passes the 0.80 bar); "
    "per-factor held-out renderings are omitted; a GO is a seed-0 screen result. The screen is seed 0; the "
    "roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.")
GO_RULE_REV6_SHA256 = "61888db4b1ec7bf6c18d29c30a7413f29b7e4d5f70efcee7fdf737c7a36a5c84"   # checked by _check_sd
# Revision 5 (2026-10-09 about 22:50, the operator's decisions on the checker's two identification gaps; SUPERSEDED by
# revision 6 before any output existed). Kept verbatim (docs/t2b/register/GO_RULE_rev5.md, DESIGN_rev5.md section 7);
# never applied.
GO_RULE_TEXT_REV5 = (
    "PRE-STATED SCREEN GO RULE, REVISION 5 (T2b-register; supersedes revisions 1-4, stated before any output exists; "
    "seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no "
    "floating-point rounding: a difference or share equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% "
    "cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is "
    "flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 TRIGGER): "
    "the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants 0-3 per "
    "field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, event); "
    "paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed canonical order "
    "plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed order, so it trains in "
    "the canonical order alone) never occur in training; targets keep the literal cmp -> calc -> after order. Outcome = "
    "exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). Two "
    "renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim one-step frame; HELD-OUT SURFACE (HS) = "
    "paraphrase 4 + field-name variant 4 + a held-out key order per object type, rotating over the held-out orders by "
    "row, on the primary strata and on the two identification strata. PRIMARY outcomes are out of range (joint amendment "
    "5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range "
    "values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'cond_oor_val' = the "
    "boundary pairs alone (other = threshold: ok false; other = threshold + 1: ok true); 'mix' = mix_oor. IDENTIFICATION "
    "strata (G1 only; neither is pooled into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with "
    "tens(other) == tens(threshold), so the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 "
    "realized: both members of a pair lie in one decade), |value| 12..40 (out of range), in-range registers and "
    "thresholds, neither branch wraps; it separates the comparison from tens-digit shortcuts that are exact on every "
    "primary conditional row (e.g. 'add iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it); "
    "'cond_oor_edge' (revision 5) = 20 branch-balanced minimal quadruples at the boundary of the env's strict 'other > "
    "threshold': other - threshold = -1 and 0 (ok false) and +1 and +2 (ok true), |value| 12..40 (out of range), "
    "in-range registers and thresholds, neither branch wraps, 10 quadruples moving the threshold and 10 the other "
    "register; with cond_oor_val it gives G1 rows at other = threshold - 1, threshold and threshold + 1, so every "
    "off-by-one variant of the comparison (one that differs from it only at other = threshold - 1, threshold or "
    "threshold + 1, e.g. '>=', '> threshold + 1', '>= threshold - 1', 'or other == threshold - 1') is contradicted on at "
    "least 1/4 of its rows. PREDICATE ACCURACY (trace side; arithmetic-independent): the share of a stratum's rows whose "
    "trace answer the strict joint parser accepts, whose cmp is an object with exactly the keys x, thr, ok in that "
    "literal order, and whose ok is a JSON boolean equal to the env's predicate (other > threshold) for that row; x and "
    "thr are not scored; a malformed trace or a wrong key order counts as wrong. FILLER (amendment 4b): the fill length "
    "is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long (Titan "
    "tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold steps), "
    "whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is identical up "
    "to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict joint parser "
    "accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each primary. DECISION "
    "POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ filler, not "
    "supervision of the intermediates (an answer-free filler of constant length per action kind, amendment 4b, does as "
    "well: the gain is attributable to extra decode positions or the answer format)', with the sub-label 'trace format "
    "failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else "
    "'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO iff ALL of: (G1) trace "
    "- plain >= +0.25 on conditional, on cond_oor_val alone, on cond_oor_edge alone AND on cond_sametens alone, AND the "
    "trace side's predicate accuracy >= 0.80 on cond_oor_edge alone AND on cond_sametens alone (a balanced stratum gives "
    "a branch-blind or tens-digit shortcut about 0.50, and an off-by-one variant scores at most 0.75 on cond_oor_edge; "
    "this clause has no plain-arm term, so a plain arm that is weak on the out-of-range arithmetic cannot let a shortcut "
    "through), each both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and "
    "on the held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) "
    "controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution "
    "non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, "
    "cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not "
    "valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. "
    "Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus "
    "in-surface, paired by row, every side and held-out-surface stratum); the filler residual covariate (constant filler "
    "tokens minus that row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out "
    "renderings (paraphrase only, names only, key order only) are omitted by choice (a GPU-budget choice, not a capacity "
    "limit). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R "
    "bank is built.")
GO_RULE_REV5_SHA256 = "241befabca5352b07b7d20c97c017654ad57ef3c40d476ab6d4897dcfe6f9abf"   # checked by _check_sd
# Revision 4 (2026-10-09 about 17:45, the operator-selected verifier fixes; SUPERSEDED by revision 5 before any
# output existed). Kept verbatim (docs/t2b/register/GO_RULE_rev4.md, DESIGN_rev4.md section 7); never applied.
GO_RULE_TEXT_REV4 = (
    "PRE-STATED SCREEN GO RULE, REVISION 4 (T2b-register; supersedes revisions 1-3, stated before any output exists; "
    "seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no "
    "floating-point rounding: a difference equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% "
    "cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar "
    "is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 "
    "TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names "
    "(variants 0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top "
    "level, state, event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully "
    "reversed canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its "
    "reversed order, so it trains in the canonical order alone) never occur in training; targets keep the literal cmp "
    "-> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed "
    "trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's verbatim "
    "one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per object "
    "type, rotating over the held-out orders by row, on the primary strata and on cond_sametens. PRIMARY outcomes are "
    "out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced "
    "pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's "
    "0..15); 'cond_oor_val' = the boundary pairs alone; 'mix' = mix_oor. IDENTIFICATION stratum (G1 only; not pooled "
    "into 'conditional'): 'cond_sametens' = 50 branch-balanced minimal pairs with tens(other) == tens(threshold), so "
    "the branch is decided by the units digit, |other - threshold| in 2..9 (2..7 realized: both members of a pair lie "
    "in one decade), |value| 12..40 (out of range), in-range registers and thresholds, neither branch wraps; it "
    "separates the comparison from tens-digit shortcuts that are exact on every primary conditional row (e.g. 'add "
    "iff tens(other) > tens(threshold) or other == threshold + 1' scores 1/2 on it). FILLER (amendment 4b): the fill "
    "length is a per-kind constant: the smallest run of ' .' units such that the filler target is at least as long "
    "(Titan tokens) as the LONGEST trace target of that kind over train + dev (one-step rows and the chains' gold "
    "steps), whatever its 'after' (swap 7, add 17, mix 17, conditional 29 units); every filler target of a kind is "
    "identical up to 'after', so its length carries no information about the answer. VALIDITY: format_ok = the strict "
    "joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on each "
    "primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: 'trace ~ "
    "filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, "
    "amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the "
    "sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side "
    "is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO "
    "iff ALL of: (G1) trace - plain >= +0.25 on conditional, on cond_oor_val alone AND on cond_sametens alone, each "
    "both in-surface and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the "
    "held-out surface; (G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: "
    "trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution "
    "non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, "
    "cond_xp1, cond_mid) and on mix_in (in-surface); and the filler side is valid. If G1-G5 all hold but the filler "
    "side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition and its "
    "failed parts. Reported, not gating: the held-out-surface results of the OFF and filler sides; the surface audit "
    "(held-out minus in-surface, paired by row, every side and held-out-surface stratum); the filler residual "
    "covariate (constant filler tokens minus that row's trace tokens) with the filler side's accuracy split at its "
    "median; per-factor held-out renderings (paraphrase only, names only, key order only) are omitted by choice (a "
    "GPU-budget choice, not a capacity limit). The screen is seed 0; the roadmap's confirmation requires the same "
    "rule in all 3 seeds before a sealed R bank is built.")
GO_RULE_REV4_SHA256 = "8f9dc0cb3af79dae2fde091e1878bcf29a6b2a2048783a52379f8da100095d54"   # checked by _check_sd
# Revision 3 (2026-10-09 about 12:24, operator decisions after PREREG-3, D-36; SUPERSEDED by revision 4 before any
# output existed). Kept verbatim (docs/t2b/register/GO_RULE_rev3.md, DESIGN_rev3.md section 7); never applied.
GO_RULE_TEXT_REV3 = (
    "PRE-STATED SCREEN GO RULE, REVISION 3 (T2b-register; supersedes revisions 1 and 2, stated before any output "
    "exists; seed "
    "by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no "
    "floating-point rounding: a difference equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% "
    "cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar "
    "is flagged in the report). TRAINING: the surface-diversity build 'sd' only (decision point 1 fired: PREREG-3 "
    "TRIGGER): the same 750 rows, each prompt rendered with a seeded rule-text paraphrase (0-3), field names (variants "
    "0-3 per field) and JSON key orders drawn from a TRAIN subset of permutations per object type (top level, state, "
    "event); paraphrase 4, field-name variant 4 and the HELD-OUT key orders (per object type the fully reversed "
    "canonical order plus 2 fixed seeded permutations where they exist; the 2-key top level has only its reversed "
    "order, so it trains in the canonical order alone) never occur in training; targets keep the "
    "literal cmp -> calc -> after order. Outcome = exact 'after' state (the decision parser reads only 'after'; a "
    "malformed trace/filler answer is wrong). Two renderings of the same rows, paired by row: IN-SURFACE = Titan's "
    "verbatim one-step frame; HELD-OUT SURFACE (HS) = paraphrase 4 + field-name variant 4 + a held-out key order per "
    "object type, rotating over the held-out orders by row, on the primary strata. PRIMARY outcomes are out of range "
    "(joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; "
    "cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); "
    "'cond_oor_val' = the boundary pairs alone; 'mix' = mix_oor. FILLER (amendment 4b): the fill is a run of ' .' of a "
    "CONSTANT length per action kind (swap 5, add 14, mix 14, conditional 26 units: the shortest run whose filler "
    "target is at least as long as every trace target of that kind in train + dev), so every filler target of a kind "
    "is identical up to 'after' and its length carries no information about the answer. VALIDITY: format_ok = the "
    "strict joint parser accepts the answer; the trace (filler) side is valid iff its in-surface format_ok >= 0.95 on "
    "each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH in-surface primaries, STOP: "
    "'trace ~ filler, not supervision of the intermediates (an answer-free filler of constant length per action kind, "
    "amendment 4b, does as well: the gain is attributable to extra decode positions or the answer format)', with the "
    "sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side "
    "is not valid, else 'neither format helped' if filler - plain < 0.10 on both in-surface primaries. Otherwise GO "
    "iff ALL of: (G1) trace - plain >= +0.25 on conditional AND >= +0.25 on cond_oor_val alone, each both in-surface "
    "and on the held-out surface; (G2) trace - plain >= +0.30 on mix, both in-surface and on the held-out surface; "
    "(G3) filler below trace: trace - filler >= +0.10 on EACH in-surface primary; (G4) controls: trace ON - OFF >= "
    "-0.05 on swap and on add_nowrap (untouched skills; in-surface); (G5) in-distribution non-inferiority: trace - "
    "plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in "
    "(in-surface); and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: "
    "filler control invalid. Otherwise NO-GO, naming every failed condition and its failed parts. Reported, not "
    "gating: the held-out-surface results of the OFF and filler sides; the surface audit (held-out minus in-surface, "
    "paired by row, every side and primary stratum); the filler residual covariate (constant filler tokens minus that "
    "row's trace tokens) with the filler side's accuracy split at its median; per-factor held-out renderings are not "
    "run (GPU budget). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a "
    "sealed R bank is built.")
GO_RULE_REV3_SHA256 = "aff13a4e9937af49daec6df178d555ff1ecb116e3da490b94f6c47bbceabe9b7"   # checked by _check_sd
# Revision 2 (2026-10-09 about 07:54, after the adversarial review; SUPERSEDED by revision 3 before any output existed).
# Kept verbatim (docs/t2b/register/GO_RULE_rev2.md, DESIGN_rev2.md section 7); never applied.
GO_RULE_TEXT_REV2 = (
    "PRE-STATED SCREEN GO RULE (T2b-register, seed by seed; point estimates on the fresh dev screen bank, decided "
    "EXACTLY on integer counts as rationals (no floating-point rounding: a difference equal to a bar meets a '>=' bar "
    "and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided "
    "within one bootstrap SE of its bar is flagged in the report). Outcome = exact 'after' state (the decision parser "
    "reads only 'after'; a malformed trace/filler answer is wrong). PRIMARY outcomes are out of range (joint amendment "
    "5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range "
    "values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'mix' = mix_oor. "
    "VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its "
    "format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH primaries, "
    "STOP: 'trace ~ filler, not supervision of the intermediates (extra decode compute and/or the length-coded "
    "magnitude/branch information in the amendment-4 fill; not attributable to decode compute alone)', with the "
    "sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side "
    "is not valid, else 'neither format helped' if filler - plain < 0.10 on both primaries. Otherwise GO iff ALL of: "
    "(G1) trace - plain >= +0.25 on conditional; (G2) trace - plain >= +0.30 on mix; (G3) filler below trace: trace - "
    "filler >= +0.10 on EACH primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched "
    "skills); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean "
    "over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in; and the filler side is valid. If G1-G5 all hold but "
    "the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed "
    "condition. Pre-stated filler diagnostics (reported, not gating): per primary stratum the filler side's format_ok, "
    "fill_exact and its after-accuracy split by fill_exact, and the bank's fill-length leakage (how well the gold fill "
    "length alone predicts wrap / branch). The screen is seed 0; the roadmap's confirmation requires the same rule in "
    "all 3 seeds before a sealed R bank is built.")
GO_RULE_REV2_SHA256 = "d4bdf397fb9c3dee96513a6d5e9c8bca1fce0a9640e85df3c1ccc41882057b95"   # checked by _check_sd
# Revision 1 (2026-10-09 about 03:04, the first build; SUPERSEDED by revision 2 before any output existed). Kept verbatim
# (docs/t2b/register/GO_RULE_rev1.md, DESIGN_rev1.md section 7); never applied.
GO_RULE_TEXT_REV1 = (
    "PRE-STATED SCREEN GO RULE (T2b-register, seed by seed; point estimates on the fresh dev screen bank; 95% cluster-"
    "bootstrap CIs reported, not gating). Outcome = exact 'after' state (decision parser reads only 'after'; a "
    "malformed trace/filler answer is wrong). PRIMARY outcomes are the out-of-range strata (joint amendment 5): "
    "'conditional' = cond_oor_val + cond_oor_thr pooled (branch-balanced by construction); 'mix' = mix_oor. GO iff ALL "
    "of: (G1) trace - plain >= +0.25 on conditional; (G2) trace - plain >= +0.30 on mix; (G3) filler below trace: "
    "trace - filler > 0 on both primaries; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched "
    "skills); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean of "
    "cond_xm1/cond_x0/cond_xp1) and on mix_in. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH "
    "primaries, STOP and report 'extra decode compute, not supervision' (sub-reason 'neither format helped' when filler "
    "- plain < 0.10 on both primaries as well). Otherwise NO-GO, naming every failed condition. The screen is seed 0; "
    "the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.")
GO_RULE_REV1_SHA256 = "c7be67e0039c0687550e952ff7be98f4d108f31817835f8926e99859d377f640"   # checked by _check_sd

# ---------------------------------------------------------------------------------------------- semantics


def branch_ok(before: dict, ev: dict) -> bool:
    """The env's conditional predicate (strict)."""
    return before[ev["other"]] > ev["threshold"]


def raw_value(before: dict, ev: dict):
    k, r = ev["kind"], before[ev["register"]]
    if k == "add":
        return r + ev["value"]
    if k == "mix":
        return r + 2 * before[ev["other"]] + ev["value"]
    if k == "conditional":
        return r + (ev["value"] if branch_ok(before, ev) else -ev["value"])
    return None


def wraps(before: dict, ev: dict) -> bool:
    raw = raw_value(before, ev)
    return raw is not None and not 0 <= raw < MOD


def op_class(before: dict, ev: dict) -> str:
    """scripts/make_t2_data.stratum('R', ...) (that module imports torch; replicated as t2_analysis.t2_stratum does)."""
    k = ev["kind"]
    if k == "swap":
        return "swap"
    if k == "conditional":
        return "conditional/" + ("gt" if before[ev["other"]] > ev["threshold"] else "le")
    raw = before[ev["register"]] + (ev["value"] if k == "add" else 2 * before[ev["other"]] + ev["value"])
    return f"{k}/" + ("wrap" if not 0 <= raw < MOD else "nowrap")


def cond_dist(before: dict, ev: dict) -> int:
    """other - threshold of a conditional (ok iff > 0)."""
    return before[ev["other"]] - ev["threshold"]


def is_oor(before: dict, ev: dict) -> bool:
    """Outside the training ranges: value, threshold, or (conditional) |other - threshold| beyond TWIN_DIST."""
    k = ev["kind"]
    if k == "swap":
        return False
    v = ev["value"]
    if not IN_V[0] <= v <= IN_V[1]:
        return True
    if k == "conditional":
        return not THR_IN[0] <= ev["threshold"] <= THR_IN[1] or abs(cond_dist(before, ev)) > TWIN_DIST[1]
    return False


def single_diff(a: tuple, b: tuple):
    """Which ONE number differs between two (before, event) conditionals: 'threshold' / 'other' / None (anything else)."""
    (b1, e1), (b2, e2) = a, b
    same_ev = {k: v for k, v in e1.items() if k != "threshold"} == {k: v for k, v in e2.items() if k != "threshold"}
    if not same_ev or e1["kind"] != "conditional":
        return None
    o = e1["other"]
    others_same = all(b1[r] == b2[r] for r in REGS if r != o)
    if b1 == b2 and e1["threshold"] != e2["threshold"]:
        return "threshold"
    if others_same and b1[o] != b2[o] and e1["threshold"] == e2["threshold"]:
        return "other"
    return None


def parts(before: dict, ev: dict):
    """(cmp, calc, after) of the joint trace for one register step; dicts in their literal key order."""
    after = env.step(FAM, before, ev)
    k = ev["kind"]
    cmp = {"x": before[ev["other"]], "thr": ev["threshold"], "ok": branch_ok(before, ev)} if k == "conditional" else None
    if k == "swap":
        calc = None
    else:
        raw = raw_value(before, ev)
        calc = {"raw": raw, "mod": raw % MOD}
        if after[ev["register"]] != calc["mod"]:
            raise AssertionError(f"calc disagrees with env.step: {before} {ev}")
    return cmp, calc, after


def _dump(obj) -> str:
    """Compact JSON in the object's OWN key order (never sorted)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def plain_text(after: dict) -> str:
    return env.dumps(after)


def trace_text(cmp, calc, after) -> str:
    return '{"cmp":' + _dump(cmp) + ',"calc":' + _dump(calc) + ',"after":' + env.dumps(after) + "}"


def filler_text(k: int, after: dict) -> str:
    return '{"fill":"' + FILL_UNIT * k + '","after":' + env.dumps(after) + "}"


def fill_units(kind: str, table: dict | None = None) -> int:
    """Amendment 4b: the constant fill length of an action kind (prompt-derived, never answer-derived)."""
    return (FILL_UNITS if table is None else table)[kind]


def targets(before: dict, ev: dict, fill_table: dict | None = None) -> dict:
    """fill_table None = the current FILL_UNITS (FILL_UNITS_REV3 only to reconstruct the revision-3 files)."""
    cmp, calc, after = parts(before, ev)
    k = fill_units(ev["kind"], fill_table)
    return {"plain": plain_text(after), "trace": trace_text(cmp, calc, after), "filler": filler_text(k, after),
            "fill_units": k, "cmp": cmp, "calc": calc, "after": env.dumps(after)}


def min_constant_fill(afters: list, longest_trace_tokens: int, count) -> int:
    """Amendment 4b (revision 4): the smallest fill n (units of ' .') such that EVERY filler target of the kind (one per
    given 'after') has at least longest_trace_tokens Titan tokens, i.e. the shortest filler target of the kind is at
    least as long as the kind's LONGEST trace target. A per-kind constant: it does not pair a row's filler with that
    row's trace."""
    n = 1
    while min(count(filler_text(n, a)) for a in afters) < longest_trace_tokens:
        n += 1
        if n > 400:
            raise ValueError("no constant fill length found")
    return n


def min_per_row_fill_rev3(afters_and_trace_tokens: list, count) -> int:
    """Revision 3's (superseded) derivation, kept for the record: the smallest n with filler(n, after_i) >= trace_i on
    every row i (each row's filler compared with THAT row's trace). It is <= min_constant_fill."""
    n = 1
    while not all(count(filler_text(n, a)) >= t for a, t in afters_and_trace_tokens):
        n += 1
        if n > 400:
            raise ValueError("no constant fill length found")
    return n


def best_cut_acc(values: list, labels: list) -> float:
    """Accuracy of the best single cut 'label iff value <= c' (or its negation): how much a scalar tells about a label."""
    n = len(values)
    if n == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: values[i])
    tot_true = sum(bool(x) for x in labels)
    best = max(tot_true, n - tot_true) / n
    ct = cf = i = 0
    while i < n:
        v = values[order[i]]
        while i < n and values[order[i]] == v:
            if labels[order[i]]:
                ct += 1
            else:
                cf += 1
            i += 1
        acc = (ct + (n - tot_true) - cf) / n
        best = max(best, acc, 1 - acc)
    return best


# ---------------------------------------------------------------------------------------------- parsing and scoring


def _loads_ordered(text: str):
    """json.loads that keeps key order and REJECTS duplicate keys."""
    def hook(pairs):
        keys = [k for k, _ in pairs]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate key")
        return dict(pairs)
    return json.loads(text, object_pairs_hook=hook)


def parse_trace(text):
    """(cmp, calc, after) or None. Keys must be exactly cmp, calc, after, in that order (no extra, no duplicates)."""
    if not isinstance(text, str):
        return None
    try:
        d = _loads_ordered(text.strip())
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(d, dict) or list(d) != ["cmp", "calc", "after"]:
        return None
    return d["cmp"], d["calc"], d["after"]


def parse_filler(text):
    """(fill, after) or None. Keys must be exactly fill, after, in that order. The fill's length is NOT scored (4b: the
    constant length is a property of the data, not an outcome)."""
    if not isinstance(text, str):
        return None
    try:
        d = _loads_ordered(text.strip())
    except (ValueError, TypeError, RecursionError):
        return None
    if not isinstance(d, dict) or list(d) != ["fill", "after"]:
        return None
    return d["fill"], d["after"]


def after_text(text, arm: str):
    """The 'after' part for the family's existing parser (plain: the text itself). BWM traces.after_text convention."""
    if arm == "plain":
        return text
    if arm == "trace":
        p = parse_trace(text)
        return None if p is None else json.dumps(p[2], separators=(",", ":"), sort_keys=True)
    if arm == "filler":
        p = parse_filler(text)
        return None if p is None else json.dumps(p[1], separators=(",", ":"), sort_keys=True)
    raise ValueError(arm)


def parse_after(text, arm: str):
    """Decision parser: the state in 'after' via env.parse_state(lenient=True), or None (malformed = wrong)."""
    a = after_text(text, arm)
    if a is None:
        return None
    return env.parse_state(FAM, a, lenient=True)


def same_json(a, b) -> bool:
    """Equal as ORDERED JSON (key order and types matter: true != 1)."""
    try:
        return _dump(a) == _dump(b)
    except (TypeError, ValueError):
        return False


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


CMP_KEYS = ["x", "thr", "ok"]                        # the cmp object's literal key order (joint format)


def pred_ok(text, row: dict) -> bool:
    """Revision 5, G1's arithmetic-independent clause: the trace answer states the env's predicate for this row. True iff
    the strict joint parser accepts the answer (keys cmp, calc, after in that order; no extra / missing / duplicate
    keys), cmp is an object with exactly the keys x, thr, ok in that literal order, and ok is a JSON boolean equal to
    the row's gold predicate (other > threshold). x and thr are NOT scored. A malformed trace, a wrong key order (top
    level or cmp), a missing / non-boolean ok and any non-conditional row give False."""
    gc = row["gold_cmp"]
    p = parse_trace(text)
    if gc is None or p is None:
        return False
    c = p[0]
    return isinstance(c, dict) and list(c) == CMP_KEYS and isinstance(c["ok"], bool) and c["ok"] == gc["ok"]


def stated_ok(text):
    """Revision 6 (the report-only shortcut-agreement analysis): the boolean a trace answer STATES in cmp.ok, read as
    strictly as pred_ok reads it (the strict joint parser, cmp keys exactly x, thr, ok in that order, a JSON boolean
    ok), else None. For a conditional row, pred_ok(text, row) == (stated_ok(text) == row's gold ok)."""
    p = parse_trace(text)
    if p is None:
        return None
    c = p[0]
    return c["ok"] if isinstance(c, dict) and list(c) == CMP_KEYS and isinstance(c["ok"], bool) else None


def score_text(row: dict, text, arm: str) -> dict:
    """Per-row score. after_ok is the outcome; pred_ok (trace side) is the outcome of G1's predicate clause (revision
    5); the other trace fields are mechanism diagnostics."""
    st = parse_after(text, arm)
    res = {"parsed": st is not None, "after_ok": st is not None and env.dumps(st) == row["gold_after"]}
    if arm == "trace":
        p = parse_trace(text)
        res["format_ok"] = p is not None
        res["pred_ok"] = pred_ok(text, row)
        gc, gk = row["gold_cmp"], row["gold_calc"]
        if p is None:
            res.update(cmp_ok=False, ok_ok=False, calc_ok=False, after_follows_calc=False, calc_follows_cmp=False)
        else:
            c, k, _ = p
            res["cmp_ok"] = same_json(c, gc)
            res["ok_ok"] = gc is not None and isinstance(c, dict) and isinstance(c.get("ok"), bool) and c["ok"] == gc["ok"]
            res["calc_ok"] = same_json(k, gk)
            ev, b = row["event"], row["before"]
            fol = False
            if gk is None:
                fol = k is None and st is not None
            elif isinstance(k, dict) and _int(k.get("mod")) and st is not None:
                exp = dict(b)
                exp[ev["register"]] = k["mod"]
                fol = env.dumps(st) == env.dumps(exp)
            res["after_follows_calc"] = fol
            cf = False
            if ev["kind"] == "conditional" and isinstance(c, dict) and isinstance(c.get("ok"), bool) \
                    and isinstance(k, dict) and _int(k.get("raw")):
                cf = k["raw"] == b[ev["register"]] + (ev["value"] if c["ok"] else -ev["value"])
            res["calc_follows_cmp"] = cf
    elif arm == "filler":
        res["format_ok"] = parse_filler(text) is not None
    return res


def fill_units_of(text):
    """Number of ' .' units in a parsed filler answer's fill, or None (unparsed / not a pure run of the unit)."""
    p = parse_filler(text)
    if p is None or not isinstance(p[0], str):
        return None
    k = len(p[0]) // len(FILL_UNIT)
    return k if p[0] == FILL_UNIT * k else None


# ---------------------------------------------------------------------------------------------- surface heuristics


def heuristic_after(name: str, before: dict, ev: dict):
    """A deterministic shortcut's answer for one step, or None where the heuristic does not apply."""
    k, r = ev["kind"], ev["register"]
    o, v, t = ev.get("other"), ev.get("value", 0), ev.get("threshold")
    b = dict(before)
    if name == "copy":
        return b
    if name == "unreduced":                          # the correct operation (correct branch) without the modulo
        if k == "swap":
            return env.step(FAM, before, ev)
        b[r] = raw_value(before, ev)
        return b
    if k == "conditional":
        x = before[o]
        dec = {"always_add": True, "always_sub": False,
               # single surface cues of the branch (no comparison of other with threshold)
               "wrong_register": before[r] > t, "x_hi": x > 48, "thr_lo": t < 48, "v_pos": v > 0,
               # rule variants: comparisons that coincide with the env's 'other > threshold' on a structural subset
               "non_strict": x >= t, "eq_plus1": x == t + 1, "units_eq_plus1": x % 10 == (t + 1) % 10,
               "units_digit_gt": x % 10 > t % 10, "tens_digit_gt": x // 10 > t // 10,
               # revision 4 digit shortcuts (Python floor // and %): exact on every primary conditional row
               "tens_or_eq_plus1": x // 10 > t // 10 or x == t + 1,
               "tens_ge_not_eq": x // 10 >= t // 10 and x != t,
               # revision 5: the off-by-one family (the env's 'x > t' with the branch flipped exactly where x - t lies
               # in the named set): obo_m1 'x > t or x == t-1'; obo_0 'x >= t' (= non_strict); obo_p1 'x > t+1' (= 'x > t
               # and x != t+1'); obo_m1_0 'x >= t-1'; obo_m1_p1 '(x > t or x == t-1) and x != t+1'; obo_0_p1 'x == t or
               # x > t+1'; obo_m1_0_p1 'x in {t-1, t} or x > t+1'
               **{n: (x > t) != (x - t in s) for n, s in OFF_BY_ONE.items()}}.get(name)
        if dec is None:
            return None
        b[r] = (before[r] + (v if dec else -v)) % MOD
        return b
    if k == "mix":
        if name == "mix_1x":
            b[r] = (before[r] + before[o] + v) % MOD
            return b
        if name == "mix_no_v":
            b[r] = (before[r] + 2 * before[o]) % MOD
            return b
    if k == "add" and name == "always_wrap" and v != 0:   # 'add => wrap by 97' (true of every training add, by design)
        b[r] = before[r] + v - (MOD if v > 0 else -MOD)
        return b
    return None


# Surface heuristics, gated <= AUDIT_MAX (0.55) on the training rows (the task's audit): no comparison of other with
# threshold is involved.
GATED_HEURISTICS = ("copy", "unreduced", "always_add", "always_sub", "wrong_register", "x_hi", "thr_lo", "v_pos")
# Rule variants: each equals the env's predicate on a structural subset (eq_plus1 on every ok-false row and every
# other = threshold + 1 row; non_strict everywhere except other = threshold; ...), so its accuracy is at least that
# subset's share: acc(eq_plus1) = P(ok false) + P(other = threshold + 1) >= 0.5 + 50/N with the roadmap's 50 rows at
# threshold + 1, and acc(non_strict) = 1 - P(other = threshold). A 0.55 bar is therefore unreachable for this class at
# N = 300 (it needs N >= 1000 for eq_plus1 alone). They are gated by IDENTIFICATION instead: each is contradicted by at
# least RULE_VARIANT_MIN_CONTRA of the 300 conditional training rows (accuracy <= 1 - that share; operator-accepted).
RULE_VARIANTS = ("non_strict", "eq_plus1", "units_eq_plus1", "units_digit_gt", "tens_digit_gt")
RULE_VARIANT_MIN_CONTRA = Fraction(1, 6)            # >= 50 of 300 rows: one core offset block
# Revision 4: the verifier's tens-digit shortcut and its mirror, derived for the env's STRICT '>': the cond_oor_val
# boundary pair is (other = threshold: false, other = threshold + 1: true), so the shortcut's equality term is
# 'other == threshold + 1' (the first true value) and the mirror's is 'other != threshold' (the last false value); for a
# non-strict env they would be 'other == threshold' / 'other != threshold - 1'. Both are exact on every primary
# conditional row of the dev bank (the mirror would err on a cond_oor_thr row with threshold 97..99 and other 90..96;
# the bank has none) and score 1/2 on cond_sametens. Reported (training audit and dev floors), NOT gated on the training
# rows: the 750 rows are fixed (never re-drawn) and the shortcut is contradicted by only 25 of the 300 conditional
# training rows (the twins at +2..+9 inside one decade; below the rule variants' 1/6 bar), the mirror by 59.
DIGIT_SHORTCUTS = ("tens_or_eq_plus1", "tens_ge_not_eq")
# Revision 5: the off-by-one family, every non-empty flip set of x - t in {-1, 0, +1} (other = threshold - 1 / threshold
# / threshold + 1). Reported on the training rows and as dev floors, NOT gated on the training rows (the 150 core rows at
# -1 / 0 / +1 contradict every single flip on exactly 50 of the 300 conditional rows, the 1/6 bar); identified on the dev
# bank by G1 (cond_oor_val's (0, +1) pairs and cond_oor_edge's (-1, 0, +1, +2) quadruples).
OFF_BY_ONE = OrderedDict([("obo_m1", (-1,)), ("obo_0", (0,)), ("obo_p1", (1,)), ("obo_m1_0", (-1, 0)),
                          ("obo_m1_p1", (-1, 1)), ("obo_0_p1", (0, 1)), ("obo_m1_0_p1", (-1, 0, 1))])
OFF_BY_ONE_FORMS = {"obo_m1": "x > t or x == t-1", "obo_0": "x >= t", "obo_p1": "x > t+1", "obo_m1_0": "x >= t-1",
                    "obo_m1_p1": "(x > t or x == t-1) and x != t+1", "obo_0_p1": "x == t or x > t+1",
                    "obo_m1_0_p1": "x in {t-1, t} or x > t+1"}
DIAG_HEURISTICS = ("mix_1x", "mix_no_v", "always_wrap")
AUDIT_MAX = Fraction(55, 100)

# ---------------------------------------------------------------------------------------------- prompts


def render_prompt(messages: list) -> str:
    """Qwen3 chat template with add_generation_prompt=True, enable_thinking=False (rcwmt.capture.render), written out
    so the data builder needs no torch. Verified equal to every register prompt of data/t2_train.jsonl (rendered by
    the real template) at build time, and to tok.apply_chat_template on every row in the tiny test."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def canonical_messages(before: dict, ev: dict) -> list:
    return env.one_step_messages(FAM, before, ev)


# Rule-text paraphrases. They keep the exact semantics (strict '>', modulo 97, simultaneous swap, mix = 2 x other +
# value) and name the event fields through placeholders, so a field-name variant renames the field in the prompt JSON
# AND in the rule text together. SD_PARAPHRASES[0] with the canonical names == env.SYSTEM_ONE. Index 4 is HELD OUT.
SD_PARAPHRASES = (
    "Apply one instruction to the supplied registers. Registers are integers modulo 97. swap exchanges {register} and "
    "{other} simultaneously. add adds {value} to {register}. mix adds (2 times {other} plus {value}) to {register}. "
    "conditional adds {value} if {other} is greater than {threshold}, otherwise subtracts {value}. Reduce updated "
    "numeric registers modulo 97; preserve other registers. Return only the complete resulting registers as a JSON "
    "object.",
    "You are given registers holding integers modulo 97 and one instruction. swap: the registers named by {register} "
    "and {other} exchange values at the same time. add: the {register} register increases by {value}. mix: the "
    "{register} register increases by twice the {other} register plus {value}. conditional: if the {other} register is "
    "strictly greater than {threshold}, the {register} register increases by {value}; otherwise it decreases by "
    "{value}. Reduce every updated register modulo 97 and leave the rest unchanged. Output only the complete resulting "
    "registers as a JSON object.",
    "Execute the single instruction on the registers (integers modulo 97). swap - exchange the {register} and {other} "
    "registers simultaneously; add - add {value} to the {register} register; mix - add 2 x the {other} register + "
    "{value} to the {register} register; conditional - when the {other} register is greater than {threshold}, add "
    "{value} to the {register} register, else subtract {value} from it. Each updated register is reduced modulo 97; "
    "untouched registers keep their values. Reply with only the complete resulting registers as a JSON object.",
    "Registers are integers modulo 97. Carry out the one instruction given. For swap, the {register} and {other} "
    "registers trade values simultaneously. For add, {value} is added to the {register} register. For mix, the "
    "{register} register gains two times the {other} register plus {value}. For conditional, the {register} register "
    "gains {value} if the {other} register is greater than {threshold} and loses {value} otherwise. Take updated "
    "registers modulo 97 and keep the others as they are. Return just the complete resulting registers as a JSON "
    "object.",
    "Update the registers by applying the single instruction below; every register holds an integer modulo 97. A swap "  # held out
    "instruction makes the {register} and {other} registers exchange their values at once. An add instruction "
    "increases the {register} register by {value}. A mix instruction increases the {register} register by {value} "
    "plus double the {other} register. A conditional instruction increases the {register} register by {value} when "
    "the {other} register exceeds {threshold}, and decreases it by {value} when it does not. Every changed register is "
    "reduced modulo 97, and every other register stays the same. Respond with nothing but the full set of resulting "
    "registers as one JSON object.",
)
TRAIN_PARAPHRASES = (0, 1, 2, 3)
HELDOUT_PARAPHRASE = 4
TEXT_FIELDS = ("register", "other", "value", "threshold")      # the fields the rule text names
# Field-name variants per field (index 0 = canonical, 1-3 = training, 4 = HELD OUT). All 35 names are distinct strings,
# distinct from 'tick' and from the register names, so a prompt's names identify their field and variant.
SD_FIELDS = OrderedDict([
    ("register", ("register", "target", "dst", "reg", "dest")),
    ("other", ("other", "source", "src", "operand", "peer")),
    ("value", ("value", "amount", "delta", "change", "offset")),
    ("threshold", ("threshold", "limit", "cutoff", "bound", "pivot")),
    ("kind", ("kind", "op", "type", "action", "opcode")),
    ("before", ("before", "registers", "state", "regs", "memory")),
    ("event", ("event", "instruction", "command", "step", "operation")),
])
TRAIN_NAME_IDX = (0, 1, 2, 3)
HELDOUT_NAME_IDX = 4
# JSON objects of the one-step register prompt, by object type, in their CANONICAL (env.dumps: sorted) key order,
# keyed by canonical role names. The register prompt has no lists.
OBJECT_KEYS = OrderedDict([("top", ("before", "event")), ("state", tuple(REGS)), ("event", EVENT_KEYS)])
HELDOUT_ORDER_SEED = 20261113
# HELD-OUT key orders per object type (C2): the fully reversed canonical order + 2 fixed permutations drawn with
# random.Random(HELDOUT_ORDER_SEED).sample from the permutations that are neither canonical nor reversed (in OBJECT_KEYS
# order; checked below). The 2-key top-level object has no third permutation: only its reversed order is held out, so
# its TRAIN subset is the canonical order alone.
HELDOUT_ORDERS = OrderedDict([
    ("top", (("event", "before"),)),
    ("state", (("Z", "Y", "X"), ("Y", "X", "Z"), ("X", "Z", "Y"))),
    ("event", (("value", "tick", "threshold", "register", "other", "kind"),
               ("other", "kind", "tick", "threshold", "register", "value"),
               ("threshold", "kind", "tick", "other", "value", "register"))),
])
TRAIN_ORDERS = OrderedDict((t, tuple(p for p in itertools.permutations(keys) if p not in HELDOUT_ORDERS[t]))
                           for t, keys in OBJECT_KEYS.items())
HS_ROTATION = 3                                     # held-out orders rotate by row: order index = j % len(orders)


def _seeded_heldout_orders() -> OrderedDict:
    rng = random.Random(HELDOUT_ORDER_SEED)
    out = OrderedDict()
    for t, canon in OBJECT_KEYS.items():
        rev = tuple(reversed(canon))
        cands = [p for p in itertools.permutations(canon) if p not in (canon, rev)]
        out[t] = (rev,) + tuple(rng.sample(cands, min(2, len(cands))))
    return out


def canonical_surface() -> dict:
    return {"paraphrase": 0, "names": {f: 0 for f in SD_FIELDS}, "orders": {t: list(k) for t, k in OBJECT_KEYS.items()}}


def draw_train_surface(rng: random.Random) -> dict:
    """One training row's surface: paraphrase, field names and key orders from the TRAIN subsets only."""
    para = rng.choice(TRAIN_PARAPHRASES)
    names = {f: rng.choice(TRAIN_NAME_IDX) for f in SD_FIELDS}
    orders = {t: list(rng.choice(TRAIN_ORDERS[t])) for t in OBJECT_KEYS}
    return {"paraphrase": para, "names": names, "orders": orders}


def heldout_surface(j: int) -> dict:
    """The combined held-out surface of the j-th held-out-surface row: paraphrase 4, field-name variant 4 of every
    field, and per object type the held-out order j mod (its number of held-out orders)."""
    return {"paraphrase": HELDOUT_PARAPHRASE, "names": {f: HELDOUT_NAME_IDX for f in SD_FIELDS},
            "orders": {t: list(o[j % len(o)]) for t, o in HELDOUT_ORDERS.items()}}


def surface_messages(before: dict, ev: dict, surface: dict | None) -> list:
    """The one-step messages of (before, event) in a surface; None = Titan's verbatim frame (env.one_step_messages)."""
    if surface is None:
        return canonical_messages(before, ev)
    if tuple(sorted(ev)) != EVENT_KEYS or tuple(sorted(before)) != tuple(REGS):
        raise ValueError(f"unexpected prompt objects: {sorted(before)} {sorted(ev)}")
    nm = {f: SD_FIELDS[f][surface["names"][f]] for f in SD_FIELDS}
    system = SD_PARAPHRASES[surface["paraphrase"]].format(**{f: nm[f] for f in TEXT_FIELDS})
    o = surface["orders"]
    for t, keys in OBJECT_KEYS.items():
        if sorted(o[t]) != sorted(keys):
            raise ValueError(f"order of {t} is not a permutation of {keys}: {o[t]}")
    state = {r: before[r] for r in o["state"]}
    event = {nm.get(k, k): ev[k] for k in o["event"]}
    top = {"before": state, "event": event}
    user = json.dumps({nm[k]: top[k] for k in o["top"]}, separators=(",", ":"))
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_NAME_OF = {name: (f, i) for f, names in SD_FIELDS.items() for i, name in enumerate(names)}


def read_prompt(messages: list) -> dict:
    """Decode a one-step register prompt in ANY surface: the canonical (before, event) plus the surface it was rendered
    in (paraphrase index, field-name variant per field, key order per object type). Raises ValueError if the prompt is
    not a known surface. Used to PROVE which surfaces the training and held-out prompts use."""
    system, user = messages[0]["content"], messages[1]["content"]
    top = json.loads(user, object_pairs_hook=lambda p: p)
    names, orders = {}, {}

    def field(name, allowed):
        if name not in _NAME_OF or _NAME_OF[name][0] not in allowed:
            raise ValueError(f"unknown key {name!r}")
        f, i = _NAME_OF[name]
        if names.setdefault(f, i) != i:
            raise ValueError(f"field {f} named two ways")
        return f
    tk = [field(k, ("before", "event")) for k, _ in top]
    if sorted(tk) != ["before", "event"]:
        raise ValueError("top-level keys")
    orders["top"] = tk
    obj = {f: v for f, (_, v) in zip(tk, top)}
    orders["state"] = [k for k, _ in obj["before"]]
    before = {k: v for k, v in obj["before"]}
    ek = []
    for k, _ in obj["event"]:
        ek.append(k if k == "tick" else field(k, ("register", "other", "value", "threshold", "kind")))
    if sorted(ek) != sorted(EVENT_KEYS) or sorted(orders["state"]) != sorted(REGS):
        raise ValueError("event / state keys")
    orders["event"] = ek
    event = {r: v for r, (_, v) in zip(ek, obj["event"])}
    paras = [i for i, p in enumerate(SD_PARAPHRASES)
             if p.format(**{f: SD_FIELDS[f][names[f]] for f in TEXT_FIELDS}) == system]
    if len(paras) != 1:
        raise ValueError("system text is not a known paraphrase with these field names")
    surf = {"paraphrase": paras[0], "names": {f: names[f] for f in SD_FIELDS}, "orders": orders}
    if surface_messages(before, event, surf) != messages:
        raise ValueError("the prompt does not re-render from its decoded surface")
    return {"before": before, "event": event, "surface": surf}


def surface_heldout_parts(surface: dict) -> dict:
    """Which factors of a surface are HELD OUT: paraphrase, field names (per field), key order (per object type)."""
    return {"paraphrase": surface["paraphrase"] == HELDOUT_PARAPHRASE,
            "names": sorted(f for f, i in surface["names"].items() if i == HELDOUT_NAME_IDX),
            "orders": sorted(t for t, o in surface["orders"].items() if tuple(o) in HELDOUT_ORDERS[t])}


def _check_sd() -> None:
    canon = {f: v[0] for f, v in SD_FIELDS.items()}
    if SD_PARAPHRASES[0].format(**canon) != env.SYSTEM_ONE[FAM]:
        raise AssertionError("SD_PARAPHRASES[0] with canonical names must equal env.SYSTEM_ONE[register_machine]")
    allnames = [n for v in SD_FIELDS.values() for n in v]
    if len(set(allnames)) != len(allnames) or set(allnames) & ({"tick"} | set(REGS)):
        raise AssertionError("field-name variants must be globally distinct")
    if any(len(v) != 5 for v in SD_FIELDS.values()) or len(SD_PARAPHRASES) != 5:
        raise AssertionError("5 paraphrases and 5 names per field (4 = held out)")
    if _seeded_heldout_orders() != HELDOUT_ORDERS:
        raise AssertionError("HELDOUT_ORDERS differ from their seeded draw")
    for t, keys in OBJECT_KEYS.items():
        if keys not in TRAIN_ORDERS[t] or tuple(reversed(keys)) not in HELDOUT_ORDERS[t] or not TRAIN_ORDERS[t]:
            raise AssertionError(f"{t}: canonical order must train, reversed order must be held out")
    b, e = {"X": 1, "Y": 2, "Z": 3}, {"kind": "mix", "other": "X", "register": "Y", "threshold": 4, "tick": 5, "value": 6}
    if surface_messages(b, e, canonical_surface()) != canonical_messages(b, e):
        raise AssertionError("the canonical surface must render Titan's verbatim frame")
    if read_prompt(canonical_messages(b, e))["surface"] != canonical_surface():
        raise AssertionError("read_prompt(canonical)")
    for txt, h, k in ((GO_RULE_TEXT_REV1, GO_RULE_REV1_SHA256, 1), (GO_RULE_TEXT_REV2, GO_RULE_REV2_SHA256, 2),
                      (GO_RULE_TEXT_REV3, GO_RULE_REV3_SHA256, 3), (GO_RULE_TEXT_REV4, GO_RULE_REV4_SHA256, 4),
                      (GO_RULE_TEXT_REV5, GO_RULE_REV5_SHA256, 5), (GO_RULE_TEXT_REV6, GO_RULE_REV6_SHA256, 6)):
        if hashlib.sha256(txt.encode("utf-8")).hexdigest() != h:
            raise AssertionError(f"revision {k} of the go rule must stay verbatim")


# ---------------------------------------------------------------------------------------------- io


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(p: Path) -> list:
    return [json.loads(line) for line in Path(p).read_text(encoding="utf-8").splitlines() if line.strip()]


def read_json(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_dev(path: Path = DEV) -> list:
    return load_jsonl(path)


def load_dev_hs(path: Path = DEV_HS) -> list:
    return load_jsonl(path)


def load_chain_bank(path: Path = CHAINS) -> list:
    """Chain parents in the rcwmt bank format (parent_id, family, split, s0, candidates[cand_id, events])."""
    return load_jsonl(path)


def flat_chains(parents: list) -> list:
    return [{"family": p["family"], "s0": p["s0"], "events": c["events"], "cand_id": c["cand_id"], "stratum": p["stratum"],
             "parent_id": p["parent_id"]} for p in parents for c in p["candidates"]]


_TOK = []


def token_counter():
    """Titan's tokenizer through the `tokenizers` library (no torch): text -> number of ids, no special tokens. The tiny
    test checks it against transformers' AutoTokenizer (what the trainer uses) on every target."""
    if not _TOK:
        from tokenizers import Tokenizer
        _TOK.append(Tokenizer.from_file(str(TOKENIZER_JSON)))
    t = _TOK[0]
    return lambda s: len(t.encode(s, add_special_tokens=False).ids)


_check_sd()
