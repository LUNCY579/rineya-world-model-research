# PREREG-L1: promotion screen for L8-15 (Titan-Sparse's delta restricted to blocks 8-15)

**DRAFT, revision 4 (revision 2 after the adversarial review of 2026-10-08; revision 3 added amendment A-OOD,
criterion C6, for DECISIONS D-28(a), section 15; revision 4 applies the verification of A-OOD, section 13: no model
reads the OOD bank before the parity gate passes, and C6's expected binding strata are stated). Not frozen. No L1
output exists**: there is no
`runs/l1_screen/` directory and no model has been run on any L1 instrument for any arm. It is a development screen.
There is no training run, and the update comes from one training draw, the parent's. It can make L8-15 **eligible**
for promotion as an **experimental** core in the private vault, and nothing more. It cannot release anything, make a
quality claim, or confirm anything. Under D-26, nothing goes to the vault or to Hugging Face without the operator's
explicit go-ahead, even after a PROMOTE verdict.

Runner: `scripts/l1_screen.py`. Tiny-model test: `scripts/l1_test_tiny.py`, 92/92 checks pass. OOD bank builder:
`scripts/l1_make_ood.py`. Every number in this document marked "measured at drafting" comes from CPU checks that loaded
no model (section 10). Section 13 lists every review item and what was done with it.

---

## 0. The question, in one paragraph

R12 found that the transition skill of Titan-Sparse-Experimental (Supa 1.7 + the top-25% magnitude-pruned RRP-015
update) is computed in transformer blocks 8-15. Its delta in blocks 16-27, 59% of its energy, contributes nothing
measurable to that skill. R14 (D-23) then found that removing the 16-27 part keeps the skill and chained decision
regret within noise, while cutting off-task first-token KL drift against Supa to 0.68x [0.62, 0.74] of
Titan-Sparse's. Both results are exploratory and come from pilot instruments.

L8-15's only reason to exist beside its parent is this: **less off-task drift at no more than a registered loss of
the skill and of retention**. PREREG-L1 tests that claim under the evidence standard that promoted
Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze (SPARSE001_PREREG.md, PROMOTION.json), on instruments that played no
part in choosing the cut, with the deviations from that standard named in section 8.

## 1. Provenance, stated first

**The 8-15 cut was chosen after seeing pilot data.** The full forking path:

1. **The cut grid was fixed before R12; the choice of the cut was not.** D-21 fixed the hybrid grid {12, 16, 20, 24}
   before the patching run. R12 (RCWM pilot bank, TEST one-frame items, 640 items) then measured every prefix and
   suffix cut of that grid (FR, HI >= 0.7, LO <= 0.2):

   | delta kept in | ledger | prerequisite | register |
   |---|---|---|---|
   | 8-11 | 0.13 LO | 0.83 HI | 0.50 MID |
   | **8-15** | **0.85 [0.68, 1.02] HI** | **1.00 [0.76, 1.31] HI** | **0.97 [0.91, 1.03] HI** |
   | 8-19 | 0.96 HI | 1.00 HI | 1.01 HI |
   | 8-23 | 0.92 HI | 1.00 HI | 1.00 HI |
   | 12-27 | 0.56 MID | 0.91 HI | 0.73 HI |
   | 16-27 | 0.09 LO | 0.00 LO | 0.00 LO |
   | 20-27, 24-27 | LO | LO | LO |

   Three prefix cuts were HI in every family: 8-15, 8-19 and 8-23. **8-15 was chosen, implicitly, as the smallest
   cut that is HI in every family**, which minimises the retained delta. Of the three it has the lowest ledger FR.
   No criterion was written down before the choice.
2. **R14 measured the candidate before this rule existed.** D-23 built two arms, `loc8_15` and its complement
   `up16_27`, and measured them on the 248 off-task prompts of `data/offtask_prompts.jsonl` and on chain_one
   decisions on the pilot TEST plans. The 248 prompts were declared at birth "measurement only (representation shift
   / first-token KL); no training or selection". **D-26 went beyond that declaration**: R14's verdict on them,
   "localisation reduces interference", is what made L8-15 a candidate core. They are therefore candidacy-exposed,
   not merely "measured".
3. So **the pilot bank and the 248 prompts were used to choose the candidate.** No decisive endpoint (C1-C6) reads
   either. 228 of the 248 (section 4) appear in two roles only: a C0 harness check on the two **anchors'** log-probs
   (no L8-15 number enters it), and the report-only bridge ratio of P6.
4. **No further forking is allowed.** No other cut is screened under this registration and no refinement is allowed
   (for example 8-13, 10-15, 8-19, or per-family cuts). Any other cut needs its own registration, written before any
   of its numbers is read. This forbids further forks; it does not undo the fork that produced 8-15, which is why the
   candidate is labelled post hoc everywhere.

**What "why_this_is_not_post_hoc" can honestly say here.** SPARSE-001's answer was "the decision rule was fixed
before any arm was built". That sentence is **not available** to L8-15: the arm existed in memory (R12, R14) and was
measured before this rule was written. The defensible statement is narrower:

> The candidate was chosen post hoc from exploratory pilot data. The promotion decision is not post hoc. It rests
> only on instruments that played no part in choosing the cut, under a rule fixed before any of them was read for
> this arm. The bytes being judged are fixed in advance: sha256 `6282b456…`, recorded in `L8_15_EXPECTED.json`,
> and they are exactly R14's `loc8_15` arm, group hash for group hash.

**Inherited exposure, disclosed rather than corrected.**

- **Winner's curse, both directions.** Titan-Sparse was *selected* on SELECT-VIEWS-001 and the arithmetic/retention
  panel (SPARSE-001: k=0.25 chosen over 0.10 and 0.50), so its numbers there are optimistic. Two consequences:
  - L8-15 shares the parent's draw, so its *absolute* numbers inherit part of that optimism. This favours L8-15 on the
    absolute thresholds C1, C3 and the absolute half of C4.
  - On the items the parent was selected on, regression to the mean biases the paired contrast L8-15 minus
    Titan-Sparse *downward*. This makes C2 and the relative half of C4 conservative.
- **Parent training overlap.** 319 of the 384 SELECT-VIEWS-001 parents were trained by some historical arm (GATE_V2
  Amendment 1). This was disclosed and allowed for Titan-Sparse, and it applies unchanged here.
- **Confirmed abstention damage in the parent (absent from the vault record).** RCI-001 (`RCI_SEAL.json`, spent 1/1)
  *confirmed* that the k=0.25 delta damages literal-`UNKNOWN` abstention: 25/128 vs Supa's 40/128, 0 gained / 15
  lost, p = 6.1e-5. The decision-level abstention screen on HSKILL-VIEWS-002 shows no detectable change (70 vs 71 of
  112, p = 1.000). The vault's Titan-Sparse README still says that no confirmation instrument exists for abstention
  (gap G2). L8-15 keeps 41% of that delta's energy, and no unspent instrument is left to confirm its abstention
  (section 11; open question Q2).
- **The two retention gates still disagree, and no resolution is registered.** SPARSE001_RESULTS: the legacy
  per-class monitor "FAILS every arm ... the disagreement with Gate v2 needs a registered resolution before anything
  is selected on either". Titan-Sparse's PROMOTION.json lists that resolution as outstanding. **L1 follows the
  Titan-Sparse promotion precedent and selects on Gate v2 terms (C1, C3, C4) regardless.** The legacy monitor is
  report-only here, and it is expected to FAIL L8-15 as it fails every arm.

## 2. The artifact

**Construction.** `theta(L8-15) = theta_supa + 1[block in 8..15] * delta`, where `delta = theta_TitanSparse -
theta_supa`.

- Byte for byte: every tensor of blocks 8-15 is Titan-Sparse's stored FP32 tensor. Every other tensor is Supa
  1.7's stored BF16 value widened to FP32, a 16-bit shift that is exact.
- The construction rests on `delta` being exactly zero in the embeddings, the final norm and blocks 0-7 (D-21).
  Gate B2 re-verifies this element by element on every build.
- No training and no model call. Nothing is ever routed through BF16, for the reason Titan-Sparse's README
  measures: BF16 storage would zero 28.5% of the kept delta.

**Identity, measured at drafting** (`l1_screen.py build --dry-hash --record`; two streamed reads of the published
cores, no model):

| field | value |
|---|---|
| weight sha256 | `6282b45676fab8b2446d4363373b94931aea16c346ba769c284b81fcb0f97fe3` |
| bytes / tensors / dtype | 2,384,234,968 / 310 / all F32 |
| sources | Supa 1.7 `962abc5b…` (BF16), Titan-Sparse `a28629c9…` (FP32), both hash-verified (B1) |
| per-group FP32 hashes | equal to R14's `loc8_15` record, which equals the composition of R14's Supa and Titan-Sparse hashes (B4) |
| share of delta energy kept | 0.4128 (blocks 8-15: 5.9, 5.7, 5.4, 5.1, 4.9, 4.7, 4.8, 4.8%) |
| share of ||delta|| kept | 0.6425 |
| record | `docs/l1/L8_15_EXPECTED.json` (sha `76f20310…`) |

**Reuse of the RCWM splice utilities.** R14's arm was spliced in memory by `run_localized.py` via
`run_patch.set_block_sources`. The vault needs a file, so `l1_screen.py build` writes one by a streaming byte
transform, with no torch involved. It reuses `run_patch`'s group key and `run_localized.file_group_hashes`' digest
definition, byte for byte. Gate B4 binds the file to R14's recorded arm. The tiny test also checks the B4 hashes
against `run_patch.group_hashes` on an independently assembled model.

**Build gates (fatal).**

| gate | condition |
|---|---|
| B1 | input hashes equal the pins |
| B2 | the D-21 premise holds, element-exact |
| B3 | an independent second derivation and a re-read of the file from disk both equal the first pass |
| B4 | group hashes equal R14 `loc8_15` |
| B5 | sha equals the frozen expectation |
| B6 | the six sidecars are byte copies of Supa's |
| B7 | create-only: an existing arm is re-verified and never overwritten |

**Measured-arm layout.** The programme's convention for a measured arm: Supa's six byte-identical sidecars, and a
`MATERIALIZATION.json` receipt with `record=CHECKPOINT_MATERIALIZATION`, `source_weight_sha256` and
`out_weight_sha256`. A vault copy would differ in one sidecar, as Titan-Sparse's does: `config.json` would set
`torch_dtype: float32` (gap G10, section 12).

## 3. Arms

All three arms are measured in **one run, in one environment**, in FP32, in this order: Supa 1.7, then Titan-Sparse,
on every instrument except the OOD bank; then the anchor parity gate (C0, CPU); then, **only if the gate passed**,
Titan-Sparse and Supa on the OOD bank (C6) and L8-15 on everything. If the gate fails, L8-15 is never loaded and no
model reads the OOD bank. Stages may span processes through the per-stage
resume; every invocation must run the frozen runner in the frozen environment (python, torch, transformers,
tokenizers, safetensors, numpy, torch CUDA build, NVIDIA driver, GPU name), checked identical at every resume (G0),
and stages from two environments are never stitched.

| arm | role |
|---|---|
| Supa 1.7 | Anchor: re-measured and checked against its published raw predictions. Baseline for drift. |
| Titan-Sparse | Anchor and parent. It is the comparator for C2, C4 and C5. |
| L8-15 | Candidate. |

Every contrast between arms is paired within this run. Two kinds of number from another environment do enter, as
they did in SPARSE-001, and are named here:
- **the B28 source answers** (47 permissive / 41 strict), the fixed reference set for `references_retained` in C4.
  They were measured on the **CPU** (`G1_RETENTION_B28.json` context: device cpu, torch 2.10.0, transformers 4.51.3).
  This is the same reference set SPARSE-001 used for Titan-Sparse's 41;
- **the anchors' published values**, used only by the C0 parity gate (section 8).

## 4. Instruments and their roles

| instrument | n | role in L1 | permission (verbatim where a record exists) | read when choosing the cut? |
|---|---:|---|---|---|
| SELECT-VIEWS-001 acquisition (`7dd53f2a…`) | 384 (128 per family) | **SELECTION**: C1, C2; anchor parity (C0) | `BUILD.json`: `"view_role":"ADMITTED_FOR_SELECTION","admissible_uses":["SELECTION","DEV"],"inadmissible_uses":["TRAIN","CONFIRMATION"]`. No `QUARANTINE.json` entry exists in any copy (see "read budget" below). | **No.** It shares no (before, event) payload with the RCWM pilot bank: 0 of 15,629 payloads (integrity check) |
| Arithmetic panel (`5f6df1c1…`) + the 47 frozen source answers (B28 receipt `ec3fca1e…`) | 150 | **SELECTION**: C3, C4; Titan-Sparse parity (C0) | "the gate's own instrument" (SPARSE001_PREREG); in no registry; not SPENT | **No.** RCWM never read it |
| Fresh off-task prompts (`docs/l1/offtask_fresh_prompts.jsonl`, `38b964bf…`) | 456 (57 reply classes x 8) | **SELECTION**: C5 | Drawn for this screen (seed 20261009) from the v3 preservation slice (`5ee0c283…`, in no registry), with no model call. Excluded, in order: the 248 R14 prompts' ids (248 rows); **every id and user text of the HSKILL-002 source pools** (1,000 rows); every template family of the 248 (1,251 rows); text equal to one of the 248. The five SRB HSKILL classes (arithmetic_short_answer, clarification, knowledge_boundary, safety_boundary, uncertainty) have no row left and drop out; 62 - 5 = 57 classes. SOURCE record: "PREREG-L1 endpoint C5 (selection) and nothing else … Spent for confirmation by that use." | **No.** No model has read them |
| A-OOD rule-extrapolation bank (`docs/l1/ood_bank.jsonl`, `ac665c11…`) | 600 (6 strata x 100) | **SELECTION**: C6 (L8-15 vs Titan-Sparse per stratum); Supa report-only. Read by a model only after the parity gate has passed | Built for this screen (amendment A-OOD, section 15; seed 20261028) with no model call. SOURCE record: "PREREG-L1 endpoint C6 (selection) and nothing else … Spent for confirmation by that use." Zero payload and zero operative-content overlap with the historical set, every RCWM bank (pilot, confirm, confirm2, s4000), the T2 files and SELECT-VIEWS-001 | **No.** Fresh; no model has read it |
| Bridge: the 248 R14 prompts minus every HSKILL-002 pool row | 228 | **C0** harness check on the anchors only (their first-token log-probs against R14's stored ones); **REPORT-ONLY** bridge ratio (P6) | Candidacy-exposed (section 1). No L8-15 number from them enters C0-C6 | **Yes** (R14) |
| Long-Context Demonstration Suite (answer key `b95cf52c…`) | 256 | **REPORT-ONLY**, never in C0-C6 (anchor parity on it is reported, not gated) | `QUARANTINE.json` (RRP2_QUARANTINE): `{"key":"Rineya_Long_Context_Demo_v1/all","bars":["TRAIN","SELECTION","CONFIRMATION"],"reason":"SPENT","sha256":"b95cf52c…"}`. The live answer key hashes to the same value. GATE_V2_PREREG: "A gate is a selection instrument by definition." | No |
| Literal `UNKNOWN` (the 64 `unanswerable` rows inside that suite) | 64 | **REPORT-ONLY**, never in C0-C6 | Same bank, same bar. "not a separate instrument and carries no separate permission" (GATE_V2_PREREG) | No |
| Arithmetic strict (as an endpoint); legacy per-class monitor (`protocol.json` G1_RETENTION) | 150 | **REPORT-ONLY** endpoints (arithmetic strict is also a Titan-Sparse parity statistic in C0) | Reported, never selected on, as in SPARSE-001 | No |

**Is any selection instrument spent?** No. SELECT-VIEWS-001 is selection-exposed: INT-001 and SPARSE-001 selected on
it. Its build record grants SELECTION and DEV. GATE_V2_PREREG states the registry's logic for such a split: entries
barred only `["TRAIN", "CONFIRMATION"]` with reason `SELECTION_EXPOSED` "are selection-*permitted*, because a split
already spent on selection takes no further harm from selection". **No fresh acquisition bank is therefore needed,
and none is registered here.** Two qualifications, disclosed:

- **No read budget exists.** GATE_V2 requirement 5 is titled "A registered read budget". Its content requires a
  `QUARANTINE.json` entry (`bars: ["TRAIN", "CONFIRMATION"]`, reason `SELECTION_EXPOSED`) "at the moment of first
  selection use — not retroactively". That entry was never written, and no numeric budget exists. L1 is one more
  unbudgeted selection read (gap G1; open question Q5).
- **It is a go/no-go detector, not a ranking instrument.** `ACQUISITION_BAND_NOT_RESOLVABLE_20260919.md`:
  "SELECT-VIEWS-001 is a **go/no-go detector** … **It is not a ranking instrument**". C1 uses it as a detector (floors).
  C2 uses it as a paired within-draw contrast, which the same note says remains valid: transform comparisons share
  their trajectory, so "their paired contrast is valid at its own discordance".
- **Fallback if the reviewer or operator rules otherwise** (for example, that two prior selections plus 319/384
  trained parents disqualify it): a fresh one-step bank in SELECT-VIEWS format built by the programme's own builder
  (`select-views-001/BUILD.json` lineage), with payload exclusion against SELECT-VIEWS-001, every RCWM bank (pilot,
  confirm, confirm2), T_BANK and the HSKILL banks, and floors published in its own FLOORS file before any model call.
  It would be its own amendment, frozen before any read. It is not part of this draft.

**Why the HSKILL-002 pools are excluded from every L1 prompt set.** HSKILL-CONFIRM-002 is
`{"barred":["TRAIN","SELECTION","DEV"],"status":"UNSPENT_AND_UNREAD","reason":"reserved for a single confirmation
decision"}`. Both HSKILL-002 banks were drawn from the same source pools (their build records carry identical
`sources` and `source_sha256`). The schema-1 fresh draw (`62faf04d…`, 496 prompts, never read by any model) contained
40 pool rows and 2 HSKILL-VIEWS-002 rows, so a C5 read would very probably have touched a CONFIRM item. The draw was
deleted and redrawn outside the pools, which guarantees disjointness from both banks **without opening the CONFIRM
file**. The bridge applies the same exclusion (248 -> 228). Whether R14 itself already read CONFIRM items among the 20
excluded rows can only be settled by a hash-only check against `HSKILL_CONFIRM_V2.jsonl`; that needs the operator's
permission (open question Q7).

**Not used, and why.**

| instrument | status (verbatim from its record) | why not used |
|---|---|---|
| RCWM pilot bank | — | chose the cut |
| RCI-001 calibration bank (128 rows) and RCI-002 calibration bank (256 rows) | `RCI_SEAL.json`: `"status": "SPENT for confirmation — it has been scored", "remains_admissible_for": ["DEV", "SELECTION"]` | Selection-admissible, and RCI-001's measures exactly the parent's confirmed literal-UNKNOWN damage (Supa and k=0.25 anchor scores exist). Not in this draft; a report-only use is open question Q2 |
| RCI-001 / RCI-002 sealed confirmation banks; T_BANK_V4; HSKILL_SOUP_V3 | spent | spent |
| HSKILL-CONFIRM-002 | `UNSPENT_AND_UNREAD`, barred TRAIN/SELECTION/DEV | the programme's last unspent confirmation bank, operator-reserved |
| HSKILL-VIEWS-002 (224 rows) | `UNSPENT`, `ADMITTED_FOR_SELECTION`, barred TRAIN/CONFIRMATION | selection-admissible abstention bank. Not in this draft; open question Q2 |

## 5. Decision rule (fixed now; verbatim as registered in `l1_screen.py`, `DECISION_RULE`)

> L8-15 is eligible for promotion as an experimental core - promotion itself still requires the operator's explicit
> go-ahead (D-26) - if and only if, in the L1 run, ALL of the following hold. C0 identity and comparability, settled
> before any L8-15 output exists: the measured L8-15 weights are the frozen bytes (sha256 in L8_15_EXPECTED.json;
> per-group FP32 hashes equal to R14's loc8_15 arm); every build and load gate passed; the forward path is the
> programme's DenseReference (vendored from pilot commit 5b4c22d); the run used the frozen runner and the environment
> recorded in the freeze, in every invocation; the full decisive instruments were run (no row subset) and no decisive
> prompt was truncated or skipped; the L1 scorers reproduce every published per-row verdict and count on the anchors'
> published SELECT-VIEWS-001 and arithmetic-panel predictions, the L1 tokenizer and decoder return every one of those
> published texts unchanged (encode, decode, strip), and every arithmetic prompt fits the registered 512-token
> context; and the anchor parity gate, evaluated after both anchors and before L8-15 is loaded, passes: Supa 1.7
> reproduces its published SELECT-VIEWS-001 texts on all but max(3, floor(3% of n)) rows, its canonical total within
> +-3 and its floor verdicts identically; Titan-Sparse does the same and also reproduces arithmetic_permissive and
> arithmetic_strict each within +-2 and references_retained within +-1 of its published 72, 63 and 41, its correct set
> on the 47 permissive and 41 strict source-correct rows up to 1 row per policy, its per-class arm_correct and
> terminated counts within +-1, and its 24 published arithmetic samples in text and termination on all but 3; and both
> anchors agree with R14's stored first-token log-probabilities on the 228 bridge prompts (top-1 agreement >= 0.99,
> mean KL(L1||R14) < 0.001 nats). If the anchor parity gate fails, L8-15 is never loaded and no arm generates on the
> OOD bank. C1 acquisition floors (GATE v2 term 3): on SELECT-VIEWS-001 every family's two-sided exact binomial 95%
> lower bound on canonical_correct exceeds its published binding floor (ledger_replay 0.2344, prerequisite_plan
> 0.0781, register_machine 0.0703). C2 acquisition non-inferiority, per family, paired within the run against
> Titan-Sparse on canonical correctness: in every family the net change (gained - lost) is >= -8 (ledger_replay), >=
> -5 (prerequisite_plan), >= -6 (register_machine), and no family shows a detected loss (lost > gained with a
> two-sided exact McNemar p < 0.05). C3 arithmetic (GATE v2 term 1): arithmetic_permissive(L8-15) >= 62 of 150. C4
> source answers retained (GATE v2 term 2 plus non-inferiority): references_retained(L8-15) >= 37 of 47 AND >=
> references_retained(Titan-Sparse, same run) - 2. C5 off-task drift on the 456 fresh prompts: the paired
> percentile-bootstrap 95% upper bound (10,000 resamples over prompts, seed 20261009) of mean first-token
> KL(L8-15||Supa) / mean KL(Titan-Sparse||Supa) is < 0.80. C6 out-of-distribution rule extrapolation (amendment A-OOD,
> D-28(a)), paired within the run against Titan-Sparse on the 600-row OOD bank (docs/l1/ood_bank.jsonl, sha256
> recorded in ood_bank.SOURCE.json and pinned in the runner; built with no model call from data that played no part in
> choosing the cut: six strata of 100 one-step transitions in Titan's trained one-step frame, L_big, L_void, R_wrap,
> R_boundary, P_inapp and P_appdel), every arm run on it only after the anchor parity gate passed, greedy with at most
> 96 new tokens, and a row correct iff rcwmt.env.parse_state(family, prediction, lenient=True) equals the oracle
> state: in every stratum the net change (gained - lost) is >= -max(3, ceil(5% of n)), which is -5 at n = 100, and no
> stratum shows a detected loss (lost > gained with a two-sided exact McNemar p < 0.05). Supa's OOD scores are
> report-only. If C0 fails the run is NON-COMPARABLE: no verdict, nothing promoted, no L8-15 endpoint computed, and
> this registration ends (a rerun is a new registration with a fresh drift draw that may not reuse these prompts, and
> with a fresh OOD bank if any arm has generated on this one). If C0 holds and any of C1-C6 fails: DO NOT PROMOTE.
> Long context and literal UNKNOWN (including every parity and scorer check on that SPENT bank), Supa's arithmetic
> against its CPU-measured source answers, the legacy per-class monitor and the bridge drift ratio are report-only and
> never enter C0-C6. No threshold may change after any L1 output exists.

C1, C3 and the absolute half of C4 are **Gate v2 applied to L8-15**, the standard Titan-Sparse passed. C2 and the
relative half of C4 ask that L8-15 be **not worse than Titan-Sparse by more than the registered margins**. C5 asks
that its reason to exist, less drift, replicate on prompts that played no part in choosing it. C6 (amendment A-OOD,
section 15) asks that the cut not have removed something that only matters **out of range**: on six strata of
one-step transitions outside the distribution that chose the cut, L8-15 may lose at most 5% of a stratum (net), and
never a loss the paired test can see.

**One margin logic for C2 and C4.** Both are paired, within-run contrasts of a deterministic transform against its
parent, so the cross-draw floor of `ACQUISITION_BAND_NOT_RESOLVABLE` (|net| >= 18-21) does not apply to them (that
note says so for transform comparisons). Each leg fails if the loss is either (a) **detected** by the paired exact
test (lost > gained, two-sided exact McNemar p < 0.05), or (b) larger than **half the parent recipe's own cross-draw
span** on that endpoint, from REP-001's three independent FP32 controls of the RRP-015 recipe (seeds 17 / 23 / 29):

| endpoint | REP-001 controls (s17, s23, s29) | span | margin = floor(span / 2) |
|---|---|---:|---:|
| ledger_replay /128 | 54, 71, 61 | 17 | **8** |
| prerequisite_plan /128 | 95, 85, 93 | 10 | **5** |
| register_machine /128 | 85, 73, 74 | 12 | **6** |
| (aggregate /384, not gated) | 234, 229, 228 | 6 | — |
| references_retained /47 | 40, 38, 36 | 4 | **2** |

These margins are **value judgments, not noise floors**: "the cut may cost less than half of what retraining the
parent moves that endpoint, and never a loss the paired test can see". For C4 the detection clause is implied by the
margin (the smallest detectable net at p < 0.05 is -6: 6 lost, 0 gained), so C4 is written with the margin alone.
C2 is **per family**, as Gate v2 requires ("Acquisition continues to be reported per family, never as an
aggregate"); the aggregate is reported, never gated. Under R12's projection (ledger about -6, section 7) the
detection clause is a real risk for ledger_replay: a net of -6 fails if it comes with zero gains (6 lost, 0 gained,
p = 0.031), and a net of -7 fails unless at least 2 views are gained.

| leg | threshold | provenance / reason |
|---|---|---|
| C0 parity | section 8 | The 4.51.3 harness is gone (gap G5). The programme's own LONGCTX note says to "compare against a base-model run from this same script". The anchors re-measured here prove that this harness reproduces the programme's instrument, before L8-15 is loaded |
| C1 floors | CP lower bound > 0.2344 / 0.0781 / 0.0703 | `FLOORS.json`, published before any model call. "The CP lower bound is the rule". At n = 128 the bound is met from 41 / 17 / 16 correct |
| C2 acquisition | per family: net >= -8 / -5 / -6 and no detected loss | the margin logic above |
| C3 arithmetic | >= 62 / 150 permissive | Gate v1/v2 term 1 (ADJUDICATION_001 D3; `bare_policy: permissive` as registered) |
| C4 retained | >= 37 of 47 and >= Titan-Sparse - 2 | Gate v2 term 2, plus the margin logic above |
| C5 drift | upper 95% CI of the KL ratio < 0.80 | D-23's own criterion, unchanged, fixed in `docs/localized/DESIGN.md` before R14 together with KL(arm‖Supa) as the primary direction. Replicated on 456 prompts that played no part in the candidacy |
| C6 OOD | per stratum (6 x 100): net >= -max(3, ceil(5% of n)) = -5 and no detected loss | D-28(a): non-inferiority to the paired parent out of range (section 15). A value judgment, like C2's margins; no cross-draw span exists for fresh rows |

**Fragile outcomes (report-only).** Any C1-C4 or C6 outcome within the matching anchor-parity tolerance of its threshold
(acquisition +-3 per family count or per family net, and per OOD stratum net; arithmetic +-2, retained +-1) is labelled FRAGILE in the results:
the harness change is tolerated up to that size on the anchors, so such an outcome could plausibly flip under the
programme's original harness. The label does not change the verdict.

**Scoring definitions (unchanged from the programme).**

| endpoint | definition |
|---|---|
| acquisition | `canonical_correct`: `json.loads(prediction)` re-serialised with `sort_keys=True, separators=(',',':')` equals the reference. Greedy, max_new_tokens 96 |
| arithmetic | `by_bare_policy.permissive.arm_correct_total` from the vendored typed scorer `retention.py`. Max 24 new tokens, prompts <= 512 tokens, truncation forbidden |
| references_retained | `permissive.source_correct_total - lost_total`, against the frozen B28 source answers (47). Gains never offset losses |
| drift | per prompt, KL between the arms' full first-answer-token distributions (log-softmax of FP32 logits, stored float16 and clamped at -60). Storage exactly as in R14; forward path DenseReference, batch 1, no padding (R14 used HF Qwen3 sdpa, batch 8, left padding; the C0 bridge leg checks the two agree). Ratio of means with a paired percentile bootstrap over prompts (`localized_analysis.ratio`) |

## 6. Reported regardless of the verdict

For all three arms (for the two anchors only if the parity gate fails, and then without OOD scores: no model reads
the OOD bank before the gate passes), labelled report-only where section 4 says so:

- **Acquisition.** Strict and canonical, per family, with CP intervals and floor verdicts; paired exact McNemar of
  L8-15 vs Titan-Sparse, per family (C2's input) and in aggregate (report-only). Every acquisition claim is reported
  as (net, discordant, exact p), never as a bare difference of totals (`ACQUISITION_BAND_NOT_RESOLVABLE` forward rule).
- **Arithmetic.** Permissive and strict, never pooled; paired McNemar of L8-15 vs Titan-Sparse.
- **Retention.** Retained (permissive and strict) and the lost-row ids, the legacy per-class monitor under both
  policies, and Gate v2 per arm. Report-only: each arm's losses against the **same-run CUDA Supa** (paired McNemar
  on the 150 rows, both policies), next to the registered losses against the CPU-measured B28 source.
- **Long context /256.** Showcase and stability; exact McNemar of each arm against the same-run Supa, of L8-15
  against Titan-Sparse, and of each arm against the published Supa (`LONGCTX_supa.json`, 161). Literal `UNKNOWN`
  under **both** counters: the substring counter `'unknown' in normalize(prediction)` (SPARSE-001's number) and
  `prediction.strip() == "UNKNOWN"` (gap G4); false abstentions on answerable rows. A long-context row that does not
  fit the 4096-token context is recorded as skipped (never truncated); that labels these report-only numbers
  non-comparable and never enters C0.
- **Drift, fresh and bridge.** KL(arm‖Supa) and KL(Supa‖arm) means and ratios with CIs; top-1 first-token flips and
  their ratio; concentration flags (Titan-Sparse flips fewer than 10 prompts; the top 10% of prompts carry more than
  half of the KL).
- **The full parity gate record** (`PARITY_GATE.json`): every decisive leg and every report-only parity row, and the
  scorer parity checks.
- **Registered predictions** (section 7), each marked held or FALSIFIED; **fragile flags** (section 5).
- **OOD rule extrapolation (C6).** Exact-state counts per stratum for all three arms, with parsed and terminated
  counts; paired exact McNemar of L8-15 vs Titan-Sparse per stratum (C6's input) and over all 600 rows (report-only);
  each arm vs Supa per stratum (report-only); and, per stratum, a report-only **binding flag**: whether C6 can fail
  there at all. L8-15 can lose only rows Titan-Sparse got right, and at n = 100 the smallest failing loss is 6 lost and
  0 gained, so a stratum where Titan-Sparse is correct on fewer than 6 rows is marked "cannot fail" (section 15).
- **SPARSE-001's selection rule** applied to {Titan-Sparse, L8-15}, report-only: floors 3/3 and arithmetic >= 62,
  then maximise retained, ties broken by acquisition.

## 7. Registered predictions (scored, never decisive)

P3, P7 and P8 are **no-loss predictions with a tolerance** equal to the instrument's parity tolerance. Equality is
their expected value; "held" can never be read as an improvement. (AMENDMENT_001 recorded the weakness of a bare
">=": "A minimum effect size should have been registered".)

| id | prediction |
|---|---|
| P1 | L8-15 clears all three SELECT-VIEWS-001 family floors. |
| P2 | Per-family net (L8-15 - Titan-Sparse): ledger_replay in [-12, 0], prerequisite_plan in [-4, +4], register_machine in [-6, +2]; aggregate in [-16, +2]. |
| P3 | No retention loss beyond 1: references_retained(L8-15) >= references_retained(Titan-Sparse) - 1. |
| P4 | arithmetic_permissive(L8-15) >= 62. |
| P5 | The fresh-prompt drift ratio's point estimate lies in [0.60, 0.76]. R14 measured 0.680 [0.617, 0.735] on its 248. |
| P6 | The bridge drift ratio (228 prompts, this harness) is within +-0.05 of R14's 0.6772 on the same 228 (recomputed from R14's stored arrays by `integrity`). |
| P7 | No long-context loss beyond 3 (report-only): long_context(L8-15) >= long_context(Titan-Sparse) - 3. |
| P8 | No abstention loss beyond 1 (report-only): UNKNOWN substring(L8-15) >= UNKNOWN substring(Titan-Sparse) - 1. |
| P9 | Both anchors reproduce their published per-row predictions on >= 99% of rows of every instrument, decisive and report-only alike (Supa's panel texts against the CPU-measured B28 included). |
| P10 | Both anchors' bridge log-probabilities agree with R14's within mean KL 1e-4 nats (10x inside the C0 bound). |

**Projection behind P1/P2.** It applies R12's hybrid FRs (0.85 / 1.00 / 0.97) to Titan-Sparse's published per-family
gains over Supa on SELECT-VIEWS-001. Titan-Sparse scored 53 / 96 / 83 and Supa 14 / 0 / 0, so the projection for
L8-15 is 47 / 96 / 80.5: nets -6 / 0 / -2.5, about -8 in aggregate.

- ledger_replay needs 41 to clear its floor; at R12's lower CI bounds (0.68 / 0.76 / 0.91) the projection is
  40.5 / 73 / 75.5, which fails C1 by a hair on ledger_replay and fails C2 in two families.
- **C1 and C2 are therefore real risks, not formalities.** C2's ledger leg is the tightest (section 5).
- C3 is the least predictable leg. The arithmetic panel measures transfer, not retention, and R12 says nothing about
  where Titan-Sparse's +25 arithmetic gain lives.

## 8. Comparability: why this harness is the programme's instrument (C0)

**The matched environment no longer exists (gap G5), and this is a named deviation.** SPARSE-001's battery ran in the
pilot clone `ChatBubble-retention-pilot-20260917` with `.venv-rab001-cu126` (Python 3.11.15, torch 2.10.0+cu126,
transformers 4.51.3). Both are deleted, and no environment on this machine has transformers 4.51.3. L1 runs in the
hermes venv (Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, tokenizers 0.22.2, safetensors 0.8.0, numpy
2.4.6), the runtime of R12/R14 and CONFIRM-002; the freeze records it, and G0 refuses any other.

**Deviation from the precedent's stop rule.** SPARSE001_PREREG: "Any arm whose metadata differs is reported
non-comparable rather than compared" (transformers 4.51.3, digest a329eb7bb0). **Under that literal rule this run
would be non-comparable.** L1 registers the anchor parity gate below as its replacement: comparability is earned by
re-measuring both anchors in this harness, not assumed, and the absolute legs C1, C3 and the absolute half of C4 are
judged in this harness. The parity tolerances (acquisition +-3, arithmetic +-2, retained +-1) are as large as typical
threshold margins; outcomes inside them are flagged FRAGILE (section 5).

1. **Forward path.** The programme's own `DenseReference` (`model.py`), its key map and its official-shape check are
   byte copies of files covered by digest `a329eb7bb0` at pilot commit `5b4c22d`, fetched with read-only
   `git cat-file`, each git blob id re-derived from the bytes (`docs/l1/vendor_5b4c22d/VENDOR.json`), hash-checked
   before import. Transformers is used only for the tokenizer and the chat template (`enable_thinking=False`,
   `add_special_tokens=False`). The greedy loop is the programme's: batch 1, KV cache, argmax, the official EOS list
   [151645, 151643], then `decode(skip_special_tokens=True).strip()`. Weights are streamed to the device as FP32; TF32
   is off and `float32_matmul_precision` is `highest`.
2. **Scorers.** The typed arithmetic scorer, `monitor_verdict` and `build_overlay` are the vendored `retention.py`.
   The acquisition, long-context and McNemar code re-implements `process_probe.py`, `long_context_probe.py` and
   `tabulate_sparse.py`; the retention leg re-implements `run.py` (`_probe_ids`, `_greedy_probe`, `_eos_ids`,
   `_score_probe`, `_retention_classes`, `load_probe_rows`) and the loader re-implements `hf_checkpoint.load_hf_candidate`
   -> `run._load`. `run.py` and `hf_checkpoint.py` at `5b4c22d` are vendored as **reference-only** byte copies, never
   imported (`VENDOR_REFONLY.json`, blob ids re-derived). **Measured at drafting** (`l1_screen.py parity-cpu`, 17/17:
   11 decisive, 6 report-only because they touch the SPENT long-context bank), on the *published* raw predictions:
   - decisive: Titan-Sparse 232/384 and 3/3 floors with identical CP bounds; Supa 14/384 and 0/3; the B28 source
     answers re-score to exactly 47/41; `monitor_verdict` reproduces Titan-Sparse's FAIL; the 24 published
     Titan-Sparse arm samples score consistently with its published lost ids under both policies; Titan-Sparse's
     FRONTIER_DET row (232, 3/3, 72, 63, 41, Gate v2 PASS);
   - report-only: 164/256 (43/121) and UNKNOWN 12 (substring and literal) for Titan-Sparse, 161/256 (41/120) and 13
     for Supa; McNemar 7 lost / 10 gained, p = 0.6291; FRONTIER_DET's 164 and 12.
3. **Tokenization and decoding.** **Measured at drafting** (decisive): every arithmetic prompt fits the registered
   512-token context (maximum 204); all 942 published anchor texts on SELECT-VIEWS-001 and the arithmetic panel
   survive encode -> `decode(skip_special_tokens=True)` -> strip unchanged. Report-only: this environment's tokenizer
   and chat template reproduce all 256 recorded long-context input-token counts row by row (0 differ; the only input
   counts the programme recorded, and counts, not ids), and the 512 long-context texts survive the round trip. The
   decisive end-to-end tokenization check is the anchors' own SELECT-VIEWS-001 texts in the parity gate.
4. **The anchor parity gate (C0), evaluated after both anchors and before L8-15 is loaded.** `run` measures Supa and
   Titan-Sparse on every instrument except the OOD bank, then scores their saved outputs on the CPU. The gate never
   reads OOD outputs; none exist yet. Text allowance: all but max(3, floor(0.03 n)) rows
   (384 -> 11, 256 -> 7, 150 -> 4, 24 -> 3).

   | leg | decisive (C0) | report-only |
   |---|---|---|
   | Supa, SELECT-VIEWS-001 | texts (allowance 11), canonical total 14 +-3, floor verdicts identical | — |
   | Titan-Sparse, SELECT-VIEWS-001 | texts (allowance 11), canonical total 232 +-3, floor verdicts identical | — |
   | Titan-Sparse, arithmetic panel (CUDA vs CUDA: both on the RTX 4070) | permissive 72 +-2, strict 63 +-2, retained 41 +-1; on the 47 permissive and 41 strict source-correct rows the correct set equals (source-correct minus the published lost ids) up to 1 row per policy; per class, arm_correct and terminated within +-1; the 24 published samples equal in (text, terminated) on all but 3 | — |
   | Supa, arithmetic panel vs B28 | — | texts and 47/41: B28 is a **CPU** measurement, so a CPU-to-GPU FP32 argmax flip here is not the environment change C0 tests; retention comparability rests on Titan-Sparse's CUDA-vs-CUDA leg |
   | Supa and Titan-Sparse, bridge log-probs (228 prompts) vs R14's stored arrays for the same arm | top-1 agreement >= 0.99 and mean KL(L1 ‖ R14) < 0.001 nats, each anchor | — |
   | Supa and Titan-Sparse, long context | — | texts (allowance 7), total +-3, UNKNOWN substring and literal +-1 (published 12 / 12 and 13 / 13) |

   The bridge leg is C5's comparability leg: it checks that the drift forward path (DenseReference, batch 1)
   reproduces R14's (HF sdpa, batch 8, left padding) on the very arrays R14's ratio came from. It uses no L8-15 number.
5. **If the gate fails**: `PARITY_FAIL.json` is written (create-only), the lease is released, L8-15 is **never
   loaded**, **no model reads the OOD bank** (the anchors' OOD stage, like every L8-15 stage, runs only after a passed
   gate), and `analyze` issues NON-COMPARABLE without computing any L8-15 endpoint or any OOD score. It refuses to
   analyse a directory that holds OOD outputs next to `PARITY_FAIL.json`. **This registration then ends.** A rerun in
   a matched environment would be a new registration with its own fresh drift draw, which may not reuse the 456
   prompts (open question Q4). After a gate failure the OOD bank is still unread. If C0 fails later instead (in
   `analyze`, after every arm has generated on the bank), the rerun also needs a fresh OOD bank, as the rule says. No
   threshold is relaxed.
6. **Binding the run to the freeze.** `analyze` refuses (no results) unless every invocation and every stage was made
   after `frozen_utc`, by the frozen runner, under the freeze record, without `--no-freeze`, in the frozen
   environment. `--no-freeze` is refused for any real core and for a real GPU, and `freeze` refuses if any L1 output
   exists anywhere under `runs/`. Every OOD stage and every L8-15 stage records the gate it followed; a stage
   without a passed gate on record is a C0 failure.

## 9. Execution and GPU plan

**Order of operations** (CPU unless marked):

| # | step | command | notes |
|---|---|---|---|
| 1 | Review and approval | — | This revision, including the D-28(a) amendment A-OOD (section 15, Q9) and its verification (section 13), then the operator's approval. |
| 2 | Freeze | `l1_screen.py freeze --approved-by <operator>` | Writes `docs/l1/PREREG_L1_FREEZE.json`, create-only, with the sha256 of this file, the runner, the tiny test, the vendored code and both manifests, the fresh prompts and their SOURCE, `L8_15_EXPECTED.json`, the queue hook draft, the OOD bank, its SOURCE record, its builder and `rcwmt/env.py`; every pinned input hash; the registered constants and predictions; the environment. It refuses if any L1 output exists anywhere under `runs/`. |
| 3 | Queue | — | Read `runs/bank_pilot/master_log.txt` first: `master_chain.sh` sources hooks only when it reaches its hook loop, after the PREREG-2 steps. If the running chain has passed it, the hook runs only after a relaunch at a safe point. Then copy `docs/l1/queue_hook_l1.sh.draft` byte for byte to `scripts/queue_hooks/50_l1_screen.sh`. The queue stays the single RCWM GPU waiter (R14 one-waiter rule). |
| 4 | Build | `l1_screen.py build` | Writes `runs/l1_screen/arm-l8_15`: two derivations of 3.6 GB input plus a re-read of the 2.38 GB output, about 2-5 min. No model. |
| 5 | Run, GPU | `l1_screen.py run --device cuda` | See below. Exit 0 = done or stopped by the parity gate; 3 = deterministic failure, never retried; 5 = resumable host stop, one retry. |
| 6 | Analyze | `l1_screen.py analyze` | Once, create-only: `runs/l1_screen/L1_RESULTS.{json,md}`. |

**The run, step by step.** A create-only run lock (`runs/l1_screen/.run.lock`, pid and process start time) refuses a
second process on the directory; a dead owner's lock is reclaimed. G0 checks the environment and the runner. G1 hashes
the three weight files (about 8 GB sequential read, 1-3 min) before anything else. Then, **with nothing held**, the
host gate waits (up to 2 h, then exit 5) for >= 6 GiB available RAM and a live watchdog (`runs/thermal_log.txt`
written within 3 min). Then one lease, `rcwm-t1:l1screen`, through `rcwmt.runtime.gpu_lease` (turn alternation, the
`.thermal_pause` flag, the dedicated-memory holder check, then `wait_for_resources`: free VRAM >= 3.6 GiB and host
RAM >= 2 GiB, read once the lease is held). Pending stages are re-derived after the lease is acquired; a directory
finished by another process meanwhile loads nothing. Before each load: the pause flag clear, a live watchdog, RAM >=
5 GiB within 5 min (else exit 5, releasing the lease); then a streamed FP32 load and G2. The passes, in order, each
stage with a progress line every 50 rows:

1. Supa: acq -> ret -> lc -> drift_fresh -> drift_bridge (load 1).
2. Titan-Sparse: the same five stages (load 2).
3. The anchor parity gate (CPU) on those outputs. If it fails, the run stops here: L8-15 is never loaded and no model
   has read the OOD bank.
4. Titan-Sparse: ood. The model stays loaded across the gate, so this needs no load.
5. Supa: ood (load 3, a reload).
6. L8-15: all six stages, ood last (load 4).

A resumed run keeps this order: the gate is re-evaluated before any pending OOD or L8-15 stage, and every such stage
records the gate it followed.

**Thermal and host safety.**
- `runtime.breathe()` after every row (the duty cycle of DECISIONS I-10..I-13; it blocks on the watchdog flag); a
  60 ms pause and a flag check every 8 decode steps.
- **No process is spawned while waiting or running (I-27).** The lease's `tasklist` and PowerShell `Get-Counter`
  helpers (2-3 processes per 20 s poll while waiting) are replaced in this process by psutil and an in-process PDH read
  of the same counter (`\GPU Process Memory(*)\Dedicated Usage`), with the I-17 semantics unchanged (a foreign
  python.exe with > 256 MiB dedicated GPU memory still blocks); `runtime.gpu_temp_c`'s `nvidia-smi` spawn is replaced
  by the in-process NVML read of `scripts/sensors_noproc.py`. Remaining spawns: only fallbacks if PDH or NVML fail,
  each at most once per 2 min (PowerShell) or per 30 s (nvidia-smi). The tiny test waits on a held lock with
  `subprocess` patched to raise and observes zero spawns.
- While running: a watchdog silent for 3 min, or a pause flag with RAM < 5 GiB for 15 min, ends the process with exit
  5 and releases the lease, so the job never holds the card through a RAM-caused pause (I-23, I-26).
- **Host footprint.** The load transient is about 1 GB (streamed, against the programme loader's 4.8 GB). The process
  also carries the CUDA context's commit: the comparable D-23 process showed 7.85 GB private against a 1.32 GB working
  set (I-26). The gates use the watchdog's own thresholds (pause < 3 GB, RAM-caused resume >= 5 GB). VRAM: weights
  2.38 GB plus at most about 0.5 GiB of attention transient (longest long-context row 1,556 tokens), about 3.2 GiB,
  inside the 0.70 memory-fraction cap.

**GPU cost** (`l1_screen.py plan`). Rates come from the programme's own `probe_terminal.json` (SPARSE-001/REP-001
batteries, RTX 4070 Laptop, FP32 DenseReference). Each arm runs 1,390 generations (790 + the 600 A-OOD rows of C6)
and 684 forward-only prompts. The OOD rows are priced at the acquisition rate (same families, same one-step frame,
same decoder and cap) scaled by the answer-length ratio (OOD gold 18.5 vs SELECT-VIEWS-001 references 18.0
characters, x1.03). One I-24 single job: one lease, three arms (the D-22 4.5 h cap is scoped to T2 pairs and does not
apply; the high band below approaches it anyway).

| | low duty (<70 C) | expected (70-74 C) | high (>=75 C) |
|---|---:|---:|---:|
| GPU hours, 3 arms, one lease (revision 4: C6, the anchors' OOD stage after the gate, 4 loads) | 1.39-1.66 | **2.14-2.58** | 3.26-4.02 |
| (revision 3: C6 with the anchors' OOD stage before the gate, 3 loads) | 1.37-1.64 | 2.12-2.56 | 3.24-4.00 |
| (revision 2, without C6, for reference) | 0.77-0.94 | 1.16-1.42 | 1.75-2.21 |

- Compute alone is 922-1,154 s per arm, of which the OOD stage is 435-523 s. In-lease overhead is 300 s: 4 streamed
  loads at about 60 s plus 15 s for G2 each. Putting the anchors' OOD stage after the gate costs one extra load
  (Supa's reload), about 75 s or 0.02 h.
- If the gate fails, neither L8-15 nor either anchor's OOD stage runs, so about two thirds of the compute is saved.
- **Not included:** watchdog pauses (I-25/I-26 stalled jobs 7-30 min), waits for the peer's turn and the lease, and the
  pre-lease host gate.
- Disk: 2.81 GiB (the 2.38 GB arm plus 0.58 GiB of float16 log-prob arrays).

## 10. Integrity status at drafting (no model loaded)

| check | result |
|---|---|
| `l1_screen.py integrity` | **70/70** (51 before A-OOD, 68 in revision 3; the new checks re-verify the OOD bank, the run order and the C6 binding threshold, section 15) |
| `l1_screen.py parity-cpu` | **17/17** (11 decisive, 6 report-only; section 8) |
| `l1_screen.py build --dry-hash --record` | B1, B2, B4 pass on the real cores; identity recorded |
| `scripts/l1_test_tiny.py` | **92/92** in 410 s, peak RSS 1,075 MiB, CUDA never initialised (83/83 before A-OOD, 89/89 in revision 3) |
| `scripts/l1_make_ood.py` (build, then `--verify`) | bank built once (46 s, no model call, peak 330 MiB); **11/11** on re-verification |

**What `integrity` covers:** every pinned input hash; the vendored code and the reference-only copies; SELECT-VIEWS-001's
role and floors and the absence of its registry entry; the long-context SPENT bar; the 0/15,629 payload overlap with the
pilot bank; the fresh prompts (456 = 57 x 8; disjoint in id, template family and text from the 248; no id or user text
of the HSKILL-002 pools; no HSKILL-VIEWS-002 source row); the bridge (228, none in the pools); R14's stored log-probs
(one prompt order for the three arms; KL ratio 0.6772 on the bridge, P6's reference); the HSKILL-002 and RCI seals'
statuses; the GATE_V2 sentences quoted in section 4; this file's section 5 equals `DECISION_RULE`; R14 identity facts;
the A-OOD bank (section 15): pinned and equal to its SOURCE record, then re-checked independently by the builder's
verifier from the file (Titan's one-step frame, oracle gold and the FDS cross-check, every stratum's definition,
uniqueness, and zero payload / operative-content overlap with every exclusion source, re-harvesting the historical
set with the reserved HSKILL_CONFIRM_V2* files skipped by name), the C6 scorer on the gold texts (600/600) and on
copy-the-input (100 on L_void and P_inapp, 0 elsewhere), and this file naming the bank's hash and carrying C6; the
run order (supa and sparse on the pre-gate stages, the gate, then the OOD stages and L8-15; nothing that reads the OOD
bank is scheduled before the gate); and the C6 binding threshold (6 of 100).

**What the tiny test covers:** the decision rule's boundaries (C2 per family, tolerance and detection; C4 both legs;
C5 strict; C6 per stratum: margin max(3, ceil(5% n)) and detection, a missing stratum or a broken pairing fails); the
margins' derivation from REP-001; fragile flags; build gates B1-B7 and tensor-exact contents; G1, G2;
the parity gate before L8-15, its legs passing on the published anchor predictions and on R14's own arrays and failing
on perturbations; the registered order observed stage by stage (supa and sparse pre-gate, the gate, sparse OOD still
loaded, supa OOD reloaded, L8-15), with every post-gate stage recording the gate it followed; a failed gate never
loading L8-15 and leaving the OOD bank unread, analyze then computing no L8-15 endpoint and no OOD score, and refusing
OOD outputs next to PARITY_FAIL.json; crash/resume equal
to an uninterrupted run file by file; a resume in another environment refused; the run lock; pending re-derived after
the lease; the cuda wiring (one lease, the host gate before it, breathe and the guard per row); zero process spawns
while waiting on a held lock; HostGuard's exits; the freeze (refuses after any output under runs/, records the
environment, binds the run: refuses a pre-freeze or --no-freeze run, accepts a frozen one); create-only analysis and
tamper refusal; the HF path as a named reason; the long-context overflow as report-only; the OOD stage (one row per
stratum on the tiny run, Titan's one-step frame, scored by `parse_state`, C6 computed per stratum in analyze) and two
C6 fixtures on tiny-model run directories whose OOD outputs are replaced by full 600-row fixtures: L8-15 identical to
Titan-Sparse passes C6, and L8-15 losing 10 of 100 rows in one stratum fails it, naming that stratum; the report-only
binding flags, checked exhaustively against the rule (a stratum can fail only if Titan-Sparse is right on >= 6 of 100),
all-binding in the fixtures and all-non-binding on the 1-row tiny strata.

**File hashes at drafting.** These change with review edits; the freeze records the final ones.

| file | sha256 |
|---|---|
| `scripts/l1_screen.py`, `scripts/l1_test_tiny.py`, this file | recorded at freeze |
| `docs/l1/L8_15_EXPECTED.json` | `76f203107eb24e5f…` |
| `docs/l1/offtask_fresh_prompts.jsonl` | `38b964bfc4a29401…` (schema 2; the schema-1 draw `62faf04d…` was deleted unread) |
| `docs/l1/offtask_fresh_prompts.SOURCE.json` | `c474681a5ef1ecce…` |
| `docs/l1/vendor_5b4c22d/VENDOR.json` | `c8ad1ae27ace45f3…` |
| `docs/l1/vendor_5b4c22d/VENDOR_REFONLY.json` | `db926b63413fa6e2…` |
| `docs/l1/queue_hook_l1.sh.draft` | `1fc8a0c4e736c804…` |
| `docs/l1/ood_bank.jsonl` (A-OOD, pinned in the runner) | `ac665c115b8c098e6bf58f6182a4a5a73b6fa1d87401afb1b3a55eeb165e940c` |
| `docs/l1/ood_bank.SOURCE.json` (pinned in the runner) | `a373686250caeb503061e267faec7b261cf089c8e6832f9789591f26886b08dd` |
| `scripts/l1_make_ood.py` (the builder; its own sha is in the SOURCE record) | `dbfd3d2e8eedd426…` |
| `rcwmt/env.py` (C6's parser and oracle, pinned in the runner) | `92abb6787588bfb0…` |

## 11. What this cannot establish

1. **No replication.** There is one training draw: the parent's seed 17. REP-001 found that retention and
   long-context numbers vary across draws by more than the differences being claimed. Splicing cannot manufacture
   replication.
2. **No confirmation.** Every decisive instrument is selection-grade. The drift instrument is spent for confirmation
   by this use. Long context and abstention are report-only on a SPENT bank. No unspent confirmation capacity exists
   for these claims (gap G9): the RCI sealed banks, T_BANK_V4 and HSKILL_SOUP_V3 are spent, and HSKILL-CONFIRM-002 is
   operator-reserved.
3. **Drift is a proxy for forgetting.** First-token KL on preservation-slice prompts measures how far behaviour moves,
   not what is lost. The slice is v3 SFT training data (`split: train`), as the 248 prompts were.
4. **Abstention is unexamined.** The parent's literal-UNKNOWN damage is confirmed (RCI-001). L8-15's is unknown,
   apart from the report-only 64-row counts (Q2).
5. **The retention gates disagree** (section 1); L1 selects on Gate v2 without a registered resolution.
6. **Serving format.** The core is FP32 only. A BF16 copy would be a different model.
7. No quality, efficacy or release claim follows from any verdict.

## 12. If PROMOTE, and only with the operator's explicit go-ahead

Draft vault records are in `docs/l1/vault_draft/`: `README.md` (model card), `CORE.json`, `PROMOTION.json` and
`EVIDENCE_MANIFEST.json`. Numbers there are placeholders filled from `L1_RESULTS.json`. **Nothing is written to the
vault by any L1 script.** The promotion itself would be a separate, operator-authorised step that follows the vault's
own rules: never overwrite a directory (archive, then write); weights hash-verified before and after the copy;
`config.json` set to `torch_dtype: float32`, the one sidecar that differs from the measured arm (gap G10); a
CATALOG.json entry wrapping CORE.json and a MODEL_REGISTRY.md row; never the legacy `_materialize_core.py`; a backup to
the private family repo only if the operator includes it in the authorisation scope. The parent's records should also
get an amendment recording RCI-001 (gap G2); that is the operator's call, separate from L1.

## 13. Review disposition (adversarial review of 2026-10-08, three lenses)

| lens / severity | item | disposition |
|---|---|---|
| post-hoc MUST-FIX | C5 prompts not excluded from HSKILL-CONFIRM-002 | **Applied.** Redrawn outside every id and user text of the HSKILL-002 pools (496 -> 456, 62 -> 57 classes); bridge 248 -> 228; integrity checks added; CONFIRM never opened. The R14 exposure question is Q7. |
| post-hoc MUST-FIX | SPENT long-context bank inside decisive C0 | **Applied, option (a).** Long-context and UNKNOWN parity, the long-context scorer and McNemar checks and the long-context token-count check are all report-only; a skipped long-context row labels only those numbers. Section 4 corrected. |
| post-hoc MUST-FIX | C2 margin justified by an inapplicable statistic | **Applied.** One margin logic for C2 and C4 (section 5): no detected loss, and at most half REP-001's cross-draw span; stated as value judgments. |
| post-hoc SHOULD | C2 gates on the aggregate | **Applied.** C2 is per family; the aggregate is report-only. |
| post-hoc SHOULD | forking path under-reported | **Applied.** Full R12 table, implicit criterion, D-23 arm set; "further forking is forbidden". Mirrored in the vault draft. |
| post-hoc SHOULD | pre-freeze run analysable as the registered run | **Applied.** `--no-freeze` refuses real cores and a real GPU; `freeze` scans all of `runs/`; `analyze` binds every invocation and stage to the freeze. |
| post-hoc SHOULD | Gate v2 vs legacy monitor disagreement | **Applied.** Section 1 and 11. |
| post-hoc SHOULD | RCI calibration banks omitted | **Applied.** Section 4 inventory; Q2. |
| post-hoc NOTE | stop-rule deviation not named | **Applied.** Section 8; fragile flags. |
| post-hoc NOTE | wording (winner's curse, P3/P7/P8, the 248's declaration, vault wording, read budget) | **Applied.** Sections 1, 4, 7; vault draft. |
| fidelity MUST-FIX | parity computed only after L8-15 | **Applied.** The gate runs between the anchors and L8-15; a failure writes PARITY_FAIL.json, never loads L8-15, and ends the registration. |
| fidelity SHOULD | Supa retention parity is CPU-vs-CUDA | **Applied.** Report-only, disclosed in sections 3 and 8; same-run CUDA Supa contrast added. |
| fidelity SHOULD | Titan-Sparse retention parity weak | **Applied**, with one deviation: per-class terminated counts within +-1, not identical, consistent with the 1-row allowance on the correct set. |
| fidelity SHOULD | 63 and 12 not gated | **Applied.** Arithmetic strict +-2 decisive; UNKNOWN (both counters) +-1 report-only, as the SPENT bank must be. |
| fidelity SHOULD | C5 has no comparability leg | **Applied.** The bridge log-prob leg is part of C0. |
| fidelity NOTE | analyze not bound to the freeze | **Applied** (see binding). |
| fidelity NOTE | digest wording; run.py / hf_checkpoint.py not vendored | **Applied.** Section 8; `VENDOR_REFONLY.json`. |
| fidelity NOTE | parity-cpu gaps (arm samples, decode, floor, environment) | **Applied.** Arm-sample and decode round-trip checks added; floor(0.03 n) stated; the environment is frozen and enforced (G0). Token ids were never recorded, so only counts can be compared, and the only recorded counts are on the SPENT bank, so that check is report-only (post-hoc MUST-FIX 2). |
| ops MUST-FIX | 2-3 spawns per 20 s while waiting | **Applied.** psutil + in-process PDH; test observes zero spawns. |
| ops SHOULD | RAM gate inside the lease | **Applied.** Host gate before the lease; short gates inside; exit 5 releases the lease. A pre-lease **VRAM** check is **not** added: while the peer holds the lease its job occupies the card, so free VRAM before acquisition says nothing; `wait_for_resources` checks VRAM (>= 3.6 GiB) once the lease is held, against a need of about 3.2 GiB. |
| ops SHOULD | duplicate run process | **Applied.** Run lock; pending re-derived after the lease. |
| ops SHOULD | resume across environments | **Applied.** G0 per invocation; section 3 reworded. |
| ops SHOULD | watchdog liveness | **Applied.** Before the lease, before each load, and every minute while running. |
| ops NOTE | hook retries a deterministic failure | **Applied.** No retry after exit 3. |
| ops NOTE | hook may be missed by the running chain | **Applied.** Section 9 step 3. |
| ops NOTE | D-22 wording, comment, pauses, progress | **Applied.** |
| D-28(a), after the review (peer, BWM-001 F-01/F-03/F-06) | in-distribution screens cannot see a shortcut that fails only out of range; OOD and rule-extrapolation probes needed before the freeze, L8-15 non-inferior to Titan-Sparse | **Applied (revision 3): amendment A-OOD, criterion C6** (section 15). Verified on 2026-10-08 (the rows below), but not yet reviewed adversarially; see Q9. |
| A-OOD verification (2026-10-08) SHOULD | The anchors' OOD stage ran before the C0 gate. A failed gate would have left the bank read by two models, and the rerun clause did not cover the bank | **Applied, option (a) (revision 4).** Every arm's OOD stage now runs only after a passed gate. Titan-Sparse stays loaded across the gate and Supa is reloaded: 4 loads, about +0.02 h. A failed gate leaves the bank unread. The rule now says no arm generates on the bank after a failed gate, and a rerun needs a fresh OOD bank if any arm generated on this one. `analyze` refuses OOD outputs next to `PARITY_FAIL.json`, and a post-gate stage with no passed gate on record is a C0 failure. `integrity` checks the run order and the tiny test observes it stage by stage. Sections 3, 5, 6, 8, 9 and 15 are updated. |
| A-OOD verification SHOULD | C6 is expected to be non-binding on 4 of 6 strata, and that was not quantified | **Applied.** Section 15 now tabulates Titan-Sparse's measured rates and states that C6 is expected to bind only on R_boundary and L_big. The results carry a report-only binding flag per stratum (Titan-Sparse correct >= 6 of 100), checked exhaustively against the rule in the tiny test. Q9 leaves an absolute floor or another comparator to the operator. The decisive rule is unchanged in substance. |
| A-OOD verification NOTE | Section 15 overstated how little the cut-choosing evidence saw of no-op and large-balance rows | **Applied.** It now states that R14's chain regret was measured on whole pilot TEST chains, which include void steps and balances up to 2,702. The pilot-chain maximum is corrected from 5,544 to 2,702 (TEST) and 7,200 (all splits), recomputed from `data/bank_pilot.json`. |
| A-OOD verification NOTE | The harvest, and every `integrity` run, byte-read the retired HSKILL-CONFIRM-001 | **Disclosed** in section 15 and Q10. The builder is pinned by its SOURCE record and is unchanged. |
| A-OOD verification NOTE | Out-of-scope records still describe C0-C5 | **Left for the caller**, because this session may not edit them: `docs/l1/vault_draft/README.md:87`; `PROMOTION.json`'s `rule_applied` template (no C6 inputs, no binding flags); DECISIONS.md (a D-29 follow-up recording A-OOD and this revision). The queue hook draft's run comment omits the OOD order; it is not wrong, and its hash is pinned. |

## 14. Open questions for the operator

| | question | default in this draft |
|---|---|---|
| **Q1** | Should C5 stay decisive? The alternative is Gate v2 + non-inferiority only, with drift report-only. | C5 is decisive. Without it, L8-15 is a weaker-or-equal Titan-Sparse with no stated advantage. |
| **Q2** | Add a **report-only** abstention endpoint before the freeze? Candidates: the RCI-001 calibration bank (128 rows, `remains_admissible_for: ["DEV","SELECTION"]`, measures exactly the confirmed literal-UNKNOWN damage; Supa and k=0.25 anchor scores exist) and/or HSKILL-VIEWS-002 (224 rows, UNSPENT, selection-admissible; parity anchors Supa 71/112, Titan-Sparse 70/112). Either needs its probe vendored and a parity check, about 3-5 GPU-min per arm. | Not implemented. **Recommended as a pre-freeze amendment**, RCI-001 calibration first. |
| **Q3** | Are the margins acceptable: C2 per family net >= -8 / -5 / -6 with no detected loss, C4 >= Titan-Sparse - 2? They are the only thresholds this draft introduces. | As written. |
| **Q4** | If the run is NON-COMPARABLE, may a matched transformers 4.51.3 environment be created (outside RCWM-T1) for a **new** registration with a fresh drift draw? | No. This registration ends either way. |
| **Q5** | SELECT-VIEWS-001's overdue `SELECTION_EXPOSED` registry entry and the missing read budget (gap G1). The records tree is read-only to this session. | Reported, not written. |
| **Q6** | Codename and directory. | `Rineya-Titan-Sparse-L815-Experimental-1.0.0-devfreeze`, codename `Titan-Sparse-L8-15`, assigned by this session; `RENAME.json` remains the vault's mechanism. |
| **Q7** | May a **hash-only** check of the 20 excluded R14 prompts against `HSKILL_CONFIRM_V2.jsonl` be run, to learn whether R14 already read CONFIRM items? If it did, an incident belongs in DECISIONS.md. | Not run; it opens the reserved file. |
| **Q8** | The C0 bridge leg uses 228 candidacy-exposed prompts, for the anchors only. Acceptable? | Yes: it validates the harness and reads no L8-15 number. |
| **Q9** | DECISIONS D-28(a) (about 16:20, 2026-10-08) requires, **before the freeze**, OOD and rule-extrapolation probes from data not used to choose the cut, with L8-15 non-inferior to Titan-Sparse, FP32 only. | **Applied in revision 3 as amendment A-OOD (C6, section 15); verified, with fixes, in revision 4.** The bank is fresh: 600 rows, built with no model call, excluded against every used set. C6 is per-stratum paired non-inferiority vs Titan-Sparse, margin max(3, ceil(5% n)) = 5 and no detected loss; no model reads the bank before the parity gate passes. **Expected binding:** Titan-Sparse is at 0-4% on L_void, R_wrap and P_inapp, and probably P_appdel (section 15 table), below the 6 of 100 a failing loss needs. So C6 is expected to bind only on R_boundary and L_big; the results flag the non-binding strata. Open for the operator: whether that is enough for D-28(a), or whether an absolute OOD floor, a different comparator (Supa or the copy floor) or more rows in the binding strata is wanted; whether the margin and strata are acceptable; and whether C6 should be reviewed adversarially before the freeze (recommended). |
| **Q10** | `rcwmt.bank.load_exclusions()` (FDS `harvest_exclusions`) reads every `*.json` / `*.jsonl` in the records tree, and `hskill-views-002/HSKILL_CONFIRM_V2.jsonl` has been there since 2026-09-18. The RCWM builders that called it (`bank_pilot`, `bank_confirm`, the T2 data and supplement, `bank_confirm2`) therefore byte-scanned the reserved file for transition payloads: a mechanical read, with no model call, keeping nothing but `{before, event}` payloads. Does this need a disclosure in the HSKILL-002 records? | A-OOD does **not** repeat it: its harvest skips `HSKILL_CONFIRM_V2*` by name before any open and still finds exactly the 6,464 payloads `load_exclusions` recorded on 2026-10-08. The harvest does, like `load_exclusions` before it, byte-read the **retired** HSKILL-CONFIRM-001 file (`hskill-views-001/HSKILL_CONFIRM.jsonl`, `RETIRED_INVALID_UNREAD`, `never_read: true`), and so does every `integrity` run, because integrity re-harvests the records tree. No confirmation capacity is lost, because that bank is retired as invalid, but its `never_read` flag is no longer literally true. Disclosure is the operator's call; the records tree is read-only to this session. |

## 15. Amendment A-OOD: rule-extrapolation probes (criterion C6; DECISIONS D-28(a))

**Why it exists.** D-28(a) (about 16:20, 2026-10-08) requires, before this registration freezes, out-of-distribution
and rule-extrapolation probes built from data that played no part in choosing the 8-15 cut, with L8-15 required to be
**non-inferior to Titan-Sparse**, its paired parent, in FP32. The reason is the peer's BWM-001 findings F-01, F-03 and
F-06: a forward LoRA learned a shortcut that the training data forced, stayed about 99.7% correct in distribution, and
failed only out of range. A shortcut of that kind is invisible to C1 and C2, which read in-distribution views only,
and to the evidence that chose the cut:

- R12's FR, and R14's one-step accuracy, were measured on **changed** pilot TEST one-frame items. Rows whose correct
  answer is "unchanged" (a void ledger event, an inapplicable action) were outside the metric that chose 8-15.
- R12's 640 one-frame items start from balances within +-99. None of the 208 ledger items has a balance of 100 or more.
- The exception is R14's chain_one k=4 regret, which D-26 cited for candidacy ("chained decision regret within
  noise"). It was measured on the 640 whole 4-step pilot TEST chains (`docs/localized/DESIGN.md`), and those chains do
  contain no-op steps and balances above 99:
  - 20 of the 208 ledger chains start with a void event, and 89 of their 832 steps are void;
  - 124 of their 624 later steps start from a balance of 100 or more, and TEST states reach |balance| 2,702.

  That exposure is indirect: a whole-chain regret, not per-step accuracy. No payload of the bank occurs in those
  chains (overlap 0, below).
- Removing blocks 16-27 could plausibly remove something that matters only at such edges (large magnitudes, a rule
  exception, a strict inequality) while leaving in-distribution accuracy intact. C6 asks this directly.

**The bank** (`docs/l1/ood_bank.jsonl`, sha256 `ac665c115b8c098e6bf58f6182a4a5a73b6fa1d87401afb1b3a55eeb165e940c`;
build record `docs/l1/ood_bank.SOURCE.json`, `a3736862…`; builder `scripts/l1_make_ood.py`, seed 20261028, a seed no
earlier RCWM-T1 builder, analysis or bootstrap used).

- **Provenance: fresh.** It was built on 2026-10-08, after the cut was chosen, with **no model call** (no torch, no
  tokenizer). No model has read it, and nothing in it was used to choose the cut. The pilot bank that chose the cut is
  excluded, together with every other set listed below.
- **Frame.** One-step transitions in Titan's trained one-step frame, `rcwmt.env.one_step_messages(family, before,
  event)`: the verbatim `SYSTEM_ONE` prompts that SELECT-VIEWS-001 and R12's one-frame items use. Events come from
  `rcwmt.env`'s recorded-marginal generators (`gen_start`, `gen_event`, `gen_action_library`).
- **Gold.** `rcwmt.env.step`. On every row it is cross-checked against the validated FDS oracle (`env.cross_check`).
- **Six strata of 100 rows.** Each stratum is built so that one specific shortcut gives a wrong answer.

| stratum | rows (definition) | shortcut it exposes | in R12's 640 one-frame items | copy-the-input scores |
|---|---|---|---:|---:|
| L_big | ledger; at least one balance with absolute value in [100, 2000] (1-4 big accounts: 17 / 48 / 25 / 10; range -1,999 to 1,988); event from the recorded marginals, status as drawn; no-op rows rejected (so all are posted: 25 add, 27 scale, 48 transfer; 67 touch a big account) | arithmetic or digit copying that holds only in the trained magnitude range | 0 of 208 | 0 |
| L_void | ledger; `void` event whose posted version would change the state (gold: unchanged) | applying an event regardless of its status | 18 of 208 (excluded from R12's changed-item FR) | 100 |
| R_wrap | register add (50) or mix (50) whose raw sum lies outside [0, 96] (83 overflow, 17 underflow; raw -6 to 268) | dropping or misapplying the modulo | 50 of 208 | 0 |
| R_boundary | register conditional with `other` = threshold - 1 / threshold / threshold + 1 (40 / 28 / 32), value != 0 mod 97 | `>=` for `>`, or a sign error at the branch | 2 of 208 | 0 |
| P_inapp | prerequisite action that is inapplicable (66 pre_on unmet, 25 pre_off set, 9 both) and would change the state if applied (gold: unchanged, the FDS oracle convention; Titan's one-step frame says an action "requires" the bits but never shows the no-op) | applying an action regardless of applicability | 55 of 224 (excluded from R12's changed-item FR) | 100 |
| P_appdel | applicable action whose delete clears a set bit that add does not restore | copying the input, or applying add only (T2 seed 0's prerequisite shortcut, R13) | 34 of 224 | 0 |

**What "OOD" means here.** Only L_big is out of range: it falls outside the range of R12's items and of every RCWM
start state. It is not outside everything Titan has seen. 30 of SELECT-VIEWS-001's 128 ledger views have a balance of
100 or more, and later states of pilot chains reach |balance| 2,702 in TEST and 7,200 across all splits. The other
five strata are **rule-edge** probes. Their kinds of rows occur in the pilot bank, in the shares above, but each
stratum keeps only rows where the rule's edge decides the answer. Two of them, L_void and P_inapp, are exactly the no-op rows that R12's changed-item metric never scored.

**Freshness: zero overlap, by payload and by operative content.** No row's `{before, event}` payload equals one from
any source below. No row's operative content does either; operative content is the answer-determining fields, by
FDS `generate.operative_key`, which drops tick, name and cost. Payloads and operative contents are also unique within
the bank. These counts were measured at build, and `integrity` recomputes them independently.

| source | payloads | overlap (payload / operative) |
|---|---:|---:|
| historical set (`rcwmt.bank.load_exclusions`, records tree) | 6,464 | 0 / 0 |
| `data/bank_pilot.json` (chose the cut) | 15,629 | 0 / 0 |
| `data/bank_confirm.json` (incl. order twins) | 13,552 | 0 / 0 |
| `data/bank_confirm2.json` | 12,163 | 0 / 0 |
| `data/bank_s4000.json` | 255,745 | 0 / 0 |
| `data/t2_train.jsonl`, `t2_eval.jsonl`, `t2_eval_xframe.jsonl`, `t2_eval_supp.jsonl` | 1,501 / 600 / 600 / 360 | 0 / 0 each |
| SELECT-VIEWS-001 (its user turn is exactly `{before, event}`) | 384 | 0 / 0 |
| union | 306,014 (299,155 operative) | 0 / 0 |

**The reserved bank was not opened.** `load_exclusions()` reads every JSON file in the records tree, and that
includes `HSKILL_CONFIRM_V2.jsonl`. The A-OOD harvest therefore re-implements it line for line with one change: files
named `HSKILL_CONFIRM_V2*` (four files) are skipped by name, before any open, and are never hashed. It finds exactly
6,464 payloads, the size `load_exclusions` recorded on 2026-10-08. Skipping can only remove payloads, so the reserved
file held none that is not also somewhere else. Earlier RCWM builders did byte-scan that file; see Q10. The harvest
does open the **retired** HSKILL-CONFIRM-001 file (`hskill-views-001/HSKILL_CONFIRM.jsonl`, status
`RETIRED_INVALID_UNREAD`, `never_read: true`), as `load_exclusions` always has, and so does every `integrity` run,
because integrity re-harvests the records tree. That bank is retired as invalid, so no confirmation capacity is lost,
but its `never_read` flag is no longer literally true (Q10).

**Scoring and decision (C6, verbatim in section 5).**

- **Generation.** Every arm generates greedily, at most 96 new tokens: the acquisition cap for these families in this
  frame, with the same DenseReference greedy loop. The stage `ood` runs inside the same lease job, **for every arm only
  after the C0 parity gate passed**: first Titan-Sparse (still loaded from its pre-gate stages), then Supa (reloaded),
  then L8-15 (its last stage). The gate reads no OOD output. If it fails, no model ever reads the bank, which stays
  unread (section 8.5). Every OOD stage records the gate it followed, and `analyze` checks that.
- **Correctness.** A row is correct iff `rcwmt.env.parse_state(family, prediction, lenient=True)` equals the oracle
  state. This is PREREG-1's lenient parser, the same for every arm. `rcwmt/env.py` is pinned in the runner and the
  freeze.
- **Per stratum, paired, L8-15 vs Titan-Sparse.** Both conditions must hold:
  - net (gained - lost) >= -max(3, ceil(0.05 n)), which is -5 at n = 100;
  - no detected loss, where a detected loss is lost > gained with a two-sided exact McNemar p < 0.05.
- C6 passes iff every one of the six strata passes. A missing stratum, or arms that cover different rows, fails it.
- **Supa** is reported (report-only), as are the aggregate over strata and each arm against Supa.

**The margin is a value judgment.** D-28(a) asks for non-inferiority but names no margin. C2's margins came from
REP-001's cross-draw spans, and no such span exists for fresh rows, so the amendment sets the margin directly: the cut
may cost at most 5% of a stratum, net, and never a loss the paired test can see.

- **At n = 100 the detection clause is implied by the margin**, as it is for C4. The smallest detectable loss is 6 lost
  and 0 gained (p = 0.031), which is net -6. A net of -5 is never detected (5 lost and 0 gained gives p = 0.0625).
- It is kept in the rule because it is what D-28(a) asks for, and it still binds if a stratum ever had fewer rows.
- **Fragile flags** (report-only) mark a stratum whose net lies within 3 of its tolerance, as for C2.
- **Binding flags** (report-only) mark, per stratum, whether C6 could fail there at all. L8-15 can lose only rows that
  Titan-Sparse got right. The smallest failing loss at n = 100 is 6 lost, 0 gained. So a stratum where Titan-Sparse is
  correct on fewer than 6 rows cannot fail C6, whatever L8-15 does, and the results say so.

**Comparability.** C6 has no published anchor, because the rows are new. It inherits comparability from C0:
- the frame family, the decoder, the greedy loop and the cap are the ones whose anchor parity C0 checks on
  SELECT-VIEWS-001;
- C6 is a within-run paired contrast, so a harness change that moves both arms alike cancels;
- `integrity` checks the scorer on the bank itself. The gold texts score 600/600. Copying the input scores 100 on the
  two no-op strata and 0 on the four must-change strata.

**What C6 cannot establish.**
- **No absolute floor, and four strata are expected to be non-binding.** C6 is relative to the parent, and the parent
  is near zero on exactly the rule edges four strata probe. Titan-Sparse's measured accuracy on the matching cells:

  | stratum | Titan-Sparse's measured accuracy on the matching cell | expected Titan-Sparse correct /100 | C6 can fail there? |
  |---|---|---:|---|
  | L_void | 0.036 on void-disc rows (4/112, T2_RESULTS section A, adapter OFF = Titan-Sparse, trained one-step frame); 0.00-0.12 on void events (R8, n 41, one-step frame) | about 4 | **no** (needs >= 6) |
  | R_wrap | 0.000 on add/wrap (n 40), mix wraps = 1 (n 22) and mix wraps >= 2 (n 58) (T2 section A); 0.008 on mix (R8, n 126) | about 0-1 | **no** |
  | P_inapp | 0.009 on inapplicable actions (R8, n 117, one-step frame); 0.042 on inapp-disc (n 24, T2's m1 frame, which states the no-op rule) | about 1-4 | **no** |
  | P_appdel | 0.020 on app-del (n 50, T2's m1 frame); never measured alone in the one-step frame (R8: 0.329 on all applicable actions, n 161) | probably a few | **probably not** |
  | R_boundary | 0.400 on each boundary-pair member, threshold = other - 1 and = other (n 40 each, T2 section A); 0.44 / 0.51 on conditionals (R8) | about 40 | yes |
  | L_big | 0.27-0.59 on posted pilot transitions with a balance above 99 (T2_RESULTS section B, adapter OFF, pilot DEV+TEST, one-step frame; add 0.415 n 53, scale 0.404 / 0.594 n 47 / 192, transfer 0.265 n 117); 0.32-0.65 on posted events in range (R8) | about 30-50 | yes |

  **So C6 is expected to bind only on R_boundary and L_big.** On L_void, R_wrap and P_inapp (and probably P_appdel),
  C6 cannot fail whatever L8-15 does: L8-15 cannot lose rows its parent never got right. These strata still show any
  gain or loss in the report-only absolute counts, and the binding flags (above) report which strata could fail in
  the actual run. On L_void and P_inapp, copying the input scores 100, so the absolute counts there are read against
  that floor. Whether an absolute OOD floor, a different comparator (for example Supa or the copy floor), or more rows
  in the binding strata is wanted is the operator's decision before the freeze (Q9). This draft keeps C6 as D-28(a)
  states it: non-inferiority to Titan-Sparse.
- **One step only, one frame.** Multi-step chains and other frames are not probed.
- **Resolution.** C6 detects large, shortcut-sized losses (10 lost and 0 gained in a stratum fails it). It cannot
  detect a cost of a few rows per stratum.
- **No replication or confirmation.** The bank is selection-grade, and this use spends it for confirmation.

**Cost.** 600 more generations per arm, plus one more streamed load (Supa's reload for its post-gate OOD stage).
Section 9 gives the price: 435-523 s of compute per arm, and the expected band rises from 1.16-1.42 to 2.14-2.58 GPU
hours for the three arms.
