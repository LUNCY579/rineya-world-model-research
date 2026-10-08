<!--
DRAFT model card for a possible vault core. NOT a vault file. Written by RCWM-T1 (PREREG-L1, D-26) under
docs/l1/vault_draft/. It may be used only if PREREG-L1's registered rule returns PROMOTE AND the operator gives an
explicit go-ahead. Every {{...}} is filled from runs/l1_screen/L1_RESULTS.json (key path given); nothing is typed by hand.
If the verdict is DO NOT PROMOTE or NON-COMPARABLE, this file is not used.
-->
---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
base_model: Rineya-Supa-1.7.0-devfreeze
tags:
  - rineya
  - experimental
  - research-artifact
  - not-release-qualified
  - fp32-only
---

# Rineya Titan-Sparse-L8-15-Experimental 1.0.0-devfreeze

**This is an experimental research core. It is not a release, it is not release-qualified, and it is not an
established replacement for Supa 1.7, Titan-Sparse or Opallo.**

It is Supa 1.7 plus **Titan-Sparse-Experimental's fine-tuning delta, kept only in transformer blocks 8-15**.

- In blocks 8-15, every tensor is Titan-Sparse's stored FP32 tensor.
- Everywhere else, every tensor is Supa 1.7's.

**No new training, and no model call to construct it.** It keeps 41.3% of Titan-Sparse's delta energy and 64.2%
of its norm. The 16-27 part of the delta is dropped. An exploratory patching study found that part contributes
nothing measurable to the transition skill.

## Where the candidate came from, stated first

**The 8-15 cut was chosen after seeing exploratory pilot data. It is not the output of a rule fixed in advance.**

| step | experiment | data | result |
|---|---|---|---|
| Located the skill | RCWM-T1 R12 (weight-spliced hybrids) | 640 pilot one-step items | Titan-Sparse's transition skill lives in blocks 8-15: FR 0.85 / 1.00 / 0.97 with the delta in 8-15 only, against 0.09 / 0.00 / 0.00 with the delta in 16-27 only. Three prefix cuts were HI in every family (8-15, 8-19, 8-23; 8-11 was not); 8-15 was chosen, without a written criterion, as the smallest of them |
| Measured drift | RCWM-T1 R14 (D-23; arms loc8_15 and up16_27) | 248 off-task prompts, declared at birth "no training or selection"; this use went beyond that | Removing 16-27 cut first-token KL drift against Supa to 0.68x [0.62, 0.74] of Titan-Sparse's |

**The promotion decision is not post hoc.**

- It was made by the rule in `evidence/PREREG_L1.md`, frozen before any of its instruments was read for this
  core (`evidence/PREREG_L1_FREEZE.json`).
- That rule used only instruments that played no part in choosing the cut:
  - SELECT-VIEWS-001, which shares no item payload with the pilot bank;
  - the 150-row arithmetic panel and its 47 source answers;
  - 456 fresh off-task prompts (57 reply classes x 8), disjoint in id, template family and text from the 248 that
    R14 used, and from every id and user text of the HSKILL-002 source pools (so from both HSKILL-002 banks).
- The pilot bank is not read by the screen. 228 of the 248 prompts serve only to check, on the two anchors, that the
  screen's forward path reproduces R14's stored log-probabilities, and as a report-only bridge ratio. No number
  of this core measured on them is evidence for this promotion.
- No further cut may be screened under this registration; that forbids further forks, it does not undo the one
  that produced 8-15.

## ⚠ This core is FP32 and must not be converted to BF16

This has the same reason as for its parent. The kept delta is a magnitude-selected subset of a 0.056%-relative
update, and BF16 storage would zero a large share of it.

- `config.json` declares `torch_dtype: float32`. This is the only sidecar that differs from Supa 1.7.
- The measured arm used Supa's byte-identical `config.json`, as the programme's measured-arm convention requires.
  The two differ only in that file.

## Measured (PREREG-L1, one run, all three arms in the same process, FP32)

| Endpoint | Supa 1.7 | Titan-Sparse | **This core** | role |
|---|---:|---:|---:|---|
| New skill, SELECT-VIEWS-001 /384 | {{summary.supa.acquisition}} | {{summary.sparse.acquisition}} | **{{summary.l8_15.acquisition}}** | selection |
| Families clearing the published floor | {{summary.supa.floors_cleared}}/3 | {{summary.sparse.floors_cleared}}/3 | **{{summary.l8_15.floors_cleared}}/3** | selection |
| Arithmetic /150, permissive | {{summary.supa.arithmetic_permissive}} | {{summary.sparse.arithmetic_permissive}} | **{{summary.l8_15.arithmetic_permissive}}** | selection |
| Arithmetic /150, strict | {{summary.supa.arithmetic_strict}} | {{summary.sparse.arithmetic_strict}} | **{{summary.l8_15.arithmetic_strict}}** | report |
| Source answers retained /47 | {{summary.supa.retained}} | {{summary.sparse.retained}} | **{{summary.l8_15.retained}}** | selection |
| Off-task drift: mean first-token KL vs Supa, 456 fresh prompts | 0 | {{summary.sparse.drift_fresh_kl_arm_supa}} | **{{summary.l8_15.drift_fresh_kl_arm_supa}}** | selection |
| Drift ratio to Titan-Sparse [95% CI] | — | 1 | **{{drift.drift_fresh.ratio_l8_15_over_sparse.kl_arm_supa.ratio}} [{{...ci95}}]** | selection |
| Long context /256 (SPENT suite) | {{summary.supa.long_context}} | {{summary.sparse.long_context}} | **{{summary.l8_15.long_context}}** | report |
| Literal `UNKNOWN` /64 (substring / exact) | {{summary.supa.unknown_substring}} / {{summary.supa.unknown_literal}} | {{...}} | **{{...}}** | report |
| Gate v2 | {{summary.supa.gate_v2}} | {{summary.sparse.gate_v2}} | **{{summary.l8_15.gate_v2}}** | — |
| Legacy per-class monitor (permissive/strict) | {{...}} | {{...}} | **{{...}}** | report |

Both anchors were re-measured in the same run and passed the anchor parity gate before this core was loaded
({{parity_gate.legs}}). The run used transformers 5.14.1, not the programme's deleted 4.51.3 harness; under
SPARSE-001's literal stop rule it would be non-comparable, and the gate is its registered replacement. Outcomes
flagged FRAGILE: {{fragile_report_only}}. The registered decision rule returned **PROMOTE (eligible)** on
{{analysed_utc}}, with conditions C0-C5 as listed in `evidence/L1_RESULTS.md`.

## Read this before believing the table

- **It is one training draw.** The delta comes from Titan-Sparse, which comes from a single RRP-015 seed.
  REP-001 showed that retention and long-context figures vary across draws by more than the differences shown
  here. Splicing cannot manufacture replication.
- **Every decisive instrument is selection-grade, and none confirms anything.**
  - The 456 drift prompts are spent for confirmation by this use.
  - Long context and literal `UNKNOWN` live on a bank marked SPENT. They are reported only and did not select
    this core.
- **Abstention.** RCI-001 *confirmed* that the parent's delta damages literal-`UNKNOWN` abstention: 25/128 vs
  Supa's 40/128, p = 6.1e-5. A decision-level screen found no change (70 vs 71 of 112). This core keeps 41% of
  that delta's energy. Its abstention has not been confirmed either way.
- **Drift is a proxy.** First-token KL measures how far behaviour moves on preservation-slice prompts, which are
  SFT training data of the base programme (`split: train`). It does not measure what is lost.
- **Acquisition inherits a winner's curse.** The parent was selected on the same acquisition and retention
  instruments. That favours this core on the absolute thresholds; the paired contrasts against the parent are, if
  anything, biased conservatively (regression to the mean on the items the parent was selected on).
- **The two retention gates disagree.** Gate v2 and the legacy per-class monitor disagree for every arm, and no
  resolution is registered. This core was selected on Gate v2 terms, as its parent was; the legacy monitor is
  reported, and is expected to fail it as it fails every arm.

## Reproducing it

The build is deterministic, with no model and no training. It uses the two published cores (Supa 1.7 `962abc5b…`
and Titan-Sparse `a28629c9…`):

```
python -B scripts/l1_screen.py build --out-dir <dir>      # RCWM-T1; gates B1-B7
```

Weight SHA-256:

```
6282b45676fab8b2446d4363373b94931aea16c346ba769c284b81fcb0f97fe3
```

The build is verified three ways:
- two independent derivations, plus a re-read of the written file, agree byte for byte;
- the per-group FP32 hashes equal those of the arm that RCWM-T1 R14 measured;
- Titan-Sparse − Supa is re-checked to be exactly zero outside blocks 8-27, element by element.

## Intended use

Research inference, and the localisation arm to carry into independent training draws of the parent recipe.

- It is not for production serving and not quality-validated.
- It is not a substitute for Supa 1.7 or Titan-Sparse in any published system.
- `release_permitted` and `quality_improvement_established` are both false.

## Limitations

1. There is one training draw and no replication.
2. The candidate was chosen post hoc from pilot data. Only the promotion decision was preregistered.
3. There is no confirmation instrument for drift, long context or abstention. The relevant banks are spent or
   operator-reserved.
4. Gate v2 has no long-context term and no abstention term.
5. The core is FP32 only, and a BF16 copy is a different model.
6. Human data review and blind semantic abstention review were **NOT PERFORMED**.
7. No broad model-quality superiority over Supa, Titan-Sparse or Opallo is established.

The codename was assigned by the RCWM-T1 session for clarity, not by the operator. `RENAME.json` is the vault's
supported mechanism if a different roster name is preferred.
