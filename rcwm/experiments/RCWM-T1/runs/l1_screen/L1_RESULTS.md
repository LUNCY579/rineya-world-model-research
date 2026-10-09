# PREREG-L1 results: DO NOT PROMOTE

Analysed 2026-10-09T07:04:19Z; forward dense; freeze binding: every invocation and stage: after the freeze, frozen runner, freeze record, frozen environment. Failed conditions: C2.

## Decision rule (registered, verbatim)

> L8-15 is eligible for promotion as an experimental core - promotion itself still requires the operator's explicit go-ahead (D-26) - if and only if, in the L1 run, ALL of the following hold. C0 identity and comparability, settled before any L8-15 output exists: the measured L8-15 weights are the frozen bytes (sha256 in L8_15_EXPECTED.json; per-group FP32 hashes equal to R14's loc8_15 arm); every build and load gate passed; the forward path is the programme's DenseReference (vendored from pilot commit 5b4c22d); the run used the frozen runner and the environment recorded in the freeze, in every invocation; the full decisive instruments were run (no row subset) and no decisive prompt was truncated or skipped; the L1 scorers reproduce every published per-row verdict and count on the anchors' published SELECT-VIEWS-001 and arithmetic-panel predictions, the L1 tokenizer and decoder return every one of those published texts unchanged (encode, decode, strip), and every arithmetic prompt fits the registered 512-token context; and the anchor parity gate, evaluated after both anchors and before L8-15 is loaded, passes: Supa 1.7 reproduces its published SELECT-VIEWS-001 texts on all but max(3, floor(3% of n)) rows, its canonical total within +-3 and its floor verdicts identically; Titan-Sparse does the same and also reproduces arithmetic_permissive and arithmetic_strict each within +-2 and references_retained within +-1 of its published 72, 63 and 41, its correct set on the 47 permissive and 41 strict source-correct rows up to 1 row per policy, its per-class arm_correct and terminated counts within +-1, and its 24 published arithmetic samples in text and termination on all but 3; and both anchors agree with R14's stored first-token log-probabilities on the 228 bridge prompts (top-1 agreement >= 0.99, mean KL(L1||R14) < 0.001 nats). If the anchor parity gate fails, L8-15 is never loaded and no arm generates on the OOD bank. C1 acquisition floors (GATE v2 term 3): on SELECT-VIEWS-001 every family's two-sided exact binomial 95% lower bound on canonical_correct exceeds its published binding floor (ledger_replay 0.2344, prerequisite_plan 0.0781, register_machine 0.0703). C2 acquisition non-inferiority, per family, paired within the run against Titan-Sparse on canonical correctness: in every family the net change (gained - lost) is >= -8 (ledger_replay), >= -5 (prerequisite_plan), >= -6 (register_machine), and no family shows a detected loss (lost > gained with a two-sided exact McNemar p < 0.05). C3 arithmetic (GATE v2 term 1): arithmetic_permissive(L8-15) >= 62 of 150. C4 source answers retained (GATE v2 term 2 plus non-inferiority): references_retained(L8-15) >= 37 of 47 AND >= references_retained(Titan-Sparse, same run) - 2. C5 off-task drift on the 456 fresh prompts: the paired percentile-bootstrap 95% upper bound (10,000 resamples over prompts, seed 20261009) of mean first-token KL(L8-15||Supa) / mean KL(Titan-Sparse||Supa) is < 0.80. C6 out-of-distribution rule extrapolation (amendment A-OOD, D-28(a)), paired within the run against Titan-Sparse on the 600-row OOD bank (docs/l1/ood_bank.jsonl, sha256 recorded in ood_bank.SOURCE.json and pinned in the runner; built with no model call from data that played no part in choosing the cut: six strata of 100 one-step transitions in Titan's trained one-step frame, L_big, L_void, R_wrap, R_boundary, P_inapp and P_appdel), every arm run on it only after the anchor parity gate passed, greedy with at most 96 new tokens, and a row correct iff rcwmt.env.parse_state(family, prediction, lenient=True) equals the oracle state: in every stratum the net change (gained - lost) is >= -max(3, ceil(5% of n)), which is -5 at n = 100, and no stratum shows a detected loss (lost > gained with a two-sided exact McNemar p < 0.05). Supa's OOD scores are report-only. If C0 fails the run is NON-COMPARABLE: no verdict, nothing promoted, no L8-15 endpoint computed, and this registration ends (a rerun is a new registration with a fresh drift draw that may not reuse these prompts, and with a fresh OOD bank if any arm has generated on this one). If C0 holds and any of C1-C6 fails: DO NOT PROMOTE. Long context and literal UNKNOWN (including every parity and scorer check on that SPENT bank), Supa's arithmetic against its CPU-measured source answers, the legacy per-class monitor and the bridge drift ratio are report-only and never enter C0-C6. No threshold may change after any L1 output exists.

| condition | holds |
|---|---|
| C0 | True |
| C1 | True |
| C2 | False |
| C3 | True |
| C4 | True |
| C5 | True |
| C6 | True |
- C2 ledger_replay: net +0 (lost 11, gained 11, p 1); tolerance -8; detected loss False; pass
- C2 prerequisite_plan: net -8 (lost 9, gained 1, p 0.02148); tolerance -5; detected loss True; FAIL
- C2 register_machine: net +0 (lost 2, gained 2, p 1); tolerance -6; detected loss False; pass
- C6 L_big (n 100): net -2 (lost 11, gained 9, p 0.8238); tolerance -5; detected loss False; pass
- C6 L_void (n 100): net +6 (lost 2, gained 8, p 0.1094); tolerance -5; detected loss False; pass
- C6 R_wrap (n 100): net +0 (lost 0, gained 0, p 1); tolerance -5; detected loss False; pass
- C6 R_boundary (n 100): net +7 (lost 7, gained 14, p 0.1892); tolerance -5; detected loss False; pass
- C6 P_inapp (n 100): net +2 (lost 0, gained 2, p 0.5); tolerance -5; detected loss False; pass
- C6 P_appdel (n 100): net -1 (lost 1, gained 0, p 1); tolerance -5; detected loss False; pass
- C6 binding (report-only; a stratum can fail only if Titan-Sparse is correct on at least the fewest rows whose loss fails it): L_big Titan-Sparse 31/100, can fail; L_void Titan-Sparse 4/100, CANNOT FAIL (needs >= 6); R_wrap Titan-Sparse 1/100, CANNOT FAIL (needs >= 6); R_boundary Titan-Sparse 35/100, can fail; P_inapp Titan-Sparse 0/100, CANNOT FAIL (needs >= 6); P_appdel Titan-Sparse 2/100, CANNOT FAIL (needs >= 6)

Fragile (within the anchor-parity tolerance of a threshold, report-only): C2 prerequisite_plan: net -8, tolerance -5; C3: arithmetic_permissive 63, bar 62; C6 L_big: net -2, tolerance -5

## Endpoints

| | Supa 1.7 | Titan-Sparse | L8-15 |
|---|---:|---:|---:|
| Acquisition /384 (canonical) | 14 | 232 | 224 |
| Families clearing floor | 0 | 3 | 3 |
| Arithmetic /150 permissive | 47 | 72 | 63 |
| Arithmetic /150 strict (report) | 41 | 63 | 55 |
| Source answers retained /47 | 47 | 41 | 43 |
| Gate v2 | False | True | True |
| Long context /256 (SPENT, report) | 161 | 164 | 161 |
| UNKNOWN substring /64 (report) | 13 | 12 | 12 |
| UNKNOWN literal /64 (report) | 13 | 12 | 12 |
| Mean first-token KL vs Supa, fresh prompts | 0.0 | 0.0992 | 0.0641 |
| Mean first-token KL vs Supa, 228 bridge prompts (report) | 0.0 | 0.0971 | 0.0658 |
| Legacy per-class monitor (report) | PASS/PASS | FAIL/FAIL | FAIL/FAIL |
| OOD exact state, all strata (aggregate report-only) | 95/600 | 73/600 | 85/600 |
| OOD L_big (C6 input; Supa report-only) | 5/100 | 31/100 | 29/100 |
| OOD L_void (C6 input; Supa report-only) | 17/100 | 4/100 | 10/100 |
| OOD R_wrap (C6 input; Supa report-only) | 0/100 | 1/100 | 1/100 |
| OOD R_boundary (C6 input; Supa report-only) | 0/100 | 35/100 | 42/100 |
| OOD P_inapp (C6 input; Supa report-only) | 68/100 | 0/100 | 2/100 |
| OOD P_appdel (C6 input; Supa report-only) | 5/100 | 2/100 | 1/100 |

## Paired contrasts (L8-15 vs Titan-Sparse, same run)

Acquisition (net, discordant, exact two-sided McNemar p): total -8 / 36 / 0.243; ledger_replay +0 / 22 / 1; prerequisite_plan -8 / 10 / 0.02148; register_machine +0 / 4 / 1

OOD rule extrapolation (net, discordant, exact two-sided McNemar p): total_report_only +12 / 54 / 0.1337; L_big -2 / 20 / 0.8238; L_void +6 / 10 / 0.1094; R_wrap +0 / 0 / 1; R_boundary +7 / 21 / 0.1892; P_inapp +2 / 2 / 0.5; P_appdel -1 / 1 / 1

- drift_fresh: KL(arm||Supa) ratio 0.646 [0.597, 0.693]; KL(Supa||arm) 0.661 [0.627, 0.697]; flags none
- drift_bridge: KL(arm||Supa) ratio 0.677 [0.610, 0.739]; KL(Supa||arm) 0.682 [0.622, 0.744]; flags none

## Registered predictions (scored, not decisive)

- P1: L8-15 clears all three SELECT-VIEWS-001 family floors -> held
- P2: per-family net (L8-15 - Titan-Sparse): ledger_replay in [-12, 0], prerequisite_plan in [-4, +4], register_machine in [-6, +2]; aggregate in [-16, +2] (pilot projection -6 / 0 / -2.5, about -8) -> FALSIFIED
- P3: no retention loss: references_retained(L8-15) >= references_retained(Titan-Sparse) - 1 -> held
- P4: arithmetic_permissive(L8-15) >= 62 -> held
- P5: fresh-prompt drift ratio point estimate lies in [0.60, 0.76] (R14 on its 248: 0.680 [0.617, 0.735]) -> held
- P6: bridge drift ratio (228 R14 prompts, this harness) is within +-0.05 of R14's 0.677 on the same 228 -> held
- P7: no long-context loss (report-only): long_context(L8-15) >= long_context(Titan-Sparse) - 3 -> held
- P8: no abstention loss (report-only): UNKNOWN substring(L8-15) >= UNKNOWN substring(Titan-Sparse) - 1 -> held
- P9: both anchors reproduce their published per-row predictions on >= 99% of rows of every instrument (decisive and report-only alike) -> held
- P10: both anchors' bridge log-probabilities agree with R14's within mean KL 1e-4 nats (10x inside the bound) -> held

SPARSE-001's selection rule applied to {Titan-Sparse, L8-15} would prefer (report-only): ['l8_15']

## Anchor parity gate (C0)

Enforced True; pass True; failed legs none.
- supa.select_views_001: pass
- supa.bridge_logprobs: pass
- sparse.select_views_001: pass
- sparse.arithmetic_panel: pass
- sparse.bridge_logprobs: pass
- (report-only) supa.arithmetic_panel_vs_b28_cpu: within tolerance
- (report-only) supa.long_context: within tolerance
- (report-only) sparse.long_context: within tolerance
- scorer parity on published artifacts: decisive 11/11, report-only 6/6

