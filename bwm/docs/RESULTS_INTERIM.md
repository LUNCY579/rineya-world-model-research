# BWM-001 native — interim results (2026-10-06, evening)

Status: **interim**. Every number below is validation-grade and comes from one training seed unless stated
otherwise. Nothing here is a confirmatory test. Sources are `DECISIONS.md` and the run directories named under each
result.

## 1. Summary

BWM-001 asks whether a frozen 0.6B Rineya model can plan better when it gets two things: a learned, goal-blind world
model, and outcome-conditioned plan proposals, with or without explicit predicted states. Everything here runs
natively on frozen **Titan-Sparse** (FP32) with small LoRA adapters, on a laptop GPU shared with another research
session.

Four results so far:

1. **The learned LM world model is accurate and extrapolates arithmetic.** It is 99.7% exact in distribution and
   99.5% exact on numbers never seen in training. However, it fails to apply the limit rule at new limit values:
   36.5% on legality-flip doubles.
2. **The failure is a shortcut, and it can be repaired without training.** Gating the frozen base model as a verifier
   by the world model's own uncertainty raises flip accuracy from 32% to 95%. In-distribution accuracy does not drop,
   and about 90% of the downstream planning gap is recovered.
3. **The first proposer arm has run.** A00 (action-only proposer, no world model) reaches 62% on held-out goals under
   tight budgets. The output format is perfect and 87% of individual steps are optimal. The F-reranked and
   state-supervised arms are running.
4. **A PlanBench Blocksworld harness is built and validated.** VAL agrees with our executor on 602 of 602 plans.
   Published LLM plans are re-scored per instance. A leakage finding forced test-signature filtering of all training
   data.

## 2. Setup

| Component | Implementation |
|---|---|
| Backbone | Rineya Titan-Sparse 1.0.0 (Qwen3-0.6B geometry, FP32, frozen, never merged) |
| Adapters | LoRA r16 / α32 on q,k,v,o,gate,up,down (10.1M params); AdamW (0.9, 0.95), lr 1e-4, WSD |
| Domain RW2 | RegisterWorld law (add1/sub1/double/halve/finish, ±limit) with **tight budgets** (optimal + 1 + 0–2) |
| Split | **held-out goal values**; no test goal appears in any training row |
| Forward model F_LM | goal-blind LoRA: (x, action, limit) → {legal, terminated, x} |
| Proposers | action-only vs state-and-action LoRA; identical rows, order and init per seed |
| Inference | K = 4 candidates (CRN seeds); execute one action, replan; F rerank primary, gate as separate operating point |

## 3. World-model findings (`runs/F_titan_main`, `runs/A1_factorized`)

### 3.1 F_LM accuracy by stratum

All rows are exact-match accuracy.

| Stratum | n | F_LM |
|---|---:|---:|
| ID transitions (limits 32/48/64, in-sample) | 1,455 | 0.997 |
| \|x\| > 64 at unseen limits 96/128 | 960 | 0.995 |
| Three-digit outputs | 290 | 0.907 |
| **Legality-flip doubles** (legal at 96/128, illegal at every training limit) | 96 | **0.365** |
| Odd held-out limits 21–61 | 479 | 0.931 |

Zero-shot Titan without the adapter scores 0% under the strict schema. It answers in its trained bare-state format
`{"x":…}`, and its content is also wrong.

### 3.2 Mechanism: a shortcut forced by the data

- In any training set whose numbers stay at or below 64, the rule "doubling |x| > 32 is illegal" is right 100% of the
  time. Adding more limits at or below 64 cannot remove it.
- With the adapter switched off, Titan answers "is |2x| ≤ L?" correctly on most flips.
- **Above** the training range the shortcut is soft: F_LM's P(legal) on flips is about 0.2–0.45.
- **Below** it (limit 21) the shortcut is confidently wrong: F_LM predicts doubling 11–14 legal with P = 0.81–0.997.

### 3.3 Training-free repairs

Accuracy columns are exact match. The last two columns are BFS-256 success.

| Legality source for doubles | Flip-type (216) | ID (873) | BFS-256 range_val | BFS-256 long_val |
|---|---:|---:|---:|---:|
| F_LM | 0.324 | 0.994 | 0.720 | 0.965 |
| Frozen-base verifier on every double | 0.917 | 0.912 | 0.737 | **0.275** |
| Calibrated P(legal) > 0.10 | 0.741 | 0.981 | 0.803 | 0.965 |
| **Uncertainty-gated verifier** (only if 0.01 < P < 0.5) | **0.949** | **0.994** | **0.863** | 0.965 |
| Code rule \|c\| ≤ L (ceiling, not learned) | 0.954 | 0.995 | 0.840 | 0.965 |

- The τ threshold and the gating band were selected on calibration limits 80 and 112, which no task uses.
- Exact dynamics reach 0.880 on range_val BFS-256.
- The gated verifier calls the verifier on about 18% of keys.
- Limit-only BFS-256 drops slightly, from 0.98 to 0.967.
- Confident errors below the training range are not repaired; odd limits stay at 0.933.

The trainable version, **F_vlab**, is a single-call F trained on 2,322 gated self-labels at limits 72–120. Its labels
agree with the exact law 99.3% of the time. Training is queued.

## 4. Planning on held-out goals (`runs/EVAL_pilot_s1`, RW2 validation)

Success rates per split; the number of F calls per task is given for ID.

| System | ID (400) | Long (200) | Range (300) | F calls per task (ID) |
|---|---:|---:|---:|---:|
| Greedy-1 under F_LM | 0.650 | 0.740 | 0.473 | 21 |
| Greedy-2 under F_LM | 0.797 | 0.830 | 0.517 | 51 |
| BFS-16 under F_LM | 0.675 | 0.780 | 0.473 | 36 |
| BFS-64 under F_LM | 0.780 | 0.870 | 0.507 | 60 |
| BFS-256 under F_LM | **1.000** | 0.965 | 0.720 | 79 |
| **A00**: action-only proposer, no F, K = 4 (200 ID tasks, CPU) | 0.620 | – | – | 0 |

A00 diagnostics:

| Diagnostic | Value |
|---|---:|
| Malformed outputs | 0% |
| First action optimal | 70.5% |
| Per-step optimal | 87.3% |
| Open-loop success (first plan executed in full) | 24.5% |
| Coverage@4 at round 1 | 38.5% |
| Success at slack 0 / 1 / 2 | 0.46 / 0.77 / 0.64 |
| Success at optimal length 1 / 4 / 8 | 1.00 / 0.62 / 0.33 |

51 of the 76 failures went wrong on the first action.

RW2 is fully searchable: BFS under the learned F solves ID at about 80 F calls per task. Proposer claims must
therefore be stated as amortization (success against compute), not raw success. The A01, A10 and A11 arms are running
on the GPU over the same 200 tasks.

## 5. PlanBench Blocksworld harness (`bench/planbench`, `bwmn/blocks.py`)

- **Sources.** The assets come from LLMs-Planning (MIT), commit fc638a1a. VAL is used unmodified.
- **Validation.** BFS optimal plan lengths match the published ground truth on 120 of 120 instances. The oracle
  planner scores 602/602, and VAL agrees with the exact executor on all 602.
- **Leakage finding.** The 602 instances are only 347 distinct problems up to renaming. 54% of random 3–5-block
  problems are renamings of a test instance. Intermediate suffix states also leak: 577 such training rows were found
  and removed. Training therefore uses 4–6 blocks, mostly 5–6, while the test is mostly 3–4 blocks.
- **Published plans re-scored with local VAL** (zero-shot, out of 600):

| Model | Valid plans |
|---|---:|
| DeepSeek-R1 | 595 |
| o1-preview | 587 |
| LLaMA-3.1-405B | 386 (paper reports 376) |
| Claude 3 Opus | 356 |
| o1-mini | 340 |
| GPT-4o | 213 |
| GPT-4 | 210 |
| LLaMA-3-70B | 205 |

- **Blocksworld world model and proposer.** Training data is built: 12k forward rows (half illegal) and 39.9k proposer
  rows. The forward model is the next GPU job.
- **Comparators.** A same-harness Qwen3-4B/0.6B comparator is built. Its extractor agrees with published verdicts on
  599/600 (GPT-4) and 596/600 (o1). Must-fix items remain before any run (see `TODO_BEFORE_RUNS.md`).

## 6. Infrastructure lessons that affect interpretation

- The laptop GPU also drives the display. The machine had driver resets and two crashes:
  - one standby-under-load bugcheck, 0x19C;
  - VRAM spill and paging.
- Jobs now run with these mitigations:
  - a 25% duty cycle;
  - a 70% VRAM cap;
  - thermal pauses, sharing a watchdog with the other session;
  - keep-awake, approved by the user;
  - crash-resilient 50-task evaluation slices.

None of these change any model's math, seeds or decoding. The ~3 h CPU A10 run lost in a crash is being redone on
the GPU.

## 7. Claims

**Allowed now (validation-grade, single seed):**

- A LoRA world model on frozen Titan learns RW2 dynamics and extrapolates arithmetic to unseen numbers.
- It does not extrapolate a parameterized legality rule. This is a data-forced shortcut.
- Uncertainty-gated verification by the frozen base repairs the soft side of that shortcut without training:
  flips go from 0.32 to 0.95, and range planning from 0.72 to 0.86.
- Our harness reproduces published PlanBench verdicts per instance.

**Not yet allowed:**

- Any statement about the T effect (state supervision) or the F effect for proposers.
- Any multi-seed or confirmatory claim.
- Any PlanBench score for Rineya.
- Any comparison to larger models.
