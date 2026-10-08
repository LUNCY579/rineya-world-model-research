# Block-localised delta: does the dispensable upper delta carry off-task drift? (design, D-23)

Status: exploratory follow-up to RESULTS_PILOT R12. Written 2026-10-08 before any localized data exists.
`runs/bank_pilot/localized/` does not exist yet. The reading rule below is DECISIONS D-23's, stated at 23:21 on
2026-10-07, together with its operationalisation, which is fixed here before the run.

## Question

In R12, Supa + Titan-Sparse's delta restricted to blocks 8-15 (HlowT@16) recovers the one-step skill in every
family (FR 0.85 / 1.00 / 0.97 for L / P / R). Supa + the delta in blocks 16-27 only (HupT@16) recovers none of it
(FR 0.09 / 0.00 / 0.00), even though blocks 16-27 hold 59% of the delta energy.

The question is whether that dispensable upper part carries off-task drift, the representational side of
forgetting. If it does, a delta localised to blocks 8-15 keeps the skill with less interference.

## Arms

Titan-Sparse = Supa + a delta that is exactly zero in the embeddings, the final norm and blocks 0-7 (D-21), and 25%
dense in blocks 8-27. Every arm is "Supa with Titan's blocks in a range":

| arm | Titan blocks | differs from Supa in | equals |
|---|---|---|---|
| supa | none | nothing | Supa (fingerprint a9952c55f5a1 asserted) |
| loc8_15 | 8-15 | blocks 8-15 | run_patch HlowT@16 |
| sparse | 8-27 | blocks 8-27 | Titan-Sparse (fingerprint 818be9c6b6a3 asserted) |
| up16_27 | 16-27 | blocks 16-27 | run_patch HupT@16 (that run built it on a Titan host; same weights) |

**Build.** There is one real model in memory and one GPU lease (`rcwm-t1:localized`).
- Supa is loaded once (runtime.load_arm, FP32, TF32 off).
- Every arm is spliced in place with run_patch's hash-checked `set_block_sources` / `splice_blocks`, in the order
  supa → loc8_15 → sparse → up16_27. That is 28 block copies in total.
- The two cores' configs and generation configs are identical apart from `torch_dtype`.

**Hard gates.** A failed gate exits with code 3 and no stage of that arm runs.
- **G1:** both cores' per-group hashes are read from their safetensors files and must equal run_patch's recorded
  R12 hashes. The loaded Supa must equal its file.
- **G2:** the cores must differ in exactly blocks 8-27.
- **G3:** every spliced block must hash-equal its source.
- **G4:** after each build, every parameter group must hash-equal the core it should hold.
- **G5:** fingerprints a9952c55f5a1 (supa) and 818be9c6b6a3 (sparse).

G1 and G2 run before the lease is requested.

**Reproduction check.** This check warns and records; it is never fatal.
- Each arm regenerates run_patch's first batch per family and compares it with that arm's run_patch text file
  (`gen_base_supa`, `gen_HlowT_16`, `gen_base_sparse`, `gen_HupT_16`). The batch uses the same items, batch size and
  token cap, so the padding is identical.
- An exact match is expected on the first 4 items per family. A mismatch is logged as a WARNING and recorded in
  `ARM.json`.

## Measures (per arm, all against Supa from the same process)

**Off-task.** The 248 neutral prompts in `data/offtask_prompts.jsonl`, run through `offtask.capture_offtask`.
- **First-token KL**, in nats, from the stored fp16 log-probs (clamped at −60, renormalised in float64), in both
  directions:
  - KL(arm‖supa);
  - KL(supa‖arm).

  Each is reported as a mean and a median with a percentile bootstrap CI over prompts (10,000 resamples).
- **Top-1 agreement** on the first token.
- **64-token greedy replies** (`run_t2.gen_prompts`, the same rendering and stop rules as run_t2's offtask_gen):
  - exact agreement with Supa, overall and per reply_class;
  - JSON-bleed rate, as defined in t2_analysis;
  - the graded **prefix divergence** 1 − (common-prefix characters / characters of the longer reply).
- **Answer-slot state shift per capture layer.** The RMS over dimensions of (h_arm − h_supa) / sd_supa, with Supa's
  per-dimension SD floored at 1e-3 × the layer median. Also reported: the relative L2 shift without Supa's top 5
  massive dimensions.

**Skill.** `generation.chain(mode="chain_one")` on the 640 pilot TEST chains (run_t2's list).
- **k=4 normalised regret per family:** `decisions.g_only(..., modes=("chain_one",))`, the same numbers as
  t2_analysis. Each arm is compared with sparse by a paired parent bootstrap (`decide.paired_bootstrap`). The
  copy-chain floor is shown alongside.
- **One-step accuracy from the first chain step.** This step uses the same prompt as run_patch's one-frame item.
  Accuracy is reported on all first steps and on changed ones. FR = (acc − A_supa) / (A_sparse − A_supa) on changed
  first steps, as R12 defines it, with a parent bootstrap CI.

**Key ratios.** drift(arm) / drift(sparse) for arm ∈ {loc8_15, up16_27}. The drift measures are:
- mean KL in both directions;
- exact greedy disagreement;
- greedy prefix divergence;
- top-1 disagreement;
- state shift per layer.

Each ratio has a paired bootstrap CI over prompts (numerator and denominator are resampled together), and the paired
difference arm − sparse is reported next to it.

**Denominator diagnostics.** These are reported next to every ratio. They are caveats and never change the verdict.
- **Quantities measured:**
  - the number of prompts on which sparse flips Supa's top-1 token;
  - sparse's greedy disagreement count and rate;
  - the share of sparse's mean KL carried by the top 10% of prompts.
- **UNINTERPRETABLE flags:**
  - top-1 ratio, if sparse flips fewer than 10 prompts;
  - exact-greedy ratio, if sparse's disagreement rate is above 0.9 (ceiling) or fewer than 10 prompts disagree
    (floor);
  - prefix-divergence ratio, if fewer than 10 prompts disagree.
- **CAUTION flags (KL ratios only):** if sparse flips fewer than 10 prompts, or if the top 10% of prompts carry more
  than 50% of sparse's KL.

## Pre-stated reading (DECISIONS D-23, verbatim)

> "localisation reduces interference" iff the KL ratio loc8_15/sparse has an upper 95% CI < 0.8 AND
> regret(loc8_15) − regret(sparse) ≤ +0.05 in every family. "No reduction" if the ratio's lower CI > 0.9.
> Otherwise inconclusive.

**Operationalisation (fixed here, before the run):**

- **KL ratio** = mean over the 248 prompts of KL(loc8_15‖supa) / mean of KL(sparse‖supa), with the paired
  percentile bootstrap 95% CI over prompts (10,000 resamples, seed 20261008).
  - **Primary direction: KL(arm‖supa).** D-23 does not name a direction, so it is fixed here.
  - The same rule applied to KL(supa‖arm) is reported as **secondary**, with a note on whether the two agree.
- **Regret** = chain_one k=4 normalised regret from `decisions.g_only`. The difference is the paired per-parent mean
  loc8_15 − sparse (`decide.paired_bootstrap` mean_diff, with NaN parents dropped jointly). Its CI is shown but is
  not part of the rule.
  - The difference must be ≤ +0.05 (with 1e-12 float slack) in each of L, P and R.
  - A family without a finite difference fails the regret leg.
- **Edge cases:** a non-finite KL CI satisfies neither KL leg. Equality does not satisfy a strict bound: an upper CI
  of exactly 0.8 is not < 0.8, and a lower CI of exactly 0.9 is not > 0.9.
- **Verdict:**
  - "localisation reduces interference" if upper CI < 0.8 and the regret leg holds in every family;
  - otherwise "no reduction" if lower CI > 0.9;
  - otherwise "inconclusive".
- **Where it is reported:** `localized_analysis.py` computes the verdict mechanically. It is written as
  `d23_verdict` in `LOCALIZED_RESULTS.json` and at the top of `LOCALIZED_RESULTS.md`, with the rule quoted verbatim.

**What the rule does not cover:**
- up16_27 is descriptive. It shows whether the upper delta carries drift on its own, but it does not enter the
  verdict.
- **Low-headroom families.** The regret leg is informative mainly for L and R. In the stored pilot data (Titan-Sparse
  chain_one, k=4; `docs/t2_review/retention_confounds_extra.log`):

  | family | sparse regret | copy chain | headroom |
  |---|---|---|---|
  | L | 0.507 | 1.011 | 0.50 |
  | P | 0.879 | 0.955 | 0.076 |
  | R | 0.386 | 1.007 | 0.62 |

  For P the +0.05 tolerance can barely fail. Any family whose headroom (copy − sparse) is below 0.10 is therefore
  marked low-headroom, and its one-step FR on changed first steps is shown beside the regret leg. That FR is the
  skill evidence for P. The mark is a caveat; it does not change the verdict.
- **Stored-chain consistency.** The stored pilot chains predate the 160-token cap and the D-14 batch budget (Supa's
  include 60 truncated texts). An `identical_chains` value below 1 against them is expected and is not a splice
  failure.
- **Limits:**
  - chain_m1, P's primary frame in T2, is not generated.
  - There is one greedy decode per prompt.
  - Pilot TEST was seen in development, and the 8/16 cut was chosen in R12.
  - The thresholds were stated in advance but are not a preregistration.
  - No DECISIONS entry was added, because DECISIONS.md is frozen for this job.

## Scheduling and cost

- **Gate (D-23):** the job starts only after the T2 s0 pair has finished. `--wait-for runs/t2/t2_s0/eval.json`
  blocks before torch is imported, and before the file hashing and the lease. It polls every 60 s, with no time
  limit unless `--wait-max-h` is given; on expiry it exits 4 with nothing loaded or written.
- **Size of the job:** 4 arms. Each arm has 248 off-task replies, up to 4 × 640 chain steps and 48 reproduction
  generations.
- **Runtime at 2.7 greedy generations/s (`--plan` on the real pilot):**

  | case | generations | hours |
  |---|---|---|
  | upper bound | 11,424 | 1.18 |
  | expected | 11,194 | 1.15 |
  | low | 9,768 | 1.00 |

  - **Expected:** sparse and loc8_15 are taken at Titan's stored 2,445 steps, and supa and up16_27 at the full
    2,560, because Supa's stored 1,134 steps predate the 160-token cap.
  - **Low:** uses Supa's stored step count.
- **Overhead:** about 5 min for the load, hashing both core files (about 3.6 GB read), the hash passes and 124
  off-task forward batches.
- **Plan for:** about 1.25 h of GPU time (1.2–1.3 h), not D-23's estimate of about 40 min. At R12's measured
  2.7–4.9 generations/s, it would be about 0.75–1.25 h including overhead.
- **Disk:** about 0.31 GB.

## Outputs and resume

The run writes to `runs/bank_pilot/localized/<arm>/`:
- `ARM.json`: the build record, group hashes, fingerprint and reproduction check;
- `STAGES.json`: per-stage records with the sha256 of each output file;
- `offtask_meta.json`, `offtask_logprobs.npy`, `offtask_last.npy`;
- `offtask_gen.jsonl`;
- `gen_chain_one.jsonl`.

**Run-level files:**
- `RUN.json`: the settings, locked across invocations;
- `DONE.json`: written only when all 12 arm-stages are done.

**Resume.** Each arm and stage resumes independently. A stage counts as done only if its record carries the arm's
fingerprint and its file hashes still match.

**Analysis.** It runs on CPU (numpy only, never imports torch) and refuses to run if any file differs from
`STAGES.json`. It writes `docs/localized/LOCALIZED_RESULTS.{md,json}`.

## Commands

From the RCWM-T1 directory, with `CUDA_VISIBLE_DEVICES` unset. Per the sharing protocol, message the peer first and
launch hidden via WMI, like the other `launch_*.cmd` jobs:

```
cd /d C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1
set PYTHONIOENCODING=utf-8
C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe scripts\run_localized.py --device cuda --out-name localized --wait-for runs/t2/t2_s0/eval.json >> runs\bank_pilot\localized_log.txt 2>&1
```

After `runs/bank_pilot/localized/DONE.json` exists:

```
set PYTHONIOENCODING=utf-8
C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe scripts\localized_analysis.py
```

The analysis sets `CUDA_VISIBLE_DEVICES=""` itself and never imports torch.

## Verification (tiny models only; no real model loaded, GPU never touched)

`scripts/localized_test_tiny.py` passes 70/70 checks on CPU in 75–79 s, with peak RSS 996 MiB. It uses two tiny
random 28-block Qwen3 stand-ins with BF16 / FP32 safetensors, and a real run_patch.py run on the same pair supplies
the reference texts and hashes. The checks cover:

- **Hashes and build gates:**
  - file-derived hashes equal `group_hashes`;
  - G1, G2 and G3 trip as designed;
  - every arm's groups hash-equal the intended core, and each arm differs from Supa in exactly the intended blocks;
  - fingerprints equal independently assembled models;
  - run_patch's texts are reproduced exactly for all four arms.
- **Crash and resume:** an interrupted run resumes with exactly the pending stages; a fresh run is byte-identical to a
  resumed one; one Supa load; DONE.json is written only at the end.
- **CUDA path:** stubbed, it uses one lease, `rcwm-t1:localized`, and its output is byte-identical to the CPU run.
- **Analysis:**
  - KL is 0 for supa vs supa and positive for the other arms;
  - regret equals `g_only`, and arm − sparse equals `paired_bootstrap`;
  - the denominator counts and flags match independent computation;
  - the D-23 verdict matches the rule applied independently to the reported numbers;
  - end to end, a no-drift loc8_15 gives "localisation reduces interference" and a full-drift loc8_15 gives "no
    reduction";
  - boundary cases of the rule are covered as pure functions.
- **Gate:** a file that appears later releases the wait. A missing file with `--wait-max-h` exits 4 before torch is
  imported, with nothing loaded or written.

Code as of this design:
- run_localized.py sha256 `2dfff5f3993834d0008d2f30b279b03eaf92e6b489f95f24bb0cc64ce413b469`
- localized_analysis.py sha256 `036043e746ed6d83e8f583d3f74e7750ea2830e920e9c76bc4cd300fbd33e514`
- localized_test_tiny.py sha256 `174bb2518531efdb2e27849ca5b239d8cf982e2c796bc3a209cb35f6d0e48b30`
