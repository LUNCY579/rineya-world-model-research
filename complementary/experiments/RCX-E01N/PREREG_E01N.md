# PREREG RCX-E01N: counterfactual rule-varied LoRA training makes Titan-Sparse read supplied ledger rules (SEALED confirmation)

**Template:** bundle `06_templates/PREREGISTRATION.md`.
**Supersedes:** `PREREG_E01N_DRAFT.md`.

**Status:** **FROZEN** when this file's sha256 is recorded in DECISIONS (CX D-26), together with the SEALED commitment.

**What was known at the freeze** (DEV battery seed 20261240; R-03, R-05, R-06): DEV results for OFF, T2 and seed 0 of
N-VR and N-CFSD, including both confirmatory contrasts (N-CFSD − N-VR, 280 parents, one seed each):
- C1 +0.020 [+0.002, +0.039] (N-VR 0.964);
- C2 +0.099 [+0.075, +0.125];
- F0 posted twins −0.015 (0.980 vs 0.995);
- format errors 0/2,240 in both arms;
- the per-wording `ho_tpl` breakdown.

This file was finalised after the N-CFSD s0 DEV result (2026-10-11 01:09). The DEV battery is spent for confirmation
and is never reused as one.

**Changes from `PREREG_E01N_DRAFT.md`** (committed 5e05a5b, 2026-10-10 20:33).
- **Before any N-CFSD output** (CX D-25; outcomes known: yes, N-VR s0 only): C2 (`ho_tpl`) was made a co-primary under
  Holm.
- **After the N-CFSD s0 DEV result** (outcomes known: yes):
  - ADVANCE may rest on either endpoint (the draft's table named only the primary), with the other endpoint's lower
    bound > −0.05 and the per-seed condition;
  - STOP needs both endpoints (the draft needed the primary only, which at C1's ceiling would have made STOP likely);
  - REVISE needs Holm p < 0.05;
  - the outcomes are tested in a stated order;
  - N-FX is reduced to one descriptive seed, and the draft's secondary Holm family of 3 and its secondary hypothesis
    are dropped;
  - "parse rate" is replaced by the format-error rate (+0.02), and both protected strata are judged by interval;
  - seed 0 of N-VR and N-CFSD reuses the DEV-pilot adapters;
  - added: the p-value definition, bootstrap seed 20261236, the train-overlap exclusion and the seed-0 sensitivity.
- **Unchanged:** the contrast and its direction, the composite, the +0.10 practical effect, 3 seeds per confirmatory
  arm, the recipe and the training rows.

**Execution scope:**
- CPU only, frozen FP32 Titan-Sparse, unmerged adapters.
- No publication, promotion or merge.
- E33 retention (RCWM D-36) is requested only for an arm that reaches ADVANCE.

## 1. Claim and scope

- **H01-N.** At a matched training budget (1,501 rows, 190 × 16 steps, D-03 recipe), a LoRA trained on **varied
  in-context ledger rules with counterfactual twins, nonce statuses and paraphrased rule wording** (N-CFSD) executes
  supplied rules better than one trained on **ordinary varied rules** (N-VR, canonical wording, real words, no twins).
- **Model and tier.** Titan-Sparse-Experimental 1.0.0, FP32, sha a28629c9…. Tier C: a component student (frozen base
  plus an unmerged LoRA), not a dense checkpoint.
- **Practical effect.** +0.10 on a confirmatory endpoint.
- **A positive result would not establish:**
  - general instruction following;
  - reading rule wordings other than the two held-out templates h0/h1 that C2 measures;
  - transfer to other families or natural-language constraints (E03);
  - retained general ability (E33 is a separate gate);
  - any dense-model change.

## 2. Custody

**Code (sha256 recorded at freeze in DECISIONS CX D-26):**
- `rcx/native/lora_cpu.py`
- `experiments/RCX-E01N/native_e01.py`
- `experiments/RCX-E01N/native_probe.py`
- `experiments/RCX-E01N/analyze_native.py`
- `rcx/rcl/{world,gen,vocab,templates,score}.py`
- `rcx/stats.py`

**Training rows.** `runs/RCX-E01N/cells/train_{N-VR,N-CFSD,N-FX}.jsonl`, sha256s as in `TRAIN_MANIFEST.json`, audited
by `AUDIT.json`. The same rows serve every seed; seeds change only the row order and the LoRA initialization
(`row_order_for(seed)`, `attach_lora(seed)`).

**Adapters.**
- Seed 0 for N-VR and N-CFSD are the DEV-pilot adapters, which already exist: sha256 `2f376882…` and `97cc48e8…`
  (full values in `TRAIN_META.json`).
- Each seed-0 adapter was trained once, in one run, before any evaluation of it. There was no selection among
  candidates.

  | Adapter | Trained | DEV eval |
  |---|---|---|
  | N-VR_s0 | 2026-10-10 17:09 | 20:32 |
  | N-CFSD_s0 | 23:09 | 2026-10-11 01:09 |

  Code: `lora_cpu.py` `ca469a13…` (unchanged) and `native_e01.py` `f023c201…` (84be407). Its training code
  (`native_rows`, `cmd_train`, constants) is identical in the frozen file; only evaluation, SEALED and CLI code changed.
- Seed 0 is DEV-gated: this confirmation, and C2, were set after seeing seed 0 on DEV (see the seed-0 sensitivity in
  §4).
- Seeds 1 and 2 are trained after the freeze under the frozen code, and are evaluated on no battery before open-sealed.

**SEALED battery.**
- Built as `gen.make_battery("sealed", 40, seed)` in the arith world, surfaces `in` and `ho_tpl`.
- `seed = int(sha256(salt)[:8], 16) % 10**8`.
- **Mint.** The salt is drawn by:

  ```
  native_e01.py mint-commit --out runs/RCX-E01N/cells --arms-required N-VR_s0,N-VR_s1,N-VR_s2,N-CFSD_s0,N-CFSD_s1,N-CFSD_s2 --n-parents 40 --surfaces in,ho_tpl
  ```

  `SEALED_COMMIT.json` holds:
  - `sha256(salt)`;
  - a salted commitment to the battery content, `sha256(salt + ":" + sha256(battery))`;
  - the shas of `rcx/rcl/{gen,vocab,templates,world}.py` and `native_e01.py`;
  - the required adapters.

  It is committed with this file. `sha256(salt)` and the `SEALED_COMMIT.json` sha256 are recorded in D-26. The salt
  stays in `SEALED_SALT.secret` (git-ignored by `*.secret`), and nothing but `open-sealed` reads it.
- **Open.** `open-sealed --expect-commit-sha <D-26 value>` runs once (`SEALED_OPEN.json` is write-once).
  - It opens only when all six adapters are completed, untampered runs:
    - `TRAIN_META.json` is present;
    - the adapter sha equals `TRAIN_META`;
    - the rows sha equals `TRAIN_MANIFEST.json`;
    - the run has 190 steps and no resume checkpoint;
    - the regenerated battery matches the commitment.
  - It records the salt, seed, battery sha, adapter shas, and the battery's overlap with every training-row file.
  - `SEALED_OPEN.json` is committed, with its sha256 logged in DECISIONS, before the first SEALED evaluation.
- **Evaluate.** `eval --battery sealed` refuses:
  - before opening;
  - on other surfaces;
  - for an adapter whose sha differs from the one recorded at opening;
  - for any N-VR/N-CFSD cell not fixed at opening.
- **Limit.** The commitment proves the battery was fixed at the freeze. It cannot stop the salt being read early; doing
  so is a protocol violation.
- **SEALED pool words** (statuses and rule sets of F2, F3 and the V strata) come from `vocab.pools("sealed")`. They are
  disjoint from TRAIN and DEV, and from PREREG-3 and the factory, with cross-split edit distance ≥ 2.
- **Shared with DEV by design** (none of these occurs in training; see AUDIT):
  - the anchors posted and void;
  - the F4 near-miss probes;
  - the five F5 posted-like strings (F5 is in the composite);
  - the `ho_tpl` templates h0 and h1.
- **Training overlap.** SEALED parents that share a semantic or skeleton hash with any training row (`train_overlap` in
  `SEALED_OPEN.json`) are dropped from every arm before analysis, and the count is reported. Four random SEALED seeds
  checked at the freeze gave 0.

## 3. Arms and fairness

| Arm | Seeds | Role |
|---|---|---|
| N-CFSD | 0, 1, 2 | treatment |
| N-VR | 0, 1, 2 | matched control (same rows count, steps, recipe and frame) |
| OFF (frozen core) | — | tier A reference |
| T2 s0 (registered) | — | reference |
| N-FX | 0 only | descriptive fixed-rule control (same generator); not in any test |

**Covariates.** For seed 0, N-CFSD has −11.4% supervised tokens vs N-VR (58,223 vs 65,726; −11.5% vs N-FX), and 50% of its rows have 3 accounts. Every
arm has the same number of oracle labels (1,501).

**No tuning.** All hyperparameters are T2's, and no DEV-informed change was made to any training recipe.
- Rows were built (06:29) and audited (06:46) on 2026-10-10, before any native training or evaluation.
- The training code is unchanged since 84be407, and `rcx/rcl` is unchanged since b128150.

## 4. Endpoints

**Confirmatory family (Holm, 2 tests).** Both are N-CFSD − N-VR:

| ID | Endpoint |
|---|---|
| **C1** | parent-macro exact transition accuracy on the composite {F2, F3, F5, V1, V2, V3, V5}, surface `in` (its 280 parents pooled, equal weight) |
| **C2** | the same composite on surface `ho_tpl` (the two held-out rule templates h0/h1: never in training; the same wordings as on DEV) |

- **Estimator.** Seed × parent paired cluster bootstrap (`rcx.stats.seed_parent_bootstrap_test`, which gives the same
  point estimate and CI as `seed_parent_bootstrap`).
  - Per parent, average the variants within each seed, then average the three seeds.
  - Resample parents (not seeds), with 10,000 resamples and bootstrap seed 20261236. The CI is the 95% percentile
    interval.
  - Inference is conditional on the six adapters, the SEALED word pools and the held-out templates. Seed variation
    enters through the per-seed condition (§5) and the seed analyses below.
- **p-value.** For each contrast, two-sided, from the same replicates as its CI: 2 × min(share of replicate
  differences ≤ 0, share ≥ 0).
  - Replicates equal to 0 (within 1e-12) count on both sides.
  - Floored at 1/10,000 and capped at 1.
  - Without zero replicates this equals 2 × min(share ≤ 0, share > 0), and p < 0.05 implies that the 95% CI excludes
    0.
- **Holm.** Applied at α = 0.05.

**Protected strata** (ADVANCE guards, not in the Holm family). Each is N-CFSD − N-VR by the same bootstrap, resamples
and seed, and non-inferiority is judged by interval:
- **F0 posted twins:** parent-macro exact on the `twin_posted` variants of the F1–F5 parents (200), surface `in`.
  Within margin if the 95% CI lower bound is > −0.05.
- **Format-error rate:** the parent-macro share of outputs that `native_probe.parse_answer` cannot parse, over all
  SEALED records (both surfaces, all strata). Within margin if the 95% CI upper bound is ≤ +0.02.

Unparseable outputs score 0 (exact and decision) in every endpoint.

**Secondary (descriptive):**
- each stratum and the error directions;
- the no-op-when-required rate (Wilson);
- V3 polarity sensitivity and both-twins exact;
- the PREREG-3-style verdict per arm;
- N-CFSD − T2;
- N-VR − OFF;
- per-seed values and the seed spread, plus the matched per-seed contrasts of C1 and C2 with a t interval (df = 2);
- seed-0 sensitivity: C1 and C2 on seeds 1–2 only, with the same bootstrap. The report states whether the qualifying
  endpoint still has point ≥ +0.10 and CI lower bound > 0 there;
- held-out-wording breakdown (h0 and h1 by polarity);
- M1/M2.

## 5. Decision rule (applied to C1 and C2 after Holm)

"CI" means the 95% percentile interval of §4, not multiplicity-adjusted. Thresholds are compared with a 1e-9
tolerance. The outcomes are tested in the order below, and the first that holds is the decision.

1. **ADVANCE.** At least one of C1 or C2 meets all of:
   - Holm-adjusted p < 0.05;
   - point estimate ≥ +0.10;
   - CI lower bound > 0;
   - each matched per-seed contrast (N-CFSD sₖ − N-VR sₖ, k = 0, 1, 2) > 0.

   And, in addition:
   - the other endpoint's CI lower bound is > −0.05. A significant drop within this margin does not block ADVANCE; it
     is reported as a trade-off;
   - both protected strata are within margin.
2. **STOP.** Both endpoints have CI upper bounds < +0.05.
3. **REVISE.** At least one endpoint has Holm-adjusted p < 0.05 with point estimate > 0. This includes an endpoint that
   reaches +0.10 while an ADVANCE condition fails; the report names which condition.
4. **INCONCLUSIVE.** Anything else.
   - It is always the outcome when any of the six N-VR/N-CFSD SEALED cells is missing. C1 and C2 are then reported
     descriptively on the seeds both arms have.
   - A missing OFF or T2 cell does not affect the decision.

**Pre-declared notes** (known at the freeze, stated here so they cannot be reinterpreted later):
- **Ceiling.** On DEV seed 0, N-VR scored 0.964 on C1 (+0.036 of headroom). C1 can therefore hardly reach +0.10, and
  its upper bound will probably be < +0.05. ADVANCE and STOP effectively rest on C2, and C1 acts as a guard or yields
  REVISE.
- **Power.** On DEV, C2 was +0.099, right at the threshold.
  - If the SEALED effect equals it, P(C2 point ≥ +0.10) ≈ 0.5 whatever the seed variance.
  - P(ADVANCE) ≈ 0.3–0.45 once the guards are included, and REVISE is the other likely outcome.
  - The +0.10 threshold predates any N-CFSD output and is not changed.

## 6. Execution and safety

**Order:**
1. mint-commit at the freeze (command in §2) with the frozen code. Record sha256(salt) and the `SEALED_COMMIT.json`
   sha256 in D-26, and commit `SEALED_COMMIT.json` with this file (never the salt);
2. train N-VR s1, N-CFSD s1, N-VR s2 and N-CFSD s2;
3. `open-sealed --expect-commit-sha <D-26 value>`; commit `SEALED_OPEN.json` and log its sha256;
4. evaluate the six adapters plus OFF and T2 on SEALED;
5. analyse once with `analyze_native.py --battery sealed`. It refuses unless every eval names the rebuilt battery's
   sha256, every N-* adapter matches open-sealed's record, and every stored score equals a re-score of the raw output.
   Its outputs are create-only;
6. optionally, train N-FX s0 (after opening; not in the commitment, and labelled as such) and evaluate it on DEV and
   SEALED, descriptively, in a separate addendum (`--out runs/RCX-E01N/ANALYSIS_SEALED_NFX.md`).

**Constraints:**
- single CPU lane, ≤ 4 threads, `rcx.guard`;
- ≥ 12 GB of free RAM before any model load;
- resumable training checkpoints guarded by a run fingerprint;
- training refuses unaudited rows.

**Failures.**
- A crashed training run resumes under the same fingerprint. A crashed evaluation is rerun; it is complete only when
  its eval file is written.
- A cell that cannot be completed is reported as missing and is never replaced by a different seed.
- open-sealed needs all six adapters, so a training cell that cannot be completed leaves the battery closed and the
  outcome INCONCLUSIVE. Opening it then for descriptive use is a logged deviation.
- A SEALED evaluation that cannot be completed is analysed with `--allow-missing` and gives INCONCLUSIVE (§5).

## 7. Freeze and deviations

The freeze sha256 and timestamp, the SEALED commitment hash and the code shas are recorded in DECISIONS CX D-26. Every
deviation is logged there with "outcomes known: yes/no". Nothing is rerun under this ID. Any of the following voids the
confirmation:
- re-creating or deleting `SEALED_COMMIT.json`, `SEALED_SALT.secret` or `SEALED_OPEN.json`;
- retraining an adapter after opening.
