# PREREG RCX-E01N (DRAFT, not frozen): does counterfactual rule-varied LoRA training make Titan-Sparse read supplied ledger rules?

**Template:** bundle `06_templates/PREREGISTRATION.md`.

**Status:** DRAFT.
- The DEV pilot (seed 0, DEV battery seed 20261240) may inform only:
  - instrument validity (floors, ceilings, parse rates);
  - feasibility (time, RAM).
- It may not change the primary comparison or its direction.
- The draft is frozen, with its sha256 recorded in DECISIONS.md, *before* any SEALED battery is minted.

**Execution scope:**
- CPU only, on frozen FP32 Titan-Sparse.
- Adapters stay unmerged. There is no publication and no promotion.
- **Retention (E33).** Before any capability claim, the retention confirm runs through RCWM's D-36 suite, in an RCWM GPU
  turn at their discretion.

## 1. Claim and scope

- **Hypothesis H01-N.** At a matched budget (1,501 rows, 190 × 16, D-03 recipe), a LoRA trained on **varied in-context
  rules with counterfactual twins, nonce statuses and paraphrased rule wording** (N-CFSD) executes supplied rules better
  on fresh vocabulary and inverted rules than a LoRA trained on **ordinary varied rules** (N-VR).
- **Secondary.** Both rule-varied students beat the **fixed-rule** student (N-FX, a T2-like data recipe) and the
  registered T2 on the rule-reading composite. N-FX is expected to show PREREG-3's TRIGGER signature on fresh words.
- **Model.** Titan-Sparse-Experimental 1.0.0 (FP32, sha a28629c9…) with LoRA r16/α32 on q,k,v,o,gate,up,down. Tier
  **C**: a trained student, component artifact (frozen base plus unmerged adapter), *not* a dense checkpoint.
- **Practical effect.** +0.10 on the primary composite.
- **What a positive result does not establish:**
  - broad natural-language instruction following;
  - transfer to other families;
  - retained general ability (that is E33's job);
  - a dense-model improvement.

## 2. Custody

**Code commit at freeze:** `<sha>`.

**Pinned inputs:**
- `rcx/native/lora_cpu.py` and `experiments/RCX-E01N/native_e01.py`;
- `rcx/rcl/{world,gen,vocab,templates,score}.py`;
- `runs/RCX-E01N/cells/train_N-*.jsonl` and `AUDIT.json`.

**Teacher.** The exact interpreter, qualified in `RCX-E01/qualification/QUALIFICATION.json`. T2 never labels anything.

**SEALED battery.**
- Built as `make_battery("sealed", N, seed)` in the arith world.
- `seed = int(sha256(salt).hexdigest()[:8], 16) % 10**8`.
- `sha256(salt)` is committed at freeze, and the salt is revealed only after all adapters are trained.
- Surfaces: `in` and `ho_tpl`.
- SEALED words are disjoint from TRAIN and DEV.

**Spent:** the DEV battery 20261240 (pilot), and all RCWM and BWM instruments (none are used).

## 3. Arms and fairness

| Arm | Data | Role |
|---|---|---|
| OFF | none | tier A |
| T2 s0 | registered | reference (RCWM) |
| N-FX | fixed rule | matched control |
| N-VR | ordinary varied | matched SFT (same rows, steps and tokens up to the reported covariates) |
| N-CFSD | counterfactual / SD | treatment |

- **Covariates** (seed 0), reported:
  - supervised tokens: N-CFSD −11.5% vs N-FX;
  - account count: 50% of N-CFSD rows have 3 accounts.
- **Seeds.** 3 per trained arm, matched by `row_order_for(seed)` and the LoRA init seed.
- **No tuning.** Hyperparameters are T2's.

## 4. Endpoints

- **Primary.** Parent-macro exact transition accuracy on the composite {F2, F3, F5, V1, V2, V3, V5}. The surface is
  `in`, which is Titan's canonical frame with the rule sentence in canonical wording.
- **Primary comparison.** N-CFSD − N-VR, by seed × parent paired cluster bootstrap with 10,000 resamples.
- **Co-primary** (added 2026-10-10 after the N-VR DEV pilot, CX D-25, before any N-CFSD output existed): the same
  contrast on surface `ho_tpl` (held-out rule paraphrases). It is Holm-corrected with the primary over a family of 2.
  - Rationale: N-VR reaches 0.964 on `in` (ceiling) but 0.594 on `ho_tpl`.
  - This is an instrument-headroom decision taken without N-CFSD data.
- **Secondary (Holm family of 3):**
  - N-VR − N-FX;
  - N-CFSD − T2;
  - N-CFSD − N-VR on surface `ho_tpl` (held-out paraphrases).
- **Protected (non-inferiority, margin −0.05):**
  - F0 posted twins;
  - parse rate.
- **Descriptive:**
  - every stratum;
  - error direction;
  - the PREREG-3-style verdict per arm (RULE / TRIGGER / MIXED on F-strata probes);
  - V3 both-twins exact;
  - V6 compositions;
  - M1/M2 magnitudes.

## 5. Decision rule

| Outcome | Condition |
|---|---|
| **ADVANCE** (request E33 retention) | primary diff ≥ +0.10 with CI lower bound > 0, and protected strata within margin |
| **REVISE** | diff > 0 with CI > 0, but < 0.10 |
| **STOP** | CI upper bound < +0.05 |
| **INCONCLUSIVE** | anything else |

## 6. Execution and safety

- Single CPU lane, at most 4 threads.
- `rcx.guard`, plus a ≥ 12 GB RAM floor before any load.
- Resumable checkpoints guarded by a run fingerprint.
- The audit gate must pass before training.
- No GPU use without a written request and both peers' agreement.

## 7. Freeze and deviations

The freeze hash and timestamp go here. All deviations are logged, and nothing is rerun under the same ID.
