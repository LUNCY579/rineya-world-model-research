# PREREG RCX-E01R (DRAFT): complement-targeted LoRA recipes make Titan-Sparse decide the unlisted case of unseen restrictive positive rule wordings (SEALED confirmation on P3)

**Template:** bundle `06_templates/PREREGISTRATION.md`. Structure, statistics and custody follow the frozen
`experiments/RCX-E01N/PREREG_E01N.md`.
**Design:** `experiments/RCX-E01R/DESIGN.md` (draft).

**Status:** **PARTLY FROZEN** (2026-10-11, about 16:25). §2 and the R1/R2/R3 specifications are frozen (CX D-37); the
rest is a DRAFT for review. The whole file becomes FROZEN only when
its final sha256 is recorded in DECISIONS (a new CX D-xx entry) together with the SEALED commitment. Markers:
- **PENDING STEP A**: depends on the step A DEV2 diagnostic (DESIGN §3). None of these places may be filled except
  mechanically, by the rule in §2.
- **PENDING BUILD / PENDING FREEZE**: hashes and counts of code and rows that do not exist yet.

**What was known when this draft was written:**
- E01N: R-05, R-06 and R-08, including the per-wording breakdown. N-CFSD (3 seeds) decides no-op on held-out (`ho_tpl`)
  wordings for:
  - **146/1,053 (0.14)** of the positive-rule (S+) items that need one;
  - 490/507 (0.97) of the negative-rule (S−) items that need one.

  Apply items are 1,120/1,173 (0.95) for S+ and 359/387 (0.93) for S−.
- The 45 P2_v2 wordings (sha256 `7e625a59…`) have been read by the lead (the CX D-34 stem pass).
- The DEV2 battery exists (seed 20261261, sha256 `38362338…`).
- **No step A output exists.** The step A chain has been waiting for RAM before its first cell, N-CFSD_s0
  (`step_a_chain.log`, 15:31–15:56). No DEV2 evaluation has started.
- P3_v2 has never been read by the lead or by any analysis. It holds 48 wordings, sha256 `88dd432b…`. Only its per-cell
  counts are known (`POOL_REFINE.json`).

**FROZEN NOW (CX D-37), before any step A output exists:** §2 (the arm choice, computed by
`experiments/RCX-E01R/decide_branch.py`) and the recipe specifications R1, R2 and R3 in §3 that §2 selects.
- This file is committed and its sha256 is logged in DECISIONS (CX D-37), together with `decide_branch.py`'s sha.
- That happened while `step_a_chain.log` showed no START line and no DEV2 prediction file existed.
- Any later change to §2 or to the R1/R2/R3 specifications is a deviation with "outcomes known: yes".
- §4–§11 stay under review until the full freeze. Changes there must not use treatment outcomes; none exist before the
  SEALED analysis.

**Proposed refinements of DESIGN.md (for review):**
1. **Operational sub-readings.** §2 gives operational criteria for the step A sub-readings. DESIGN §3 names "EXPLICIT >
   IMPLICIT" and "EXPLICIT ≈ IMPLICIT" without a criterion. Two branches are added: an instrument anchor check and a
   near-ceiling branch.
2. **Separate arms.** The two treatment recipes are trained as separate arms, never combined into one arm.
3. **R1 covers both polarities.** R1's training pool is authored for both polarities, not only positive rules. Novel
   wordings for positive rules only would make "unfamiliar wording" a cue for positive polarity (§3).
4. **R2's twins could be mirrored (open).** Proposed: mirror R2's complement pairs for negative-rule parents, so that
   "unlisted" alone never predicts the target (§3). The parallel `recipes.make_r2` currently adds them for S+ parents
   only.
5. **ADVANCE rests on C1.** ADVANCE rests on the targeted endpoint C1. C2 must also be significantly positive (§7).
6. **P3 is not read at mint.** The SEALED commitment binds a P3-free skeleton plus the P3 file sha. The battery builder
   therefore reads P3 only at opening (§4).

**Execution scope:**
- CPU only. Frozen FP32 Titan-Sparse, with unmerged adapters.
- No publication, promotion or merge.
- E33 retention is requested from RCWM only for an arm that reaches ADVANCE (RCWM R27).

## 1. Question and claim

**Question.**
- N-CFSD reads supplied ledger rules on its training wordings and on most held-out wordings.
- On unseen *restrictive positive* wordings, it decides an unlisted status as "apply" (R-08). That error,
  `applied_when_noop`, accounts for 286–350 of each N-CFSD cell's `ho_tpl` errors.
- Can a training recipe aimed at the complement case fix this, at N-CFSD's training budget, without trading away what
  N-CFSD already does? The candidate recipes are:
  - complement coverage (R1);
  - complement-focused wording twins (R2);
  - a predicate-first trace target (R3).
- The test is on unseen, blind-authored positive wordings.

**H01-R (one hypothesis per treatment arm X).**
- At a matched budget, an adapter trained with recipe X decides required no-ops under positive rules on P3 wordings
  more accurately than N-CFSD (C1).
- The improvement also shows in exact transitions (C2).
- The protected strata stay within margin.
- Matched budget means 1,501 rows, 190 × 16 steps, the D-03 recipe and the same training seeds.

**Model and tier.**
- Titan-Sparse-Experimental 1.0.0, FP32, sha `a28629c9…`.
- Tier: native-C. Each adapter is a component student (a frozen base plus an unmerged LoRA), not a dense checkpoint.

**Practical effect.** +0.25 on C1 (justified in §7).

**What a positive result (ADVANCE for arm X) would establish, within this synthetic ledger task family:**
- At this budget, recipe X makes the adapter decide the complement case of restrictive positive rules correctly
  substantially more often than N-CFSD. This holds on wordings that no model or analysis has seen.
- It does so while keeping, within the stated margins:
  - training-wording performance;
  - posted twins;
  - output format;
  - apply decisions;
  - negative-rule no-ops.

**What it would not establish:**
- **Reading of arbitrary natural-language rule wordings.**
  - P3 comes from one blind-authoring protocol: three registers, label-blind verification (CX D-34).
  - Inference is conditional on its 48 templates, of which 23 are positive.
  - If R1 is trained, its pool comes from the same protocol (§3). An R1 gain is therefore transfer *within that
    authoring distribution*.
- **Reading of compositions** (no V6 stratum on `p3`), transfer to other families, or natural-language constraints
  (E03).
- **That the predicate step causes an R3 gain.** R3 also adds output tokens before the answer, and there is no
  token-matched filler arm (§8).
- **Retained general ability.** E33 is a separate gate, run by RCWM and only after ADVANCE.
- **Any dense-model or merged-model change.**
- **That a STOP or REVISE rules a recipe out at other budgets or row shares.**

## 2. Arm choice from step A (FROZEN, CX D-37; before any step A output exists)

**Computation.**
- Run `python -B experiments/RCX-E01R/decide_branch.py` once, after the step A chain ends. It writes
  `runs/RCX-E01R/dev2/BRANCH_STEP_A.json` (create-only).
- It reads the DEV2 battery (sha `38362338…`, checked) and the raw predictions of **N-CFSD s0, s1 and s2**. Each
  `eval_<tag>.json` must name the battery sha. It does not rely on `DIAG_DEV2.json`.
- N-VR s0 is descriptive only.

**Quantities.** All are pooled over the three seeds and use the required-no-op items under **positive** rules (S+) on
surface `p2`:
- p_f = k/n per form: LF = EXPLICIT_LISTED_FIRST, CF = EXPLICIT_COMPLEMENT_FIRST, IM = IMPLICIT;
- p̄, pooled over the three forms;
- E, pooled over LF and CF; I = p_IM;
- S = max_f p_f − min_f p_f.
- **Bootstrap.** The 95% interval of E − I uses a template-then-parent cluster bootstrap:
  - resample P2 templates within each group with replacement, then parents within each sampled template;
  - the statistic is the count ratio;
  - 10,000 resamples, seed 20261267.
- **Reported only:** q_f and q̄, the same quantities for negative rules.
- **Instrument anchor:** a_in, the no-op accuracy on `in` (both polarities).

**Branches.** The branches are tested in this order, and the first match is the branch. Thresholds use a 1e-9
tolerance.

| # | Condition | DESIGN §3 reading | Treatment arms | Then |
|---|---|---|---|---|
| B0 | Any of: an N-CFSD seed missing; a battery sha mismatch; any n = 0; a_in < 0.85 | instrument fault | none | No mint and no training. The step A instrument is investigated and logged. R-08 SEALED gave 0.97 on `in`. |
| B1 | min_f p_f ≥ 0.80 | **H-alt** | none | The P3 coverage check (§2.1); RCM then moves to E02 |
| B3 | E − I ≥ 0.15 **and** the bootstrap lower bound of E − I > 0 | **H-E01R, EXPLICIT > IMPLICIT** (form-specific deficit) | **R1 and R3** | §3–§8 |
| B2 | p̄ ≥ 0.65 | near ceiling, uniform | none | P3 stays sealed. Documented; RCM moves to E02. |
| B4 | otherwise | **H-E01R, EXPLICIT ≈ IMPLICIT, both low** (uniform deficit), or a non-directional intermediate | **R2 and R3** | §3–§8 |

**Rationale.**
- **R3 in every training branch.** RCWM R23 and BWM D-54 both favour an explicit predicate step. R3 also leaves the
  training inputs byte-identical to N-CFSD's.
- **B3.** A directional deficit, explicit wordings read but implicit ones not, is a coverage gap. The direct fix is
  new training wordings (R1).
  - B3 is tested before B2, so a strong form-specific deficit is never hidden by a high pooled rate.
  - The cluster bootstrap guards against template idiosyncrasy: each form has only 6–9 P2 templates.
- **B4.** A uniform deficit gives no form to aim at. R2 instead trains the complement decision densely and invariantly
  across wordings.
- **B2.** With p̄ ≥ 0.65, the remaining headroom is below 0.35, which makes C1's +0.25 threshold (§7) impractical at
  about 20 h of CPU lane.

**Fallbacks** (fixed now; each is logged in DECISIONS with "outcomes known: step A only"):
- **B3, R1 short of wordings.** If the R1 pool falls below its minimum after the audit and one blind top-up round,
  R1 is replaced by R2 before the mint.
- **A failed row audit.** If an arm's rows fail the training-row audit for a reason that cannot be fixed without
  changing the recipe, the arm is dropped before the mint, and the Holm family shrinks to the remaining arm.
- **After the mint,** no arm is added, replaced or re-specified.

> **PENDING STEP A** (filled from `BRANCH_STEP_A.json` only): branch = ___; p_LF / p_CF / p_IM = ___; p̄ = ___;
> E − I = ___ [lo, hi]; S = ___; q_f = ___; a_in = ___; treatment arms = ___.

### 2.1 P3 coverage check (branch B1 only)

**Procedure.**
- No training.
- Mint the SEALED battery of §4 with `--arms-required N-CFSD_s0,N-CFSD_s1,N-CFSD_s2`, then open it.
- Evaluate N-CFSD s0–s2 and OFF, and analyse once.

**Pre-stated reading.** "H-alt holds on P3" if all of the following hold for the C1-level quantity (N-CFSD's POS no-op
decision accuracy on `p3`, the seed-pooled parent macro). Otherwise the reading is "H-alt not confirmed on P3".
- point ≥ 0.80;
- the 95% CI lower bound ≥ 0.70 (parent bootstrap, 10,000 resamples, seed 20261265);
- the point estimate ≥ 0.70 in each form.

**Consequences.**
- Either way E01R ends without training, and P3 is spent.
- If H-alt is not confirmed, the result is logged as the sealed estimate of the residual. A later recipe then needs a
  freshly authored pool (P4).
- This is not a confirmatory contrast. There is no E33 request: N-CFSD's E01N outcome stays REVISE (CX D-33).

## 3. Arms and fairness

| Arm | Seeds | Role | Status |
|---|---|---|---|
| **N-CFSD** | 0, 1, 2 (existing) | control | Rows `train_N-CFSD.jsonl` sha256 `de23a923…` (1,501 rows, 376 parents). Adapters `97cc48e8…`, `8e710e46…`, `652b497d…` (CX D-30; full values in each `TRAIN_META.json`). Re-evaluated only, never retrained. |
| **X1** = R1 (B3) or R2 (B4) | 0, 1, 2 | treatment | **PENDING STEP A** |
| **X2** = R3 | 0, 1, 2 | treatment | trained only in B3/B4 |
| OFF (frozen core) | — | tier-A reference, descriptive | |
| N-VR s0 | — | optional descriptive reference; never in a test | |

**Control.**
- All three N-CFSD adapters were trained before P3 existed:
  - P3 was fixed about 14:25 on 2026-10-11 (D-34);
  - s0 finished on 2026-10-10 at 23:09;
  - s2 finished before the 08:40 opening on 2026-10-11 (D-30).
- Their P2 behaviour (step A) is what the treatments were chosen against. The control itself is fixed: there is no
  selection among control adapters.

**Recipe (every treatment adapter).** Exactly N-CFSD's:
- D-03: LoRA r16/α32/dropout 0.05 on frozen FP32 Titan-Sparse, 190 steps × 16, lr 1e-4 WSD, AdamW (0.9, 0.95),
  `max_len` 512, CPU.
- It runs through the unchanged `lora_cpu.train_sft`, with exactly the arguments of `native_e01.cmd_train`; a test
  checks this.
- 1,501 rows; Titan's frame (`native_probe.rcl_system`).
- **Matched seeds.** Training seed k ∈ {0, 1, 2} sets `attach_lora(k)` and `row_order_for(1501, 190, 16, k)`, exactly
  as for N-CFSD_sk. X_sk and N-CFSD_sk are therefore matched pairs.
- **No tuning.** No hyperparameter differs from N-CFSD's.
- **No early look.** No treatment adapter is evaluated on DEV, DEV2 or any battery before the SEALED analysis.

### Treatment recipes

**R1, complement coverage (B3 only; FROZEN spec).**
- **Pool.** The training pool `TR_v1`: 141 candidates, 2 blind authors (workflow wf_b87c2ed3-8e0), both polarities,
  all 6 cells.
  - The authors never saw any RCX template, nor the P2, P3, R-06 or R-08 wordings.
  - All 141 passed unanimous label-blind verification by 3 verifiers at list sizes 1, 2 and 3. In the same
    verification run, the protocol rejected 12/12 planted ambiguous wordings and passed 11/11 clear decoys and 23/23
    P2 re-tests (CX D-37).
- **Audit, by code, before any row is built.** A wording is dropped for:
  1. a status word, or a stem (≥ 4 shared leading letters) of a real status-like word;
  2. near-duplicating an RCX reference wording (t\*, h\*, frames, clauses): word Jaccard ≥ 0.5, or a shared 5-word
     sequence;
  3. near-duplicating any **P2_v2 or P3_v2** wording under the stricter rule: Jaccard ≥ 0.35, or a shared 4-word
     sequence;
  4. within-pool near-duplicates (Jaccard ≥ 0.6, or a shared 6-word sequence): one member is kept per cluster.
- **P3 custody in the audit.** The P3 comparison is a dedicated routine. It reads `P3_SEALED_v2.json.secret` and prints
  **counts per cell only**. Dropped texts go only to a git-ignored `*.secret` file. No P3 id, text or pairing is ever
  output.
- **Minimum.** At least 6 surviving wordings in each of the 6 cells. One blind top-up round is allowed, with the same
  training-author prompt; otherwise the §2 fallback applies.
- **Rows.** `recipes.make_r1(1501, seed=20261253, train_pool=<audited pool>, p_pool=0.5)`.
  - 20261253 is N-CFSD's data seed, so parents, states, events, targets and row order are N-CFSD's.
  - Each single-atom row gets a pool wording of the same polarity with probability 0.5: the form is drawn uniformly,
    then the wording. Compositions keep their frames.
  - **Required test.** Apart from `tid`/`input`/`parent_id`/`group`, the records equal
    `gen.make_train("CF_SD", 1501, 20261253)`. The non-pooled prompts and targets equal `train_N-CFSD.jsonl`
    (`de23a923…`) byte for byte.
  - **The row audit uses an E01R reader.** It parses the registered `tr:` wordings back to their AST and gold, with a
    unique reading required. The h\* fragments are computed from the original t\*/h\* only.
- **Covariates.** Supervised tokens equal N-CFSD's; processed tokens differ by prompt length.

**R2, complement-focused wording twins (B4 only; FROZEN spec).**
- **Generator.** `recipes.make_r2(1501, seed=20261264)` over CF_SD parents (`gen.train_parent_variants("CF_SD", …)`),
  with the TRAIN word source (50% nonce).
- **Twin pairs for single-atom rules.** Every parent whose rule is a single status atom gets k ∈ {2, 3} pairs, each
  under a distinct training wording (t0–t3):
  - **S+ parents** (as implemented): `comp_unlisted` (a fresh unlisted status → no-op) and `comp_listed` (a listed
    status → apply);
  - **S− parents (mirrored; adopted now):** `comp_unlisted` (→ apply) and `comp_listed` (→ no-op).

  So "unlisted" alone never predicts the target. Both members of a pair share state, event and surface. For a size-1
  set, a `comp_listed` twin that duplicates the parent's base or status_swap record is redrawn under a different
  wording (otherwise dropped).
- **Rows.** Whole parents only; the remainder up to 1,501 is filled with base-only parents.
- **Pre-declared confound.** R2 has fewer parents than N-CFSD (376), and a different rule-type mix. Rows by rule type
  (S+, S−, composition), parents and supervised tokens are reported per arm. An R2 effect is a recipe effect, not
  isolated from the rule-type shift.
- `make_r2` is updated to this specification (with tests) before the full freeze.

**R3, predicate-first trace target (B3 and B4).**
- **Implementation.** `experiments/RCX-E01R/trace_target.py`.
- **Inputs.** The same records as N-CFSD (`gen.make_train("CF_SD", 1501, 20261253)`): prompts byte-identical, same
  order.
- **Target.** Only the target changes, to two lines:
  - **line 1:** the rule's checks on this event and the resulting decision, e.g.
    `check status="vervil" listed=no only-listed:false -> noop`. It is derived from the rule AST, the event and (for
    BGE) the state, never from the wording;
  - **line 2:** N-CFSD's unchanged target (`target_json(after)`).

  The membership test and the decision are thus stated before the state (DESIGN §4; the E01 `cmp → after` hypothesis;
  RCWM R23).
- **Scoring (FROZEN).** For R3 outputs, the balances are taken from the **last non-empty line**, parsed by
  `native_probe.parse_answer`.
  - A one-line answer, such as the bare native JSON, is accepted.
  - Line 1's well-formedness and consistency are reported separately and never scored.
  - G3 for R3: `after is None` under this rule.
  - Control and R1/R2 outputs are parsed by `native_probe.parse_answer` on the whole output, unchanged.
- **Generation length.** `max_new` = max(160, the tokenizer-measured longest R3 gold target + 16), the same for
  **every** arm. It is recorded at build time, before the full freeze.
- **Smoke test.** Before `open-sealed`, each treatment adapter is run on 50 TRAIN-row prompts, never DEV or SEALED.
  The share well-formed and the share truncated are logged. If more than 2% are truncated, the run aborts before
  opening.
- **Prompt conflict.** The frame still ends "Return only the complete resulting balances as a JSON object". Keeping
  the prompts identical across arms is deliberate.
- **Covariates.**
  - R3 adds supervised tokens: line 1 has a mean of 57–68 bytes, and full targets are ≤ 128 bytes in the train sets
    (**PENDING BUILD** in tokens).
  - Rows stay far below `max_len` 512; the audit checks this.
  - Training is expected to take about 20–30% longer.
- **Alternative format (review).** `world.target_trace`'s JSON `cmp/calc/after` format is closer to RCWM's joint
  format. It is not adopted here, because the two-line format is implemented and tested.

**Covariates (all arms, reported in `AUDIT_E01R.json`; nothing is matched on them):**
- supervised and processed tokens per seed;
- parents;
- no-op share;
- rows by rule type, wording source and account count;
- for N-CFSD (seed 0): 58,223 supervised tokens.

> **PENDING BUILD.** Rows sha256 and covariates per arm: ___.

## 4. Battery and custody

**Code (sha256 recorded at the freeze; PENDING FREEZE):**
- Drafted in parallel (untracked when this draft was written; to be reviewed):
  - `experiments/RCX-E01R/recipes.py` (R1, R2);
  - `experiments/RCX-E01R/trace_target.py` (R3);
  - `tests/test_e01r_recipes.py` and `tests/test_e01r_trace.py`.
- Still to write (**PENDING CODE**):
  - `experiments/RCX-E01R/confirm_e01r.py`: rows, the counts-only P3 audit, train, mint-commit, open-sealed, eval;
  - `experiments/RCX-E01R/analyze_e01r.py`: `confirmatory()`, `protected()`, `decide()`, the experiment summary;
  - `tests/test_e01r_confirm.py`: fake tokenizer only; includes a truth table for `decide()`.
- Imported unchanged, re-hashed at the freeze (the CX D-26/D-34 values if unchanged):
  - `rcx/native/lora_cpu.py` (`ca469a13…`);
  - `experiments/RCX-E01N/native_e01.py`;
  - `native_probe.py` (`e8e8723f…`);
  - `rcx/rcl/{world,gen,vocab,templates,score}.py`;
  - `rcx/rcl/wording_audit.py` (`cd6a1ef5…`);
  - `rcx/stats.py` (`06a30c10…`);
  - `experiments/RCX-E01R/run_e01r.py` (`dff678c1…`).

**Pools:**
- P2_v2: `7e625a59…`, 45 wordings.
- **P3_v2**: `88dd432bf59ba82617035d3a96280be74ef9fce233cfafa03e539cb489bcf052`, 48 wordings. Per cell, from
  POOL_REFINE `kept_by_cell.P3`:
  - POS LF 7, CF 10, IM 6;
  - NEG LF 7, CF 9, IM 9.
- R1 pool: **PENDING STEP A / BUILD** (only in B3).

**Battery (SEALED, `in` + `p3`).**
- **`in` surface.** Built with `gen.use_world("arith")` and `gen.make_battery("sealed", 40, seed, strata=STRATA,
  surfaces=("in",))`.
  - STRATA = `run_e01r.STRATA`: the single-atom strata F1–F5, V1, V2, V3, V5 and V7.
  - 400 parents and 880 records, in the canonical t0 wording.
- **`p3` surface.** A copy of every `in` record, with the rule rendered by a P3 wording and everything else canonical
  (as `run_e01r`'s `p2`).
  - Each parent gets one form, rotated within its stratum (rank mod 3, in `FORMS` order).
  - For each polarity, the parent draws an index j = `Random(f"{seed}-{pid}").randrange(c)`, where c is the committed
    count of that (polarity, form) cell.
  - Only at opening does j map to the j-th P3 wording of the cell, in id-sorted order.
  - V3 and V5 twins keep their parent's form.
  - 880 records.
- **Size.** 1,760 records in total, about 33 min per non-trace evaluation (E01N SEALED: 2,240 records in 2,485 s).
- **No compositions.** P3 wordings are single-atom (one `{L}` slot), and the composition frames are fixed, already-spent
  text. V6, M1 and M2 are therefore not in the battery.

**Mint** (once, after the freeze and **before any treatment training**):
- **Seed.** seed = int(sha256(salt)[:8], 16) % 10**8, as in E01N.
- **Commitment.** The commitment binds a **P3-free skeleton**: every `in` record, plus, for each `p3` record, its
  parent_id, variant, polarity, form and j, without text.
  battery_commit = sha256(salt + ":" + sha256(skeleton jsonl bytes)).
- **`SEALED_COMMIT.json` holds:**
  - sha256(salt);
  - battery_commit;
  - n_parents 40, the strata and the surfaces;
  - the P3 file sha256 and its per-cell counts;
  - `arms_required` = N-CFSD_s0–s2 plus X1_s0–s2 and R3_s0–s2;
  - the shas of the generator modules, `confirm_e01r.py`, `analyze_e01r.py` and `TRAIN_MANIFEST_E01R.json`;
  - `minted_at`.
- **The mint reads no P3 file.** It uses only the committed counts.
- **Salt.** It is stored in `runs/RCX-E01R/cells/SEALED_SALT.secret` (git-ignored, write-once).
- **Records.** `SEALED_COMMIT.json` is committed with the frozen prereg. sha256(salt) and the `SEALED_COMMIT.json`
  sha256 are logged in DECISIONS.

**Open** (`open-sealed --expect-commit-sha <DECISIONS value>`; runs once; `SEALED_OPEN.json` is write-once). It refuses
unless every `arms_required` adapter is a completed, untampered run:
- for the control: the adapter sha equals the D-30 value and E01N's `TRAIN_META.json`;
- for the treatments:
  - `TRAIN_META.json` is present;
  - the adapter sha matches it;
  - the rows sha equals `TRAIN_MANIFEST_E01R.json`;
  - the run has 190 steps and no resume checkpoint.

It then:
1. regenerates the skeleton and checks it against the commitment;
2. reads P3_v2. This is the first read of P3 by any code other than the R1 audit. It checks the sha `88dd432b…` and the
   per-cell counts, then renders `p3`;
3. records:
   - the salt, the seed and the rendered battery's sha256;
   - the adapter shas;
   - the battery's training overlap with every training-row file (N-CFSD and the treatments): semantic hashes,
     skeleton hashes, and status words found in prompts;
4. drops parents with any hash overlap from every arm before analysis (the E01N rule) and reports the count.

`SEALED_OPEN.json` contains the salt. It is committed, with its sha256 logged, before the first SEALED evaluation.

**P3 text stays hidden until the analysis.**
- No command prints P3 text.
- The lead does not view P3 text or the rendered `p3` records until the analysis has run.
- Evaluation and analysis rebuild the battery in memory.

**Evaluate** (`eval --battery sealed`).
- **It refuses:**
  - before opening;
  - for an adapter other than the one recorded at opening;
  - for any cell of a committed arm that was not fixed at opening.
- **Decoding.** Greedy, with `max_new` = `TRACE_MAX_NEW` (160; R3 traces need more than E01N's 64). Every arm uses the
  same batching settings.
- **Outputs.** Raw outputs plus scores, written create-only.

**Analyse.** `analyze_e01r.py` runs once, and its outputs are create-only. It refuses unless:
- every eval names the rendered battery's sha256;
- every adapter matches the open-sealed record;
- every stored score equals a re-score of the raw output.

**Training-row audit (before the freeze; training refuses unaudited rows).** These are E01N's `audit_native` hard
checks, applied to every treatment file:
- the frame is byte-exact;
- the gold is re-derived from the prompt text:
  - for R3, the full trace target;
  - for R1, the R1 wordings are registered so that `read_back` parses them;
- only TRAIN words and the posted/void anchors appear;
- there is no h0/h1 fragment, held-out combination, size-3 set or M1/M2 magnitude;
- there is no RCWM held-out text;
- there is zero semantic and zero skeleton overlap with the E01N DEV battery (seed 20261240) and with the DEV2 battery
  (seed 20261261, both surfaces).

**Shared with earlier batteries by design.**
- **The sealed word pools.** They are also used by the opened E01N SEALED battery (seed 88527345), whose results were
  seen only in aggregate. No sealed word occurs in any training row.
- The anchors posted and void.
- The F4 near-miss probes and the F5 posted-like strings.
- The canonical t0 wording on `in`.

**Limit.** The commitment proves that the battery was fixed at the mint. It cannot stop the salt or P3 from being read
early; doing so is a protocol violation and voids the confirmation (§11).

## 5. Endpoints

All endpoints are parent macros (variants averaged within a parent, then over seeds), and every contrast is X − N-CFSD.
Unparseable outputs score 0 for both decision and exact.

**Confirmatory family (Holm, 4 tests; 2 per treatment arm):**

| ID | Endpoint | Records | Parents |
|---|---|---|---|
| **C1_X** | **POS no-op decision accuracy on `p3`** (the R-08 residual) | Every required-no-op record under an S+ rule on `p3`: the F1–F5 probes, V1 `unlisted`, V5 `pos_x_status_posted`, V7 `unlisted` when S+. Value = `decision_ok`; for no-op items this equals `exact`. | ≈ 300 (≤ 1 item each; ≈ 100 per form) |
| **C2_X** | composite exact on `p3` | E01N's composite {F2, F3, F5, V1, V2, V3, V5}, `exact` | 280 |

**Protected strata** (ADVANCE guards, outside the Holm family). Each is X − N-CFSD by the same bootstrap and seed, and
is judged by interval:

| ID | Stratum | Within margin if | Parents |
|---|---|---|---|
| G1 | `in` composite exact | 95% CI lower bound > −0.03 | 280 |
| G2 | F0 posted twins: exact on the `twin_posted` variants of F1–F5, `in` | lower bound > −0.05 | 200 |
| G3 | format-error rate, all SEALED records | upper bound ≤ +0.02 | 400 |
| G4 | **POS apply** decision accuracy on `p3`: F `twin_posted`, V1 `listed`, V3 `pos`, V5 `pos_x_status_x`, V7 S+ `listed` | lower bound > −0.05 | ≈ 340 |
| G5 | **NEG apply** decision accuracy on `p3`: V2 `unlisted`, V5 `neg_x_status_posted`, V7 S− `unlisted` | lower bound > −0.05 | ≈ 100 |
| G6 | **NEG no-op** decision accuracy on `p3`: V2 `listed`, V3 `neg`, V5 `neg_posted_status_posted`, V7 S− `listed` | lower bound > −0.05 | ≈ 140 |

**Why G4–G6.** C1 can be won by degenerate policies, and each is caught by a guard:

| Degenerate policy | Effect on C1 | Caught by |
|---|---|---|
| "no-op whenever the status is unlisted" (polarity-blind) | wins C1 | G5 (NEG apply collapses) |
| "no-op everywhere" | wins C1 | G4 and G5 |
| "apply every listed status" | — | G6 |

C2 alone guards weakly, because the C1 items carry about a third of its parent weight.

**Secondary (descriptive; no test):**
- **Per form and per template.**
  - C1 and G4–G6 levels and contrasts per form (LF, CF, IM), with Wilson intervals.
  - Per P3 template (ids and counts only).
  - A **template-cluster sensitivity** for C1: resample P3 templates within each cell, then parents; 10,000 resamples,
    seed 20261268. The report must state "C1 gain shown across P3 wordings: yes/no", according to whether this
    interval's lower bound is > 0.
- **Arm contrasts.** X1 − R3 head to head, and each arm against OFF (and against N-VR s0 if run).
- **Seeds.** Per-seed values and spread, and the matched per-seed contrasts with a t interval (df = 2).
- **Behaviour.**
  - Every stratum, including F1, F4 and V7, on both surfaces;
  - error directions;
  - V3 polarity sensitivity;
  - the PREREG-3-style verdict;
  - the `in` analogue of C1.
- **R3 trace diagnostics** (`trace_target.predicate_consistency`):
  - well-formed rate;
  - the stated decision's accuracy, including C1 recomputed from it;
  - agreement between the balances and the stated decision;
  - line-1 exactness and coherence;
  - `listed=` accuracy, which tests membership.

  On S+ unlisted items, "listed=no only-listed:true" or "-> apply" is the complement-inference error itself.
- **Covariates** (§3).

## 6. Statistics

As in PREREG_E01N §4:
- **Estimator.** `rcx.stats.seed_parent_bootstrap_test`, a seed × parent paired cluster bootstrap:
  - per parent, average the variants within each seed, then average the three seeds;
  - resample parents, not seeds;
  - **10,000 resamples, bootstrap seed 20261265**, for every confirmatory and protected contrast;
  - the CI is the 95% percentile interval.
- **Pairing.** Over the parents present in both arms, after the train-overlap drop.
- **p-value.** Two-sided, from the same replicates: 2 × min(share of replicate differences ≤ 0, share ≥ 0).
  - Replicates within 1e-12 of 0 count on both sides (`P_TIES = "both"`).
  - Floored at 1/10,000 and capped at 1.
- **Holm.** Applied at α = 0.05 across all primary tests of both treatment arms: {C1_X1, C2_X1, C1_R3, C2_R3}.
  - The family is fixed by `arms_required` at the mint. If an arm is dropped before the mint (§2), the family is the
    remaining arm's 2 tests.
  - After the mint, a test that cannot be computed confirmatorily (a missing cell) enters Holm with p = 1.
- **Per-seed contrast.** The parent macro of X_sk minus that of N-CFSD_sk, on the endpoint's records, for k = 0, 1, 2.
- **Conditioning.** Inference is conditional on:
  - the 9 adapters;
  - the sealed word pools;
  - the 48 P3 wordings, of which 23 are positive.

  Seed variation enters through the per-seed condition (§7), and wording variation through the template-cluster
  sensitivity (§5).
- **Tolerance.** Thresholds are compared with a 1e-9 tolerance.

## 7. Decision rule

**Per treatment arm X.** "CI" is the 95% percentile interval of §6, not multiplicity-adjusted. The outcomes are tested
in the order below, and the first that holds is the decision.

1. **ADVANCE.** All of:
   - **C1_X:**
     - Holm-adjusted p < 0.05;
     - point estimate ≥ **+0.25**;
     - CI lower bound > 0;
     - each matched per-seed contrast ≥ **+0.10**.
   - **C2_X:** Holm-adjusted p < 0.05 and point estimate > 0.
   - **Guards:** G1–G6 all within margin.
   - **Cells:** all six cells (X s0–s2 and N-CFSD s0–s2) analysed.
2. **STOP.** C1_X CI upper bound < +0.10, **and** C2_X CI upper bound < +0.05.
3. **REVISE.** C1_X or C2_X has Holm-adjusted p < 0.05 with point estimate > 0. This includes a C1 at or above +0.25
   whose other ADVANCE conditions fail; the report names each failed condition.
4. **INCONCLUSIVE.** Anything else. It is always the outcome when any of the arm's or the control's six cells is
   missing; C1 and C2 are then reported descriptively.

**Experiment-level outcome.** Tested in order:
1. ADVANCE if any arm reaches ADVANCE (each such arm is named);
2. STOP if every arm reaches STOP;
3. REVISE if any arm reaches REVISE;
4. otherwise INCONCLUSIVE.

If both arms reach ADVANCE, the lead candidate is the arm with the larger C1 point estimate (a tie goes to the arm with
fewer supervised tokens).

**Changes from E01N's rule:**
- ADVANCE rests on the targeted C1 only. In E01N it could rest on either endpoint.
- C2 must be significantly positive, which is stronger than E01N's "other endpoint lower bound > −0.05".
- The per-seed bar is ≥ +0.10 instead of > 0. C1 has large headroom, and a gain carried by one seed would not be a
  dependable recipe.
- There are six protected strata.

**Threshold justification (from R-08):**
- **C1 +0.25.**
  - On `ho_tpl`, N-CFSD's polarity gap was 0.97 − 0.14 = 0.83 (S− vs S+ no-op), and +0.25 is about 30% of it.
  - If P3 resembles `ho_tpl`, +0.25 takes the control's 0.14 to about 0.39. Anything smaller leaves "apply" as the
    dominant answer on unlisted statuses.
  - C1's items carry 90/280 = 0.32 of C2's parent weight:
    - F2, F3 and F5 probes: half of each parent;
    - V1 `unlisted`: half;
    - V5: a quarter.

    So +0.25 on C1 corresponds to about +0.08 on C2, commensurate with E01N's +0.10 composite effect.
- **Per seed +0.10.** This is 40% of the practical effect. A seed below it has not materially learned the complement
  case.
- **STOP at +0.10 on C1.** This equals E01N's practical effect. An upper bound below it means less than an eighth of
  the gap was closed. **STOP at +0.05 on C2** is E01N's STOP bound.
- **G1 −0.03.** N-CFSD's SEALED `in` gain over N-VR was +0.018 [+0.006, +0.030]. A loss beyond 0.03 would give back
  more than that gain's upper bound, i.e. drop to N-VR's level on training wordings.
- **G2–G6 (−0.05 / +0.02).** These are E01N's margins. R-08's over-restriction on held-out OR wordings (apply 84/87 →
  66/87, −0.21) is the size of effect G4 and G5 exist to catch.

**Pre-declared notes (power, stated honestly):**
- **Baseline.**
  - The control's C1 level on `p3` is unknown. The P2 proxy is **PENDING STEP A**: p̄ = ___, so the headroom is
    1 − p̄ = ___.
  - Under B3/B4, p̄ < 0.65, so the headroom is > 0.35. R-08's `ho_tpl` value was 0.14.
- **Sampling error.**
  - C1 has about 300 parents. For rates between 0.2 and 0.6, with items repeated across seeds, the parent-bootstrap
    SE of a C1 contrast is about 0.025–0.03 (CI half-width about 0.05–0.06).
  - Seed-to-seed variation of a 3-seed mean adds an unknown amount. E01N's C2 per-seed SD was about 0.03 at a +0.09
    effect, and a targeted endpoint may vary more.
- **P(point ≥ +0.25), assuming a total SD of about 0.05:**

  | True C1 effect | +0.35 | +0.30 | +0.25 | +0.20 |
  |---|---|---|---|---|
  | P(point ≥ +0.25) | ≈ 0.98 | ≈ 0.84 | ≈ 0.50 | ≈ 0.16 |

  ADVANCE is therefore likely only if the true effect is at least about +0.30 *and* no guard fails. An effect near
  +0.25 is a coin flip, and REVISE is then the likely outcome. Any effect of at least about +0.08 gives a CI lower bound
  > 0, so REVISE at least.
- **C2.** A +0.30 C1 effect implies about +0.10 on C2, with SE ≈ 0.011 (E01N's C2 CI half-width was 0.022). C2 binds
  only if the recipe also costs negative-rule, V3 or arithmetic accuracy.
- **Holm.** At the p floor of 1e-4, the Holm-adjusted p is ≤ 4e-4, so Holm binds only for weak effects.
- **Main risks.**
  - R3: format (G3) and arithmetic inside `after` (C2).
  - R1 and R2: over-restriction (G4, G5).
- **Wordings.** There are 23 positive P3 wordings. A C1 gain carried by a few of them can still pass. The
  template-cluster sensitivity line (§5) makes that visible but does not change the decision.

## 8. Consequences

**ADVANCE (arm X).**
- **Records.** A new R-xx in EXPERIMENT_RESULTS (native-C, confirmatory) and a CX D-xx in DECISIONS.
- **E33 request.** A `docs/REQUESTS.md` entry (CX Q-0x) to **RCWM-T1** for **E33 retention**, following RCWM R27:
  - RCWM 5c, the D-36 gate, on X s0–s2;
  - a **fresh sealed draw minted by RCWM at run time**;
  - about 0.1–0.2 GPU-h per adapter;
  - only adapter directories and shas are exchanged;
  - N-CFSD s0 is added as a reference if RCWM's protocol needs a trained comparator (RCWM's call);
  - RCX never writes turn or lease files; RCWM's DECISIONS is authoritative.

  The request is also delivered as a session message.
- **CX Q-03 cross-scoring** is extended to X:
  - RCWM's PREREG-3-style map runs on X, and RCX's battery on T2b-ledger;
  - each side runs its own instrument;
  - confirmation banks are not shared.
- **Claims.** No retention, dense-model or general-ability claim is made until E33 reports. The adapters stay local
  and unmerged.
- **If X = R3.** Before any mechanism claim ("the predicate step causes it"), a token-matched filler arm (`target_fill`,
  the D-34/4b format) is needed. It is proposed as a follow-up and is not part of E01R.
- **Next.** RCM moves to E02 (computed predicates), with X as the base-recipe candidate (the E02 prereg decides).

**REVISE.**
- No E33 request and no Q-03 extension. The report names the failed condition.
- Follow-up options (any of them needs a new prereg):
  - **C1 ≥ +0.25 but a guard failed:** revise the recipe against that guard. For example, a G5 failure calls for
    stronger polarity-balanced twins.
  - **C1 significant but below +0.25:** combine X⊕R3, or raise the row share.
- Every follow-up needs a newly authored blind pool (P4) and a new sealed battery, because P3 is spent after this
  analysis.

**STOP (every arm).**
- Coverage, twins and trace do not fix the residual at this budget. This is logged as a negative result, with the
  per-form profile.
- RCM moves to E02.
- The complement limitation stays attached to every citation of N-CFSD (as in R-08's limits).
- No further E01R recipe is tried without a new hypothesis.

**INCONCLUSIVE.**
- A descriptive report only.
- Missing cells are never replaced, and nothing is rerun under E01R. P3 is spent.

**B0, B1 and B2 (§2).** No training. The outcome is documented in DECISIONS, and RCM moves to E02 (under B1, after the
coverage check).

## 9. Seeds

**Search.** The working tree was grepped on 2026-10-11 for `202612`, excluding `*.secret` files and `pools/`; the four
readable pool JSON files were checked separately. No committed file uses any of 20261263–20261269 or
20261281–20261288. The only hits are in the untracked drafts `recipes.py` and `tests/test_e01r_recipes.py` (see
below). Neighbouring seeds already in use:
- 20261261: the DEV2 battery;
- 20261262: the step A descriptive bootstrap;
- 20261270–20261280: E07/BW;
- 20261289 and up: E07/BW.

The git history was not searched: a pickaxe search was stopped to keep load off the lane. The untracked
`recipes.py`, drafted in parallel, *suggests* 20261263 (R1) and 20261264 (R2) as defaults; the table below fixes the
roles.

| Seed | Role |
|---|---|
| 20261263 | **Not used.** It is `recipes.py`'s suggested R1 default and is superseded: R1 reuses N-CFSD's data seed 20261253 (§3). Left unassigned so that no accidental default run can collide with a fixed role. |
| 20261264 | R2 data generator (`recipes.make_r2`): parents, complement-pair statuses and wordings (B4 only) |
| **20261265** | Bootstrap seed for every confirmatory and protected contrast (§6) and for the B1 coverage-check CI |
| 20261266 | R1 label-blind verification items: list sizes, probe statuses, order (B3 only) |
| 20261267 | Reserved: an R1 top-up verification round, only if needed (B3 only) |
| 20261268 | Descriptive template-cluster bootstrap (§5) |
| 20261269 | Reserved, unassigned |
| 20261281–20261288 | Reserved for E01R follow-ups; not used by this prereg |

**Seeds outside the block:**
- **SEALED battery seed:** derived from the salt; unknown until opening.
- **Training seeds 0, 1, 2:** set the LoRA initialization and row order, identical to N-CFSD's.
- **R1 and R3 base rows:** N-CFSD's (E01N data seed 20261253), unchanged.
- **R3:** draws no randomness.

## 10. Execution and safety

**Order:**
1. Step A, which is already queued: four DEV2 evals, then `analyze`. Then the branch is read mechanically by §2 and
   logged in DECISIONS.
2. Under B3/B4, build the treatment rows. R1 authoring and verification run as subagents with no CPU-lane load. Then
   run the audit and write `TRAIN_MANIFEST_E01R.json`. Unit tests run only when the lane check prints LANE_FREE.
3. Fill every PENDING field, review, and freeze: the sha256 of this file and the code shas go into DECISIONS.
4. Run `mint-commit`. Commit `SEALED_COMMIT.json` and log its hashes.
5. Train the six treatment adapters in sequence (s0, s1, s2 of each arm).
6. Run `open-sealed`. Commit `SEALED_OPEN.json` and log its sha256.
7. Evaluate on SEALED: N-CFSD s0–s2, X1 s0–s2, R3 s0–s2 and OFF; optionally N-VR s0.
8. Analyse once with `analyze_e01r.py`.

**Compute (lane hours; PENDING STEP A):**
- Training: about 13–17 h. A non-trace adapter takes 1.7–2.6 h (CX D-29, R-06); an R3 adapter about 2.1–3.2 h.
- Evaluation: about 6–7 h (about 33 min per non-trace cell, about 60 min per R3 cell).
- Total: about 20–24 h.
- B1 needs only about 1.7 h.

**Constraints:**
- **Single CPU lane.** One model job at a time. The chain scripts log `START <tag>` / `END <tag>` lines that the lane
  check reads.
- **Threads.** `RCX_THREADS` ≤ 4 for lane jobs and 1 for anything else. `CUDA_VISIBLE_DEVICES` is empty; the GPU is
  never used.
- **Subagents.** No CPU-heavy work while a lane job runs; only checks that finish in under 5 s.
- **RAM.** Wait for ≥ 12.5 GB of free RAM before each model load; `need_ram` refuses below 12 GB.
- **Guard.** `rcx.guard.ThermalGuard` runs between micro-batches.
- **Control adapters** are read-only, from `runs/RCX-E01N/cells`. E01R writes only under `runs/RCX-E01R/`.

**No retries under the same id:**
- A completed adapter is never retrained, and an eval file is never overwritten.
- The mint and open records are write-once.
- The analysis runs once, and its outputs are create-only.
- A crashed training run may continue only from its own checkpoint, under the same run fingerprint (the same
  computation).
- A cell that cannot be completed is reported as missing and is never replaced by another seed or id.

**Failures:**
- A training cell that cannot be completed leaves open-sealed refusing, and the affected arm is INCONCLUSIVE. Opening
  the battery anyway for descriptive use is a logged deviation.
- A SEALED evaluation that cannot be completed is analysed with `--allow-missing`, and its arm is INCONCLUSIVE.

## 11. Freeze and deviations

At the freeze, the following are recorded in a new CX D-xx:
- this file's sha256 and timestamp;
- the code and row shas;
- sha256(salt) and the `SEALED_COMMIT.json` sha256.

Every deviation is logged with "outcomes known: yes/no". Nothing is rerun under this ID.

**Any of the following voids the confirmation:**
- re-creating or deleting `SEALED_COMMIT.json`, `SEALED_SALT.secret` or `SEALED_OPEN.json`;
- retraining a completed adapter at any time, or training any adapter after opening;
- reading the salt, or P3 text, before opening, by any path other than the code paths named in §3–§4;
- evaluating a treatment adapter on any battery before the SEALED analysis;
- changing §2 after any step A output exists.

## Open items for review (draft only; to be resolved before the freeze)

1. **The §2 cut-offs.** These are judgement calls:
   - 0.85 (anchor);
   - 0.80 (H-alt, DESIGN);
   - 0.65 (near ceiling);
   - 0.15 (spread).

   B2's "no training" is the most debatable.
2. **R1 vs R2 per branch.** The mapping (form-specific → R1, uniform → R2) is a reasoned guess. Should R1's authoring
   use *different* registers from P3's, for a stronger transfer test, at the cost of a likelier STOP?
3. **R2's mirrored S− twins and R1's two-polarity pool.** Both depart from DESIGN §4's positive-only wording.
4. **The C1 threshold.** +0.25 is fixed, not scaled to the step A baseline.
5. **Six guards.** They lower P(ADVANCE) under a true effect. Margins: −0.03 for G1, −0.05 for G2 and G4–G6, +0.02 for
   G3.
6. **R3 and its filler control.** R3 without a filler control limits any mechanism claim (§1).
7. **The R3 target format.** This draft adopts `trace_target.py`'s two-line `check … -> dec` + native JSON format.
   `world.target_trace`'s JSON `cmp/calc/after` format is the alternative.
8. **Alignment with the parallel code.** `recipes.make_r1` must be called with seed 20261253, not its default 20261263.
   The P3 near-duplicate check must be a separate counts-only routine and must never be passed through `audit_pool`'s
   `extra_refs`, whose reasons name the matching reference key.
9. **New code and tests** must exist and be reviewed before the freeze: `confirm_e01r.py`, `analyze_e01r.py`, and
   tests that use a fake tokenizer only. Among them:
   - a test that the P3-free skeleton commitment and the rendered battery are deterministic;
   - a test that `make_r1(seed=20261253)` reproduces N-CFSD's records apart from their wordings.
