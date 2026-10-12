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

> **STEP A (filled mechanically from `BRANCH_STEP_A.json`, CX D-39):** branch = **B4**; p_LF / p_CF / p_IM = 0.509 / 0.346 / 0.639; p̄ = 0.496;
> E − I = −0.213 [−0.504, +0.076]; S = 0.293; q_LF / q_CF / q_IM = 0.917 / 0.747 / 0.667; a_in = 0.996; treatment arms = **R2, R3**.

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

**Reading of the parts kept byte-identical.** The header and §1–§3 predate step A (CX D-37) and are not edited. Where
they leave something open, CX D-38, CX D-39 and §4–§11 govern:
- **Branch B4** (CX D-39; §2's record). The treatment arms are **E-R2** (recipe R2) and **E-R3** (recipe R3), seeds 0,
  1 and 2 each, against the control N-CFSD s0–s2. In §3's table, X1 = E-R2 and X2 = E-R3.
- **Not applicable under B4:** R1, the TR_v1 pool, `audit_tr_pool.py`, `make_r1`, §2's R1 fallback and §2.1 (B1). No
  R1 code runs, so **no code reads P3 before `open-sealed`**.
- **Smoke.** §3's frozen rule ("the run aborts before opening") is applied without any exemption route (§10).
- **N-VR s0**, "optional" in §3, is **not** evaluated on the E01R SEALED battery.
- **Stale markers.** The header's PENDING markers, its "What was known" list and its refinement 4 describe the state at
  CX D-37. Refinement 4 (mirrored S− pairs) is part of the frozen R2 spec. §3's "PENDING STEP A" for X1 is resolved
  above. §3's "`make_r2` is updated …" is done in code and checked by `tests/test_e01r_recipes.py`. §3's "PENDING
  BUILD" values (rows shas, covariates, R3 tokens) are recorded in `AUDIT_E01R.json`, `TRAIN_MANIFEST_E01R.json` and
  the freeze entry (§11), not in §3.
- **"Recorded in DECISIONS … together with the SEALED commitment"** (header). This is read as two DECISIONS entries,
  linked by the code sha: the freeze entry first, the mint entry second (§11).

**Code** (sha256 recorded at the full freeze; PENDING FREEZE):
- **On the B4 path:**
  - `experiments/RCX-E01R/recipes.py`: `make_r2` (with the mirrored S− pairs), `recipe_stats` and `native_rows`;
  - `experiments/RCX-E01R/trace_target.py`: R3 targets, the last-line parser and the trace diagnostics;
  - `experiments/RCX-E01R/audit_e01r.py`: the training-row audit, with the E01R reader;
  - `experiments/RCX-E01R/native_e01r.py`: the commands `build`, `train`, `smoke`, `mint-commit`, `open-sealed` and
    `eval`;
  - `experiments/RCX-E01R/analyze_e01r.py`: §5–§7, run once (§10);
  - `experiments/RCX-E01R/decide_branch.py` (frozen, CX D-37, `6835b040…`; it has run once, CX D-39);
  - tests (a fake tokenizer and a fake model only; no weights or tokenizer files are loaded):
    `tests/test_e01r_recipes.py`, `test_e01r_trace.py`, `test_e01r_audit.py`, `test_native_e01r.py`,
    `test_e01r_branch.py` and `test_analyze_e01r.py`.
- **Imported unchanged, re-hashed at the freeze** (the CX D-26/D-34/D-36 values if unchanged):
  - `rcx/native/lora_cpu.py` (`ca469a13…`);
  - `experiments/RCX-E01N/native_e01.py`;
  - `experiments/RCX-E01N/native_probe.py` (`e8e8723f…`);
  - `rcx/rcl/{world,gen,vocab,templates,score}.py` and `rcx/rcl/wording_audit.py` (`cd6a1ef5…`);
  - `rcx/stats.py` (`06a30c10…`);
  - `rcx/lane.py` (CX D-36; re-hashed for reference, but not bound: below);
  - `experiments/RCX-E01R/run_e01r.py` (`dff678c1…`).
- **Not on the B4 path:** `audit_tr_pool.py` and the R1 functions of `recipes.py` (`make_r1`, `audit_pool`,
  `select_pool`). They stay in the repository and no E01R step runs them. `audit_tr_pool.py` is bound nevertheless.

**Bound code.** `SEALED_COMMIT.json` records the sha256 of every file in this list, and the commitment binds it:
- `native_e01r.py`, `analyze_e01r.py`, `audit_e01r.py`, `audit_tr_pool.py`, `decide_branch.py`, `recipes.py`,
  `trace_target.py`, `run_e01r.py`;
- `rcx/stats.py`, `rcx/rcl/wording_audit.py`, `rcx/rcl/{gen,vocab,templates,world,score}.py`;
- `native_e01.py`, `native_probe.py` and `rcx/native/lora_cpu.py`.

**`rcx/lane.py` is not bound.** It is shared CPU-lane infrastructure (CX D-36). It decides only when a job may start,
and has no effect on rows, adapters, outputs or scores. A peer's fix to the lock must not block E01R.

**Code sha.** The code sha is the sha256 of the canonical JSON of the bound-code map {file name: sha256}: `json.dumps`
with sorted keys, separators `,` and `:` and ASCII escapes, encoded as UTF-8. `native_e01r.py code-sha` prints it
(pure; no model, tokenizer or P3). The freeze entry logs it. `mint-commit --expect-code-sha256 <value>` refuses unless
the current bound files give exactly that value (§11).

**Bound-code drift.** Every command that runs under the commitment refuses if any bound file's current sha256 differs
from `SEALED_COMMIT.json`:
- `train` and `smoke`, before the model is loaded;
- `open-sealed`, before P3 is read;
- `eval`, before decoding, and before post-processing a stored raw-generations file;
- `analyze_e01r.py`, before anything else.

The analysis also refuses if a record written under the commitment names a sha for a bound file that differs from the
commitment. These records are the `TRAIN_META.json` of each E-R2 and E-R3 adapter, every `SMOKE_<tag>.json` and every
`eval_sealed_<tag>.json`; each of them records the bound-code shas.
- **The `TRAIN_META.json` check covers E-R2 and E-R3 only.** The control's `TRAIN_META.json` files were written by
  E01N before the commitment existed. They record E01N-time shas, so they cannot match it.
- **The control's custody is the CX D-30 adapter-sha chain:** the D-30 value = the adapter file = E01N's
  `TRAIN_META.json` and `SEALED_OPEN.json` = the E01R opening record = the eval record.

**Pools:**
- **P2_v2:** `7e625a59…`, 45 wordings. Read by the lead; used only by the DEV2 battery (step A, done).
- **P3_v2:** `88dd432bf59ba82617035d3a96280be74ef9fce233cfafa03e539cb489bcf052`, 48 wordings.
  - Per cell, from POOL_REFINE `kept_by_cell.P3`: POS LF 7, CF 10, IM 6; NEG LF 7, CF 9, IM 9.
  - So far it has never been read, either by the lead or by any code. Under B4 the first read is `open-sealed` (below).
- **TR_v1:** not used (B4).

**Battery (SEALED, `in` + `p3`).**
- **`in` surface.** Built with `gen.use_world("arith")` and `gen.make_battery("sealed", 40, seed, strata=STRATA,
  surfaces=("in",))`.
  - STRATA = `run_e01r.STRATA`: the single-atom strata F1–F5, V1, V2, V3, V5 and V7.
  - 400 parents and 880 records, in the canonical t0 wording.
- **`p3` surface.** A copy of every `in` record, with the rule rendered by a P3 wording and everything else canonical
  (as `run_e01r`'s `p2`). 880 records.
- **P3 draw (exact sequence; pinned by `tests/test_native_e01r.py`, `test_pinned_draw_sequence`).** For each parent,
  in `gen.make_battery`'s order:
  1. form = FORMS[rank mod 3], with FORMS = (LF, CF, IM) and rank = the parent's index within its stratum. This is a
     rotation, not a random draw.
  2. rng = `random.Random(f"{seed}-{parent_id}")`: one per parent, used for nothing else.
  3. j_POS = `rng.randrange(n[POS, form])`, then j_NEG = `rng.randrange(n[NEG, form])`. Both are always drawn, in this
     order, whatever the parent's polarity. n is the committed per-cell count.
  4. At opening, a `p3` record of polarity P (S+ → POS, S− → NEG) gets the j_P-th wording of cell (P, form), with the
     cell's P3 ids sorted. V3 and V5 twins keep their parent's form and use j_POS or j_NEG by their own polarity.
- **Size.** 1,760 records in total. At `max_new` 160, a cell takes about 26–47 min (non-trace arms) or 34–85 min (R3)
  (§10).
- **No compositions.** P3 wordings are single-atom (one `{L}` slot), and the composition frames are fixed, already-spent
  text. V6, M1 and M2 are therefore not in the battery.

**Mint** (`native_e01r.py mint-commit`; once, after the full freeze and **before any treatment training**):
- **Seed.** seed = int(sha256(salt)[:8], 16) % 10**8, as in E01N (`native_e01.sealed_seed`).
- **Commitment.** The commitment binds a **P3-free skeleton**: every `in` record, plus a copy of each `in` record as
  its `p3` record, with `tid`, `input` and `p3_template` set to None and `p3_pol`, `p3_form` and `p3_j` added. There is
  no P3 text in it.
  battery_commit = sha256(salt + ":" + sha256(skeleton jsonl bytes)).
- **Arms are bound to the branch.** `--arms-required` must be exactly:
  - N-CFSD_s0, N-CFSD_s1 and N-CFSD_s2;
  - plus s0, s1 and s2 of every treatment arm of `BRANCH_STEP_A.json` (B4: E-R2 and E-R3).

  Those treatment arms must also equal the arms of `TRAIN_MANIFEST_E01R.json`. Any other set is refused.
- **Dropping an arm.** This is possible only at the mint, only for the frozen §2 reason, and only through
  `--drop-arm <arm> --reason "<text>"`. The §2 reason is that the arm's rows fail the training-row audit for a reason
  that cannot be fixed without changing the recipe.
  - The mint refuses a drop unless `AUDIT_E01R.json` records the arm as failing on its manifest rows: `arm_ok` false
    for that arm, with a failure entry naming it. An arm that passes the audit cannot be dropped, whatever the reason
    given.
  - The drop and its reason are logged in DECISIONS (outcomes known: step A only) before the mint, and they are
    recorded in `SEALED_COMMIT.json`.
  - At least one treatment arm must remain. The Holm family then has 2 tests (§6).
  - After the mint, no arm is added, dropped, replaced or re-specified.
- **`SEALED_COMMIT.json` holds:**
  - sha256(salt) and battery_commit;
  - n_parents 40, the strata, surfaces and forms, and the seed derivation and the P3 draw rule as text;
  - the P3 file name, sha256, n and per-cell counts, taken from `POOL_REFINE.json`;
  - `arms_required`, `treatment_arms` and any dropped arm with its reason;
  - the `BRANCH_STEP_A.json` sha256 and branch;
  - `max_new`, from `TRAIN_MANIFEST_E01R.json`;
  - the sha256 of `TRAIN_MANIFEST_E01R.json` and `AUDIT_E01R.json`;
  - the bound-code shas (above);
  - `minted_at`.
- **The mint reads no P3 file.** It uses only the committed counts (test `test_mint_reads_no_p3`).
- **Refusals.** The mint refuses:
  - if a commitment, salt or open record already exists;
  - if `arms_required` is not the branch-bound set above (after any logged drop);
  - if an arm's rows differ from the manifest or lack a passing `AUDIT_E01R.json`;
  - unless `AUDIT_E01R.json` records the per-seed token counts (`tokens.per_seed`: supervised and processed tokens for
    training seeds 0, 1 and 2) of every kept arm. §7 uses them for the lead-candidate tie-break;
  - unless `--expect-code-sha256` equals the code sha of the current bound files (above), i.e. the value logged in the
    freeze entry;
  - if a treatment adapter directory already exists;
  - if `analyze_e01r.py` is missing.
- **Salt.** It is stored in `runs/RCX-E01R/cells/SEALED_SALT.secret` (git-ignored, write-once).
- **Records.** The mint follows the freeze entry (§11). `SEALED_COMMIT.json` is committed, and sha256(salt), the
  `SEALED_COMMIT.json` sha256 and the code sha are logged in a second DECISIONS entry, the mint entry.

**Smoke test** (specified in §3, frozen; procedure in §10). Each treatment adapter is tested after its training and
before opening, on TRAIN rows only. There is no exemption: one failing adapter means the battery is never opened.

**Open** (`open-sealed --expect-commit-sha <DECISIONS value>`; runs once; `SEALED_OPEN.json` is write-once). Every
check that can refuse runs before P3 is read, and reading P3 is the last step. There is no partial opening and no
opening for descriptive use.
1. **Refusal checks.** None of them reads P3. `open-sealed` refuses unless all of these hold:
   - **commitment:** `SEALED_COMMIT.json` exists, its sha256 equals `--expect-commit-sha`, and no `SEALED_OPEN.json`
     exists;
   - **manifest and audit:** `TRAIN_MANIFEST_E01R.json` and `AUDIT_E01R.json` are unchanged since the mint, and the
     manifest's `max_new` equals the committed one;
   - **row files:** every training-row file (N-CFSD, E-R2 and E-R3) is present and matches its D-30, manifest and
     commitment sha;
   - **adapters:** **every** `arms_required` adapter is a completed, untampered run:
     - for the control: the adapter sha equals the D-30 value, E01N's `TRAIN_META.json` and E01N's `SEALED_OPEN.json`
       record;
     - for the treatments: `TRAIN_META.json` is present; the adapter sha matches it; the rows sha equals
       `TRAIN_MANIFEST_E01R.json`; the run has 190 steps and no resume checkpoint is left;
   - **smoke:** a passing `SMOKE_<tag>.json` exists for each treatment adapter's sha, the committed `max_new` and 50
     rows (§10);
   - **salt:** `SEALED_SALT.secret` exists and its sha256 equals the commitment's;
   - **skeleton:** the P3-free skeleton, regenerated from the salt's seed and the committed counts, reproduces
     battery_commit. On a mismatch it refuses and names the bound files changed since the mint;
   - **bound code:** no bound file differs from the commitment (§4, bound-code drift).
2. **Descriptive records, computed on the skeleton.** The skeleton carries every field of the battery except the P3
   wording, so these values are those of the rendered battery:
   - the UTF-8 byte length of the longest R3 gold target, plus 1 for EOS. The R3 target does not depend on the rule's
     wording. Since a byte-level BPE token covers at least one byte, this bounds the gold length in tokens. A flag
     records whether `max_new` covers it (descriptive);
   - the battery's training overlap with every training-row file: recorded and read-back semantic hashes, skeleton
     hashes, and status words found in prompts.
3. **P3 read (last).** `open-sealed` reads P3_v2. This is the first read of P3 by any code.
   - The sha256 is checked against the commitment before parsing; then the ids, the single-slot form and the per-cell
     counts. No error message quotes P3 text.
   - The wordings are registered as `p3:<id>` in memory only, and `p3` is rendered.
4. **Record.** `SEALED_OPEN.json` records:
   - the salt, the seed, and the rendered battery's sha256 and per-surface counts;
   - the adapter shas;
   - `max_new`, the R3 gold byte bound and its flag;
   - the training overlap, with the list of parents that have any hash overlap. The analysis drops them from every arm
     (§5.1) and reports the count.

`SEALED_OPEN.json` contains the salt. It is committed, with its sha256 logged, before the first SEALED evaluation.

**P3 text stays hidden until the analysis.**
- P3 is read only by `open-sealed`, `eval` and `analyze_e01r.py`, in memory.
- No command prints P3 text, and the rendered battery is never written to disk.
- The lead does not view P3 text or the rendered `p3` records until the analysis has run.
- Evaluation and analysis rebuild the battery in memory and check it against the opening record.
- If E01R ends before opening (§10), P3 stays unread.

**Evaluate** (`eval --battery sealed --arm <arm> --seed k`).
- **Cells:** N-CFSD s0–s2, E-R2 s0–s2 and E-R3 s0–s2 (the nine confirmatory cells), then OFF (descriptive). `eval`
  refuses N-VR.
- **It refuses:**
  - before opening;
  - if the battery rebuilt in memory differs from the opening record;
  - for an adapter other than the one recorded at opening;
  - for any cell of a committed arm that was not fixed at opening;
  - if `max_new` in the manifest or in `SEALED_OPEN.json` differs from the commitment;
  - if any bound file differs from the commitment (§4), both before decoding and before post-processing a stored
    raw-generations file.
- **Decoding.**
  - Greedy, through the unchanged `lora_cpu.generate`.
  - Its default batching: batch size = min(32, 3000 // (prompt tokens + `max_new`)).
  - `max_new` is the committed value of the frozen §3 rule: max(160, the tokenizer-measured longest R3 gold target +
    16). It is measured at build over the R3 training targets and the E01N DEV battery's `in` records of the E01R
    strata (seed 20261240).
  - The same `max_new` and batching apply to every arm, including N-CFSD and OFF.
- **Parsing (frozen §3).** R3 outputs are parsed from the last non-empty line; every other arm's output is parsed whole
  by `native_probe.parse_answer`.
- **Outputs** (create-only):
  - the raw generations, with the prompt token lengths and the decode log that map each output to its EOS flag,
    stored before any post-processing;
  - `eval_sealed_<tag>.json`: scores, composites, the truncation count (outputs with no EOS within `max_new`) and the
    bound-code shas;
  - `preds_sealed_<tag>.jsonl.gz`: raw outputs and per-record scores, and for R3 `trace_target.predicate_consistency`.
- **Unknown truncation.** If the decoded rows cannot be mapped back to the battery's records, the eval does not fail.
  It records the truncation as unknown (`truncated` null, with a flag), and the analysis accepts and reports it (§5.1).
- **Console.** `eval` prints only that the cell completed, with its counts (records, truncated outputs, wall time). It
  prints no score, composite, `p3` outcome or decision.
- **Lane.** `train`, `smoke` and `eval` refuse unless the CX D-36 lane lock is held by `rcx`.

**Analyse** (§10). `analyze_e01r.py` runs once:
- **No output option.** It writes `ANALYSIS_E01R.md` and `ANALYSIS_E01R.json` at fixed paths in `runs/RCX-E01R/cells/`.
- **Start marker.** Once the refusal checks below have passed, and before it computes any statistic, it creates the
  marker `ANALYSIS_E01R.started`, create-only. It refuses at once if the marker or either output already exists.

**Refusal checks.** It refuses unless:
- every eval names the rendered battery's sha256;
- every adapter matches the open-sealed record;
- every eval used the committed `max_new`;
- every stored score equals a re-score of the raw output;
- every stored truncation flag equals the one re-derived from the raw file's decode log (`native_e01r.map_finished`).
  A cell whose eval recorded the truncation as unknown is accepted, and reported as such;
- the bound code matches the commitment (above);
- `--missing` names exactly the cells declared missing in DECISIONS (§10):
  - it refuses if a named cell has a complete eval;
  - it refuses if an unnamed confirmatory cell lacks one;
  - it refuses a named cell unless §10's missing-cell check holds for it: the failed attempts recorded in
    `ATTEMPTS_E01R.jsonl`, or an adapter or eval-record failure.

**Training-row audit** (`audit_e01r.py`, before the full freeze). `train` and `mint-commit` refuse rows without a
passing `AUDIT_E01R.json` for their sha. These are E01N's `audit_native` hard checks, applied to E-R2 and E-R3:
- exactly 1,501 rows, with the manifest's sha and count;
- the frame is byte-exact, the user turn is canonical, and the rule sentence reads back uniquely and re-renders
  byte-exact (the E01R reader of review item 5; under B4 no registered `tr:` wording exists);
- the gold is re-derived from the prompt:
  - E-R2: `target_json(after)`;
  - E-R3: exactly two lines; line 2 equals `target_json(after)`, and line 1 equals `trace_target.check_line(…)` and
    parses back to the gold decision;
- the recorded semantic hash equals the read-back one;
- only TRAIN words and the posted/void anchors appear; there are no DEV or SEALED words;
- there is no h0/h1 fragment, held-out template, held-out rule factor, size-3 set or M1/M2 magnitude;
- there is no RCWM held-out text;
- there is no `p2:`, `p3:` or `tr:` wording id;
- there is zero semantic and zero skeleton overlap with:
  - the E01N DEV battery (seed 20261240);
  - the DEV2 battery (seed 20261261, both surfaces);
  - the opened E01N SEALED battery (seed 88527345, `in` + `ho_tpl`);
- **R2 follows its frozen spec.** `build` generates E-R2's rows with `recipes.make_r2(1501, seed=20261264)`, and
  `tests/test_e01r_recipes.py` checks that generator:
  - every single-atom parent gets k ∈ {2, 3} complement pairs, less any pair dropped by the frozen redraw rule, each
    under a distinct training wording;
  - for S+ parents, `comp_unlisted` → no-op and `comp_listed` → apply; for S− parents the mirror;
  - both members of a pair share state, event and surface;
  - only whole parents appear, and the remainder up to 1,501 rows is filled with base-only parents.

  The audit's hard checks above apply to every E-R2 row;
- **R3 control match:** prompts byte-identical to N-CFSD's, in order, and line 2 equal to N-CFSD's target;
- with the tokenizer, no row is longer than `max_len` 512 tokens.

The audit also reports the §3 covariates per arm and for the control, including the supervised and processed tokens
per training seed (`tokens.per_seed`; the mint requires them for every kept arm).

The count of E-R2 `comp_listed` twins that duplicate their parent's base or status_swap record (same wording, rule,
state and event) is computed by `analyze_e01r.py` from the committed `train_E-R2.jsonl`. These occur in size-2 sets,
which the frozen spec does not redraw (§5.5).

**Shared with earlier batteries by design.**
- **The sealed word pools.** They are also used by the opened E01N SEALED battery (seed 88527345), whose results were
  seen only in aggregate. No sealed word occurs in any training row.
- The anchors posted and void.
- The F4 near-miss probes and the F5 posted-like strings.
- The canonical t0 wording on `in`.

**Limit.** The commitment proves that the battery was fixed at the mint. It cannot stop the salt or P3 from being read
early; doing so is a protocol violation and voids the confirmation (§11).

## 5. Endpoints

Every contrast is X − N-CFSD, for X ∈ {E-R2, E-R3} (the committed `treatment_arms`).

### 5.1 Inputs and record scores

- **Battery.** The opened SEALED battery, rebuilt in memory and checked against `SEALED_OPEN.json`. Each record carries
  `parent_id`, `group` (the stratum), `variant`, `surface` (`in` or `p3`), `rule_ast.op` (S+ or S−) and
  `gold.decision` (`apply` or `noop`).
- **Cells.** One prediction per record for each of N-CFSD_s0–s2, E-R2_s0–s2 and E-R3_s0–s2.
- **Record score.** `score.classify(rec, parse(out, keys))`, with keys = `list(rec["gold"]["after"])`:
  - parse is the last-non-empty-line parser for E-R3, and `native_probe.parse_answer` on the whole output for N-CFSD
    and E-R2 (frozen §3);
  - `exact` ∈ {0, 1} and `decision_ok` ∈ {0, 1}. For R3 these come from the parsed balances, never from line 1's stated
    decision;
  - **format error** = 1 if parse returns None, else 0;
  - unparseable outputs score 0 for both `exact` and `decision_ok`;
  - truncated outputs (cut at `max_new`) are scored as they stand.
- **Re-score.** The analysis recomputes every score from the raw output and refuses on any difference from the stored
  score.
- **Truncation re-check.** The analysis re-derives each output's truncation flag from the raw file's decode log
  (`native_e01r.map_finished`) and refuses on any difference from the stored flag. Where the eval recorded the
  truncation as unknown (the decoded rows could not be mapped to records), the cell is accepted and its truncation is
  reported as unknown. Truncation never enters a score other than through the output text.
- **Train-overlap drop.** Every parent in `SEALED_OPEN.json` `train_overlap.parent_ids` is removed from every endpoint,
  guard and descriptive quantity, for every arm, before anything is computed. The count is reported.

### 5.2 Record selection

Each quantity uses the records below, after the drop. Each set is also defined by a predicate on (surface, strata,
`rule_ast.op`, `gold.decision`). The analysis checks that the predicate and the variant list select the same records,
and refuses otherwise. In V7 the polarity is drawn per parent, so "V7 (S+)" means the V7 parents whose rule is S+;
there are about 20 of each polarity, the exact counts set by the draw.

**Confirmatory (Holm family, §6):**

| ID | Surface | Records (stratum: variant) | Predicate | Score | Parents |
|---|---|---|---|---|---|
| **C1** | `p3` | F1–F5: `probe`; V1: `unlisted`; V5: `pos_x_status_posted`; V7 (S+): `unlisted` | op = S+ and gold = noop | `decision_ok` | ≈ 300 (200 + 40 + 40 + ≈ 20) |
| **C2** | `p3` | every variant of F2, F3, F5, V1, V2, V3, V5 (E01N's composite) | stratum ∈ composite | `exact` | 280 |

**Protected strata (ADVANCE guards; not in the Holm family):**

| ID | Surface | Records (stratum: variant) | Predicate | Score | Parents |
|---|---|---|---|---|---|
| G1 | `in` | every variant of F2, F3, F5, V1, V2, V3, V5 | stratum ∈ composite | `exact` | 280 |
| G2 | `in` | F1–F5: `twin_posted` | F stratum and variant = `twin_posted` | `exact` | 200 |
| G3 | `in` and `p3` | every record of all 10 strata | all | format error | 400 |
| G4 | `p3` | F1–F5: `twin_posted`; V1: `listed`; V3: `pos`; V5: `pos_x_status_x`; V7 (S+): `listed` | op = S+ and gold = apply | `decision_ok` | ≈ 340 |
| G5 | `p3` | V2: `unlisted`; V5: `neg_x_status_posted`; V7 (S−): `unlisted` | op = S− and gold = apply | `decision_ok` | ≈ 100 |
| G6 | `p3` | V2: `listed`; V3: `neg`; V5: `neg_posted_status_posted`; V7 (S−): `listed` | op = S− and gold = noop | `decision_ok` | ≈ 140 |

**Non-anchor sub-quantities (descriptive; same estimator):**

| ID | Surface | Records (stratum: variant) | Predicate | Score | Parents |
|---|---|---|---|---|---|
| C1-nF | `p3` | V1: `unlisted`; V5: `pos_x_status_posted`; V7 (S+): `unlisted` | C1 without the F strata | `decision_ok` | ≈ 100 |
| G4′ | `p3` | V1: `listed`; V3: `pos`; V5: `pos_x_status_x`; V7 (S+): `listed` | G4 without the F strata | `decision_ok` | ≈ 140 |
| G5′ | `p3` | V2: `unlisted`; V7 (S−): `unlisted` | G5 without V5 | `decision_ok` | ≈ 60 |

A parent contributes at most one record per seed to C1, C1-nF, G2, G4, G4′, G5, G5′ and G6. It contributes 2 or 4
records to C2 and G1, and up to 8 to G3.

### 5.3 Per-parent statistic, seed pooling and contrasts

For a quantity Q, an arm A (cells A_s0, A_s1, A_s2) and a parent π of Q's parent set:
- v_A,k(π) is the mean score over Q's records of π in cell A_sk (records weighted equally);
- v_A(π) = (v_A,0(π) + v_A,1(π) + v_A,2(π)) / 3, so the seeds are weighted equally;
- the level is L_A(Q) = the mean of v_A(π) over Q's parents (parents weighted equally);
- the **point contrast** is d_X(Q) = L_X(Q) − L_N-CFSD(Q) = the mean over π of [v_X(π) − v_N-CFSD(π)];
- the **matched per-seed contrast** is d_X,k(Q) = mean_π v_X,k(π) − mean_π v_N-CFSD,k(π), for k = 0, 1, 2
  (`rcx.stats.parent_macro` of each cell).

Q's parent set is the same for every arm: it is fixed by the battery and the drop. The interval and p-value are in §6.

### 5.4 Why G4–G6, and why the non-anchor sub-quantities

C1 can be won by degenerate policies, and each is caught by a guard:

| Degenerate policy | Effect on C1 | Caught by |
|---|---|---|
| "no-op whenever the status is unlisted" (polarity-blind) | wins C1 | G5 (NEG apply collapses) |
| "no-op everywhere" | wins C1 | G4 and G5 |
| "apply every listed status" | — | G6 |
| "apply only when the status is `posted`" (anchor trigger) | wins most of C1 | G4 and G5, through their non-anchor items |

**The anchor dilution.** The F strata use the fixed posted rule, so their `twin_posted` items carry the anchor status
`posted`, which every training regime pairs with "apply". These items are 200 of G4's ≈ 340 parents. V5's
`neg_x_status_posted` items carry `posted` too, and they are 40 of G5's ≈ 100.
- A recipe that loses about 0.12 on the ≈ 140 non-anchor parents of G4 while the anchor items stay at ceiling moves G4
  by only about 0.05, and can pass G4's margin.
- G4′ and G5′ remove the anchor items, so they show such a loss undiluted. C1-nF shows whether a C1 gain reaches
  beyond the posted rule.

C2 alone guards weakly, because the C1 items carry about a third of its parent weight. A policy that wins C1 at the
cost of a large G4–G6 loss also cannot reach REVISE: §7 makes it STOP (degenerate trade-off).

**Pinned by tests.** `tests/test_analyze_e01r.py` runs four degenerate policies end to end through
`analyze_e01r.py`, on synthetic cells: always no-op; no-op on `p3` only; no-op if and only if the status is unlisted;
and apply only when the status is `posted`. Each must end STOP (degenerate trade-off), and none ADVANCE or REVISE.

**INFO: partial polarity-blind policies** (stated, not acted on; nothing here changes a decision).
- Suppose a treatment follows the polarity-blind policy ("no-op whenever the status is unlisted") on a share f of the
  `p3` items, and behaves like the control otherwise.
- **C1 gain.** Its expected d(C1) is at most f, since each affected item can gain at most 1 (about f × 0.5 at step A's
  headroom).
- **G5 loss.** G5's point falls by about f × the control's S− apply level (0.93 on `ho_tpl`, R-08). It reaches the
  −0.15 degenerate point only from f ≈ 0.16.
- So a mixture on about 15% of the `p3` items or less is not caught by the degenerate rule:
  - **It can still reach REVISE**, if its C1 gain is significant. The report then names the failed ADVANCE conditions:
    d(C1) < +0.25, and G5 whenever its lower bound misses −0.10.
  - **It never reaches ADVANCE.** Its expected d(C1) is at most 0.15, about 3 SE or more below +0.25 (§7), and as f nears
    0.15 its G5 lower bound also misses the margin.

**Pre-stated readings (descriptive; they never change a decision):**
- A treatment whose G4′ or G5′ point contrast is ≤ −0.15 is labelled **"degenerate on non-anchor items"** in the
  report. The label is repeated wherever its outcome is cited.
- "C1 gain beyond the posted rule: yes" if C1-nF's CI lower bound is > 0; otherwise "not shown".

### 5.5 Declared asymmetries

1. **Parser leniency (R3).**
   - R3's last-line parser accepts an output whose last non-empty line is a valid answer, whatever precedes it.
     Whole-output parsing (N-CFSD, E-R2) fails such an output, because it reads any preamble as part of the answer.
   - The two parsers agree on every output with one non-empty line. They differ only on outputs with several
     non-empty lines:
     - a preamble before a one-line answer is accepted by the last-line parser and rejected by whole-output parsing.
       This is the relevant direction, and it favours R3 on every endpoint and on G3;
     - an answer spread over several lines is the reverse case. E01N's native outputs were compact JSON of at most 39
       bytes.
   - **Symmetric re-score (descriptive).** Every arm's raw outputs are re-scored with the last-line parser. C1, C2
     and G1–G6 are recomputed under it with the §6 estimator.
   - Per cell, the report gives the number of records on which the two parsers disagree.
   - Pre-stated reading: if any §7 condition for an arm holds under one scoring and fails under the other, the report
     says so and names the condition. The decision uses the frozen scoring.
2. **R2 duplicate twins.**
   - The frozen R2 spec redraws a `comp_listed` twin that duplicates its parent's base or status_swap record (same
     wording, rule, state and event) only for size-1 sets.
   - In size-2 sets such duplicates are kept, and they count towards the 1,501 rows.
   - Their count is computed by `analyze_e01r.py` from the committed `train_E-R2.jsonl` (the file whose sha the
     commitment binds) and reported as a covariate, together with `recipe_stats`' `dup_rows_within_parent`
     (`TRAIN_MANIFEST_E01R.json`).
3. **Declared in §3 (frozen) and repeated here.** R2's parent count and rule-type mix differ from N-CFSD's. R3 adds
   supervised and generated tokens. Neither is matched.

### 5.6 Secondary (descriptive; no test)

- **Per form and per template.**
  - C1 and G4–G6 levels and contrasts per form (LF, CF, IM). Wilson intervals are computed on the record counts pooled
    over the three seeds.
  - Per P3 template: ids and counts only.
  - **Template-cluster sensitivity for C1.** 10,000 resamples, seed 20261268. In each resample:
    - for each form f, draw with replacement as many POS P3 templates of f as C1 uses;
    - for each drawn template, draw with replacement as many of its C1 parents as it has;
    - the statistic is the mean of v_X(π) − v_N-CFSD(π) over all drawn parents.

    The report states "C1 gain shown across P3 wordings: yes/no", according to whether this interval's lower bound is
    > 0.
- **Arm contrasts.** E-R2 − E-R3 head to head (paired, §6 estimator), and each arm against OFF.
- **Seeds.** Per-seed values and spread, and the matched per-seed contrasts with a t interval (df = 2).
- **Behaviour.**
  - Every stratum, including F1, F4 and V7, on both surfaces;
  - error directions;
  - V3 polarity sensitivity;
  - the PREREG-3-style verdict;
  - the `in` analogue of C1.
- **Sub-quantities and re-score:** C1-nF, G4′ and G5′ (§5.2, §5.4), and the symmetric re-score (§5.5).
- **R3 trace diagnostics** (`trace_target.predicate_consistency`):
  - well-formed rate;
  - the stated decision's accuracy, including C1 recomputed from it;
  - agreement between the balances and the stated decision;
  - line-1 exactness and coherence;
  - `listed=` accuracy, which tests membership.

  On S+ unlisted items, "listed=no only-listed:true" or "-> apply" is the complement-inference error itself.
- **Generation.** Truncation counts per cell ("unknown" where the eval could not map its outputs, §5.1), and each
  treatment adapter's smoke profile (TRAIN prompts; §10).
- **Covariates** (§3, and §5.5 item 2).

## 6. Statistics

**Estimator.** For each quantity Q of §5 and each treatment arm X, one call:

`rcx.stats.seed_parent_bootstrap_test(per_seed_a=[N_s0, N_s1, N_s2], per_seed_b=[X_s0, X_s1, X_s2], resamples=10000,
seed=20261265, alpha=0.05, p_floor=None, ties="both")`

- Each element maps parent_id → the list of Q's record scores of that parent in that cell (§5.2, after the drop).
- `diff` is d_X(Q) (§5.3). `lo` and `hi` are the 95% percentile interval. They equal `rcx.stats.seed_parent_bootstrap`
  with `resample_seeds=False` exactly.
- **This is a seed × parent paired cluster bootstrap.**
  - Within each parent, variants are averaged per seed, and then the three seeds are averaged.
  - Each replicate draws as many parents as Q has, with replacement, from the sorted parent list, using
    `random.Random(20261265)`. Its value is the mean of the drawn parents' contrasts v_X(π) − v_N-CFSD(π).
  - Seeds are not resampled.
- **10,000 resamples.** Every call starts from the same bootstrap seed, **20261265**. This covers C1, C2, G1–G6, the
  sub-quantities, the symmetric re-score and E-R2 − E-R3.
- **CI.** The 2.5% and 97.5% quantiles of the replicates (`rcx.stats._quantile`, linear interpolation). It is not
  multiplicity-adjusted.
- **p-value.** Two-sided, from the same replicates: p = 2 × min(#{d* ≤ 0}, #{d* ≥ 0}) / 10,000.
  - Replicates within 1e-12 of 0 count as 0 and are counted on **both** sides (`ties="both"`).
  - Floored at 1/10,000 and capped at 1.
- **Holm.** `rcx.stats.holm` at α = 0.05 over the raw p of the four primary tests {C1_E-R2, C2_E-R2, C1_E-R3,
  C2_E-R3}.
  - The family is fixed by `treatment_arms` in `SEALED_COMMIT.json`. If an arm was dropped at the mint (§4), the
    family is the remaining arm's 2 tests.
  - A test that cannot be computed confirmatorily (a missing cell, §10) enters Holm with p = 1.
  - G1–G6 and every descriptive quantity are not tested and are not in the family.
- **Conditioning.** Inference is conditional on:
  - the 9 adapters;
  - the sealed word pools;
  - the 48 P3 wordings, of which 23 are positive.

  Seed variation enters through the per-seed condition (§7), and wording variation through the template-cluster
  sensitivity (§5.6).
- **Tolerance.** Every threshold comparison uses ε = 1e-9 (as `analyze_native.py`):
  - x ≥ t means x ≥ t − ε;
  - x > t means x > t + ε;
  - x < t means x < t − ε;
  - x ≤ t means x ≤ t + ε.

## 7. Decision rule

**Notation, per treatment arm X.**
- d(Q) is the point contrast X − N-CFSD (§5.3).
- [lo(Q), hi(Q)] is its 95% percentile CI (§6).
- p_H(Q) is the Holm-adjusted p (§6).
- d_k(C1), for k = 0, 1, 2, are the matched per-seed C1 contrasts.

All comparisons use the §6 tolerance.

**Derived conditions:**
- **sig(X):** (p_H(C1) < 0.05 and d(C1) > 0) or (p_H(C2) < 0.05 and d(C2) > 0).
- **guards_ok(X):** all of:
  - lo(G1) > −0.03;
  - lo(G2) > −0.05;
  - hi(G3) ≤ +0.02;
  - lo(G4) > −0.05;
  - lo(G5) > −0.10;
  - lo(G6) > −0.08.
- **degenerate(X):** d(G4) ≤ −0.15, or d(G5) ≤ −0.15, or d(G6) ≤ −0.15. These are point estimates (CX D-38).

**Ordered rule, per treatment arm X.** The outcomes are tested in this order, and the first that holds is the
decision:

0. **INCONCLUSIVE (missing).** At least one of the six SEALED cells X_s0–s2 and N-CFSD_s0–s2 is missing (§10).
1. **ADVANCE.** All of:
   - p_H(C1) < 0.05;
   - d(C1) ≥ +0.25;
   - lo(C1) > 0;
   - d_k(C1) ≥ +0.10 for each of k = 0, 1, 2;
   - p_H(C2) < 0.05 and d(C2) > 0;
   - guards_ok(X).
2. **STOP.** hi(C1) < +0.10 and hi(C2) < +0.05.
3. **STOP (degenerate trade-off).** sig(X) and degenerate(X).
4. **REVISE.** sig(X) and d(C1) > 0.
5. **INCONCLUSIVE.** Otherwise.

**Experiment-level outcome:**
- If the battery is never opened (a smoke failure or an incomplete training cell; §10), E01R is **INCONCLUSIVE**. There
  is no analysis, and P3 stays unread.
- Otherwise the outcomes are tested in this order:
  1. ADVANCE if any arm reaches ADVANCE (each such arm is named);
  2. STOP if every arm in the Holm family reaches STOP of either kind;
  3. REVISE if any arm reaches REVISE;
  4. otherwise INCONCLUSIVE.
- If both arms reach ADVANCE, the lead candidate is the arm with the larger d(C1). A tie (within ε) goes to the arm with
  fewer supervised tokens (the seed mean, `AUDIT_E01R.json`).

**Report.** `analyze_e01r.py` reports:
- every clause value;
- every later outcome whose conditions also hold;
- for REVISE, each failed ADVANCE condition;
- the §5.4 labels and the §5.5 re-score reading.

**Changes from E01N's rule:**
- ADVANCE rests on the targeted C1 only. In E01N it could rest on either endpoint.
- C2 must be significantly positive, which is stronger than E01N's "other endpoint lower bound > −0.05".
- The per-seed bar is ≥ +0.10 instead of > 0. C1 has large headroom, and a gain carried by one seed would not be a
  dependable recipe.
- There are six protected strata, with the CX D-38 margins.
- A significant gain whose G4, G5 or G6 point contrast is ≤ −0.15 is STOP (degenerate trade-off).
- REVISE also needs d(C1) > 0. A significant C2 gain with C1 ≤ 0 is INCONCLUSIVE, because it does not bear on the
  targeted residual.

**Guard margins (CX D-38; fixed before any step A END line; G4–G6 from design constants only):**

| Guard | Within margin if | Basis |
|---|---|---|
| G1 `in` composite | lo > −0.03 | the draft margin, kept by CX D-38 (below) |
| G2 F0 posted twins | lo > −0.05 | E01N's margin |
| G3 format-error rate | hi ≤ +0.02 | E01N's margin |
| G4 POS apply (≈ 340 parents) | lo > −0.05 | m ≥ 3.24·SE under the D-38 constants |
| G5 NEG apply (≈ 100) | lo > −0.10 | SE ≈ 0.028 |
| G6 NEG no-op (≈ 140) | lo > −0.08 | SE ≈ 0.024 |

**Threshold justification:**
- **C1 +0.25 (fixed under every branch, CX D-38).**
  - On `ho_tpl`, N-CFSD's polarity gap was 0.97 − 0.14 = 0.83 (S− vs S+ no-op), and +0.25 is about 30% of it.
  - If P3 resembles `ho_tpl`, +0.25 takes the control's 0.14 to about 0.39. Anything smaller leaves "apply" as the
    dominant answer on unlisted statuses.
  - Taking step A's P2 proxy p̄ = 0.50 as the control's level, +0.25 would mean about 0.75. That proxy was known only
    after CX D-38 (below).
  - C1's items carry 90/280 = 0.32 of C2's parent weight:
    - F2, F3 and F5 probes: half of each parent;
    - V1 `unlisted`: half;
    - V5: a quarter.

    So +0.25 on C1 corresponds to about +0.08 on C2, commensurate with E01N's +0.10 composite effect.
- **Per seed +0.10.** This is 40% of the practical effect. A seed below it has not materially learned the complement
  case.
- **STOP at +0.10 on C1.** This equals E01N's practical effect. An upper bound below it means that less than a fifth of
  the step A headroom (about 0.50) was closed. **STOP at +0.05 on C2** is E01N's STOP bound.
- **G1 −0.03.** N-CFSD's SEALED `in` gain over N-VR was +0.018 [+0.006, +0.030]. A loss beyond 0.03 would give back
  more than that gain's upper bound, i.e. drop to N-VR's level on training wordings.
- **G2 −0.05 and G3 +0.02** are E01N's margins.
- **G4 −0.05, G5 −0.10 and G6 −0.08** (CX D-38).
  - Each is 3.24·SE rounded up to the next 0.01, at about 340, 100 and 140 SEALED parents, so that m ≥ 3.24·SE. This
    assumes a per-parent SD of 0.2 for each arm's seed-averaged value and a correlation of 0 between the arms.
  - Each still catches R-08's over-restriction on held-out OR wordings (apply 84/87 → 66/87, −0.21).
- **Degenerate trade-off at a point estimate ≤ −0.15** (G4–G6; CX D-38, prereg review finding 4).
  - −0.15 lies between the guard margins (−0.05 to −0.10) and R-08's −0.21.
  - A point rule is used, not a CI bound. At G5's SE of about 0.028, a rule on the lower bound would label a true loss
    of about −0.095 degenerate half the time.
  - A REVISE invites a follow-up of the same recipe. An observed loss of 0.15 or more in a protected decision class
    suggests that the C1 gain is a shifted default (the degenerate policies of §5.4), not complement reading.
- **REVISE needs d(C1) > 0** (CX D-38, review finding 9). REVISE names a recipe worth refining for the C1 residual.

**Guard pass probabilities (stated honestly; nothing here changes a margin).**

*What D-38's constants imply.* The per-parent SD of the paired difference is √2 × 0.2 = 0.28, and SE = 0.28 / √n.
With a true guard effect δ, P(within margin) ≈ Φ((m + δ)/SE − 1.96).

| Guard | Parents | Margin m | SE | m/SE | P(pass \| δ = 0) | P(pass \| δ = −0.02) |
|---|---|---|---|---|---|---|
| G4 | ≈ 340 | 0.05 | 0.015 | 3.26 | 0.90 | 0.50 |
| G5 | ≈ 100 | 0.10 | 0.028 | 3.54 | 0.94 | 0.81 |
| G6 | ≈ 140 | 0.08 | 0.024 | 3.35 | 0.92 | 0.71 |
| G4–G6 jointly (independence) | | | | | **0.78** | **0.28** |

- **Per guard.** P(pass | 0) ≥ 0.90 holds by construction, but only under these constants.
- **Jointly** the probability is lower: about 0.78 under independence, and at most 0.90 (the smallest single guard).
  The three guards use different items, partly from the same V3, V5 and V7 parents.
- **G4 binds.** A true loss of only 0.02 in POS apply halves its pass probability.
- **Sensitivity** (stated, not acted on). Single-item parents with accuracies of 0.7–0.9 can have per-arm SDs near 0.3.
  - At per-arm SD 0.3 and arm correlation 0.5, the paired SD is 0.30: the per-guard probabilities are 0.87–0.92 and
    the joint one 0.70.
  - At per-arm SD 0.3 and correlation 0, the paired SD is 0.42: the per-guard probabilities are 0.58–0.65 and the
    joint one 0.23.
- **G1–G3 are not covered by D-38's constants.** Under them, G1 (280 parents, m = 0.03) would pass only about 0.43 of
  the time at δ = 0. G1–G3 rely instead on near-ceiling behaviour on `in` and on format:
  - R-08's N-CFSD − N-VR intervals imply SEs of about 0.006 (`in` composite) and 0.010 (F0 twins), and a format-error
    rate near 0. That gives P(pass | 0) ≈ 1.00.
  - A recipe that changes the output format (R3) may not share that behaviour. G3 exists to catch it.

**Power for C1 and the decision (stated honestly):**
- **Baseline.**
  - The control's C1 level on `p3` is unknown.
  - The step A analogue, the same selection on DEV2's `p2` with N-CFSD s0–s2 pooled, is p̄ = 0.496 (R-10; DEV, on
    wordings read by the lead). So the headroom is about 0.50.
  - R-08's `ho_tpl` value was 0.14.
- **Sampling error.**
  - C1 has about 300 parents, with one item per parent and seed. The seed-averaged values are 0, 1/3, 2/3 or 1.
  - A per-parent paired-difference SD of 0.4–0.6 gives a parent-bootstrap SE of about 0.025–0.035 (CI half-width about
    0.05–0.07).
  - The bootstrap does not include seed-to-seed variation. E01N's C2 per-seed SD was about 0.03 at a +0.09 effect, and
    a targeted endpoint may vary more.
- **The effect needed for REVISE (Holm over 4 tests).**
  - When C1_X is the only test that reaches significance, Holm multiplies its p by 4. It then needs a raw p < 0.0125
    (z ≈ 2.50).
  - At an SE of 0.025–0.035, a true C1 effect of about +0.06–0.09 reaches this half the time, and one of about
    +0.08–0.12 reaches it 80% of the time.
  - When other tests in the family are also small, the step-down multiplier drops to 3, 2 or 1. At the p floor of
    1e-4, the Holm-adjusted p is ≤ 4e-4.
  - REVISE also needs d(C1) > 0 and no G4–G6 point ≤ −0.15.
- **P(ADVANCE).** This is approximately P(d(C1) ≥ +0.25) × P(all guards pass).
  - P(d(C1) ≥ +0.25) assumes a total SD of the C1 point of about 0.05 (parent sampling plus seed variation).
  - The other ADVANCE conditions are close to 1 when the true effect is ≥ +0.25, so they are left out of the table:
    - Holm on C1 and C2;
    - d(C2) > 0: C2 ≈ 0.32 × C1, with an SE of about 0.011–0.015;
    - the per-seed bar: ≥ 0.95 at a per-seed SD ≤ 0.07.
  - The guard factors assume true guard effects of 0.

  | True C1 effect | +0.35 | +0.30 | +0.25 | +0.20 |
  |---|---|---|---|---|
  | P(d(C1) ≥ +0.25) | 0.98 | 0.84 | 0.50 | 0.16 |
  | P(ADVANCE), D-38 guard constants (joint 0.78) | 0.76 | 0.66 | 0.39 | 0.12 |
  | P(ADVANCE), paired SD 0.42 (joint 0.23) | 0.23 | 0.20 | 0.12 | 0.04 |

  ADVANCE is therefore likely only if the true C1 effect is at least about +0.30 *and* the guards' true effects are
  near 0. An effect near +0.25 is at best a coin flip before the guards, and REVISE is then the likely outcome.
- **C2.** A +0.30 C1 effect implies about +0.10 on C2 (E01N's C2 CI half-width was 0.022). C2 binds only if the recipe
  also costs negative-rule, V3 or arithmetic accuracy.
- **Main risks.**
  - R3: format (G3), arithmetic inside `after` (C2), and non-terminating outputs (the smoke test, §10).
  - R2: over-restriction (G4, G5) and the rule-type shift (§3).
- **Wordings.** There are 23 positive P3 wordings. A C1 gain carried by a few of them can still pass. The
  template-cluster sensitivity (§5.6) makes that visible but does not change the decision.

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
- **Claims.**
  - No retention, dense-model or general-ability claim is made until E33 reports. The adapters stay local and
    unmerged.
  - A §5.4 label ("degenerate on non-anchor items") and the §5.5 re-score reading go with every citation of the
    outcome.
- **If X = E-R3.** Before any mechanism claim ("the predicate step causes it"), a token-matched filler arm
  (`target_fill`, the D-34/4b format) is needed. It is proposed as a follow-up and is not part of E01R.
- **Next.** RCM moves to E02 (computed predicates), with X as the base-recipe candidate (the E02 prereg decides).

**REVISE.**
- No E33 request and no Q-03 extension. The report names the failed condition.
- Follow-up options (any of them needs a new prereg):
  - **d(C1) ≥ +0.25 but a guard failed** (for G4–G6, a point above −0.15 with a lower bound at or below the margin):
    revise the recipe against that guard. For example, a G5 failure calls for stronger polarity-balanced twins.
  - **C1 significant but below +0.25:** combine R2 and R3, or raise the row share.
- Every follow-up needs a newly authored blind pool (P4) and a new sealed battery, because P3 is spent after this
  analysis.

**STOP (degenerate trade-off; arm X).**
- X moved C1 or C2 significantly, but a G4, G5 or G6 point contrast is ≤ −0.15. The recipe may have shifted a default
  rather than taught complement reading.
- Logged as a negative result for X, naming the guard and its point and interval.
- No REVISE-style follow-up of X. A successor needs a new hypothesis about the trade-off (for example
  polarity-balanced supervision), a new prereg and P4.
- At the experiment level it counts as STOP (§7).

**STOP (every arm).**
- Neither complement-focused twins nor a predicate-first trace fixes the residual at this budget without a trade-off.
  This is logged as a negative result, with the per-form profile and, for any degenerate-trade-off arm, the guard that
  failed.
- RCM moves to E02.
- The complement limitation stays attached to every citation of N-CFSD (as in R-08's limits).
- No further E01R recipe is tried without a new hypothesis.

**INCONCLUSIVE after opening** (an arm with a missing cell, or a pattern that meets no other outcome).
- A descriptive report only. Missing cells are never replaced by another seed or id (§10).
- P3 is spent once opened.

**INCONCLUSIVE before opening** (a smoke failure, or a training cell that cannot be completed; §10).
- The battery is never opened, and nothing is analysed. P3 stays unread and unspent.
- The smoke files (TRAIN prompts only) and the cause are logged in DECISIONS with "outcomes known: TRAIN-prompt smoke
  only".
- The six adapters are never evaluated on any battery under E01R. A successor prereg decides whether to use them, mints
  its own battery and may use P3.

**B0, B1, B2 and B3 (§2).** Not applicable: the branch is B4 (CX D-39).

## 9. Seeds

**Search.**
- **Working tree (2026-10-11).** Grepped for `202612`, excluding `*.secret` files and `pools/`; the readable pool JSON
  files were checked separately.
- **Tracked files (2026-10-12; `git grep` in `complementary/`, `pools/` excluded).** The reserved seeds appear only:
  - in this file;
  - 20261264 in `recipes.py`, `native_e01r.py` and their tests (the R2 role);
  - 20261267 in `decide_branch.py` (the §2 bootstrap, FROZEN).

  Re-checked after commit `5b36b9c` (`git grep` for 20261264, 20261265 and 20261268): they now also appear in
  `analyze_e01r.py` and `tests/test_analyze_e01r.py` (20261265 and 20261268, in their §6 and §5.6 roles) and in
  `NEXT_STEPS.md` (20261264, the R2 role). There is no other use.

  The six readable pool files contain no `202612xx` literal. 20261263 no longer appears in any code: `ca40a38` changed
  `recipes.py`'s R1 default to 20261253.
- **Git history** (review item 19). `git log --all -S<seed>` over all local and remote branches, one seed per command,
  run only when the lane is free:
  - **20261263:** two commits only, both RCX E01R: `1bb6c32` (the CX D-37 freeze, where it was `recipes.py`'s
    suggested R1 default) and `ca40a38` (its removal from `recipes.py`). No other experiment or branch uses it.
  - **PENDING (lane free):** 20261264, 20261265 and 20261268, the seeds this prereg still assigns and has not yet used.
    They are searched before the full freeze and recorded here.
  - **Not searched:** 20261267, which is frozen and already used once (CX D-39), and the unassigned seeds (below).
    Each of these is searched before any later assignment.

  The 20261263 search ran at about 01:28, seconds after step A's first cell had started; the lane check run with it
  already printed LANE_BUSY. It took about 8 s of git I/O and no Python. No further search was run while the lane was
  busy.

**Neighbouring seeds already in use:**
- 20261240: the E01N DEV battery;
- 20261253: N-CFSD's data seed (E01N);
- 20261261: the DEV2 battery;
- 20261262: the step A descriptive bootstrap (`run_e01r.unpaired_parent_boot`);
- 20261270–20261280: E07/BW;
- 20261289 and up: E07/BW.

| Seed | Role |
|---|---|
| 20261263 | **Not used.** It was `recipes.py`'s suggested R1 default and is superseded (R1 is not run under B4). It is left unassigned so that no accidental default run can collide with a fixed role. |
| 20261264 | R2 data generator (`recipes.make_r2`): parents, complement-pair statuses and wordings |
| **20261265** | Bootstrap seed for every contrast of §5–§7: C1, C2, G1–G6, C1-nF, G4′, G5′, the symmetric re-score and E-R2 − E-R3 |
| 20261266 | **Not used** (it was the R1 top-up verification, B3 only). Unassigned. |
| **20261267** | **The §2 step A cluster bootstrap** of E − I (`decide_branch.BOOT_SEED`; FROZEN, CX D-37; used once, CX D-39) |
| 20261268 | Descriptive template-cluster bootstrap (§5.6) |
| 20261269 | **Not used** (it was the guard-power computation, replaced by CX D-38's design constants; nothing resamples). Unassigned. |
| 20261281–20261288 | Reserved for E01R follow-ups; not used by this prereg |

**Seeds outside the block:**
- **SEALED battery seed:** derived from the salt; unknown until opening.
- **Training seeds 0, 1, 2:** set the LoRA initialization and row order, identical to N-CFSD's.
- **R3 base rows:** N-CFSD's (E01N data seed 20261253), unchanged.
- **R3:** draws no randomness.
- **Smoke test:** draws no randomness (fixed row indices, §10).

## 10. Execution and safety

**Order (branch B4):**
1. **Step A** (done; CX D-39, R-10). `decide_branch.py` ran once and wrote `BRANCH_STEP_A.json`: B4, arms R2 + R3.
2. **Rows.** Run `native_e01r.py build --arms E-R2,E-R3`. It loads only the tokenizer, and it writes:
   - the rows;
   - `TRAIN_MANIFEST_E01R.json`, with `max_new` and the R3 gold length in tokens.

   Then `audit_e01r.py --tok` writes `AUDIT_E01R.json`. Unit tests run only when the lane check prints LANE_FREE.
3. **Freeze.** Fill the remaining PENDING fields and complete the review. Then freeze: this file's sha256, the code
   sha (printed by `native_e01r.py code-sha`) and the code, rows, manifest and audit shas go into DECISIONS (the
   freeze entry, §11).
4. **Mint.** Run `mint-commit --expect-code-sha256 <the freeze entry's code sha> --arms-required
   N-CFSD_s0,N-CFSD_s1,N-CFSD_s2,E-R2_s0,E-R2_s1,E-R2_s2,E-R3_s0,E-R3_s1,E-R3_s2` (§4). Use `--drop-arm` only as §4
   allows. Commit `SEALED_COMMIT.json` and log its hashes in the mint entry (§11).
5. **Training and smoke.** In lane chains, train the six treatment adapters (E-R2 s0–s2 and E-R3 s0–s2).
   - Each adapter's smoke test follows its training.
   - **All six adapters are trained whatever the smoke results.** A smoke failure stops neither the chain nor any
     later training.
6. **Opening.** Only if all six adapters are complete and all six smoke tests pass: run `open-sealed
   --expect-commit-sha …`, then commit `SEALED_OPEN.json` and log its sha256. Otherwise E01R ends INCONCLUSIVE
   before opening (§7, §8).
7. **Evaluation.** Evaluate on SEALED, in this order:
   - the nine confirmatory cells: N-CFSD s0–s2, E-R2 s0–s2 and E-R3 s0–s2;
   - then OFF (descriptive).

   N-VR is not evaluated. Each cell's eval files are committed once the cell has ended.
8. **Analysis.** Run `analyze_e01r.py` once, with `--missing` naming exactly the cells declared missing, if any.
   - It has no output option: the report and its JSON go to fixed paths (§4).
   - It creates the create-only marker `ANALYSIS_E01R.started` after its refusal checks pass and before its first
     statistic, and it refuses if the marker exists. A run that refuses computes, prints and writes no statistic. Such
     a run is not the analysis, and fixing its cause (for example declaring a cell missing, §10) and running again is
     allowed.
   - Once the marker exists, the analysis has run, whether or not it finished.

**Smoke test (procedure for the frozen §3 rule; review item 16).**
- **When.** After each treatment adapter's training and before `open-sealed`, in the lane chain:
  `native_e01r.py smoke --arm <arm> --seed k`. `smoke` refuses after opening, for an adapter that is not a completed
  run, and if any bound file differs from the commitment (§4).
- **Prompts.** 50 rows of the arm's own training file, at fixed indices i × 1501 // 50 for i = 0–49 (no randomness).
  These are TRAIN prompts only, never DEV, DEV2 or SEALED.
- **Decoding.** Exactly as in the SEALED eval: greedy, `lora_cpu.generate`, the committed `max_new` and the same
  batching.
- **Logged** in `SMOKE_<tag>.json` (create-only, with the raw outputs and the bound-code shas):
  - the truncated count, i.e. outputs with no EOS within `max_new`;
  - the well-formed share: for R3, both lines parse (`trace_target.parse_trace_answer`); for E-R2, `parse_answer`
    returns balances;
  - the scorable and exact shares;
  - wall time.
- **Pass:** a truncated share ≤ 2%, i.e. at most 1 of 50. The other shares are descriptive and never gate.
- **Records.** Each SMOKE file's sha and counts are logged in DECISIONS with "outcomes known: TRAIN-prompt smoke only".
- **Reading a failure.** `max_new` exceeds every R3 training target by at least 16 tokens. A truncation on a TRAIN
  prompt therefore means that the adapter does not stop after its answer: it is a property of the adapter, not of
  `max_new`.
- **If any adapter fails** (the frozen rule: the run aborts before opening; there is no exemption route):
  - the battery is never opened, and nothing is re-minted or deleted;
  - no adapter is retrained, and `max_new` is not changed;
  - the remaining adapters are still trained and smoke-tested (step 5), so the record is complete;
  - E01R is **INCONCLUSIVE (smoke failure)**, and P3 stays unspent (§8).

**Crashes and retries.** A training, smoke or evaluation run that crashes may be resumed or re-run deterministically,
with at most **2 further attempts** per cell. A crash here means an exception, an OOM kill, a power or host failure, or
a forced stop.
- Each attempt is logged in the chain log and in DECISIONS, with "outcomes known: no", or "TRAIN-prompt smoke only" for
  smoke.
- **Attempts log.** `train`, `smoke` and `eval` also record every attempt in `ATTEMPTS_E01R.jsonl`, next to
  `SEALED_COMMIT.json`. The log is append-only and is never edited or deleted.
  - A `start` event is appended once the gates have passed (the lane is held by the chain, and RAM suffices for a
    model load). An `end` event with status ok is appended when the cell's outputs are written.
  - A `start` without an ok `end` is a failed attempt: an exception or a refusal after the start, a kill, an OOM or a
    power loss. DECISIONS gives each failed attempt's cause.
- **What does not count as an attempt:** a refused start (lane held by a peer, RAM below 12 GB), a refusal by a custody
  check that runs before the lane, and a ThermalGuard pause.
- **Training.**
  - `train` returns without training only if both the adapter and its `TRAIN_META.json` exist. An adapter directory
    without `TRAIN_META.json` is an incomplete run: `train` resumes it from its checkpoint. Without a checkpoint it
    refuses, and the partial directory is handled as an unreadable checkpoint (below).
  - It resumes from its own checkpoint under the same run fingerprint: the same seed, rows, row order, code, recipe and
    `RCX_THREADS`.
  - If the checkpoint is unreadable, the partial run directory is renamed `<tag>.crashed-<n>` and kept, and the cell
    is trained again from step 0 under the same fingerprint.
  - Bit-identity is not claimed for CPU training. The completed adapter's sha is what `open-sealed` fixes.
- **Smoke and evaluation** are re-run from scratch, with the same adapter, prompts or battery, `max_new` and batching.
  Greedy decoding makes them deterministic.
- **If the raw generations of an eval were stored** before a later step failed, the re-run post-processes them without
  decoding again.

**Missing cells.** A cell is "missing" only for one of these technical reasons, each logged in DECISIONS before the
analysis runs:
1. a training run that still fails after its 2 further attempts. Because `open-sealed` then refuses, E01R ends
   INCONCLUSIVE before opening;
2. a SEALED evaluation that still fails after its 2 further attempts;
3. an adapter whose file no longer matches the sha fixed at opening, and that cannot be restored byte-identically;
4. raw outputs that are lost or fail the analysis integrity checks (battery sha, adapter sha, `max_new`, bound code,
   re-score, truncation re-check), and cannot be regenerated by a deterministic re-run within the attempt limit.

No other reason is accepted. In particular, a cell's scores are never a reason (`eval` prints none). A missing
confirmatory cell makes its arm INCONCLUSIVE (missing). A missing control cell makes both arms INCONCLUSIVE
(missing). A missing OFF cell affects nothing.

**A cell is missing only after the stated further attempts.** `analyze_e01r.py` checks this for every cell named in
`--missing`. It refuses the name unless one of these holds:
- `ATTEMPTS_E01R.jsonl` shows at least 3 failed `eval` attempts on the cell: the first attempt and its 2 further
  attempts (reasons 2 and 4);
- the cell's adapter file no longer matches the sha fixed at opening (reason 3, which has no attempt limit);
- the cell's eval record exists but fails an integrity check (reason 4). Eval files are create-only and never
  overwritten, so such a cell cannot be re-run under its id. The record itself proves the completed attempt, so
  this holds even if the "ok" event after it was lost in a crash.

A torn line in `ATTEMPTS_E01R.jsonl` (a crash or power loss mid-write) is reported and ignored, and later events are
appended on a new line. A torn line can only hide an attempt, never add one, so it cannot help a cell reach the
attempt limit. A well-formed line that is not an attempt event refuses.

**Compute (CPU-lane hours; re-estimated for `max_new` 160; review item 18).**
- **Basis.** The E01N SEALED evals had 2,240 records each at `max_new` 64.
  - Each cell took 2,139–2,648 s (36–44 min).
  - ThermalGuard pauses account for 0–520 s of that, so the unpaused time was 1,990–2,150 s (0.89–0.96 s per record).
  - The outputs were native JSON: mean 29 bytes, max 39 bytes, in every cell including OFF.
- **Mechanism.**
  - `lora_cpu.generate` sizes each batch as 3000 // (prompt tokens + `max_new`).
  - With prompts of about 125–150 tokens, a batch holds about 15 rows at `max_new` 64 and about 10 at 160, giving about
    1.5× as many batches.
  - A batch decodes until its longest output reaches EOS. For a non-trace arm, `max_new` 160 therefore changes the
    number of batches, not the tokens generated.
- **Assumptions.** None of these was measured; no tokenizer was loaded.
  - Prefill cost scales with prompt tokens and does not change.
  - Decoding is 30–70% of the unpaused time at `max_new` 64.
  - A decode step at batch 10 costs between 2/3 of a batch-15 step (compute-bound) and the same (memory-bound).
  - R3 outputs are about twice as long as native ones, so R3 batches decode for about twice as many steps. Line 1
    adds 57–68 bytes, assumed to be about 18–25 tokens, to the native answer's 20 or so.
  - Guard pauses add 0–25%.
- **Per 1,760-record cell.** At `max_new` 64 this is 26–28 min unpaused.
  - At `max_new` 160, a non-trace cell takes ×1.0–1.35: 26–38 min unpaused, or 26–47 min with pauses.
  - An R3 cell takes ×1.3–2.4: 34–68 min unpaused, or 34–85 min with pauses.
- **Totals (B4):**
  - **Training: 11–18 h.** That is three E-R2 adapters at 1.7–2.6 h each (N-CFSD took 6,005–9,302 s) and three E-R3
    adapters at 2.0–3.4 h each (20–30% longer; §3).
  - **Smoke tests:** about 0.3–0.5 h for six adapters (50 prompts each, plus model loads).
  - **SEALED evals: 4.7–9.7 h, central about 7 h.** These are 10 cells: seven non-trace (N-CFSD s0–s2, E-R2 s0–s2 and
    OFF) and three E-R3.
  - **Total:** about 16–28 h, central about 21 h.
- **Refinement.** The build records the R3 gold length in tokens, and the smoke tests measure wall time at the committed
  `max_new` before opening. Both are descriptive; nothing gates on them.

**Constraints:**
- **Single CPU lane.** One model job at a time, under the CX D-36 lock (`rcx/lane.py`). `train`, `smoke` and `eval`
  refuse unless `rcx` holds it. The chain scripts log `START <tag>` / `END <tag>` lines that the lane check reads.
- **Threads.** `RCX_THREADS` ≤ 4 for lane jobs and 1 for anything else. `CUDA_VISIBLE_DEVICES` is empty; the GPU is
  never used.
- **Subagents.** No CPU-heavy work while a lane job runs; only checks that finish in under 5 s.
- **RAM.** Wait for ≥ 12.5 GB of free RAM before each model load; `need_ram` refuses below 12 GB.
- **Guard.** `rcx.guard.ThermalGuard` runs between micro-batches.
- **Control adapters** are read-only, from `runs/RCX-E01N/cells`. E01R writes only under `runs/RCX-E01R/`.

**No retries under the same id beyond the crash rule:**
- A completed adapter is never retrained, and an eval or smoke file is never overwritten.
- The mint and open records are write-once.
- The analysis runs once. Its start marker and outputs are create-only.
- `ATTEMPTS_E01R.jsonl` is append-only.
- A cell that cannot be completed is reported as missing and is never replaced by another seed or id.
- There is no descriptive opening of the battery outside this sequence.

## 11. Freeze and deviations

**Two entries, in order.** The header says this file becomes FROZEN when its final sha256 is "recorded in DECISIONS
… together with the SEALED commitment". This is read as two CX D-xx entries, linked by the code sha (§4):
1. **The freeze entry** (first). It records:
   - this file's sha256 and timestamp, with a check that everything before §4 is byte-identical to commit `c4cba9e`;
   - the sha256 of every bound code file (§4), the code sha, the sha256 of `rcx/lane.py` (for reference; not bound)
     and the sha256 of every E01R test file;
   - the E-R2 and E-R3 rows shas, `TRAIN_MANIFEST_E01R.json` (with `max_new` and the R3 gold length in tokens) and
     `AUDIT_E01R.json`, with the §3 covariates and the per-seed token counts;
   - the unit-test result, from a run made only when the lane is free;
   - the §9 seed-history result;
   - the CX D-38 margins, by reference (no guard-power computation is run).
2. **The mint entry** (second). The mint runs with `--expect-code-sha256` equal to the freeze entry's code sha, and
   refuses otherwise. The entry records sha256(salt), the `SEALED_COMMIT.json` sha256, the code sha and any
   `--drop-arm` with its reason.

No bound file changes between the two entries. A change found by the mint's code-sha check means a new freeze entry
before any mint.

Every deviation is logged with "outcomes known: yes/no". Nothing is rerun under this ID, except as the §10 crash rule
allows.

**Custody of the revisions.**
- **First revision.** §4–§11 and the open items were revised on 2026-10-12 while step A's first cell ran.
  `BRANCH_STEP_A.json` did not exist, no step A output was read, and §2 and the R1/R2/R3 specifications were left
  byte-identical to CX D-37.
- **This revision** was made on 2026-10-12, after step A (R-10, CX D-39). Outcomes known: step A DEV2 only, i.e. the
  branch and R-10's P2 rates. No treatment adapter exists.
  - It integrates CX D-38, which was fixed before any step A END line. No threshold, margin or guard was changed after
    step A.
  - Step A enters only through CX D-39's branch (the arms) and through the C1 baseline in §7's power notes.
  - The header and §1–§3 are byte-identical to `c4cba9e`.
- **Consistency revision** (2026-10-12, after commit `5b36b9c`). Outcomes known: step A DEV2 only, as above. No
  treatment rows, manifest, audit, commitment or adapter exists.
  - It aligns §4–§11 with the code fixes of the consistency review:
    - `smoke`, `open-sealed` and `eval` refuse on bound-code drift, and `rcx/lane.py` is unbound;
    - the opening's refusal checks all precede the P3 read;
    - the attempts log and the attempt-based missing-cell check;
    - the run-once analysis (fixed paths, start marker), and the truncation re-check;
    - `--expect-code-sha256`, `--drop-arm` tied to the audit, and the mint's token-count check;
    - `eval` prints no outcome.
  - It adds the §5.4 tests note and the INFO caveat.
  - No endpoint, estimator, threshold, margin, guard, decision rule or seed changed. The header and §1–§3 are
    byte-identical to `c4cba9e` and `5b36b9c`.

**Any of the following voids the confirmation:**
- re-creating or deleting `SEALED_COMMIT.json`, `SEALED_SALT.secret` or `SEALED_OPEN.json`;
- retraining a completed adapter at any time, or training any adapter after opening;
- reading the salt or P3 text before opening, or reading P3 text other than in memory through `open-sealed`, `eval` and
  `analyze_e01r.py`;
- opening while an `arms_required` adapter is incomplete or lacks a passing smoke test (there is no exemption);
- adding, dropping, replacing or re-specifying an arm after the mint;
- evaluating any arm at a `max_new` other than the committed one;
- evaluating a treatment adapter on any battery before the SEALED analysis (the TRAIN-prompt smoke test excepted);
- running the analysis more than once (deleting its start marker included), or declaring a cell missing for a reason
  not listed in §10;
- editing or deleting entries of `ATTEMPTS_E01R.jsonl`;
- changing §2 after any step A output exists.

## Open items for review (draft only; to be resolved before the full freeze)

**Resolved in this revision (2026-10-12, after step A; see §11 custody):**
- **CX D-38 integrated:**
  - the margins G1–G6, and C1 +0.25 under every branch;
  - six fixed guards;
  - the degenerate trade-off on a G4–G6 point estimate ≤ −0.15;
  - REVISE requires d(C1) > 0.

  The PENDING STEP A guard-power computation and the proposed margin rule are removed. §7 states what the D-38
  constants imply (former items 1 and 8).
- **Branch B4 (CX D-39):**
  - the arms are E-R2 and E-R3;
  - R1, TR_v1, `audit_tr_pool.py` and §2.1 are not applicable;
  - P3 is not read before opening;
  - N-VR s0 is not evaluated (§4).
- **Smoke.** No exemption route: any smoke failure means no opening, E01R INCONCLUSIVE, and P3 unspent. All six
  adapters are trained regardless (§10). This resolves the second half of former item 3.
- **Mint, post-mint rules and custody:**
  - the mint binds the arms to the branch, with `--drop-arm` only before the mint (§4);
  - the post-mint crash, attempt and missing-cell rules (§10);
  - the bound-code list, verified by `train`, `smoke`, `open-sealed`, `eval` and the analysis (§4).
- **Specification:**
  - §5–§7 are now a complete specification of record selection, statistics and the ordered decision rule;
  - the non-anchor sub-quantities (§5.4) and the declared asymmetries (§5.5) are added.
- **Header and §1–§3** are left byte-identical. Their stale statements are read as §4 says (former item 7).

**Resolved in the consistency revision (2026-10-12; §11 custody):**
- **Code.** `analyze_e01r.py` with `tests/test_analyze_e01r.py`, and the `native_e01r.py` and `audit_e01r.py`
  changes, implement this text (former remaining item 1):
  - bound-code drift refused by every command, and `rcx/lane.py` not bound;
  - the opening's refusal checks before the P3 read;
  - the attempts log, the missing-cell attempt check, and the run-once analysis;
  - the truncation re-check, with unknown truncation flagged;
  - `--expect-code-sha256`, `--drop-arm` only for an arm that fails the audit, and the token-count check;
  - `eval` refuses N-VR and prints no outcome;
  - the R2 duplicate-`comp_listed` count, computed by the analysis (§4, §5.5).
- **Freeze and mint** are two DECISIONS entries, linked by the code sha (§11).
- **INFO caveat** on partial polarity-blind policies (§5.4).

**Remaining** (the only open items; nothing else is pending before the full freeze):
1. **Build** (with the tokenizer only, §10 step 2):
   - the E-R2 and E-R3 rows;
   - `TRAIN_MANIFEST_E01R.json`, with `max_new` and the R3 gold length in tokens;
   - `AUDIT_E01R.json`, with the §3 covariates and the per-seed token counts (`tokens.per_seed`) that the mint
     requires.
2. **Seeds.** The git-history search for 20261264, 20261265 and 20261268, one seed per command, with the lane free
   (§9).
3. **Tests.** Every E01R test file is run once, in full, with the lane free. The result goes into the freeze entry
   (§11).
