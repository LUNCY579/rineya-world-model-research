# PREREG_ITEM7_SEEDS, Addendum A (DRAFT): report-only metrics, and the registration of the B4 futility screen

| | |
|---|---|
| Status | **DRAFT**, written 2026-10-11 ~20:50 CDT; revised 2026-10-12 ~02:30 CDT for RCX's co-sign review (`COSIGN_REVIEW_B4_SCREEN.md`, commit 2657079, `9dd352ee…`; 13 required edits). BWM accepted all 13 as D-80 (DECISIONS.md; also recorded in RCX's `complementary/COLLABORATION.md`, commits f3031f6 and af969cf). Revised again 2026-10-12 ~06:50 CDT for the final hash round (D-80 follow-up, D-81): the screen runner's canary sha set and RCX's low-severity items applied. Not registered. Becomes `research/PREREG_ITEM7_ADDENDUM_A.md` when its sha256 is recorded in DECISIONS. §B also needs RCX's co-signature (D-76a), which binds to the final shas in this header (§E). |
| Screen prereg (final) sha256 | recorded at finalisation: the sha256 of `research/PREREG_B4_SCREEN.md`, which is finalised first, written as the full sha256 in backticks (the screen's freeze refuses unless this row names the sha256 of its `--prereg` file) |
| Deadline | Registered before `scripts/bw_seeds.py freeze` writes any `runs/FREEZE_I7_*` (panel `top_now_cpu` item 6). |
| State at writing | Re-checked 2026-10-12 01:41 CDT. `runs/FREEZE_I7_*`: none. `runs/BW_SEEDS/`: absent (no item-7 evaluation output exists). `BW_Fdelta_s1`, `BW_Fdelta_s2`: trained. `BW_Pa_v2_s2`: step 300/1000, cut by the 21:54 reboot, resumes from its checkpoint. `BW_Pa_v2_s3`: not trained. |
| Base prereg | `PREREG_ITEM7_SEEDS.md`, sha256 `956a0dc4117c948065f40258be3a7a3e418f51c2b199c4bbb8e7f1337d88985d`. **It stays byte-identical.** |
| Addendum code | `scripts/item7_addendum_a.py` sha256 `2a78ba81d6d905ad4cd1feda29057b3a0d43fbca7a9b281c14c2789f8ec8f4e7`; tests `tests/test_item7_addendum_a.py` (9 pass) sha256 `1fcc461f039fb74ccb2d8ab144518bbb493833268a7d2d1ebc91b75af2669479` |
| Screen runner | `scripts/b4_screen.py` sha256 `0b576b1f4136f7ced9b12bdc71219a6435cdd1733bd733d1908d241469df750f`; tests `tests/test_b4_screen.py` sha256 `8d2e2237a7de68cce2aed4c4909ed2452e578315bf09fe1a8944846a857824fc`. These are the same rows as the screen prereg's header, and they replace every earlier runner and test sha. The screen's freeze refuses unless this text names this runner sha. |
| Decision source | D-75 (user, ~16:10: decision 1, GPU order with the B4 screen first after item 7); D-76a (RCX agreement on the screen); D-77; D-80 and its follow-up; D-81; F-09; `research/NEXT_ANALYSIS_PANEL_2026-10-11.json` `final.top_now_cpu` items 2, 3 and 6. |

## 0. What this addendum does and does not change

**Adds:**
- §A: report-only metrics for item 7's existing runs;
- §B: the registration of a separate one-arm B4 screen that reuses R0's S-beam run as its control.

**Does not change anything registered in `PREREG_ITEM7_SEEDS.md`:**
- the systems R0/R1/R2, the S-beam and S-direct configurations, the fresh set and the training recipes;
- the two registered tests ("the world-model effect replicates") and the release-gate reading;
- the report-only list and the "Not claimed" section.

`scripts/bw_seeds.py` (`b783a191…`) and its `analyze` stage are unchanged. No metric below is a test, none has a reading
that can change item 7's outcome, and none adds a GPU run to item 7. The screen's GPU run is a separate job after item
7's analysis (D-75.1). **Item 7 runs and is analysed exactly as registered, whatever the screen does.**

**Pins at writing** (rechecked when the addendum is registered):

| File | sha256 |
|---|---|
| `PREREG_ITEM7_SEEDS.md` | `956a0dc4117c948065f40258be3a7a3e418f51c2b199c4bbb8e7f1337d88985d` |
| `scripts/bw_seeds.py` | `b783a1916bce655cf93e3d55bedcb276dae3dab2bbdc214e00ff633c828f3b08` |
| `scripts/blocks_eval.py` | `b3666424151c78368500535d2571f0eb5b9e546b11764696d99f427a7794acb6` |
| `scripts/bw_run_tests.py` (merge) | `de7022d1eacebab271f7624e962d40d58aeee435494da079a622a03f80a34542` |
| `bwmn/lm.py` | `88a1e3a26b518cebd17e0b8197d1dd10872d549a4a27a69744ed59266895c0ac` |
| `bwmn/blocks_planner.py` | `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960` |
| `bwmn/blocks.py` | `b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83` |
| `runs/FREEZE_B2_beam.json` / `runs/FREEZE_B2_direct.json` | `412a6ee5…` / `167276d3…` |
| `runs/BW_val_large/instances_seeds_fresh.json` | `6fccb94547184d8ba44a2fdc19ad6c3f1fca584c67c906964281746c0541b3fc` |
| `research/PREREG_FREPLAY.md` (definitions used in §A.1) | `994d74c0b721724a49cd24774b5ecf0fc12027d058718575f276b100b3e975f8` |
| `scripts/freplay.py` (P-15's registered script; imported read-only, pinned by the addendum code) | `866ca5b696b465d0de77d0882b38323df50f1bb3bcc216ff68aaadbb0beb0a3b` |
| `bwmn/flops.py` (sha recorded at analysis) | `bbada67b554668d5351e2670d744043dd0c5661ce95dd1ac41b8229cbcee1625` at writing |
| Titan-Sparse `tokenizer.json` (§A.4, tokenizer only) | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `scripts/item7_addendum_a.py`, `scripts/b4_screen.py` and their tests | header rows above |

## A. Report-only metrics for item 7

All of §A applies to R0, R1 and R2. Beam-only metrics (A.1, A.3 and the beam half of A.2) apply to `{R}_beam`;
S-direct has no F. All of §A is implemented by `scripts/item7_addendum_a.py`: CPU only, no model, no torch.

### A.0 Inputs are hashed before they are read

**What it writes.** Before any §A analysis reads an item-7 output, a hashing-only step
(`python -B scripts/item7_addendum_a.py hash`) writes `runs/BW_SEEDS/ADDENDUM_A_INPUTS.json`. It holds the sha256, line
count and size of:
- every `plans.jsonl`, `summary.json` and `fcache.jsonl` under `runs/BW_SEEDS/{R0,R1,R2}_{beam,direct}/`, merged and
  per shard (every expected file must exist);
- the six `runs/FREEZE_I7_*.json`.

**Why.** This file is the sha table that `PREREG_FREPLAY.md` §9 requires for new sets ("an amendment that names its
files and shas before any of their outputs are read"). The files are named here, and that step fixes their shas before
any parse. `bw_seeds.py analyze` (the registered item-7 tests) is not affected by this rule.

**Rules of the step:**
- It is written once.
- It needs `runs/BW_SEEDS/.run_done`, and it parses nothing.
- **It refuses unless the B4 screen is frozen** (`runs/FREEZE_B4S_a_beam.json`). Its output closes the screen's
  late-freeze window (§B.3).
- With `--without-screen-freeze`, it records that the screen was never frozen, and the screen is then void.
- The analysis re-hashes every file of the table before parsing and refuses on any difference or on any new item-7
  output file.

### A.1 Per-system F audit (FREPLAY metrics)

**Sets.** `i7_R0_beam`, `i7_R1_beam`, `i7_R2_beam`, and, for the screen, `b4s_a_beam` (§B.6).

| Set | F adapter (sha from `FREEZE_I7_{R}_beam.json` `fwd_sha256`) | Caches | Outcomes | Initial states |
|---|---|---|---|---|
| `i7_R0_beam` | `runs/BW_Fdelta_s0/adapter` (`2ed0bb00…`) | `runs/BW_SEEDS/R0_beam/shard_{0..3}of4/fcache.jsonl` | `runs/BW_SEEDS/R0_beam/plans.jsonl` (+ shard files) | seeds_fresh json (`6fccb945…`), ids `seeds_fresh/<index>` |
| `i7_R1_beam` | `runs/BW_Fdelta_s1/adapter` | `runs/BW_SEEDS/R1_beam/shard_{0..3}of4/fcache.jsonl` | `runs/BW_SEEDS/R1_beam/plans.jsonl` | same |
| `i7_R2_beam` | `runs/BW_Fdelta_s2/adapter` | `runs/BW_SEEDS/R2_beam/shard_{0..3}of4/fcache.jsonl` | `runs/BW_SEEDS/R2_beam/plans.jsonl` | same |

**Definitions:** `PREREG_FREPLAY.md` §§2-7, unchanged and by reference (sha `994d74c0…`). They cover:
- the unit: a unique (state, action) key within a set; the last line wins within a file; the cross-file conflict rule;
- the classes and subtypes, the consistency assertions and the second code path;
- the strata (block count × action type) and the rates with exact Clopper-Pearson 95% CIs;
- the per-instance forest, the coverage check, and the flagged and support-exposed instances;
- the two registered readings.

To avoid the name clash with item 7's systems, FREPLAY's readings are called:
- **FREPLAY-R1:** oracle-equivalence bounds [L, U] on exact-F success;
- **FREPLAY-R2:** the support share, false_legal_rejected_by_support_check / illegal keys.

**Wording consequence (FREPLAY §7), applied to item 7's own statements.** If FREPLAY-R2 > 5% in a system's set, any
statement that the system's verified plans were valid says "verified by the learned F plus a domain-agnostic
delta-consistency check (a predicted deletion must be a fact of the state)".

**Additional per-system report** (descriptive; no reading):
- F-flag accuracy on schema-valid keys;
- the delta-exact rate;
- harmful keys by class;
- the spread of these rates across the three F seeds (min and max).

No equality test is made between seeds; a difference is reported, not interpreted.

**Implementation.**
- `scripts/item7_addendum_a.py analyze` imports `scripts/freplay.py` **read-only** (sha-pinned `866ca5b6…`;
  `freplay.py` is not edited).
- It calls FREPLAY's own functions: `read_cache`, `build_units`, `strata`, `forest`, `coverage`, `reading_r1`,
  `reading_r2` and the classifier with its assertions.
- Only the instance loader is new: seeds_fresh, with the instance-to-shard map read from the shard `plans.jsonl` files.
- The sha of `freplay.py` is recorded with the result.

### A.2 Failure taxonomy

**S-beam, per system** (from `plans.jsonl`; success = the VAL verdict):

| Class | Definition |
|---|---|
| solved | VAL success, verified or not; unverified successes are listed |
| verified_but_invalid | `verified` true and VAL false (FREPLAY-R1 predicts 0 when no instance is flagged) |
| dead_end | `fail == "dead_end"` (the beam emptied) |
| budget | `fail == "budget"` (E24 reached) |

For every unverified failure, the report gives:
- the number of unmet goal facts in the executor's state after the submitted plan, as a distribution. The submitted
  plan is the best node's prefix: `plan_beam` submits the node with the fewest unmet facts, then the shortest prefix;
- the **stalled-at-1** count: exactly one unmet goal fact, the F-09 "deceptive minima" pattern (on B2-hard, 28 of 37
  failures);
- a prefix that is invalid under the executor, counted separately.

**S-direct, per system:**

| Class | Definition |
|---|---|
| solved | VAL success |
| unparseable | `fail == "unparseable"` (candidate 0 had no parseable plan) |
| invalid_action | parsed, executor `valid` false |
| goal_not_reached | parsed, every action applicable, goal not met |

A cross-cutting flag is added: **at_cap**, `gen_tokens ≥` the frozen S-direct M (256, read from
`FREEZE_I7_{R}_direct.json`). Such outputs may be truncated (P-13).

**Cross-tabs (descriptive):**
- R1 and R2 vs R0 by class, per configuration;
- per system, beam class × direct outcome.

VAL-executor disagreements are counted.

### A.3 Round-capped anytime curves (S-beam)

**Exact round curve.**
- S_r = the number of instances with VAL success and `rounds ≤ r`, for r = 1 up to max(12, the largest observed
  `rounds`).
- By D-48c/D-50 (per-instance CRN seeds keyed by round label, batch-independent sampling, frozen memoised F), the first r
  rounds of `plan_beam` do not depend on any later cap. So S_r equals the success of the same system capped at r
  rounds, with the E24 expansion cap unchanged.
- Reported per system, with R1 − R0 and R2 − R0.

**Approximate token curve.** The number of successes with `gen_tokens ≤ T`, for T = 2k, 4k, 6k, 8k, 10k, 12k and 16k
proposer tokens. It is approximate because a token cap would cut inside a round.

### A.4 FLOPs and token ledger (estimates, labelled as such)

**Logged, exact:**
- per instance: proposer `gen_tokens` and `rounds`;
- per run: `f_model_queries` (cold cache per shard, so a key queried in two shards counts twice), the expansions total
  (Σ over shards of `expansions_per_instance` × n; `blocks_eval.py` keeps no per-instance expansions) and `wall_s`
  (planning only).

**Derived exactly from the caches** (given the tokenizer):
- per shard, every unique (state, action) key, which that shard computed once (a torn last line is skipped);
- the union over shards (FR-02's unique count), reported beside;
- per key, the F prompt length (`B.forward_prompt_delta(s, a)`) and the generated length (the cache text + EOS, capped
  at LMForwardBW's 40-token limit).

Tokens are counted with the Titan-Sparse `tokenizer.json` (sha-pinned) through the `tokenizers` library only: no
weights and no transformers, sequential, on CPU.

**FLOPs** use `bwmn/flops.py` (sha recorded at analysis), one formula for every term:
- **F:** Σ over shards and their unique keys of `flops_breakdown(fp_F, prompt_k, gen_k)`, with
  `fp_F = flop_params(Titan-Sparse config, extra_nonemb = the F adapter's LoRA parameters)`, which are read from the
  adapter's safetensors header.
- **S-beam proposer (ESTIMATE):**
  - Per instance, x_i = min(E + W − 1, 1 + W × (rounds − 1)) = min(27, 1 + 4 × (rounds − 1)) bounds the expansions:
    a round adds at most W, and `plan_beam` stops at the first round end with at least E (a budget failure can
    exceed E = 24 by up to W − 1).
  - The x_i are scaled by one factor so that their sum equals the logged run total.
  - Each expansion costs `shared_prefix_breakdown(fp_Pa, P, [g], sample_pairs = beam_proposer_pairs_bound(g, 80))`.
    P = the tokenised proposer prompt of the instance's initial state (imagined states have the same blocks, and their
    prompts differ by a few tokens). g = the instance's `gen_tokens` spread evenly over its scaled expansions.
- **S-direct:** `flops_breakdown(fp_Pa, P_init, gen_tokens)`.

**Reported:** FLOPs and tokens per instance and per solved instance, per system and configuration, and the beam/direct
ratio. No iso-FLOP claim is made from these numbers, and none is compared with API models.

**Output:** `runs/BW_SEEDS/ADDENDUM_A_REPORT.{json,md}`, written once.

### A.5 Operational note (not a registration)

bundleI's `i7-run` has `est_s` 600. But R0/R1/R2 evaluation needs about 9-10.5 GPU-h: 3 × about 3.3-3.5 h of S-beam,
plus three short S-direct runs. The job resumes per shard, so it works; plan for 2-3 BWM turns.

## B. Registration of the B4 futility screen (D-75.1, D-76a)

### B.1 What

- One arm, **a** = RCX's hybrid hook `LearnedUnmet.from_file(gnn_s20261276.pt, invalid_policy="fallback",
  hybrid_lambda=1.0)` as `unmet=` in the frozen B2 `plan_beam`, run on seeds_fresh-40 with exactly R0's S-beam
  arguments.
- The full design, pins, rule and outputs are in `research/PREREG_B4_SCREEN.md` (drafted as
  `PREREG_B4_SCREEN_DRAFT.md`). It is finalised first and its sha256 goes into this header.

### B.2 The control is R0's S-beam run, as item 7 produces it

- `runs/BW_SEEDS/R0_beam/` is used unchanged and is not re-run.
- `bw_seeds.py freeze` writes R0's beam args as `dict(FREEZE_B2_beam args, pa=runs/BW_Pa_v2_s1/adapter)`, with
  `fwd_adapter = runs/BW_Fdelta_s0/adapter`: exactly the B2 beam configuration.
- R0's beam run is therefore B4's arm c (the default key) on these 40 instances, with the same F (`2ed0bb00…`), Pa
  (`ee1c5aec…`), eval_seed 0, CRN seed function and shard partition (index % 4).
- The screen runner refuses to freeze or run unless:
  - the R0 record's args equal FREEZE_B2_beam's and the adapter shas match;
  - each R0 shard's `summary.json` records exactly the frozen arguments on `seeds_fresh`, shard `i/4` (co-sign edit 7).
- The R0 files the screen reads must equal §A.0's sha table; otherwise the screen is VOID (co-sign edit 9).
- **Pairing check:** an instance solved in round 1 by either arm must be identical in both arms (plan and proposer
  tokens). If this check or the R0 shard-argument check above fails, there is an INSTRUMENT FAULT: no decision, and c
  and a are re-run by the screen runner in one GPU turn (screen prereg §3).

### B.3 Timing: the screen is fixed before any R0 output exists (co-sign edit 9)

- **Normal path.** `scripts/b4_screen.py freeze` writes `runs/FREEZE_B4S_a_beam.json` immediately after
  `bw_seeds.py freeze` and before `bw_seeds.py run`. It records:
  - R0's freeze sha, the args, the hook spec and every pin;
  - the snapshot of RCX's files and the library and driver versions;
  - the rule, the pre-check result, and the shas of this addendum, the screen prereg, the runner and its tests.
- **The guard.** The freeze refuses if any item-7 output exists. That means any file under `runs/BW_SEEDS/` (R0_direct
  reveals per-instance outcomes, and a running shard's `progress.txt` shows per-round solve counts), or bundleI's log
  `runs/BW_SEEDS.stdout` once it shows that a run has started.
- **Late freeze is mandatory, not optional.** If `b4s-freeze` fails because an item-7 output exists, the late freeze
  (`--allow-after-r0`) is mandatory, and it records the files present. There is no "or dropped".
- **The window closes.** Once `runs/BW_SEEDS/SEEDS_REPORT.json` or `runs/BW_SEEDS/ADDENDUM_A_INPUTS.json` exists, or
  bundleI's log `runs/BW_SEEDS.stdout` contains item 7's `release_gate_reading`, the freeze refuses outright, and the
  screen is void. A.0 therefore refuses unless the screen is frozen (§A.0).
- **The screen runs whatever item 7's outcome is.**
- **Nothing can be chosen after R0's outputs are known, even with a late freeze.** Everything about the arm is fixed by
  the screen prereg: module, call, args, instances, drop rule, decision rule and VOID rule.
- **Queue (the lead's change, D-81; not made by this draft).**
  - bundleI runs `i7-freeze` → `b4s-freeze` → `i7-run` → `i7-a0-hash` (§A.0) → `i7-analyze` →
    `i7-addendum-analyze`.
  - A failing or refusing `b4s-freeze` stops the bundle before any R0 output exists, so the queue cannot void the
    screen.
  - Should an item-7 output nevertheless exist before the screen is frozen, the late freeze must come before
    `i7-analyze` and A.0.

### B.4 The decision rule (verbatim from the screen prereg §5.2)

> GO iff g − l ≥ 2, where g = a-only and l = R0-only VAL successes on the paired instances; otherwise NO-GO (net ≤ 1
> is always NO-GO). The one-sided sign p and the two-sided McNemar p are reported only and are not part of the rule
> (D-77). If more than 5% of hook calls return a value in the penalty band (invalid state or critic error;
> value ≥ 1,000), a GO is not credited to the learned critic alone and c+pen is mandatory in the full B4 (D-76a). The
> screen is VOID on any hook model, encode, nonfinite or failsafe error, model_evals = 0 (in any hook shard), a
> failed canary, or R0 outputs that differ from §A.0's table; it is an INSTRUMENT FAULT if the round-1 identity check
> or the R0 shard-argument check fails, and INSTRUMENT_FAULT_RERUN (terminal, no reading) if the pre-registered re-run
> fails the identity check again. Precedence: VOID > INSTRUMENT FAULT > GO/NO-GO.

The net ≥ 3 with p ≤ 0.20 rule of the earlier draft is superseded (D-77) and is not part of the frozen rule. The null
false-GO rate of the registered rule is about 0.30-0.36 at plausible discordance and at most 0.44 at any discordance
(screen prereg §5.3).

### B.5 What item 7 gains and loses

- **Item 7's tests:** nothing changes. R0's outputs get one more registered reader.
- **seeds_fresh exposure:** after the screen, the set has been used by four S-beam systems (R0, R1, R2, a) and three
  S-direct systems, plus c and a again after an instrument fault. The screen's result selects nothing in item 7.

### B.6 Report-only additions tied to the control (registered here because they read R0's outputs)

- **c ≡ c+pen certificate on R0.**
  - Every F-accepted prediction in R0's beam caches is checked with the hook's `is_consistent`, with the parent
    state's blocks as `objs`. So a vanished goal block, which the hook penalises, is caught although the cache does not
    record the goal.
  - If none is inconsistent, c+pen would have made exactly R0's choices, and R0 serves as the c+pen control.
    Otherwise the screen reports R0 as c only.
- **R0 failures stalled at 1 unmet goal fact** (D-76a), and how many of them arm a solves (A.2's definition, on the
  true state). Also reported, symmetrically: a's solve rate on R0's other failures, and the mirror quantity (a's
  failures and R0's solves among them).
- **F cache cross-check, R0 vs a:** keys answered in both runs must have identical F text (a check of D-48c/D-50).
- **The arm's F audit:**
  - set `b4s_a_beam`: caches `runs/BW_B4_SCREEN/a_beam/shard_{0..3}of4/fcache.jsonl`, F `BW_Fdelta_s0`, under A.1's
    definitions;
  - run with `scripts/item7_addendum_a.py faudit-screen`, on files whose shas the screen's analysis recorded in
    `runs/BW_B4_SCREEN/ANALYSIS_INPUTS.json` before parsing them;
  - output `runs/BW_B4_SCREEN/FAUDIT_b4s_a_beam.json`, written once.

### B.7 Independence of B4, and what the co-signature covers (co-sign edit 5)

- B4's 110 instances (seed 20261410) exclude seeds_fresh's signatures, and the generator checks this at generation.
- The screen's 40 are never pooled into B4.
- B4's n, α, Holm family, arms and endpoints are not changed on screen data, and the screen's net and p are not
  quoted as effect estimates.
- **Co-signing the screen does not co-sign B4.** Route (D-80): B4's prereg declares the screen's outcome as known at its
  registration and forbids any change to arm (a): its module, call and pins (screen prereg §8.1).

### B.8 Outcomes known at registration (co-sign edit 12)

As listed in the screen prereg §0.
- **Known:** the B2 hard and mirror results (round-1 solves 0/110 and 1/110), H2, F-09 (28/37 failures stall at 1
  unmet fact), FREPLAY, RCX R-04, E07b VAL-B/TEST-B (VAL-B selected A3 with λ = 1), E11, the screen's stub dry runs
  and the D-77 operating-characteristics simulation.
- **Not known:** any item-7 output or any seeds_fresh outcome.
- **State at the 21:54 reboot:** `BW_Pa_v2_s2` was at step 300/1000 (checkpointed), and `BW_Pa_v2_s3` was not trained.

## C. Order of operations

1. **Registration.**
   - Finalise `PREREG_B4_SCREEN.md`, then this addendum (its header takes the screen prereg's final sha).
   - Record both shas in DECISIONS.
   - RCX co-signs §B and the screen prereg on the final shas (§E). Before that: RCX sent the canary (`9e9ad339…`,
     commit 42cbfcc), BWM wrote its sha into the runner, BWM's CPU pre-flight reproduced every probe exactly (screen
     prereg §10), and the runner/test shas in both headers were updated (D-80 follow-up).
2. **bundleI (D-81):** train `BW_Pa_v2_s2` and `BW_Pa_v2_s3` → `bw_seeds.py freeze` → `b4_screen.py freeze` →
   `bw_seeds.py run` → the A.0 hashing step (`item7_addendum_a.py hash`) → `bw_seeds.py analyze` (the registered
   item-7 tests). A failing `b4_screen.py freeze` stops the bundle before `bw_seeds.py run`.
3. **CPU, in the same bundle after item 7's analysis:** A.1-A.4 (`item7_addendum_a.py analyze`).
4. **The next BWM GPU turn:** `b4_screen.py run` (about 3.5 GPU-h), then `b4_screen.py analyze` (CPU), then
   `item7_addendum_a.py faudit-screen` (CPU). Then B6 (D-75.1).

## D. Not claimed; limits

- Nothing in §A is a test. Seed spread in F behaviour is described, not tested. A non-significant difference is not
  read as equivalence, as in the base prereg.
- §A.4 is an estimate where the logs lack per-expansion detail, and it is labelled as one. No iso-FLOP or "cheaper
  than model X" claim follows from it.
- §B registers a futility screen, not a B4 claim. Its limits are listed in the screen prereg §9.

## E. Finalisation and co-signature (co-sign edit 11)

- **What the co-signature binds.** RCX's co-signature binds to the final shas: this text's, the screen prereg's, and
  the runner and test shas in the headers.
- **What finalisation may change.** Finalisation drops "DRAFT" from the title and changes only the Status row and the
  "Screen prereg (final) sha256" row of this header. `scripts/b4_screen.py final-diff --draft D --final F` refuses any
  other change, and RCX re-verifies that diff.
- **The binding is checked.** The screen's freeze refuses unless the "Screen prereg (final) sha256" row names, as a
  full sha256 in backticks, the sha256 of the screen prereg it is given (`--prereg`).
