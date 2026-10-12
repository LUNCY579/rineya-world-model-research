# PREREG_ITEM7_SEEDS, Addendum A (DRAFT): report-only metrics, and the registration of the B4 futility screen

| | |
|---|---|
| Status | **DRAFT**, written 2026-10-11 ~20:50 CDT. Not registered. Becomes `research/PREREG_ITEM7_ADDENDUM_A.md` when its sha256 is recorded in DECISIONS; §B also needs RCX's co-signature (D-76a). |
| Deadline | Registered before `scripts/bw_seeds.py freeze` writes any `runs/FREEZE_I7_*` (panel `top_now_cpu` item 6: by the end of BWM turn 3, when `BW_Pa_v2_s3` finishes). |
| State at writing | `runs/FREEZE_I7_*`: none. `runs/BW_SEEDS/`: absent (no item-7 evaluation output exists). `BW_Fdelta_s1`, `BW_Fdelta_s2`: trained. `BW_Pa_v2_s2`, `BW_Pa_v2_s3`: not trained. |
| Base prereg | `PREREG_ITEM7_SEEDS.md`, sha256 `956a0dc4117c948065f40258be3a7a3e418f51c2b199c4bbb8e7f1337d88985d`. **It stays byte-identical.** |
| Decision source | D-75 (user, ~16:10: decision 1, GPU order with the B4 screen first after item 7); D-76a (RCX agreement on the screen); F-09; `research/NEXT_ANALYSIS_PANEL_2026-10-11.json` `final.top_now_cpu` items 2, 3 and 6. |

## 0. What this addendum does and does not change

**Adds** (§A) report-only metrics for item 7's existing runs, and (§B) registers a separate one-arm B4 screen that reuses
R0's S-beam run as its control.

**Does not change** anything registered in `PREREG_ITEM7_SEEDS.md`: the systems R0/R1/R2, the S-beam and S-direct
configurations, the fresh set, the training recipes, the two registered tests ("the world-model effect replicates"),
the release-gate reading, the report-only list, or the "Not claimed" section. `scripts/bw_seeds.py` (`b783a191…`) and
its `analyze` stage are unchanged. No metric below is a test, none has a reading that can change item 7's outcome, and
none adds a GPU run to item 7. The screen's GPU run is a separate job after item 7's analysis (D-75.1).

**Pins at writing** (rechecked when the addendum is registered):

| File | sha256 |
|---|---|
| `PREREG_ITEM7_SEEDS.md` | `956a0dc4117c948065f40258be3a7a3e418f51c2b199c4bbb8e7f1337d88985d` |
| `scripts/bw_seeds.py` | `b783a1916bce655cf93e3d55bedcb276dae3dab2bbdc214e00ff633c828f3b08` |
| `scripts/blocks_eval.py` | `b3666424151c78368500535d2571f0eb5b9e546b11764696d99f427a7794acb6` |
| `scripts/bw_run_tests.py` (merge) | `de7022d1eacebab271f7624e962d40d58aeee435494da079a622a03f80a34542` |
| `bwmn/blocks_planner.py` | `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960` |
| `bwmn/blocks.py` | `b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83` |
| `runs/FREEZE_B2_beam.json` / `runs/FREEZE_B2_direct.json` | `412a6ee5…` / `167276d3…` |
| `runs/BW_val_large/instances_seeds_fresh.json` | `6fccb94547184d8ba44a2fdc19ad6c3f1fca584c67c906964281746c0541b3fc` |
| `research/PREREG_FREPLAY.md` (definitions used in §A.1) | `994d74c0b721724a49cd24774b5ecf0fc12027d058718575f276b100b3e975f8` |
| `scripts/freplay.py` (at writing) | `866ca5b696b465d0de77d0882b38323df50f1bb3bcc216ff68aaadbb0beb0a3b` |
| `bwmn/flops.py` (at writing) | `bbada67b554668d5351e2670d744043dd0c5661ce95dd1ac41b8229cbcee1625` |
| `research/PREREG_B4_SCREEN_DRAFT.md` / `scripts/b4_screen.py` | recorded when finalised (draft runner at writing `2bb8c548…`; `cb2a7f63…` after the independent verifier's two fixes) |

## A. Report-only metrics for item 7

All of §A applies to R0, R1 and R2. Beam-only metrics (A.1, A.3, the beam half of A.2) apply to `{R}_beam`; S-direct
has no F.

### A.0 Inputs are hashed before they are read

Before any §A analysis reads an item-7 output, a hashing-only step writes `runs/BW_SEEDS/ADDENDUM_A_INPUTS.json`: the
sha256 and line count of every `plans.jsonl`, `summary.json` and `fcache.jsonl` under `runs/BW_SEEDS/{R0,R1,R2}_{beam,
direct}/` (merged and per shard), plus the six `runs/FREEZE_I7_*.json`. This file is the sha table that
`PREREG_FREPLAY.md` §9 requires for new sets ("an amendment that names its files and shas before any of their outputs
are read"): the files are named here, and their shas are fixed by that step before any parse. `bw_seeds.py analyze`
(the registered item-7 tests) is not affected by this rule.

### A.1 Per-system F audit (FREPLAY metrics)

**Sets.** `i7_R0_beam`, `i7_R1_beam`, `i7_R2_beam` (and, for the screen, `b4s_a_beam`, §B.6).

| Set | F adapter (sha from `FREEZE_I7_{R}_beam.json` `fwd_sha256`) | Caches | Outcomes | Initial states |
|---|---|---|---|---|
| `i7_R0_beam` | `runs/BW_Fdelta_s0/adapter` (`2ed0bb00…`) | `runs/BW_SEEDS/R0_beam/shard_{0..3}of4/fcache.jsonl` | `runs/BW_SEEDS/R0_beam/plans.jsonl` (+ shard files) | seeds_fresh json (`6fccb945…`), ids `seeds_fresh/<index>` |
| `i7_R1_beam` | `runs/BW_Fdelta_s1/adapter` | `runs/BW_SEEDS/R1_beam/shard_{0..3}of4/fcache.jsonl` | `runs/BW_SEEDS/R1_beam/plans.jsonl` | same |
| `i7_R2_beam` | `runs/BW_Fdelta_s2/adapter` | `runs/BW_SEEDS/R2_beam/shard_{0..3}of4/fcache.jsonl` | `runs/BW_SEEDS/R2_beam/plans.jsonl` | same |

**Definitions:** `PREREG_FREPLAY.md` §§2-7, unchanged and by reference (sha `994d74c0…`): the unit (unique (state,
action) key within a set; last line wins within a file; the cross-file conflict rule), the classes and subtypes, the
consistency assertions and the second code path, the strata (block count × action type), the rates with exact
Clopper-Pearson 95% CIs, the per-instance forest, the coverage check, the flagged and support-exposed instances, and the
two registered readings. To avoid the name clash with item 7's systems, FREPLAY's readings are called **FREPLAY-R1**
(oracle-equivalence bounds [L, U] on exact-F success) and **FREPLAY-R2** (support share = false_legal_rejected_by_support_check
/ illegal keys).

**Wording consequence (FREPLAY §7), applied to item 7's own statements.** If FREPLAY-R2 > 5% in a system's set, any
statement that the system's verified plans were valid says "verified by the learned F plus a domain-agnostic
delta-consistency check (a predicted deletion must be a fact of the state)".

**Additional per-system report** (descriptive; no reading): F-flag accuracy on schema-valid keys, the delta-exact rate,
harmful keys by class, and the spread of these rates across the three F seeds. No equality test is made between seeds;
a difference is reported, not interpreted.

**Implementation:** `scripts/freplay.py`, extended only by adding these sets and their pins (no definition change); its
sha is recorded with the result. The FREPLAY track owns that code.

### A.2 Failure taxonomy

**S-beam, per system** (from `plans.jsonl`; success = VAL verdict):

| Class | Definition |
|---|---|
| solved | VAL success (verified or not; an unverified success is listed) |
| verified_but_invalid | `verified` true and VAL false (FREPLAY-R1 predicts 0 when no instance is flagged) |
| dead_end | `fail == "dead_end"` (the beam emptied) |
| budget | `fail == "budget"` (E24 reached) |

For every unverified failure: the number of unmet goal facts in the executor's state after the submitted plan (the
best node's prefix: `plan_beam` submits the node with the fewest unmet facts, then the shortest prefix), as a
distribution, and the **stalled-at-1** count (exactly one unmet goal fact; the F-09 "deceptive minima" pattern). A
prefix that is invalid under the executor is counted separately.

**S-direct, per system:**

| Class | Definition |
|---|---|
| solved | VAL success |
| unparseable | `fail == "unparseable"` (candidate 0 had no parseable plan) |
| invalid_action | parsed, executor `valid` false |
| goal_not_reached | parsed, every action applicable, goal not met |

plus, as a cross-cutting flag, **at_cap**: `gen_tokens ≥ 256` (the frozen M; such outputs may be truncated, P-13).

**Cross-tabs (descriptive):** R1 and R2 vs R0 by class, per configuration; and per system, beam class × direct outcome.

### A.3 Round-capped anytime curves (S-beam)

**Exact round curve:** S_r = the number of instances with VAL success and `rounds ≤ r`, for r = 1..12 and up to the
largest observed `rounds`. By D-48c/D-50 (per-instance CRN seeds keyed by round label, batch-independent sampling,
frozen memoised F), the first r rounds of `plan_beam` do not depend on any later cap, so S_r equals the success of the
same system capped at r rounds (the E24 expansion cap unchanged). Reported per system, with R1 − R0 and R2 − R0.

**Approximate token curve:** the number of successes with `gen_tokens ≤ T`, T = 2k, 4k, 6k, 8k, 10k, 12k, 16k proposer
tokens. Approximate, because a token cap would cut inside a round.

### A.4 FLOPs and token ledger (estimates, labelled as such)

**Logged, exact:** per instance proposer `gen_tokens` and `rounds`; per run `f_model_queries` (cold cache per shard,
so a key queried in two shards counts twice), `expansions_per_instance` (mean only; `blocks_eval.py` keeps no
per-instance expansions) and `wall_s` (planning only).

**Derived exactly from the caches:** unique F keys per system (union over shards; the FR-02 count), and per key the F
prompt length (`B.forward_prompt_delta(s, a)`) and generated length (the cache text + EOS), tokenised with the
Titan-Sparse tokenizer files (tokenizer only, no weights; CPU).

**FLOPs** with `bwmn/flops.py` (sha recorded at analysis; `bbada67b…` at writing), one formula for every term:

- **F:** Σ over unique keys of `flops_breakdown(fp_F, prompt_k, gen_k)`, with `fp_F = flop_params(Titan-Sparse config,
  extra_nonemb = the F adapter's LoRA parameters)`.
- **S-beam proposer (ESTIMATE):** per expansion `shared_prefix_breakdown(fp_Pa, P, [G], sample_pairs =
  beam_proposer_pairs_bound(G, 80))`. P = the tokenised proposer prompt of the instance's initial state (imagined
  states have the same blocks; their prompts differ by a few tokens). G = the instance's `gen_tokens` spread over its
  expansions. Per instance, expansions are bounded by min(24, 1 + 4 × (rounds − 1)); the run total uses the logged mean
  exactly.
- **S-direct:** `flops_breakdown(fp_Pa, P_init, gen_tokens)`.

**Reported:** FLOPs and tokens per instance and per solved instance, per system and configuration, and the beam/direct
ratio. No iso-FLOP claim is made from these numbers, and none is compared with API models.

### A.5 Operational note (not a registration)

bundleI's `i7-run` has `est_s` 600, but R0/R1/R2 evaluation needs about 9-10.5 GPU-h (3 × about 3.3-3.5 h of S-beam
plus three short S-direct runs). The job resumes per shard, so it works; plan for 2-3 BWM turns.

## B. Registration of the B4 futility screen (D-75.1, D-76a)

### B.1 What

One arm, **a** = RCX's hybrid hook `LearnedUnmet.from_file(gnn_s20261276.pt, invalid_policy="fallback",
hybrid_lambda=1.0)` as `unmet=` in the frozen B2 `plan_beam`, run on seeds_fresh-40 with exactly R0's S-beam arguments.
The full design, pins, rule and outputs are in `research/PREREG_B4_SCREEN_DRAFT.md`, which is finalised as
`research/PREREG_B4_SCREEN.md` and hashed together with this addendum.

### B.2 The control is R0's S-beam run, as item 7 produces it

- `runs/BW_SEEDS/R0_beam/` is used unchanged and is not re-run.
- `bw_seeds.py freeze` writes R0's beam args as `dict(FREEZE_B2_beam args, pa=runs/BW_Pa_v2_s1/adapter)` with
  `fwd_adapter = runs/BW_Fdelta_s0/adapter`, i.e. exactly the B2 beam configuration. R0's beam run is therefore B4's
  arm c (the default key) on these 40 instances: same F (`2ed0bb00…`), Pa (`ee1c5aec…`), eval_seed 0, CRN seed function
  and shard partition (index % 4).
- The screen runner refuses to freeze or run unless the R0 record's args equal FREEZE_B2_beam's and the adapter shas
  match.

### B.3 Timing: the screen is fixed before any R0 output exists

- `scripts/b4_screen.py freeze` writes `runs/FREEZE_B4S_a_beam.json` immediately after `bw_seeds.py freeze` and before
  `bw_seeds.py run`. It records R0's freeze sha, the args, the hook spec, every pin, the rule, the pre-check result and
  the shas of this addendum, the screen prereg and the runner.
- It refuses if any R0 beam output exists (any file under `runs/BW_SEEDS/R0_beam/`, including a running shard's
  `progress.txt`), unless a late freeze is passed explicitly (`--allow-after-r0`), in which
  case the list of R0 files present is recorded.
- Everything about the arm (module, call, args, instances, drop rule, decision rule) is fixed by the screen prereg, so
  nothing can be chosen after R0's outputs are known, even with a late freeze.
- **Queue (the lead's change, not made by this draft):** insert `b4s-freeze` between bundleI's `i7-freeze` and
  `i7-run`. If it fails, item 7 proceeds unchanged; the release gate is never blocked by the screen.

### B.4 The decision rule (verbatim from the screen prereg §5.2)

> GO iff g − l ≥ 2, where g = a-only and l = R0-only VAL successes on the paired instances; otherwise NO-GO (net ≤ 1
> is always NO-GO). The one-sided sign p and the two-sided McNemar p are reported but do not gate (D-77). If more than
> 5% of hook calls return a penalty-band value, a GO is not credited to the learned critic alone and c+pen is mandatory
> in the full B4 (D-76a).

### B.5 What item 7 gains and loses

- **Item 7's tests:** nothing changes. R0's outputs get one more registered reader.
- **seeds_fresh exposure:** after the screen, the set has been used by four S-beam systems (R0, R1, R2, a) and three
  S-direct systems. The screen's result selects nothing in item 7.

### B.6 Report-only additions tied to the control (registered here because they read R0's outputs)

- **c ≡ c+pen certificate on R0:** every F-accepted prediction in R0's beam caches checked with the hook's
  `is_consistent`, with the parent state's blocks as `objs` (so a vanished goal block, which the hook penalises, is
  caught although the cache does not record the goal). If none is inconsistent, c+pen would have made exactly R0's choices, and R0 serves as the c+pen
  control; otherwise the screen reports R0 as c only.
- **R0 failures stalled at 1 unmet goal fact** (D-76a), and how many of them arm a solves (A.2's definition).
- **F cache cross-check R0 vs a:** keys answered in both runs must have identical F text (a check of D-48c/D-50).
- **The arm's F audit:** set `b4s_a_beam` (caches `runs/BW_B4_SCREEN/a_beam/shard_{0..3}of4/fcache.jsonl`, F
  `BW_Fdelta_s0`) under A.1's definitions.

## C. Order of operations

1. Finalise and register this addendum and `PREREG_B4_SCREEN.md` (shas into DECISIONS); RCX co-signs §B and the screen
   prereg.
2. bundleI: train `BW_Pa_v2_s2`, `BW_Pa_v2_s3` → `bw_seeds.py freeze` → `b4_screen.py freeze` → `bw_seeds.py run` →
   `bw_seeds.py analyze` (the registered item-7 tests).
3. CPU, after item 7's analysis: the A.0 hashing step, then A.1-A.4.
4. The next BWM GPU turn: `b4_screen.py run` (about 3.5 GPU-h), then `b4_screen.py analyze` (CPU). Then B6 (D-75.1).

## D. Not claimed; limits

- Nothing in §A is a test. Seed spread in F behaviour is described, not tested; a non-significant difference is not
  read as equivalence (as in the base prereg).
- §A.4 is an estimate where the logs lack per-expansion detail (labelled); no iso-FLOP or "cheaper than model X" claim
  follows from it.
- §B registers a futility screen, not a B4 claim; its limits are listed in the screen prereg §9.
