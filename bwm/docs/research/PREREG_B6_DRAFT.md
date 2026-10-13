<!-- DRAFT produced 2026-10-13 ~08:05 CDT by a read-only design workflow (technical map, 3 designs, 2 judges, 1 synthesis); NOT frozen, NOT approved by the user. Items marked [VERIFY] must be checked and items marked [FILL AT FREEZE] must be filled before the prereg is hashed or the freeze runs. -->

# Addendum B6 (DRAFT): round-2 search distillation at 11-15 blocks, with a position-matched control

| | |
|---|---|
| Status | **DRAFT.** Written 2026-10-13 ~08:05 CDT. Not frozen, not hashed, not approved by the user. Proposed final file: `PREREG_B6_DISTILL_R2.md` at the bundle root (like `PREREG_B3_DISTILL.md`). Becomes registered when its sha256 is recorded in DECISIONS. |
| Written before | Any B6 seed is consumed and any B6 set, problem, solution, row, mix, adapter or freeze record exists. Checked 2026-10-13 07:55 CDT: no `runs/BW_DISTILL_r2/`, no `runs/FREEZE_B6*`, no `instances_b6_*`; seeds 2026141x-2026142x appear in the repo only as panel proposals (§4.3). The B4 screen (bundleJ) was running (arm a, started 07:00); its outcome is unknown at writing. |
| Decision source | D-75.1 (user: after item 7, the B4 screen, then B6 = "round-2 search distillation aimed at 11-15 blocks, with a matched control", then item 9). B3 design and bar: D-56, D-58, P-11. Scope corrections: P-13, P-14 (b)-(c). Wording: P-15. Seed/F-recipe limits: P-17. Contamination rules: P-18, E-05. Control-skill rows: X-04. Pins: D-59, D-63, B4-screen pins (`b4_screen.py`). RCX scope: D-64. Seed ranges: D-67 follow-up. Night windows: D-84. Cost lessons: I-21, E-04. Forensics: F-09. Why the misplaced key is not used: X-05, D-70 (B4 arm a0). Panel: `research/NEXT_ANALYSIS_PANEL_2026-10-11.json` rank 1 (B6). |
| Base prereg | `PREREG_B3_DISTILL.md` sha256 `e6aa884b8af74243e23624b451e96249175e65d4978e0e826a2123cdab4fe74a`. **It stays byte-identical.** All B2/B3 rules carry over unless changed here: one plan per instance, no execution feedback; Clopper-Pearson 95% CIs; exact two-sided McNemar for paired contrasts. |
| Tier | A (model-native). The teacher is BWM's own system (Pa_v3 + learned F in the frozen beam); no external module; the exact simulator and VAL never produce or filter a label. |
| Runner / tests | `scripts/bw_b6.py` sha256 [FILL AT FREEZE]; `tests/test_bw_b6.py` sha256 [FILL AT FREEZE] (n pass [FILL AT FREEZE]). |
| Canonicaliser / audit | `scripts/bw_canon.py` sha256 [FILL AT FREEZE]; `tests/test_bw_canon.py` sha256 [FILL AT FREEZE]; `scripts/bw_b6_audit.py` (independent method B) sha256 [FILL AT FREEZE]. |
| Peer conditions | B6 is BWM's and needs no co-sign. **But** the B4 fresh-set recipe (seed 20261410) is co-signed with RCX (PREREG_B4_SCREEN §8.1); the cross-exclusion in §4.6 needs RCX's written agreement **before this text is hashed** [FILL AT FREEZE: RCX record id/commit]. |

## 0. Outcomes known at registration

**Known before this text was written** (all used in the design; none is a B6 outcome):
- B2: hard S-beam 73/110, S-direct 5/110 (P-08); mirror 61/110 (P-10).
- B3: P-11 Pa_v3 vs Pa_v2c on b3fresh 22 vs 8 (+18/−4, p 0.0043). By stratum (recomputed from
  `runs/BW_B3_EVAL/plans_{v3,v2c}_*.jsonl`): b3fresh 6-10 +11/−3, 11-15 +7/−1; mirror 6-10 +9/−1, 11-15 +7/−0.
  Pa_v3 at M256 truncated 25/110 on b3fresh, all at 11-15 (11:1, 12:1, 13:5, 14:13, 15:5), none successful; every
  successful 11-15 plan has ≤ 32 actions.
- H1 13/110 (P-13; 12 of 13 at 6-10). H2 88/110 vs 73/110 (P-14; 11-15 +17/−4; 12-15 +14/−4 of 62). H2 hard 11-15
  failures: 16 budget, 4 dead end; most stall at one unmet goal fact (F-09). Beam at 11 blocks is near ceiling (H2 hard
  12/12; B2 on generated sets 14/16).
- P-15 FREPLAY (support-check share 13-28%); P-16 (post hoc); P-17 (seed spread R1 vs R0 +5/−5; Fdelta_s2 has 159
  harmful false-illegal keys at 12-15 blocks, Fdelta_s0 0; 15 blocks 0/3); P-18 (all registered test sets clean except
  seeds_fresh/0; val_v2 303/400 leaky; rules 1-3).
- X-05 (GBFS, exact dynamics: misplaced_blocks 19/75 vs unmet 0/75 at 200 expansions); X-07 (RCX E11 used B3 plans).

**Not known:** any B6 output; the B4 screen's reading (record it in this header at freeze, [FILL AT FREEZE]; no B6
setting depends on it, only scheduling, §8).

## 1. Question and hypotheses

**Question.** Round 1 distilled F-verified 6-10-block plans into the proposer; its direct gain stayed in range, while
the remaining failures of the best system (H2) are at 11-15 blocks. If the round-1 system (H2: Pa_v3 + `BW_Fdelta_s0`
in the frozen beam) solves fresh 11-15-block problems and its F-verified plans are distilled into Pa_v3, does the
proposer plan better at 11-15 blocks than after the same extra training without those plans? If so, does the gain
carry into the unchanged frozen search?

**Hypotheses.**
- **H-A (confirmatory, P1).** S-direct(Pa_v4_s1) > S-direct(Pa_v3c_s1) on `b6test` (fresh, 11-15 blocks).
- **H-B (confirmatory, P2; tested only if P1 is MET, R does not BLOCK and the user approves the GPU time).**
  S-beam(Pa_v4_s1) > S-beam(Pa_v3c_s1) on `b6beam` (fresh, 12-15 blocks), frozen B2/H2 beam with only `pa` swapped.
- **H-A2 (replication, outside the confirmatory family).** The H-A contrast holds again with training seed 2.

**Minimum effects of interest (fixed now).** P1: +8 pp (the round-1 out-of-range discordance, p10 .095 / p01 .014).
P2: +15 pp (round 1's system effect at 11-15 was +17/−4 of 74 = +17.6 pp; at 12-15 +14/−4 of 62 = +16 pp).

## 2. Systems and arms

**Shared components (all arms, all stages).**

| Component | Value |
|---|---|
| Core | Titan-Sparse FP32, frozen (D-02) |
| F | `runs/BW_Fdelta_s0/adapter` sha256 `2ed0bb0064bc6b11ec9c661840a38094e468462cea2ff833cda3bf234ead65ae`, delta format. Never substituted (P-17: Fdelta_s2's error profile differs). |
| Planner | `bwmn/blocks_planner.py` sha256 `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960`, imported read-only, default `unmet` key (no hook). |

**Arms.**

| Arm | Adapter dir | Init | Training rows | Role |
|---|---|---|---|---|
| **Pa_v4_s1** (v4) | `runs/BW_Pa_v4_s1` | `runs/BW_Pa_v3_s1/adapter` (`48c9d279bd8813311f3232bc867a88d96604a93347270a16ac57829662e4c678`) | `mix_v4` = D2 + Rep | treatment |
| **Pa_v3c_s1** (v3c) | `runs/BW_Pa_v3c_s1` | same | `mix_v3c` = S + Rep | matched control; the comparator of P1 **and** P2 |
| Pa_v3_s1 (v3) | existing | — | — | incumbent; retention-guard comparator; report-only on b6test |
| Pa_v4_s2, Pa_v3c_s2 | `runs/BW_Pa_{v4,v3c}_s2` | Pa_v3_s1 | same mixes, `--seed 2` | A2 only (§5.5) |

**The matched control, exactly.** N = |D2| after the filters of §3.4. P = `runs/BW_DISTILL_r1/mix_v3.jsonl` (8,508
rows: 4,254 D1 + 4,254 v2; 422 failure-outcome rows; sha256
`7dadf5c9ab271383489d91c3394d44fac580bc574ffd72a6eec430035af0c656`), Pa_v3's own training mix. rng =
`random.Random(20261424)`; perm = rng.sample(range(8508), 8508).

| Symbol | Definition |
|---|---|
| Rep | If N ≤ 8,508: P[perm[0..N−1]]. If N > 8,508: all of P in perm order, then N − 8,508 rows of `runs/BW_data_v2/proposer_actions_train.jsonl` not in P (by exact (prompt, target) text), drawn by the same rng without replacement. |
| S (stand-in) | The success-outcome rows (outcome read from the prompt JSON) of P[perm[N..]], in perm order, first N. If fewer than N, top up with success-outcome v2 rows not in P and not in Rep, drawn by the same rng without replacement. Top-up count reported. |
| `mix_v4.jsonl` | D2 at indices 0..N−1 (solution order, then t), Rep at N..2N−1 |
| `mix_v3c.jsonl` | S at indices 0..N−1, Rep at N..2N−1 |

Only `prompt` and `target` are written (as bw_b3_prep.py:63-66).

**Matched by construction.** `train_generic.py` fixes the row order from the row count and `--seed` only
(train_generic.py:47-52; passed as `row_order` to `lm.train_sft`). With 2N rows and the same seed both arms see the same
Rep row in the same batch slot at every step, D2[i] and S[i] in the same slots, identical failure-outcome rows (only in
Rep), and the same steps, batch, lr, schedule, init and max-len. This removes B3's position and failure-row mismatches
(B3: 422 vs 804 failure rows).

**Not matched, reported.** Supervised tokens (D2 targets are longer; expected v4 ≈ 1.4-1.6×), prompt tokens, and the
block range of the substituted rows (that difference is the treatment). The control isolates "D2 content vs more of the
distribution Pa_v3 already learned"; it does **not** separate search-found content from exposure to long valid 11-15
targets (§14).

## 3. Training data

### 3.1 Search problems (`b6search`)

- **Generator:** `B.generate(rng, n, full_goal_frac=1.0)` (blocks.py:194-213), the draw used by `VL.make`
  (blocks_val_large.py:42-56): init towers p_stack 0.6; goal = every on-pair of a p_stack-0.85 target; goal unmet at
  init. Implemented as a new function in `bw_b6.py`; `blocks_val_large.py` and `bw_distill.py` are not edited.
- **Seed** `random.Random(20261420)`. **Sizes:** 11-15 blocks, 32 each (n = 160), drawn in 16 blocks of 10 with 2 per
  size; the size order within a block is shuffled by the same rng, then generation continues on it. Search chunk k =
  block k, so every completed prefix is size-balanced. Ids `b6search/0..159`.
- **Exclusions:** exact renaming class (§4.1) of the universe U (§4.2), of b6test, b6beam and b6ret, and within-set
  duplicates. Rejected draws are redrawn from the same stream and counted by reason and size.
- **No `VL.invariant` exclusion** (coarse: on r1 suffix rows it would wrongly drop 16/36, 23/36 and 4/14 of the 6-10
  instances of b3fresh, mirror and seeds_fresh, where exact matches are 0). The share of draws the legacy rule
  (invariants of hard, val_large, mirror, b3fresh, seeds_fresh, distill-240) would have rejected is computed and
  reported for every B6 set (about 5% expected at 11-15 [VERIFY]).
- **Search cost cap (outcome-independent, by index prefix only):** search uses at most 3 BWM turns counted from the first
  search chunk. Chunks not started by the end of the third search turn are dropped. If fewer than 8 chunks (80
  problems) are complete by then, search continues until chunk 7 is complete. The cap rarely binds at the central
  estimate (§8). Wall time correlates with teacher failure; this is disclosed.
- **File:** `runs/BW_DISTILL_r2/problems.json`, written once (refuse overwrite), sha into DECISIONS before any GPU job.
- **Disclosure:** the generator imitates hard's whole-configuration tall-tower goals, with taller towers than hard
  (P-10); the 11-15 target was chosen after the hard strata were known (P-14), as B3's 6-10 range was.

### 3.2 Teacher and solver

`runs/FREEZE_B6_teacher.json` is written before any search output. It holds `runs/FREEZE_B3_H2.json` (sha256
`a433e499e23cf6fb1edd353c50b8cc9466e224b8adca56ef5731306ce46bda51`) args with two stated values:

| Arg | Value | Note |
|---|---|---|
| K | **4** | r1 precedent (bw_distill.py:132); selection K4-M80 0.790 vs K8-M80 0.807 (DECISIONS:430) |
| eval_seed | **7** | stated this time (r1 used 7 silently, bw_distill.py:137) |
| pa / fwd | Pa_v3_s1 / Fdelta_s0, delta | |
| W, E, R, M, P, max_depth | 4, 24, 12, 80, 1, 80 | frozen B2 beam values; `plan_beam` default max_depth |
| batch tokens | proposer 6,000; F 5,000 | as bw_distill.py:70-72 |

- One pinned `BP.plan_beam` call per chunk of 10, with a pass-through recording F wrapper (as bw_distill.py:78-96). No
  planner edit, no hook. Sampling seeds depend on (eval_seed, id, round, candidate) (blocks_planner.py:117) and outputs
  do not depend on batch composition (D-48c, D-50), so chunking changes cost only.
- **Per-problem record** (`runs/BW_DISTILL_r2/solutions.jsonl`, appended per chunk; `chunk_log.jsonl` per chunk):
  id, n, objs, init, goal, verified, plan, f_states, rounds, expansions, gen_tokens, fail, chunk, chunk_wall_s, chunk F
  model queries. Analysis only: `exact_valid_for_analysis_only` (`B.execute`), `val_valid_for_analysis_only` (VAL via
  `B.write_problem` + `B.val_check`, as blocks_eval.py:139-151), 2 × `must_move` lower bound, repeated-F-state count,
  u*_F (unmet goal facts at the end of the submitted plan under F's recorded states; for unverified plans this is the
  best node, blocks_planner.py:338-339), u*_exact, and the count of exact-state mismatches along f_states.
- Unique F keys are counted from `fcache.jsonl` per chunk and in total (E-04; summaries undercount after restarts).

### 3.3 Label rule (unchanged from B3)

- A plan becomes training data iff `plan_beam` returned verified = True: F predicts every step legal and
  delta-supported and the goal is reached (blocks_planner.py:241-259, 309-316; blocks.py:349-361).
- The exact simulator and VAL never filter. No plan shortening or loop removal; the loop share is reported. Maximum plan
  length = path (≤ max_depth 80) + one rollout (≤ R 12) = 92 actions [VERIFY against plan_beam].
- Wording follows FREPLAY scope: "verified by the learned F plus a domain-agnostic delta-consistency check" (P-15).

### 3.4 Rows and filters

**Rows** (bw_distill.py:106-125 semantics): each suffix t of each verified plan gives prompt =
`B.proposer_prompt(F_state_t, goal)` with outcome "success", target = `B.target_actions(plan[t:])`. Row metadata: id,
n, t, rem, u_t.

**Filters, in order, each counted in `ROW_FILTER.json`:**
1. **Symmetric contamination drop (P-18 rule 2).** Drop the row if the exact class of (F_state_t, goal, objs) belongs
   to any evaluation, selection or test set in U(c)-(d), to b6test, b6beam or b6ret, or to the B4 fresh set if it exists
   when `rows` runs (§4.6). Expected 0.
2. **Malformed F states.** The canonicaliser is total (§4.1): a state with a non-functional or non-injective `on`
   relation, duplicate or unknown-arity facts gets a MALFORMED code, never an exception. Such a state cannot be
   isomorphic to a well-formed instance (tested), so filter 1 does not drop it; it is kept (labels are never filtered
   by validity) and counted. Expected 0 for Fdelta_s0 (P-15: 0 wrong states).
3. **Overlength drop.** Drop rows with prompt + target + EOS > **768** tokens, measured exactly as `lm.encode_pair`
   (lm.py:313-321), because `lm.train_sft` raises on any overlength row (lm.py:413-414). Basis: a tokenizer audit found
   15-block prompts up to 228 tokens and 4.38 tokens per action (max 4.55), so a 92-action row is ≈ 646-648 tokens,
   above B3's 640. Expected 0 drops at 768. `TOKEN_AUDIT.json` [FILL AT FREEZE] records the audit.
4. No deduplication; no exact-simulator filter.

D2 = the rows left. `DISTILL_META.json` records problems, verified, exact-valid and VAL-valid counts by size, rows,
label precision, the share of F-verified plans that are exactly invalid, malformed-state rows and every drop.

### 3.5 Teacher futility (pre-registered)

If fewer than 32 plans are verified or N < 800, B6 stops after `rows`: `TEACHER_FUTILITY.json`, reading
**TEACHER-FUTILITY** (not evidence against distillation), no training, no test set touched; the teacher statistics are
reported and the GPU passes on.

**Expected yield (estimate, not a gate).** About 45-60% verified (H2 hard 11-15 73% at K8; B2 on generated 11-15 sets
44/100; K4 costs ≈ 2 pp), i.e. 72-96 plans × ≈ 38 actions ≈ 2,700-3,700 rows.

### 3.6 Class export

`TRAIN_CLASSES_r2.json`: the sha256 of the exact class of every D2 row and every b6search problem (S top-up rows are v2
rows, already in U). It feeds B4's 20261410 draw, RCX, RCWM and any later fresh set (P-18 rule 1).

### 3.7 Training recipe (identical for both arms; nothing selected)

```
python scripts/train_generic.py --rows runs/BW_DISTILL_r2/mix_v4.jsonl  --init-adapter runs/BW_Pa_v3_s1/adapter \
  --steps 400 --batch 16 --lr 0.0001 --seed 1 --max-len 768 --out runs/BW_Pa_v4_s1
python scripts/train_generic.py --rows runs/BW_DISTILL_r2/mix_v3c.jsonl --init-adapter runs/BW_Pa_v3_s1/adapter \
  --steps 400 --batch 16 --lr 0.0001 --seed 1 --max-len 768 --out runs/BW_Pa_v3c_s1
```

- Saved LoRA continued: r16, α32, dropout 0.05 on q, k, v, o, gate, up, down (lm.py:281-289); AdamW (0.9, 0.95),
  weight decay 0; WSD restarted (10% warmup, last 20% to a 0.1 floor; lm.py:379-387); checkpoints every 50 steps,
  resume only from a checkpoint; final adapter only. No early stopping, no checkpoint or step selection.
- 6,400 draws ≈ 0.9-1.2 epoch of 2N at the expected N (B3: 0.75). Under `--init-adapter`, `--seed` sets the row order
  and the train_sft seed (dropout) [VERIFY].
- **Memory smoke (GPU, before any search):** `b6-memsmoke` trains 3 steps at batch 16 on synthetic 760-768-token rows
  into a scratch dir and records peak memory and tokens/s (Pa_v3 peaked at 3.01 GiB at 640). If peak > [FILL AT FREEZE:
  threshold from card VRAM] GiB, the bundle stops and max-len is revisited with the user before any D2 exists.

## 4. Test sets and contamination hygiene (P-18)

### 4.1 Canonicaliser (CPU, before anything else)

- Promote `research/audit_evidence/bwm_h2_scratch/leakhunt/canon.py:32-71` (exact port-BFS form; sha256
  `1207b622d26a0e3bf49fe36085e8e0107916bc51e00746bf1a9bc831780c5dfa`) into the **new** `scripts/bw_canon.py`.
  Nothing in `bwmn/` changes.
- **Totality fix (required).** canon.py:19-20 and :26 use `assert` for non-functional/non-injective `on` facts and
  arity. These become explicit checks returning `("MALFORMED", reason, sorted facts)`, safe under `python -O`; duplicate
  facts are de-duplicated first.
- **Validation before any set is generated** (`tests/test_bw_canon.py`):
  (a) equals `B.signature` on ≥ 3,000 random problems at n ≤ 7, including holding states;
  (b) invariant under 1,000 random renamings at 8-15 blocks;
  (c) separates 1,000 single-fact perturbations;
  (d) agrees with method B (P-18 individualisation-refinement) on ≥ 2,000 problems at 8-15 blocks;
  (e) P-18 positive controls: seeds_fresh/0 ≡ v2 row line 20184; 211/240 distill problems found in distill_rows;
      72/602 PB instances in the superseded pre-B-02 file;
  (f) malformed inputs return MALFORMED, never raise, never equal a well-formed code.
- **Archive first:** P-18's audit code exists only in the session scratchpad (`audit_common.py`, `audit_main.py`,
  `audit_b.py`, `audit_large_n.py`, `c_verify.py`; present at 07:55). Copy them with shas to
  `research/audit_evidence/bwm_p18_scratch/` (D-74 precedent). Promote method B as `scripts/bw_b6_audit.py`.

### 4.2 Exclusion universe U

`runs/BW_DISTILL_r2/EXCLUSION_UNIVERSE.json` records, per source, path, sha256 and class count:

- **(a) Every proposer training or held-out row file**, including suffix and failure-outcome rows:
  `runs/BW_data_v2/proposer_actions_train.jsonl` (`6b46743a2837c511902493adf90b231e795b4849d2df81963d4494421277709c`,
  53,682 rows) and every other `proposer_*` file in `runs/BW_data/` and `runs/BW_data_v2/` (actions and states, train
  and val) [FILL AT FREEZE: shas]; `runs/BW_DISTILL_r1/distill_rows.jsonl`
  (`e0d1c72822a3cee7151afec4a2503429552e18f061e0f17ad197264c0a41928b`); mix_v3 (`7dadf5c9…`); mix_v2c
  (`42312ac1ea6dc4c15dbe7fec9b140a193c6c7c656e90b54630e2c9338608aa46`). Forward files carry no goals (P-18).
- **(b) Training problems:** distill-240 (`BD.problems(240)`, bw_distill.py:35-46).
- **(c) Evaluation and selection sets:** hard (110); PB-602 (generated_basic_3 + generated_basic); mirror
  (`2f61e38bf2f99e2c5f7524d8642618d51d1e2d83573123524b3d8e41155d53f2`); val_large
  (`9be244f2a97fc76e1a6f473f8ada14e74be1e38062ba290336dd44dc44eb656e`); b3fresh
  (`575c45b14c88f482728cc04e9f7234eabc7d160033f52e3f7f05fa8f4ca9844d`); seeds_fresh
  (`6fccb94547184d8ba44a2fdc19ad6c3f1fca584c67c906964281746c0541b3fc`); val_v1 (`runs/BW_data/instances_val.json`,
  `baa6fc76…`); val_v2 (`runs/BW_data_v2/instances_val.json`, `ca1dd0e0…`).
- **(d) Peer lists:** RCX E07 (`runs/BW_B4_SCREEN/rcx_snapshot/runs/RCX-E07/main/data/instances.json`,
  `1e39f9bbc25a7e857ccbfee97c996638fed63d606c334efef88fc3b7a32f4e31`); E11a
  (`research/b4_exclusions/rcx_e11a_instances.json`, `6cbec9cfc87d5ef3929086663ea34905221730633cb54b33309158861b71f876`);
  E11b (`rcx_e11b_instances.json`, `47c44d315ec54c6c54d49d179bc067194330371fbb9a7540dbdc02d22e166212`); E07b and E08 if
  received before `sets` runs (requested now; they do not block, since no BWM adapter trained on RCX data; late lists
  fall under V5).
- At 11-15 blocks no existing training row can collide (v2 is 3-6 blocks, D1 6-10); the check still runs and is
  recorded. At 6-10 (b6ret) the v2 6-block rows and D1 rows do bind.

### 4.3 Sets and seeds

All sets use the §3.1 generator, exclude U plus every earlier B6 set, are written once (refuse overwrite) by one CPU
step (`bw_b6.py sets`) **before any B6 GPU job**, and are hashed into DECISIONS.

| Set | File | Seed | Blocks × n | n | Use |
|---|---|---|---|---|---|
| b6test | `runs/BW_val_large/instances_b6_test.json` | 20261421 | 11-15 × 40, size order | 200 | P1, A2 (S-direct) |
| b6beam | `runs/BW_val_large/instances_b6_beam.json` | 20261422 | 12-15 × 24, in 12 blocks of 8 (2 per size); shard k = block k | 96 | P2 (S-beam); **sealed** until `FREEZE_B6B.json` |
| b6ret | `runs/BW_val_large/instances_b6_ret.json` | 20261423 | 6-10 × 20 | 100 | retention guard R (S-direct) |
| b6search | `runs/BW_DISTILL_r2/problems.json` | 20261420 | 11-15 × 32 | 160 | teacher problems (§3.1) |

- **Generation order:** b6test → b6beam → b6ret → b6search. Mix sampling uses 20261424. **20261425-29** are reserved
  for a regeneration under V2. **20261419** is the scratch smoke-test seed (scratch root only; never a registered set).
- **Seed bookkeeping:** the panel proposed 20261420-22 for B6 and 20261423 for the B-best confirmation set (panel JSON
  :86, :116); a repo grep found no other use of 2026141x-2026142x. Record 20261419-29 as consumed by B6 in DECISIONS
  before `sets` runs; B-best needs a new seed.
- **Not measured:** a fresh 3-5-block set (rule 1 would distort it: v2 covers most of that class space). PB-602 (clean,
  P-18) gives report-only 3-5-block retention instead (§5.9).

### 4.4 Custody

- No system output on b6test or b6ret before `FREEZE_B6.json`; b6beam is read only by `sets` and `audit` before
  `FREEZE_B6B.json`. Inspection before then is limited to counts by size, exclusion records and shas.
- Lists go to RCX and RCWM **for exclusion only**; no peer evaluates on any B6 set. RCX reads other B6 artifacts only
  after `B6A_REPORT.json` exists (D-64 analogue).

### 4.5 Independent audit

`bw_b6_audit.py` (method B) runs after the row filter and before training, and must report 0 overlaps
(`OVERLAP_AUDIT.json`):
- between {b6test, b6beam, b6ret} and {U(a), U(b), D2, mix_v4, mix_v3c, b6search};
- between {b6search, D2} and every set in U(c)-(d).
A non-zero count triggers V2 (§9).

### 4.6 B4 interplay and peers

- B4's fresh set (seed 20261410) is generated only with the B4 freeze (D-70 follow-up) and is not generated by B6.
- **Rule: the later-generated side excludes the earlier.** B4's draw must exclude b6test, b6beam, b6ret, b6search and,
  if they exist at that time, `TRAIN_CLASSES_r2.json`, mix_v4 and mix_v3c, by exact class. If B4's set exists when B6's
  `rows` stage runs, B6's symmetric filter and audit add it.
- Because B4's recipe is co-signed (PREREG_B4_SCREEN §8.1), this amendment needs RCX's written agreement before this
  text is hashed [FILL AT FREEZE]. Request RCX's E07b and E08 lists now; notify RCWM.
- If the B4 screen reads GO, the placement of the full B4 (about 7 turns) relative to B6 is the user's decision (D-75.1
  does not place it).

## 5. Registered tests and readings

### 5.1 Freeze and direct runner

`runs/FREEZE_B6.json` is written once, after both seed-1 adapters exist and before any b6test or b6ret output. It holds:
adapter shas (v4_s1, v3c_s1, v3_s1); shas of this prereg, b6test, b6ret, b6beam (sealed), problems.json,
EXCLUSION_UNIVERSE, solutions.jsonl, DISTILL_META, ROW_FILTER, OVERLAP_AUDIT, TOKEN_AUDIT, MIX_META, mix_v4, mix_v3c,
TRAIN_CLASSES_r2; code shas (bw_b6.py, bw_canon.py, bw_b6_audit.py, planner, lm.py, blocks.py, blocks_eval.py,
train_generic.py); library versions; the direct settings **K 1, greedy candidate 0, M 512, eval_seed 0, proposer batch
tokens 6,000**; the b6beam shard map; the decision rules of this section.

**M 512** is a new frozen direct setting: it fits a 92-action plan (≈ 419 tokens + EOS). The frozen M256 truncated 25/74
of Pa_v3's 11-15 outputs on b3fresh and "can only understate" (P-14 (c)).

**Runner** (`bw_b6.py run-direct`, modelled on bw_b3_eval.py:46-86): verifies every sha; builds
`BP.LMProposerBW(K=1, max_new=512, max_batch_tokens=6000)`; `BP.plan_direct(eval_seed=0)`; scores with `BP.score`; VAL
on every plan (blocks_eval.py:139-151 method). One cell per (arm, set); a completed cell is never re-run; cells are
deadline-aware (§8). Success = the exact executor's verdict; any VAL disagreement is reported and VAL governs (0
disagreements so far: P-08, P-10, P-13, P-14, P-17).

**Statistics** (code as bw_b3_eval.py:89-99): b = v4-only successes, c = comparator-only; Δ = 100(b − c)/n pp;
p = `binomtest(b, b+c, 0.5).pvalue` (two-sided exact McNemar), 1 if b + c = 0; Clopper-Pearson 95% CI per arm plus the
exact conditional CI of b/(b + c).

### 5.2 P1 (primary): S-direct-512 on b6test (n = 200), Pa_v4_s1 vs Pa_v3c_s1

- **MET** iff p < 0.05 **and** Δ ≥ +5.0 pp (b − c ≥ 10). **PARTIAL** iff Δ > 0 and not MET. **NO** iff Δ ≤ 0.
- **B3-bar line (named secondary, always reported beside P1):** Δ ≥ +10.0 pp and p < 0.05 (PREREG_B3_DISTILL:48). No
  text may say "met the B3 bar" unless this line holds.
- **Why not B3's bar as the rule:** at a true +10 pp it gives P(MET) ≈ 0.53 at any n, and 0.23 at the minimum effect
  (§6). The +5 pp floor only guards against trivially small significant effects. **Disclosure:** the floor and the
  minimum effect were set after round-1's 11-15 stratum (+7/−1 of 74) was known. **This departure from a
  user-approved convention (D-58) needs the user's sign-off before the hash (§16, decision 1).**

### 5.3 Retention guard R: S-direct-512 on b6ret (n = 100)

- **BLOCK** iff Pa_v4 < Pa_v3 with c > b and exact two-sided p < 0.05 (v4 vs the incumbent). BLOCK only prevents
  Stage B and adoption; it never creates a claim.
- Flag (descriptive) if Δ(v4 − v3) ≤ −5 pp. Also reported: v4 vs v3c and v3c vs v3 on b6ret.

### 5.4 Gate files (written only by `bw_b6.py analyze-a`)

- `P1_MET.json` iff P1 is MET (triggers A2). Otherwise `P1_NOT_MET.json`.
- `B6A_GO.json` iff P1 is MET and R does not BLOCK (permits the Stage B approval request). Otherwise `B6A_NO.json`.
- A hand-written or edited gate file voids everything downstream of it (V8).

### 5.5 Stage A2: seed replication (runs automatically iff `P1_MET.json` exists)

- Train Pa_v4_s2 and Pa_v3c_s2 (identical mixes and args, `--seed 2`); freeze in `FREEZE_B6_A2.json`; S-direct-512 on
  b6test.
- **REPLICATES** iff the seed-2 contrast meets the P1 rule; otherwise SAME SIGN (Δ > 0) or DOES NOT REPLICATE. Seed
  spread (v4_s1 vs v4_s2, v3c_s1 vs v3c_s2) reported.
- The recipe-level claim ("round-2 distillation improves …") requires REPLICATES. No seed picking: Stage B and any
  adoption use the seed-1 adapters.

### 5.6 P2 (Stage B, conditional): S-beam on b6beam (n = 96), Pa_v4_s1 vs Pa_v3c_s1

- **Preconditions:** `B6A_GO.json` exists **and** the user's explicit approval of the GPU time is recorded in DECISIONS.
  bundleK2 is queued only after that. Nothing runs automatically.
- **Freeze:** `runs/FREEZE_B6B.json` = `FREEZE_B3_H2` args verbatim (K 8, W 4, E 24, R 12, M 80, P 1, max_depth 80,
  step_cap 24, delta Fdelta_s0, eval_seed 0) with only `pa` set per arm, plus all shas.
- **Runner:** `bw_b6.py run-beam` builds exactly what blocks_eval.py:106-131 builds; a CPU parity test (exact F +
  oracle proposer on 5 val_large instances) must give byte-identical plans to `blocks_eval.py` before the freeze.
  `blocks_eval.TEST_SETS` is not edited. Separate F cache per arm (cost accounting, E-04).
- **Shards:** 12 per arm (shard k = b6beam block k, 8 instances, 2 per size). Job order v4/s0, v3c/s0, v4/s1, v3c/s1, …
  A killed shard restarts from its F cache (D-50, I-22).
- **Interim** (n1 = 48: shards 0-5 of both arms; `analyze-b --interim` only): **STOP for futility** iff b1 − c1 ≤ +1.
  Shards 6-11 then stay unexposed. Non-binding; no interim efficacy stop.
- **Final** (n = 96): **MET** iff b > c and p < 0.05; PARTIAL iff b > c otherwise; NO iff b ≤ c. A futility stop
  reads NOT MET (stopped at interim).
- **Optional third arm (user decision, +50% cost):** Pa_v3_s1 (incumbent) on the same shards, report-only, giving an
  adoption reading v4 vs v3. Not in the confirmatory family.

### 5.7 Overall reading

| P1 | R | Stage B | B6 reading |
|---|---|---|---|
| MET | no BLOCK | P2 MET | **MET-SYSTEM**: proposer gain at 11-15 and frozen-search gain at 12-15, both vs the matched control |
| MET | no BLOCK | P2 PARTIAL/NO/stopped | **MET-PROPOSER**: system gain not shown |
| MET | no BLOCK | not approved | **MET-PROPOSER**: system not tested |
| MET | BLOCK | not run | **MET-PROPOSER, RETENTION BLOCK** |
| PARTIAL | — | not run | **PARTIAL** |
| NO | — | not run | **NO** |
| (§3.5) | — | — | **TEACHER-FUTILITY** |
| (§9) | — | — | **VOID** / **INCOMPLETE** |

Every MET line carries the B3-bar line and the A2 status (REPLICATES / SAME SIGN / DOES NOT REPLICATE / not run).

### 5.8 Adoption rule: does B6 change the default system?

- Release system = FREEZE_B2_beam in `release/…-0.1.0-devfreeze` (Pa_v2_s1). Default BWM system for later work = the
  FREEZE_B3_H2 configuration (Pa_v3_s1).
- **ADOPT** Pa_v4_s1 as the default BWM proposer (write `runs/FREEZE_B6_beam.json` = FREEZE_B3_H2 args with pa →
  Pa_v4_s1, plus a DECISIONS entry) iff all hold: (1) P1 MET; (2) R no BLOCK; (3) P2 MET; (4) every plan Pa_v4's beam
  verified on b6beam is valid under the exact executor and VAL (else hold pending review); (5) no open VOID or fault;
  (6) if the optional Pa_v3 arm ran, v4-only ≥ v3-only. Otherwise **KEEP INCUMBENT**.
- Without the optional arm, adoption rests on beating the matched control (Pa_v3 + replay training); P-11 found the
  extra-training control alone did nothing at S-direct (+6/−3), and S3 (§5.9) reports v3c vs v3 on b6test. Adoption
  text must say v4 vs the incumbent was not tested at system level.
- Adoption never changes the release package (D-65.1), never runs hard or PB-602 in the beam, never edits any FREEZE_B2_*
  or FREEZE_B3_* file (P-18 rule 3), and never changes B4's registered arms.

### 5.9 Secondary analyses (report-only, unadjusted, labelled)

- **S1.** P1 by block size and the 13-15 stratum.
- **S2.** M256-derived reading for v4 and v3c on b6test (success at M512 with gen_tokens ≤ 256 [VERIFY the boundary
  against `lm.sample_crn_shared`'s `n_tokens`/`hit_eos` (lm.py:571-647) and bw_b3_eval.py:83]), with its own B3-bar
  line. Batch size depends on max_new (lm.py:531), so this rests on canary (ii). Reported only if canary (ii)
  passes.
- **S3.** v3c vs v3 on b6test (extra replay training alone). **S4.** v4 vs v3 on b6test.
- **S5.** Truncation at 512, unparseable and illegal-step rates, solved-plan length and its ratio to 2 × must_move.
- **S6. Retention:** b6ret (all three pairings); PB-602 at 3-5 blocks, v4 vs v3c only; the 6-10 strata of b3fresh and
  mirror (from S7 cells). Descriptive; "retention loss" only via R's BLOCK rule.
- **S7. Cross-round ladder:** v4 and v3c S-direct-512 on b3fresh and mirror; their M256-derived readings read against
  the P-11 cells (v2, v2c, v3) as v2 → v2c → v3 → v3c → v4, all 110 and the 6-10 / 11-15 strata. Requires canaries
  (i) and (ii).
- **Canary** (before any P-11 cell is reused): (i) Pa_v3 at M256 re-decodes the 16 b3fresh ids ≡ 0 (mod 7); plan and
  gen_tokens must equal `runs/BW_B3_EVAL/plans_v3_b3fresh.jsonl`. (ii) Pa_v3 at M512 on the same ids must equal (i) on
  every id not truncated in (i) (batch invariance across max_new, D-48c). On a failure of (i) the P-11 cells are not
  reused; on a failure of (ii) S2 and S7 are not reported. The primary is unaffected (both its arms are new).
- **S8. Teacher:** verified, exact-valid, VAL-valid by size; label precision and the exactly-invalid share; rounds,
  expansions, proposer tokens, unique F keys (cache-counted); fail type; share stalled at u*_F = 1; exact-state
  mismatches; loop share; 2 × must_move for verified vs unverified (yield selection bias); legacy-invariant rejection
  share; malformed-state rows.
- **S9. Rows and mixes:** rows, filter and overlength drops by source, supervised and prompt tokens per arm, Rep/S
  sources and top-up count, failure rows per arm (equal by construction), block range.
- **S10. FREPLAY** R1/R2 on the teacher's F cache (P-15 method; `freplay.py` classifier imported unchanged), with CIs.
- **S11. Training:** loss curves, wall, tokens/s, peak memory (incl. the memory smoke).
- **S12. Disclosure:** goal-tower-height bins of b6test and b6beam vs hard 11-15 (P-10 method, instance files only).
- **S13. Stage B mechanism and cost:** size strata; solved within 2/4/6/8 rounds; budget vs dead end; best-node u*;
  share of failures stalled at one unmet fact; trap/escape-bound breakdown (one unmet `on(x, y)` with y clear and x
  under k blocks, escape bound 2(k+1) + 2r; exact simulator on outputs, analysis only); proposer tokens and unique F keys
  per instance from caches; FREPLAY R1/R2 on the new keys; verified ⇒ valid count.
- **S14.** A2 seed spread. **S15.** V5 sensitivity (all-instance numbers beside any registered exclusion).

## 6. Power

Exact trinomial enumeration with the exact two-sided McNemar test, one direction counted (computed for this draft;
reproduces the judges' and the rigor design's figures at n = 150).

**P1, S-direct, b6test.** P(p < .05 with b > c) / P(MET under §5.2) / P(MET under the B3 bar):

| True effect (p10 / p01) | Anchor | n = 150 | **n = 200** |
|---|---|---|---|
| Null (.05/.05 … .12/.12) | — | ≤ .017 / ≤ .017 / ≤ .008 | **≤ .018 / ≤ .018 / ≤ .002** |
| +5 pp (.08/.03) | — | .37 / .37 / .04 | **.50 / .48 / .02** |
| **+8 pp (.095/.014)** | round-1 11-15 (+7/−1 of 74) | .85 / .85 / .27 | **.95 / .93 / .23** |
| +8 pp, higher discordance (.14/.06) | M512 raises the control's rate | .52 / .52 / .32 | **.68 / .68 / .29** |
| +10 pp (.13/.03) | B3 bar | .86 / .86 / .54 | **.94 / .94 / .53** |
| +10 pp, higher discordance (.17/.07) | | .66 / .66 / .53 | **.80 / .80 / .53** |
| +12.7 pp (.164/.036) | round-1 overall (P-11) | .94 / .94 / .82 | **.99 / .99 / .84** |
| +15 pp (.18/.03) | | .99 / .99 / .94 | **1.00 / 1.00 / .96** |

n = 200 (not 150) because realistic M512 discordance (.14/.06) leaves n = 150 at 0.52; the extra 50 instances cost
about 0.7 GPU-h over three arms.

**Retention guard R** (b6ret, n = 100, v4 vs v3): P(BLOCK) null .011-.015; true −5 pp .18; −10 pp .54; −15 pp .92.
It is a guard, not a non-inferiority test.

**P2, two-stage S-beam** (n1 = 48, n = 96, futility at b1 − c1 ≤ +1):

| True effect (p10 / p01) | P(MET) | P(stop at interim) | E[n] |
|---|---|---|---|
| Null (.05/.05 … .125/.125) | ≤ .014 | .67-.76 | 60-64 |
| +8 pp (.14/.06) | .33 | .22 | 85 |
| +10 pp (.15/.05) | .51 | .14 | 90 |
| +12.5 pp (.175/.05) | .68 | .08 | 92 |
| **+15 pp (.20/.05), minimum effect** | **.81** | .04 | 94 |
| H2-like at 12-15 (.226/.065) | .80 | .04 | 94 |
| +15 pp, higher discordance (.23/.08) | .70 | .06 | 93 |

(n = 80 with interim 40: .73 at +15 pp; n = 100: .83.) Stage B cannot reliably detect system gains below about
+12 pp; a non-MET P2 is **not** evidence of no system gain.

**A2.** If the seed-2 contrast were independent of seed 1, P(both MET) ≈ .86 at (.095/.014) and ≈ .46 at (.14/.06)
(approximation; the arms share data).

**Why no hard run is pre-authorised.** A once-only Pa_v4 hard beam against H2's per-instance results (88/110; at most
22 movable) has power .18 at round-1 relative rates (conversion .54, loss .068), .33 at (.40, .03) and .68 at
(.55, .03); it would be hard's fourth exposure. Any hard run is a separate user decision after the report.

## 7. Multiplicity

- **Confirmatory family {P1, P2} in fixed sequence**, each at α = 0.05: P2 is tested only if P1 is MET (and R does not
  BLOCK). FWER ≤ 0.05 without splitting α. The Stage B interim is futility-only and non-binding.
- R can only block; A2 is a conjunction requirement for the recipe claim; neither inflates FWER.
- Everything else (S1-S15, strata, the B3-bar line, M256-derived readings, ladder) is report-only, unadjusted,
  labelled, never claimed. No adjustment across the B3 and B6 families.
- B6 has no selection set and chooses no setting; only `est_s` values are operational.

## 8. Schedule and cost

A turn is 16,200 s (bundle.py:1-13). A non-first sub-job starts only if `est_s` ≤ remaining (bundle.py:39-48). All long
jobs are **deadline-aware**: they read `BWM_LEASE_ACQUIRED`, start a chunk, cell or shard only if 1.3 × the largest unit
time seen so far (or the unit's est) fits, and otherwise exit **75**; they exit 0 (letting bundle.py write the marker,
:69-71) only when fully done. bundle.py returns a sub-job's non-zero rc as its own [VERIFY that the orchestrator requeues
rc 75 from a sub-job without counting an attempt, QP-3]. Every job after `b6-rows` first checks for
`TEACHER_FUTILITY.json` and, if present, writes its marker as skipped and exits 0.

**Cost estimates** (point; range ±40-50%):

| Item | Basis | GPU-h |
|---|---|---|
| Memory smoke | 3 steps at 768 | 0.1 |
| Search, 160 at K4 | r1 77 s per 6-10 problem × 2.43 (H2's 11-15/6-10 token ratio) = 3.1 min, raised for more failures on generated sets (B2 mirror 11-15 30/74 vs hard 41/74; failures ≈ 14.8k vs 8.4k tokens per solve at K8) → **3.5-3.8 min/problem**, ≈ 37 min per chunk of 10 | **9.7** (7-14) |
| rows, filters, audit, mix, classes | CPU inside the bundle | 0.25 |
| Train Pa_v4 | ≈ 1.55-1.6M tokens at ≈ 250 tok/s (Pa_v3: 1,272,786 tokens / 5,099 s) | 1.8 |
| Train Pa_v3c | ≈ 1.27M tokens | 1.4 |
| Direct core: v4, v3c × b6test-200; v4, v3, v3c × b6ret-100 | ≈ 16 s per 11-15 instance at M512, ≈ 6 s at 6-10 [VERIFY]; B3 cell v3/b3fresh 950 s per 110 at M256 | 2.3 |
| Direct report-only: canary; v3 × b6test; v4, v3c × {b3fresh, mirror}; v4, v3c × PB-602 | B2 PB-602 direct 418 s | 2.4 |
| **Stage A total** | | **≈ 18 (13-25) ≈ 4 turns of work; plan 5 BWM turns** |
| A2 (iff P1 MET) | 2 trainings + 2 b6test cells | ≈ 5.0 (≈ 1.1-1.5 turns) |
| Stage B interim (48 × 2 arms) | 35 s per 1k proposer tokens (median of 36 shards); ≈ 11k tokens per 12-15 instance at K8 → ≈ 6.5 min [VERIFY] | ≈ 10.4 (≈ 2.5 turns) |
| Stage B full (96 × 2 arms) | same | ≈ 20.8 (≈ 5 turns); +50% with the optional Pa_v3 arm |

**Totals.** Null at P1: ≈ 5 BWM turns. P1 MET, Stage B not approved: ≈ 6-6.5. Stage B approved: +2.5-3 (futility stop;
67-76% under the null) or +5-6 (full). For comparison, B3 took ≈ 19 GPU-h.

**bundleK** (`runs/bundles/bundleK.json`, budget_s 16,200, queued `after: runs/bundles/bundleJ.done` and after the
user's placement decision §16.3):

| Job | est_s | Marker / notes |
|---|---|---|
| `b6-memsmoke` | 900 | `runs/BW_DISTILL_r2/MEMSMOKE.json`; scratch output only |
| `b6-search-00` … `b6-search-15` | 2,800 each | one job per chunk of 10; `chunk_k.done`; deadline-aware; chunk log |
| `b6-rows` | 900 | rows, filters, audit, futility check, mixes, classes; marker `MIX_META.json` (or futility) |
| `b6-train-v4` | 7,800 | resumable from 50-step checkpoints |
| `b6-train-v3c` | 5,800 | resumable |
| `b6-freeze` | 60 | `FREEZE_B6.json` (refuses overwrite) |
| `b6-direct-core` | 3,600 | cells (v4,b6test), (v3c,b6test), (v4,b6ret), (v3,b6ret), (v3c,b6ret); cell-adaptive |
| `b6-analyze-a` | 120 | `B6A_REPORT.json`, gate files (§5.4) |
| `b6-direct-report` | 3,600 | canary → (v3,b6test) → ladder (4 cells) → PB-602 (2 cells) |
| `b6-a2-train-v4s2`, `b6-a2-train-v3cs2` | 7,800 / 5,800 | run iff `P1_MET.json`; else skipped marker |
| `b6-a2-freeze`, `b6-a2-direct`, `b6-analyze-a2` | 60 / 3,600 / 60 | same gate |
| `b6-freplay`, `b6-report-a` | 300 / 60 | CPU |

bundleK2 (Stage B: `b6b-freeze`, 24 shard jobs at est_s 4,000, `b6b-interim`, `b6b-analyze`) is written now but queued
only after the user's approval is recorded.

**Turn plan (central).** T1: memory smoke, chunks 00-06. T2: chunks 07-13. T3: chunks 14-15, rows/audit/mix, Pa_v4
(Pa_v3c does not fit). T4: Pa_v3c, freeze, direct core, P1/R reading and gates. T5: report-only cells; A2 trainings if
P1 MET. T6: A2 cells and analysis; Stage A report; Stage B approval request.

**Calendar** [VERIFY at queue time]. About 2 BWM turns per day under strict alternation and the D-84 windows (no new
bundle 23:00-07:00 through 10-15 07:00, possibly extended). The B4 screen ends about 10:30 on 10-13 and RCWM's turn
follows; the CPU prework (§11) will not be done for the 10-13 afternoon turn. Realistic T1 = **10-14 07:00**; P1 reading
about 10-15 evening to 10-16; Stage A report about 10-16; A2 about 10-17. Add the full B4 if the user places it first.

**Memory/RAM.** GPU peak ≈ 3-3.6 GiB (measured by the smoke). A GPU bundle leaves ≈ 10 GB host RAM (D-84); all CPU
stages are under 1 GB on one lane, and CPU jobs over 5 minutes take the shared lane lock (D-76b).

## 9. Void, fault and stopping rules

- **V1 Timing.** This prereg is hashed into DECISIONS before `sets` runs; sets and problems are hashed before any GPU
  job; `FREEZE_B6_teacher.json` precedes any search output; no b6test/b6ret output before `FREEZE_B6.json`; no b6beam
  read except by `sets`/`audit` before `FREEZE_B6B.json`. A violation voids the affected readings.
- **V2 Contamination before the freeze.** An overlap between a B6 test set and U, or a non-zero `OVERLAP_AUDIT`, means
  a generator or filter bug: void and regenerate the affected sets with 20261425+ (recorded before any model output on
  them); an overlap involving D2 is removed by the filter and must audit to 0 before training.
- **V3 Teacher futility** (§3.5): TEACHER-FUTILITY; nothing trained or tested.
- **V4 Instrument faults.** Any sha mismatch makes the runner refuse with no output. The parity tests (beam path vs
  blocks_eval; canon vs `B.signature` and method B) must pass before the freezes. A torch version difference from the
  teacher run is report-only.
- **V5 Overlap found after the freeze** (including late RCX lists). The registered reading of that test excludes the
  overlapping instances; the all-instance numbers are reported beside it. If more than 5% of a set overlaps, that
  test's reading is VOID.
- **V6 Training faults.** Resume from the last checkpoint. A non-finite loss gets one full re-run with identical args
  (recorded); a second non-finite loss or second crash of the same run voids Stage A (or A2).
- **V7 Eval faults.** A cell killed at the cap re-runs from scratch next turn (greedy, deterministic). A completed cell
  is never re-run. A beam shard restarts from its cache; restarts logged. A missing primary cell after two failures:
  INCOMPLETE, no reading, no imputation.
- **V8 Gate tampering.** A hand-written or edited gate file (`P1_MET`, `B6A_GO`, interim files) voids everything
  downstream.
- **V9 Peeking.** The only analyses are `analyze-a`, `analyze-a2`, `analyze-b --interim`, `analyze-b`. No interim in
  Stage A; one futility look in Stage B. No sample size changes after the freeze except the outcome-independent search
  cap (§3.1).
- **V10 Wording.** "No simulator labels" only at label precision 1.000, otherwise report the wrong-label share; the
  FREPLAY scope wording is mandatory.
- **After the freeze**, any deviation is an erratum and makes the affected result post hoc.

## 10. Code, pins and freeze records

**Pins (verified 2026-10-13 07:55 CDT; re-checked at freeze). No pinned file is edited.**

| File | sha256 |
|---|---|
| `bwmn/blocks_planner.py` | `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960` (D-63; notify RCWM and RCX before any change; none planned) |
| `bwmn/lm.py` | `88a1e3a26b518cebd17e0b8197d1dd10872d549a4a27a69744ed59266895c0ac` |
| `bwmn/blocks.py` | `b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83` |
| `bwmn/__init__.py` | `e3b0c442…` (empty) |
| `scripts/blocks_eval.py` | `b3666424151c78368500535d2571f0eb5b9e546b11764696d99f427a7794acb6` (B4-screen pin; imported read-only for `_json_set`, `load_set`; no new `TEST_SETS` entry) |
| `scripts/bw_run_tests.py`, `scripts/bw_seeds.py` | `de7022d1…`, `b783a191…` (B4-screen pins; not touched) |
| `scripts/train_generic.py` | `96057cdfb0fd27516e51d54713db64af2d23cf9ac9634d3c8a6b3053e8ccfb6e` |
| `scripts/bundle.py` | `5b1d299289c14796b464d8402b75d443b4af0dd4954f15d4a5107745faf3b51f` |
| `scripts/bw_distill.py`, `scripts/bw_b3_prep.py` | `2b2d5182…`, `3828130c…` (hardcoded seeds/exclusions, imported elsewhere; not edited) |
| `runs/FREEZE_B3_H2.json` / `FREEZE_B2_beam.json` / `FREEZE_B2_direct.json` | `a433e499…` / `412a6ee5…` / `167276d3…` |
| Adapters Pa_v3_s1 / Fdelta_s0 | `48c9d279…` / `2ed0bb00…` |
| New code (bw_b6.py, bw_canon.py, bw_b6_audit.py, tests) | [FILL AT FREEZE] |

**New files only:** `scripts/bw_canon.py`, `scripts/bw_b6_audit.py`, `scripts/bw_b6.py` (stages: `universe`, `sets`,
`teacher-freeze`, `search`, `rows`, `mix`, `classes`, `audit`, `freeze`, `run-direct`, `analyze-a`, `a2-freeze`,
`analyze-a2`, `freeze-b`, `run-beam`, `analyze-b`), `tests/test_bw_canon.py`, `tests/test_bw_b6.py`, and the bundle
files.

**Stub tests (CPU)** in `tests/test_bw_b6.py`: exact F + scripted proposer smoke of search and rows on seed 20261419 in a
scratch root, with its artifact saved (E-01(h)); only verified plans kept; row states equal F states; per-chunk resume
and the exit-75 path; futility path; mix invariants (indices, identical Rep positions, equal failure rows, contingency
branches N > 8,508 and S top-up); guards (overwrite, sha mismatch, re-run, missing gate file, hand-written gate);
run-beam parity with blocks_eval; `analyze` reproducing P-11 from its plan files (22 vs 8, +18/−4, p 0.0043) and every
row of the §5.7 table on synthetic plan files.

**Freeze records:** `FREEZE_B6_teacher.json`, `FREEZE_B6.json`, `FREEZE_B6_A2.json`, `FREEZE_B6B.json`, and
`FREEZE_B6_beam.json` only on ADOPT.

## 11. CPU work before the freeze, in order (no GPU)

1. Archive P-18's scratch audit code to `research/audit_evidence/bwm_p18_scratch/` with shas.
2. Promote and test `bw_canon.py` (totality fix, tests (a)-(f)) and `bw_b6_audit.py`.
3. Write `bw_b6.py` and the stub tests; run the token audit (`TOKEN_AUDIT.json`).
4. Coordinate: RCX's written agreement to §4.6; request E07b/E08; notify RCWM.
5. DECISIONS entry: seeds 20261419-29 consumed by B6 (panel proposals 20261420-23 marked consumed); the exclusion
   universe manifest (paths, shas).
6. Settle the §16 decisions with the user; finalise this text (drop DRAFT, fill [FILL AT FREEZE] items that exist by
   then); hash it into DECISIONS.
7. `bw_b6.py universe` → `sets` (b6test → b6beam → b6ret → b6search) → `audit` on the sets (expect 0) → hash into
   DECISIONS → send lists to RCX and RCWM (exclusion only).
8. Write `FREEZE_B6_teacher.json`; write bundleK and bundleK2; queue bundleK only.

## 12. Outputs

- **Records:** `PREREG_B6_DISTILL_R2.md`; DECISIONS entries (seeds and manifest, prereg hash, set hashes, results,
  approvals); `runs/bundles/bundleK.json`, `bundleK2.json`.
- **Sets:** `runs/BW_val_large/instances_b6_{test,beam,ret}.json`; `runs/BW_DISTILL_r2/{problems.json,
  EXCLUSION_UNIVERSE.json, SETS_META.json}` (seeds, counts, rejected draws by reason, legacy-invariant share).
- **Teacher and data:** `runs/FREEZE_B6_teacher.json`; `runs/BW_DISTILL_r2/{solutions.jsonl, chunk_log.jsonl,
  fcache.jsonl, distill_rows.jsonl, ROW_FILTER.json, OVERLAP_AUDIT.json, DISTILL_META.json, TOKEN_AUDIT.json,
  MEMSMOKE.json, mix_v4.jsonl, mix_v3c.jsonl, MIX_META.json, TRAIN_CLASSES_r2.json, FREPLAY_TEACHER.json}` (or
  `TEACHER_FUTILITY.json`).
- **Adapters:** `runs/BW_Pa_v4_s1`, `runs/BW_Pa_v3c_s1` (+ `_s2` on A2), each with `metrics.json`.
- **Stage A:** `runs/FREEZE_B6.json`; `runs/BW_B6_EVAL/{results.jsonl, plans_<arm>_<set>.jsonl, CANARY.json,
  B6A_REPORT.json, P1_MET|P1_NOT_MET.json, B6A_GO|B6A_NO.json}`; `FREEZE_B6_A2.json`, `B6A2_REPORT.json`.
- **Stage B:** `runs/FREEZE_B6B.json`; `runs/BW_B6_BEAM/<arm>/shard_k/{plans.jsonl, summary.json, fcache.jsonl,
  progress.txt}`; `B6B_INTERIM.json`, `B6B_REPORT.json`; `runs/FREEZE_B6_beam.json` (only on ADOPT).
- **Summary:** `runs/BW_B6_EVAL/B6_REPORT.md`; backups and export after the report (D-73 practice).

## 13. Allowed claims and wording

- **P1 MET:** "Continued on generated 11-15-block plans that its own learned world model verified (verified by the
  learned F plus a domain-agnostic delta-consistency check; no simulator labels [only if label precision = 1.000]),
  the proposer alone solves X/200 fresh generated 11-15-block hard-style problems vs Y/200 for the same extra training
  on its earlier training data (paired +b/−c, exact McNemar p = …; B6 rule p < 0.05 and ≥ +5 pp). B3-bar line: [met /
  not met]." Add "replicated with a second training seed (…)" only on REPLICATES; otherwise "one training seed".
- **P2 MET (added):** "…and inside the unchanged frozen search with only the proposer swapped, X/96 vs Y/96 fresh
  generated 12-15-block problems against the same matched control (paired, p = …)."
- **Not allowed:** "met the B3 bar" unless the B3-bar line holds; "simulator-free" for the student or the system; "the
  model's own search" without the F-verification qualifier; any claim from a mix, step, adapter or setting chosen after
  seeing B6 outputs; any hard-set or published-model comparison ("beats").

## 14. Not claimed

- Seed robustness of the recipe unless A2 REPLICATES (then two seeds only); seed robustness of any B6 system.
- Robustness of labels to the F seed (teacher and verifier are Fdelta_s0 only; P-17).
- That search-found content, rather than exposure to 11-15 prompts with long valid targets or more supervised tokens,
  causes the effect (no scripted-plan exposure control; token gap reported).
- That S-direct gains imply system gains (that is P2), or "no system gain" from a non-MET P2 (underpowered below about
  +12 pp).
- Attribution of round 1's H2 gain (P-14's confound concerns Pa_v3 vs Pa_v2 and stays open; P2 is v4 vs v3c).
- System-level superiority over the incumbent Pa_v3 unless the optional Pa_v3 beam arm runs.
- Hard, PB-602 beam or any published-model comparison; release changes.
- Generality beyond this generator (tall-tower full goals, names, training format; F-08), beyond 15 blocks, or beyond
  Blocksworld; non-inferiority at 3-10 blocks (retention is descriptive plus the R guard).
- Any fix of the one-unmet-fact ranking minima (B4's question); compute-matched comparisons; optimality of 400 steps.

## 15. Options considered and rejected

| Option | Why not |
|---|---|
| Faithful B3 copy with M256 primary on 11-15 | Censors exactly where the treatment acts (25/74 Pa_v3 truncations at 11-15; teacher's 15-block solves ≈ 51 actions vs ≈ 58-action cap); P(MET) .31 at the round-1 out-of-range analog |
| Size cycle with 60 problems at 15 blocks | ≈ 1 search turn for ≈ 12 plans (15-block yield ≈ 15-25%) |
| Misplaced-blocks pass-B teacher (`unmet=` hook) | It is B4 arm a0 (tier B, hand-written heuristic): turns B6 into heuristic-assisted distillation, pre-empts the co-signed B4, cannot separate passes |
| F-state loop cut; ×2 row weighting | New label transforms on top of the round-2 change; unnecessary at M512 / max-len 768 |
| K8 or E48 teacher | ≈ 2× cost for ≈ +2-3 pp yield (R-05) |
| 240 search problems | +4 GPU-h; 160 gives ≈ 2.7-3.7k rows |
| B3's observed ≥ +10 pp bar as the rule | P(MET) ≈ .5 at a true +10 pp at any n; kept as the named B3-bar line (decision 1) |
| b6test n = 150 | Power .52 at realistic discordance (.14/.06) |
| Stage B vs the incumbent Pa_v3 only | Leaves the extra-training confound open at system level; v3c is the registered comparator, v3 is an optional report-only arm |
| b6beam including 11 blocks | Near ceiling (H2 12/12, B2 14/16 on generated sets); ≈ 20% of Stage B GPU for concordant pairs |
| Reusing b3fresh, mirror or seeds_fresh as primary | Already exposed and informed this design |
| val_v2 retention | 303/400 leaky (P-18) |
| `VL.invariant` exclusion | Coarse; distorts fresh sets; exact classes are the criterion (legacy share reported) |
| Token-matched control; exposure (scripted-plan) control; unconditional seed-2 copies | Each +0.4-1.3 turns; scoped in §14 |
| Shared F cache across beam arms | Small saving; muddies per-arm cost accounting |
| Single-stage beam | The interim saves ≈ 1.3-2 turns on average under the null for ≤ .01 power |

## 16. Decisions for the user before the hash

1. **P1 decision rule.** Registered draft rule: p < 0.05 and ≥ +5 pp (power .93 at the minimum effect, .68 at higher
   discordance), with the B3 bar as a mandatory named line. Alternative: keep the B3 bar as the rule (power .23-.53).
   This changes a convention the user approved (D-58), so it needs explicit sign-off.
2. **Stage B scope** (asked now so its design is fixed before outcomes; running it still needs approval later): n = 96
   (power .81 at +15 pp; ≈ 2.5 turns interim, ≈ 5 full) or n = 80 (.73; ≈ 2 / ≈ 4.2); with or without the optional
   report-only Pa_v3 arm (+50%).
3. **Placement:** where B6 goes relative to the full B4 if the B4 screen reads GO (≈ 7 turns), and whether B6's Stage B
   or A2 may delay item 9.
4. **A2 automatic on P1 MET** (≈ 1.1-1.5 turns; draft default yes) or asked after the P1 reading.
5. **PB-602 report-only cells** (≈ 0.3 GPU-h; a further exposure of a registered B2 test set to new adapters; draft
   default include).
6. Confirm no hard run is pre-authorised (§6); any hard run is asked after the report.

## 17. How this draft was built (synthesis record)

- **Inputs:** a technical map of the B3 pipeline and three designs (faithful B3 replication, maximum-effect, rigour and
  budget), scored by two independent judges.
- **Base design: rigour.** Both judges chose it (totals 40 and 38 vs faithful 32/32 and effect 30/25), and neither
  found a fatal flaw in it. Faithful's M256 primary censors the target stratum; effect's pass-B misplaced key
  duplicates B4 arm a0.
- **Grafts adopted:** P2 comparator Pa_v3c (attribution-clean; both judges) with Pa_v3 as an optional report-only arm;
  b6beam restricted to 12-15 (judge 2); `TRAIN_CLASSES_r2.json` (both); V5 late-overlap rule, the reading table and the
  V8 gate-tampering rule (judge 1); u*, exact-state mismatches and the trap/escape-bound breakdown (judge 1); PB-602
  and v4-vs-v3c retention reports (both); deadline-aware exit-75 jobs (judge 2); P-11 ladder on b3fresh and mirror,
  the determinism canary and the M256-vs-M512 check (both); eval_seed 7 stated, per-chunk wall and cache-counted F keys
  (both); hard-run power as the reason none is pre-authorised (judge 1).
- **Must-fix items and where they are addressed:** P1 rule sign-off, B3-bar line and disclosure (§5.2, §16.1);
  Stage B comparator (§5.6); P-18 code archive and canonicaliser promotion and tests (§4.1, §11); canonicaliser
  totality (§4.1, §3.4); peak-memory smoke at 768 (§3.7); max-len from the token audit (§3.4); seeds 20261419-29 and
  panel seeds consumed (§4.3, §11); B4 cross-exclusion with RCX agreement and user placement (§4.6, §16.3); per-chunk
  search jobs, chunk log, index-prefix cap, legacy-invariant share (§3.1, §8); realistic power, b6test raised to 200
  (§6); corrected cost (3.5-3.8 min/problem) and calendar (T1 10-14 07:00) (§8); separate `P1_MET.json` trigger for A2
  (§5.4); retention also vs v3c, BLOCK defined vs v3 only (§5.3).
- **Not adopted:** faithful's M256 primary, size cycle and H2b; effect's misplaced pass B, loop cut, ×2 weighting,
  automatic beam stage, and b6beam as a subset of the primary set.
