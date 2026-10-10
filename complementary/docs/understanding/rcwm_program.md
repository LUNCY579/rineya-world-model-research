# RCWM (Rineya Consequence World Model) research report, as of 2026-10-09 22:20 CDT

**Paths used below.**
- **B** = `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1`. This is the live working copy and it is newer than the repo.
- **R** = `C:/Users/clclc/BusinessProject/WorldModel/rineya-world-model-research/rcwm/experiments/RCWM-T1`. This is the repo mirror on `main`, HEAD `a223884`, about 12:34 on 10-09.

**Labels.** [M] = measured or recorded in files. [P] = plan or proposal. [I] = my inference.

## 0. Repo vs local bundle: the bundle is newer

The repo DECISIONS.md has 665 lines and ends at D-37. The bundle DECISIONS.md has 713 lines and was last modified at 22:06. The bundle adds these entries:
- **R17 (12:09):** retention dev selection.
- **D-38 (about 19:10):** the 5a prerequisite screen rule is frozen and hook `scripts/queue_hooks/75_t2bp.sh` is installed.
- **I-31:** a second user restart at 18:38, recovered at 22:02. BWM's stale lock was cleared at 22:05:27 on the user's instruction.

Files in the bundle modified within the last 24 hours that are not in the repo (or newer there):
- `docs/ledger_beam/PREREG_LB_DRAFT.md` (17:41), plus `LB_DRYRUN.*`, `LB_POWER.json`, `GEN_PLAN_LB.json` and `RUN_TEST_*.json`
- `scripts/lbeam_run.py` (22:13), `lbeam_analysis.py` (22:12), `lbeam_test_run.py` (22:20), and `docs/ledger_beam/queue_hook_lbeam.sh.draft` (22:13)
- revision 4 of the 5a kit: `t2bp_*.py` (17:24–17:40), `docs/t2b/prereq/{DESIGN,SCREEN_RULE,DRYRUN,TINY_TEST,AUDIT*}`
- revision 4 of the 5b kit: `t2br_*.py`, `docs/t2b/register/*`, rebuilt `data/t2br_*` (17:38), and `data/t2br_superseded/rev3/`
- `runs/bank_pilot/sparse/lik_rank.jsonl` (17:50)
- `runs/thermal_log.txt`, `runs/bank_pilot/master_log.txt`

`scripts/t2br_common.py` was modified at 22:20. [I] A CPU workflow is editing the register kit right now: per I-31, the 5b identification fix wf_84e8f08d-166 and the PREREG-LB revise step both resumed at about 22:05.

## 1. RCWM environments, oracles and data formats

**Core module: `B/rcwmt/env.py`.** It imports only the standard library, so it can be imported without torch.

Families, in `FAMILIES` order:

| Family | State | Event | Dynamics |
|---|---|---|---|
| `ledger_replay` (L) | `{"A","B","C","D"}` integer balances | `{account, amount, kind∈{add,scale,transfer}, other, status, tick}` | Applied only if `status == "posted"` (exact string), otherwise a no-op. add: `+amt`. scale: `*amt`. transfer: `a -= amt; other += amt`. |
| `prerequisite_plan` (P) | 8-bit int mask (0..255) | `{add, cost, delete, kind:"action", name, pre_off, pre_on}` | Applicable iff `(s & pre_on) == pre_on and (s & pre_off) == 0`, then `(s & ~delete) | add`; otherwise a no-op. |
| `register_machine` (R) | `{"X","Y","Z"}` mod 97 | `{kind∈{swap,add,mix,conditional}, other, register, threshold, tick, value}` | mix: `r + 2·o + v`. conditional: `+v` if `reg[o] > thr` (strict), else `−v`. All `% 97`. |

**Key functions.**
- Dynamics: `step(fam, state, ev)`, `rollout(fam, s0, events)` and `applicable()`.
- Features: `features(fam, states, events)`, plus `bits()`.
- Generators: `gen_start`, `gen_event` and `gen_action_library(rng, m, n_actions=6)` for P.
- Prompts: `one_step_messages(fam, before, ev)` and `multi_step_messages(fam, before, events)`.
- Parsing: `parse_state(fam, text, lenient=False)`. With `lenient=True` it unwraps `{"after":…}`, rewrites `a op b = c`, and falls back to the last complete `after`.
- Serialisation: `dumps()` is compact JSON with `sort_keys=True`.
- System texts: `SYSTEM_ONE` (Titan's verbatim one-step frame) and `SYSTEM_MULTI` (the "in order" variant, which states the P no-op rule).

**Oracle path.**
- `env.cross_check(fam, s, ev)` compares `env.step` against `fds_oracles()`.
- `fds_oracles()` imports `oracles.ORACLES` from the hard-coded path `C:/Users/clclc/BusinessProject/ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1/oracles.py`, which was validated 384/384 against the recorded references.
- `env.load_marginals()` reads `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/process-transition-probe-001/CANDIDATE_VIEWS.jsonl` and checks its sha (`45674251…`).
- PREREG-3 adds an independent `scripts/p3_common.rule_gold()`. Its gold is `env.step == rule_gold == FDS oracle`.

**Prompt rendering.**
- Rendering uses the Qwen3 chat template with `enable_thinking=False`. The rendered string ends `<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- The user turn is the compact sorted JSON `{"before":…,"event":…}` (one-step) or `{"before":…,"events":[…]}` (multi-step).
- Decoding is greedy, with EOS [151645, 151643] and PAD 151643. The cap is 160 new tokens in PREREG-2/3.
- `rcwmt/generation.generate_texts` batches with left padding, at most 32 prompts and batch × (prompt + new) ≤ 8,000 tokens.
- `generation.chain(model, tok, chains, mode)` runs self-rollouts in two modes: `chain_one` (Titan's frame) and `chain_m1` (the multi-step rule with one event).

**Other `rcwmt` modules.**
- `bank.py`: parents × K plans; split by salted SHA-256 (`"RCWM-T1-parent-split-v1"`, 70/15/15); `load_exclusions()`.
- `data.py`: `phi_from_state`, `PHI_SPEC`. φ is L = sign_neg×4 + changed×4 + clip(final/100, ±3)×4; R = hi>48, parity, changed (3 each); P = 8 final bits.
- `decide.py`: goals, `regret_table`, `paired_bootstrap`, `symbolic()`.
- `decisions.py`: `g_only`; 256 test goals, seed 1001.
- Also `fidelity.py`, `controls.py`, `heads.py`, `capture.py` (layers 0, 4, …, 28) and `pvr.py`.

**Dataset formats [M].**
- **`data/bank_pilot.json`**
  - Top-level keys: `{schema, bank, seed 20261005, n_parents_per_family 200, k_candidates 8, horizon 4, n_actions 6, …, parents[600]}`.
  - Each parent is `{parent_id, family, split, s0, library, candidates:[{events, cand_id}]}`.
- **`data/bank_confirm.json` and `bank_confirm2.json`:** sealed confirmation banks, 150 parents per family.
- **`data/t2_train.jsonl`:** 1,501 rows, each `{prompt (fully rendered chat string), target, family, stratum, frame, before, event}`. This is what the vendored BWM trainer consumes.
- **`data/bank_p3_oor.jsonl`** (sha `0fd18295…`): 1,500 one-step rows plus 80 chains. Rows are `{id, kind, stratum, group, base, before, event, key_order, messages, gold, gold_rule, posted_gold, meta}`.
- **Ledger beam sets:** `docs/ledger_beam/dev2_primary.jsonl` (120 instances) and `dev2_oor.jsonl` (30).

**Spent or forbidden instruments.**
- Never open `HSKILL_CONFIRM_V2*`. Builders skip it by name (I-28).
- Spent: pilot TEST, `t2_eval`/supp, R14's 248 off-task prompts, L1's 456 prompts and its OOD bank, SELECT-VIEWS-001 (spent for the L8-15 comparison), and ledger-beam dev seeds 20261141–43.

## 2. PREREG-3: "rule or trigger" (frozen 02:38:31, run 06:34–07:19 on 10-09)

**Hypothesis [M].**
- PREREG-2's T2 LoRA took ledger void accuracy from 0.04–0.13 to 1.00. But T2's 501 ledger rows contained only statuses posted 375 / void 126, amounts −9..12, |before| ≤ 98, accounts A–D and sorted keys.
- Four readings were registered:
  - RULE: "If status is not posted, leave balances unchanged" at any magnitude, naming or key order.
  - A semantic cue.
  - A "contains posted" substring cue.
  - TRIGGER: the literal token `void`, with in-range arithmetic only.

**Design [M].** `PREREG_3.md`, scripts `p3_make_bank.py`, `p3_common.py`, `p3_run.py` and `p3_analysis.py`.
- 100 bases × 15 strata, changing one factor at a time. Every row has a posted, in-distribution twin. Kind quota: transfer 45, scale 33, add 22.
- Strata:
  - T_posted and C_void (controls);
  - U_pending, U_draft, U_cancelled, U_reversed, U_held;
  - U_neutral (approved / active / ok / dax / wug);
  - N_voidcase (Void / VOID / voided);
  - N_postedlike (unposted / not_posted / "not posted");
  - M_bal (4–5-digit balances);
  - M_amt (|amount| 13..500);
  - A_renamed (E–H, or word names);
  - A_many (5–8 accounts);
  - F_keyorder.
- Chains: CH_S (in range, mixed statuses) and CH_X (out of range), K = 4.
- Arms: OFF = Titan-Sparse with `disable_adapter()`; T2 seeds 0, 1 and 2, unmerged in one PeftModel; Supa 1.7; and COPY.
- The bar is accuracy ≥ 0.80 with Wilson lower bound ≥ 0.70.
- RULE requires all 8 unseen-status strata at the bar in all 3 seeds, plus the twin guard (T_posted ≥ 0.80 and copied ≤ 0.10).
- TRIGGER requires all of:
  - the 7 non-void unseen strata at ≤ 0.30;
  - C_void ≥ 0.80;
  - more than half of the errors applied the event as if posted.

**Verdict: STATUS: TRIGGER [M]**, from `B/runs/bank_p3_oor/analysis/PREREG_3_RESULTS.md`.

| stratum | OFF | s0 | s1 | s2 | Supa |
|---|---|---|---|---|---|
| T_posted | 0.47 | 0.87 | 0.90 | 0.91 | 0.01 |
| C_void | 0.05 | 0.99 | 1.00 | 1.00 | 0.11 |
| U_pending / draft / reversed / held | 0.03–0.04 | 0.01–0.03 | 0.01–0.02 | 0.01–0.02 | 0.10–0.14 |
| U_cancelled | 0.04 | 0.12 | 0.05 | 0.18 | 0.12 |
| U_neutral | 0.02 | 0.02 | 0.01 | 0.02 | 0.10 |
| N_voidcase | 0.06 | **0.95** | **0.94** | **0.99** | 0.13 |
| N_postedlike | 0.03 | 0.03 | 0.02 | 0.05 | 0.13 |
| M_bal / M_amt | 0.28 / 0.10 | 0.50 / 0.19 | 0.57 / 0.25 | 0.60 / 0.14 | ≈0.01 |
| A_renamed / A_many | 0.37 / 0.41 | 0.71 / 0.84 | 0.76 / 0.90 | 0.77 / 0.90 | 0.06 / 0.04 |
| F_keyorder | 0.21 | 0.37 | 0.47 | 0.37 | 0.03 |

**What was learned.** The shortcut is a token cue, not the rule.
- `void` and its near-miss spellings (Void, VOID, voided) became the no-op cue.
- Any other non-posted status gets applied: 1,605 of 2,030 errors were applied_as_posted (s0 496/675, s1 570/687, s2 539/668).
- Holm McNemar ON > OFF is significant on N_voidcase in all seeds (p_Holm ~1e-25), and on U_cancelled only for s2 (p_Holm 0.028).

**Group verdicts.**
- Magnitude, add/transfer: DOES NOT GENERALISE, with a partial gain.
  - M_bal|at: 0.46–0.61 against twins 0.84–0.91.
  - Retained gain: −0.28 to −0.34, CIs excluding 0.
- Magnitude, scale: DOES NOT GENERALISE (0.44–0.53 against twins 0.91–0.97).
- Naming: MIXED. Letters 0.82–0.90; word names 0.60–0.66.
- Format: DOES NOT GENERALISE. Top-level swap 0.46–0.64; all keys permuted 0.28–0.30.

**Base models and chains.**
- Neither base reads the rule text. Status-sensitive pairs are 0.00–0.03 for both OFF and Supa.
- CH_S k=4 exact: T2 0.05–0.125, OFF 0.
- CH_X k=4 exact: 0.000 for every arm.

**Consequences [M].**
- D-35: every claim now needs a surface-form audit (names, key order, list order, held-out trigger words).
- R16: both the BWM and RCWM LoRA world models are locked to JSON key order but invariant to names. BWM measured reversed keys at 0.458.

## 3. T2 adapter, forgetting guard and retention suite

**T2 adapter [M].**
- Weights: `B/runs/t2/t2_s{0,1,2}/adapter/` (adapter_model.safetensors about 40 MB). The s0 sha is `92bc7c74…`, and s0 is PREREG-2's registered artifact.
- Recipe: LoRA r16, alpha 32, dropout 0.05, on q/k/v/o/gate/up/down; peft 0.20.0. Frozen FP32 Titan-Sparse; 190 steps × batch 16; lr 1e-4 with a WSD schedule.
- Data and trainer: 1,501 rows (`data/t2_train.jsonl`, sha `bf5a2276…`); vendored trainer `vendor_bwm/scripts/train_generic.py`.
- s0 training: wall time 1,858.8 s, peak 2.61 GiB recorded.
- Packaging (D-28): never dense-merged; always loaded unmerged.

**Confirmed results [M].**
- PREREG-2, on 150 fresh ledger parents, chain_one at k=4:
  - regret 0.544 → 0.056 (−0.488 [−0.535, −0.443]);
  - exact state 0.029 → 0.698.
- Across seeds, prerequisite false-copy rose by +0.53 / +0.75 / +0.80 (a copy shortcut). Register mix stayed 0 → 0.

**Forgetting guard (PREREG-2 §5b) [M].**
- Statistic: r = mean KL(OFF‖ON) / mean KL(Supa‖Sparse), on the first answer token over 248 off-task prompts. The CI comes from a 62-reply-class cluster bootstrap.
- Rule: PASS iff the one-sided 95% upper bound of r < 1; FAIL iff the lower bound > 1.
- Result: PASS at 0.50 [0.34, 0.70].
- But cumulative drift KL(Supa‖ON) is 1.94× KL(Supa‖Sparse), and Supa/ON 64-token agreement is 0.37. KL(Supa‖Sparse) = 0.129 nats.
- D-19 G8 is the older version: KL(OFF‖ON) ≤ KL(Supa‖Sparse).

**Retention suite (5c; `docs/retention/DESIGN.md`; scripts `ret_*.py`) [M].**
- Frozen at 08:03 (`RET_FREEZE.json`).
- Design: 8 code-graded categories × 75 items × 3 levels, greedy decoding with 48 tokens. Two graders, a typed primary and a lenient one.
- Dev selection (R17, Supa-only dev draw `dev_20261121`, 2,025 prompts, 388 s), 5 of 8 categories selected:
  - unit_conv L1 (0.68)
  - pct_ratio L1 (0.52)
  - kv_recall L3 (0.72)
  - unknown_twin L2 (0.64)
  - format_follow L1 (0.72)
- Excluded: arith_word (0.00–0.03), date_time and count_sort.
- GO rule: RETAINED unless there is a Holm-McNemar detected loss in a category, a category net below −11 (of 75), or a pooled net below −4% of items, under either grader.
- No confirm run has happened yet [M]. Each confirmation mints its own sealed draw.

## 4. Active and queued work, and GPU safety rules

**Live state at 22:20 [M]** (process list and lock files).
- BWM holds the GPU lease:
  - holder pid 32060, label `bwm:bundle-bundleD`;
  - job `bw_run_tests.py` → `blocks_eval.py --set mirror … shard_5of8`;
  - acquired 22:05:30, cap 16,200 s, so it releases by about 02:35 on 10-10.
- `t2bp_run.py` is not computing on the GPU yet. It is the waiting RCWM job: pid 42092, label `rcwm-t1:t2b-prereq`, since 22:03. Its log says "waiting: lock held by another session".
- GPU memory in use: 6,303 of 8,188 MiB (the BWM eval).

**5a screen (D-38) [P/M].**
- Arms: D-int, S-trace and B-bits, surface-diversity build `sd_s0`.
- Stages: T, E1, E2, E3 (held-out surface) and E4 (report-only).
- Time: about 3.83 h (one turn) at the observed rate, or 6.10 h (two turns) at the planning rate.
- Ranking: R = (K4 + IAW + OOR + HS)/4.
- STOP_P_TRAINING fires if every arm scores below 0.20 on int-add-wrong.
- The F-filler control (1.44–2.43 h) runs only if S-trace wins.

**Queue order after hook 75** (`scripts/master_chain.sh`): lik supa, pvr stage1, offtask supa, pvr retro, steer, arms randdelta / exp / transition / sparse_a0.5 / a2.0 / qwen3, hstates, then the non-binding pilot regeneration. Hook slot 70 is reserved for PREREG-LB and slot 80 for the 5b register screen. Neither is installed.

**Lease and turn protocol [M].**
- Files:
  - `C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock` (the lease);
  - `rineya-gpu-0.turn.json` (`{"last": …}`);
  - `rineya-gpu-0.waiting.{rcwm,bwm}.json`;
  - `*.stale-<pid>.bak` backups;
  - `B/runs/.yield*.json` (one-shot courtesy yields).
- Code: `rcwmt/runtime.gpu_lease(label, job)`, in this order:
  1. `yield_to_peer`;
  2. `alternate_turn` (strict one-for-one alternation, up to 45 min);
  3. the thermal flag check;
  4. `lease_vendored.lease`, where a foreign process blocks only if it has more than 256 MiB of dedicated GPU memory (I-17), and orphan reclaim only if the holder pid is dead, worker_pids is empty and orphan_after has passed;
  5. `wait_for_resources` (VRAM ≥ 3.6 GiB, host RAM ≥ 2 GiB).
- Turn cap: 4.5 h per turn, D-22 stage rule.
- One-waiter rule (R14): never more than one RCWM process waiting.

**Thermal and health [M].**
- `B/scripts/thermal_watchdog.py` (v3.6, pid 24388, relaunched 22:02) is the only writer of the flag `B/runs/.thermal_pause`. BWM's `bwmn/lm.py` also reads that flag.
- Watchdog thresholds:
  - GPU: pause at ≥ 76 °C, resume at ≤ 66 °C, sampled every 1.5 s.
  - Battery: on battery it sets the pause flag.
  - Host RAM: below 3 GB sets the pause.
  - CPU zone TZ01: ≥ 98 °C sets the pause.
  - Power request: holds ES_SYSTEM_REQUIRED|DISPLAY_REQUIRED only while a lease is held on AC.
- Logs to `runs/thermal_log.txt`. Stop file: `runs/.thermal_stop`.
- In-process guards (`runtime.breathe`, `_duty_ratio`, `ThrottleProcessor`):
  - duty pause 0.5× below 70 °C, 1.5× at 70–74 °C, 3.0× at ≥ 75 °C;
  - backup guard pauses at 80 °C until 72 °C;
  - a pause every 8 decode steps.
- BWM's `health_log.py` lives at `C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/scripts/health_log.py`. It writes `runs/health.log` and `health_now.json` every 30 s and never writes the flag.

**CPU-only torch recipe (I-21).** Set `CUDA_VISIBLE_DEVICES=""` before importing torch, then patch `torch.cuda.is_available` and `device_count` to report no GPU. Using `-1` corrupted the heap.

## 5. Joint work with BWM

The roadmaps are `R/docs/roadmap/RCWM_ROADMAP_2026-10-08.md` and BWM's `native_v1/research/JOINT_ROADMAP_2026-10-08.md`.

**Joint trace format (D-34) [M].**
- One JSON object with keys in literal order `{"cmp","calc","after"}`. It is never key-sorted, and parsers reject extra keys or a different order.
- Pipelines read only `after`.
- Filler control (amendment 4b): `{"fill":" . . .","after":…}`, with the fill length constant per (family, action kind) so it carries no information about the answer.
- Instances:
  - P: `cmp={on_miss, off_hit, viol, ok}`;
  - R: `cmp={x, thr, ok}` (strict `>`), `calc={raw, mod}`.
- BWM reference implementation: `native_v1/bwmn/traces.py`.

**Joint items [P].**
- **Item 3 (PREREG-LB):** T2 as the world model F inside BWM's `plan_beam`.
  - Co-primaries: T2 vs OFF paired success over all instances, and the same contrast on the needs-lookahead stratum.
  - Instances: N = 180, minted at the freeze.
  - Config: enum W2 E12, max_depth 7, with a range guard (|b| ≤ 98, trained amounts, posted/void only).
  - About 2.5 GPU-h. Runner `lbeam_run.py`, lease `rcwm-t1:lbeam`.
  - Draft only. It needs BWM's co-sign with a salt commitment.
- **5a:** prerequisite representation (RCWM runs it).
- **5b:** register trace vs plain vs filler (joint format, RCWM runs it). Its go rule includes a cond_sametens identification stratum.
- **5c:** retention suite.
- **#6:** an explicit legality checker trained on minimal pairs, shared across domains.
- **#8:** exact arithmetic inside the step.
- **Shared CPU work:** a shortcut-audit checklist.

## 6. Reusable interfaces (importable read-only)

- **`B/rcwmt/env.py`.** Pure-Python dynamics, prompts and parser. On import, `sys.path.insert(0, ".../RCWM-T1")` and set `sys.dont_write_bytecode=True` so nothing is written into the bundle.
  - Caveat: `load_marginals()` and `cross_check()` need the external FDS and records paths, which exist.
- **`B/scripts/p3_common.py`** (torch-free): `rule_gold`, `messages(before, ev, ko)` for key-order variants, `parse_general(text, keys)`, `score_onestep`, `wilson`.
- **`B/scripts/lbeam_problems.py`** (torch-free): ledger planning states and actions, `step_state`, `goal_met`, `unmet`, `bfs_plan`, `closure`, `load_set`, `planner_instances`, `score`, `load_exclusions` for freshness.
- **`B/scripts/lbeam_f.py`.**
  - `LedgerF(arm ∈ {T2, OFF, ORACLE, IGNORE_STATUS}).predict(keys)` returns `(next_state, True)` or `None`.
  - Wrappers: `RangeGuardF`/`guarded()`, `BudgetF`, `SurfaceAuditF`.
  - `SimulatedF` presets for sizing: SIM_T2, SIM_T2_P3, SIM_OFF.
  - `load_registered_t2()` is GPU-only and must run inside the lease.
- **`B/scripts/lbeam_proposer.py`:** `EnumerateProposer`, `beam_kwargs`, `select_fixed`.
- **`t2bp_common.py`, `t2br_common.py`** (torch-free): renderers, surface diversity and held-out surfaces, strict trace parsers, heuristics.
- **`ret_grade.py`, `ret_gen.py`:** graders and the item generator.
- **CPU reference package:** `R/../reference/rcwm/` (tiny transformer, masks, metrics; 60 tests).

## 7. What RCWM explicitly plans next (to avoid duplicating)

**Running or queued [P].**
1. The 5a screen, then seeds 1–2 of the winner. If every arm fails, P moves to a code-side bit interface.
2. The 5b register screen (hook 80 after its freeze), then 3 seeds and a sealed R bank.
3. PREREG-LB (hook 70).
4. Retention confirm runs, one per adapter.

**Roadmap items not yet started [P].**
- **#5 dissect T2:** LoRA on blocks 8–15 vs 16–27, an α sweep, and self-distillation (SD) and natural-marginal (NAT) controls.
- **#6 long horizons and seed consensus:** k = 8/16, consensus certificates.
- **#8 no-LLM learners at matched labels:** S-roll and a digit transformer on Titan's 629 ledger rows, scored on the out-of-range strata. CPU only.
- **PREREG-4:** one T3 adapter combining ledger with the winning P and R formats, with a sealed bank_confirm3, around the end of week 2.

**Leftover mechanism jobs in the queue:** likelihood ranking for Supa, PVR, steering, the dose and random-delta arms, Qwen3, Transition.

**Explicit "do not do" list.**
- Dense-merging adapters.
- Tuning on spent instruments.
- More ON/OFF deferral or voting.
- Latent single-pass composition.
- Self-label distillation now.
- Reporting tool tiers as model ability.

## 8. The models

All paths are under `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/`, in `rcwmt/runtime.PATHS` [M].

| Model | Path / size | Notes |
|---|---|---|
| Supa 1.7 (former "Taria") | `Rineya-Supa-1.7.0-devfreeze/model.safetensors`, BF16, 1.19 GB | 596,049,920 parameters; Qwen3 architecture, 28 layers, hidden 1024, 16 heads / 8 KV heads, vocab 151,936, tied embeddings. Fingerprint `a9952c55f5a1`. |
| Titan-Sparse-Experimental 1.0.0 | `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze`, **FP32, 2.38 GB**, sha `a28629c9…` | Supa plus the per-tensor top-25% \|delta\| of the RRP-015 control update (0.056% of the weight norm, 83.6% of the delta norm retained). Must not be converted to BF16. Fingerprint `818be9c6b6a3`. |
| Other arms | `Rineya-Titan-Experimental-1.0.0-devfreeze`, `Rineya-Titan-Transition-1.0.0-devfreeze`, and Qwen3-0.6B at `C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de2…` | |

**Loading (`runtime.load_arm`).**
- `AutoModelForCausalLM.from_pretrained(dtype=float32, device_map={"":"cuda"}, attn_implementation="sdpa", low_cpu_mem_usage=True)`.
- TF32 off, `set_per_process_memory_fraction(0.70)`.
- T2 is attached with `PeftModel.from_pretrained(base, adapter_dir, adapter_name=…)`.
- Delta arms are built on the GPU from safetensors mmap.
- Runtime: hermes venv `C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv` (Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0).

**VRAM [M].**
- Card: RTX 4070 Laptop, 8,188 MiB, which also drives the display.
- FP32 model: about 2.4 GB.
- FP32 KV cache: 224 KiB per token. Batch 64 once spilled 1.37 GB to host memory (I-07), so the budget is now 8,000 tokens per batch, about 1.8 GB of KV in the worst case.
- Floor: 3.6 GiB free.
- T2 LoRA training peaked at 2.61 GiB (recorded).
- [I] A complementary GPU job needs about 3–6 GB, must go through the shared lease and alternation, and will not fit while another job holds the card.