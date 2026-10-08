# Research-reader report: BWM-001, the Rineya lineage, recent work, launchpad, and prior world-model / verifier attempts

I did not modify any files, load any model or run any training. All numbers below are quoted from the files cited.

---

## 1. BWM-001: what it is, how it relates to RCWM-001, and what code can be reused

**Location:** `C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle` (2.8 MB, v1.0, dated 5 Oct 2026).

**Idea.** "Outcome-Conditioned World Modeling." It formalizes a "reverse world model" as an outcome-conditioned trajectory proposer `Q_theta(tau | h_t, g, w, G*, H)`, paired with:
- a goal-blind grounded forward model `F_phi(s_next | h_t, action, m)`,
- an exact evaluator `R_w`,
- a plan → execute-first-action → observe → replan loop.

The v0 architecture is one frozen 0.6B Rineya backbone with a LoRA proposer adapter, plus a small independent forward sidecar, an exact evaluator, and bounded MPC.

The design is built around avoiding "success-conditioned fantasy." The forward model never sees the desired return. The analytic luck counterexample is: safe action EV 0.8 vs risky action EV −0.8, yet filtering for return ≥1 keeps only the risky action. Cycle-consistency weight is 0 (decision D04).

**Core experiment (2×2 factorial):**
- T = explicit predicted-state supervision; F = forward-model planning.
- Arms A00/A10/A01/A11. Primary contrast is **A11 − A01** (does state prediction add value given grounded planning).
- Controls: frozen base, success-only SFT, goal-conditioned BC, action-only best-of-K, heuristic + same forward model, exact-simulator oracle.
- Promotion: ≥5 pp executed-success gain, CI lower bound >0, retention lower bound >−2 pp.
- Planning numbers: n=621 parents (discordance .20, effect .05); 1,024 confirmation parents; ~1,955–2,048 retention cases.
- Pilot: 3 seeds (11/23/37) × 2 adapter families = 6 runs × 200 updates; mb1, acc8, len256; rank-8 q/v LoRA (≈1,146,880 params on stock geometry); K=4 candidates, horizon 4, replan after 1 action; ≈409,600 token positions per run.
- Gates G0–G6. Only G0 has reference evidence; G1–G6 are "NOT EXECUTED" (`program/ACCEPTANCE_GATES.json`).
- `evidence/NATIVE_STATUS.json`: `UNQUALIFIED_NOT_RUN`. No Rineya checkpoint, no CUDA, no retention test.

**Measured (CPU, Linux build host, torch 2.10.0+cpu, no transformers or peft):**
- 65 tests pass in 1.17 s.
- RegisterWorld-v1: integer x, |x|≤20, actions `add1/sub1/double/halve/finish`; step cost −0.01, violation −0.1.
- 551 episodes from 200 parents; splits train 368 / val 87 / cal 41 / test 55. Digest `1f5db7a1…b56d`.
- 3,561 role rows: forward 1660/374/172/253; policy and reverse 368/87/41/55 each.
- `SmallForward` 47-256-256-164 = 120,228 params; minibatch loss 5.115→0.212 over 100 updates in 0.18 s. This is a train-batch figure only.
- Demo: `select_plan` rejects 1 injected state inconsistency and selects `add1,halve,halve,add1,finish` (−9→−1, utility 0.95).
- The T0 pool has only 625 start/goal pairs, so it cannot support a 1,024-parent confirmation.

**Reusable code (all CPU, original, no dependencies except optional torch):**

| Module | Contents |
|---|---|
| `src/rineya_bwm/core.py` | `Task` (hash-split 70/10/10/10), `exact_step`, `shortest_plan` (BFS teacher), `execute`, `returns_to_go`, `generate_episodes(n,seed)`, `validate_dataset`, `make_sft(role)` (protocol `bwm-json-raw-v1`, prompt `MODE policy/forward/reverse …OUTPUT`), `EmpiricalForward` (count model with unsupported-transition rejection), `EmpiricalReverse`, `select_plan`, `luck_counterexample` |
| `src/rineya_bwm/torch_forward.py` | the forward sidecar |
| `src/rineya_bwm/tokenization.py` | `encode_target_only` (prompt masked to −100, EOS trained, unstable boundaries and overlength rejected) |
| `src/rineya_bwm/analysis.py` | Wilson interval, paired parent cluster bootstrap, factorial interaction, sample size |
| `scripts/native_adapter.py` | HF Qwen3 q/v LoRA qualify/train scaffold |

Notes on `scripts/native_adapter.py`:
- It refuses configs with `model_type != qwen3` or with `auto_map`. Titan configs are plain `Qwen3ForCausalLM` with no auto_map, so they would pass.
- Its default `--min-free-host-gib 7.4` will block on a host with ~2 GB free RAM.
- It needs CUDA, transformers and peft.
- It is labeled unqualified.

**Relation to RCWM-001.** RCWM-001 is the sibling bundle and the current working directory (`…/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05`). It has the same date and the same Titan constraints (596,049,920 params, 8 GB). Neither bundle cross-references the other.
- **RCWM's focus is the evaluator.** It learns reusable *consequence representations* plus a goal-conditioned linear utility readout (successor-feature style), tested against a **matched scalar Transformer evaluator** (registry G1-B2 vs B3/B4).
- **BWM's focus is the proposer.** It tests whether generating future states helps the proposer, with an *exact* evaluator.
- They are complementary: BWM's forward model plus exact evaluator would become RCWM's learned consequence predictor once exact evaluation is unavailable.
- RCWM measured: 60 CPU tests; a 40,325-param `FeatureTrajectoryTransformer` trained 250 steps.
- RCWM's preserved known failure: on a safe prerequisite chain it predicts object-gained = **0.0638 vs exact 1.0**, despite low average MSE.
- RCWM's `specs/NATIVE_8GB_RUNBOOK.md` requires a prefix-vs-bulk logit parity preflight. This matters because that check has already failed on Titan (§3).

---

## 2. Family lineage and evaluation instruments

**Anchor:** Qwen3-0.6B: 28 layers, hidden 1024, 16 query / 8 KV heads, head dim 128, vocab 151,936, tied head, 596,049,920 params.

**Lineage:**
1. **Phase 0–8 teacher (Aug).** 19-layer Pulse 1.0-rc1 (~454M, Phase-2B layer drop; human commercial PASS on 2026-08-05) → Aegis 1.1 (+ dual-SFT v1 LoRA, 16,410 rows: 12k short-response + 4,410 RAG; the RAG specialist).
2. **28-layer cross-depth stitches (no 28L fine-tuning).** The 19 kept blocks are blended into the official 28L scaffold:
   - Halo 1.2 (α=0.5 over a v1@5e-4 + v2@1e-4 delta soup)
   - Halo-Uncensored 1.5 (α=1.0)
   - Lyra 1.4
   - Pulse 1.6 (`late_rineya` α shape)
   - Nimbus 1.6 (uniform α=0.40, Qwen3 → Halo-U)
3. **Supa 1.7 / Baya 1.7 (Aug 26).** Nimbus + an incomplete Stage-A long-context QLoRA (checkpoint 200, trained on Halo; NaN eval loss; stopped at step 250) merged at scale 0.8125 / 1.25. Supa SHA is `962abc5b…d24b`.
4. **Supa-A1 (Aug 27).** Same tensor plus a baked +5.15625 logit bias on token 51378 `UNKNOWN`. Supa-A1.1 is an FP16 gated variant used in SupaExperiment.
5. **Routed systems.** System48/46/45/42 are cascades and cannot be collapsed into one weight file.
6. **Opallo 1.0.** Supa-derived arithmetic candidate: 73/150, but long context 161→135 and literal UNKNOWN 13→0.
7. **Titan (Sept), in `Chat-Bubble/Rineya_Published_Cores/`:**
   - **Titan-Experimental** (Sept 17, RRP-015). Supa + 384 solver-verified Forge-RL-v2 one-step transitions; tail blocks 8–27; 28 updates; LR 1e-5 with WSD schedule; 54,172 tokens; FP32 training, BF16 export (SHA `53720167…`).
   - **Titan-Sparse** (SPARSE-001). Supa + top 25% of that update by |δ| per tensor; **FP32 only**, because BF16 zeroes 28.49% of the kept delta (SHA `a28629c9…`). The update is 0.056% of the weight norm; 25% of its elements carry 84% of its norm.
   - **Titan-Transition** (CONFIRM-002 TH25, promoted 2026-09-26). Titan-Sparse + 0.25·P16(δ_H), where δ_H is a rank-16 boundary/abstention delta (FP32, SHA `a4d73353…`).

**Instruments:**

- **Showcase /64 and Stability /192.** These come from `Rineya_Long_Context_Demo_v1`: frozen synthetic long-context reading, about 1,119–1,556 input tokens under a 4,096 cap. Protocol: greedy, seed 17, 24 new tokens, thinking off, strict exact match.
  - Each scenario has 4 correlated rows (base, paraphrase, counterfactual, unanswerable), so the units are 16 / 48 packs.
  - **All256 is the union of the two, not an independent test.**
  - Showcase has been used for selection many times. Stability has been inspected. For Titan both are **"SPENT"**: report-only and barred for training, selection or confirmation.
  - Reference scores: Qwen3-0.6B 14/64; Supa 40/121/161; Supa-A1 43/125/168; System48 48/128/176; Qwen3-4B (NF4) 43/64; Qwen3-8B (NF4) 53/64.
  - Titan scores were measured on the RAB-001 harness, where Supa = 41/161: Titan 162 (its FP32 parent 157, so the 162 is a BF16 artifact per AMENDMENT_001) and Titan-Sparse 164.
- **Arithmetic /150.** A 150-row word-problem monitor drawn from the v3 corpus, covering 3 of 39 reply classes (percentage_reasoning, ratio_proportion, time_duration). Scored permissive and strict.
  - Companion endpoints: "source answers retained /47" (Supa's 47 correct rows) and a legacy per-class zero-loss monitor that is nearly unpassable.
  - Scores: Supa 47/41; Titan 69/62; Titan-Sparse 72/63.
- **Other instruments:**
  - SELECT-VIEWS-001 /384 transition acquisition: 128 per family, with prepublished floors 0.2344 / 0.0781 / 0.0703.
  - Literal `UNKNOWN` /64.
  - CONFIRM-002 banks: transition /864, boundary /336, answerable /109.
  - Puzzle /450.
  - Forge IID and transfer /200.
  - Older chat DEV 510 and RAG DEV 752×2.

---

## 3. Recent research directions (late Sept – Oct 5) and findings

- **CONFIRM-002 (Sept 25).** Results file: `Rineya-Titan-Transition…/evidence/CONFIRM002_RESULTS.md`.
  - C1 PASS: transition skill kept, 430→434 (lower bound −2 vs a −21.5 margin).
  - C2 PASS: boundary abstention +67/336, p=1.4e-13.
  - C3 FAIL: share of boundary skill retained P_H=0.859, lower bound 0.718 < 0.75.
  - C4 PASS: 109/109 answerable rows still attempted.
  - C5 FAIL: full-strength δ_H mostly imports formatting. Strict transition score drops 430→260 because 342/864 answers gain a trailing period.
  - This is the first confirmed second-skill composition.
- **REP-001 / pruning (Sept 17).** Pruning's benefit replicates in direction across 3 of 3 draws but not in magnitude:
  - Long-context deltas +7 / +0 / +15; retained +1 / +1 / +4.
  - The worst control draw scored 144/256 (p=0.0115); its pruned version scored 159.
  - Pruning acts as a *repair*.
- **RRP-007 three-seed process arm** (1,536 transition rows; `ChatBubble-post-training-Mimo/PROCESS_TRANSITION_ARM.md`).
  - Acquisition is **289/384 in all three seeds**.
  - Arithmetic panel 71 / 58 / 65; seed 23 fails retention.
  - Summary: "gains the new skill reliably and forgets unreliably."
  - A rehearsal arm (648 rows) lost 27 long-context rows (p=2.53e-5).
- **Centralized MiMo E0/E1** (Sept 28–30; `experiments/post_training_mimo_centralized_v1/RESEARCH_LEDGER.md`). Goal: a Titan BF16 q/v LoRA run.
  - Repeatedly blocked by host RAM. The admission budget is 3.3125 GiB; observed free RAM was 0.22–3.5 GiB.
  - Capture qualification **failed numerically**. Online vs bulk logprob error reached 0.0414 against a 0.03125 tolerance. The cause traces upstream to hidden states: prefix vs full hidden states differ by max 1.25, mean 0.0554.
  - No E1 run happened.
- **Sept 29.**
  - RVS-001 (verified resampling on Titan): implemented and CPU-tested; native efficacy unmeasured.
  - EFE-001 (FUNC / KV / DEQ tracks): CPU reference implementations only.
  - CDE-001 (compute-depth elasticity): offline only; native blocked.
- **TAPS campaign v2 (Oct 1).** Restricted negative result:
  - DEV 67/192 vs 69/192 for the fixed and random controls; validation 34/96 for all arms.
  - Even an oracle reaches only 70/192.
  - **Query-blind floors (71/192, 42/96) exceed the model scores.**
  - Increasing recurrence scale harms UNKNOWN likelihood on 48/48 cases.
- **CRS-001 (Oct 1).** Centralized reasoning system (evidence store, typed operations, verifier, procedure library). Design recorded; execution not authorized.
- **SBHM-001 (Oct 4).** Hybrid-memory recurrence on Titan-Transition, FP32:
  - Local-only arm C 172/224 vs local+recurrent arm D 171/224, so **no recurrence benefit**.
  - Frozen source 51/224.
  - Both C and D damage 4 of 22 source-correct retention cases.
  - Missing-evidence handling collapses to null on all 32 cases.
  - KV saving is only 3.24%.
  - Peak VRAM ≈2.49 GiB.
- **BAG-001 (Oct 4).** BF16 attention-gradient audit; ARMR CPU score 12–13/16.
- **Oct 5 quality push:**
  - **Titan Joint Quality Pilot v1→v3.** v1 found that query-blind policies had *perfect structural shortcuts*. v3 repairs the instrument with opposite-query pairs: 2,000 TRAIN / 384 PILOT, ≤54,172 tokens per arm. Acceptance needs ≥5 pp more semantic reasoning than Titan-Sparse. Native run not dispatched.
  - **Quality pattern research.** Historical exact-match "gains" hid a repeatable source-correct regression.
  - **matched_mlp M1.** The single authorized attempt was consumed with **zero model loads**: the Windows venv launcher broke the parent-ownership check.
  - **CE8 / COMP-RS S1** (3,000-row MaxRL/CE acquisition arm): only accounting and census reviews.

**Meta-finding:** most work since ~Sept 26 is qualification and custody infrastructure. Native results are scarce, and host RAM is the binding constraint.

---

## 4. Launchpad (`Rineya-Research-Launchpad-20261005`)

There is no README. The folder holds `inventory.json` plus 11 prompt files. Every prompt is offline-only (inspection, fixtures, one reviewed native proposal) **except #02**, which authorizes one native M1 mechanics attempt with no training.

| # | Task |
|---|---|
| 01 | Non-recurrent adapter: evidence reconciliation; serving bias 3.75 vs historical 0 / 5.156 |
| 02 | MLP-SHIP-001 M1 mechanics on the 156,225-param sidecar; budget 900 s / 6144 MiB RSS / 7168 MiB reserved |
| 03 | M2 readiness: 96 CAL / 192 CHECK rows across retrieval, counting/arithmetic and state_tracking; 25% unanswerable |
| 04 | GD-001 trajectory-snapshot attachment |
| 05 | DSD-001 privileged-context self-distillation |
| 06 | RDD-001 distillation dynamics (rollout source × KL direction plus SFT control) |
| 07 | RSD-001 recursive self-distillation, explicitly "secondary to CE8/pruning/matched_mlp" |
| 08 | PREFIX-ISOLATION-1 reconciliation |
| 09 | Geometric third-skill parent qualification |
| 10 | TAPS closeout |
| 11 | SBHM closeout / justified hold |

Stated priority order: **CE8, pruning, matched_mlp first.**

`Rineya-Research-Git/self-distillation-axes.git` is a shallow bare repo with a single commit, `e173766` (DSD-001), on branch `research/rineya-self-distillation-axes-v1`. It has no README.

---

## 5. Prior world-model, consequence, planning, value-head and verifier attempts

1. **One-step transition learning (consequence prediction) is the strongest positive result in the family.**
   - Frozen Supa and Opallo score **below the copy-the-input floor** on one-step transitions: Supa 14/384, with 0/128 on both prerequisite_plan and register_machine.
   - SFT on verified transitions brings acquisition to 289/384 (1,536 rows) or 234/384 (384 rows).
   - It is not echoing: correct on 251/346 rows where the state actually changes.
   - It **transfers** to arithmetic: time_duration 10→23 under both permissive and strict scoring.
   - It composes with abstention (Titan-Transition).
   - Retention is the fragile part.
2. **Multi-step reasoning is at a ceiling** (`PUZZLE_CEILING.md`).
   - Supa scores 102/450 against a guessing floor of 0.229; Opallo 96.
   - Forge nulls held across schedule, depth, dose, architecture, weight averaging and an 11.5× reasoning-data dose.
   - The memo calls this a "capacity ceiling" for end-to-end multi-step reasoning at 596M.
   - *Implication:* chaining a learned one-step model with external search/execution (BWM-style MPC) targets exactly this gap.
3. **Planning environment and data already exist.** `ChatBubble-ced-native-loader-qualification/rineya_reasoning_forge_rl_v2/` (171 MB, released Sept 14) contains:
   - 8 families, 20k canonical worlds.
   - `forge_rl/environment.py::PlanningEnvironment`: a reset/step interface with STOP / IMPOSSIBLE, prerequisite_plan only.
   - `oracle.replay_transition`, `verify_solution`, `verify_trace`.
   - `process/train_transitions.jsonl.gz`: **36,752 labels = 18,376 valid/invalid `proposed_after` pairs** (ledger_replay 16k, register_machine 16k, prerequisite_plan 4,752). This is ready-made data for training a transition verifier.
   - 2,500 continuous-replanning episode worlds × 4 stages; 32,000 preference pairs.
   - **It is far richer than BWM's RegisterWorld.**
4. **Verifiers and RL: designed and CPU-tested only.** None has a native run, and RLVR infrastructure is absent.
   - RVS-001 (verified resampling)
   - PRLC-001 (GRPO with a deterministic verifier)
   - MAXRL-DATA-001 (32,800 verifiable problems)
   - GFlowRL, DREAM-001
   - `reasoning_posttraining_v1/verifier_audit.py`
5. **Value-like internal signals.**
   - Generic uncertainty (entropy, top-1/top-2 margin, logprob) **failed**; "the model is often confidently wrong."
   - Direct UNKNOWN-token mass has AUC ≈0.8. The S1 gate's +7 on Stability (121→128) is stable, but its significance is not robust.
   - TAPS' coherence signal matched the true loss direction on 91/188 cases, no better than a constant.
6. **Small sidecars.** matched_mlp is a 156,225-param GELU sidecar shared across layers [3, 8, 13, 18, 23, 27] on Supa-A1.1.
   - DEV expected loss −41.2% at bias 0.
   - Answerable 202/720 vs 109/720 at bias 5.156.
   - Behavior at the live serving bias of 3.75 is unmeasured, and native M1 has not passed.

**Constraints for any new experiment:**
- Host RAM is the binding limit: ~2 GB free; MiMo admission blocked at 3.3 GiB; a 1.4–1.5 GiB RSS was observed for BF16 captures.
- Use Titan-Sparse or Titan-Transition in FP32 only (2.38 GB weights).
- Prefix-vs-bulk numerical parity is unresolved.
- Showcase/Stability are spent. Use fresh, pack-disjoint, parent-disjoint banks with prepublished shortcut floors.
- Environment: `Chat-Bubble/.venv` has torch 2.12.1+cu126, transformers 5.14.1, peft 0.19.1, bitsandbytes 0.50.0. Historical Titan runs used transformers 4.51.3 / 4.57.1, and transformers 5.x changes `apply_chat_template`'s return type.