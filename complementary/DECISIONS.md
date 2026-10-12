# DECISIONS: RCX complementary session

Each entry records the timestamp (CDT), the decision, the reason, whether outcomes were already known when it was made,
and the evidence path.

## 2026-10-09

- **CX D-01 Branch, worktree and folder (22:20).**
  - Created branch `research/rineya-complementary-intelligence-v1` from `origin/main` at `a223884`.
  - It lives in a separate worktree, `WorldModel/rineya-complementary-worktree`. The shared checkout
    `rineya-world-model-research` stays on `main` and is not touched.
  - All RCX files live under `complementary/`.
  - A read-only, detached worktree of `origin/research/rineya-high-quality-dataset-v1` was added in this session's
    scratchpad for inspection only. It will be removed with `git worktree remove` when no longer needed.
  - Outcomes known: n/a.
- **CX D-02 Understanding pass (22:25).**
  - Seven parallel read-only readers covered: the bundle core, tracks, protocol, RCWM, BWM, the dataset factory, and
    coordination.
  - A synthesis was then written: `docs/understanding/SYNTHESIS_2026-10-09.md` and the per-area reports.
  - No peer file was modified.
- **CX D-03 Scope split with peers (22:35).**
  - RCX takes **rule-conditioned execution** for RCM-001: rules supplied in context and varied, with opposite meanings,
    nonce words and counterfactual twins.
  - It does **not** take fixed-rule surface diversification, which is RCWM T2b 5a/5b.
  - **Reason:** RCWM PREREG-3 (REPORT §3c) shows two things:
    - T2 learned a `void` trigger;
    - "neither base model reads the rule text" (status sensitivity 0.00–0.03).
  - T2b repairs the fixed rule's surface. Nobody tests whether a model can *read and execute a supplied rule*, which is
    the bundle's RCM research question.
  - E10 search distillation is BWM's B3, so RCX does not run it.
  - Joint item #6 (legality-checker LoRA) remains a peer item unless handed over.
  - Requests were sent (`docs/REQUESTS.md`, CX Q-01, Q-02).
- **CX D-04 Compute posture (22:35).**
  - CPU only. GPU is leased by BWM (bundleD, until about 02:35 on 10-10) with RCWM waiting (5a, 3.8–6.1 GPU-h), then
    BWM B3.
  - Host RAM available is about 5–7 GB, below the 12 GB CPU-lane load floor (D-39). So no 0.6B model is loaded on CPU
    now.
  - tz01 is 94–96 °C against the shared 98 °C GPU-pause threshold. So RCX limits itself to 4 threads and pauses at tz01 ≥
    97 °C.
- **CX D-05 Labels (22:35).**
  - Programme label RCX; decision IDs `CX D-nn`; requests `CX Q-nn`; runs `RCX-Exx`.
  - Reason: `RCI-001` already names a spent calibration bank (`rineya-dataset-retention-records/rci001-20260918`).
  - The seed range is `202612xx`, disjoint from BWM (20261005–11), RCWM (20261121, 20261141–43) and the factory
    (20261009).
- **CX D-06 RCWM confirms the scope split (about 22:55).**
  - RCWM's reply to CX Q-01 (`docs/REQUESTS.md`): no rule-varying ledger arm on RCWM's side, and imports are allowed
    with sha pins.
  - **Binding for RCX:**
    - never call `rcwmt.bank.load_exclusions()` or touch `HSKILL_CONFIRM_V2*`;
    - PREREG-3's 16 strings never appear as RCX statuses (already enforced by `vocab.EXCLUDED`);
    - any future Titan/Supa training data must exclude RCWM's 5a/5b held-out paraphrase #4, field-name variant #4 and
      held-out key orders.
  - Retention confirms for RCX native candidates go to RCWM.
  - Qualification check Q1 already passed against the pinned files: 20,000 posted-rule cases, with
    `world.transition == env.step == p3_common.rule_gold`.
- **CX D-07 BWM confirms the scope split (about 23:00).**
  - BWM's reply to CX Q-02 (`docs/REQUESTS.md`): B3 is BWM's, and RCX will analyse no B3 output before `B3_REPORT.json`
    exists. E07/E08 are allowed on fresh instances with BWM's exact exclusion recipe (`bw_b3_prep.py::fresh`). The
    reserved test sets are read only to build exclusions and are never evaluated on (wording corrected 2026-10-10 at
    BWM's request; see CX D-13). E04 BW minimal pairs will come from `traces.bw_parts`.
  - **New binding machine rules for RCX:**
    - **one CPU lane at a time** (no concurrent heavy CPU jobs);
    - a 4 GB free-RAM floor before any job over 2 GB;
    - honor RCWM's flag.
  - The RCX guard already pauses on the flag, at tz01 ≥ 97 °C and on low RAM.

## 2026-10-10

- **CX D-08 E01 pilot: first learnability failure (about 23:18 on 10-09). Outcomes known: yes, DEV pilot only.**
  - `VR-ans`: CharLM d128/L3, 32k examples, 3,000 steps, 617 s.
    - It collapsed to COPY: IID exact 0.522 and decision 0.523; 711 of 717 IID errors are false_copy.
    - The DEV `in` composite is 0.478, with every stratum at about 0.5.
    - Every `ho` surface except `ho_tpl` gives format errors (unseen names, extra accounts and held-out key orders).
  - **Reading.** At this budget, an answer-only from-scratch character model neither reads the rule nor does the
    arithmetic. The decision is never supervised directly, so the model hedges toward copying the state.
  - **Consequence for the design.** Arm-vs-arm comparisons are uninformative until every arm learns its *own training
    distribution*. This calls for learnability diagnostics before the factorial:
    1. `FX-ans`: is a fixed trigger learnable?
    2. `VR-trace`: does explicit `listed`/`ok` supervision unlock rule reading?
    3. Then, as needed, capacity/steps and a pretrained small-LM learner.
  - **Prereg rule.** The `ho` composite is not measurable for from-scratch character models, because they cannot emit
    unseen names. Under the draft prereg rule, the CPU tier's primary endpoint therefore falls back to P-in. `ho`
    surface robustness is measured via the canonicalizer.
- **CX I-01 Background job stall (2026-10-09 about 23:31 to 2026-10-10 about 06:02).**
  - The background runner finished `FX-ans` but did not start `VR-trace` until about 06:02. The machine stayed awake:
    the watchdog kept logging and the peers ran GPU jobs.
  - No data were lost. `VR-trace` started fresh, with no resumed checkpoint.
  - Cause unknown. The likely explanation is that the host suspended the session's background shell.
  - **Lesson:** long queues are checked for progress, not assumed to be running.
- **CX D-09 E01 learnability diagnostics: the arith world is not learnable by the CPU char model at this budget.
  Outcomes known: yes (DEV pilot).**
  - Results at d128/L3, 32k examples:

    | Arm | Steps | IID exact | IID decision | Note |
    |---|---|---|---|---|
    | `FX-ans` (T2 analog) | 3,000 | 0.169 | 0.627 | 687 of 1,247 errors are arithmetic |
    | `VR-trace` | 3,000 | 0.409 | 0.584 | |
    | `VR-ans` | 3,000 | 0.522 | 0.523 | collapsed to COPY |

  - **Reading.** Two-digit add/transfer arithmetic plus a rule decision is too much for the CPU tier's budget. Even the
    fixed trigger is not cleanly learned. Trace supervision of `listed`/`ok` does not unlock the decision.
  - **Decision.** Add the arithmetic-free **copy world** (SPEC §1.1: `set`, `swap`). The CPU mechanism tier then studies
    applicability C with the calculation A factored out, as in the bundle's RCM factorization C/A/assembler.
  - **Qualification.** The independent oracle was extended blind, from the spec only (29 oracle tests). Qualification
    passes in both worlds: Q2/Q3 on 76,320 records, plus Q1. All 89 tests pass.
  - The arith world stays the native-tier instrument, because pretrained cores already compute.
  - **Provenance note.** The `code_sha256` of the arith-world pilot summaries for `FX-ans` and `VR-trace` was computed
    at the end of each run, after `gen.py` and `world.py` had been edited to add the copy world. The arith-world
    behaviour is byte-identical: qualification Q2/Q3 reproduced the same records.
  - **Next.** Learnability of `VR-ans` and `CFSD-ans` in the copy world, then the native tier-A/B probe
    (`experiments/RCX-E01N/native_probe.py`: OFF, T2 s0 and Supa on the arith battery, CPU inference with ≥ 12 GB RAM).
- **CX D-10 The character-model mechanism tier is closed as non-informative for E01 (about 06:50). Outcomes known: yes
  (DEV).**
  - In the copy world (no arithmetic), at 4,000 steps:
    - `VR-ans`: IID exact 0.531, decision 0.601;
    - `CFSD-ans`: 0.505 and 0.545.
  - In all five configurations the decision stays near chance, *including on each arm's own training distribution*.
  - **Decision.** Do not run the 9-arm character factorial: any arm difference would reflect learnability noise, not
    the data factor. Two replacement paths:
    1. The **symbol-token learner** (SymLM), with identifier words as single tokens and frozen random embeddings.
       Being built.
    2. The **native tier**, Titan-Sparse LoRA cells on CPU with the D-03 recipe. This is the decisive test for Rineya.
  - The kit is under adversarial review (wf_53c3f5a8-51d) before any 0.6B load.
- **CX D-11 RGM-001 E07/E08 infrastructure built (about 09:50).**
  - Code: `rcx/bw/*`, 19 tests.
  - **BWM imports.** Pinned and executed from verified bytes, so no BWM bytecode is written. `bw_distill.problems`
    runs from its pinned source without importing torch. Regenerating `b3_fresh` in memory reproduces BWM's file sha
    exactly, which validates the exclusion recipe.
  - **Exclusions** (`experiments/RCX-E07/EXCLUSIONS.json`): 349 renaming signatures and 637 invariants (n = 6..15).
  - **Exact h\***: full state spaces up to n = 8 (695,417 states), and A\* per state for n ≥ 9. It agrees 100% with
    `B.bfs_plan`.
  - **Design consequence.** Full-goal instances at small n are nearly exhausted by the exclusions (n = 3: 0 survive;
    n = 4: 25). E07 therefore trains on n = 4..7 and mixes in partial goals. Row-level exclusion stays on by default.
    It is conservative and goes beyond BWM's instance-level recipe; its effect on coverage is reported.
  - **Seeds:**

    | Seed(s) | Use |
    |---|---|
    | 20261270 | training instances |
    | 20261271 | validation instances |
    | 20261272 | E07 test, 7–12 blocks |
    | 20261273 | E08, 8–15 blocks |
    | 20261274 | state sampling |
    | 20261275 | split |
    | 20261276–79 | model seeds |
    | 20261280–89 | unit tests |
- **CX D-12 Native-tier cost on CPU (about 09:55).**
  - **Inference.** The native tier-A/B probe (probe1: OFF, T2 and SUPA; 40 parents; `in` and `ho_tpl`; 2,240 battery
    prompts plus 800 verbatim-frame controls per arm) runs at about 1.5 s per prompt. Prefill of about 150 tokens
    dominates. The guard pauses on the shared flag, including GPU-temperature reasons, as a courtesy to the peers' GPU
    jobs.
  - **Training estimate.** A D-03 LoRA cell (190 × 16, about 0.46M processed tokens) is about 3 h on 4 CPU threads, plus
    about 1 h of evaluation.
  - **Plan.**
    1. Finish probe1 on CPU, about 3.5 h in total.
    2. Decide between CPU training (seed 0 of N-FX, N-VR and N-CFSD ≈ 12 h of the single lane) and a GPU-slot request
       to both peers (≈ 0.75 GPU-h per cell). The request goes only after the probe's baseline justifies the cells,
       and never in competition with leased work.
- **CX I-02 Single-lane violations, and record corrections requested by BWM (about 10:15).**
  - **Violations.**
    - At about 09:44, a SymLM builder agent's learnability test (40k records, batch 256) ran while the native probe
      (pid 13188, 4 threads) held the CPU lane. tz01 was 96.1 °C.
    - At 10:09 the E07 runner agent started a smoke run (pid 122664) concurrently.
    - Both breach CX D-07 (one heavy CPU job, ≤ 4 threads) while peers' GPU jobs were running.
  - **Actions.**
    - pid 122664 was terminated at about 10:14.
    - The E07 agent was instructed to run no CPU jobs until the lane is free.
    - **New rule for RCX:** sub-agents may only edit code and run checks costing less than 5 s of CPU while any RCX
      lane job runs. All training and smoke runs are executed by the coordinating session, one at a time.
  - **Record corrections (BWM message, about 10:10):**
    - (a) The COLLABORATION entry wrongly dated "2026-10-10 about 23:20" is re-dated to the arrival window it actually
      fell in.
    - (b) The B3 gate now reads "no B3 reads before `runs/BW_B3_EVAL/B3_REPORT.json` exists" (COLLABORATION,
      ROADMAP).
    - (c) BWM reserved and selection sets are "read only to build exclusions; never evaluated on" (COLLABORATION,
      D-07).
- **CX D-13 BWM FYIs recorded.**
  - BWM now records the Q-02 boundary in its own DECISIONS.md.
  - D-54 Amendment 1: `PREREG_D54_KEYORDER.md` d2e0d86c… and `bw_keyorder.py` 94561044…. D-54 runs inside BWM's B3
    bundle after B3, with per-item outputs in `runs/BW_KEYORDER/items.jsonl`.
  - The `blocks_planner.py` pin stays at 18a5f22b… for RCX's E08. `blocks_val_large.py` and `bw_distill.py` are
    unchanged during B3, and RCX's sha checks fail closed if they change.
- **CX D-14 Native probe1 result and next native steps (about 14:30). Outcomes known: yes (DEV, descriptive).**
  - Result: `EXPERIMENT_RESULTS.md` R-03.
    - OFF and T2 s0 have zero sensitivity to in-context rules: they produce a no-op in 1/280 required-no-op cases, and
      V3 sensitivity is 0.00.
    - T2 shows the TRIGGER signature on fresh words (void 1.00, near-miss 0.97, unseen 0.00).
  - **Consequence.** The native E01 cells (N-VR, N-CFSD, plus N-FX as a matched fixed-rule control) are well
    motivated. The baseline rule-reading rate is about 0, so any learned rule sensitivity is measurable.
  - **Run order in the single CPU lane:**
    1. E07 smoke (2 min);
    2. native profile (3 steps);
    3. N-VR train and eval;
    4. N-CFSD train and eval;
    5. SUPA probe arm, when RAM ≥ 12 GB;
    6. N-FX;
    7. the E07 full run.
  - The SUPA arm was refused at 11.4 GB free RAM, which is the fail-closed behaviour working as intended.
- **CX D-15 Cross-scoring with RCWM agreed in principle (about 14:40).**
  - RCWM logged R-03 as R19.
  - **Protocol:**
    - adapters are exchanged as directories plus sha256 only;
    - each programme scores the other's adapter on its own sealed instrument and sends results;
    - there is no access to the other side's confirmation banks.
  - CX Q-03 will be sent once N-VR or N-CFSD and T2b-ledger exist.
- **CX D-16 New BWM reserved set to exclude (BWM message, about 14:50).**
  - The set is BWM item 7, seed replication: `native_v1/runs/BW_val_large/instances_seeds_fresh.json`.
    - sha256 6fccb94547184d8ba44a2fdc19ad6c3f1fca584c67c906964281746c0541b3fc, verified by RCX.
    - 40 instances from generator seed 20261014.
    - Preregistered in BWM `PREREG_ITEM7_SEEDS.md` (956a0dc4…).
  - **Actions:**
    - add its `blocks_val_large.invariant`s to the RCX exclusion union;
    - avoid seed 20261014;
    - never evaluate on it without a joint prereg.
  - **Status:** pending until the E07 review reconcile finishes, to avoid conflicting edits to `rcx/bw/excl.py`.
    `EXCLUSIONS.json` must be rebuilt before the E07 full run, and `run_e07` refuses stale exclusions.
  - **BWM pins noted:** `blocks_eval.py` b3666424… (additive change), `blocks_planner.py` unchanged at 18a5f22b….
- **CX D-17 E07 runner reviewed (wf_28dada83-a37); fixes and decisions (about 15:00).**
  - **Defects confirmed and fixed by the review:**
    - leakage: BWM's selection set val_v2 was not excluded. A smoke train instance and an ID test instance were val_v2
      renamings. Fixed with `excl.with_selection_sets`;
    - Spearman pooled across sizes (now within each size);
    - the best-handcrafted baseline was selected on a different statistic from the one compared;
    - FlatMLP's E08 cells beyond n_max were misleading (`n_na` now marked);
    - the bootstrap seed is now allocated as `20261275:bootstrap`;
    - a seed guard;
    - an E08 goal-class guard.
  - **Applied by RCX:**
    - D-16: BWM `instances_seeds_fresh.json` is in the invariant union, but kept out of the b3_fresh regeneration
      check, which predates it;
    - `load_exclusions` now refuses a stale EXCLUSIONS.json (missing reserved sources);
    - BWM's v1 val set (`runs/BW_data/instances_val.json`) is added as a selection set, read only to build exclusions.
  - **Decision.** Train and val keep one instance per goal class: only p(n) − 1 goal classes exist per size (4, 6, 10,
    14 for n = 4..7), and duplicates of a class only add renamed duplicate rows at n ≤ 6. At n = 7, states per
    instance go from 3,000 to 10,000 to compensate.
  - **Before the full run (lane free):**
    1. rebuild EXCLUSIONS.json with `python -B -m rcx.bw.excl`, and check that the b3_fresh regeneration still matches;
    2. run `tests.test_bw`;
    3. run a new smoke into a fresh `--out`.
  - **Stale outputs.** The old smoke outputs are void; never resume an old `--out`.
- **CX D-18 N-VR seed 0 trained; E07 exclusions rebuilt; E07 full run started (about 19:25).**
  - **N-VR s0 (native, CPU, D-03 recipe).**
    - Wall time 9,769 s, of which 2,205 s were courtesy pauses on the shared thermal flag.
    - Final loss (mean of the last 50 steps) 0.014.
    - Adapter sha256 `2f376882…`, unmerged. Weights stay local and are git-ignored; the sha is in `TRAIN_META.json`.
    - Evaluation waits for ≥ 12 GB of free RAM (10.4 GB at 19:21).
  - **EXCLUSIONS.json rebuilt with D-16 seeds_fresh.** The invariant union went from 637 to 677. The b3_fresh
    regeneration still matches BWM's file. No BWM bytecode was written.
  - **Checks.** `tests.test_bw` 20/20 OK. A fresh smoke (`scratchpad/e07_smoke_r2`) confirms the review fixes are live.
  - **E07 full run** to `runs/RCX-E07/main`; estimate 0.6–1.4 h of CPU.
- **CX D-19 E07/E08 result R-04 (about 20:05). Outcomes known: yes.**
  - **Findings.**
    - The GNN generalizes structurally: OOD policy accuracy 0.998 vs MLP 0.563 vs h_add 0.848.
    - The GNN is the best GBFS critic at ≤ 200 expansions, but plateaus at 65/100 at 2,000, where misplaced_blocks
      reaches 87 and h_add 89.
    - BWM's unmet-goal ranking is weak (14/100 at 500).
  - **Decision.**
    - E07b refinement, preregistered before any run, on a fresh E08 instance set with new seeds 20261290–20261299. Arms:
      1. GNN-base;
      2. training extended to n = 8 with full goals;
      3. a pairwise successor-ranking loss;
      4. a hybrid: GBFS key (GNN, misplaced_blocks), or GNN + λ·misplaced, with λ chosen on val.
    - Primary endpoint: GBFS success at 2,000 expansions on n = 12–15.
    - Share R-04 with BWM now, framed as tier B and exact dynamics.
- **CX D-20 PREREG_E07B frozen (about 20:20). Outcomes known for E07b: no.** These things were known at the freeze:
  R-04's results on its own E08 set and E07 test, which are context only. `PREREG_E07B.md` sha256 `8a4cbde047f17eeb415ad9353459cc17f99e4f81c3a6008ba090db074af29cfb`.
  - Arms: A0 base, A1 n = 8 with full goals, A2 with a ranking loss, A3 GNN + λ·misplaced.
  - Primary: BEST (chosen on VAL-B) vs misplaced_blocks, success at 2,000 expansions on TEST-B n = 12–15.
  - Seeds 20261290–99 are reserved.
- **CX D-21 Joint B4 with BWM accepted in principle (about 20:00).**
  - BWM logged R-04 as X-02. B4 runs after BWM's B3, D-54 and item 7, under a co-signed prereg.
  - **RCX deliverables:**
    1. a robust `unmet(state, goal)` hook wrapping the E07b winner;
    2. the h\* label recipe for BWM's model-native value LoRA (arm b);
    3. a cache and the bookkeeping that `bw_select` uses.
  - The hook adapter is built now (CPU, code plus tiny tests). It is evaluated only inside the co-signed B4.
- **CX D-22 B4 hook built (`rcx/bw/b4_hook.py`; notes in `experiments/RCX-E07/B4_HOOK_NOTES.md`; 11 tests).**
  - **Interface facts read from BWM's `blocks_planner.py`** (sha 18a5f22b):
    - `unmet=` is keyword-only, and is bound by truthiness;
    - it is called positionally as `(state, goal)` on the initial state and on every imagined rollout state;
    - its value is only compared (lower is better), and must not be NaN.
  - **Hook properties.**
    - It never raises and always returns a finite value.
    - An inconsistent imagined state gets PENALTY 1000 plus misplaced_blocks on a sanitised copy.
    - It runs at 1 torch thread, with the caller's setting restored.
    - It undoes `rcx/bw/models.py`'s process-wide CUDA patch, so it is safe inside BWM's GPU process.
  - **For the B4 prereg:**
    - The fallback penalty gives the GNN arm a consistency preference that the default ranking lacks. B4 should add a
      control arm: the default unmet ranking with the same invalid-state penalty.
    - Report invalid-state counts per arm.
    - Consider `round_to` to keep plan_beam's prefix-length tie-break meaningful.
- **CX D-23 E07b implemented (`run_e07b.py`); prereg clarification accepted before any E07b run.**
  - **The issue.** PREREG_E07B's A1 asks for "the 10 largest-available fresh goal classes" at n = 8.
    - n = 8 has 21 classes. R-04's E07 test states use 10 of them and R-04's E08 set uses 11.
    - The strict reading, avoiding every R-04 class, leaves 5 and is infeasible.
  - **Accepted reading.** Avoid the classes of R-04's E07 n = 8 test states, which are the only n = 8 states evaluated
    in E07b (the policy endpoint). TEST-B is n ≥ 10 and is disjoint from R-04 E08 by instance key. This leaves 11, and
    the top 10 by generator frequency are used.
  - This protects every E07b endpoint from leakage, so it is recorded as a clarification, not a deviation.
  - **Other resolutions:**
    - the n = 8 instance draw uses seed 20261298 (with derived sub-seeds);
    - A2 hinge: all (optimal, non-optimal) successor pairs, margin 0.5, weight 1;
    - paired tests use seed 1 of each arm;
    - ties go to the smaller λ, then to arm order A1, A2, A3;
    - budget monotonicity is spot-checked by direct reruns.
  - **Estimate.** Full run 1.1–1.6 h; a smoke runs first.
- **CX D-24 RSM-001 unblocked by BWM's B3 report (about 20:30).**
  - **Plan (to be preregistered, after E01N seed 0 and E07b).** E11 option discovery and composition, using BWM's 211
    F-verified B3 plans as the teacher signal:
    - mine recurring macro-actions;
    - test whether an option-augmented action space or a macro library cuts GBFS/beam expansions on fresh RCX
      instances (exact dynamics);
    - test generalization to option compositions (AB, BA, conditional) unseen in mining.
  - No RCX evaluation on b3fresh or mirror; only BWM's result files are read.
- **CX D-25 N-VR seed 0 DEV result R-05 (about 20:45). Outcomes known: yes (DEV pilot).**
  - **Result.**
    - Canonical wording: rule reading is induced, with composite 0.964 vs OFF 0.325 and T2 0.400, and V3 sensitivity
      1.00.
    - Held-out paraphrases: wording-bound, with `ho_tpl` composite 0.594.
  - **Consequence for PREREG_E01N.** The primary-endpoint choice between `in` and `ho_tpl` must be made on instrument
    grounds before freezing, not by looking at N-CFSD. Proposed, and recorded now before N-CFSD exists: keep the
    primary on `in` as drafted, and make the `ho_tpl` contrast N-CFSD − N-VR a co-primary under Holm. Both test the
    same hypothesis (counterfactual/SD data improves rule reading), but `ho_tpl` is where N-VR shows headroom, and the
    `in` contrast has a ceiling at 0.964.
  - **Lane order:** E07b smoke (3 min); N-CFSD train (needs ≥ 12 GB RAM) and eval; E07b full; N-FX; SUPA.
- **CX D-27 E07b result R-07: primary outcome CLOSED (about 02:00 on 10-11; computed mechanically by the frozen
  rule).**
  - The A3 hybrid (λ = 1) becomes the RCX module for joint B4 arm (a).
  - A1 (n = 8 full-goal) and A2 (ranking loss) are not pursued further.
  - This is shared with BWM as an input to the B4 prereg.
- **CX D-26 PREREG_E01N FROZEN; SEALED commitment minted (2026-10-11 about 01:55). Outcomes known: yes for DEV seed 0
  of every arm (see the prereg header); no for SEALED.**
  - `PREREG_E01N.md` sha256 `975cca9a8041d53d12a08a2924ae565dcaf07eb8df09420936c8eadac6afa059`.
  - **SEALED commitment.**
    - `SEALED_COMMIT.json` sha256 `4f5f9780199067e652b12594fb76b776a8d84e6b2330212ae520faff31a15b32`, to be used as `open-sealed --expect-commit-sha`.
    - sha256(salt) = `8e5e08f1b7dc7a4fb39e33a874c1b5deba28b6074389f3de3be52c6d50943765`.
    - 40 parents per stratum, surfaces `in` and `ho_tpl`.
    - Required adapters: N-VR_s0..s2 and N-CFSD_s0..s2.
    - The salt is in `SEALED_SALT.secret`, git-ignored and unread until open-sealed.
  - **Frozen code (sha256):**

    | File | sha256 |
    |---|---|
    | `rcx/native/lora_cpu.py` | `ca469a13782f00fc8fd13c393e469d7166c57230e04b5a7d06927c541007e2d8` |
    | `experiments/RCX-E01N/native_e01.py` | `e4f630e0fbe3e1d1699f0ddac7ee498510923e3b0fe57272be85c026ccfcc0a5` |
    | `experiments/RCX-E01N/native_probe.py` | `e8e8723f50d8f1120047661cb9932a6e23d1a6310be9cb347a1909fe6096f8e0` |
    | `experiments/RCX-E01N/analyze_native.py` | `74866ab20d995d40486a64f2927f685747a9e3a4048b664272792e2b8298fdb3` |
    | `rcx/rcl/world.py` | `51e00a15c57e5ecfcaa53b8395b61d85118908378853d9e1d8cd5f3c56b7501d` |
    | `rcx/rcl/gen.py` | `fd89a80d90be4a66d4b0e875ec56e01993042d0e2d23de5f56f6508940fae744` |
    | `rcx/rcl/vocab.py` | `de9f1adbc0f4eec96939bf5037688ded0ef6f3ee75a7af4bcb3c5708505d8e55` |
    | `rcx/rcl/templates.py` | `c0d6f35544100d91bf2f0fec7095a4d7dba4b3dbb7fb79630450ce601eb5f642` |
    | `rcx/rcl/score.py` | `771fd12ca3755f054db23df974db753b2a12120da5c9d52d6134293b0477d22e` |
    | `rcx/stats.py` | `06a30c10acb6998033619560aac1b1130542a4344d0edf53e85e63386acf24b2` |

  - **Review.** The prereg was reviewed before the freeze (wf_45d6b236-470: statistics, custody, implementation), and all
    edits E1–E17 were applied.
    - Analysis switches: `P_TIES=both`, `DECISION_ORDER=ADVANCE,STOP,REVISE`, `ADVANCE_PER_SEED=True`,
      `PROTECTED_BASIS=ci`, `DROP_TRAIN_OVERLAP=True`.
    - Tests: 67 pass (native kit, stats, world).
  - **Seed 0 provenance.** The seed-0 `TRAIN_META` records `native_e01.py` `f023c201` (84be407). Its training code is
    identical to the frozen file (AST-checked by the reviewer).
  - **Next:**
    1. train N-VR s1, N-CFSD s1, N-VR s2 and N-CFSD s2;
    2. open-sealed;
    3. evaluate all arms on SEALED;
    4. analyse once.
- **CX D-28 RSM-001 E11 drafted (`experiments/RCX-E11/`, `rcx/bw/{macros,subgoal}.py`, 13 tests); coordinator
  decisions on its open questions (about 05:25).**
  - **Decisions:**
    - Q1: the primary is E11a with GBFS + A3 at 200 expansions, mined macros vs primitives, by exact McNemar, with
      Holm-graded controls.
    - Q2: the per-state A3 hook, to keep exact B4 parity.
    - Q3: sizes 120 and 96; power is not simulated (stated).
    - Q4: replicate seeds derived from the E11 block 20261256–59.
    - Q5: a single clear-to-table achiever.
    - Q6: the goal-structure holdout is the composition test; the registry's AB/BA template holdout is not run.
    - Q7: the shuffled-plan control excludes the mined macros (tested).
    - Q8: the random control excludes the mined macros' full keys (step pattern plus goal annotation) but not bare
      step patterns. Only 3 sound length-2 patterns exist and the mined library uses 2, so stricter exclusion would make
      a length-matched random control impossible.
  - **Teacher data.** BWM `BW_DISTILL_r1/solutions.jsonl` sha256 `fdf80151…`, read-only. All 211 verified plans replay
    exactly under the simulator. No off-limits file was read.
  - **Next.** Run the smoke when the CPU lane is free, then freeze `PREREG_E11.md`, then the full run (about 2–4 h).
- **CX D-29 Confirmation chain progress.**
  - N-VR s1: adapter `52adbde9…`, 6,084 s.
  - N-CFSD s1: adapter `8e710e46…`, 6,189 s.
  - N-VR s2 started at 05:19; N-CFSD s2 follows.
  - No SEALED access so far: the salt is unread.
- **CX D-30 Confirmation adapters complete; SEALED battery opened (2026-10-11 08:40). Outcomes known for SEALED: no.**
  - **Adapters**, all 190 steps with no resume checkpoints (sha256 prefixes):

    | Arm | s0 | s1 | s2 |
    |---|---|---|---|
    | N-VR | `2f376882…` | `52adbde9…` | `876ac52d…` |
    | N-CFSD | `97cc48e8…` | `8e710e46…` | `652b497d…` |

  - **Opening.** `open-sealed --expect-commit-sha 4f5f9780…` ran and the battery commitment was checked: seed 88527345,
    2,240 records, battery sha256 `c87765ea…`.
  - **Training overlap: zero.** 0 semantic parents, 0 skeleton parents, and no SEALED status word in any training
    prompt.
  - `SEALED_OPEN.json` sha256 `22ccc4344fa547944d3a999ecd2f17cae901cb81774059805bf77b327a9796b6`. It contains the
    revealed salt and is committed before the first SEALED evaluation, per PREREG_E01N §6.
  - **Next.** The SEALED evaluation chain: N-VR and N-CFSD s0–s2, OFF and T2 (`confirm_sealed_eval_chain.sh`), then
    a single analysis with `analyze_native.py --battery sealed`.
- **CX D-31 PREREG_E11 FROZEN (2026-10-11 about 08:45). Outcomes known: smoke plumbing only (disclosed in the
  header).**
  - `experiments/RCX-E11/PREREG_E11.md` sha256 `c67f036d530ed1f961f84caf2e9b2e598562190ccf1722069379807046b5f40c`, pinned in `run_e11.py`.
  - **Smoke** (`runs/RCX-E11/smoke`, 28 s):
    - every gate passed (G1–G3, BWM bytecode clean);
    - A3 hook accounting is exact;
    - teacher custody: 240 rows, 211 verified, 211 trajectories matching;
    - budget monotonicity: identical.
  - **Seeds:**
    - 20261256: E11a instances, plus derived R, S, mono and bootstrap seeds;
    - 20261257: E11b small sets;
    - 20261258: E11b execution set, plus derived seeds;
    - 20261259: the selector, with derived replicate model seeds 816062654 and 250692302.
  - **Settings:** `exclude_noop_macros=True`; `b_label_target=achiever`; arms P, M, R, S and G (G descriptive);
    compute thresholds 500, 1,000, 2,500 and 5,000.
  - **Scheduling.** The full run follows the SEALED E01N evaluation chain in the single CPU lane (estimate 1–2 h).
- **CX D-32 Lane sequencing fix (2026-10-11 12:45). No outcome is involved.**
  - The original wrapper would have started E11 the moment the SEALED chain ended. The one SEALED analysis would then
    have run beside E11, breaking the single-lane rule (I-02).
  - I stopped only the wrapper. `confirm_sealed_eval_chain.sh` kept running untouched: OFF was in progress and no cell
    was restarted. The six adapter evals (raw outputs only) were committed in 2dd1ad0. The analysis has not been run.
  - `confirm_post_sealed_chain.sh` waits for `SEALED CHAIN DONE`, then checks that all 8 `eval_sealed_*` files exist
    (otherwise it stops). It then runs `analyze_native.py --battery sealed` once (outputs are create-only) and, only if
    that exits 0, runs the frozen E11.
- **CX D-33 RCX-E01N confirmation outcome: REVISE (2026-10-11 14:07; R-08). Applied as preregistered, with no
  reinterpretation.**
  - `analyze_native.py --battery sealed` ran once (create-only outputs, rc 0, 11 s):
    - C1 `in`: +0.018 [+0.006, +0.030], Holm p 0.002.
    - C2 `ho_tpl`: +0.090 [+0.069, +0.113], Holm p 0.0002.
    - All per-seed diffs are > 0. The protected strata are within margin.
    - ADVANCE failed only on the +0.10 point threshold.
  - **Consequences fixed by PREREG_E01N:**
    - No E33 retention request to RCWM and no CX Q-03 cross-scoring; both are conditional on ADVANCE.
    - No retained or dense-model claim.
    - The N-CFSD adapters stay local, unmerged component artifacts.
  - **The held-out wording pool `ho_tpl` is spent for confirmatory use.** Per-wording results were seen on DEV (R-06)
    and SEALED (R-08). The SEALED battery is likewise opened and used.
  - **Next recipe: RCX-E01R.** The target is the specific residual failure: unseen restrictive *positive* phrasings
    are read as "apply". The design must use:
    - a new held-out wording pool, authored blind to the training pools and audited for near-duplicates;
    - a new salt-committed SEALED battery;
    - matched compute against N-CFSD as the control.

    The first step is a descriptive diagnostic on the existing adapters before any new training. Details are in
    NEXT_STEPS.
- **CX D-34 RCX-E01R wording pools fixed (2026-10-11 about 14:25). No model has seen them, and no outcome exists.**
  - **Authoring** (workflow wf_d006d143-69e):
    - 3 blind authors produced 143 unique templates: 6 cells (POS/NEG × EXPLICIT listed-first / EXPLICIT
      complement-first / IMPLICIT), 23–24 per cell.
    - All 143 passed unanimous label-blind verification by 3 verifiers at list sizes 1, 2 and 3.
    - **Limitation:** no planted ambiguous controls were included, so the verifiers' sensitivity to ambiguity is
      untested.
  - **Transfer.** The pool was written to `POOL_ALL.json.secret` (git-ignored), sha256 `aa86ea65…`. Its FNV-1a32
    (`9d8266ab`) matched the workflow payload, so the transcription was exact.
  - **First audit and split** (`split_pool.py`, POOL_SPLIT.json; `wording_audit.py` as of 52e4c65):
    - drops: 9 for status words, 11 near-duplicates of RCX templates, 22 within-pool near-duplicates;
    - split: P2 49 / P3 49 / spare 3, by cell order under a fresh secret salt.
  - **Second pass, stem drops (`refine_pool.py`, POOL_REFINE.json).**
    - Reading P2 showed inflections of status words: "posting"/"post" (anchor "posted") and "rejection" ("rejected").
      `wording_audit.stem_hits` drops any wording with a word sharing ≥ 4 leading letters with a real status-like word.
    - On P2 it flags exactly those 4 wordings and nothing else.
    - It removes items only, to avoid moving read P2 wordings into P3. Result: **P2_v2 45, P3_v2 48** (sha256
      `88dd432b…`, never read by the lead), spare 3.
  - **DEV2 battery for step A built:** seed 20261261, 1,320 records (`in` 660, `p2` 660), sha256 `38362338…`, all 45
    P2_v2 templates used. It is evaluated after E11, in the single CPU lane: N-CFSD s0–s2, N-VR s0.
- **CX D-35 RCX-E11 outcomes (2026-10-11 15:30; R-09). Applied by `run_e11.py`'s frozen decision functions.**
  - **E11a ADVANCE.**
    - M vs P @200: 77 vs 52, discordant pairs 25/0, p = 6e-8.
    - Holm attribution vs R (p = 0.0078) and S (p = 0.00024). M − G = 14.
    - Compute claims are worded per heuristic call. Per primitive step, M is ahead only from 1,000 steps.
  - **E11b REVISE.**
    - Primary −0.074 [−0.139, +0.0006].
    - The order is longer than ready-first (+0.018).
  - **Follow-ups (in NEXT_STEPS; none preregistered yet):**
    - (i) Offer BWM the mined library M as a candidate action-abstraction arm for its beam planner (B5 idea; BWM's
      call).
    - (ii) E11c: replace achiever labels with execution-aware labels (OPT_A-derived subgoal orders), since label
      learning is at ceiling.
    - (iii) Report the heuristic dependence of E11a's specificity with every citation of R-09.
- **CX D-36 Shared CPU-lane lock adopted (2026-10-11 about 16:12; agreed with BWM, its D-76).**
  - `rcx/lane.py`, with 5 tests. Lock `%TEMP%/rineya-cpu-lane.json` = {holder, job, pid, started, expected_end,
    token}, created with O_EXCL.
  - **Waiting markers:** `rineya-cpu-lane.waiting.<party>`.
  - **Staleness:** a lock whose holder pid is dead is reclaimed (renamed `.stale-<pid>.bak`).
  - **RCX policy:**
    - if BWM holds the lane, write the marker and wait (BWM yields between instances of its resumable Qwen3-4B CPU
      comparator);
    - if the lane is free but RAM is short (a GPU turn), wait **without** a marker;
    - hold the lane for a whole block, then release it on exit.
  - **Priority:** RCX's E01R blocks take precedence. BWM gets the gaps and the nights.
  - **Step A restarted under the lock** (`step_a_chain_v2.sh`). The v1 chain was stopped while still waiting for RAM,
    before any eval; nothing was lost.
- **CX D-37 PREREG_E01R §2 (the arm choice) and the R1/R2/R3 recipe specifications FROZEN before any step A output
  (2026-10-11 20:22).**
  - **State at the freeze.** `step_a_chain.log` has 0 START lines and no DEV2 prediction file exists. The chain has
    been waiting for RAM since 15:31 (last at 11.5 GB < 12.5 GB).
  - **Frozen artifacts:**

    | File | sha256 |
    |---|---|
    | `experiments/RCX-E01R/PREREG_E01R_DRAFT.md` | `4ceb8250de737d4a0fe55feb7d4303ebb27773008ac552436f230423d9b23700` (only §2 and the R1/R2/R3 specifications are frozen; §4–§11 remain a draft) |
    | `experiments/RCX-E01R/decide_branch.py` | `6835b0402120a2847423d8af08814cc89f311292b110365487b39d61636080d6`, with 6 tests |
    | `experiments/RCX-E01R/recipes.py` | `fbc5e309…` (R2's mirrored S− pairs are still to be added, per the frozen spec) |
    | `experiments/RCX-E01R/trace_target.py` | `8800a07d…` |
    | `pools/TR_v1_candidates.json` | `b8e46463…` |
    | `pools/VERIFIER_CALIBRATION.json` | `b9f42451…` |
  - **Branches, in order:**
    1. **B0** instrument fault: a missing or sha-mismatched N-CFSD seed, n = 0, or a_in < 0.85;
    2. **B1** H-alt: min_f p_f ≥ 0.80, leading to the P3 coverage check;
    3. **B3** E − I ≥ 0.15 with the template-cluster bootstrap lower bound > 0, leading to R1 + R3;
    4. **B2** p̄ ≥ 0.65, no training;
    5. **B4** otherwise, leading to R2 + R3.
  - **Review fixes adopted** (workflow wf_b87c2ed3-8e0, reviewer items 1–4, 7 and 9):
    - B3 is tested before B2, with a directional criterion and a cluster bootstrap;
    - all 3 seeds are required;
    - R2 is mirrored for S− parents, and its rule-type shift is a declared confound;
    - R3 is scored from the last non-empty line, with a pre-run smoke test on TRAIN prompts only;
    - R1 uses the TR_v1 pool, audited against P2_v2 and P3_v2 with stricter thresholds (Jaccard ≥ 0.35, or a shared
      4-gram), with counts only for P3.
  - **Verifier calibration.** In one label-blind run (3 verifiers), the protocol:
    - rejected **12/12 planted ambiguous** wordings;
    - passed **11/11 clear decoys** and **23/23 P2_v2 re-tests**;
    - passed 141/141 TR_v1 training candidates.

    This removes the D-34 limitation: the verifiers' ambiguity sensitivity is now measured.
  - **Remaining review items** (5, 6, 8, 10–19: the E01R row reader, guard power, the degenerate-REVISE rule, P3
    leakage hardening in `audit_pool`, seeds against git history, eval-time re-estimate, and others) are resolved
    before the full freeze. They do not depend on step A.
- **CX D-38 PREREG_E01R guard margins, C1 threshold, guard count and degenerate rule FIXED before any step A END line
  (2026-10-12 01:55). The values come from design constants only; no step A or DEV2 output was read.**
  - **What it replaces.** The "PENDING STEP A" margin revision in §7 of the reviewed draft (sha `ac50a4ab…`, committed
    together with this entry) is superseded. The margins are fixed now from the expected SEALED parent counts, with
    an assumed per-parent SD of 0.2 and correlation 0. The pass condition is |m| ≥ 3.24·SE, which gives P(pass | true 0)
    ≥ 0.90.

    | Guard | Margin | Basis |
    |---|---|---|
    | G1 | −0.03 | unchanged |
    | G2 | −0.05 | unchanged |
    | G3 | +0.02 | unchanged |
    | G4 (≈ 340 parents) | **−0.05** | |
    | G5 (≈ 100) | **−0.10** | SE ≈ 0.028 |
    | G6 (≈ 140) | **−0.08** | SE ≈ 0.024 |

  - **C1 threshold and guards.** The C1 ADVANCE threshold stays **+0.25** under every branch, and the six guards are
    fixed. The B3 high-p̄ case (headroom near +0.25) is an accepted limitation, not a later decision.
  - **Degenerate trade-off.** "STOP (degenerate trade-off)" triggers when a G4–G6 **point estimate** is ≤ −0.15, not
    a CI bound (prereg reviewer finding 4).
  - **REVISE** also requires a C1 point estimate > 0 (prereg reviewer finding 9).
  - **Remaining items** from the E01R build reviews are resolved before the full freeze:
    - prereg findings 2, 3, 5–8 and 10–16;
    - code findings 1–10 and T1–T6;
    - in particular: no smoke exemption (a smoke failure means no opening); P3 read only if the visible R1 audit meets
      the minimum; the mint binds the arms to the branch; non-anchor sub-guards are reported.

    None depends on step A outcomes.
- **CX D-39 RCX-E01R step A done; the frozen branch rule gives B4 → treatment arms R2 + R3 (2026-10-12 03:20; R-10).**
  - **Run.** Step A ran 01:27–03:20 under the CPU-lane lock: N-CFSD s0–s2 and N-VR s0, all rc 0.
  - **Branch rule.** `decide_branch.py` (`6835b040…`, frozen in D-37) ran once and wrote `BRANCH_STEP_A.json`:
    - p_LF / p_CF / p_IM = 0.509 / 0.346 / 0.639;
    - p̄ = 0.496;
    - E − I = −0.213 [−0.504, +0.076];
    - S = 0.293;
    - q_LF / q_CF / q_IM = 0.917 / 0.747 / 0.667;
    - a_in = 0.996.

    So the branch is **B4**.
  - **The R1 path is unused.** `audit_tr_pool.py` is never run, so P3 is not read before opening.
  - **Next.** The remaining E01R review items are resolved for the B4 path. Then the full freeze, row build, audit,
    SEALED mint (before any training), and 6 adapters (R2 and R3, seeds 0–2), scheduled on the CPU lane with peers.
