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
