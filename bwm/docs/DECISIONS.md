# Decision log — BWM-001 native pilot on Titan (native_v1)

Each entry: date (CDT), decision, reason, evidence. Earlier entries are never edited; corrections are new entries.

## 2026-10-05

- **D-01 Prompt frame.** Use Titan's chat frame (system rule text + user compact JSON, thinking disabled, generation
  prefix `<think>\n\n</think>\n\n`), not the spec's raw `bwm-json-raw-v1`. Reason: Titan's transition prior was
  trained and evaluated in this frame; the raw frame never shows it. Evidence: zero-shot Titan answers forward prompts in
  its own bare-state convention `{"x":…}` (format engaged, semantics wrong: ignores limit, sign errors on halve).
- **D-02 Backbone.** Titan-Sparse (FP32, sha a28629c9…), frozen, never merged, FP32 end to end (BF16 including autocast
  would round its update away).
- **D-03 Adapter recipe.** LoRA r16/alpha32/dropout 0.05 on q,k,v,o,gate,up,down; AdamW (0.9, 0.95); WSD 10/70/20;
  clip 1.0; loss on target + EOS only (supervised positions only); gradient checkpointing (measured: without it peak
  memory 17.6 GiB, spilling into shared memory; with it 3.0–3.3 GiB).
- **D-04 RW2 domain.** RegisterWorld-v1 law with tight budgets (opt+1+slack). Reason: v1 greedy floor 95.9%.
- **D-05 F_LM qualifies (G2).** Forward LoRA on Titan-Sparse, 1,455 ID transitions, 300 updates: one-step accuracy
  0.9966 on the full ID table (in-sample: training covers every ID key) and 0.9948 on |x| > 64 at limits 96/128.

## 2026-10-06 — design critique (3 of 4 critic lenses completed; synthesis done by the lead)

- **D-06 Goal-held-out split (blockers A1/SC-01/S2).** Suffix relabelling turned training sub-problems into test
  prompts (lookup floor ≈ 0.94) and the limit never binds. Now: family = (start, goal); split by hash of the goal
  value (65/10/5/20); limit is a nuisance token drawn per family. Verified: 0 training rows carry a test goal.
- **D-07 F rule (blockers S1/SC-02).** Primary F-on rule = `rerank`: best predicted-successful candidate, else the first
  valid candidate (the no-F rule). The draft's "abstain unless predicted success" becomes the named operating point
  `gate` (A01g/A11g), reported with coverage. `would_abstain` is logged every round.
- **D-08 Common random numbers.** Candidate 0 greedy; candidates 1..K-1 sampled with per-candidate seeds
  hash(eval_seed, family, t, j) via a custom batched Gumbel-max sampler (greedy parity with HF generate 200/200;
  sample invariant to batch composition 25/25).
- **D-09 Training budgets.** Training slack drawn from 0..6 (budgets up to 15) so OOD-long budgets are not novel tokens.
- **D-10 Baselines and controls.** Exact-dynamics greedy-1/2 floors; greedy-1/2 and BFS (16/64/256 calls per decision)
  under the same F_LM; random proposer + F; frozen Titan ± F; outcome-flip (`failure` requested) and budget+4 controls;
  novel-limit-only split; shuffled-state training control; A01 at K=8/12 for the token-matched view; exact-F oracle arms.
- **D-11 Framing.** RW2 is exhaustively searchable (BFS under exact F with 256 calls/decision: ID 1.00, long 0.98,
  range 0.89 success). Proposer claims are therefore stated as amortization: success vs F calls and tokens.
- **D-12 Statistics.** Unit = parent; clusters = held-out goal values (37 ID-test goals); crossed seed × goal-cluster
  bootstrap; seed-level t interval and per-seed exact McNemar as sensitivity analyses.
- **D-13 Shared GPU.** All GPU jobs run under the shared lease (`bwmn/gpu_lease.py`, wrapper `scripts/gpu_run.py`),
  agreed with the concurrent RCWM-T1 session.
- **Rejected/deferred:** 5-seed requirement is the target but the decision is staged (pilot seed first, then seeds
  2–5); E5 (pruning an unmerged, router-gated adapter) deferred — no retention question while the adapter is off by
  default; F_MLP one-hot sidecar dropped (≡ exact on ID, cannot represent range states).

## 2026-10-06 (continued) — incidents and fixes

- **I-01 CUDA fault, P_actions_s1 attempt 1.** `CUDA error: unknown error` at ~step 120 (10:41) while another session's
  unleased GPU test shared the card (VRAM 6.45 GB vs our ~3.3 GiB). Preserved at `runs/_failed/P_actions_s1_attempt1`.
  Peer confirmed its process and now leases every GPU touch.
- **I-02 Paging stall, P_states_s1 attempt 1.** State-and-action batches carry up to ~2,240 supervised tokens; the full
  [N, 151,936] FP32 logits + softmax + gradient (~4 GB) pushed the card into WDDM shared memory (7.8 GB, no progress
  past step 1 for 7 min). Preserved at `runs/_failed/P_states_s1_fail105009`.
- **D-14 Chunked checkpointed cross-entropy** (`lm.chunked_ce`, chunks of 256 supervised positions, logits recomputed in
  backward). Verified exact on the worst-case batch: loss 0.606332 vs 0.606332, max gradient difference 4.1e-7 (max
  gradient 0.14). Live peak memory for state-and-action training: 3.42 GiB. All proposer trainings use this loss;
  F_titan_main (trained earlier with the unchunked loss) is unaffected in definition (same objective).
- **D-15 Queue hygiene.** Queue scripts are idempotent (`runs/lib.sh: train_job` skips finished adapters, preserves
  failures under `runs/_failed`, retries once). Note: stopping a background task does not kill Git-Bash child
  processes; stop queues by killing the `bash.exe` processes by command line.
- **I-03 Paging, pilot eval attempt 1 (12:03).** HF `DynamicCache` grows by `torch.cat`, so the KV cache transiently
  exists twice; a 12,000-token batch budget put the eval process at 7.18 GB dedicated + 0.6 GB shared. Preserved at
  `runs/_failed/EVAL_pilot_s1_paging_121008`. **D-16:** generation batch budget 6,000 tokens (`lm.sample_crn`,
  `lm.generate`); proposer decode caps 80 (actions) / 180 (states) tokens (longest training target ~103).
- **I-04 Machine reboot 13:15:10.** All GPU jobs on both sessions died (pilot eval had finished only the model-free
  baselines). **D-17:** GPU work now runs through `scripts/orchestrate.py`, launched via WMI `Win32_Process.Create`
  (outside the session's process tree), reading an editable `runs/jobs.json`, idempotent via completion markers,
  retry-once, failures preserved.
- **D-18 GPU fairness with the RCWM-T1 session.** Lease is not FIFO; our wrapper adds a 30 s courtesy gap before every
  acquire and honours one-shot explicit yields (`runs/.yield.json`).
- **F-01 (finding, validation-grade) F_LM extrapolates arithmetic but not the limit rule.** Over the frozen F table
  (3,705 keys): ID 0.9966 (in-sample), |x| > 64 at new limits 0.9948, three-digit outputs 0.907, but the 96
  legality-flip doubles (|x| in 33–64 doubled at L = 96/128: legal, yet illegal at every training limit) only 0.365.
  F predicts "illegal" from the magnitude pattern of the three training limits instead of comparing with the limit
  in context (and errs both ways: −52 double at L=96 predicted legal → −104). Cost on planning (validation, model-free
  planners under F_LM vs exact dynamics): BFS-256 range 0.720 vs 0.89; greedy-2 range 0.517 vs ~0.68.
  Follow-up queued (E2-xlim): F trained with extra limit values {16,24} or {8,16,24,40,56} (all numbers ≤ 64).
  F_titan_main remains the frozen F of the factorial.
- **I-05 Orchestrator deaths (13:17, 13:49, 13:50).** Direct WMI launches of the venv `python.exe` (uv trampoline)
  exited with STATUS_CONTROL_C_EXIT (0xC000013A) right after spawning a child, even with SIGINT ignored (reproduced with a
  dummy script under both WMI and Task Scheduler; children in a new process group survived). WMI -> `bash.exe` ->
  python survived repeated tool calls in a side-by-side test. **D-19:** launch via `runs/start_orchestrator.sh`
  (WMI -> bash -> orchestrate.py). Orphaned job children from dead orchestrators were killed before relaunch so no
  duplicate job could run.
- **I-06 Machine stability (reported by the RCWM-T1 session, verified in the System log).** nvlddmkm TDR resets
  (event 153/14) at 10:42 (our P-actions crash), 10:49, 12:40, 13:14:51 (19 s before the 13:15 restart), 14:02–14:04;
  unexpected shutdowns (Kernel-Power 41) at 23:11 on 10/5 and 10:30 on 10/6 — both coincide with our lost sessions.
  The RTX 4070 Laptop also drives the user's display; sustained FP32 compute and VRAM spills starve the compositor.
  **D-20 safeguards in `bwmn/lm.py`:** `set_per_process_memory_fraction(0.70)` (fail fast instead of paging);
  `gpu_yield()` = synchronize + 50 ms sleep after every training step / generation batch / scoring batch, 10 ms every
  16 decode steps; generation batch budget 3,000 tokens; scoring batch 16; thermal guard (nvidia-smi at most every
  20 s; pause at >= 84 C until <= 78 C); one GPU process at a time under the lease. Cost: slower jobs; results are
  unaffected (no change to math, seeds or decoding).
- **I-07 Thermal (reported by RCWM-T1).** GPU reached 86–91 C under both sessions' jobs; our first in-process guard
  (84 C, sampled every 20 s) never fired in the pilot eval (cause not pinned down; temperature reads work in that
  process context). **D-21:** guard redesigned in `lm.thermal_guard`: honours the shared watchdog flag
  `RCWM-T1/runs/.thermal_pause` on every yield; samples every 10 s; pauses at >= 82 C until <= 75 C; unreadable
  temperature = hot (fail safe); inter-batch yield stretched to 0.3 s while >= 78 C; temperature logged every 5 min.
  Verified: with the peer flag set the guard paused 162 s and resumed at 62 C. Stop rule: any further TDR under a
  BWM job halts BWM GPU work and is reported to the user.
- **I-08 92 C at 145.8 W (14:26) during the pilot eval** (shared watchdog log): the flag reacts only after a burst;
  this card climbs ~20 C in < 3 s at full boost. **D-22 proportional duty cap** (`lm.gpu_yield`): after every
  generation batch / 8 decode steps / scoring batch / training microbatch, synchronize and idle
  max(0.12 s, 3.0 x GPU time since the previous yield) (~25% duty; `BWM_DUTY_RATIO`), then thermal guard.
  Generation forwards <= 2,000 tokens; scoring batches of 8; training now uses 4-row microbatches with
  supervised-token-weighted gradient accumulation (same gradient as the full-batch token mean up to float order;
  seed-1 adapters were trained before this change with full 16-row forwards). Expected cost: jobs ~4x slower.
- **D-23 Re-plan under the slowdown.** Pilot read first: A00/A10/A01/A11 on 200 ID-validation tasks for the
  step-800 and step-400 seed-1 adapters; seeds/steps for stage 3 decided from that read.
- **D-24 CPU evaluation lane.** Measured CPU generation (Ryzen 9 7940HX, FP32): 32 greedy proposals in ~25 s at 8 or
  16 threads (bandwidth-bound) — roughly 3–6x slower than the duty-capped GPU but free of GPU/display/thermal risk.
  Evaluations now run in a second queue (`orchestrate.py --cpu`, `runs/jobs_cpu.json`): BWM_DEVICE=cpu, 8 threads,
  no duty cap, 8,000-token generation batches; GPU queue keeps training/E2. Rule: all arms of one eval run (a paired
  comparison) execute on the same device, so CPU/GPU float differences never enter a paired contrast. The F table is
  the frozen GPU-computed table in every case.
- **I-09 GPU idle ~1.5 h behind a stale lock (15:03–16:07).** The lease's foreign-holder rule ("python > 512 MB
  present in nvidia-smi compute apps") was always true because, under WDDM, nvidia-smi lists every process that merely
  touched the CUDA driver — including our CPU-lane eval (torch.cuda.is_available()) and an agent's CPU probe — so the
  waiting job never reached the stale-lock reclaim path. **D-25:** (a) CPU-lane processes never call into CUDA
  (`lm.cuda_ok()` checks BWM_DEVICE first); (b) vendored `gpu_lease.foreign_gpu_holder` counts a python process as a
  holder only if its dedicated GPU memory (Windows counter `\GPU Process Memory(*)\Dedicated Usage`) exceeds 256 MiB
  (CUDA-touching CPU processes show ~160 MB; model jobs 2.5–7 GB); falls back to the original rule if counters fail.
  Reported to the RCWM-T1 session.
- **D-26 Conditional one-for-one GPU alternation** (RCWM-T1's protocol): waiting markers
  `rineya-gpu-0.waiting.{bwm,rcwm}.json`, `rineya-gpu-0.turn.json` on release; a side whose turn was last waits only
  while the other side has a live waiting job (45 min cap). No idle GPU when only one side has work.

## 2026-10-06 — solutions and benchmark plan (research workflow wf_258efb87)

- **DEV-01 Protocol deviation (recorded, quarantined).** During the fix investigation an agent ran BFS planners on
  RW2 *test* splits (range/test, limitonly/test) in its scratchpad. Those numbers (e.g. limitonly 0.967 vs 0.940)
  are quarantined: no decision may use them; every fix is selected on validation splits only.
- **F-02 (finding, validation/CPU probes) the shortcut is forced by the data, and the backbone already has the
  skill.** In any training set with numbers <= 64, "double with |x| > 32 is illegal" is right 100% of the time, so
  extra limits <= 64 (E2-xlim) cannot remove it. With the adapter off, Titan answers "is |2x| <= L?" correctly on
  94/96 flips; with legality forced, F's arithmetic is right on 320/320. Correcting only the 96 flip keys moves
  BFS-256 range_val 0.720 -> 0.867 (exact 0.880). The legality-token margin flags F's errors at AUROC 0.923.
- **D-27 Plan.** Track A (fix F): A1 factorized F (F_LM candidate + adapter-off verifier; code-check ceiling;
  calibrated P(legal) threshold) — no training, CPU; A2 F_vlab (single-call F trained on frozen-backbone
  self-labels at limits 72–120, no ground-truth label above 64); A3 slack-scratchpad vs boolean target (mechanism),
  xlimB as raw control, xlimA cancelled. Common gate FLIP-EVAL v1: pooled flip-type >= 0.90 (Wilson LB >= 0.85),
  odd held-out limits {21,37,45,53,61} >= 0.97, no stratum regression > 1 pp; passing arms get full tables and
  validation BFS. Track B: PlanBench Blocksworld (600, Task 7, Hard, unsolvable), VAL-scored, paired per instance
  against published plans re-scored with local VAL (reproduces GPT-4 210/600, o1-preview 587/600); then Mystery
  Blocksworld Regime A (renaming robustness) and Regime B (rules from prompt). Track C: same-harness Qwen3-4B
  comparators (needs >= 12 GB available RAM, CPU lane idle), FLOPs as the primary cost measure. Comparability ledger
  to be preregistered before any PlanBench run; "beats GPT-4" is not a claim (specialist + search vs zero-shot
  generalists); headline candidates are iso-FLOP non-inferiority vs Qwen3-4B-thinking and generalization (Hard,
  Mystery Regime B).
- F_titan_main remains the preregistered F of the RW2 factorial; F_fact / F_vlab enter only as added arms via a
  PREREG amendment before the freeze.
- **B-01 PlanBench harness (Track B) built and validated.** Vendored LLMs-Planning assets (MIT; commit fc638a1a;
  VAL LGPL-3, unmodified) under `bench/planbench/` (test-only; SOURCE.md). `bwmn/blocks.py` (exact 4-op semantics,
  PDDL parser, BFS oracle, generator, VAL wrapper): BFS optimal lengths match the published ground truth on 120/120
  checked instances; oracle planner 602/602 with VAL agreeing with the exact executor on 602/602 (0 disagreements).
  **Leakage finding:** the 602 PlanBench instances are only 347 distinct problems up to block renaming, and 54% of
  random 3–5-block problems from the same generator are renamings of a test instance; training data
  (`scripts/blocks_data.py`) excludes every test signature, so it is 4–6 blocks (4: 224, 5: 3,233, 6: 1,456 train
  instances) while the test is mostly 3–4 blocks. **Ledger** (`bench/planbench/LEDGER.json`): all published Task-1
  plans re-scored with local VAL per instance — zero-shot: DeepSeek-R1 595/600, o1-preview 587, LLaMA-3.1-405B 386
  (paper 376), Claude-3-Opus 356, o1-mini 340, GPT-4o 213, GPT-4 210, LLaMA-3-70B 205; one-shot GPT-4 206.
- **D-28 GPU economy.** GPU = training only. Zero-shot gradings and adapter gradings run on the CPU queue (jobs with
  an `after` dependency on the training marker; orchestrator gained dependency support). Order on GPU: E2 r256 (in
  flight) -> BW forward model -> BW action proposer -> remaining E2 trainings -> xlimB.
- **D-29 Track A execution.** A1 (factorized F; 3,476 keys; CPU lane 2) running. Smoke test (32 keys) showed the
  verifier fails at |c| == L ("is 32 <= 32" -> false) and the forced decode copies x for illegal add1/sub1, so every
  factorized variant applies its legality source to doubles only (F_LM keeps add1/sub1/halve/finish). Lane-2 chain:
  A1 analysis -> A2 self-labelling at L in {72,80,88,104,112,120} (|x| <= 64; doubles labelled by the frozen-backbone
  verifier, add1/sub1 by F_LM) -> `vlab_sft.jsonl` (exact ID table + self-labels; no exact label above 64) -> GPU job
  `a2-train-Fvlab` (dependency on the label file) -> CPU grading. Both orchestrators are being handed over at their
  next job boundary to the dependency-aware version (`runs/{cpu,gpu}_handover.sh`).
- **PREREG_PLANBENCH.md** drafted before any BWM system touched a PlanBench instance (to be frozen with adapter
  hashes and planner settings selected on generated validation instances only).
- **X-01 (cross-session evidence, RCWM-T1 pilot, reported 2026-10-06, not re-measured here).** Titan-Sparse's
  one-step skill on its own families, on steps whose input state was correct: register swap 1.00, add 0.84 (missing
  mod 97), conditional 0.44–0.51, mix 0.008; ledger posted scale 0.65 / add 0.53 / transfer 0.32, void events
  0.00–0.12 (applied anyway); prerequisite applicable 0.33, inapplicable 0.009 (always applied — the 384 training
  positives never contained inapplicable actions). Consistent with F-01/F-02: Titan learned surface operations, not
  conditional rules. Design consequence already adopted: every BWM forward model trains on illegal/no-op transitions
  explicitly (RW2 ~20%, Blocksworld ~50% illegal rows). Our trainer (`bwmn/lm.py`, `scripts/train_generic.py`) was
  shared with that session for a rule-coverage LoRA (vendored copy, not imported in place).
- **D-30 CPU-queue triage.** Measured CPU eval cost: A00 (200 tasks) 65 min; A10 > 2.5 h on the contended CPU.
  The pilot read keeps A00/A10/A01/A11 on 200 ID-validation tasks; the step-400 comparison is reduced to A00;
  `eval-pilot-core` (long/range) and `eval-pilot-diag` are deferred (`runs/jobs_cpu_deferred.json`). E2 gradings
  (~25 min each) stay. Prompt-prefix sharing across the K candidates will be introduced only for fresh evaluation
  runs (never mid-comparison), so every arm of a paired comparison uses one implementation.
- **I-10 Bugcheck 0x19C at 19:10 (WIN32K_POWER_WATCHDOG_TIMEOUT).** The laptop entered Modern Standby on idle
  timeout at 19:08:38 while jobs ran (our two CPU lanes + the peer's GPU job) and bugchecked; reboot 19:10:20.
  The earlier unexplained shutdowns (23:11 on 10/5, 10:30 on 10/6) fit the same idle-standby pattern. Lost: the
  pilot A10 arm (~4 h of CPU, not checkpointed); A1 resumed from its per-key JSONL.
- **D-31 Keep-awake (user-approved via AskUserQuestion, 19:2x).** Each orchestrator holds
  SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED) while a job runs and releases it
  when its queue is idle (display stays on during runs). **D-32 Crash-resilient evaluation:** evaluation arms run as
  50-task slices (`run_eval.py --slice`), merged by `merge_parts.py` (refuses partial merges); valid because sampling
  uses per-candidate seeds, so results do not depend on batch composition. Remaining pilot arms (A10/A01/A11, and the
  step-400 A00) use the shared-prefix sampler (`--share-prefix`; identical outputs verified 24/24, ~38% faster); the
  pilot A00 (step 800) ran with the original sampler. A second CPU lane runs as an orchestrator (`--cpu2`): A1 pass ->
  A1 analysis -> A2 self-labelling -> emit.
- **F-03 (finding, Track A1, validation-grade; selection on calibration limits 80/112 only; single F seed).**
  Training-free fixes for F_LM's rule extrapolation, from one CPU pass over 3,476 keys (`runs/A1_factorized/`):
  | legality source for doubles | flip-type (216) | ID (873) | odd limits (479) | BFS-256 range_val | long_val |
  |---|---:|---:|---:|---:|---:|
  | F_LM | 0.324 | 0.994 | 0.931 | 0.720 | 0.965 |
  | frozen-base verifier, all doubles | 0.917 | 0.912 | 0.810 | 0.737 | 0.275 |
  | calibrated P(legal) > 0.10 | 0.741 | 0.981 | 0.881 | 0.803 | 0.965 |
  | **uncertainty-gated verifier** (verifier only if 0.01 < P(legal) < 0.5) | **0.949** | **0.994** | 0.933 | **0.863** | 0.965 |
  | code rule |c| <= L (ceiling, not learned) | 0.954 | 0.995 | 0.983 | 0.840 | 0.965 |
  Exact dynamics: range_val BFS-256 0.880. The gated verifier passes the flip gate (205/216, Wilson LB ~0.91) with no ID
  regression, recovers ~90% of the range planning gap, and calls the verifier on ~18% of keys; limitonly_val BFS-256
  0.98 -> 0.967 (the shortcut had pruned useless moves). It fails the odd-limit gate (0.933 < 0.97) because that
  weakness is pre-existing and *confident*: at L=21 (below every training limit) F_LM predicts doubling |x| in 11..14
  legal with P(legal) 0.81–0.997 — the mirror image of the upper shortcut. **Mechanism:** the fine-tune encoded the
  training limits' magnitude pattern; above the training range the shortcut is soft (uncertain, flips at P ~0.2–0.45)
  and confidence gating repairs it; below the range it is confident and only a rule check repairs it. The blanket
  verifier is unusable (it errs on ordinary doubles, e.g. "64 <= 64"; long_val collapses to 0.275).
- **D-33 A2 labelling switched to the gated rule (before any label was emitted).** Self-labels for F_vlab use F_LM's
  own prediction everywhere except doubles with 0.01 < P(legal) < 0.5, which take the frozen-backbone verifier.
- **D-34 Pilot moved to the GPU lane (all four arms on one device).** CPU state-and-action decoding advanced one
  planning round per ~20 min for 50 tasks (projected ~24 h for the remaining arms). The pilot now runs as two GPU
  jobs on the same 200 ID-validation tasks: A00+A01 (`pilotG-actions`) and A10+A11 (`pilotG-states`), duty-capped,
  shared-prefix sampler, output `runs/EVAL_pilot_gpu_s1`. The earlier CPU A00 (0.620) remains the CPU reference for the
  CPU step-400 comparison only; no paired contrast mixes devices. The aborted CPU A10 slice is recorded under
  `runs/_failed/cpu_pilot_A10_slice_aborted`.
- **D-35 CPU lanes never touch the CUDA driver** (peer finding I-21: with CUDA hidden, CPU forwards could die from heap
  corruption on torch 2.6.0+cu124; our CPU lanes still showed 165 MiB CUDA contexts because transformers/accelerate call
  torch.cuda.is_available()). `bwmn/lm.py` patches torch.cuda.is_available/device_count to False/0 at import when
  BWM_DEVICE=cpu (applies to newly started jobs).
- **B-02 Suffix-row leakage closed (found by the adversarial review of the Mystery generator; same flaw in our
  Blocksworld proposer data).** Proposer rows for intermediate states (state_t, goal) can be renamings of PlanBench
  test problems even when the instance's (init, goal) is not. `blocks_data.py` now drops every suffix row whose
  signature is a test signature: 577 rows (1.4%) removed; forward-model data verified byte-identical (sha256 prefixes
  e3a8a93c / 8915b370), so `bw-train-F` is unaffected; the leaky v1 data is kept under `runs/_superseded/` and was never
  trained on. Proposer training (`bw-train-Pa`) uses the clean file.
- **Gates before running external comparators / Mystery (from the same review; tracked in TODO_BEFORE_RUNS.md):**
  comparator plan extraction must not be one-sided (port PlanBench's own extractor as a second scorer; drop sub-bullet
  commentary; table/inline-list formats), an exclusive run lock + re-checked RAM gate, attention FLOPs included, resume
  config pinned; Mystery generator must apply the suffix-signature guard, substring vocabulary filter, single-letter
  objects for part of training, containment-based Blocksworld-fragment exclusion, a token budget, and signature-
  clustered CIs (603 Mystery instances = 262 distinct problems).
- **A2 self-labels ready.** Gated labelling over 2,322 keys (|x| <= 64 at limits 72–120): 0 dropped; agreement with
  the exact law 2,305/2,322 = 0.993 (analysis only; the exact law was never used to make or select labels).
  `vlab_sft.jsonl` = 1,455 exact in-distribution rows + 2,322 self-labels = 3,777 rows; GPU job `a2-train-Fvlab` queued.
- **P-01 (pilot finding, CPU, paired on the same 200 ID-validation tasks, seed 1).** Action-only proposer A00:
  step 400 = 0.440 vs step 800 = 0.620 (+0.18; 49 vs 13 discordant; exact McNemar p = 4.8e-6); first-action optimality
  0.61 -> 0.705; coverage@4 at round 1 0.205 -> 0.385. The proposer is far from converged at 800 updates (~0.14 epoch).
  **D-36:** queue a 1,600-update action-only run (seed 1, checkpoint at 1,200) to locate the operating point before
  committing GPU time to more seeds; GPU order now bw-F -> pilot (actions, states) -> F_vlab -> bw-Pa -> P_actions_1600.
- **D-37 CPU training measured and rejected** (~56 s/update for 16 x ~170-token rows on 8 threads vs ~7.6 s
  duty-capped GPU). CPU lane 1 instead runs **A01 on CPU** (4 slices) to pair with the CPU A00 (same device, same 200
  tasks): an early read of the F-rerank effect. The four-arm GPU pilot remains the clean comparison.
- **P-02 (early, n = 50, not conclusive).** A01 (F rerank) 0.66 vs A00 0.62 on the first 50 paired tasks (2 vs 0
  discordant). In 39/50 A01 episodes some round had no candidate predicted to reach the goal, so the rerank fell back to
  the first candidate (= A00 behaviour): with an accurate F, the bottleneck is proposer coverage at K = 4.
  **D-38:** add A01 at K = 12 on the first 100 tasks (CPU) to measure success vs candidates/compute.

## 2026-10-07

- **R-01 Restart.** The 23:12:42 reboot was a clean Start-menu restart (event 1074), not a crash; lanes were idle
  until 09:06, then relaunched; the GPU lease reclaimed the peer's stale lock under its own rules.
- **P-03 (pilot finding, CPU, seed 1, 200 ID-validation tasks, 8 held-out goal clusters).** F rerank within the
  action-only family: A01 0.675 vs A00 0.620 (+0.055; 11 wins / 0 losses; exact McNemar p = 0.001; goal-cluster
  bootstrap 95% CI [+0.020, +0.094]). Open-loop success 0.245 -> 0.385; first-action optimality 0.705 -> 0.750; cost
  ~25 cached F calls per task, same generated tokens. 76.5% of A01 episodes had a round with no candidate predicted to
  reach the goal (fallback = A00 behaviour), so proposer coverage limits the F effect. Still below model-free search
  under the same F (greedy-2 0.797 at ~51 F calls; BFS-64 0.780). Validation-grade, single seed; the four-arm GPU pilot
  and the K = 12 run are pending.
- **I-11 bw-train-F attempt 1 died (09:27:38) with a driver-level `CUDA error: out of memory`** at step ~55 while its
  peak VRAM was 2.77 GiB (cap 5.6 GiB): host RAM had dropped to 0.66 GB free (our CPU-lane grade_forward 3.9 GB +
  run_eval 3.65 GB + the peer's CPU smoke test 3.2 GB). **D-39 host-RAM floor:** `lm.load_core` first waits
  (`lm.wait_host_ram`) for free physical RAM and commit >= 3 GB (GPU loads, max 30 min, lease held) or >= 12 GB
  (CPU-lane loads, set by the orchestrator, so >= 6 GB stays free for either session's GPU job; agreed with RCWM-T1).
- **I-12 Hard reset at ~09:31:30** (Kernel-Power 41, BugcheckCode 0, no dump, no sleep/standby, no power button, lid
  open, no WHEA errors). Load at the time: our two 8-thread CPU lanes (A01k12 slice, E2 grading) + the peer's GPU job
  starting (GPU ~3.6 W at 09:31:23 per the shared watchdog). Same signature as three resets on 10/5 (09:16, 17:54,
  23:11); cause not determined (power or thermal trip, or memory pressure; the ACPI zone tz01 read 86 C at idle 2.5
  min after boot, but it also reads ~95 C under light load, consistent with the Ryzen boost target). Lost: ~25 min of
  each CPU job and the first ~21 min of bw-train-F. **D-40 mitigations:** (a) one CPU lane at 6 threads (cpu2 queue
  folded into `jobs_cpu.json`; `runs/.stop_cpu2`); (b) `scripts/health_log.py` (detached) logs tz01, CPU %, RAM,
  commit and GPU temp/power every 30 s with fsync to `runs/health.log` and `runs/health_now.json`;
  (c) `lm.cpu_thermal_guard` pauses CPU-lane jobs between batches at tz01 >= 98 C until <= 90 C and honours the
  shared peer flag; (d) **resumable long jobs:** `lm.train_sft(ckpt_dir=...)` checkpoints trainable tensors, AdamW,
  RNG states and loss history every 50 steps (atomic), used by train_generic/train_proposer/train_forward; verified
  on CPU that an interrupted-and-resumed 4-step run equals the uninterrupted run bit for bit (max |diff| 0.0, same
  loss and token counts). `grade_forward` appends greedy outputs per 480-key chunk to an fsynced cache and resumes
  from it (torn last line dropped). None of these changes alters any model's math, data, seeds or decoding.
- **R-02 Restart and early bw-train-F.** The GPU orchestrator was restarted at 10:00 to load bundle support (D-41);
  the peer's job ended early, so bw-train-F ran 11:58-13:29 (600 steps, 5,440 s, final loss 0.0008, peak 2.78 GiB).
- **D-41 Bundled GPU turns (agreed with RCWM-T1).** `scripts/bundle.py` runs several queued GPU jobs inside one
  lease turn: <= 4.5 h of wall time from the first acquire (lock record and turn.json carry `bundle_start`); a
  sub-job starts only if its estimate fits; a sub-job still running at the cap is terminated (trainings resume from
  checkpoints); a crash ends the bundle; exit 75 = budget used, requeued by the orchestrator without counting a
  failure. Unit-tested with dummy jobs (fit/skip, cap termination, crash). RCWM stays at one job per turn.
- **D-42 CPU guard honours the peer flag only for machine-wide reasons** (RAM, tz01, battery): a 15-min pause of a
  CPU test on a GPU-heat flag was observed.
- **F-04 (finding, offline, no model calls; `scripts/f03_ladder.py`, `runs/A1_factorized/ladder.json`).** Deferral
  ladder for doubles (1,954 keys) at the band's verifier-call rate (31%): F_LM alone 0.886; verifier always 0.760;
  random deferral 0.847; band (0.01, 0.5) 0.979; deployable symmetric rule |P - 0.5| < 0.478 (threshold set on the
  calibration limits to the band's call rate there) 0.979 at 31.6% calls; oracle 0.992. Strata: flips band 0.949
  vs symmetric 0.917 (at 68% vs 91% calls); below-range odd limits band 0.945 vs symmetric 0.977; ID 0.997 vs 1.000.
  The verifier is right on all 26 of F_LM's confident errors (outside the band), so the remaining loss is a gating
  problem, not a verifier problem. Planning (validation): BFS-256 range 0.720 (F_LM) -> 0.863 (band) / 0.847
  (symmetric); greedy-2 range 0.517 -> 0.650 / 0.640; limit-only 0.980 -> 0.967 / 0.957. **Reading:** the gain comes
  from confidence-based deferral to the frozen base as such (random deferral at the same rate hurts); the band's
  asymmetry trades below-range repairs for cheaper flip repairs. Novelty must be stated as empirical (selective
  prediction / learning-to-defer is established; the literature sweep found no instance of gating a LoRA world
  model's own backbone, adapter off, as its verifier).
- **F-05 (finding, CPU, `runs/BW_F_s0/grade_size.json`): the Blocksworld world model size-generalizes.** BW_F (LoRA
  on frozen Titan, 600 steps on 12k transitions from 4-6-block problems, ~50% illegal) graded by exact match against
  the simulator: in-distribution validation 0.995 (600 rows; legality 0.997); fresh random states by block count:
  3-8 blocks 1.000 each, 10 blocks 1.000, 12 blocks 0.967, 15 blocks 0.967 (legality 1.000 / 0.983; 60 keys per
  size). Errors at 12-15 blocks are copy errors in long fact lists (a dropped or added on-pair), not rule errors.
  PlanBench Blocksworld-hard has 6-15 blocks (110 instances, names a..o as in training), so a learned-world-model
  planner is viable there; per-step 0.967 compounds over 20-40-step plans, so a delta-format F (predict only the
  changed facts) is the natural next repair. Shared-prefix greedy decoder verified identical to the plain decoder
  on 48/48 keys.
- **D-43 Proposer data v2 for the Blocksworld-hard track (before any model saw a hard instance).** Hard goals are
  whole target configurations (5-12 on-pairs; 6-15 blocks; optimal plans 20-40), while v1 proposer goals had 1-4
  pairs. `runs/BW_data_v2` (proposer rows only; BW_F keeps its v1 data): same 3-6-block generator, but half the
  instances take every on-pair of a tall-tower target (p_stack 0.85) as the goal; Blocksworld-hard signatures
  (<= 7 blocks) are excluded in addition to the PlanBench-600 ones (instances and suffix rows; 634 suffix rows
  dropped). 53,682 train rows (goal pairs 1-5), all <= 234 tokens. Regenerating v1 with the new code is byte-identical.
  bundleA now trains `BW_Pa_v2_s1` (1,000 steps, seed 1) instead of the v1 Pa; the v1 Pa is dropped.
- **D-44 Planner for long problems: `blocks_planner.plan_beam`** (proposer-guided beam search in imagination; F
  only, no execution feedback; children = state at each candidate's point of best goal progress; no revisits;
  width W, expansion cap). Harness test with the exact simulator and a scripted myopic proposer (2-6-action
  fragments, 20% perturbed): 110/110 Blocksworld-hard solved (mean 49 expansions, mean plan 35 steps). Also
  `plan_direct` = the no-F SFT arm (proposer candidate 0 submitted as is).
- **P-04 (pilot finding, CPU, seed 1, first 100 ID-validation tasks, paired): proposer coverage was the bottleneck.**
  F rerank with K = 12 candidates (A01k12) 0.82 vs K = 4 (A01) 0.67: 15 wins / 0 losses, exact McNemar p = 6e-5;
  vs the proposer alone (A00) 0.63: 19 / 0, p = 4e-6. Round-1 coverage (some candidate predicted to reach the goal)
  0.34 -> 0.60 and open-loop success 0.34 -> 0.60; generated tokens per task 296 -> 870 (2.9x); F calls are cached
  table lookups. For reference on all 400 ID-validation tasks, model-free search under the same F: greedy-2 0.797,
  BFS-64 0.780, BFS-256 1.000. Validation-grade, single seed; the success-vs-compute comparison against search at
  matched LM calls is the follow-up (research plan item 6).
- **P-05 (compute accounting, same 100 tasks; `runs/EVAL_pilot_s1/compute_table_first100.json`).** Tokens processed
  per task by the shared 0.6B backbone (generated + prefill once per round + ~130 per unique F query):
  A00 0.63 @ ~1.0k; A01 K=4 0.67 @ 4.1k; A01 K=12 0.82 @ 6.3k; greedy-1 under F 0.60 @ 2.8k; greedy-2 0.77 @ 6.7k;
  BFS-16 0.62 @ 4.8k; BFS-64 0.81 @ 7.6k; BFS-256 1.00 @ 9.3k. **Reading:** in RW2 (five actions) the proposer + F
  reranker sits on the same success-compute frontier as model-free search under F (slightly above greedy-2/BFS-64
  at lower cost) but does not beat BFS-256, which reaches 1.00 at ~1.5x the tokens. No amortization claim for RW2.
  The amortization question moves to Blocksworld, where ground actions grow as 2n^2 and uniform search under F is
  far costlier.
- **I-13 Health logger died at 15:31** (no traceback captured; most likely `os.replace` onto `health_now.json`
  while a reader held it, a Windows sharing violation). Gap 15:31-15:54 in `runs/health.log`. Fixed: replace
  retried, every sample wrapped (errors to `runs/health.err`), relaunched 15:54; the monitor now alerts on a stale
  health file. CPU-lane jobs did not pause during the gap (stale reading -> no pause, by design).
- **F-06 (finding, F-03/F-04 follow-up; `scripts/f04_twosided.py`, `runs/A1_lowcal/`).** Added below-range
  calibration limits 24/28 (106 doubles; one CPU pass, 4 min; used by no task or grading table). F_LM alone is
  0.906 there (confident 'legal' errors). Re-selecting on calibration 80/112/24/28: the band family (hi <= 0.5) and
  the extended band (hi up to 0.999) both re-select band(0.01, 0.5) and do not change below-range accuracy; the
  symmetric gate (selected t) reaches all-doubles 0.980, odd limits 0.980 (band 0.945), ID 1.000, illegal 0.989,
  below-range calibration 0.943, but flips 0.921 (band 0.949) at 93% verifier calls on flips, and range planning
  BFS-256 0.840 (band 0.863). **Reading:** no confidence-only gate repairs both sides: the residual below-range
  errors are confident (P(legal) > ~0.98) while the verifier is worse than F_LM on most below-range keys (always-
  verify 0.811 on odd limits vs oracle 1.000). Repairing them needs a non-confidence signal (input-support or
  hidden-state novelty; research plan item 7). The band stays the F-03 operating point.
- **Diagnostic (not a selection input): Pa_v2 at step 300/1000, direct planner (no F, one greedy plan), CPU.**
  val_v2 (24 instances): 0.54 success, 100% parseable (4 blocks 1.00, 5 blocks 0.47, 6 blocks 0.00); val_large
  (12 instances, 6-15 blocks): 0.00 success, 92% parseable, plans locally sensible (unstack/put-down then build).
  As the literature predicts, a plain SFT plan generator does not reach larger problems; whether F + beam search
  does is the B2 question. Training continues to step 1,000.
- **D-45 Symmetric bundling for RCWM-T1 (agreed 20:4x).** The peer may run a T2 train->eval pair for one seed under
  a single lease (`rcwm-t1:t2pair:<tag>`) on the same conditions as D-41 (<= 4.5 h from first acquire, fit-or-wait,
  stop at the cap between resumable stages, release on crash, all guards); everything else of theirs stays one job
  per turn. Strict one-for-one alternation unchanged.
- **D-46 bundleB reordered for an earlier signal:** Pa_v2 remainder -> selection with the state-format F
  (direct / beam K4 / beam K8 on val_v2-100 + val_large-60) -> delta-F training -> selection of the delta configs ->
  RW2 pilot evals. The beam-search CPU probe was stopped (CPU decoding too slow to finish; no result recorded).
- **R-03 bundleA ended as designed.** BW_Pa_v2_s1 trained 1,000 steps (18:08-22:15, 14,849 s; final loss 0.046;
  peak 2.90 GiB); delta-F training did not fit the remaining 1,351 s, so the bundle ended the turn (exit 75), the
  lease was released with `bundle_start` in turn.json, and the peer's patching job acquired at once. bundleB
  (state-F selection first, D-46) is next in the GPU queue.
- **D-47 Selection throughput fix (23:48).** The first selection turn (bundleB) produced the no-F results only:
  direct Pa_v2 (K = 1) solves val_v2 0.99 (50/100... n = 100; 4-6 blocks) but val_large 0.117 (n = 60; 6-7 blocks
  1.00, 8: 0.29, 9: 0.13, 10: 0.33, 11-15: 0.00). The state-format beam selection could not finish within a turn
  (full-state outputs of up to 114 tokens, 4-row GPU batches, 25% duty: ~1.5 h per beam round), so it was stopped
  and the card released 4 h early (bundleB marked done; a2-train-Fvlab's queued wait was cancelled so bundleC runs
  next). Changes, none of which alters a model's output: `LMForwardBW` gets a persistent JSONL output cache
  (verified with a stub: second pass 0 new queries, identical results) and a 5,000-token GPU decode budget; delta F
  outputs are capped at 40 tokens (training max 25); `plan_beam(max_rollout=R)` simulates only the first R actions
  of each candidate (harness test still 110/110). Selection grid now: direct (done) and beam K4/K8, W4, E24, R12,
  with the delta F only, on val_large-30 (a subset of the 60) and val_v2-50; F format chosen on transition accuracy
  (state F_s0 vs delta, CPU grading on generated keys). bundleC = delta-F training -> delta selection -> RW2 pilot.
- **P-06 (selection-grade validation, GPU, generated instances only): world model + search rescues larger
  problems.** BW_Pa_v2 + delta BW_F, beam K4 W4 E24 R12 (`runs/BW_SELECT_v1/select.jsonl`): val_large (30 hard-style
  6-15-block instances) 0.50, every F-verified plan valid (verification precision 1.00); val_v2 (50, 4-6 blocks) 0.98.
  No-F direct arm on the same generator: val_large 0.117 (n = 60, a superset of the 30), val_v2 0.99. The K8 config
  and the paired breakdown on the identical 30 are pending (bundleC was capped mid-K8).
- **I-14 RAM emergency (06:05).** (a) A third-party helper, RevoUninHelper.exe (elevated), holds ~380k handles and
  grows: apparently one per process ever started, so the thousands of short nvidia-smi/PowerShell processes both
  sessions spawn stay as kernel zombies; available commit fell from 39 GB (10-07 09:38) to 5 GB. (b) My own fault:
  `wait_host_ram` loaded anyway after 3 h below the floor, so the delta-F CPU grading loaded at 06:05:35 and
  available RAM fell to 0.3 GB; killed at 06:06 but stuck in termination (4.39 GB resident), as is the peer's GPU
  job (8.15 GB), which still holds the GPU lock. Stop-Process on the helper: access denied (not escalated).
  **Fixes:** RAM floor never loads anyway (CPU waits; GPU exits 75 = give the turn back after 30 min); GPU sensors
  via NVML ctypes (`bwmn/nvml.py`), no subprocess per sample; both BWM queues halted (`runs/.stop`, `.stop_cpu`)
  by agreement with RCWM until the user ends the helper or reboots. Detached launches (WMI -> bash) currently fail
  under the memory starvation, so the health logger is down.
- **R-04 Recovery (2026-10-08).** The machine was restarted at 08:22:49 and the user ended RevoUninHelper.exe
  (it had restarted at boot). RAM 23.6 GB available, 43 GB commit free. Relaunched 08:25-08:26: health logger
  (NVML; no subprocess per sample), CPU lane (delta-F grading first), GPU orchestrator with a one-shot yield
  (`runs/.yield.json`) so the peer, whose turn was lost to the stall, goes first. Event monitoring now runs as one
  long-lived python process (`scripts/watch_events.py`) instead of shell loops spawning tail/grep/wc every 20 s.
- **D-48 Final-run throughput and fairness (before any test run).** (a) S-direct must be able to emit a whole hard
  plan (20-40 actions, ~4.4 tokens each): the proposer's 120-token default (~27 actions) is raised to 256 for the
  direct arm. (b) The beam planner simulates only R = 12 actions per candidate, so its proposer now generates at most
  80 tokens and keeps the complete leading actions of a truncated output (`blocks.parse_plan_prefix`). This changes
  which candidates exist, so it is re-selected on the validation sets, not assumed equivalent. (c) Proposer sampling
  budget on the GPU 2,000 -> 6,000 tokens (the 250-token hard-style prompts had forced one prompt per batch); outputs
  do not depend on batch composition (per-candidate CRN). (d) `scripts/bw_freeze.py` applies the pre-registered rule
  (max mean success over val_large + val_v2, <= 2,000 F queries/instance, ties -> fewer proposer tokens) and writes
  the freeze records with adapter and PREREG hashes; `scripts/bw_run_tests.py` then runs the frozen S-beam and
  S-direct once each on hard-110, mirror-110 and PlanBench-600 (VAL-checked; a run interrupted by the turn cap
  restarts from scratch with identical seeds and a warm F cache). blocks_eval refuses test runs whose settings differ
  from the freeze record (verified). bundleC = selection -> freeze -> tests -> RW2 pilot.
- **F-07 (finding, CPU, generated keys; F-format selection): the delta world model removes the large-state copy
  errors.** Same 12k transitions, delta output (facts added/deleted) vs full state, exact match by block count
  (60 keys each): 12 blocks 0.983 vs 0.967, 15 blocks 1.000 vs 0.967; 3-10 blocks 1.000 both except 5 blocks
  0.983 (delta); in-distribution validation 0.990 vs 0.995. Remaining delta errors are legality errors (calls an
  illegal pick-up/unstack legal); several are caught at planning time because a predicted deletion of a fact the
  state lacks makes the answer unsupported (`parse_forward_delta` returns None). Delta format kept for B2 (selected on
  generated-key accuracy at hard-set sizes, before any test run).
- **D-49 Packaging (user decision, 2026-10-08 ~09:05).** After the frozen Blocksworld-hard result, build a LOCAL
  devfreeze package `Rineya-Titan-Sparse-BWM-Blocksworld-0.1.0-devfreeze` in the Published-Cores format (README model
  card, CORE.json, evidence/: PREREG, freeze records, test summaries): unmerged LoRA adapters (delta BW_F, BW_Pa_v2)
  on frozen FP32 Titan-Sparse + planner/eval code. Pushing to Hugging Face is a separate step requiring the user's
  explicit approval; recommended gates first: frozen hard/mirror/600 results with a significant gain over S-direct,
  2-3 adapter seeds, model-card review, private repo before public.
- **R-05 Frozen B2 runs begin (2026-10-08).** Selection finished 13:55 (12,303 s for the remaining grid);
  `bw_freeze.py` selected beam K8 W4 E24 R12 M80 P (mean success 0.807 over val_large 0.633 / val_v2 0.98; eligible
  alternatives K4-M80 0.790, K4 0.740). Freeze records written 13:55:02. **S-direct on Blocksworld-hard (frozen,
  run once): 5/110 = 4.5%** (VAL agrees on all 110; 8 blocks 0.29, 9 blocks 0.21, all other sizes 0).
- **D-50 Sharded test runs.** The frozen S-beam needs ~8 GPU-h on the 110 hard instances (validation: 7,700 s per 30),
  more than a 4.5 h turn, and an interrupted run restarts from scratch. The unsharded hard-beam run (started 15:43,
  no round completed, no results) was stopped at 16:03 and its directory removed; S-beam test runs now use 8 shards
  (instances with index % 8 == i; outputs independent of shard composition by construction: per-instance CRN seeds,
  batch-independent sampling, frozen memoised F), each run once, merged by `bw_run_tests.merge` (unit-tested).
  bundleD = sharded tests -> RW2 pilot. The GPU orchestrator had died between 14:56 and 15:06 (`^C` in its stderr,
  probably a console-close event when this assistant session restarted); relaunched 16:03.
- **D-51 Records repository (user-approved 2026-10-08 ~19:00).** BWM-001 records, code and results are exported to
  the PRIVATE GitHub repo LUNCY579/rineya-world-model-research under `bwm/` (created by the RCWM session at the
  user's request) by `tools/export_bwm.py` from a separate clone (`Rineya_BWM_001_Research_Bundle/gh_export`):
  docs, code, run records; no adapter weights, training data or caches (listed with size + SHA-256 in
  `bwm/EXCLUDED_LARGE_FILES.json`); host health logs not exported; secret/email scan must pass before writing.
  First push: commit ea466a3 (254 files, 14 MB). Re-export after each milestone; pull --rebase before push.
- **P-07 (INTERIM, frozen B2 configuration, PlanBench Blocksworld-hard shards 0-2 = 42 of 110 instances; no
  decision depends on it, the remaining shards run unchanged).** S-beam 25/42 = 0.595 (Clopper-Pearson 95% CI
  0.433-0.744); S-direct on the same 42 instances 1/42; paired: beam-only 24, direct-only 0, exact McNemar
  p = 1.2e-7. VAL agrees with the executor on every plan. By block count: 6: 1/1, 7: 1/1, 8: 3/4, 9: 5/5,
  10: 4/6, 11: 1/2, 12: 1/3, 13: 5/8, 14: 4/10, 15: 0/2. Shard 3 was cut at the turn cap (restarts next turn);
  ~1.5 h per shard, so shards 3-7 need about two more BWM turns.
- **I-15 GPU used outside the lease (my fault), 23:08-00:06 on 10-08.** At bundleD's turn cap, `bundle.py` called
  `terminate()` on its direct child (`bw_run_tests.py`) only; the grandchild `blocks_eval.py` (hard-beam shard 3,
  ~3 GB VRAM) kept running until it finished at 00:06, overlapping the peer's L1 screen (whose analysis passed its
  parity gate; peer informed, offered a re-run turn). Shard 3's output is valid (frozen configuration, ran to
  completion): 10/14. **Fix:** `bundle.py` kills the whole process tree at the cap (`taskkill /F /T`); a one-off
  guard kills leftover eval processes when the current turn (old code loaded) ends at 06:33.
  *I-15 follow-up (peer, 02:2x):* no overlap occurred. The peer's L1 job was waiting in its lease, whose
  foreign-dedicated-VRAM check held it until the stray process exited (acquired 00:06:04); all L1 passes ran alone.
  Cost: ~1 h delay to the peer. Our lease already has the same foreign-VRAM check (D-25).
- **D-52 Shared trace-target schema (joint roadmap item 5b, agreed with RCWM 2026-10-09).** One JSON object in fixed
  key order `{"cmp": .., "calc": .., "after": ..}` (explicit order, never sorted; parsers reject other orders). RW2:
  moves `cmp = {"cand": c, "slack": L-|c|, "ok": slack>=0}`, halve `{"x", "rem", "ok"}`, finish `null`;
  `calc` = c if ok else x; `after` = existing `{"legal","terminated","x"}`. Blocksworld: `cmp` = each precondition fact
  with its truth + ok; `calc` = delta; `after` = existing delta answer. Arms: plain / trace / filler (token-matched).
  Primary outcome: out-of-range strata. Replaces the A3 slack-scratchpad plan.
- **P-08 (FINAL, frozen B2 configuration, run once; PlanBench Blocksworld-hard, 110 instances, 6-15 blocks,
  20-40-step optimal plans, planning without execution feedback, VAL-scored).** **S-beam 73/110 = 0.664**
  (Clopper-Pearson 95% CI 0.567-0.751); **S-direct 5/110 = 0.045** (0.015-0.103); paired beam-only 68 /
  direct-only 0, exact McNemar p = 6.8e-21. VAL agrees with the executor on all 220 plans. Every plan S-beam
  verified in imagination was valid (73/73; all 37 failures were unverified, i.e. the system knew it had not found a
  plan). By block count (beam/direct/n): 6: 1/0/1, 7: 1/0/1, 8: 6/2/7, 9: 14/3/14, 10: 10/0/13, 11: 9/0/12,
  12: 8/0/13, 13: 11/0/18, 14: 11/0/21, 15: 2/0/10. Cost per instance: ~8.0k proposer tokens + ~187 unique F
  queries. Published zero-shot references (different regime, aggregate only; unpaired Fisher tests):
  vs o1-preview 26/110 p = 2.4e-10; vs o1-mini 10% p = 1.2e-18; vs DeepSeek-R1 53.6% (~59/110) two-sided p = 0.073
  (one-sided 0.037), so "above R1" is NOT established at the two-sided 5% level. Training saw <= 6 blocks.
  Mirror (generated, same block counts): S-direct 6/110 = 0.055; S-beam pending (8 shards).
- **D-53 plan_beam domain hooks (joint roadmap item 3).** `plan_beam(..., goal_met=None, unmet=None)`; defaults
  reproduce the Blocksworld behaviour exactly (harness regression after the change: 110/110 with the identical
  79,745 exact-F calls), so the frozen B2 configuration's remaining runs (mirror S-beam, PlanBench-600) are
  unaffected. Interface sent to RCWM for the ledger planning test.
- **F-08 (finding, CPU; `scripts/bw_surface_audit.py`, `runs/BW_Fdelta_s0/surface_audit.json`): the Blocksworld
  world model is invariant to names and list order but NOT to JSON key order.** Same 120 (state, action) keys
  (5 and 10 blocks, ~50% illegal), exact match: training format 1.000; unseen block names (p-z, two-letter)
  0.992; shuffled fact lists 1.000; reversed key order (user and state objects) 0.458; all changes 0.458. The
  model learned the transition rule over symbols, but tied to one serialization: the format, not the names, is
  the trigger. Same direction as RCWM PREREG-3 (T2, reordered keys 0.37-0.47). The frozen B2 results are
  unaffected, because the planner always renders states in the training format; claims are scoped accordingly
  ("generalizes in size and object names; format-locked"). Repair candidate: key-order augmentation in training
  data (joint item 5b surface diversification).
- **D-54 Serialization augmentation design (joint with RCWM, after F-08 / PREREG-3).** Augment INPUT only: randomize
  key order and list order of the prompt's state/action JSON; targets keep their literal order (trace targets
  cmp -> calc -> after, never sorted). Primary test on HELD-OUT input orders: train on a subset of key permutations
  and evaluate on permutations never seen in training (e.g. fully reversed), so augmentation is tested as rule
  learning rather than a wider trigger set.
- **D-55 Filler amendment 4b (RCWM finding, adopted).** A per-row length-matched filler leaks the answer (RCWM:
  fill length alone predicts wrap 0.957-0.970 and branch up to 0.85), giving a filler-trained model a hidden
  channel. The filler length is now a CONSTANT per (domain, action kind), equal to the longest trace head of that
  kind in train+dev (`traces.filler_length`); verified on Blocksworld that filler prefixes are identical within a kind
  regardless of legality. The trace-vs-filler residual is reported as a covariate. RCWM-side deviations recorded:
  R uses strict '>', R calc = {raw, mod}, cmp null for R add/mix, parser rejects duplicate keys.
- **D-56 Search distillation (joint roadmap item 4), pipeline built.** `scripts/bw_distill.py`: (search, GPU)
  240 fresh generated hard-style problems (6-10 blocks; seed 20261010, disjoint from val_large/mirror; hard-set
  signatures/invariants and val_large/mirror invariants excluded) solved by the frozen B2 planner at K=4; ONLY
  F-verified plans are kept (teacher = the model's own world model; the exact simulator scores label precision for
  analysis only). (rows, CPU) every suffix of each verified plan, with F-predicted states, becomes a proposer row.
  Next: continue Pa_v2 on new + v2 rows, then the frozen S-direct comparison on generated hard-style sets and a
  pre-registered once-only run on the hard set (new addendum). Smoke test with exact F + scripted proposer: 16/24
  verified, 304 rows, label precision 1.0. Queued as bundleE after bundleD's remaining frozen tests.
- **D-57 Joint item 3 (ledger beam) prereg drafted; planner code frozen.** Draft v1 at
  `research/PREREG_LEDGER_BEAM_DRAFT.md` (co-primaries T2 vs OFF paired, all instances and the needs-lookahead
  stratum, enum W2E12; key secondary the T2/ORACLE ratio; N = 180 fresh; training-surface claim). RCWM runs and scores.
  `bwmn/blocks_planner.py` is frozen at SHA-256 18a5f22b... until that run is complete (no edits on the BWM side).
- **P-09 (INTERIM, frozen B2, mirror shards 0-4 = 70 of 110 generated hard-style instances; no decision depends on
  it).** S-beam 38/70 = 0.543 (CI 0.419-0.663); S-direct on the same 4/70; paired +34 / -0, p = 1.2e-10. Hard set,
  same shard indices: 44/70 = 0.629. Interim hard-minus-mirror gap +8.6 pp (pre-registered flag at > 10 pp; final
  pending). Mirror S-direct (all 110): 6/110. The mirror goals (tall-tower targets, p_stack 0.85) are not
  distribution-matched to the hard set's goal structure, so a gap in either direction is partly difficulty, not
  only contamination.
- **D-58 Addendum B3 (search distillation) pre-registered before any distillation data existed**
  (`PREREG_B3_DISTILL.md`, sha e6aa884b…; `PREREG_PLANBENCH.md` left byte-identical to its frozen sha adfc4354…).
  - **Arms:** Pa_v3 continues `BW_Pa_v2_s1` on distillation rows + an equal number of v2 rows. Pa_v2c is the control:
    the same extra training on v2 rows only. Both use 400 steps, batch 16, lr 1e-4, seed 1, max-len 640, and rows
    over 640 tokens are dropped explicitly and counted.
  - **Primary:** S-direct (greedy, M 256) on a fresh generated hard-style set of 110
    (`instances_b3_fresh.json`, seed 20261011, hard block-count mix, excluding hard/val_large/mirror/distill
    invariants, sha 575c45b1…), Pa_v3 vs Pa_v2c, paired McNemar. The reading is "met" only if the difference is
    >= +10 pp and p < 0.05.
  - **Conditional hard runs:** once each, only if the reading is met (H1 S-direct, H2 S-beam with only Pa swapped).
  - **Automation:** the whole chain is queued as bundleE: search -> rows -> mix -> train Pa_v3 -> train Pa_v2c ->
    `FREEZE_B3.json` (SHAs) -> eval -> analyze. Nothing in it chooses a setting.
  - **Code:** `train_generic.py --init-adapter`, `bw_b3_prep.py`, `bw_b3_eval.py`.
  - **Queue:** the old pilotG GPU replications moved from bundleD to bundleF at the end of the queue (lower value now).
- **D-59 PREREG-LB (joint roadmap item 3) co-signed by BWM: revision 2,
  `RCWM-T1/docs/ledger_beam/PREREG_LB_DRAFT.md`, sha256 26c62144d89dde15e7d18ad119f9361ce6f93562bd3d358f26808e7aefbd788d.**
  - **Salt commitment:** `commit:0b6b5ea92bd193a3bf75866a645be1abd50f99a991e01d8b852701c3e13de8cf`. BWM holds the salt
    out of band and reveals it only to the RCWM operator at the freeze.
  - **What I checked:** my v1 draft (b2896ba3…) is quoted verbatim; the planner pin is 18a5f22b…, and the runner
    verifies it at import and every turn and imports the module from my tree read-only; the kit SHAs match.
  - **Statistics:** exact McNemar is computed on the discordant pairs, with Holm m = 2. The status table
    (PASS/SHORT/NS/REVERSED) and the 12 branches are exhaustive over 32 combinations, and every s1 = PASS branch keeps
    my co-primary-1 meaning ("enables planning", qualified where co-primary 2 fails or reverses).
  - **Seed:** the commit-reveal seed rule leaves no party able to compute the draw in advance.
  - **Amendments accepted:** (a)-(f) and the operational definitions (A0, F as the first computed answer across
    stages, void-twin exposure, the three-way audit reading).
  - **Appendix B:** defaults accepted, Q1-Q10:
    - all stages kept, primary first;
    - beam vs one-shot left out;
    - only blocks_planner.py binding;
    - co-primary-2 bar +0.20;
    - batch invariance >= 99%;
    - audit margin 0.05;
    - N = 180;
    - salt procedure as written;
    - data/ subdirectories in the union;
    - automatic analysis after a verdict-eligible stop.
  - **BWM commitments:**
    - `bwmn/blocks_planner.py` unchanged until the registered analysis has run;
    - no import-breaking edits to `bwmn/__init__.py`, `blocks.py` or `lm.py` (the planner module imports them in
      RCWM's GPU process), with notice to RCWM before any edit to them while LB is queued or running.
  - **D-59 follow-up:** at RCWM's freeze request, BWM revealed the salt by session message. Before sending, I
    checked locally that it is 39 characters and that its sha256 equals the committed 0b6b5ea9…. The salt is now
    recorded in RCWM's draw SOURCE record.
  - **PREREG-LB frozen by RCWM, 2026-10-09 23:12:28.**
    - PREREG_LB.md 0f414cbc…, PREREG_LB_FREEZE.json 1b7e7a54…; `--verify` rc 0.
    - Draw: 180 instances, sealed 4fff0738…, not inspected. Exclusion union: 315,081 payloads from 37 sources.
    - Queued as RCWM's next GPU job (`rcwm-t1:lbeam`), which follows bundleD's release.
    - BWM keeps `blocks_planner.py`, `__init__.py`, `blocks.py` and `lm.py` stable until LB's analysis has run.
- **P-10 (FINAL, frozen B2 configuration, every test set run once; settings chosen on generated sets only).
  Exact simulator and local VAL agree on every plan (0 disagreements).**

  | Set | S-beam (F + search) | S-direct (same proposer, no F) | Paired, exact McNemar |
  |---|---|---|---|
  | PlanBench Blocksworld-hard (110; 6-15 blocks) | 73/110 = 0.664 (CI 0.567-0.751) | 5/110 = 0.045 | +68 / -0, p = 6.8e-21 |
  | Generated hard-style mirror (110) | 61/110 = 0.555 (CI 0.457-0.649) | 6/110 = 0.055 | +55 / -0, p = 5.6e-17 |
  | PlanBench Blocksworld (602; 3-5 blocks) | 602/602 = 1.000 (CI 0.994-1.000) | 584/602 = 0.970 | +18 / -0, p = 7.6e-6 |

  - **Cost per instance (S-beam):** hard 187 F queries and 8.0k proposer tokens; PlanBench-602 3.0 F queries and
    279 proposer tokens.
  - **Pre-registered contamination flag: RAISED.** Hard minus mirror is +10.9 pp, above the registered 10 pp
    (unpaired Fisher p = 0.13, not significant).
  - **Post-hoc, descriptive** (`scripts/bw_mirror_gap.py`, `runs/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json`; the flag
    stands as computed):
    - The move lower bound is matched (hard 21.7 vs mirror 21.0).
    - The goal structure differs. The mirror's tallest goal tower averages 8.1 vs 6.9 on hard, and 29/110 mirror
      instances have a goal tower of >= 10 vs 13/110 on hard.
    - Success falls with goal-tower height in both sets, and the bins are similar: <= 6: hard 51/61, mirror 30/33;
      7-9: 20/36 and 25/48; >= 10: 2/13 and 6/29.
    - Reweighting the mirror to the hard set's bin mix gives 0.699 vs hard 0.664 (gap -3.5 pp).
    - Reading: the raw gap is explained by the mirror's taller goal towers, not by an advantage specific to the
      PlanBench instances. It is still reported as flagged.
  - **Allowed claim (B2):** trained only on <= 6-block generated problems, the frozen 0.6B core with two LoRA
    adapters, planning without feedback, solves 73/110 Blocksworld-hard; the world model + search accounts for
    +68/-0 over the same proposer alone. Published zero-shot references are R1 53.6% and o1-preview 23.6%, in a
    different regime.
- **D-60 Vault copy reverted (2026-10-10 ~03:20).** At 03:05 I had copied the rebuilt package into
  `Rineya_Published_Cores/` as a new folder (no CATALOG/REGISTRY edits).
  - **Why reverted:** D-49 authorizes a LOCAL package only, not placement in the vault, and the user never
    explicitly approved the vault step. I had only announced it in status messages.
  - **What was removed:** the vault copy, after `diff -r` confirmed it was byte-identical to
    `native_v1/release/…`, which remains the package of record.
  - **Next:** the vault placement waits for the user's explicit go-ahead. RCWM raised the question.
- **D-61 PREREG-LB report co-signed by BWM after an independent re-score** (`scripts/lb_rescore.py`,
  `runs/LB_RESCORE/LB_RESCORE.json`). Inputs were read only: the draw (sealed 4fff0738…) and the episode files.
  - **Re-score method:** BWM's own ledger simulator, goal test, unmet hook and greedy needs-lookahead procedure; exact
    McNemar via scipy; Holm; the branch table; and the paired bootstrap. No RCWM analysis code was imported.
  - **Agreement with `PREREG_LB_RESULTS.json` (887b4c88…):** every registered quantity agrees:
    - success: T2 127/180, OFF 14/180, ORACLE 160/180, IGNORE_STATUS 63/180, T2 W4E24 134/180;
    - needs-lookahead set: 52 instances, identical per instance;
    - CP1: +113/−0, Holm p 3.9e-34, PASS;
    - CP2: 23/52 vs 3/52, +20/−0, Holm p 1.9e-6, PASS;
    - T2/ORACLE 0.794 [0.730, 0.855], NEAR false;
    - branch YES.
  - **Rendered report:** headline, reading, qualifiers and limits are the registered strings.
  - **Notes, report-only and changing no reading:**
    - The W4E24 ratio of 0.802 must not be cited as near-oracle, because the bar applies to the primary
      configuration only.
    - select_fixed (T2 0.867) chooses among stored candidates that include the optimal plan, so it is not comparable
      with beam success.
    - T2's verified-plan precision is 0.952: about 5% of T2's "verified" plans fail under the true dynamics.
    - The realized needs-lookahead share is 29% vs about 36% expected from dev. It was reported, not chosen.
    - The D-35 audit (5/180 off-surface) makes the planning claim training-surface only, as registered.
  - **Joint reading:** a near-oracle one-step ledger model planned inside BWM's domain-agnostic beam planner,
    unchanged except for the hooks. It reached 79% of the exact simulator's success, with every discordant
    instance in T2's favour, but only on its training surface.
- **D-62 D-54 key-order augmentation pre-registered and queued** (`PREREG_D54_KEYORDER.md`, written before any
  augmented adapter existed; `scripts/bw_keyorder.py`).
  - **Arm FK** = `BW_Fdelta_ko_s0`: the same 12,000 transitions and targets as F0 (`BW_Fdelta_s0`), the same recipe
    (600 steps, seed 0). Prompts are re-rendered: 50% canonical; otherwise one of 21 training state-key orders × 2
    top-level orders, with lists shuffled in half of those rows. Data: 5,901 canonical + 6,099 augmented rows,
    42 distinct orders, max 175 tokens.
  - **Held out:** the state orders (table,on,hand,clear), which is F-08's reversal, (hand,table,clear,on) and
    (on,clear,table,hand).
  - **Evaluation:** 240 fresh keys (seed 20261013) in 5 modes, F0 vs FK paired.
  - **Reading:** "generalises to unseen key orders" iff FK >= 0.95 on both held-out modes and canonical is
    non-inferior within 0.01. "Trigger widened" iff the training orders are >= 0.95 but a held-out mode is < 0.90.
    Otherwise "partial".
  - **Queue:** bundleG (train about 1.7 GPU-h, then eval), placed after bundleE (B3) and before bundleF.
- **I-16 Reboot recovery (2026-10-10; boot 09:24:04).** The user rebooted during bundleE, which had held the card
  05:19-09:23, i.e. 4h04m of its 4.5 h.
  - **State at the reboot:** B3 search finished chunks 0-6, 168/240 problems, 148 F-verified. `solutions.jsonl`
    (168 rows) and `fcache.jsonl` (12,086 rows) had no torn lines. Chunk 7 was lost (about 26 GPU-min; its F calls
    are cached).
  - **Recovery:**
    - BWM's own stale lock (pid 63900, worker 247728, both dead; no compute on the card) was archived as
      `rineya-gpu-0.lock.stale-63900.bak` and removed.
    - turn.json was written by hand (`last: bwm`, released at boot), because the reboot killed the bundle before
      its release write.
    - A one-shot `runs/.yield.json` (label_prefix `rcwm-t1:`, max 6 h, give-up 90 min) was set, and the health
      logger and orchestrator were restarted at 09:27:06.
    - RCWM's 5a took the card at 09:27:42, as agreed.
  - **Fix for a gap the audit found (QP-2):** honour_yield writes no waiting marker, so at about 10:27 BWM
    hand-wrote `waiting.bwm.json` for the yielding gpu_run (pid 16628, `bwm:bundleE`). A cap-deferral re-call by
    RCWM therefore sees a live BWM waiter. A code patch is pending, for future launches only.
- **D-63 Joint plan after the reboot (RCWM D-45 / JOINT_PLAN_2026-10-10), adopted with BWM amendments; post-reboot
  audit actions.** The audit was 15 read-only agents with adversarial verification: 0 blockers, 0 high. Every P-10
  number reproduces.
  - **Turn order (strict alternation):**
    1. RCWM 5a;
    2. BWM bundleE, turn A: search remainder → rows → mix → Pa_v3;
    3. RCWM 5b;
    4. BWM bundleE, turn B: Pa_v2c → FREEZE_B3 → eval → analyze → D-54 FK train → key-order eval;
    5. RCWM T2b-ledger / screen winners;
    6. BWM: if B3 reads "met", the registered conditional H1/H2 runs first (prepared as freeze copies and a bundle
       on CPU beforehand); otherwise item 7 seeds; item 9 transfer is CPU scoping only.
  - **Commitments:**
    - BWM mirrors PREREG-4 with FK inside the frozen B2 planner on a held-out key order.
    - `bwmn/blocks_planner.py` stays pinned at 18a5f22b…, since B3 H2, RCX E08, T2b-ledger and PREREG-4 depend on
      it. RCWM and RCX get notice before any change.
  - **Queue changes:**
    - bundleE est_s raised to measured upper bounds (search 5600, Pa_v3 7600, Pa_v2c 6600, FK 6600, key-order eval
      1500), so no registered arm is split by a cap.
    - bundleE.json sha 327133cf… → a9c0c384…, with D-54's two jobs appended.
    - bundleG was removed from the queue (merged).
    - 8 legacy single GPU jobs (a2-train-Fvlab, train-P-actions-s1-1600, six e2-*) and bundleF (pilotG) were moved
      verbatim to `runs/jobs_gpu_deferred.json`, each with a reason. This is a deferral, not a cancellation. Each
      single job would cost a full alternation turn, and none is in the joint plan.
    - jobs.json sha 11089997… → d4a0abe0…. The queue is now bundleE only. The orchestrator exits when it is empty,
      so it must be relaunched when the next bundle is added.
  - **D-54 Amendment 1 (D-62a)**, written before any FK adapter or output existed, with the shared RCWM wording:
    - reading = absolute bar + canonical non-inferiority + significantly above F0 on each held-out mode; "trigger
      widened" = training orders ≥ 0.95 while staying near F0 on held-out;
    - added instrumentation: per-item `items.jsonl`, CP CIs, adapter/rows/script/prereg hashes, an F0 ≠ FK assert;
    - files: `PREREG_D54_KEYORDER.md` 6b704388… → d2e0d86c… (original quoted inside); `bw_keyorder.py` 48902cde… →
      94561044…;
    - a stubbed CPU end-to-end test (fake model, scratch root) produced 2,400 items, provenance and the expected
      reading.
  - **Infrastructure drafts, applied only at safe boundaries:**
    - persist orchestrator attempt counts and halt instead of skipping on `.failed` (QP-3);
    - honour_yield writes the waiting marker (QP-2);
    - blocks_eval counts F-cache hits (FR-02);
    - build_devfreeze formats unrounded means (FR-01).
- **D-64 RCX scope agreement (CX Q-02, 2026-10-09 ~22:40), recorded here as authoritative.** This was previously
  only in a private note.
  - RCX is CPU-only, single lane, ≤ 4 threads. It pauses on tz01 ≥ 97 °C or RCWM's flag, and applies a 4 GB
    free-RAM floor to any job over 2 GB and 12 GB to 0.6B loads.
  - B3 is BWM's. RCX reads B3 artifacts only after `runs/BW_B3_EVAL/B3_REPORT.json` exists.
  - E07/E08 use fresh seeds 202612xx with BWM's exclusions: `test_signatures`, `hard_signatures`, and invariants of
    hard, val_large, mirror, b3fresh and distill-240.
  - No evaluation on hard, PB-600, mirror or b3fresh without a joint prereg; val_large and val_v2 are BWM's selection
    sets.
  - D-54 is BWM's; RCX adds a canonicalization control and avoids seeds 20261012/13.
  - Item 6 is joint (BWM + RCWM); RCX contributes a CPU critic baseline built from `traces.bw_parts`.
  - GPU stays a BWM/RCWM two-party alternation.
  - On 2026-10-10 at 09:44 RCX ran two heavy CPU jobs at once. BWM asked it to keep one lane and fix three record
    wordings.
- **E-01 Errata from the post-reboot audit (no registered number or reading changes):**
  - **(a) FR-01:** the hard mean tallest goal tower is 6.85 (377/55). P-10's 6.9 is right. The release README's
    6.8 comes from double rounding in build_devfreeze.py:118 and is fixed at the next approved rebuild.
  - **(b) FR-02:** the mirror S-beam F cost is 215.9 unique F queries per instance. The summary and CORE.json give
    183.6, which counts only the final attempt's cache misses, because shards 0, 2 and 5 restarted with a warm cache.
    Hard (186.9) and PlanBench (3.0) are exact.
  - **(c) FR-03:** registered B2 secondary items missing from P-10, now reported or recorded as not run:
    - hard success by optimal plan length (S-beam/S-direct/n; audit recomputation from info.json):

      | Optimal length | S-beam | S-direct | n |
      |---|---:|---:|---:|
      | 20-24 | 27 | 5 | 30 |
      | 25-29 | 18 | 0 | 20 |
      | 30-34 | 17 | 0 | 30 |
      | 35-40 | 11 | 0 | 30 |

    - paired against published per-instance results on the zero-shot 600: o1-preview +13/−0 (p = 2.4e-4); DeepSeek-R1
      +5/−0 (p = 0.0625, **not claimable**);
    - not run: S-replan secondary arm, fresh-600 control, FLOPs/F-token accounting. They were not run in the frozen
      sequence; FLOPs and F-tokens can be estimated post hoc from fcache.
  - **(d) PX-07:** the LB exclusion union (my D-59 follow-up quoted "315,081 payloads from 37 sources") is 322,899
    payloads / 314,850 operative / 58 sources per the sealed `LB_CONFIRM_DRAW.SOURCE.json`. Validity is unaffected
    (overlap 0/0). RCWM was asked for the matching erratum.
  - **(e) RC-10:** the D-60 vault revert happened at about 03:07 (vault mtime 03:07:25), not about 03:20.
  - **(f) RC-09:** D-61's "every registered quantity agrees" means the 10 machine-compared keys in
    `LB_RESCORE.json` (counts, needs-lookahead set, CP1/CP2, Holm, statuses, ratio and CI, NEAR, branch) plus hand
    checks (ORACLE 160, IGNORE 63, W4E24 134/167, audit 5/180, rendered strings).
  - **(g) RC-11:** in D-62, 76 of the 6,099 "augmented" rows equal canonical text (6,023 effectively
    non-canonical). The D-54 prereg file was saved at 05:49:46, before the data at 05:50.
  - **(h) The D-56 smoke-test claim** (16/24 verified, 304 rows) has no saved artifact.
- **I-17 Second reboot (user restart from the Start menu at 14:08:59, event 1074; boot 14:09:22).**
  - **BWM's turn:** bundleE had held the card from 12:59:19, after RCWM 5a's cap deferral, and the yield worked as
    intended. It released cleanly at 14:08:59 and recorded `turn.last=bwm`, after 70 of its 270 min.
  - **B3 progress:** the search reached 216/240 problems, 188 F-verified. The exact simulator agrees on all 188
    (analysis only). Files are intact; one chunk of 24 is left.
  - **Recovery:** RCWM 5a (B-bits arm) took the card at 14:22:36. BWM relaunched the health logger and orchestrator
    with a one-shot yield to `rcwm-t1:`, and hand-wrote `waiting.bwm.json` for pid 18212. BWM did not ask for a
    make-up turn.
  - **QP-2 patch:** `scripts/gpu_run.py` `honour_yield` now writes the waiting marker itself. This affects future
    launches only; the running launcher keeps its loaded code.
  - **5c:** adapter dirs and shas were sent to RCWM, which will run the frozen retention suite on BW_Fdelta_s0 and
    BW_Pa_v2_s1 in an RCWM turn (label `rcwm-t1:retention`).
  - **D-64 follow-up:** RCX logged the 09:44 concurrent-lane breach as CX I-02. It killed a second overlapping smoke
    run at about 10:14 and adopted a stricter rule: sub-agents may only edit code or run checks under 5 s while an
    RCX lane job runs. It also fixed the three record wordings (commit 2b55b72) and recorded BWM's pins as CX D-13.
    RCX finding R-03 (commit 980c360) is a third independent instance of the TRIGGER pattern (after BWM F-01 and
    RCWM PREREG-3): on its fresh-word ledger battery, T2's no-op fires on void (1.00) and its near-misses (0.97) but
    never on unseen non-posted words (0.00); rule-polarity sensitivity is 0.00.
- **D-65 User decisions (2026-10-10 ~14:40, explicit: "go with your recommendations on all five decisions").**
  1. **Vault: not now.** The package stays in `release/` only. Re-ask the user once the item 7 seed gate and the 5c
     retention check are done and the F-08 "format-locked" caveat is in the model card. A Hugging Face push still
     needs its own explicit approval.
  2. **Deferred GPU jobs** (`runs/jobs_gpu_deferred.json`: 8 legacy single jobs and pilotG) stay deferred.
  3. **B3 conditional hard runs first:** if B3's primary reading is "met", the registered H1 (S-direct with Pa_v3 on
     hard) and H2 (frozen B2 beam with only Pa swapped to Pa_v3, sharded) run before item 7.
  4. **Item 7 (seed replication, the release gate):** two fully independent replicate systems, R1 = (F seed 1, Pa
     seed 2) and R2 = (F seed 2, Pa seed 3), plus the anchor R0 = (BW_Fdelta_s0, BW_Pa_v2_s1). All three run with the
     frozen B2 configurations on a fresh generated hard-style set of n = 40. That is about 22 GPU-h. Pre-registered
     before any seed trains (`PREREG_ITEM7_SEEDS.md`).
  5. **Item 9 (transfer):** PlanBench Logistics, the strongest generality claim (a different domain, same PlanBench
     harness and VAL). Scoping and prereg on CPU first; no GPU until the prereg is frozen.
- **D-66 Decisions 3 and 4 implemented; queue after B3.**
  - **B3 conditional hard runs** (`scripts/bw_b3_hard.py`, sha 321ecd3e…):
    - A `b3-gate` sub-job, added to bundleE after b3-analyze, reads the registered B3 reading. On "met" it writes
      `FREEZE_B3_H1/H2.json` (the frozen B2 direct and beam args with only `pa` → `runs/BW_Pa_v3_s1/adapter`, plus
      the Pa_v3, FREEZE_B3, prereg and report SHAs) and `runs/BW_B3_EVAL/H_GO.json`. Otherwise it writes
      `H_NO.json`.
    - bundleH (H1 hard S-direct; H2 hard S-beam in 8 shards; paired analysis vs B2) is queued with
      `after: H_GO.json`, so it runs before item 7 only if B3 is met.
    - A stub test of both branches passed; the H2 args differ from B2 beam only in `pa`.
  - **Item 7 (`PREREG_ITEM7_SEEDS.md`, sha 956a0dc4…; `scripts/bw_seeds.py`, b783a191…):**
    - Fresh set `instances_seeds_fresh.json`: n = 40, seed 20261014, sha 6fccb945…. Block mix 6:1, 8:3, 9:5,
      10:5, 11:4, 12:5, 13:6, 14:8, 15:3; 637 invariants excluded. Generated before any seed training.
    - Added to `blocks_eval.py` as test set `seeds_fresh` (TEST_SETS guard; additive change; blocks_eval sha
      0f2175fb… → b3666424…).
    - bundleI: train F seeds 1 and 2, then Pa seeds 2 and 3 (exact s0/s1 recipes; est 6600/15500 s), then
      `FREEZE_I7_*`, then R0/R1/R2 × (direct, beam in 4 shards), then analyze. Expected about 6 BWM turns.
  - **Queue order:** bundleE → bundleH (only if met) → bundleI.
- **X-01 Cross-programme note: RCWM 5a screen (R20, dev bank, pre-stated rule rev 4): winner B-bits.**
  - **B-bits** (bitstring state plus an explicit applicable flag): IAW 0.83, 4-step exact 0.84, false copy 0.07.
  - **Decimal:** IAW 0.10, false copy 0.73, i.e. the copy shortcut.
  - **Joint cmp/calc/after trace:** IAW 0.59. It helped the predicate strata but induced copying (0.29), so it was
    excluded.
  - **Held-out surface:** B-bits keeps its no-ops but its apply accuracy halves (4-step 0.84 → 0.42).
  - **Relevance to BWM:**
    - Before any RW2 trace (5b mirror / A3) is pre-registered, add a false-copy metric and a bitstring-interface arm.
    - It supports "expose the predicate in the interface" over "teach the conversion": the "copy with the core,
      compute with a module" reading.
- **X-02 Cross-programme note: RCX R-04** (commit 0890ed1; tier B, an external specialist; exact dynamics; CPU).
  - **The result:** a 124k-parameter relational GNN value function trained on exact h* for 4-7 blocks transfers
    out of distribution.
    - One-step policy accuracy at 8/10/12 blocks: 0.998, vs h_add 0.848 and unmet_goal_facts 0.062.
    - GBFS on 100 fresh 8-15-block full-goal problems at low expansion budgets: GNN 49/57 at 50/200 expansions, vs
      unmet_goal_facts 2/8.
    - Above about 500 expansions misplaced_blocks and h_add overtake it.
  - **Reading for BWM:** `plan_beam`'s ranking key (unmet goal facts) is a weak critic even with exact dynamics. Part
    of B2's hard failures (budget and dead ends at 12-15 blocks) may be ranking, not F or Pa. Not claimed until tested.
  - **Candidate B4, after B3 / D-54 / item 7, pre-registered first:**
    - (a) RCX's GNN as the `unmet=` hook in the frozen B2 beam (read-only, co-signed, fresh instances, RCX's cache).
      This is an external-module result.
    - (b) A model-native critic: a value LoRA V on the same frozen Titan-Sparse, trained on h* (or on F-verified
      search distances for a simulator-free variant), as the `unmet=` hook.
    - (c) A paired comparison against the frozen B2 ranking, with Pa and F unchanged.
- **D-67 Item 9 (PlanBench Logistics) scoped on CPU; draft prereg saved; nothing frozen; no GPU.**
  - **Scoping:** a read-only workflow (3 scouts, 2 designers, 1 synthesis).
  - **Draft:** `research/PREREG_LOGISTICS_DRAFT.md` (sha 56d36602…). Scoping notes and both designs are in
    `research/ITEM9_LOGISTICS_SCOPING.json`.
  - **Vendored assets:** the only local copy was in a Temp scratchpad clone of LLMs-Planning (fc638a1a), so it is
    now copied into `bench/planbench/logistics/` (domain, generated_basic 286, generated 286, t1 prompts with
    Fast-Downward astar(lmcut) ground truth, generator source) and `bench/planbench/results/logistics/` (published
    results), recorded in SOURCE.md. These files are TEST ONLY.
  - **Key verified facts:**
    - Local VAL accepts 285/285 ground-truth plans.
    - PlanBench Logistics t1: 4-7 packages in 212 instances; optimal plan lengths 12-48.
    - Published per-instance results: o1-preview zero-shot PDDL 188/200 (strong); GPT-4 one-shot 28/200.
    - PDDL action tokens: about 13 per action; the longest ground-truth target is 619 tokens.
  - **Draft design:**
    - train F and Pa on generated 1-3-package problems;
    - select K/W/E on generated LG-VAL sets with a futility gate;
    - test once on PB-OUT-80 (20 per package count 4-7) plus a generated mirror;
    - paired S-beam vs S-direct;
    - new modules only (`bwmn/logistics.py`, `bwmn/domain_planner.py` wrapping the pinned planner).
  - **BWM defaults on its open questions** (user to confirm at freeze time):
    - PB-OUT-80;
    - Pa 1000 steps (recipe identity with B2);
    - PDDL operator names;
    - VAL semantics (self-moves are legal no-ops);
    - K fixed at 4;
    - the drafted futility gate;
    - grid rationale from generated instances only, with any PlanBench-derived design numbers disclosed as history;
    - seed replication as a later separate prereg;
    - Stage C conditional runs registered but each needing explicit user GPU approval.
  - **Cost and order:** about 6 BWM GPU turns (16-38 GPU-h), queued after bundleI (item 7).
  - **Next:** CPU implementation (CPU-2..CPU-5), resolve the [VERIFY] items, then a freeze with the user's approval.
    Seeds 20261101-05 are to be confirmed unused by RCWM and RCX.
  - **D-67 follow-up:** the Logistics seed block moves to **20261401-20261405**.
    - Why: RCX flagged that RCWM's PREREG-3 out-of-range bank used 20261103, and RCWM uses the whole 202611xx range.
    - Ranges now: BWM 2026100x-2026101x and 202614xx; RCWM 202611xx; RCX 202612xx.
    - The draft was updated in place (still unfrozen).
  - **Seed-registry note:** RCWM listed every seed it uses: 20260917/18/22; 20261006-09; 20261020; 20261028;
    20261103-04; 20261109-16; 20261121-23; 20261125-26; 20261141-43; 20261161-64; 20261199.
    - 202614xx is clear.
    - BWM's existing 20261006-09 (BW_data v1/v2, val_large, mirror) coincide numerically with RCWM's 20261006-09.
      The generators and domains differ, so no data overlap; recorded for unambiguity.
    - From now on new BWM seeds come only from 202614xx-202615xx.
- **P-11 (FINAL, registered, run once): B3 search distillation, primary reading MET**
  (`runs/BW_B3_EVAL/B3_REPORT.json`; FREEZE_B3 adapters v3 48c9d279…, v2c f7ac53bf…, v2 ee1c5aec…).
  - **Primary** (S-direct, greedy, M 256, on the fresh 110 hard-style problems `b3fresh`):

    | Arm | Solved | 95% CI |
    |---|---|---|
    | Pa_v3 | 22/110 = 0.200 | 0.130-0.287 |
    | Pa_v2c (same extra training, no search plans) | 8/110 = 0.073 | — |

    Paired: +18 / −4, exact McNemar p = 0.0043; +12.7 pp, which clears the +10 pp bar.
  - **Secondary, report-only:**
    - Pa_v3 vs the original Pa_v2: 22 vs 5 on b3fresh (+17/−0, p = 1.5e-5); 22 vs 6 on the mirror (+18/−2,
      p = 4.0e-4).
    - Pa_v3 vs Pa_v2c on the mirror: 22 vs 7 (+16/−1, p = 2.7e-4).
    - The extra-training control alone does nothing: v2c vs v2 is +6/−3 (p = 0.51) and +3/−2 (p = 1.0).
    - In range vs out of range (6-10 vs 11-15 blocks), b3fresh: v3 15/36 and 7/74; v2c 7/36 and 1/74.
    - The v2/mirror cell reproduces B2's frozen S-direct 6/110 exactly.
  - **Distillation:** 211/240 search plans F-verified; label precision 1.000 (all 211 valid under the exact
    simulator, analysis only), so the "trained without simulator labels" wording is allowed. 4,254 rows, none over
    640 tokens.
  - **Truncation** (report-only): 25/110 (b3fresh) and 21/110 (mirror) of Pa_v3's plans reached the 256-token cap,
    and none of them succeeded. Solved plans are longer for v3 (mean 21.5 vs 14.8 actions). The frozen M = 256
    understates Pa_v3. No re-run with a larger M without a new addendum.
  - **Independent check:** I re-scored all six cells with the exact simulator from the plan files. Every count and
    both primary McNemar tests reproduce.
  - **Allowed claim:** "Trained on plans its own world model verified (no simulator labels), the proposer alone solves
    22/110 fresh hard-style problems vs 8/110 for the same extra training without them (paired p = 0.004)."
  - **Consequence:** the registered H1/H2 hard runs are unlocked (H_GO.json; `FREEZE_B3_H1/H2` written by the gate).
    bundleH's two jobs were merged into bundleE after D-54, so they use the rest of BWM's next turn. The separate
    bundleH queue entry was removed. Queue: bundleE (FK → key-order eval → H1/H2) → bundleI (item 7).
- **X-03 Cross-programme note: RCX R-05** (commit 5e05a5b; DEV pilot, one seed, not a claim).
  - **Setup:** a rule-varied LoRA (N-VR), D-03 recipe on frozen Titan-Sparse.
  - **Canonical rule wording:** it executes in-context ledger rules (composite 0.964 vs OFF 0.325 and T2 0.400;
    rule-polarity sensitivity 1.00; unseen statuses 1.00).
  - **Held-out paraphrases of the rule:** it drops to 0.594, falling back to the base model's apply-everything prior.
  - **Reading:** the same surface-anchoring family as BWM F-08 (key order) and RCWM PREREG-3/5a (held-out surface).
    Training-surface diversity (D-54 here; T2b-ledger and N-VR paraphrase cells on the peer side) is the shared next
    test.
- **D-68 GitHub export clone moved out of the bundle** (2026-10-11 ~00:35, at RCWM's request), from
  `Rineya_BWM_001_Research_Bundle/gh_export` to `C:/Users/clclc/BusinessProject/WorldModel/bwm-gh-export`.
  - **Why:** RCWM's frozen 5c freshness gate scans the whole BWM bundle as an external corpus. The clone mirrors
    RCWM's own retention dev draw (`rcwm/.../retention/draws/dev_20261121.jsonl`), which produced 202 false "outside"
    hits. The gate refused to mint the 5c draw for BWM's adapters; no draw was spent.
  - **State:** the clone was clean and pushed (1442515). `tools/export_bwm.py` locates the repo relative to itself,
    so nothing else changes.
  - **Side benefit:** RCWM's banks and draws no longer sit inside any freshness union that scans the BWM bundle.
- **X-04 Cross-programme note: RCWM 5b register trace screen (R23), registered NO-GO** (failed the mix and swap
  gates). Its two lessons are central for BWM:
  - **(1) Traces teach computed conditional predicates.** On out-of-range conditionals the joint cmp/calc/after
    trace scores 0.90, vs 0.39 for plain answers and 0.41 for the constant-length filler control. On held-out
    surfaces: 0.66 vs 0.24 vs 0.22. Trace − filler = +0.49, so the gain comes from supervision, not extra decode
    compute. cmp.ok is 0.99-1.00 on adversarial identification strata. This is the positive half of "computed
    predicates aren't learned from bare-answer rows" (BWM F-01, RW2 `|2x| <= L`). It makes the RW2 legality trace
    arm (A3, 5b mirror) a strong candidate. Pre-register it with RCWM's false-copy metric (X-01).
  - **(2) Narrow LoRA training overwrote an untouched operation:** swap went 1.00 → 0.00 in every arm, with no swap
    rows in training. Rule for all future BWM adapters: training data includes control-skill rows for every
    operation the adapter must keep.
    - B3 already did this (mix_v3 = distillation rows + equal v2 rows), and D-54 FK uses the same transitions.
    - For item 9 Logistics and any RW2 trace arm, add explicit control rows, and report retention of untouched
      operations.
- **D-69 Item 9 CPU-1/CPU-2 done: `bwmn/logistics.py` (eb683a6f…), `tests/test_logistics.py` (423ef5b5…),
  `scripts/lg_published.py` (5a39ede2…).** Built by an implement → adversarial review (VAL semantics; spec and
  planner interface) → fix workflow, then re-run independently: 33/33 tests pass (58 s, CPU, no torch).
  - **Semantics:**
    - 285/285 PlanBench ground-truth plans valid under the executor and local VAL;
    - self-moves are legal no-ops, as VAL treats them;
    - type-wrong actions are rejected;
    - VAL's plan-text reader is reproduced from probes (39 pinned cases, 6,000 random texts).
  - **A* teacher:** optimal on all 285 PlanBench instances (equal to Fast Downward's lmcut plans; at most 36k
    expansions at 4-7 packages). This retires the draft's "[VERIFY] teacher failures at 4-5 packages".
  - **Signature:** a renaming-invariant signature with 285 classes over 286 files (instances 47 and 48 are
    isomorphic). Exact against brute-force isomorphism on 161 small states.
  - **Tokens:** delta target ≤ 26; Pa prompt ≤ 348; generated 1-3-package Pa rows up to 690.
  - **Published re-score** (`runs/LG_published/PUBLISHED_RESCORE.json`): 1,450 plans in 8 files with 0
    disagreements among the executor, VAL and the published flags. o1-preview zero-shot PDDL 188/200; GPT-4 one-shot
    NL 28/200; Llama-3.1-405B zero-shot PDDL 19/200.
  - **BWM design decisions on the review's open items** (draft updated in place, still unfrozen):
    - (a) LG-SPEC-1: max-len 768 for both Logistics adapters instead of 640, with peak memory re-measured in the
      bundleL1 smoke test;
    - (b) LG-SPEC-2: self-moves fixed at exactly 5% of legal F rows (uniform drawing would make about 36%).
  - **Remaining for item 9 before the freeze:**
    - CPU-3 `domain_planner.py`;
    - CPU-4 harness check;
    - CPU-5 `lg_*` scripts;
    - the [VERIFY] items;
    - the user's confirmation of the defaults.
- **P-12 (registered, run once): D-54 key-order augmentation, reading PARTIAL** (`runs/BW_KEYORDER/eval.json`;
  FK adapter ed26cbf7…; prereg with Amendment 1 d2e0d86c…; 240 fresh keys per mode; F0 vs FK paired).

  | Mode | F0 | FK | Paired (FK-only / F0-only) |
  |---|---|---|---|
  | canon | 0.992 | 0.963 | 0 / 7, p = 0.016 |
  | train_order | 0.550 | 0.925 | |
  | heldout_rev | 0.454 | **0.917** (CI 0.874-0.948) | 130 / 19, p = 1.6e-21 |
  | heldout_other | 0.496 | **0.950** | 120 / 11, p = 2.6e-24 |
  | heldout_both | 0.454 | 0.858 | |

  - **Why "partial":** FK is far above F0 on every held-out order, but heldout_rev is below the 0.95 bar and
    canonical accuracy fell by more than the 0.01 margin.
  - **Independent check:** all 2,400 prompts rebuilt from the seed with identical hashes; held-out prompts verified to
    use H1-H3 only; every item re-scored from its text with 0 mismatches.
  - **Report-only mechanism:**
    - FK's held-out accuracy (0.917/0.950) equals its accuracy on trained orders (0.925), so the gain transfers to
      unseen orders. This is evidence against "trigger widened".
    - Nearly all FK errors are on ILLEGAL actions: canon 100/109 illegal vs 131/131 legal. FK predicts the action's
      effects with `legal: true` when a precondition fails, e.g. pick-up while holding.
    - So input augmentation bought order invariance at the price of the computed legality predicate. This is the
      same failure class as F-01 (RW2) and RCWM's 5a/5b findings.
  - **Next candidate** (to be pre-registered; not run): FK plus the joint cmp/calc/after trace of Blocksworld
    preconditions (`traces.bw_parts`), which RCWM's 5b showed teaches computed predicates (X-04). Add a false-copy /
    false-apply metric and control rows. FK is NOT used in any frozen planner.
- **P-13 (registered conditional run H1, run once): Pa_v3 S-direct on PlanBench Blocksworld-hard**
  (`runs/BW_B3_HARD/h1_direct/`; freeze `FREEZE_B3_H1.json` = the B2 direct args with only `pa` → Pa_v3).
  - **Result:** 13/110 = 0.118, vs the frozen B2 S-direct 5/110. Paired +10 / −2, exact McNemar p = 0.039. VAL agrees
    on all 110 (0 disagreements).
  - **By blocks:** 6:1.00, 7:0.00, 8:0.14, 9:0.50, 10:0.23, 11:0.08, 12-15:0.00.
  - **Truncation** (report-only): 24/110 plans reached the 256-token cap; none succeeded.
  - **Reading:** search distillation transfers to the real hard set without search. The size of the gain is limited
    out of range (≥ 12 blocks: 0) and by the frozen token cap.
  - **H2** (frozen B2 beam with Pa_v3, 8 shards) started at 01:55. It continues across BWM turns before item 7.
- **X-05 RCX E07b (pre-registered, sha 8a4cbde0…; R-07, commit d9c993c): CLOSED with a hybrid winner.**
  - **Setup:** GBFS with exact dynamics on 75 fresh full-goal problems of 12-15 blocks.
  - **Result:** the hybrid (GNN + 1.0 × misplaced_blocks) solves 39/75 at 200 expansions, vs misplaced alone 19,
    the GNN 28, h_add 13, and unmet-goal (BWM's beam key) 0. At 2,000 expansions: 55 vs 54 vs 35 vs h_add 66 vs 3.
    Plan suboptimality 1.045 vs h_add 1.97.
  - **B4 arms updated** (pre-register later, co-signed with RCX), all as `unmet=` hooks in the frozen B2 `plan_beam`:
    - (a) RCX hybrid, an external learned module;
    - (a0) misplaced_blocks alone, a hand-written zero-cost control that separates "learned value" from "any better
      heuristic";
    - (b) a BWM value LoRA on Titan-Sparse;
    - (c) the default unmet-goal count.
- **D-70 B4 (learned critic as `unmet=` hook) draft received from RCX** (commit 117e53a; copied to
  `research/PREREG_B4_DRAFT.md`, sha 777f5870…). It is a draft: BWM owns execution, the instances and the final
  text, and it is frozen only after H2 and item 7.
  - **Arms (as drafted):** c (default), c+pen, a0 (misplaced + penalty), a (RCX hybrid + penalty), b (BWM value
    LoRA). Primary Holm family of 2: a vs c+pen and b vs c+pen. 110 fresh hard-style instances at the frozen W4/E24.
  - **BWM answers to the draft's §8:**
    1. The frozen config is `runs/FREEZE_B2_beam.json` args: K 8, W 4, E 24, R 12, M 80, P 1, max_depth 80, delta F
       `BW_Fdelta_s0`, Pa `BW_Pa_v2_s1`, eval_seed 0.
    2. Arm (b) has two variants: b-h* (h* labels, simulator-supervised) in the primary family, and b-sf (distances
       along F-verified search plans, simulator-free) as a secondary.
    3. The Pa_v3 cross is decided after H2.
    4. Instance seed 20261410, n = 110, hard block mix, BWM + RCX exclusions.
    5. An exact-F oracle ceiling, report-only, if the budget allows.
    6. RCX exports the h* labelling recipe; BWM generates labels in its own seed range under the exclusions.
  - **D-70 follow-up:** RCX exported the h* labelling recipe (commit e85acc5:
    `complementary/experiments/RCX-E07/HSTAR_LABEL_RECIPE.md`; `complementary/rcx/bw/space.py` sha 29d0b420…,
    numpy and stdlib only).
    - Exact backward-BFS h* for n ≤ 8; A* h* beyond that, with over-budget states recorded as missing, never dropped.
    - Splits by goal class, never by state.
    - BWM will vendor a copy at a pinned sha, or re-implement it, when B4's arm (b) data is built. Labels are
      generated in BWM's seed range under the joint exclusions.
- **D-71 Co-sign of RCWM R25: 5c retention of BWM's release adapters** (run confirm_1708484032065101819).
  **Both RETAINED, GO (5/5 binding) under the registered RULE_CONFIRM.** This is a pass of the low-power retention
  guard, not a non-inferiority result.
  - **Pins:** RET_FREEZE.json b32ef148…; sealed draw 864481a3… (375 items = 5 categories × 75); provenance.json
    4b4a925d… (= DONE.json); RET_RESULTS.json fae050af…; RET_RESULTS.md 1af4c3ac…. The adapter shas match my files
    (BW_Fdelta_s0 2ed0bb00… / 3038c779…; BW_Pa_v2_s1 ee1c5aec… / f6a2c4ef…).
  - **Independent check** (`research/RETENTION_5C_COSIGN_CHECK.json`): my own graders, written from DESIGN §4, and
    my own implementation of the §7 rule, without importing ret_analysis.py.
    - All 1,077 compared fields reproduce exactly, and the KL/AUROC blocks match.
    - Freeze 14/14, draw and generation hashes verify; the draw regenerates byte-identically.
  - **Net vs OFF out of 375, primary / lenient grader.** The pass margin is −15 pooled and −11 per category, and a
    failure under either grader counts.
    - BW_Fdelta_s0: −3 / −5.
    - BW_Pa_v2_s1: +9 / −10. The graders disagree in sign; the lenient CI reaches −20.
  - **Report-only caveats** (all go into the card; text saved in `research/RETENTION_CARD_TEXT.md`):
    - **BW_Fdelta_s0 unit_conv −7:** 30 → 23/75, lost 8 / gained 1, Holm 0.195. These are real arithmetic errors.
      The pass is fragile: without its single gain the result is NO-GO.
    - **BW_Fdelta_s0 unknown_twin −6 (primary only):** mostly format drift. It answered UNKNOWN on 75/75.
    - **BW_Pa_v2_s1 abstention 0.933:** 5/75 unanswerable questions got invented values (OFF 1.000).
    - **Base model:** Titan-Sparse already loses unit conversion against Supa 1.7 (49 → 30, Holm 0.0008), and both
      adapters add a further, non-significant drop.
    - **Scope:** 5 of 8 categories; one level each; templated short answers; one seed; each adapter tested alone,
      under the suite's prompt.
  - **Release-gate status:** retention ✓ (with caveats); seeds (item 7) pending; F-08 format caveat to add. Vault or
    HF stays the user's decision.
  - **Erratum (E-02):** D-68's clone move happened at about 00:23, not about 00:35. Directory mtimes are 00:22:57,
    and the 00:23:52 re-mint scanned 492 vs 1,300 files.
  - **D-70 follow-up (B4 exclusions):** add RCX's E11 instance keys to the B4 fresh set's exclusions: E11a test
    (seed 20261256) and E11b execution (seed 20261258). RCX will send the key-list paths and shas when E11 finishes.
    The B4 fresh set is NOT generated before that list is in hand.
- **P-14 (FINAL, registered conditional run H2, run once): Pa_v3 in the frozen B2 S-beam on PlanBench Blocksworld-hard**
  (`runs/BW_B3_HARD/h2_beam/`, `runs/BW_B3_HARD/B3_HARD_REPORT.json`; freeze `FREEZE_B3_H2.json` = the `FREEZE_B2_beam`
  args with only `pa` → `runs/BW_Pa_v3_s1/adapter` (48c9d279…); 8 shards, every shard summary matches the freeze;
  finished 2026-10-11 13:07).
  - **Result:** **88/110 = 0.800** (Clopper-Pearson 95% CI 0.713-0.870) vs the frozen B2 S-beam **73/110 = 0.664**
    (0.567-0.751). Paired +20 / −5 (both 68, neither 17), exact McNemar p = 0.0041. VAL agrees on all 110
    (0 disagreements). Every plan the beam verified in imagination was valid (88/88). All 22 failures were unverified
    (budget 17, dead end 5).
  - **Compute** (report-only): about 8.2k proposer tokens and 198 unique F queries per instance, vs 8.0k and 187 for
    B2. The gain does not come from more search.
  - **Multiplicity** (report-only; not registered): Holm over the two hard contrasts gives H2 0.0082 and H1 0.039.
  - **By block count** (report-only; the stratum tests are post hoc):
    - 6-10 blocks: 34/36 vs 32/36 (+3/−1, p = 0.63);
    - 11-15 blocks: 54/74 vs 41/74 (+17/−4, p = 0.0072).
    - Most of H2's gain lies outside the 6-10-block distillation range. H1 is the opposite: 12 of its 13 successes are
      at 6-10 blocks.
  - **Leak audit** (two independent read-only audits): no leak.
    - **No hard problem in the training data:** 0 of the 240 distillation problems is a renaming of a hard
      (init, goal). The same holds for every row of mix_v3, mix_v2c, distill_rows and the 53,682 v2 rows. Method:
      exact canonical form, validated against `B.signature` on 3,100 problems.
    - **Exclusion only:** hard touched the data only through exclusion (10 near-miss candidates rejected, none of
      them an isomorph).
    - **Overlap at baseline:** goal-configuration overlap is at the generated-set level (hard 31/36 at ≤ 10 blocks;
      mirror 33/36; b3fresh 31/36). 17 of the 20 v3-only wins are on instances whose goal is not among the
      distillation goals.
    - **No hard reads:** no script reads hard reference plans or hard results.
    - **Settings and timing:** the H2 args are the B2 beam args, chosen on val_large/val_v2 before any hard run,
      with only `pa` changed. FREEZE_B3_H2 (10-10 20:29) predates H1 and H2.
    - **Shard restarts:** both were turn-cap aborts, replayed from cache.
  - **Published references** (report-only; different regime: zero-shot general models given PDDL text, with no domain
    training, no learned simulator and no search; unpaired, two-sided Fisher):
    - DeepSeek-R1 53.6% (59/110), p = 5.2e-5;
    - o1-preview 23.6% (26/110), p = 2.5e-17.
    - This does not license "beats o1/R1 at planning".
  - **Reading:** the frozen B2 world model, search settings and compute are unchanged; swapping in only the
    self-distilled proposer raises Blocksworld-hard success from 73 to 88/110. Scope:
    - the registered contrast is Pa_v3 vs the original Pa_v2, so it does not separate the search-plan content from the
      extra training. There is no Pa_v2c beam arm on hard. On S-direct, the extra-training control alone did nothing
      (P-11: +6/−3, p = 0.51);
    - one training seed, one run;
    - the B3 design (6-10-block hard-style tall-tower goals) was chosen after B2's hard results were known. No mix,
      step count, adapter or search setting was chosen on hard.
  - **Allowed claim:** "A frozen 0.6B Rineya with two LoRA adapters (~20M parameters), planning without feedback,
    solves 88/110 Blocksworld-hard instances (CI 0.713-0.870), vs 73/110 for the same frozen search with the original
    proposer (paired +20/−5, p = 0.004). It was trained on generated problems of at most 6 blocks and then (proposer
    only) on generated 6-10-block plans its own learned world model had verified (no simulator labels). Published
    zero-shot reference: R1 53.6%, o1-preview 23.6% (different regime)."
    - B2's wording "Trained only on <= 6-block problems" does NOT apply to Pa_v3 (H1 or H2).
    - Neither the student nor the system may be called "simulator-free". "No simulator labels" covers only the
      distillation increment (label precision 1.000, 211/211).
  - **Corrections recorded with this entry:**
    - (a) **o1-mini 10% is not zero-shot.** It is the LRM-Modulo (verifier-in-loop, 4 iterations) score; o1-mini
      direct is 0.9% (1/110) (arXiv 2410.02162 Table 5; upstream per-instance file at fc638a1a). P-08's "vs o1-mini
      10% p = 1.2e-18" is withdrawn as a comparison B2 forbids. PREREG_PLANBENCH.md:117 is hash-frozen, so it gets an
      erratum, not an edit. The release README:40 and SOLUTIONS_AND_BENCHMARK_PLAN.md:44 must drop the row or restate
      it as 0.9% direct.
    - (b) **P-13's reading is scoped.** H1's transfer is within the 6-10-block distillation range: 12 of 13
      successes, +9 of the +10 discordant pairs, and 1/74 vs 0/74 at 11-15 blocks (B3's registered out-of-range
      stratum, not "≥ 12").
      - The 256-token cap can only understate H1 (24 truncated plans are unscored). It is not shown to "limit" it.
      - H1's 13/110 is below o1-preview's zero-shot 26/110 (Fisher p = 0.033), so the proposer alone is not
        competitive with the zero-shot LRMs.
      - P-13's "H2 continues" is superseded by this entry.
    - (c) **P-11:** "The frozen M = 256 understates Pa_v3" becomes "can only understate Pa_v3".
    - (d) **"Aggregate only" is false** (B2's published-reference header; release README:42). Upstream per-instance
      hard results exist at the vendored commit fc638a1a. Paired comparisons against them are post hoc and are not
      cited until the files are vendored with sha256 and re-scored by the local VAL.
    - (e) **R1 citation:** cite R1's 53.6% to the PlanBench leaderboard / upstream per-instance file, not "TMLR 2025"
      (unverified).
    - (f) **gh-export README:59:** "Trained only on plans its own world model verified" → drop "only".
  - **Audit record:** `research/H2_LEAK_CLAIMS_AUDIT.json` (leak hunt, claims audit, judge; 3 agents).
  - **D-70 follow-up (B4 design, agreed with RCX):**
    - **Primary family unchanged:** P1 a vs c+pen and P2 b vs c+pen, both under Pa_v2. Under Pa_v3 (88/110) a critic
      contrast has at most 22 movable instances.
    - **Pa_v3 cross is secondary:** each hook arm under Pa_v3 on the same 110; a vs c+pen under Pa_v3 (unadjusted,
      labelled secondary); a descriptive 2×2 interaction (compound vs substitute); strata 6-10 vs 11-15.
    - **Minimal secondary set if GPU is short:** c+pen and a under Pa_v3.
    - **E11 exclusion lists:** E11a `complementary/runs/RCX-E11/main/a/instances.json` (seed 20261256, about
      14:20) and E11b `.../b/data/instances.json` (seed 20261258, about 16:30); key = `D.instance_key`. B4's fresh
      set is generated only after both shas are recorded.
- **X-06 Cross-programme note: RCX E01N sealed confirmation, REVISE** (R-08, commit f7ee0cf).
  - **Result:** surface-diversified rule training (N-CFSD) vs plain rule-varied training (N-VR), on held-out rule
    paraphrases: +0.090 [+0.069, +0.113], Holm p 0.0002, positive in every seed. On training wordings: +0.018. Only
    the pre-registered +0.10 point threshold failed.
  - **Cost:** N-CFSD over-restricts some held-out OR wordings (apply 84/87 → 66/87) and makes slightly more
    false-copy errors.
  - **Reading:** this is the third independent instance of the D-54 pattern (BWM P-12; RCWM 5a held-out). Surface
    diversity buys held-out robustness at a small cost to a computed decision. The shared remedy candidate is
    trace supervision of the predicate (X-04).
  - **B4 exclusion list 1/2 received and verified:** RCX E11a, 120 instances (n = 12/14/15 × 40, seed 20261256), sha
    6cbec9cf… (verified from commit b0ccde8). Vendored as `research/b4_exclusions/rcx_e11a_instances.json`.
    - Key: `instance_key` = raw canonical at n ≥ 8, renaming signature at n ≤ 7. The B4 draw will also exclude by
      renaming class (`goal_key` and VL.invariant).
    - E11b is still pending (about 16:30).
  - **B4 exclusion list 2/2 received and verified:** RCX E11b, 581 instances (exec 128 at n = 10-15, seed 20261258;
    train/val/test_id/ood_small 453 at n = 4-7, seed 20261257), sha 47c44d31… (verified from commit f447563).
    Vendored as `research/b4_exclusions/rcx_e11b_instances.json`.
    - RCX's B4 key lists are now complete: E11a 6cbec9cf… and E11b 47c44d31….
    - The B4 fresh set (seed 20261410) may be generated, but only together with the B4 freeze, after item 7. It
      excludes both lists by key and by renaming class.
- **X-07 Cross-programme note: RCX E11 (frozen prereg; R-09, commit 5e7a0ba), using BWM's B3 verified plans as
  teacher, read-only.** RCX re-verified 211/211 with `B.execute`, and the F trajectories match.
  - **E11a, ADVANCE** (tier B; exact dynamics, GBFS):
    - 8 macros mined from the 211 plans raise GBFS + A3-hybrid from 52 to 77 of 120 fresh 12-15-block full-goal
      problems at 200 expansions (+25/−0, p = 6e-8).
    - They beat matched random sound macros (69) and macros mined from action-shuffled plans (63).
    - With misplaced_blocks, any macro library helps about equally.
    - The library is mostly tower chains.
  - **E11b, REVISE:** the learned subgoal order is bounded by its label definition.
  - **BWM candidate B5, after B4, pre-registered first:** RCX's macro library M as an action-abstraction arm in
    `plan_beam`. A macro expands into k primitive steps, each verified by F, with E11's random/shuffled macros as
    matched controls. RCX offers to export M with its sha.
  - **X-07 follow-up (B5 inputs vendored and verified):** RCX macro library export (commit 0671794) in
    `research/b5_macros/`:
    - `libraries.json` b61b4a05… (M treatment, R/S matched controls, 8 macros each; G 2 descriptive; per-macro
      exhaustive validation at ≤ 6 blocks);
    - `MACRO_LIBRARY_EXPORT.md` 90ca1060…;
    - `macros.py` (as `.txt`, not importable from the bundle) e644f6b1….
    - **B5 design notes to adopt:** check that imagined states are consistent before binding macros (shared with the
      B4 penalty), use F-queries per success as the compute-matched unit, and pre-register both the R and S controls.
- **D-72 Item 9 CPU-3 done: `bwmn/domain_planner.py` (7f440d0e…) and `tests/test_domain_planner.py` (16dbe3e2…).**
  Built by implement → adversarial review (parity/pin; Logistics integration) → fix. An interrupted first attempt
  left drafts, which were finished rather than discarded.
  - **Design:**
    - Domain-generic subclasses `LMForwardD` / `LMProposerD`, which set `dom` before the inherited cache loader runs.
    - `ExactD` and `score_d`.
    - The planner functions are re-exported unchanged.
    - The import refuses (ImportError) unless `blocks_planner.py` is sha 18a5f22b…. The pinned file is unchanged
      and never patched.
  - **Tests** (28/28; 61/61 together with `test_logistics.py`; re-run independently):
    - with `dom = blocks`, byte-identical prompts, parses, cache bytes and candidates vs `LMForwardBW`/`LMProposerBW`
      on about 400 real cached keys;
    - Logistics `plan_beam`/`plan_direct` with exact F and scripted proposers agree with the executor and VAL;
    - stub-LM caching and refusal paths.
  - **Review fixes:** 5 confirmed issues (1 medium: a malformed imagined state crashed a parse), all fixed with
    regression tests.
  - **Remaining for item 9:**
    - CPU-4 harness check (40 instances at 4-7 packages, seed 20261405);
    - CPU-5 `lg_*` scripts;
    - the [VERIFY] items;
    - the freeze with user confirmation.
- **D-73 Full backups (user request, 2026-10-11).**
  - **GitHub (records and code, no weights; D-51):** BWM on main at f96b62e, with the key records verified on the
    remote (DECISIONS through D-71/X-07; all PREREGs; B3, H2, D-54, LB and 5c reports).
    - RCWM is fully pushed at bb7850e. Its own adapters up to 50 MB are now in the repo, by RCWM's choice; its 7.13
      GB of regenerable arrays are listed in `rcwm/EXCLUDED_LARGE_FILES.json`.
    - RCX was asked to push its last live logs.
  - **Full local archive:** `scripts/backup_bundle.py` →
    `WorldModel/_backups/bwm_native_v1_20261011_1541.zip`.
    - Contents: all of native_v1, including the 12 adapters (463 MB), training data, B3 data, the release package,
      vendored bench files, and the session memory.
    - 3,432 files, 627 MB, sha 8ded0d28…. Every member was re-verified against the sha256 manifest: 0 failures,
      0 errors.
  - **Off-machine:** at the user's choice, the archive, manifest and a restore README were copied to
    `C:/Users/clclc/OneDrive/Rineya_backups/` (sha verified).
    - **Caveat:** at copy time OneDrive.exe was not running and no account was registered, so the cloud upload
      happens only once the user starts OneDrive and signs in.
    - The base Titan-Sparse core is not in the archive; it lives in the vault.
