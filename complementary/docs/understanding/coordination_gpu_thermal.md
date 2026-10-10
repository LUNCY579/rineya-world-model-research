# Coordination mechanisms, GPU and thermal rules, and repo conventions in the WorldModel workspace

Legend: **[M]** means measured or read directly from a file or live state. **[P]** means a plan or proposal as written in the files. **[I]** means my own inference.

---

## 1. Existing coordination files and their conventions

### 1.1 The GPU lease (the main mechanism)

**[M] Lock file:** `C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock`. There is only one card ("card 0", an RTX 4070 Laptop that also drives the display).

**[M] Implementations.** All are vendored from `ChatBubble-titan-clare-sparse-v1/experiments/fresh_dual_skill_instrument_v1/gpu_lease.py`:
- BWM: `Rineya_BWM_001_Research_Bundle/native_v1/bwmn/gpu_lease.py`. Mirrored at `rineya-world-model-research/bwm/code/bwmn/gpu_lease.py` and `RCWM-T1/vendor_bwm/bwmn/gpu_lease.py`.
- RCWM: `Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/rcwmt/lease_vendored.py`, wrapped by `rcwmt/runtime.py::gpu_lease(label, job)`.
- BWM wrapper: `native_v1/scripts/gpu_run.py <label> -- <script.py> [args]`, driven by `scripts/orchestrate.py` and `scripts/bundle.py`.

**[M] Lock record (JSON).** Fields:
- `created`, `pid`, `token` (uuid4 hex), `label`, `job`
- `worker_pids` (BWM writes the child pid)
- `orphan_after_unix` (= created + `ORPHAN_AFTER_SECONDS = 900`)
- `staleness_rule`
- BWM bundles also carry `bundle_start` and `bundle_budget_s: 16200`.

**[M] Acquiring the lock.**
1. Poll every 20 s.
2. If `foreign_gpu_holder()` is None, create the lock with `open(LOCK, "x")` (exclusive create).
3. If the file already exists, call `_try_reclaim()`. Reclaim happens only if ALL of these hold: the holder pid is dead, `worker_pids` is empty, and `orphan_after_unix` has passed. The old record is backed up to `rineya-gpu-0.lock.stale-<pid>.bak`.
4. Both programmes call it with `timeout_seconds=6*3600`.

**[M] Releasing the lock.** Release is token-checked: the file is unlinked only if its `token` is still ours.

**[M] Foreign-holder rule (I-17 / D-25).** A python.exe counts as "on the card" if its dedicated GPU memory exceeds 256 MiB. This is read from the Windows counter `\GPU Process Memory(*)\Dedicated Usage`. BWM's copy falls back to the old rule (nvidia-smi compute-apps + RSS > 512 MiB) when counters fail.

**[M] Label prefixes in use:**
- `rcwm-t1:<job>`, for example `rcwm-t1:t2b-prereq`, `rcwm-t1:retention`, `rcwm-t1:t2pair:<tag>`
- `bwm:<label>` and `bwm:bundle-<name>`
- Historic examples: `fds1:gate`, `titan-probe`. Earlier participants included DOSELORA-001/ARMPERP-001 and the "Rineya Delta multi-reasoning session".

### 1.2 Turn alternation (two-party only)

**[M]** From RCWM I-16 and BWM D-26, implemented in `runtime.alternate_turn` and `gpu_run.wait_for_turn`:
- **Waiting markers:** `%TEMP%/rineya-gpu-0.waiting.rcwm.json` and `...waiting.bwm.json`, each `{pid,label,since}`. They exist while a job waits and are deleted once the lease is acquired.
- **On release:** write `%TEMP%/rineya-gpu-0.turn.json` = `{"last": "rcwm"|"bwm", label, released_unix[, bundle_start]}`.
- **Yield rule:** before acquiring, a job yields only if `turn.last == own side` AND the peer has a live waiting marker. The wait is capped at 2700 s (45 min), polling every 15 s. "No idle GPU when only one side has work."
- **BWM courtesy gap:** BWM adds 30 s, but only between two of its own consecutive jobs.
- **[I]** The protocol is hard-coded for the two sides "rcwm" and "bwm". A third party has no marker slot.

**[M] One-shot yields:**
- RCWM: `RCWM-T1/runs/.yield*.json`, format `{applies_to_prefix, peer_label, max_wait_s, consumed, consumed_by, peer_seen, waited_s}`.
- BWM: `native_v1/runs/.yield.json`, format `{label | label_prefix, max_wait_s, give_up_if_not_seen_s, note}`. It is renamed to `.yield_consumed_<unix>.json` after use. Three consumed files exist.

**[M] Turn length and bundling (BWM D-41/D-45, RCWM I-24/D-22):**
- A turn is at most 4.5 h (16200 s) of wall time from first acquire.
- A sub-job starts only if its estimate fits the remaining budget.
- Exit code 75 means "budget used, requeue". RCWM's t2bp runner uses exit 5 for "deferred by turn cap" and exit 3 for a deterministic stop that is never retried.
- A crash releases the lease.
- RCWM otherwise runs one job per turn. The exception is T2 train→eval pairs under one lease, `rcwm-t1:t2pair:<tag>`.
- After I-15/I-29, `bundle.py` kills the whole process tree at the cap with `taskkill /F /T`.

**[M] RCWM "one-waiter rule" (R14).** Only one `rcwm-t1:*` job may wait for or hold the lease at a time. It is enforced by `one_waiter_problem()` in `scripts/p3_run.py` and `scripts/t2bp_run.py`. RCWM jobs are queued via `scripts/master_chain.sh`, which globs `scripts/queue_hooks/*.sh` once. Hooks in use: 50, 60, 65 and 75; **70 is reserved for the joint PREREG-LB**.

### 1.3 Thermal, power and RAM rules

**[M] Shared watchdog:** `RCWM-T1/scripts/thermal_watchdog.py` (v3.6), described as "Shared by both Claude sessions". It is the **only flag writer** (RCWM I-30: "BWM's health logger never writes the flag").
- **Files:**
  - flag `RCWM-T1/runs/.thermal_pause` (contents = reason: a temperature, `battery`, `ram x`, `tz01 x`)
  - stop file `runs/.thermal_stop`
  - log `runs/thermal_log.txt` (one line per minute)
- **Sampling:** every 1.5 s. The CPU zone tz01 is polled every 30 s in a thread. Sensors are read in-process via NVML/PDH (`scripts/sensors_noproc.py`); nvidia-smi is a fallback at most 2 per minute.
- **Pause conditions:** GPU ≥ **76 °C**; on battery; available RAM < **3 GB**; ACPI `\_TZ.TZ01` ≥ **98 °C**.
- **Resume conditions:** all of AC power, GPU ≤ **66 °C**, RAM ≥ 3 GB (≥ 5 GB if the pause was RAM-caused), and tz01 < 98 (≤ 90 if the pause was tz01-caused). These were made reason-specific in v3.4/v3.5 after stalls I-25 and I-26.
- **Power:** while on AC AND the lease file exists, the watchdog holds `SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED)`. This was added after the Modern Standby bugcheck 0x19C (I-19/I-20).

**[M] In-process guards:**

| | RCWM (`rcwmt/runtime.py`) | BWM (`bwmn/lm.py`) |
|---|---|---|
| Temperature guard | pause ≥ 80 °C until ≤ 72 °C | pause ≥ 82 °C until ≤ 75 °C and flag gone; 0.3 s yields at ≥ 78 °C; unreadable temperature = hot |
| Duty cycle | `_duty_ratio` 0.5 / 1.5 / 3.0 at < 70 / 70–74 / ≥ 75 °C; `BREATHE_S` 0.12 | `BWM_DUTY_RATIO` 3.0 (about 25% duty) |
| Memory | `set_per_process_memory_fraction(0.70)`; `VRAM_FLOOR_GIB 3.6`; `HOST_FLOOR_GIB 2.0` | `MEM_FRACTION` 0.70 |
| Shared flag | honoured | honoured (`PEER_FLAG` = RCWM-T1 `.thermal_pause`) |

BWM's CPU guard (`cpu_thermal_guard`) pauses at tz01 ≥ 98 °C until ≤ 90 °C. It honours the peer flag only for machine-wide reasons: RAM, tz01, battery (D-42).

**[M] RAM floors (D-39, agreed by both sides):**
- GPU model load waits for ≥ 3 GB free. After 30 min it exits 75 and gives the turn back.
- CPU-lane loads wait for ≥ 12 GB (`BWM_RAM_FLOOR_GB="12"` set by the orchestrator; the code default is 8). The point is that "≥ 6 GB stays free for either session's GPU job".
- The BWM CPU lane uses 6 threads.

**[M] CPU-only torch recipe (RCWM I-21, BWM D-35):**
- Set `CUDA_VISIBLE_DEVICES=""` before importing torch, and patch `torch.cuda.is_available`/`device_count` to False/0.
- `CUDA_VISIBLE_DEVICES=-1` crashed in 5 of 5 runs.

**[M] Process-spawn hygiene (I-27 / BWM I-14).** RevoUninHelper.exe leaked kernel objects in proportion to process spawns (about 40 per minute from the watchdog). Rule: no subprocess per sample; long-lived watchers only.

**[M] Detached launch.** Use WMI → bash → python (BWM D-19) or hidden `.cmd` launchers (RCWM `scripts/launch_*.cmd`, `ShowWindow=0`).

### 1.4 Schedule

**[M] No clock-based windows exist.** Scheduling is strict turn alternation. The joint roadmap says "about 2-3 turns per programme per day" and has a Day 1–5 table (see §4).

**[M] Live state when I read it (2026-10-09 22:21 -0500):**
- **Lock:** held by `bwm:bundle-bundleD` (pid 32060, worker 31372). Created 22:05:30; budget ends **02:35:30 on 10-10**.
- **RCWM waiter:** `rcwm-t1:t2b-prereq` (pid 42092, since 22:03:30). This is the 5a screen, 3.83 GPU-h, one or two turns.
- **turn.json:** `last: rcwm` (`rcwm-t1:lik:sparse`, 17:50:55).
- **Watchdog:** no `.thermal_pause`. Latest log line: t=67 °C, RAM 12.0 GB, tz01 96.1.
- **Stale backups:** seven `.stale-*.bak` files exist.
- **BWM GPU queue (`runs/jobs.json`):** bundleD → bundleE (B3 distillation, 8 sub-jobs, about 11,940 s estimated) → a2-train-Fvlab … → bundleF (pilotG replications).

---

## 2. Messages or requests addressed to a complementary session, or mentioning RCM/RGM/RVM/RSM

**[M] There are none in the three target directories.** I searched for `COLLABORATION`, `complementary session`, `third session`, `Request:`, `request to`, and messages to a peer. All hits are RCWM↔BWM only. No file-based mailbox exists. Agreements are recorded as "agreed by message" or "confirmed by message" (RCWM D-34; `.yield.json` `agreed_with`). **[I]** That suggests out-of-band session messaging, recorded afterwards in each programme's DECISIONS.md.

**[M] The only "complementary" statement relevant to the programmes:** `RCWM-T1/docs/understanding/bwm-and-lineage.md:58`: "They are complementary: BWM's forward model plus exact evaluator would become RCWM's learned consequence predictor once exact evaluation is unavailable."

**[M] RCM-001 / RGM-001 / RVM-001 / RSM-001 are not in the three directories.** They exist only on the remote branch `origin/research/rineya-high-quality-dataset-v1`, under `datasets/rineya_high_quality_factory_v1/`. That is a third session's CPU-only "dataset factory" with ten tracks, listed in `DATASET_BLUEPRINT.json`:

| Track | Name | Priority |
|---|---|---|
| RCM-001 | Causal and counterfactual rule models | P0 |
| RGM-001 | Relational graphs and neuro-symbolic | P0 |
| RVM-001 | Validity critics, value models, successor features | P0 |
| RSM-001 | Hierarchical policies and reusable skills | P0 |
| REM-001 | (memory track) | P1 |
| RMM-001 | (modular experts track) | P1 |
| RBM-001 | (belief-state track) | P1 |
| RJP-001 | — | P2 |
| RDP-001 | — | P2 |
| ROM-001 | — | P2 |

Approved records per its `NEXT_SESSION_HANDOFF.md`: RCM 1440 (180 families), RGM 432 (180), RVM 384 (192), RSM 24 (12). Its handoff rules include:
- "never switch or clean the research checkout"
- "Create no sealed evaluation under current exposed custody"
- "Git backup uses a separate dedicated worktree"

**[M] ID collision: RCI-001 is already taken.** It is an existing calibration bank: `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/rci001-20260918/RCI_SEAL.json` (128 rows; RCI-002 has 256 rows). Status: "SPENT for confirmation… remains admissible for DEV and SELECTION". It is cited in `l1_screen.py:2706`, `PREREG_L1_DRAFT.md` and the L1 vault drafts as the confirmed abstention-damage result (Titan-Sparse 25/128 vs Supa 40/128, p = 6.1e-5).

---

## 3. Rules for committing to the shared repo

**[M] Repo:** `https://github.com/LUNCY579/rineya-world-model-research`, PRIVATE, default branch `main`. It was created by the RCWM session on 2026-10-08 (RCWM D-31, BWM D-51).

**[M] `README.md` "Rules for contributors" (verbatim):**
- "Files over 25 MB stay out; list them in an exclusion manifest instead."
- "No secrets, tokens or personal data."
- "Run `git pull --rebase` before every push. Each session commits only inside its own folder."
- "Nothing here is a release. Promotion of any core into the private vault, or upload to Hugging Face, requires the operator's explicit go-ahead."

**[M] Folder table in the root README:** `rcwm/` (RCWM-T1 mirror), `bwm/` ("Maintained by the BWM-001 session"), `tools/` (shared export scripts). No other folder is registered.

**[M] `.gitignore`:** `*.npy *.bin *.pt *.pth *.ckpt *.onnx model.safetensors __pycache__/ *.pyc *.lock .thermal_* tools/.hash_cache.json`.

**[M] Tools:**
- `tools/export_rcwm.py`: mirrors the RCWM bundle into `rcwm/`, one process, never writes into the source. `MAX_BYTES` 25 MB; allow-listed T2 adapters up to 50 MB; writes `rcwm/EXCLUDED_LARGE_FILES.json` with size and sha256.
- `tools/export_bwm.py`: copies from `native_v1` into `bwm/` and replaces `bwm/` except `bwm/README.md`. It excludes all adapter weights ("user decision 2026-10-08: no weights in the repo"), training data, caches and host health logs. A secret/email regex scan must pass before anything is written; the only allowed email is `noreply@anthropic.com`.

**[M] One clone per session:**
- RCWM uses `WorldModel/rineya-world-model-research`.
- BWM uses its own clone, `Rineya_BWM_001_Research_Bundle/gh_export`, currently at 5f8f885, behind main's a223884.
- The dataset session uses its own branch and a dedicated worktree. Its branch touches only `datasets/`; `git diff --stat main..origin/<dataset-branch> -- README.md .gitignore tools bwm rcwm` is empty.

**[M] Current branches and worktrees:**
- `main` (a223884)
- local `research/rineya-complementary-intelligence-v1`, checked out at `WorldModel/rineya-complementary-worktree`, at a223884, clean, no remote yet
- a detached worktree `.../scratchpad/dataset_ro` at 7a7d055, inside this session's scratchpad. **[I]** It was probably created by a sibling reader. It stays registered in the shared repo until `git worktree remove`.

**[M] Commit conventions observed:**
- Message prefixes: `rcwm: …`, `bwm: …`, `Sync rcwm: …`, `dataset(s): …`.
- `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` trailer.
- LF line endings: RCWM I-30 says "scripts write files with `newline='\n'`".
- Repo size: pack 19.9 MiB, 1194 tracked files (rcwm 903, bwm 287). The largest tracked files are the three T2 adapters at 40.4 MB each.

---

## 4. The joint BWM–RCWM plan: items and owners

**[M] Source:** `native_v1/research/JOINT_ROADMAP_2026-10-08.md`, mirrored at `bwm/docs/research/`. It was merged by BWM from both brainstorms (RCWM D-32). Owners: **[B]** BWM-001, **[R]** RCWM-T1, **[J]** joint. Status below is from the DECISIONS logs as of 10-09 evening.

| # | Item | Owner | Status [M] |
|---|---|---|---|
| 1 | Finish frozen Blocksworld runs (hard, mirror, PlanBench-600) | B | Hard **FINAL**: 73/110 vs 5/110, p = 6.8e-21 (P-08). Mirror interim 38/70 (P-09). PB-600 pending. |
| 2 | Out-of-range audit, rewritten as PREREG-3 "rule or trigger?" | R | **Done 07:19 10-09: STATUS: TRIGGER.** Key order 0.37–0.47 vs twins 0.87–0.91. |
| 3 | Beam search over the RCWM ledger, with T2 as F in BWM's planner | J (R runs, B code) | RCWM kit ready (D-37, 73/73 checks). BWM hooks (D-53). PREREG-LB draft by BWM (D-57). `blocks_planner.py` frozen at sha 18a5f22b…. RCWM writes `lbeam_run.py`. Hook 70 reserved. About 1.7 GPU-h. |
| 4 | Search distillation (expert iteration) | B | Pipeline built (D-56). Addendum B3 preregistered (D-58, sha e6aa884b…). Queued as bundleE. |
| 5a | Precondition representation screen | R | Rule frozen (D-38, SCREEN_RULE sha 530a9dbd…). Hook 75 installed. It is the current RCWM waiter. |
| 5b | Trace-in-answer targets `{"cmp","calc","after"}` + filler control | J | Format agreed (RCWM D-34 = BWM D-52). Filler amendment 4b (constant length) adopted by both (D-36/D-55). 5b identification fix in progress (wf_84e8f08d-166). |
| 5c | Correctness-based retention suite | J | Frozen (D-36). Dev selection R17: 5 of 8 categories. |
| 6 | Shared legality checker (minimal pairs) | J | Made concrete through 5b. Not started. |
| 7 | Two more seeds of BW_F / Pa_v2 | B | Not started. |
| 8 | Exact arithmetic inside the step | R | Not started. |
| 9 | Transfer to a second domain | B | Not started (only a mystery-Blocksworld smoke run exists). |

**[M] Zero-GPU items:**
- [B] novelty signal for confident errors (F-06); devfreeze package and repo export.
- [R] MLP-vs-Titan hybrid analysis; no-LLM learners at 629 matched rows; k = 8/16 and seed-consensus certificates.
- [J] shared shortcut-audit checklist, now formalised as the **D-35 joint rule**: every claim gets a surface-form audit (names, key order, list order, held-out trigger words) as well as the out-of-range audit.
- Joint finding R16 / F-08: both LoRA world models are key-order-locked and name-invariant.
- D-54: augment input key and list order only; test on held-out orders.

**[M] "Do not do" list:**
- disagreement or confidence deferral as the prerequisite fix
- dense merges of rule LoRAs
- blanket verifier calls
- tuning on spent or reserved instruments (pilot TEST, t2_eval/supp, R14's 248, L1's sets)
- latent single-pass composition
- reporting tool tiers as model ability

**[P] Turn order (Day 1–5 table):**

| Day | BWM | RCWM |
|---|---|---|
| 1 | #1 hard shards | PREREG-2 → L1 |
| 2 | #1 mirror + PB-600 → #4 traces | #2 → #5 T2b s0 |
| 3 | #4 train + eval | #3 → #5 s1 |
| 4 | #7 seeds | #5 s2 → #8 |
| 5 | #6 joint → devfreeze → #9 scoping | #6 joint |

The release gate is unchanged: nothing goes to Hugging Face or the vault without the user's explicit go-ahead.

**[M] Nothing in the joint plan assigns work to a third or complementary programme.**

---

## 5. Recommended conventions for `complementary/docs/COLLABORATION.md`

Everything in this section is **[I]**, grounded in the facts above.

**Scope and where to write**
- Write only under `complementary/`. Do not touch `rcwm/`, `bwm/`, `tools/`, `datasets/`, the root `README.md` or `.gitignore`.
- To add a row for `complementary/` to the root README's folder table, propose it by message to both sessions and get the operator's OK first. The dataset session never edited shared files.
- Stay on `research/rineya-complementary-intelligence-v1` in its own worktree (`rineya-complementary-worktree`). Push to an `origin` branch of the same name. Never check out, reset or clean `rineya-world-model-research` (RCWM's live export target) or `gh_export` (BWM's). Do not merge to `main` without the operator.
- If you ever commit to `main`: run `git pull --rebase` before every push, and use the commit prefix `complementary:` with the Co-Authored-By trailer.

**Programme and ID naming**
- Do not use **RCI-001/RCI-002**: they are existing spent calibration banks.
- Do not use RCM/RGM/RVM/RSM/REM/RMM/RBM/RJP/RDP/ROM-001 (dataset tracks), RCWM-001/RCWM-T1, BWM-001, or SPARSE-001, SELECT-VIEWS-001, HSKILL-*, RRP-015, REP-001, RAB-001.
- `RCX-0xx`, `CPL-0xx` and the label `rci:` had zero hits across the three directories. Re-grep the records trees under `C:/Users/clclc/BusinessProject/` before adopting any of them.
- Prefix decision and incident IDs with the programme in cross-references, for example "CX D-01" versus "RCWM D-34" / "BWM D-52". Both peers reuse D-/I-/R-/F- numbering independently.
- **GPU lease label prefix:** use something distinct like `rcx:`. Avoid `cmp:` (it is the trace key) and avoid `rcwm-t1:` and `bwm:`. RCWM's one-waiter check matches only `rcwm-t1:*`, so a distinct prefix will not falsely trip it.

**GPU (state "CPU-only by default")**
- GPU use requires an explicit request to both sessions and the operator's go-ahead.
- If granted:
  - Use the same lock file and a copy of `rcwmt/lease_vendored.py`, which is token-checked and has the 256 MiB dedicated-VRAM holder check.
  - **Never** run any GPU work outside the lease. I-05 and I-15/I-29 are the precedents: one CUDA crash and about 1 h of peer delay.
  - Do **not** write `turn.json` or the `waiting.{rcwm,bwm}` markers. The protocol is two-party, and a foreign `last` value would reset strict alternation.
  - Take the card only when the lock is absent and neither side's waiting marker names a live pid, or through a pre-agreed one-shot `.yield` file on each peer's side.
  - Keep each turn ≤ 4.5 h. Kill the whole process tree (`taskkill /F /T`) at the end. Use `set_per_process_memory_fraction(0.70)`, a duty cap of ≥ 1.5×, and check `RCWM-T1/runs/.thermal_pause` between batches.
  - Never write the flag. Never reclaim a lock whose `worker_pids` list is non-empty; ask the owner (I-31 precedent).

**CPU work**
- Use `CUDA_VISIBLE_DEVICES=""` plus the `torch.cuda` patch (never `-1`), and ≤ 6 threads.
- Keep ≥ 6 GB RAM free for the GPU job: load CPU models only when ≥ 12 GB is available.
- Pause at tz01 ≥ 98 °C until ≤ 90 °C. Honour the flag for RAM, tz01 and battery reasons.
- No per-sample subprocess polling; read `thermal_log.txt` or `health_now.json` instead.

**Requests and messages**
- Keep a dated `complementary/docs/REQUESTS.md`. Each entry should state: an ID, the addressee (RCWM-T1 or BWM-001), the files read (read-only, with path and sha256), the resources (CPU threads, RAM, GPU-h, turn), and the status.
- Also deliver every request by message. Peers do not read our folder, and agreements are logged in their DECISIONS.md, not in shared files.

**Data hygiene**
- Read peer results only as pinned citations (commit, path, sha256).
- Never modify frozen files (`blocks_planner.py` 18a5f22b…, `rcwmt/runtime.py`, freeze-bound docs).
- Never copy sealed or spent bank content into `complementary/`.
- Keep generated data outside the RCWM/BWM bundles. Their freshness and exclusion harvesters scan the records trees and the BWM bundle (more than 15k files; I-28).

**Files**
- ≤ 25 MB each, listed in `complementary/EXCLUDED_LARGE_FILES.json` with size and sha256.
- Run the `export_bwm.py` secret/email regex before every push.
- LF line endings.
- No weights unless the operator explicitly approves.