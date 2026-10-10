# COLLABORATION: complementary session (RCX) with BWM-001, RCWM-T1 and the dataset factory

This is the shared coordination record for the complementary-intelligence session. All of its work is on branch
`research/rineya-complementary-intelligence-v1`, under `complementary/`.

**Session.** "Rineya complementary intelligence research". It is a Claude Desktop session started 2026-10-09 at about
22:13 CDT.

**Labels.**
- Programme label: **RCX** (Rineya Complementary eXperiments).
- Decision IDs: `CX D-nn`.
- Request IDs: `CX Q-nn`.
- Run IDs: `RCX-Exx-...`.
- We do **not** use `RCI-001` as a run or bank label. That ID already names a spent calibration bank in
  `rineya-dataset-retention-records/rci001-20260918/`. The bundle's `RCI-001` remains only the name of the planning
  bundle.

**Specification.** The research bundle `rineya_complementary_research_2026-10-09/` (RCI-001 v1.0), at the workspace
root, outside the repository.

## 1. Who does what (as read on 2026-10-09 at about 22:25 CDT)

Sources: `docs/understanding/*.md`. These are read-only surveys written by this session; peers' own records are
authoritative.

| Programme | Session | Owns (do not duplicate) | Live state |
|---|---|---|---|
| **BWM-001**: forward world models and planning | `Rineya_BWM_001_Research_Bundle` session | <ul><li>Blocksworld/RW2 frozen runs (hard 73/110 vs 5/110; mirror; PB-600)</li><li>**B3 search distillation (= bundle E10)**</li><li>D-54 key-order augmentation</li><li>second domain (#9)</li><li>planner code for joint item 3</li><li>frozen `bwmn/blocks_planner.py` sha `18a5f22b…` (D-57)</li></ul> | GPU lease `bwm:bundle-bundleD` until about 02:35 on 10-10; then bundleE (B3) is queued |
| **RCWM-T1**: consequence world models on frozen Titan-Sparse | `Rineya_Consequence_World_Model_Bundle_2026-10-05` session | <ul><li>PREREG-1/2/3</li><li>T2 adapters (unmerged FP32)</li><li>**T2b kits: 5a prerequisite and 5b register (surface-diversified, fixed rule)**</li><li>5c retention suite (D-36 gate)</li><li>PREREG-LB (joint item 3)</li><li>roadmap #5 (dissect T2, SD/NAT controls)</li><li>#6 long horizons</li><li>#8 no-LLM learners on T2's ledger rows</li><li>PREREG-4 (T3)</li><li>the thermal watchdog</li></ul> | waiting in the lease as `rcwm-t1:t2b-prereq` (5a screen, about 3.8–6.1 GPU-h) |
| **Dataset factory** (`research/rineya-high-quality-dataset-v1`) | a separate CPU-only dataset session | approved pilot data for all ten RCI tracks (RCM 1,440 records, RGM 432, RVM 384, RSM 24) | No model trained or evaluated. No sealed split. Its dev_challenge split is exposed. |
| **RCX** (this session) | complementary research | see §2 | CPU only |

## 2. What RCX takes on, and why it is complementary

| Track | RCX scope | Explicitly *not* RCX's scope (belongs to a peer) |
|---|---|---|
| **RCM-001** | **Rule-conditioned execution.** The rule is *supplied in context* and varies across examples: opposite meanings, nonce status words, new status sets, compositions. The question is whether a learner reads the rule instead of keying on a trigger token. This covers the E01 surface × trace factorial with counterfactual twins, the rule-swap/ablation tests and the PREREG-3-style RULE/TRIGGER verdicts on **fresh** vocabularies. It also covers a qualified rule-parameterized interpreter, an anti-shortcut battery and shortcut floors. | <ul><li>Fixed-rule surface diversification on Titan-Sparse (T2b 5a/5b)</li><li>retuning or regenerating RCWM's frozen banks</li><li>SD/NAT dissection of T2 (#5)</li><li>no-LLM learners on T2's own rows (#8)</li><li>PREREG-4</li></ul> |
| **RGM-001** | <ul><li>E07: relational representation comparison (text vs canonical vs flat MLP vs GNN) with structural OOD (more objects, held-out names and orders)</li><li>E08: learned heuristic with exact dynamics</li></ul> | <ul><li>Blocksworld LM forward models</li><li>D-54 augmentation</li><li>the second domain</li></ul> |
| **RVM-001** | <ul><li>Successor features</li><li>value / dead-end critics to rank beam candidates</li><li>a CPU logistic legality-critic baseline</li></ul> | joint item #6 (legality-checker LoRA) unless its owner hands it over |
| **RSM-001** | E11 option composition and E12 DAgger. **No B3 reads before `runs/BW_B3_EVAL/B3_REPORT.json` exists.** | E10 search distillation (that is BWM's B3) |

## 3. Resource policy (binding on RCX)

- **CPU only by default, in a single CPU lane** (one heavy CPU job at a time, per BWM I-12).
  - At most **4** torch threads. The BWM CPU lane uses 6, and the agreed ceiling is 6.
  - Use `CUDA_VISIBLE_DEVICES=""` plus the `torch.cuda` patch before importing torch, never `-1` (RCWM I-21).
- **Thermal courtesy.**
  - The CPU zone tz01 already sits at about 94–96 °C, and the shared watchdog pauses **everyone's GPU work** at
    tz01 ≥ 98 °C.
  - RCX runners therefore read the watchdog's `thermal_log.txt` (no subprocess per sample, per I-27) and pause at tz01 ≥
    97 °C until ≤ 93 °C.
  - They also pause whenever `.thermal_pause` exists, whatever the reason.
- **RAM.**
  - No 0.6B model is loaded on CPU unless at least 12 GB of host RAM is free (D-39).
  - RCX small-model runs stay under 1.5 GB resident.
- **GPU.**
  - None without:
    1. a written request to both peers in `docs/REQUESTS.md` plus a message;
    2. explicit operator approval;
    3. the shared lease `C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock`, with label prefix `rcx:`.
  - RCX never writes `rineya-gpu-0.turn.json`, `rineya-gpu-0.waiting.*.json` or `.thermal_pause`. The turn
    protocol is two-party, so a third party must use a pre-agreed one-shot `.yield` or wait for an idle card.
- **Data hygiene.**
  - RCX writes nothing inside the BWM or RCWM bundles, because their freshness harvesters scan those trees.
  - Peer code is imported read-only with `sys.dont_write_bytecode = True`, and the sha256 of every imported peer file is
    recorded.
  - RCX never opens these sealed or spent RCWM instruments:
    - `HSKILL_CONFIRM_V2*`
    - `bank_confirm*.json`
    - `bank_p3_oor.jsonl` rows
    - the retention suite's sealed draws
  - BWM's reserved sets (hard-110, mirror, PB-600, b3fresh) and its selection sets (val_large, val_v2):
    - RCX **reads hard-110, val_large, mirror_hard and b3fresh only to build exclusions** (renaming signatures and
      `blocks_val_large.invariant`), exactly as BWM's `bw_b3_prep.py::fresh` does;
    - it **never evaluates on any of them**;
    - see `rcx/bw/excl.py` and `experiments/RCX-E07/EXCLUSIONS.json`, corrected 2026-10-10 at BWM's request.
  - RCX reuses PREREG-3's *stratum taxonomy and verdict logic* only, with fresh words and seeds.
- **Seeds.**
  - Seeds already in use elsewhere (avoid):
    - BWM 20261005–20261011
    - RCWM `dev_20261121` and 20261141–43
    - the factory 20261009
  - RCX uses the **2026-12-xx** range (`202612xx`) and records every seed in `DECISIONS.md`.

## 4. Interfaces RCX may import read-only (sha pinned at first import)

| Purpose | Module | Owner |
|---|---|---|
| Ledger/register/prerequisite dynamics, prompts, parser | `RCWM-T1/rcwmt/env.py` | RCWM |
| PREREG-3 gold and scorer helpers | `RCWM-T1/scripts/p3_common.py` | RCWM |
| Blocksworld / RW2 simulators, BFS, VAL | `native_v1/bwmn/blocks.py`, `rw2.py` | BWM |
| Beam planner (frozen) | `native_v1/bwmn/blocks_planner.py` | BWM |
| Paired statistics, split audit | `08_reference_tools/paired_metrics.py`, `audit_splits.py` | bundle |
| Factory data/oracles (no writes; no `generate()` / `catalog` / `status`) | `datasets/rineya_high_quality_factory_v1/` | factory |

## 5. Requests and messages

Open requests are listed in `docs/REQUESTS.md`. Every request is also delivered by session message, because peers record
agreements in their own DECISIONS.md.

## 6. Findings shared with peers

Each entry gives a date, a finding, its evidence path, and the peers it was sent to.

- **2026-10-10 about 14:35, R-03, sent to RCWM and BWM.**
  - Frozen Titan-Sparse and T2 s0 ignore in-context ledger rules: a no-op in 1 of 280 cases that require one, and
    rule-polarity sensitivity 0.00.
  - T2 shows PREREG-3's TRIGGER signature again on fresh RCX words: void 1.00, near-miss 0.97, unseen 0.00.
  - Evidence: `EXPERIMENT_RESULTS.md` R-03 and `runs/RCX-E01N/probe1/`, commit 980c360.

## 7. Peer agreements received

- **2026-10-09 about 22:55, RCWM-T1.**
  - No overlap with rule-conditioned E01.
  - Imports allowed with pins. Never call `bank.load_exclusions()` and never touch HSKILL files.
  - Retention confirms go through RCWM (CX D-06).
- **2026-10-09 about 23:00, BWM-001.**
  - B3 is BWM's. No B3 reads before `runs/BW_B3_EVAL/B3_REPORT.json` exists.
  - E07/E08 use fresh instances with BWM's exclusion recipe, and the reserved sets stay untouched.
  - E04 BW pairs come from `traces.bw_parts`.
  - Machine rules: a single CPU lane, and a 4 GB floor for jobs over 2 GB (CX D-07).
- **BWM-001 (FYI), received after 2026-10-09 23:17 and before 2026-10-10 06:05.** The earlier entry wrongly said
  "2026-10-10 about 23:20"; RCX did not record the exact arrival time, so this was corrected at BWM's request on
  2026-10-10.
  - D-54 key-order augmentation is preregistered as bundleG, after B3: `native_v1/PREREG_D54_KEYORDER.md` (sha
    6b704388…).
  - Held-out state-key orders H1–H3: (table, on, hand, clear), (hand, table, clear, on) and (on, clear, table, hand).
  - RCX must avoid seeds 20261012 and 20261013. Its own seeds are 202612xx, so they are clear.
  - Any RCX Blocksworld canonicalization control treats H1–H3 as the held-out orders.
- **2026-10-10, synced from origin/main (RCWM D-39, D-45).**
  - RCWM records the RCX boundary as D-39, matching CX D-06.
  - D-45 adds an RCWM item, **T2b-ledger**: a surface-diversified **fixed-rule** ledger adapter. Its sealed
    confirmation covers:
    - PREREG-3's out-of-range map on held-out surfaces;
    - non-inferiority to T2 s0;
    - 5c retention;
    - LB-style planning.
  - This is RCWM's fixed-rule arm. It is complementary to RCX's **rule-varied** native cells (N-VR, N-CFSD). A planned
    cross-evaluation would score each programme's adapters on the other's instrument (RCX RCL battery vs PREREG-3-style
    map) once both exist, to be proposed as CX Q-03 at that time.
  - Shared verdict wording (RCWM D-45 and BWM D-54), adopted by RCX: "generalises to unseen X" / "trigger widened" /
    "partial".
- **2026-10-10 about 14:40, RCWM-T1.**
  - R-03 is logged as RCWM **R19**, citing 980c360.
  - RCWM's PREREG-5 (T2b-ledger) limits will state that fixed-rule surface diversification can at most earn "follows the
    frame's fixed rule on unseen surfaces", never "reads a supplied rule".
  - **Cross-scoring is accepted in principle:**
    - RCWM's PREREG-3-style map runs on RCX N-VR and N-CFSD;
    - RCX's RCL battery runs on T2b-ledger;
    - each side runs its own instrument and exchanges **adapter directories and shas only**;
    - neither side sees the other's confirmation banks before scoring;
    - any GPU runs use the turn protocol plus REQUESTS.md.
  - CX Q-03 is to be sent when the adapters exist.
