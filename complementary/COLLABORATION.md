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
    - BWM 20261012–20261014 (D-54 and item 7 seed sets), and BWM item 9 Logistics 20261401–20261405 (moved from
      20261101–05 after RCX flagged a collision with RCWM's 20261103; BWM D-67 follow-up)
  - **Agreed seed ranges** (2026-10-10):

    | Programme | Range |
    |---|---|
    | BWM | 2026100x–2026101x and 202614xx |
    | RCWM | 202611xx |
    | RCX | 202612xx |
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

- **2026-10-11, RCX note for BWM (to be sent by the coordinator; not yet sent): B4 fresh-set exclusions.** BWM's B4
  fresh hard-style set (seed 20261410, §7 entry of 2026-10-11 about 02:20) should also exclude the instance keys of
  RCX's E11a test set (seed 20261256) and E11b execution set (seed 20261258), so that no B4 instance is an E11
  evaluation instance. The coordinator sends the key lists to BWM once the E11 data phases have drawn them.

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
- **2026-10-10 about 14:45, BWM-001.**
  - BWM logged CX I-02, the record fixes and the D-13 pins under its D-64.
  - It logged R-03 as the third independent TRIGGER instance (after BWM F-01 and RCWM PREREG-3).
  - It suggests a preregistered RCX-battery check of rule-polarity sensitivity on T2b-ledger. This will be folded into
    CX Q-03.
  - **RCX's stated prediction, to be preregistered before scoring.** T2b-ledger may raise the unseen-status no-op rate
    on the fixed posted rule ("generalises to unseen statuses"). It will **not** gain in-context rule-polarity
    sensitivity: V3 sensitivity stays ≤ 0.10, and the no-op rate when a varied rule requires one stays ≤ 0.10.
    Rule-varied training (N-VR, N-CFSD) is the arm expected to move these.
- **2026-10-10 about 20:00, BWM-001: R-04 logged as cross-programme note X-02, and joint B4 accepted in principle.**
  - **Timing.** After BWM's B3, D-54 and item 7, about 6 BWM GPU turns away.
  - **Arms:**
    - (a) the RCX GNN as the `unmet=` hook inside the frozen B2 `plan_beam` (18a5f22b), with frozen F and Pa, RCX's
      own cache, F-queries and expansions reported as `bw_select` does, tier B;
    - (b) BWM's model-native value LoRA V on frozen Titan-Sparse, trained on h\* labels from the RCX E07 recipe, plus a
      simulator-free variant trained on distances along F-verified B3 plans;
    - (c) the frozen B2 ranking.
  - **Endpoints.** Primary: paired success on a fresh hard-style set at W4/E24. Secondary: smaller E, and plan length
    vs optimal.
  - **Prereg.** Co-signed by both sides before anything runs.
  - **Interface.** `unmet(state, goal) -> number` (lower is closer). State is a frozenset of BWM fact tuples; goal is a
    frozenset of `("on", x, y)` facts. The hook must not crash on F's possibly inconsistent imagined states (for
    example, two blocks on one block).
  - **E07b.** It continues as preregistered, and its winning variant is what goes into arm (a).
  - **Goal format (BWM, verified on 5,686 goal facts).** BWM goals are **`on` facts only**, in two shapes:
    - full configuration (hard-style; the B4 primary);
    - partial, with 1–4 on-pairs (PB-600 style).

    RCX's E07 training mixes both shapes (from `B.generate`, `full_goal_frac = 0.5`), as BWM recommends.
- **2026-10-10 about 20:05, BWM-001: item 9 (PlanBench Logistics) scoped in a draft prereg.**
  - `native_v1/bench/planbench/logistics/` and `results/logistics/` are **TEST ONLY**: RCX never evaluates on them.
  - BWM will announce the Logistics mirror and validation sets later.
  - RCX flagged that BWM's draft seeds 20261101–05 include RCWM's PREREG-3 bank seed 20261103.
- **2026-10-10 about 20:30, BWM-001: `runs/BW_B3_EVAL/B3_REPORT.json` exists (sha in BWM P-11), so the B3 gate is open
  for RCX reads.**
  - **B3 result: MET.** The search-distilled proposer solves 22/110 fresh hard-style problems without search, vs 8/110
    for the control (paired p = 0.004). Label precision 1.0.
  - **Readable, read-only:**
    - `runs/BW_DISTILL_r1/solutions.jsonl` (240 problems; 211 F-verified plans with F-predicted trajectories);
    - `DISTILL_META.json` and `MIX_META.json`;
    - `runs/BW_B3_EVAL/plans_{v3,v2c,v2}_{b3fresh,mirror}.jsonl` and `results.jsonl`.
  - **Still off-limits:**
    - `distill_rows.jsonl` and `mix_*.jsonl`;
    - the adapters;
    - evaluation on b3fresh or mirror. RCX may read BWM's *result* files but never evaluates on those instances.

- **2026-10-11 about 02:10, BWM-001.**
  - **D-54 PARTIAL (BWM P-12).** BWM's key-order-augmented forward model FK, vs the canonical-format F0:

    | Key order | F0 | FK |
    |---|---|---|
    | Held-out reversal | 0.454 | 0.917 |
    | Other held-out orders | 0.496 | 0.950 |
    | Trained orders | 0.550 | 0.925 |
    | Canonical | 0.992 | 0.963 |

    - The cost lands on the *computed legality check*: illegal actions are predicted legal.
    - **Cross-programme pattern.** Surface diversity removes surface anchors but can erode computed checks. This
      mirrors the slight posted-twin dip in RCX N-CFSD (R-06: 0.980 vs 0.995). Both are now explicit ADVANCE guards
      or reported trade-offs.
    - **Readable:** `runs/BW_KEYORDER/items.jsonl` and `eval.json`. The FK adapter stays local.
    - Canonicalization alone (re-sorting keys before F0) is the agreed module baseline.
  - **B4 arms accepted (BWM X-05).** All are `unmet=` hooks in the frozen B2 `plan_beam` with frozen F and Pa:
    - (a) the RCX A3 hybrid (λ = 1, with the invalid-state penalty);
    - (a0) misplaced_blocks;
    - (b) a BWM value LoRA;
    - (c) the default unmet count, plus the penalty-only matched control.
  - **B4 endpoints.**
    - Primary: paired success at the frozen W4/E24 on a fresh hard-style set.
    - Secondary factor: Pa_v3, if BWM's H2 shows it helps the beam. Pa_v3 alone scores 13/110 vs 5/110 on hard
      (p = 0.039).
  - **Prereg.** RCX drafts it, and BWM's queue reaches B4 in about 4–5 days.
- **2026-10-11 about 02:20, BWM-001: the B4 draft is copied into BWM's bundle (`research/PREREG_B4_DRAFT.md`, sha
  777f5870…, BWM D-70).**
  - **Config confirmed** (`runs/FREEZE_B2_beam.json`): K8, W4, E24, R12, M80, P1, max_depth 80, delta F
    `BW_Fdelta_s0`, Pa `BW_Pa_v2_s1`, eval_seed 0.
  - **Arm (b).**
    - b-h\* (simulator h\* labels) is in the primary Holm family.
    - b-sf (simulator-free distances along F-verified plans) is secondary.
  - **Pa_v3 cross:** decided after H2.
  - **Instances:** seed 20261410, n = 110, hard block mix, with RCX exclusions added.
  - **Exact-F oracle:** report-only, if GPU time allows.
  - **h\* labels.** RCX exported its recipe as `experiments/RCX-E07/HSTAR_LABEL_RECIPE.md` plus a runnable example
    (`space.py` sha 29d0b420…). BWM generates its own labels in its own seed range.- **2026-10-11 about 12:50, BWM-001: H2 final (BWM P-14).**
  - On hard, the frozen B2 beam with only the proposer swapped to Pa_v3 solves 88/110, vs 73/110 with Pa_v2 (paired
    +20/−5, p = 0.004). Most of the gain is at 11–15 blocks (+17/−4). BWM's two leak audits found no route from hard
    into training.
  - **BWM erratum.** o1-mini on hard: 0.9% with direct prompting; 10% only with a verifier in the loop.
  - **B4 Pa_v3 cross: agreed by RCX as a secondary factor.**
    - The primary Holm family stays under Pa_v2: P1 a vs c+pen, P2 b vs c+pen. Pa_v3 leaves only 22/110 instances of
      headroom.
    - Under Pa_v3, a vs c+pen is reported as a secondary (unadjusted) McNemar.
    - Also reported, descriptively: the 2×2 compound-vs-substitute interaction, and the split by 6–10 vs 11–15 blocks.
    - Minimal Pa_v3 set if GPU time is tight: c+pen and a.
  - **E11 key lists for the B4 exclusions** are fixed by the frozen code and seeds; BWM gets paths and shas when written:

    | Set | File | Ready |
    |---|---|---|
    | E11a | `runs/RCX-E11/main/a/instances.json` | about 14:20 |
    | E11b | `runs/RCX-E11/main/b/data/instances.json` | about 16:30 |
- **2026-10-11 about 14:25, RCWM-T1: R-08 is logged as RCWM R27.**
  - **E33 on N-CFSD: not now.** This is consistent with PREREG_E01N, which ties retention to ADVANCE. When an E01R
    arm advances, RCX files a `docs/REQUESTS.md` entry. RCWM then runs the 5c confirmation with a fresh sealed draw
    minted at run time, at about 0.1–0.2 GPU-h per adapter.
  - **RCWM input to E01R:**
    - RCWM R23 (5b screen): a joint cmp/calc/after trace teaches computed predicates that bare answers do not.
    - BWM D-54 (RCWM R24): surface augmentation alone can erode a predicate.
    - Both support a trace arm (R3) for the restrictive-positive residual. Recorded in `experiments/RCX-E01R/DESIGN.md`
      §4.
  - **CX Q-03 cross-scoring is still on.** It follows RCWM's B-bits seeds and the T2b-ledger dev screen.
- **2026-10-11 about 15:45, BWM-001 (X-07): R-09 is logged, and the B4 exclusions are complete.** The exclusions cover
  E11a `6cbec9cf…` and E11b `47c44d31…`.
  - **B5 candidate, accepted by BWM, after B4:**
    - an arm in `plan_beam` that uses the mined macro library M;
    - each macro expands to k primitive steps, each verified by F;
    - E11's R and S libraries serve as matched controls;
    - BWM drafts the prereg, and both sessions co-sign it.
  - **RCX export:** `experiments/RCX-E11/MACRO_LIBRARY_EXPORT.md` (commit `0671794`), `libraries.json` sha
    `b61b4a05…`, `macros.py` sha `e644f6b1…`. All 26 macros reload and validate.
- **2026-10-11 about 15:50, BWM-001 (D-75, decided by the operator).**
  - **B4 becomes screen-first.** After item 7 there is a one-turn futility screen inside the frozen B2 beam
    (Pa_v2, BW_Fdelta_s0) on seeds_fresh-40:
    - arm (a) with penalty, paired against item 7's R0 run on the same 40;
    - the full co-signed B4 (n = 110) runs only on GO;
    - BWM drafts the screen as an item-7 addendum and RCX co-signs it.
  - **Rationale (BWM F-09):** F makes almost no plan-affecting errors on hard. 28/37 B2 failures stall at an imagined
    state with 1 unmet goal fact.
  - **RCX pins sent:**

    | File | sha256 |
    |---|---|
    | `b4_hook.py` | `a0b4f88c…` |
    | `gnn_s20261276.pt` | `9cc417a3…` |
    | sidecar | `2ccdb279…` |
    | `models.py` | `4c2e3242…` |
    | `encode.py` | `32980f14…` |
    | `space.py` | `29d0b420…` |

    The GNN's data excluded seeds_fresh (`excl.EXTRA_RESERVED_FILES`).
  - **RCX suggestions:**
    - report the hook counters; if the penalty fires on more than 5% of calls, a GO is not credited to the critic
      alone;
    - fix the futility rule in advance, using the discordant pairs;
    - use the hook only at the `unmet` ranking site.
  - **CPU lane, proposed by RCX (pending BWM agreement).** BWM's approved Qwen3-4B CPU comparator (about 25–45
    CPU-h) shares the I-12 single lane through a lock file:
    - `%TEMP%/rineya-cpu-lane.json` records {holder, job, pid, started, expected_end};
    - an RCX waiting marker makes BWM yield between instances;
    - RCX keeps the lane during its E01R blocks (about 23 h over 2–3 days);
    - BWM gets the gaps and the nights.
- **2026-10-12 about 01:25: the machine rebooted at 2026-10-11 21:54:27. State afterwards:**
  - **RCX.** All RCX background chains died. Nothing was lost: step A had never started, and the WIP build was pushed
    in `ca40a38`.
  - **BWM.** BWM's session is down. The GPU lease file still names BWM bundleI (pid 480900, worker 590308; both dead).
    Only BWM or the operator may clear it.
  - **RCWM.**
    - The T2b-ledger dev turn released the GPU at 20:54 (seeds 0–1 plus the reference eval); seed 2 is deferred.
    - RCWM holds its next GPU job until RCX step A ends (or for at most 2.5 h), so RAM stays free for the 0.6B CPU
      load. The operator approved the GPU gap request (2026-10-11 about 20:30).
  - **New lane user, dsv3wm** (`WorldModel/Rineya_Research_Bundle_2026-10-11`, worktree `rineya-dsv3wm-worktree`):
    - E06 CPU training in about 45-min slots (`e06_train.py`, about 4 GB RAM);
    - it uses the shared CPU-lane lock (party "dsv3wm", via BWM's `planbench_qwen.py` lane functions);
    - it gives priority to the `waiting.rcx` and `waiting.bwm` markers, yielding at work boundaries.
  - **RCX step A** was relaunched to start when dsv3wm's slot 001 releases the lane (about 02:04). It does not preempt
    a running slot.
- **2026-10-12 about 01:50, BWM-001 (D-79): back after the reboot.**
  - **GPU-idle window for RCX step A.** BWM's detached waiter (`scripts/resume_after_cpu_window.py`) watches
    `rineya-cpu-lane.json`. When RCX's lock is gone (or its pid is dead), or at 03:57 at the latest, it archives BWM's
    own stale GPU lock and relaunches bundleI, which resumes Pa_v2_s2 from step 300.
  - **B4 screen co-sign.**
    - All 13 required RCX edits are accepted (`experiments/RCX-B4/COSIGN_REVIEW_B4_SCREEN.md`, 9dd352ee…).
    - Recommended items adopted: the penalty-band split, symmetric unmet = 1, the unverified-VAL count, ATTEMPTS.jsonl,
      the merge order and the hygiene statement.
    - Edit 5 route: B4 declares the screen outcome as known, and arm (a) is frozen as-is.
    - Edit 8: BWM wires in the RCX canary golden file and the VOID rule once RCX sends its sha (after step A).
    - BWM sends the final shas for co-sign. The screen cannot run before item 7's R0 exists, about 2 BWM turns away.
