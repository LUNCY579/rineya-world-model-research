# PREREG RCX-E05 (DRAFT): goal-conditioned endpoint-and-length successor features with GPI re-rank Blocksworld moves zero-shot for held-out goal weights

**Template:** bundle `06_templates/PREREGISTRATION.md`.
**Design:** `experiments/RCX-E05/DESIGN.md` rev 3 (rev 2 was sha256 `8e5e91bcff89…`; rev 3's sha is pinned at T1). It
resolves critique items F1–F9 of `CRITIQUE_RVM_DESIGNS.md` (E05 section), records F10–F12, and resolves review items
B1–B4, M1–M6 and L1–L10 of `REVIEW_PREREG_E05_DRAFT.md`. Where this file is more specific than DESIGN, this file binds.
**Track:** RVM-001 (RCI-001 registry E05 "Goal-factorized value transfer"; RCX roadmap P0-c). An RCX free lane
(COLLABORATION.md §2); no peer co-sign is needed. Adjacent to dsv3wm RDW-001 E07/E08 (boundary note, COLLABORATION.md
2026-10-12 about 12:55).

**Status:** **DRAFT, not frozen.** Written 2026-10-12 after CX D-43; revised the same day after the adversarial review
(workflow wf_317f243c-6eb, draft sha256 `ddaa23b8…`) and CX D-44. No E05 code exists, nothing has run, and no E05
output (pilot or test) has been seen.
- It becomes `PREREG_E05.md` at T3 (§11): this text plus the **PILOT FILL** (§9.2) and nothing else. Its sha256 and
  the full code digest are then recorded in a DECISIONS CX D-entry.
- Fixed now and never changed at the fill: every threshold, the decision rule and its order, the sizing formula, the
  cut order, the seeds and the arm definitions.

**What will be known at the freeze** (DEV only; selection seed 20261245; written into the PILOT FILL):
- DEV-ID and DEV-OOD nreg for every pilot arm (SF-GPI, SF-noGPI, SF-L, UVFA, UVFA+aux, the PGV_400 and FT-UVFA_400
  grids, ENUM-MIS±, GREEDY, STOP, RAND, ORACLE, EXACT-GPI(L)), hence the DEV values of D1, D2 and D_G; the selected
  PG400 also on the 24 DEV-OOD H-K/H-NEG parents (M5);
- σ̂_ID, σ̂_OOD, σ̂₂,ID and σ̂₂,OOD, the gate values of G-P1 (incl. Q9) to G-P5, and the measured throughput.

The DEV contrasts are visible to whoever writes the fill. That is why the fill is mechanical (`pilot-report` computes
every value) and why sizing uses DEV **variance** only. No TEST parent is drawn and no confirmation replicate is trained
before the freeze.

**Changes from DESIGN rev 1** (the critiqued text, sha256 `94d315246ac7…`). Full table in DESIGN "Changes from rev 1".

| # | Resolution in this prereg |
|---|---|
| F1 | Seeds 20261242–50 (CX D-43; E04 uses 20261206–14). 20261245 is selection only; replicates are 20261246–48 (App. A) |
| F2 | P1 rule: Holm p < 0.05, CI lower bound > 0 **and** point ≥ 0.03. Power stated for that rule at the planned 0.05; resizing rule fixed now (§8) |
| F3 | P1 is an intersection–union test: the macro **and** H-ID alone must pass. Per-family results always reported (§6.2) |
| F4 | Stabilizer augmentation for per-goal arms; FT-UVFA_400 added (step 0 among its early-stopping candidates); competence bar DEV-ID nreg ≤ 0.50, fixed now; P2's comparator PG400 fixed at the pilot; TEST validity conditions (§5.3, §7.1) |
| F5 | ENUM± handles negative weights (complete falsifier menus, persistence rule, holding-aware admissible ĥ); STOP-D branch (§5.2, §6.3, §7) |
| F6 | Batched multi-model PGV/FT measured at the pilot (G-P5). Protected: PGV_400 on all primary parents, PGV_1600/6400 on a fixed 48-parent subset (§10.3) |
| F7 | dsv3wm RDW E07/E08 cited; boundary note sent; no dsv3wm data or code used |
| F8 | One parent per (signed) support class per TEST family; H-ID split into seen (H-ID-S) and unseen (H-ID-U) supports; only H-ID-U carries "new preference directions" (§4, §7) |
| F9 | G-P4 gates on value error < c/2 and on one-step top-1, each relative to the copy-state floor (whose acc½ is exactly 0 on the DEV-ID bank, disclosed) (§9.1) |
| F10 | Learnability readout (SF-L, SF-noGPI vs exact references) and GPI optimism bias are named secondaries |
| F11 | Naming: "goal-conditioned endpoint-and-length successor features (exact Monte Carlo targets, no TD)", the one string used everywhere |
| F12 | Mechanism and instrument study; G1/STOP-D and k\* are the decision-driving deliverables |

**Changes from the adversarial review** (`REVIEW_PREREG_E05_DRAFT.md`; all text-only, before any code):

| # | Resolution in this prereg |
|---|---|
| B1 | ENUM-MIS± uses ĥ = max(`misplaced_compact`(s′, G∖H), h_hold(s′, G)), H = the holding facts of G; admissible, no all-tie. Rev 2's bound was 0 for every goal with a `holding` fact (`space.py` line 208, `g.has_holding`). New ENUM tests incl. a hand case and a non-tie assertion; pilot check Q9 vs GREEDY on DEV-OOD H-NEG (§5.2, §9.1) |
| B2 | Q7 = ORACLE exactly 0 and STOP exactly 1 at every start, RAND < 1 on the primary macro; GREEDY dropped from the gate and disclosed as ≡ STOP on positive families from hand-empty starts (checked as an identity). Competence bar DEV-ID nreg ≤ 0.50. FT early stopping includes step 0 (§3, §5.3, App. B of DESIGN) |
| B3 | G-P4b terminal branch (P1 comparator-invalid → ADVANCE unavailable, P1 descriptive); TEST validity conditions V-P1, V-P2 before the rule; SF-GPI floor ≤ 0.50 in every primary family for ADVANCE; cannot-ADVANCE analyze tests (§5.3, §7.1) |
| B4 | Every VOID check runs in `oracle-test` before any learned-arm TEST computation; VOID may be declared only there. Later defects → INCONCLUSIVE (defect) + design review, no redraw, full reporting (§7.2, §10.4, §11) |
| M1 | `rcx/lane.py` and `rcx/guard.py` are not pinned (shared operational infrastructure; shas recorded for reference). Windows read from `quiet_windows.json` (+ an extras file) at every phase start; GPU-lease and E01R-night rules; D-43's RAM < 11 GB rule retired (§10.2) |
| M2 | Safe points ≤ 2 min apart from the G-P5 rate, inside batches; tested (§10.2) |
| M3 | Bootstrap streams `random.Random(derive(20261250, "boot:<stratum>"))` in `rcx.stats` draw order; exact equivalence test (§6.4) |
| M4 | One pre-specified k-curve escalation; if it fails too, k\* is descriptive only (§5.3, §9.1) |
| M5 | Selected PG400 also run on the 24 DEV-OOD H-K/H-NEG parents; σ_plan,2 from max(σ̂₂,ID, σ̂₂,OOD) (§8) |
| M6 | Parent key = primitive integer vector of 8·w̄ (scale equivalence), canonical under renaming; disjointness asserted on it (§4.1) |
| L1–L10 | Complete falsifier menus (L1); low-effect power stated (L2); conditional-on-replicates inference and condition 4 on H-ID (L3); STOP-F/STOP-M gaps closed (L4); committed-sha refusal (L5); exclusions on label rows and banks, fail-closed handling (L6); pilot report published if abandoned, n = 6 wording (L7); consistency fixes (L8); RAM floor and BWM confirmation (L9); open items updated (L10) |

**Execution scope:**
- CPU only, ≤ 4 torch threads, < 1.5 GB RSS per process. No language model, no GPU, no 0.6B load.
- No peer data. Every label comes from the exact oracle at run time.
- Tier: instrument plus mechanism (an external specialist, tier B). **Nothing here is a Rineya capability claim.**
- No model is published, promoted or handed to a planner under this ID.

## 1. Claim and scope

**H-E05.** On the exact Blocksworld state space for n = 7 (65,990 states, BWM 4-op semantics, fixed dynamics; **n = 6
instead if the G-P4 fallback is taken**, fixed in the PILOT FILL and then substituted everywhere in this claim), with
preferences linear in 56 observable fact features (42 at n = 6), a goal-conditioned successor-feature critic ψθ(s, z),
trained once on exact labels for 64 training goal weights and composed by generalized policy improvement (GPI),
achieves on held-out goal weights:
- **(P1)** lower normalized closed-loop regret than a matched-capacity scalar goal-conditioned value (UVFA) trained on
  the identical (state, weight) labels, **both** on the macro over H-ID, H-K and H-NEG **and** on H-ID alone, provided
  the UVFA is a valid comparator (competence bar, §5.3, §7.1);
- **(P2)** regret non-inferior, within δ_NI = 0.05, to the better competent per-goal scalar critic given 400 fresh exact
  labels of each held-out goal (PG400: PGV_400 from scratch or FT-UVFA_400, fixed at the pilot; competent means
  nreg ≤ 0.50 on DEV-ID and on the TEST primary macro).

The SF critic uses **zero** labels per new goal.

**Effect sizes (fixed).** Planned effect for P1: 0.05 nreg. Minimum point estimate for a pass: 0.03. NI margin for P2:
0.05. Dominance margin for STOP-D: 0.05. Competence bar for comparators and floor for SF-GPI under ADVANCE: 0.50.
- **Stated plainly (L2).** The planned effect 0.05 lies above the predicted H-ID gap (+0.00 to +0.04, §7). P1-ID power
  at a true 0.03 is 0.22–0.35 at σ = 0.20 (n_ID 96–152) and never above 0.5; at a true 0.04 it is 0.39–0.59 (§8).
  P1-ID is sized to detect 0.05 and will usually miss an effect at the low end of the prediction.

**Naming (F11).** In this deterministic, undiscounted episodic MDP, ψ^π(s) is the (endpoint facts, length) of a
deterministic policy, learned by supervised regression on exact rollout targets with no TD learning. Every claim says
"goal-conditioned endpoint-and-length successor features (exact Monte Carlo targets, no TD)".

**Framing (F12).** No consumer exists today (BWM goals are hard `on` facts). E05 is a mechanism and instrument study. Its
decision-driving deliverables are G1/STOP-D (does a zero-label symbolic planner already serve soft goals here?) and k\*
(the per-goal label budget at which a scalar critic matches zero-shot SF). P1 and P2 are the scientific tests.

**A positive result (ADVANCE) would establish**, at tier B with exact dynamics and n = 7 only (n = 6 only under the
G-P4 fallback): zero-shot transfer of a learned endpoint-and-length SF critic with GPI to unseen preference weights
inside the feature span, with SF-GPI regret ≤ 0.50 in every primary family; superiority over a competent scalar
goal-conditioned critic at equal labels, including on in-distribution weights; non-inferiority to the better competent
per-goal critic with 400 labels per goal; no domination by the zero-label ENUM-MIS± planner; and a measured k\*.
"New preference directions" is allowed only under the H-ID-U qualifier (§7); otherwise the claim is "re-weighting of
seen fact patterns".

**It would not establish:** any Rineya native ability; transfer across n or domains; robustness to LM-imagined states or
LM-proposed candidates; value for hard conjunctive goals (H-CONJ) or under changed dynamics (H-DYN); any gain for BWM's
beam or an LM planner; the bundle's "+5 pp goal success at matched compute" target.

## 2. Custody

### 2.1 Code

**Existing files, pinned (8-hex prefixes here; full sha256 values are re-verified and recorded at T2 and T3):**

| File | sha256 | Use |
|---|---|---|
| `rcx/bw/space.py` | `29d0b420…` | state space, CSR graphs, `CompiledGoal`, `goal_mask`, `backward_bfs`, `misplaced_compact` |
| `rcx/bw/bwm_import.py` | `84140c7e…` | read-only BWM `blocks.py` access for exclusions |
| `rcx/bw/excl.py` + `experiments/RCX-E07/EXCLUSIONS.json` | `eb31d9eb…` + `2f211de3…` | leakage control (re-pinned if BWM's B4 fresh set is added, §4.4) |
| `rcx/bw/models.py`, `rcx/bw/encode.py` | `4c2e3242…`, `32980f14…` | R-04 RelGNN for GNN-ENUM± (inference only); CUDA-off patch |
| `runs/RCX-E07/main/models/gnn_s20261276.pt` | `9cc417a3…` | frozen R-04 GNN |
| `rcx/stats.py` | `06a30c10…` | `bootstrap_p_two_sided`, `holm`, `mcnemar_exact`, `_interval`, `seed_parent_bootstrap_test` (reference for the unit test) |

**Operational files, not pinned (M1).** `rcx/lane.py` (`4a1bc447…` after CX D-44; rev 2's `03f93d59…` is superseded) and
`rcx/guard.py` (`08fa782d…`) are shared infrastructure that other parties' decisions change (lane.py is "not bound
code" per D-41/D-44). They decide only *when* a phase runs, never *what* it computes (resumption equivalence is
tested, §2.1 tests). They are therefore outside the code digest. Their shas, and that of E05's own
`lane_e05.py`, are recorded for reference in every phase's provenance record.

**New files (to be written; none exists at this draft):**

| File | Content |
|---|---|
| `rcx/bw/soft.py` | `phi_table`; `soft_values` (max-plus value iteration, optional edge mask); `policy_table` (greedy with the tie rule; endpoints and lengths by pointer doubling); `sf_table`; `gpi_scores`; `height_limit_mask(space, h=3)`; `conj_values` |
| `experiments/RCX-E05/families_e05.py` | family generators; `canon_key`; `support_key(w, signed)`; `stabilizer(w)`; renaming of states and weights; seen/unseen quotas; one-parent-per-support assertion; exclusion adaptor; `derive(seed, tag)` |
| `experiments/RCX-E05/enum_e05.py` | ENUM± target enumeration (complete falsifier menus, satisfiability, persistence rule); ENUM-MIS± with the holding-aware ĥ (§5.2), ENUM-H\*±, GNN-ENUM±; all-tied-decision diagnostics; GREEDY, RAND, STOP, EXACT-GPI(L), BEST-SINGLE(L), ORACLE |
| `experiments/RCX-E05/learners_e05.py` | SF, UVFA, UVFA+aux; `BatchedMLP` for PGV and FT; renaming and stabilizer augmentation; trainer with checkpoints (epoch ends, batch ends, mid-batch safe points); early stopping (FT candidates include step 0); library ψ̂ cache |
| `experiments/RCX-E05/run_e05.py` | subcommands `code-digest`, `pilot-a`, `pilot-b`, `pilot-report`, `draw-test`, `oracle-test`, `train`, `pgv`, `ft`, `eval`, `extras`; calls the lane gate of `lane_e05.py`; keyed append-only outputs; `ATTEMPTS.jsonl`; provenance manifest; the committed-sha refusal (§2.2) |
| `experiments/RCX-E05/lane_e05.py` | **operational, outside the code digest (M1):** start condition, safe-point yield checks and the readers of the windows files, the GPU lease, the E01R chain logs and the quiet-release file (§10.2). Imports nothing from the outcome code; a change is an operational deviation, logged, and needs `test_e05_lane.py` to pass |
| `experiments/RCX-E05/analyze_e05.py` | stratified paired cluster bootstrap; IUT P1; NI P2; Holm; the §7.1 validity determinations; the §7.2 rule and qualifiers; tables; k\*; create-only report |

**Tests (each < 5 s on n = 3–5 spaces, literal seeds 0–9, never reported as results):**

| Test file | Must show |
|---|---|
| `tests/test_e05_oracle.py` | Q1 value iteration = brute force (n = 5, 200 random w incl. negatives and `clear`, c ∈ {0.125, 0.25, 0.75}); Q2 H-CONJ vs `GoalLabels` at n = 5; Q3 rollouts = tables and V\* = w̄·ψ exactly; Q4 GPI ≥ every library policy; Q5 renaming invariance and the conjugate-policy ψ identity; Q6 height-limit mask = direct rule check; Q7 identities: ORACLE nreg = 0 and STOP nreg = 1 exactly, and GREEDY nreg = 1 exactly at every hand-empty start of a positive on/ontable goal (n = 5) |
| `tests/test_e05_families.py` | canonical and support keys invariant under random σ (9 two-fact on/ontable support classes); **scale equivalence (M6):** (w, c) and (λw, λc) share one parent key (all-1 at c = 0.125 ≡ all-2 at 0.25; all-3 at 0.75 ≡ all-1 at 0.25), w and 2w at the same c do not; signed keys; stabilizer correctness and \|Stab(w)\| ≥ (n − m)! for a goal mentioning m blocks; split and support disjointness on the primitive key; one parent per support; quotas; exclusion adaptor on synthetic signatures (start, bank and label rows); derived-stream determinism |
| `tests/test_e05_enum.py` | **menus (L1):** every menu fact implies ¬f, and on random n = 5 states f is false iff some fact of M(f) holds; satisfiability pruning; a hand case where a penalty must be avoided (ENUM± avoids it, positive-only ENUM does not); `holding` targets; **admissibility:** ĥ_MIS(s, G) ≤ h\*(s, G) (`backward_bfs`) on 500 random (s, G) at n = 5 incl. holding targets; **B1 hand case** (§5.2) run through ENUM-MIS± itself: it picks unstack(a, b) at s0 and then STOPs (nreg 0), its successor scores at s0 are not all tied, and two test-local negative controls tie (rev 2's ĥ ties all three successors; `misplaced_compact` on G∖H without h_hold ties unstack(a, b) with pick-up(c), which the tie rule then wrongly picks); **non-tie assertion:** over 20 random H-NEG weights at n = 5 × 20 random hand-empty starts with ≥ 2 moves, the share of (w, s) with all ENUM-MIS± move scores equal is ≤ 0.5 and below the rev-2 control's share, and every weight has a start with ≥ 2 distinct scores; ENUM-H\*± = ORACLE on single-target goals; positive-only goals reduce to rev 1's ENUM |
| `tests/test_e05_learners.py` | `BatchedMLP` = B independent models (same init and data order, 50 steps, max abs diff < 1e-5); per-model early stopping; **FT candidates include step 0** (an FT run whose validation loss never improves returns the zero-shot UVFA weights exactly); augmentations preserve labels exactly; checkpoint = independent shorter run; **mid-batch checkpoint and resume = the uninterrupted run** (max abs diff ≤ 1e-6 at fixed threads); SF-shuffled permutes within weight only; CUDA-off patch |
| `tests/test_e05_analyze.py` | stratified bootstrap = `rcx.stats.seed_parent_bootstrap_test` **exactly** (diff, lo, hi, p) on one stratum with seed `derive(20261250, "boot:<stratum>")`, R = 10,000, `p_floor` = 1/R, ties 'both' (M3); IUT p = max; NI p; Holm with P2 dropped and with P1 comparator-invalid; validity conditions never change the Holm family; every §7.2 branch on synthetic inputs (INCONCLUSIVE defect and missing, STOP-D, ADVANCE, STOP-M only when neither test counts, STOP-F incl. a significant-but-small P1 and a dropped or TEST-invalid P2, REVISE incl. P1-M-only, a failed SF floor and a library ceiling ≥ 0.5, INCONCLUSIVE); **cannot-ADVANCE cases (B3):** copy-state SF (SF-GPI values = GREEDY's: exactly 1 on H-ID and H-K), constant UVFA (UVFA H-ID nreg > 0.50, else ideal inputs), PG400 > 0.50 on the TEST macro, P1 comparator-invalid at the pilot; condition 4 on the macro and on H-ID; qualifiers; tolerance; refusal on a non-PASS oracle-test report, a manifest or digest mismatch, or a manifest sha absent from the committed `DECISIONS.md` |
| `tests/test_e05_lane.py` | temporary `RINEYA_LANE_DIR` and `%TEMP%`, injected clock, RAM, windows files, GPU lease file, E01R logs and pid liveness: refuses with any `waiting.*` marker, with the lane held, with an unreadable windows or lease file, with free RAM < 4.0 GB or unknown, in day mode with a live GPU holder (pid or any worker pid) or a projected end later than 10 min before the next window opens, inside a window unless E01R has no remaining block there (release file, or E01R's newest `lane:` line is the "would end too late" refusal for this window's end) and the phase ends ≥ 5 min before the window closes; never writes a waiting marker, a GPU file or the release file; at safe points yields on a marker, a new live GPU holder, the 10-min pre-window mark, 15 min elapsed or a guard pause; **safe-point spacing (M2):** with injected step and cell costs for every (model size, B, batch) and evaluation unit in the plan, consecutive checks are ≤ 90 s apart at the planning rate and ≤ 120 s at a 2× slower rate, and a planned unit projected > 30 s is rejected; a yield mid-batch writes a checkpoint that resumes exactly (with `test_e05_learners`) |

**Code digest.** `run_e05.py code-digest` = sha256 over the sorted lines `"<relpath>\t<sha256>\n"` of the new source
files except `lane_e05.py`, and the pinned existing files above (operational files excluded, M1). The T2 (pilot) digest
covers every new outcome source file except `analyze_e05.py`; the T3 digest covers all of them. Test-file shas are
recorded beside each digest, with the pass log.

### 2.2 Data and manifests

- **No dataset file is consumed.** All data come from the exact oracle and the seeds in App. A.
- `runs/RCX-E05/pilot/PILOT_MANIFEST.json` (written by `pilot-a` under the T2 digest): TRAIN and DEV parents, label
  indices, starts, banks, rejection counts. Its sha256 goes into the PILOT FILL.
- `runs/RCX-E05/confirm/attempt-<k>/TEST_MANIFEST.json` (written by `draw-test` after T3; refuses unless
  `--expect-prereg-sha` and `--expect-code-digest` match): TEST parents per family and stratum, starts, one-step banks,
  the 48-parent subset and the per-goal label index sets. Its sha256 is committed to DECISIONS **before any arm is
  evaluated on TEST**. The oracle runs used to draw TEST are the only TEST computation before that commit.
- **Committed-sha refusal (L5).** `oracle-test`, `pgv`, `ft`, `eval`, `extras` and `analyze_e05.py` refuse unless the
  full 64-hex `TEST_MANIFEST.json` sha256 occurs in the **committed** `DECISIONS.md` (`git show HEAD:<repo
  path>/DECISIONS.md`), not merely in the working copy.
- `runs/RCX-E05/confirm/attempt-<k>/ORACLE_TEST_REPORT.json` (create-only, written by `oracle-test`): every VOID check
  of §10.4 with its value and PASS or VOID. `pgv`, `ft`, `eval`, `extras` and `analyze_e05.py` refuse unless it exists
  for the attempt and says PASS (B4).
- **Never opened:** RCWM sealed instruments; PlanBench Logistics; BWM `distill_rows.jsonl` and `mix_*.jsonl`; any
  dsv3wm data or code (RDW E07/E08 included); factory data; any `*.secret` file.

### 2.3 Models

- **Selection models** (seed 20261245) choose configs on DEV and are **never evaluated on TEST**. Their shas are logged.
- **Confirmation replicates** 20261246–48 of SF, UVFA and UVFA+aux are trained after T3 at the PILOT-FILL configs, once
  each. There is no selection among replicates. A crashed run resumes from its checkpoint under the same fingerprint
  (seed, config, label-set sha, code digest).
- **Per-goal fits:** one fit per (parent, arm, k); FT-UVFA_400 one fit per (parent, replicate). No refit, no per-goal
  tuning.
- **R-04 GNN:** frozen, inference only.

## 3. Task and instrument (exact)

- **States and moves.** n = 7: 65,990 states (37,633 hand-empty, 28,357 holding), 186,578 move edges
  (`space.successors`, verified equal to BWM `B.step`).
- **Features.** φ(s) ∈ {0,1}^56: on(i,j) (42), ontable(i) (7), clear(i) (7). A held block has none of these facts.
- **Goal.** w ∈ ℤ^56 (small integers) and step cost c; w̄ = (w, −c). Training goals use c = 0.25. Values are exact
  dyadic float64, so ties are exact.
- **Reward.** Each move costs c. STOP is always legal and collects w·φ(s). V\*_w(s) = max(w·φ(s), max over legal s′ of
  [−c + V\*_w(s′)]), by max-plus value iteration over the forward CSR graph from the STOP values.
- **Policies.** π_w is greedy on V\*_w; ties go to STOP first, then the lowest action code. ψ^π(s) = (φ(endpoint),
  length); for a move to s′, ψ^π(s, a) = (0, 1) + ψ^π(s′); V\*_w = w̄·ψ^{π_w}.
- **GPI.** For test weight w′ and code set C: Q(s, a) = max over z ∈ C of w̄′·[(0, 1) + ψ̂(s′, z)]; Q(s, STOP) =
  w′·φ(s).
- **Decision protocol (every arm).** Candidates = STOP plus the exact legal moves. Every arm scores STOP exactly as
  w′·φ(s) and receives w′. Argmax with the tie rule above (exact float comparison, no tolerance). Closed loop from s0
  until STOP, 40 moves (forced STOP), or a revisited state (forced STOP at the current state, logged as a loop).
- **Starts.** Hand-empty states with gap V\*_{w′}(s0) − w′·φ(s0) ≥ 1.0, not excluded (§4.4), uniform without
  replacement, 16 per parent. A parent with fewer than 16 is rejected and logged. H-DYN starts also satisfy the height
  limit.
- **GREEDY ≡ STOP on positive families (disclosed, B2).** From a hand-empty state every move (pick-up, unstack) only
  removes on/ontable facts and can add only `clear` facts. For a goal with non-negative weights on on/ontable facts and
  none on `clear` (DEV-ID, H-ID, H-K, H-COST, H-DYN, and H-CONJ's linearization), every move scores w′·φ(s′) − c′ <
  w′·φ(s0), so GREEDY stops at s0 and its nreg is **exactly 1** at every start. GREEDY is informative only on H-NEG,
  H-KN and H-SPAN and on holding states of the one-step bank. It is therefore not a floor in any gate or competence bar;
  the identity itself is checked (Q7).
- **Normalized regret.** J = w′·φ(s_final) − c′·T; nreg = clip((V\* − J) / (V\* − w′·φ(s0)), 0, 1). 0 is optimal; 1 is
  no better than stopping at once.
- **One-step bank.** 64 states per parent, uniform over states where STOP is not optimal; identical for every arm.
  regret1 = (max Q\* − Q\*(chosen)) / c′, and the top-1 optimal rate.
- **Instrument checks** (DESIGN App. B): Q1–Q6 as unit tests and at the pilot; Q7–Q9 at the pilot; Q3, Q4, Q7 and Q8
  again on TEST inside `oracle-test`, before any learned-arm TEST computation (§10.4). **Q7 (B2):** ORACLE nreg = 0
  and STOP nreg = 1 exactly at every start of every family; RAND's primary macro nreg < 1 (DEV: over DEV-ID, DEV-OOD
  H-K and DEV-OOD H-NEG; TEST: over H-ID, H-K and H-NEG); GREEDY nreg = 1 exactly at every start of the positive
  families listed above (an identity; a failure is an instrument bug). GREEDY is no longer required to lie strictly
  between the floors.

## 4. Goal families, sizes and splits

### 4.1 Keys

- **Parent key (M6):** (v, reward type, dynamics), where v is the **primitive integer vector of 8·w̄** = (8w, −8c)
  divided by the gcd of its entries (c ∈ {0.125, 0.25, 0.75}, so 8·w̄ is integral), canonical under block renaming
  (minimum over the 5,040 permutations). (w, c) and (λw, λc) are the same task (identical policies and nreg), so they
  get one key: e.g. all-1 weights at c = 0.125 ≡ all-2 at c = 0.25, and all-3 at c = 0.75 ≡ all-1 at c = 0.25. For
  H-CONJ, v is the canonical G with the primitive vector of (8R, −8c). Dynamics is "free" or "height-3" (H-DYN).
- **Support class:** the canonical unweighted support of w; signed (fact, sign) for families with negative weights; the
  class of G for H-CONJ. **Seen** means equal to the support class of some library goal.
- **One parent per support class** in every TEST family (asserted). The parent is therefore the support-class cluster.

### 4.2 Families

| Family | Role | Generator |
|---|---|---|
| TRAIN (library L) | training | 64 goals; K ~ {1: 0.1, 2: 0.3, 3: 0.3, 4: 0.3} facts from on ∪ ontable (P(on) = 0.75); weights iid {1, 2, 3}; c = 0.25; 400 uniform labelled states each |
| DEV-ID | tuning, gates, σ̂ | TRAIN-like, K ∈ {2, 3, 4}; 16 seen-support + 16 unseen-support parents |
| DEV-OOD | σ̂_OOD, pipeline check | 12 parents each of H-K, H-NEG, H-COST, H-SPAN |
| **H-ID** (primary) | goal split | K uniform {2, 3, 4}, TRAIN-like. **H-ID-S:** support ∈ supports(L), min(40, available) parents. **H-ID-U:** support ∉ supports(L) ∪ supports(DEV-ID), the remaining n_ID − n_S |
| **H-K** (primary) | composition | K ∈ {5, 6}, weights {1, 2, 3}, c = 0.25 |
| **H-NEG** (primary) | sign | K ∈ {2, 3, 4}; one negative fact at K = 2, one or two at K ≥ 3; negatives {−1, −2}; ≥ 1 positive; on ∪ ontable |
| H-COST | numerical | TRAIN-like (K 2–4), c ∈ {0.125, 0.75}; 40 parents |
| H-SPAN | outside span | TRAIN-like (K 1–3) plus 1–2 `clear` facts weighted {1, 2}; 40 |
| H-KN | combined | K ∈ {5, 6} with 1–2 negatives; 24 |
| H-CONJ | assumption boundary | 2–3 satisfiable `on` facts G, reward 3·1[G ⊆ s], c = 0.25; SF and UVFA get w = (3/\|G\|)·1_G; 40 |
| H-DYN | assumption boundary | TRAIN-like weights under height-limit-3 dynamics; learned arms trained on unrestricted dynamics; 40 |

Only H-ID, H-K and H-NEG are primary. Secondary families are reported per family and never pooled with the primary.

### 4.3 Sizes

| Set | Parents | Per parent |
|---|---|---|
| TRAIN | 64 (+ 49 single-fact base policies only if G-P2 triggers, then for every learned arm) | 400 labels (800 if the G-P4 fallback is taken) |
| DEV-ID | 32 | 16 starts, 64 bank states, 400 per-goal labels; 1,600/6,400 on 8 parents |
| DEV-OOD | 48 | 16 starts; 400 per-goal labels on the 24 H-K/H-NEG parents (selected PG400 only, M5) |
| **TEST primary** | n_ID ∈ [96, 152], n_K = n_NEG ∈ {56, 72}, set by §8 | 16 starts, 64 bank states, 400 per-goal labels |
| 48-parent subset | 16 per primary family (8 per H-ID stratum), from `20261242:pgv-subset` | nested 400 ⊂ 1,600 ⊂ 6,400 labels |
| TEST secondary | 184 | 16 starts, 64 bank states |

### 4.4 Leakage controls (asserted at generation; counts logged)

1. Weighted classes (the M6 parent key, scale-canonical) are disjoint across TRAIN, DEV-ID, DEV-OOD and every TEST
   family. Support classes are unique within each TEST family. H-ID-U supports are disjoint from supports(L) ∪
   supports(DEV-ID).
2. States are shared by design (n = 7 has 26 state orbits). The unit of novelty is the (state, weight) pair.
3. Every TRAIN label row, every per-goal label row (DEV and TEST), every one-step bank state and every DEV/TEST start
   is read as a BWM instance (state, positive on/ontable facts of the parent) and checked with
   `excl.row_exclusion_reason` on `with_selection_sets(load_exclusions())` (L6); H-CONJ (start, G) pairs with
   `excl.exclusion_reason`. Matches are removed from each parent's sampling pool before drawing (so counts stay fixed)
   and counted. If BWM's B4 fresh set (seed 20261410, n = 110; a different file from `instances_seeds_fresh.json`,
   which `EXTRA_RESERVED_FILES` already holds) exists by T3, it is added through `excl.EXTRA_RESERVED_FILES` and the
   shas are re-pinned in the PILOT FILL.
   - **Fail-closed after T3 (L6).** `load_exclusions` raises if a BWM source changed. Before `draw-test`: the phase
     refuses; the pinned inputs are restored, or, if BWM really changed them, re-pinned by a logged deviation ("outcomes
     known: no") in a CX D-entry before TEST is drawn. Between `draw-test` and `oracle-test`'s PASS: `oracle-test`
     records VOID (exclusion basis changed) and the attempt is redrawn after the re-pin. After the first learned-arm
     TEST computation, exclusions are never reloaded (every TEST check is complete and recorded by then); a later
     source change is reported descriptively (post hoc collision count against the new sources), not a defect.
   - A reserved set that appears after T3 is never added to this attempt; its collisions with TEST are counted and
     reported descriptively.
4. No RCX evaluation set (E07, E07b, E11) contains an n = 7 instance (asserted).
5. The R-04 GNN was trained on hard on-only goals at n = 4–7; `ontable`, `clear` and `holding` targets are out of
   distribution for it (disclosed).
6. **Rejection log:** duplicate class, duplicate support, exclusion, too few starts, unsatisfiable, quota.

## 5. Arms and fairness

### 5.1 Inputs and architectures

- State x_s (70 dims): on-matrix (49), ontable (7), holding (7), clear (7). Goal/policy code z (57 dims): weights / 3,
  plus c.
- Trunk 3 × 256, LayerNorm, ReLU; AdamW (lr 1e-3, wd 1e-4, constant); batch 256.
- Parameters: SF 180,537; UVFA 166,145; UVFA+aux 180,794; PGV width 256 151,553; width 64 13,313.

### 5.2 Arms

| Arm | Definition | New labels per goal | Role |
|---|---|---|---|
| **SF-GPI** | ψθ(x_s, z) → 56 sigmoid endpoint facts + softplus T̂; BCE + Huber(T/10); GPI over C = L ∪ {w′} | 0 | treatment |
| **UVFA** | V_θ(x_s, w) → V\*; Huber; identical labels, trunk and schedule | 0 | **P1 comparator** |
| UVFA+aux | shared trunk; scalar head used at test; ψ head auxiliary (λ = 1) | 0 | attribution |
| SF-noGPI / SF-L / SF-GPI-5 | C = {w′} / C = L / {w′} + 4 nearest library codes (cosine) | 0 | role of GPI and of the w′ index; compute-matched view |
| SF-shuffled | SF with ψ targets permuted across states within each weight (replicate 20261246) | 0 | shuffled-label control (descriptive) |
| **PGV_400** | per-goal V_θ(x_s) from scratch on the parent's 400 labels; stabilizer augmentation; early stopping on 40 of the 400 (every 100 steps, best kept) | 400 | P2 candidate |
| **FT-UVFA_400** | each UVFA replicate fine-tuned on the same 400 labels, goal input fixed to w̄′, all parameters, AdamW lr 3e-4, batch 64, stabilizer augmentation, same early-stopping split, validation loss checked **at step 0** (the zero-shot UVFA) and every 25 steps, best kept, ties to the earlier step; 3 fits per parent, averaged | 400 | P2 candidate |
| PGV_1600 / PGV_6400 | width 256; batch 128 / 256; 2,000 / 2,500 steps; stabilizer augmentation; early stopping on 10%; not tuned (only the single escalation R_ext of G-P3b, §5.3) | 1,600 / 6,400 | k\* (48-parent subset) |
| SF-FT_400 | SF replicate 20261246 fine-tuned on the ψ targets of the same 400 labels (FT's selected steps) | 400 | descriptive |
| ENUM-MIS± | ENUM± targets scored with the admissible holding-aware bound ĥ_MIS (below; B1) | 0 (symbolic) | **guard G1 and STOP-D** |
| ENUM-H\*± | ENUM± with exact h\* per target (`backward_bfs`) | 0 | oracle-information reference |
| GNN-ENUM± | ENUM± with the frozen R-04 GNN as ĥ | 0 | existing RCX critic, zero-shot |
| GREEDY | one-step lookahead on w′·φ(s′) − c′, plus the STOP rule; the copy-state floor (= SF-GPI with ψ̂(s′, z) = (φ(s′), 0)). ≡ STOP on positive families (§3) | 0 | floor (ranking floor in G-P4 only) |
| STOP / RAND | always stop / uniform over {STOP} ∪ moves (`20261244:rand`) | 0 | floors |
| EXACT-GPI(L) / BEST-SINGLE(L) | GPI over exact library policy tables / best single library policy at s0 | – | mechanism ceiling / reuse reference |
| ORACLE | π\*_{w′} | – | regret 0 |

**ENUM± (F5).** For w′ with positive facts P and negative facts N, targets are G = G⁺ ∪ R with G⁺ ⊆ P and, for each
f ∈ N, "accept" or one falsifier from M(f).
- **Menus (L1): every single fact that implies ¬f.** M(on(a,b)) = {ontable(a), holding(a), clear(b), holding(b)} ∪
  {on(a,x) : x ∉ {a,b}} ∪ {on(x,b) : x ∉ {a,b}} (14 facts at n = 7; rev 2's 3-fact menu missed on(a,x), on(x,b) and
  holding(b)); M(ontable(a)) = {holding(a)} ∪ {on(a,x) : x ≠ a}; M(clear(a)) = {holding(a)} ∪ {on(x,a) : x ≠ a}. Each
  menu covers ¬f: every state where f is false satisfies some menu fact (unit test). Targets per parent before pruning:
  H-NEG ≤ 2² × 15² = 900, H-KN ≤ 2⁴ × 15² = 3,600, H-K ≤ 2⁶ = 64.
- G is kept only if `CompiledGoal(G).consistent` and `goal_mask(G).any()`. U(G; s′) = Σ_{f∈G⁺} w_f + Σ_{f∈N}
  w_f·1[f ∈ φ(s′) and G ∪ {f} satisfiable]; score(s′) = max over G of [U(G; s′) − c′·(1 + ĥ(s′, G))]. Facts false now
  that might become true are ignored (optimistic; disclosed).
- **ĥ for ENUM-MIS± (B1).** With H the `holding` facts of G: ĥ_MIS(s′, G) = max(`misplaced_compact`(s′, G∖H)[1],
  h_hold(s′, G)), where h_hold = 0 if H is empty or s′ holds the block H requires, 1 if the hand is empty, and 2 if s′
  holds another block. Both terms are admissible (a state satisfying G satisfies G∖H; a required holding needs a
  pick-up or unstack, preceded by a put-down or stack if another block is held), so ĥ_MIS ≤ h\*(s′, G). For goals
  without `holding` facts it equals rev 2's bound.
  - **Why (confirmed in `rcx/bw/space.py`).** `misplaced_compact` returns a bound of 0 whenever `g.has_holding`
    (line 208). Under rev 2 the target P ∪ {holding(a)} (satisfiable whenever a is unconstrained by P) scored
    Σw(P) − c′ for every successor, with no penalty charged, beat every other target, and all moves tied: ENUM-MIS± then
    behaved like STOP on H-NEG.
  - **Why h_hold, beyond the review's minimal fix.** On the hand case below, `misplaced_compact`(s′, G∖H) alone still
    ties unstack(a,b) with pick-up(c), and the tie rule (lowest action code, PICK = 0 < UNSTACK = 3) picks the wrong
    move. h_hold breaks that tie without losing admissibility.
  - **Hand case (`test_e05_enum.py`).** n = 4, hand empty, a on b, c and d on the table; w′: on(a,b) = −2,
    ontable(d) = +1; c′ = 0.25. The optimum is unstack(a,b) then STOP while holding a (V\* = 0.75, w′·φ(s0) = −1).
    ENUM-MIS± scores unstack(a,b) 0.75, pick-up(c) 0.25 and pick-up(d) 0.25, so it plays the optimum (nreg 0).
- **Diagnostic.** For every ENUM± arm, the share of decisions with ≥ 2 legal moves whose move scores are all equal is
  logged per family and reported.

### 5.3 Tuning (DEV only, selection seed 20261245) and the P2 comparator

- **Competence bar b = 0.50 nreg (B2; fixed now).** A comparator is competent on a set if its mean nreg there is
  ≤ 0.50.
  - **Why this bar.** Rev 2's "≤ nreg(GREEDY) − 0.20" meant ≤ 0.80 on DEV-ID, because GREEDY ≡ STOP = 1 there (§3):
    nearly floor level. 0.50 is fixed and outcome-free, is met by neither floor (STOP and GREEDY are exactly 1 on
    DEV-ID; RAND is expected near 1), means "recovers at least half of the available gain on average", and is the scale STOP-M
    and G-P2 already use. The alternative "≤ ENUM-MIS± on DEV-ID" was rejected: ENUM-MIS± is the guard under test (B1
    showed how its quality can fail), so tying P2's existence to it would couple the comparator to the guard, and its
    DEV value is unknown now, so the bar could turn out trivially lax or unreachable. BEST-SINGLE(L) was rejected
    because it depends on the library composition (G-P2).
- **Zero-shot arms.** Epochs {20, 40} at lr 1e-3 (20 is a checkpoint of the 40-epoch run); per arm, by DEV-ID macro
  nreg.
- **UVFA validity (G-P4b, B3).** The UVFA passes iff acc½(UVFA) − acc½(copy) ≥ 0.20 **and** its DEV-ID nreg ≤ 0.50.
  If it fails, one extension {80 epochs, lr 3e-4}. **Terminal branch:** if it still fails, **P1 is declared
  comparator-invalid before the freeze** (PILOT FILL): P1 leaves the Holm family and is reported descriptively, and
  ADVANCE is unavailable. The confirmation still runs for P2, G1/STOP-D and k\*.
- **PGV_400 grid:** width {64, 256} × steps {500, 2,000}, batch 64 (500 is a checkpoint). **FT grid:** steps
  {100, 400} (100 is a checkpoint), step 0 always among the early-stopping candidates. Tuned once on DEV-ID; never per
  goal.
- **Competence (G-P3).** A candidate is competent if its best config reaches DEV-ID nreg ≤ 0.50.
- **PG400** = the competent candidate with the lower DEV-ID nreg (ties to FT-UVFA_400). If neither is competent, each
  grid is extended once (PGV: width 128, 4,000 steps, lr 3e-4; FT: 1,000 steps, lr 1e-4). If still neither is
  competent, **P2 is dropped**: P2 leaves the Holm family and ADVANCE is unavailable. If P1 is also comparator-invalid,
  the Holm family is empty and only VOID, INCONCLUSIVE, STOP-D, STOP-M or STOP-F can result (§7.2).
- **M5.** Pilot B then fits the selected PG400 config (selection seed; for FT, the selection UVFA) on the 24 DEV-OOD
  H-K and H-NEG parents with their 400 per-goal labels, for σ̂₂,OOD (§8). Not run if P2 is dropped.
- **TEST validity (B3, §7.1).** At TEST, P2 counts only if PG400's primary macro nreg ≤ 0.50, and P1 counts only if
  UVFA's H-ID nreg ≤ 0.50.
- **k-curve sanity and its single escalation (G-P3b, M4).** On the 8 DEV-ID parents, PGV_6400 mean nreg ≤ 0.25 and
  the mean nreg is non-increasing over k = 400, 1,600, 6,400 (1e-9 tolerance). If not, one pre-specified escalation
  R_ext is run on the same 8 parents and applied identically to all three k of the curve: width 128, lr 3e-4,
  4,000 steps, batch 64 / 128 / 256 for k = 400 / 1,600 / 6,400, stabilizer augmentation, early stopping on 10% every
  200 steps. If R_ext passes, the TEST k-curve (48-parent subset) uses R_ext for all three k, and PGV_400 on all
  primary parents keeps its G-P3 config for P2. If R_ext also fails, the original recipe is kept and **k\* is
  descriptive only** (reported, flagged, not decision-driving). Nothing else about per-goal training may change after
  DEV is seen.
- PGV_400 runs on every primary parent in every case; the non-selected candidate is reported descriptively.

### 5.4 Fairness constraints

- SF, UVFA and UVFA+aux see the identical (state, weight) label set, trunk, schedule, selection budget and seeds. SF's
  richer supervision (a 57-vector per label) is disclosed and controlled by UVFA+aux.
- PGV_400, FT-UVFA_400 and SF-FT_400 see the identical 400 labels per parent.
- Zero-shot arms use joint renaming augmentation of (s, w); per-goal arms use the stabilizer augmentation σ ∈ Stab(w′)
  (exact and label-preserving; |Stab| logged). |Stab(w′)| ≥ (7 − m)! for a goal mentioning m blocks, since the
  unmentioned blocks permute freely; it is 1 only for goals mentioning ≥ 6 blocks with no internal symmetry (L8).
- Every arm receives w′, faces the same legal candidates, starts and banks, and scores STOP exactly.
- **E04 independence:** no arm ranks an illegal candidate (CX D-43).

### 5.5 Accounting (logged per arm)

One-off labels, labels per new goal, supervised numbers, optimizer steps, parameters, wall time, network calls per
decision, h/misplaced calls and ENUM± target counts per parent. Totals: zero-shot arms 25,600 one-off, 0 per goal;
PGV_400/FT 400 per goal (83,200 at n_P = 208; 118,400 at 296); PGV_1600 76,800 and PGV_6400 307,200 on the subset.
Break-even: SF's one-off budget equals 400 labels on 64 new goals.

## 6. Endpoints and statistics

### 6.1 Primary endpoint

The **macro-average over H-ID, H-K and H-NEG of the family mean of parent-level normalized closed-loop regret** (lower
is better).
- A parent's value for one model is the mean over its 16 starts. For replicated arms (SF, UVFA, UVFA+aux, FT-UVFA) it
  is the mean over the 3 replicates. PGV is one fit per parent; deterministic arms need no seed.
- The H-ID family mean is the parent-weighted mean over H-ID-S and H-ID-U.

### 6.2 Confirmatory family (Holm, family-wise α = 0.05)

**P1, superiority at equal labels, intersection–union (F2, F3).** D1 = nreg(UVFA) − nreg(SF-GPI); positive favours SF.
- Conjuncts: **P1-M** (macro D1) and **P1-ID** (D1 on H-ID). p_P1 = max(p_M, p_ID). An intersection–union test needs
  no adjustment across its conjuncts, so the H-ID requirement costs no Holm budget.
- **P1 passes** iff all hold:
  1. Holm-adjusted p_P1 < 0.05;
  2. both point estimates ≥ 0.03;
  3. both 95% CI lower bounds > 0;
  4. the macro D1 > 0 **and** the H-ID D1 > 0 for each matched replicate pair (SF sₖ vs UVFA sₖ, k = 20261246, 47, 48)
     (L3: extended to H-ID, the binding conjunct, as a guard against a one-seed artifact).
- **Conditional inference (L3).** The parent bootstrap resamples parents only; replicate (seed) variance is not
  resampled. P1 and P2 inference is therefore **conditional on the 3 trained replicates** (and on the one fit per
  parent of PGV), and is stated so. Condition 4 is the only seed-level check; the §8 power tables ignore it.

**P2, non-inferiority at zero vs 400 new labels (F4).** D2 = nreg(SF-GPI) − nreg(PG400) (macro).
- One-sided p_P2 = max(1/R, share of bootstrap replicates with D2\* ≥ δ_NI), δ_NI = 0.05.
- **P2 passes** iff Holm-adjusted p_P2 < 0.05 (equivalently, at the Holm level the one-sided upper bound of D2 is < 0.05).
- If P2 was dropped at the pilot, the Holm family is {P1}; if P1 was declared comparator-invalid at the pilot, it is
  {P2}; if both, it is empty.

Holm: `rcx.stats.holm` over the family fixed at the freeze. The TEST validity conditions (§7.1) never change that
family; they only decide whether a passing test counts.

### 6.3 Guard G1 and STOP-D (outside Holm)

D_G = nreg(SF-GPI) − nreg(ENUM-MIS±) (macro; positive means the zero-label planner is better).
- **G1 holds** iff the point D_G < 0.05.
- **STOP-D condition** iff the point D_G ≥ 0.05 **and** its 95% CI lower bound > 0.

### 6.4 Estimator, intervals and p-values

- **Unit of independence:** the parent (= the support-class cluster). Starts, bank states, renamed variants and
  replicates are within-cluster repeats.
- **Stratified paired cluster bootstrap**, R = 10,000. Strata: H-ID-S, H-ID-U, H-K, H-NEG (and each secondary family
  for its own tables). Within each stratum, parents are resampled with replacement; family means and the macro are
  recomputed.
- **Index streams (M3), drawn exactly as `rcx.stats` draws them.** One stream per stratum h:
  `random.Random(derive(20261250, "boot:<h>"))`. The stratum's parents are ordered as `rcx.stats._order` orders
  parent keys (sorted canonical parent keys). The R × n_h index matrix is drawn resample by resample, n_h calls of
  `randrange(n_h)` each (the draw order of `rcx.stats._boot_means`), once, and shared by every contrast (common random
  numbers).
- **Arithmetic order.** A parent's value is computed as `rcx.stats` does: per replicate, the mean over starts
  (`sum / len`); then the mean over replicates in seed order. A contrast is bootstrapped on the per-parent paired
  difference d(p) = y_b(p) − y_a(p) (`rcx.stats`' b − a), and a stratum's resampled mean is the Python `sum` of the
  drawn d values in draw order divided by n_h. Hence, on a single stratum, E05's interval and p equal
  `rcx.stats.seed_parent_bootstrap_test(per_seed_a, per_seed_b, resamples=10000, seed=derive(20261250, "boot:<h>"),
  p_floor=1e-4, ties='both')` exactly (unit test). The family and macro means are then formed from the stratum means.
- **CI:** 95% percentile interval by `rcx.stats._interval` (linear interpolation; 2.5th and 97.5th). The "upper bound"
  in futility rules is its 97.5th percentile.
- **Two-sided p:** `rcx.stats.bootstrap_p_two_sided(boot, floor=1/R, ties='both')`. With 10,000 replicates, p < 0.05
  implies the 95% interval excludes 0; condition 3 of P1 is still checked.
- **Tolerance.** Thresholds are compared with 1e-9: "x ≥ t" means x ≥ t − 1e-9, "x < t" means x < t − 1e-9, "x > t"
  means x > t + 1e-9.
- `analyze_e05.py` runs **once** per attempt and refuses unless every eval file names the committed `TEST_MANIFEST`
  sha256 and the T3 code digest, that sha is in the committed `DECISIONS.md`, and the attempt's
  `ORACLE_TEST_REPORT.json` says PASS. A refusal writes nothing and is not an analysis run. Outputs are create-only.

### 6.5 Secondary (descriptive, prespecified, no decision weight)

- Per family, per stratum (H-ID-S, H-ID-U, K, conflicts) and per replicate D1, D2, D_G; secondary families (H-COST,
  H-SPAN, H-KN, H-CONJ, H-DYN), labelled "assumption boundary" where predicted. H-K and H-NEG D1 are labelled "includes
  extrapolation outside UVFA's training weight support".
- One-step regret1 and top-1 on the identical banks; loop rate.
- **k\*:** the smallest per-goal budget at which PGV matches SF-GPI on the 48-parent subset (log-k interpolation,
  bootstrap CI, censored at < 400 and > 6,400); the same for FT-UVFA where defined.
- **Attribution:** SF-GPI vs UVFA+aux; SF-noGPI and SF-L vs SF-GPI; **learnability readout** SF-L vs EXACT-GPI(L) and
  SF-noGPI vs ORACLE; **GPI optimism bias** on the bank (mean of max_z ŵ·ψ̂ minus exact Q of the chosen action) for
  |C| ∈ {1, 5, 65}.
- SF-GPI-5; ENUM-MIS± on every retained family, and GNN-ENUM± and ENUM-H\*± where not cut (§10.3); rev 1's
  positive-only ENUM-MIS and rev 2's ENUM-MIS± (holding bound 0) on H-NEG (the size of the F5 and B1 fixes); the ENUM±
  all-tied-decision shares; SF-shuffled.
- PG400's TEST nreg per family and its DEV-OOD value; UVFA's H-ID nreg (the validity margins of §7.1).
- Model quality on TEST states: endpoint-fact accuracy, T MAE, value MAE and the c/2 value-exact rate, against the
  copy-state floor.
- Binary view: per parent, optimal on ≥ half of the starts, by exact McNemar (`rcx.stats.mcnemar_exact`).

## 7. Decision rule

### 7.1 Determinations made before the rule (B3; by `analyze_e05.py`, from the PILOT FILL and the TEST evaluation)

- **Competence bar** b = 0.50 nreg (§5.3).
- **P1-valid** iff P1 was not declared comparator-invalid at the pilot (G-P4b) **and** (V-P1) the UVFA's TEST H-ID
  mean nreg (3-replicate mean) ≤ b.
- **P2-valid** iff P2 was not dropped at the pilot (G-P3) **and** (V-P2) PG400's TEST primary macro nreg ≤ b.
- **P1 counts** iff P1-valid and P1 passes (§6.2). **P2 counts** iff P2-valid and P2 passes (§6.2).
- Validity conditions only stop a passing test from counting. They never change the Holm family fixed at the freeze
  and never block a STOP branch: futility or misfit shown against a weak comparator is still futility or misfit.
- **G1** and the **STOP-D condition** as in §6.3.
- **SF floor:** SF-GPI's family-mean nreg ≤ b on each of H-ID, H-K and H-NEG (point estimates).

### 7.2 Ordered rule (tested in this order; the first that holds is the decision)

1. **VOID.** Declared only by `oracle-test`, before any learned-arm TEST computation, when a check of §10.4 fails. An
   attempt ending in VOID is never analyzed. Fix the fault, redraw TEST from `20261242:test:<attempt+1>` and log the
   attempt. At most two redraws; a third VOID gives INCONCLUSIVE and a design review.
2. **INCONCLUSIVE (defect or missing).** A defect found after the first learned-arm TEST computation (§10.4), or a
   missing protected-core cell (§10.3). The attempt is reported in full, descriptively; no other rule is applied; no
   redraw under E05; a design review follows a defect.
3. **STOP-D (dominated).** The STOP-D condition holds. A zero-label symbolic planner beats the learned SF; the SF
   line stops on this testbed. P1/P2 are reported as mechanism findings only.
4. **ADVANCE.** P1 counts, P2 counts, G1 holds, and the SF floor holds.
5. **STOP-M (mechanism misfit).** EXACT-GPI(L) primary macro nreg ≥ 0.5 **and** neither P1 nor P2 counts (L4: a
   counting test that works through the z = w′ index is not overridden by the library ceiling; it goes to REVISE with
   the GPI note below).
6. **STOP-F (futile).** P2 does not count (it failed, was dropped at the pilot, or is comparator-invalid on TEST;
   L4) **and** (macro D1 upper bound < 0.05 **or** H-ID D1 upper bound < 0.05). This applies even if P1's significance
   conditions hold (an interval entirely below 0.05 excludes the planned effect) and whatever P1's validity.
7. **REVISE.** P1 counts or P2 counts. The report names every failed ADVANCE condition: G1 in the middle zone
   (D_G ≥ 0.05, CI includes 0); the SF floor; "label-cost claim only" (P2 counts, P1 does not); "P1 counts, P2 does
   not"; a test that passed but does not count (comparator-invalid, with the comparator's nreg); "P1-M passes, P1-ID
   does not, P2 counts" (follow-up: UVFA-wide/SF-wide arms trained on weights covering sign and cardinality); and,
   if EXACT-GPI(L) ≥ 0.5, "library ceiling ≥ 0.5: any gain is through the z = w′ index" (GPI qualifier applies).
8. **INCONCLUSIVE.** Otherwise. A positive point with a wide interval is not a pass; a nonsignificant loss is not
   equivalence.

If P1 was comparator-invalid at the pilot, ADVANCE cannot result; if P2 was also dropped, neither REVISE nor ADVANCE
can result.

**Qualifiers** (change wording, never the decision):
- **Attribution:** if macro nreg(UVFA+aux) ≤ nreg(SF-GPI) + 0.02, the gain is credited to vector consequence
  supervision, not to test-time recomposition.
- **GPI:** if macro nreg(SF-noGPI) ≤ nreg(SF-GPI) + 0.01, the claim says "goal-conditioned endpoint predictor read out
  linearly", not "GPI".
- **New directions (F8):** "new preference directions" needs H-ID-U D1 point ≥ 0.03 with CI lower bound > 0 **and**
  H-ID-U D2 upper bound < 0.05 (unadjusted). Otherwise: "re-weighting of seen fact patterns".
- **Naming (F11):** "goal-conditioned endpoint-and-length successor features (exact Monte Carlo targets, no TD)".
- **Guard quality:** if Q9 failed at the pilot without a bug (§9.1), every G1 statement adds "against a guard that
  failed its DEV sanity check on H-NEG".
- **k\*:** if the k-curve failed G-P3b twice (§5.3), k\* is labelled "descriptive only".

**Pre-declared notes** (stated now so they cannot be reinterpreted):
- **By construction.** UVFA never sees negative weights or K > 4 in training, so part of any H-K/H-NEG advantage is
  extrapolation outside its input support. This is why P1-ID is a conjunct. The predicted H-ID gap is about +0.00 to
  +0.04, below the planned 0.05, so **P1-ID is the binding constraint** and its power at the predicted gap is low (§1,
  §8).
- **GREEDY ≡ STOP** on every positive family (§3); GREEDY rows are reported but carry information only on H-NEG,
  H-KN and H-SPAN.
- **H-ID-S vs H-ID-U** differ in K mix (two-fact supports are nearly exhausted by L); stratified, disclosed.
- **ENUM± optimism** about facts that may become true favours SF in D_G slightly on H-NEG; disclosed.
- **Priors** (DESIGN §1.5; noted, never used): P1 about 0.25–0.3; P2 about 0.5; STOP-D about 0.25; ADVANCE about
  0.15; REVISE about 0.4.

**Consequences.**
- ADVANCE: E06 design (SF-GPI as a budgeted soft-goal search heuristic vs ENUM-MIS± and GNN-ENUM±; relational SF for
  n = 6 → 7 → 8); native-A probe only after E01R; FYI to BWM about a soft-goal family.
- STOP-D: the learned-SF line closes on Blocksworld; ENUM-MIS± is reported as the zero-label soft-goal reference.
- STOP-M / STOP-F: no rerun under this ID; k\* and the attribution tables are still reported.
- REVISE: the named follow-up needs its own prereg.
- INCONCLUSIVE (defect): design review; any rerun is a new prereg with fresh seeds, never a redraw of this one.
- Every outcome: R-entry in EXPERIMENT_RESULTS; FYI to BWM, RCWM and (relayed) dsv3wm for cross-reading with RDW
  E07/E08.

## 8. Sizing, power and resizing (fixed now; F2)

**Assumptions.** σ = parent-level SD of the paired difference; normal approximation at Holm step 1 (two-sided
z = 2.2414 for P1; one-sided z = 1.96 for P2); planned effect 0.05; power = Φ((0.05 − max(z·SE, 0.03)) / SE); macro
SE = (σ/3)·√(1/n_ID + 1/n_K + 1/n_NEG). For honesty: under rev 1's rule (point ≥ 0.05) power at a true 0.05 is ≤ 0.5
at any n.

**P1-M power at a true 0.05:**

| σ | (96, 56, 56) | (152, 72, 72) |
|---|---|---|
| 0.15 | 0.97 | 0.98 |
| 0.20 | 0.89 | 0.95 |
| 0.25 | 0.71 | 0.84 |

**P1-ID power at a true 0.05:**

| n_ID | σ = 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|
| 96 | 0.97 | 0.85 | 0.58 | 0.39 |
| 120 | 0.99 | 0.92 | 0.69 | 0.48 |
| 152 | 0.99 | 0.95 | 0.80 | 0.59 |

**P1-ID power below the planned effect (L2; same formula).** The predicted H-ID gap is +0.00 to +0.04.

| n_ID | true 0.03: σ = 0.10 / 0.15 / 0.20 / 0.25 | true 0.04: σ = 0.10 / 0.15 / 0.20 / 0.25 |
|---|---|---|
| 96 | 0.50 / 0.39 / 0.22 / 0.14 | 0.84 / 0.64 / 0.39 / 0.25 |
| 120 | 0.50 / 0.48 / 0.27 / 0.18 | 0.86 / 0.75 / 0.48 / 0.31 |
| 152 | 0.50 / 0.50 / 0.35 / 0.22 | 0.89 / 0.79 / 0.59 / 0.39 |

At a true 0.03 power can never exceed 0.5, because the point must reach 0.03. These figures are stated, not hidden;
the design is not resized for them.

Joint P1 power lies between max(0, a + b − 1) and min(a, b); the conjuncts are positively correlated, so it is close to
the weaker one.

**P2 NI power** (δ_NI = 0.05):

| σ₂ | D2 = 0, (96, 56, 56) | D2 = 0, (152, 72, 72) | D2 = +0.02, (96, 56, 56) | D2 = +0.02, (152, 72, 72) |
|---|---|---|---|---|
| 0.15 | 1.00 | 1.00 | 0.80 | 0.90 |
| 0.20 | 0.94 | 0.98 | 0.55 | 0.68 |
| 0.25 | 0.80 | 0.90 | 0.39 | 0.49 |
| 0.30 | 0.64 | 0.77 | 0.29 | 0.37 |

**Resizing rule** (applied once by `pilot-report`, before TEST is drawn; DEV variance only):
1. σ̂_ID = SD over the 32 DEV-ID parents of y_UVFA(p) − y_SF(p); σ̂_OOD = the same over the 24 DEV-OOD H-K and H-NEG
   parents; σ̂₂,ID = SD over DEV-ID of y_SF(p) − y_PG400(p); **σ̂₂,OOD (M5)** = the same over the 24 DEV-OOD H-K and
   H-NEG parents, from the selected PG400 fitted there in pilot B. Selection-seed models at the selected configs
   (single models, so conservative for 3-replicate means). If P1 is comparator-invalid, n_ID and n_K are still set
   from σ̂_ID and σ̂_OOD (P2 and D_G use the same parents); if P2 is dropped, the σ̂₂ terms are skipped.
2. σ_plan = 1.15·σ̂ (about the 80% upper-confidence factor for an SD from 24–32 parents: 1.16 at 23 df, 1.13 at 31);
   σ_plan,2 = 1.15·max(σ̂₂,ID, σ̂₂,OOD).
3. n_ID = min(152, max(96, ⌈3,802·σ²_plan,ID⌉)), with 3,802 = ((2.2414 + 0.8416)/0.05)². Lookup: σ̂_ID ≤ 0.138 → 96;
   0.15 → 114; 0.16 → 129; 0.17 → 146; ≥ 0.174 → 152. The formula binds.
4. n_K = n_NEG = 56, raised to 72 if, at the chosen n_ID, the planned P1-M power with σ_plan = 1.15·max(σ̂_ID, σ̂_OOD) or
   the planned P2 power at D2 = 0 with σ_plan,2 is < 0.80.
5. **Caps.** If the protected core does not fit 150 CPU-min at the G-P5 rates, or fewer distinct unseen support classes
   exist than H-ID-U needs, n_ID is reduced toward 96, then n_K and n_NEG toward 56. Planned powers are recomputed and
   written into the PILOT FILL. **Power below 0.80 at the cap is stated, not hidden** (e.g. P1-ID 0.59 at σ = 0.25,
   n_ID = 152).

## 9. Pilot gates and PILOT FILL

### 9.1 Pilot gates (DEV only; engineering decisions recorded before the freeze, outcomes known on DEV)

| Gate | Condition and consequence |
|---|---|
| G-P1 | Q1–Q9 pass on the pilot oracle (Q7 as in §3). **Q9 (ENUM-MIS± sanity, B1):** on the 12 DEV-OOD H-NEG parents, ENUM-MIS± mean nreg < GREEDY's (strict, 1e-9) and its all-tied-decision share < 0.5 (pilot only: on TEST the guard's quality is an outcome, reported, never a VOID condition). A Q9 failure triggers an audit of `enum_e05.py` against its tests: a bug found is fixed (T2 digest re-recorded, affected pilot phase re-run, logged); if none is found, ENUM-MIS± stays as specified (the guard is never redesigned after DEV is seen), the failure is written into the PILOT FILL, and the guard-quality qualifier applies (§7) |
| G-P2 | DEV-ID EXACT-GPI(L) macro nreg ≥ 0.5 → add the 49 single-fact base policies to L for every learned arm |
| G-P3 (F4, B2) | Competence bar DEV-ID nreg ≤ 0.50 as in §5.3 → PG400 identity, one extension, or "P2 dropped"; then the selected PG400 on the 24 DEV-OOD H-K/H-NEG parents (M5) |
| G-P3b (M4) | On 8 DEV-ID parents: PGV_6400 nreg ≤ 0.25 and PGV mean nreg non-increasing in k → original k-curve recipe; else the single escalation R_ext of §5.3, re-checked once; if R_ext also fails, k\* is descriptive only |
| G-P4 (F9) | On the DEV-ID bank pairs (32 × 64), selection-seed models: acc½ = share with \|V̂ − V\*\| < c/2 (V̂ = w̄·ψ̂(s, w) for SF, V_θ for UVFA, w·φ(s) for the copy-state floor). **Disclosed (L8):** bank states exclude STOP-optimal states, and on DEV-ID (c = 0.25, integer weights) V\* − w·φ(s) is then a positive multiple of 0.25 > c/2, so acc½(copy) = 0 exactly and the thresholds below are absolute. GREEDY's top-1 is 0 on hand-empty bank states of these positive goals (it stops), so the ranking comparison bites on holding states. **(a)** acc½(SF) − acc½(copy) ≥ 0.40 **and** top-1(SF-noGPI) − top-1(GREEDY) ≥ 0.20. **(b) UVFA validity (B3):** acc½(UVFA) − acc½(copy) ≥ 0.20 **and** DEV-ID nreg(UVFA) ≤ 0.50, else one UVFA extension; if it still fails, P1 is comparator-invalid (§5.3). If (a) fails: 64 × 800 labels for every arm; if again, n = 6 (all n-dependent counts replaced by their n = 6 values; thresholds unchanged; the claim is restated for n = 6, §1) |
| G-P5 (F6) | Measured steps/s per model size and batch (single and batched B ∈ {16, 64}), VI s per weight, evaluation seconds per (arm, parent, start) cell incl. the extended ENUM± menus → projected confirmation CPU-min at the §8 sizes; §10.3 cuts applied until ≤ 150; the safe-point plan of §10.2 checked against these rates (no unit > 30 s) |

### 9.2 PILOT FILL (the only text added at T3; copied verbatim from `PILOT_REPORT.json`)

| Field | Value |
|---|---|
| T2 pilot code digest; test-file shas; `PILOT_MANIFEST.json` sha256 | ⟨FILL⟩ |
| n (7, or 6 by G-P4 fallback); TRAIN labels per goal (400/800) | ⟨FILL⟩ |
| Library composition (64, or 64 + 49) and the G-P2 value | ⟨FILL⟩ |
| G-P1 values: Q7 (ORACLE, STOP, RAND DEV primary macro, GREEDY identity), Q9 (ENUM-MIS± vs GREEDY on DEV-OOD H-NEG, all-tied share; audit outcome if failed) | ⟨FILL⟩ |
| Selected epochs for SF, UVFA, UVFA+aux; UVFA extension used?; G-P4 values (acc½ deltas, top-1 delta, UVFA DEV-ID nreg); **P1 valid / P1 comparator-invalid** | ⟨FILL⟩ |
| PGV_400 and FT-UVFA_400 selected configs; extensions used?; DEV-ID nreg of each against the bar 0.50; **PG400 = PGV_400 / FT-UVFA_400 / P2 dropped**; PG400's DEV-OOD H-K/H-NEG nreg | ⟨FILL⟩ |
| G-P3b: PGV_6400 nreg and monotonicity on the 8 DEV-ID parents; R_ext used?; k\* decision-driving or descriptive only | ⟨FILL⟩ |
| DEV-ID and DEV-OOD macro nreg for every pilot arm; DEV D1, D2, D_G | ⟨FILL⟩ |
| σ̂_ID, σ̂_OOD, σ̂₂,ID, σ̂₂,OOD; σ_plan values | ⟨FILL⟩ |
| n_ID, n_S (H-ID-S), n_U, n_K = n_NEG; available unseen support classes | ⟨FILL⟩ |
| Planned powers: P1-M, P1-ID, P2 at D2 = 0 and +0.02 | ⟨FILL⟩ |
| G-P5 rates; projected confirmation CPU-min; cut depth (items of §10.3 applied); largest evaluation and training unit (s) for the safe-point plan | ⟨FILL⟩ |
| Exclusions sha (re-pinned if the B4 fresh set was added) | ⟨FILL⟩ |
| Reference shas of `lane_e05.py`, `rcx/lane.py`, `rcx/guard.py` at the pilot (not part of the digest) | ⟨FILL⟩ |
| T3 full code digest (incl. `analyze_e05.py`); test pass log | ⟨FILL⟩ |

## 10. Execution, lane and safety

### 10.1 Order

1. T1: CX D-entry adopting this draft and DESIGN rev 3, including the E05 lane rule of §10.2 (which retires D-43's
   RAM < 11 GB rule for E05). Message to BWM asking it to confirm the residual overlap and thermal points of §10.2
   under D-84 (L9).
2. Write the code and tests (no lane). All tests pass; record the T2 digest (`run_e05.py code-digest`) in DECISIONS.
3. `pilot-a` (Phase 1a), `pilot-b` (Phase 1b), `pilot-report` → `PILOT_REPORT.json`.
   - **L7.** If E05 is abandoned or postponed after the pilot for any reason (including a large DEV contrast), the
     `PILOT_REPORT.json` and a short descriptive note ("pilot, DEV only, not a confirmation") are still published as an
     EXPERIMENT_RESULTS entry and referenced in DECISIONS.
4. T3: PILOT FILL → `PREREG_E05.md`; record its sha256 and the T3 digest in DECISIONS; commit.
5. T4: `draw-test --expect-prereg-sha <T3> --expect-code-digest <T3>`; commit the `TEST_MANIFEST.json` sha256.
6. `oracle-test`: **every VOID check** of §10.4, before any learned-arm TEST computation; writes
   `ORACLE_TEST_REPORT.json` (PASS or VOID). On VOID: fix, redraw (step 5 with `<attempt+1>`), log.
7. `train` (replicates 20261246–48 of SF, UVFA, UVFA+aux; SF-shuffled). Replicates may be trained between T3 and T4;
   they never touch TEST until step 8.
8. Only after PASS: `pgv` (PGV_400 on all primary parents, then the k-curve on the subset), `ft`, `eval`, `extras`
   (unless cut). The first of these to touch TEST is "the first learned-arm TEST computation" (§10.4).
9. `analyze_e05.py --manifest-sha <T4> --code-digest <T3>`, once.
10. T6: R-entry, DECISIONS entry, FYIs.

### 10.2 Lane rules (binding; CX D-43 as amended by D-44 and BWM D-84; this rule is recorded at T1)

- **Party** `rcx-e05`, via `rcx.lane.try_acquire` only. E05 **never writes a waiting marker** (never calls
  `rcx.lane.acquire`), never writes any `rineya-gpu-0.*` file, and never writes the quiet-release file.
- **Inputs, read at every phase start and at every safe point by `lane_e05.py`** (nothing is hard-coded):
  - **Windows:** `runs/RCX-E01R/quiet_windows.json` (the D-84/D-51 windows; at this draft 23:00–07:00 on the nights
    starting 10-12, 10-13 and 10-14, `end_margin_min` 5), united with `runs/RCX-E05/extra_windows.json`, into which the
    coordinator copies any window agreed later in a DECISIONS or COLLABORATION entry before it reaches
    `quiet_windows.json` (e.g. the 10-15/16 and 10-16/17 nights BWM accepted at 16:40, once approved). An unreadable
    or unparseable windows file → refuse; an absent extras file means no extra windows.
  - **GPU lease** `%TEMP%/rineya-gpu-0.lock` (read only). It has a **live holder** iff its `pid` or any entry of its
    `worker_pids` is alive (`rcx.lane.pid_alive`). No file means no holder; an unreadable file → refuse.
  - **E01R state:** `%TEMP%/rineya-quiet-release.rcx.json`, `runs/RCX-E01R/cells/train_chain.log` and
    `post_train_chain.log`, and the lane lock's holder.
- **Start condition** (all must hold; otherwise the phase does not start and nothing is written):
  1. The lane is free (no lock after stale-pid reclaim) and **no `rineya-cpu-lane.waiting.*` marker of any party**
     exists.
  2. **Free RAM ≥ 10.0 GB** (`rcx.guard.available_ram_gb`; unknown → refuse). This is BWM's D-84 condition, so that a
     GPU bundle of about 8 GB starting mid-phase cannot push the box below the shared 6 GB floor. E05 stays < 1.5 GB RSS.
  3. `rcx.guard`: tz01 < 97 °C (or resumed at ≤ 93 °C) and no `.thermal_pause` file.
  4. No stray python process.
  5. Projected phase length L = work at the G-P5 rate (19 GFLOP/s before G-P5 exists) × 1.2 ≤ 15 min;
     `expected_end` = start + 15 min.
  6. **Either day mode:** now is outside every window, the GPU lease has **no live holder**, and start + L ≤ (next
     window start − 10 min);
     **or night-tail mode:** now is inside window W, **E01R has no remaining block in W**, the GPU lease has no live
     holder, and start + L ≤ W.end − 5 min. "E01R has no remaining block in W" iff the quiet-release file exists, or
     the newest `lane:` line of E01R's active chain log (the post-train log once it has started, else the training
     log) is `rcx.lane.acquire`'s refusal "inside the window ending ⟨W.end⟩, but a … block would end too late" and E01R
     does not hold the lane. Anything else, including an unreadable log, counts as "has a remaining block".
- **Retired rule (M1).** D-43's "available RAM < 11 GB while E01R is pending" no longer applies to E05. Since D-44, E01R
  takes lane blocks only inside windows, so the rule protected nothing outside them, and in daytime it would starve E05:
  RAM is plentiful exactly when no GPU job holds it, which is when D-84 lets E05 run. Inside windows its purpose (E05
  never takes time E01R can use) is served by the night-tail condition. E05 still never starts within 10 min of a
  window opening, and never inside a window while E01R may still take a block there.
- **Safe points (M2).** A safe-point check runs after every optimizer step or evaluation cell once ≥ 60 s have passed
  since the previous check, and at every epoch end and batch end. Units are sized from the G-P5 rates so that no single
  uninterruptible unit (one optimizer step of a `BatchedMLP` batch; one (arm, parent, start) evaluation cell) exceeds
  30 s; checks are then ≤ 90 s apart at the measured rate and ≤ 120 s at half of it. (Rev 2's "batches of ≤ 64 models"
  were 6.5 min per PGV_400 batch at 19 GFLOP/s.) At a check, the job checkpoints (every model's weights, AdamW
  states, early-stopping snapshots, data-order RNG state and step count), releases the lane and exits if any of these
  holds: a `waiting.*` marker exists; 15 min have elapsed; in day mode, the GPU lease has a live holder or the next
  window opens within 10 min; in night-tail mode, W.end − 5 min is reached; the guard asks to pause; **free RAM < 6.0 GB** (the
  shared "≥ 6 GB free while any GPU job runs" rule, BWM D-84 condition). Resumption reproduces the uninterrupted run (tested). Outputs are keyed and append-only.
- **Interaction with others.** E05 stops ≥ 10 min before each window opens and enters a window only after E01R has
  declared that its next block cannot fit, so E01R should never find E05 holding the lane when it could start a block;
  if it does, its `acquire` writes `waiting.rcx` and E05 yields at the next check. Overlap with a GPU bundle is limited
  to a bundle that starts mid-phase, until the next check (≤ 2 min).
- **BWM confirmation (2026-10-12):** accepted, with the RAM conditions above. E05 holds the lane as party `rcx-e05`, so
  BWM's post-07:00 hold (which watches holder `rcx`) never treats an E05 phase as an E01R adapter. E05 still exits
  ≥ 5 min before every window end.
- **BWM confirmation at T1 (L9, as asked).** BWM is asked to confirm under D-84 (a) that this ≤ 2-min residual overlap with a
  newly started GPU bundle is acceptable, and (b) the thermal note: tz01 idles at 94–96 °C, and E05's 4-thread slices
  obey the guard (pause at 97 °C, resume at ≤ 93 °C). If BWM does not accept (a), E05 runs in night-tail mode only.
- Torch threads ≤ 4; `OPENBLAS_NUM_THREADS=1` for the oracle; `CUDA_VISIBLE_DEVICES=""` and the torch.cuda patch before
  torch import.
- **Only the coordinator runs phases.** Subagents edit code and run < 5 s checks while any lane job runs.

### 10.3 Budget, protected core and cut order

Planned at 19 GFLOP/s (pessimistic) / 48 (optimistic) from R-04 (FlatMLP 158,721 parameters, 20,724 steps at batch 64
in 26.1–66.1 s): pilot ≈ 16.5 / 7.5 CPU-min, plus ≈ 2.5 / 1 for PG400 on the 24 DEV-OOD parents (M5) and, only if
G-P3b escalates, ≈ 3.2 / 1.3 for R_ext on 8 DEV-ID parents (M4): ≤ 22 / 10 (cap 30); confirmation ≈ 128–147 / 60–70
CPU-min (cap 150), in about 10–12 slices. The 14-fact on-menus (L1) raise ENUM±'s target counts on H-NEG and H-KN
(≈ 2–4× there); G-P5 measures this, and cut items 4 and 6 absorb it. R_ext, if adopted, lowers the k-curve cost (462
instead of 815 GFLOP per subset parent). G-P5 replaces these with measured rates.

**Protected core (never cut):** the TEST oracle; SF, UVFA, UVFA+aux × 3 replicates; SF-noGPI, SF-L, SF-GPI-5;
SF-shuffled; **PGV_400 on all primary parents**; PG400 on all primary parents (3 replicates if FT); **the k-curve
(PGV_1600 and PGV_6400, and PGV_400 too under R_ext) on the 48-parent subset**; ENUM-MIS±, GREEDY, STOP, RAND, ORACLE,
EXACT-GPI(L), BEST-SINGLE(L) on every primary family and on every secondary family not removed by cut items 3 or 7
(L8); ENUM-H\*± and GNN-ENUM± on H-ID and H-K; the one-step banks.

**Cut order** (at G-P5, until the projection is ≤ 150 CPU-min): 1. budget curve; 2. SF-FT_400; 3. H-KN; 4. GNN-ENUM± on
H-NEG; 5. FT-UVFA_400 to replicate 20261246 only, if FT is not PG400; 6. ENUM-H\*± on H-NEG to the first 24 parents in
manifest order; 7. H-DYN. If the protected core alone exceeds 150, sizes shrink by §8 step 5 before TEST is drawn; if
it still exceeds 150, a CX D-entry raises the cap (time reasons only) or postpones. Nothing in the protected core is
cut after TEST is drawn.

### 10.4 VOID, defects and failures (B4)

- **VOID checks, all inside `oracle-test`, before any learned-arm TEST computation:** Q3 and Q4 on the TEST oracle;
  Q7 (§3); Q8 (weighted-class disjointness on the M6 key, support uniqueness, H-ID-U unseen-ness, the exclusion checks
  of §4.4 on starts, bank states and per-goal label rows, the n = 7 assertion vs RCX test sets, the 48-parent subset as
  drawn); the manifest sha (file = committed sha, in the committed `DECISIONS.md`) and the T3 code digest. Each value
  goes into `ORACLE_TEST_REPORT.json`.
- **VOID may be declared only there.** On VOID nothing learned has touched TEST: replicates are kept (they never see
  TEST) and no per-goal fit has started. Fix the fault, redraw from `20261242:test:<attempt+1>`, log. At most two
  redraws; a third VOID gives INCONCLUSIVE and a design review.
- **After the first learned-arm TEST computation** (the first `pgv`, `ft`, `eval` or `extras` call of the attempt), no
  VOID exists. Any defect found later (a bug, a custody breach, an outcome-affecting change, a mismatch caught by
  `analyze_e05.py`) gives **INCONCLUSIVE (defect)** and a design review: no redraw, no re-analysis, and the attempt is
  reported in full with the defect described. A refusal that writes nothing (e.g. a wrong CLI argument) is not a
  defect; the command is rerun correctly.
- A crashed or yielded phase resumes from its checkpoint under the same fingerprint; completed keyed outputs are never
  recomputed.
- A replicate or fit that cannot be completed is **missing**; it is never replaced by another seed. A missing
  protected-core cell gives INCONCLUSIVE (§7.2 item 2).
- Every confirmation attempt, including VOID redraws, is logged in `runs/RCX-E05/ATTEMPTS.jsonl`. A redraw is never
  called a fresh confirmation of the same bank.

## 11. Freeze points and deviations

| Point | What is fixed | Recorded in |
|---|---|---|
| T0 (now) | Nothing; DESIGN rev 3 and this draft | – |
| T1 adoption | Scope, the dsv3wm boundary, seeds 20261242–50, E04 independence, the §10.2 lane rule (retiring D-43's RAM < 11 GB rule for E05), DESIGN rev 3 sha; BWM asked to confirm §10.2 (L9) | CX D-entry |
| T2 pilot-code freeze | Digest of `soft.py`, `families_e05.py`, `enum_e05.py`, `learners_e05.py`, `run_e05.py` and the pinned files (not `lane_e05.py`, `rcx/lane.py`, `rcx/guard.py`, whose shas are recorded for reference); tests pass. TRAIN and DEV are drawn by this code | DECISIONS ("E05 pilot code digest") |
| T3 prereg freeze | `PREREG_E05.md` = this draft + PILOT FILL; full digest incl. `analyze_e05.py` | DECISIONS (prereg sha + digest) |
| T4 TEST draw | `TEST_MANIFEST.json` from `20261242:test:1`; sha committed before any arm is evaluated on TEST | DECISIONS |
| T5 confirmation | `oracle-test` (the only place VOID can be declared), replicates, per-goal fits, evaluation, **one** analysis run | `runs/RCX-E05/confirm/…`, ATTEMPTS.jsonl |
| T6 report | R-entry; DECISIONS entry; FYI to BWM, RCWM and (relayed) dsv3wm | – |

- After T2, any change to a T2-digest file before T3 re-runs the affected pilot phase and is logged.
- After T3, any change to a bound file is a logged deviation marked "outcomes known: yes/no". A change to
  `lane_e05.py` is an operational deviation (logged; `test_e05_lane.py` must pass); it cannot change outputs.
- Between T4 and `oracle-test`'s PASS, a fault that could affect TEST outcomes VOIDs the attempt (redraw).
- **After the first learned-arm TEST computation (B4)**, the following give INCONCLUSIVE (defect) and a design review,
  never a VOID or a redraw: any change that could affect TEST outcomes; evaluating a selection model on TEST; editing
  or re-creating a committed manifest; retraining a replicate; a second analysis run on the same attempt. The attempt
  is reported in full.

**Open items (L10).**
1. The T1 CX D-entry is still to be recorded (with the BWM message of §10.1).
2. Re-checked at this revision: `experiments/RCX-E04/DESIGN.md` now uses seeds from the 20261206–14 block; a grep finds
   20261242–50 only in `DECISIONS.md` and the E05 files. Re-checked at T1.
3. BWM's B4 fresh hard-style set (seed 20261410, n = 110) is not yet in `EXCLUSIONS.json`; `EXTRA_RESERVED_FILES`
   already holds `instances_seeds_fresh.json` (BWM item 7, D-16). If the B4 file exists by T3 it is added and re-pinned
   in the PILOT FILL; otherwise §4.4 item 3 applies.
4. The extra E01R nights (10-15/16, 10-16/17) await the operator's approval; once approved they reach E05 through the
   windows files (§10.2), with no code change.

## Appendix A. Seeds (block 20261242–20261250, CX D-43; E04 uses 20261206–14)

| Seed | Purpose |
|---|---|
| 20261242 | Family generation and splits. TRAIN/DEV at the pilot; TEST `20261242:test:<attempt>` after T3; subset `20261242:pgv-subset` |
| 20261243 | TRAIN labelled-state sampling; renaming-augmentation stream; SF-shuffled permutation `20261243:shuffle` |
| 20261244 | Starts and one-step banks (per split); RAND (`20261244:rand`) |
| **20261245** | **Selection only:** pilot model init, per-goal grids on DEV, G-P3/G-P4, σ̂. Never a TEST replicate |
| **20261246, 20261247, 20261248** | **Confirmation replicates** for SF, UVFA, UVFA+aux and FT-UVFA_400 from each; SF-shuffled, budget curve and SF-FT_400 use 20261246 |
| 20261249 | Per-goal label sampling (nested 400 ⊂ 1,600 ⊂ 6,400; shared by PGV, FT, SF-FT; also the 400 labels of the 24 DEV-OOD H-K/H-NEG parents) and PGV init, derived per parent and k |
| 20261250 | Analysis bootstraps: `random.Random(derive(20261250, "boot:<stratum>"))` (§6.4) |

derive(seed, tag) = int.from_bytes(sha256(f"{seed}:{tag}")[:8], "big"). Unit tests use literal seeds 0–9.
