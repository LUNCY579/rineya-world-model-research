# PREREG RCX-E05 (DRAFT): goal-conditioned endpoint-and-length successor features with GPI re-rank Blocksworld moves zero-shot for held-out goal weights

**Template:** bundle `06_templates/PREREGISTRATION.md`.
**Design:** `experiments/RCX-E05/DESIGN.md` rev 2 (sha256 `8e5e91bcff89…` at this draft; re-pinned at T1). It resolves critique items F1–F9 of `CRITIQUE_RVM_DESIGNS.md`
(E05 section) and records F10–F12. Where this file is more specific than DESIGN, this file binds.
**Track:** RVM-001 (RCI-001 registry E05 "Goal-factorized value transfer"; RCX roadmap P0-c). An RCX free lane
(COLLABORATION.md §2); no peer co-sign is needed. Adjacent to dsv3wm RDW-001 E07/E08 (boundary note, COLLABORATION.md
2026-10-12 about 12:55).

**Status:** **DRAFT, not frozen.** Written 2026-10-12 after CX D-43. No E05 code exists, nothing has run, and no E05
output (pilot or test) has been seen.
- It becomes `PREREG_E05.md` at T3 (§11): this text plus the **PILOT FILL** (§9.2) and nothing else. Its sha256 and
  the full code digest are then recorded in a DECISIONS CX D-entry.
- Fixed now and never changed at the fill: every threshold, the decision rule and its order, the sizing formula, the
  cut order, the seeds and the arm definitions.

**What will be known at the freeze** (DEV only; selection seed 20261245; written into the PILOT FILL):
- DEV-ID and DEV-OOD nreg for every pilot arm (SF-GPI, SF-noGPI, SF-L, UVFA, UVFA+aux, the PGV_400 and FT-UVFA_400
  grids, ENUM-MIS±, GREEDY, STOP, RAND, ORACLE, EXACT-GPI(L)), hence the DEV values of D1, D2 and D_G;
- σ̂_ID, σ̂_OOD and σ̂₂, the gate values of G-P2 to G-P5, and the measured throughput.

The DEV contrasts are visible to whoever writes the fill. That is why the fill is mechanical (`pilot-report` computes
every value) and why sizing uses DEV **variance** only. No TEST parent is drawn and no confirmation replicate is trained
before the freeze.

**Changes from DESIGN rev 1** (the critiqued text, sha256 `94d315246ac7…`). Full table in DESIGN "Changes from rev 1".

| # | Resolution in this prereg |
|---|---|
| F1 | Seeds 20261242–50 (CX D-43; E04 uses 20261206–14). 20261245 is selection only; replicates are 20261246–48 (App. A) |
| F2 | P1 rule: Holm p < 0.05, CI lower bound > 0 **and** point ≥ 0.03. Power stated for that rule at the planned 0.05; resizing rule fixed now (§8) |
| F3 | P1 is an intersection–union test: the macro **and** H-ID alone must pass. Per-family results always reported (§6.2) |
| F4 | Stabilizer augmentation for per-goal arms; FT-UVFA_400 added; competence gate vs GREEDY (margin 0.20) on DEV; P2's comparator PG400 fixed at the pilot (§5.3) |
| F5 | ENUM± handles negative weights (falsifier menus, persistence rule); STOP-D branch (§6.3, §7) |
| F6 | Batched multi-model PGV/FT measured at the pilot (G-P5). Protected: PGV_400 on all primary parents, PGV_1600/6400 on a fixed 48-parent subset (§10.3) |
| F7 | dsv3wm RDW E07/E08 cited; boundary note sent; no dsv3wm data or code used |
| F8 | One parent per (signed) support class per TEST family; H-ID split into seen (H-ID-S) and unseen (H-ID-U) supports; only H-ID-U carries "new preference directions" (§4, §7) |
| F9 | G-P4 gates on value error < c/2 and on one-step top-1, each relative to the copy-state floor (§9.1) |
| F10 | Learnability readout (SF-L, SF-noGPI vs exact references) and GPI optimism bias are named secondaries |
| F11 | Naming: "goal-conditioned endpoint-and-length successor features (exact Monte Carlo targets, no TD)" |
| F12 | Mechanism and instrument study; G1/STOP-D and k\* are the decision-driving deliverables |

**Execution scope:**
- CPU only, ≤ 4 torch threads, < 1.5 GB RSS. No language model, no GPU, no 0.6B load.
- No peer data. Every label comes from the exact oracle at run time.
- Tier: instrument plus mechanism (an external specialist, tier B). **Nothing here is a Rineya capability claim.**
- No model is published, promoted or handed to a planner under this ID.

## 1. Claim and scope

**H-E05.** On the exact Blocksworld state space for n = 7 (65,990 states, BWM 4-op semantics, fixed dynamics), with
preferences linear in 56 observable fact features, a goal-conditioned successor-feature critic ψθ(s, z), trained once
on exact labels for 64 training goal weights and composed by generalized policy improvement (GPI), achieves on held-out
goal weights:
- **(P1)** lower normalized closed-loop regret than a matched-capacity scalar goal-conditioned value (UVFA) trained on
  the identical (state, weight) labels, **both** on the macro over H-ID, H-K and H-NEG **and** on H-ID alone;
- **(P2)** regret non-inferior, within δ_NI = 0.05, to the better competent per-goal scalar critic given 400 fresh exact
  labels of each held-out goal (PG400: PGV_400 from scratch or FT-UVFA_400, fixed at the pilot).

The SF critic uses **zero** labels per new goal.

**Effect sizes (fixed).** Planned effect for P1: 0.05 nreg. Minimum point estimate for a pass: 0.03. NI margin for P2:
0.05. Dominance margin for STOP-D: 0.05.

**Naming (F11).** In this deterministic, undiscounted episodic MDP, ψ^π(s) is the (endpoint facts, length) of a
deterministic policy, learned by supervised regression on exact rollout targets with no TD learning. Every claim says
"goal-conditioned endpoint-and-length successor features (exact Monte Carlo targets)".

**Framing (F12).** No consumer exists today (BWM goals are hard `on` facts). E05 is a mechanism and instrument study. Its
decision-driving deliverables are G1/STOP-D (does a zero-label symbolic planner already serve soft goals here?) and k\*
(the per-goal label budget at which a scalar critic matches zero-shot SF). P1 and P2 are the scientific tests.

**A positive result (ADVANCE) would establish**, at tier B with exact dynamics and n = 7 only: zero-shot transfer of a
learned endpoint-and-length SF critic with GPI to unseen preference weights inside the feature span; superiority over a
scalar goal-conditioned critic at equal labels, including on in-distribution weights; non-inferiority to the better
competent per-goal critic with 400 labels per goal; no domination by the zero-label ENUM-MIS± planner; and a measured
k\*. "New preference directions" is allowed only under the H-ID-U qualifier (§7); otherwise the claim is "re-weighting
of seen fact patterns".

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
| `rcx/stats.py` | `06a30c10…` | `bootstrap_p_two_sided`, `holm`, `mcnemar_exact`, `seed_parent_bootstrap_test` (reference for the unit test) |
| `rcx/lane.py`, `rcx/guard.py` | `03f93d59…`, `08fa782d…` | lane lock (`try_acquire`, `read_lock`, `release`); thermal and RAM guard |

**New files (to be written; none exists at this draft):**

| File | Content |
|---|---|
| `rcx/bw/soft.py` | `phi_table`; `soft_values` (max-plus value iteration, optional edge mask); `policy_table` (greedy with the tie rule; endpoints and lengths by pointer doubling); `sf_table`; `gpi_scores`; `height_limit_mask(space, h=3)`; `conj_values` |
| `experiments/RCX-E05/families_e05.py` | family generators; `canon_key`; `support_key(w, signed)`; `stabilizer(w)`; renaming of states and weights; seen/unseen quotas; one-parent-per-support assertion; exclusion adaptor; `derive(seed, tag)` |
| `experiments/RCX-E05/enum_e05.py` | ENUM± target enumeration (falsifier menus, satisfiability, persistence rule); ENUM-MIS±, ENUM-H\*±, GNN-ENUM±; GREEDY, RAND, STOP, EXACT-GPI(L), BEST-SINGLE(L), ORACLE |
| `experiments/RCX-E05/learners_e05.py` | SF, UVFA, UVFA+aux; `BatchedMLP` for PGV and FT; renaming and stabilizer augmentation; trainer with checkpoints; library ψ̂ cache |
| `experiments/RCX-E05/run_e05.py` | subcommands `code-digest`, `pilot-a`, `pilot-b`, `pilot-report`, `draw-test`, `oracle-test`, `train`, `pgv`, `ft`, `eval`, `extras`; lane gate (§10.2); keyed append-only outputs; `ATTEMPTS.jsonl`; provenance manifest |
| `experiments/RCX-E05/analyze_e05.py` | stratified paired cluster bootstrap; IUT P1; NI P2; Holm; the §7 rule and qualifiers; tables; k\*; create-only report |

**Tests (each < 5 s on n = 3–5 spaces, literal seeds 0–9, never reported as results):**

| Test file | Must show |
|---|---|
| `tests/test_e05_oracle.py` | Q1 value iteration = brute force (n = 5, 200 random w incl. negatives and `clear`, c ∈ {0.125, 0.25, 0.75}); Q2 H-CONJ vs `GoalLabels` at n = 5; Q3 rollouts = tables and V\* = w̄·ψ exactly; Q4 GPI ≥ every library policy; Q5 renaming invariance and the conjugate-policy ψ identity; Q6 height-limit mask = direct rule check |
| `tests/test_e05_families.py` | canonical and support keys invariant under random σ (9 two-fact on/ontable support classes); signed keys; stabilizer correctness; split and support disjointness; one parent per support; quotas; exclusion adaptor on synthetic signatures; derived-stream determinism |
| `tests/test_e05_enum.py` | falsifier menus; satisfiability pruning; a hand case where a penalty must be avoided (ENUM± avoids it, positive-only ENUM does not); `holding` targets; ENUM-H\*± = ORACLE on single-target goals; positive-only goals reduce to rev 1's ENUM |
| `tests/test_e05_learners.py` | `BatchedMLP` = B independent models (same init and data order, 50 steps, max abs diff < 1e-5); per-model early stopping; augmentations preserve labels exactly; checkpoint = independent shorter run; SF-shuffled permutes within weight only; CUDA-off patch |
| `tests/test_e05_analyze.py` | stratified bootstrap = `rcx.stats.seed_parent_bootstrap_test` on one stratum; IUT p = max; NI p; Holm with P2 dropped; every §7 branch on synthetic inputs (VOID, missing, STOP-D, ADVANCE, STOP-M, STOP-F incl. a significant-but-small P1, REVISE incl. P1-M-only, INCONCLUSIVE); qualifiers; tolerance; refusal on a manifest or digest mismatch |
| `tests/test_e05_lane.py` | temporary `RINEYA_LANE_DIR`, injected clock and RAM: refuses with any `waiting.*` marker, with RAM ≥ 11 GB while E01R is pending, inside or running into a night window, and with the lane held; never writes a marker; yields at a safe point when a marker appears; caps a phase at 15 min |

**Code digest.** `run_e05.py code-digest` = sha256 over the sorted lines `"<relpath>\t<sha256>\n"` of the new source
files and the pinned existing files above. The T2 (pilot) digest covers every new source file except
`analyze_e05.py`; the T3 digest covers all of them. Test-file shas are recorded beside each digest, with the pass log.

### 2.2 Data and manifests

- **No dataset file is consumed.** All data come from the exact oracle and the seeds in App. A.
- `runs/RCX-E05/pilot/PILOT_MANIFEST.json` (written by `pilot-a` under the T2 digest): TRAIN and DEV parents, label
  indices, starts, banks, rejection counts. Its sha256 goes into the PILOT FILL.
- `runs/RCX-E05/confirm/attempt-<k>/TEST_MANIFEST.json` (written by `draw-test` after T3; refuses unless
  `--expect-prereg-sha` and `--expect-code-digest` match): TEST parents per family and stratum, starts, one-step banks,
  the 48-parent subset and the per-goal label index sets. Its sha256 is committed to DECISIONS **before any arm is
  evaluated on TEST**. The oracle runs used to draw TEST are the only TEST computation before that commit.
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
- **Normalized regret.** J = w′·φ(s_final) − c′·T; nreg = clip((V\* − J) / (V\* − w′·φ(s0)), 0, 1). 0 is optimal; 1 is
  no better than stopping at once.
- **One-step bank.** 64 states per parent, uniform over states where STOP is not optimal; identical for every arm.
  regret1 = (max Q\* − Q\*(chosen)) / c′, and the top-1 optimal rate.
- **Instrument checks** (DESIGN App. B): Q1–Q6 as unit tests and at the pilot; Q3, Q4, Q7 and Q8 again on the TEST
  oracle before any learned arm is evaluated there.

## 4. Goal families, sizes and splits

### 4.1 Keys

- **Parent key:** (w, c, reward type), canonical under block renaming (minimum over the 5,040 permutations).
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
| DEV-OOD | 48 | 16 starts |
| **TEST primary** | n_ID ∈ [96, 152], n_K = n_NEG ∈ {56, 72}, set by §8 | 16 starts, 64 bank states, 400 per-goal labels |
| 48-parent subset | 16 per primary family (8 per H-ID stratum), from `20261242:pgv-subset` | nested 400 ⊂ 1,600 ⊂ 6,400 labels |
| TEST secondary | 184 | 16 starts, 64 bank states |

### 4.4 Leakage controls (asserted at generation; counts logged)

1. Weighted classes are disjoint across TRAIN, DEV-ID, DEV-OOD and every TEST family. Support classes are unique within
   each TEST family. H-ID-U supports are disjoint from supports(L) ∪ supports(DEV-ID).
2. States are shared by design (n = 7 has 26 state orbits). The unit of novelty is the (state, weight) pair.
3. Every TRAIN label row and every DEV/TEST start is read as a BWM instance (state, positive on/ontable facts) and checked
   with `excl.row_exclusion_reason` on `with_selection_sets(load_exclusions())`; H-CONJ (start, G) pairs with
   `excl.exclusion_reason`. Matches are dropped and counted. If BWM's B4 fresh set (seed 20261410) exists by T3, it is
   added through `excl.EXTRA_RESERVED_FILES` and the shas are re-pinned in the PILOT FILL.
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
| **FT-UVFA_400** | each UVFA replicate fine-tuned on the same 400 labels, goal input fixed to w̄′, all parameters, AdamW lr 3e-4, batch 64, stabilizer augmentation, same early-stopping split; 3 fits per parent, averaged | 400 | P2 candidate |
| PGV_1600 / PGV_6400 | width 256; batch 128 / 256; 2,000 / 2,500 steps; stabilizer augmentation; early stopping on 10%; not tuned | 1,600 / 6,400 | k\* (48-parent subset) |
| SF-FT_400 | SF replicate 20261246 fine-tuned on the ψ targets of the same 400 labels (FT's selected steps) | 400 | descriptive |
| ENUM-MIS± | ENUM± targets scored with the admissible `misplaced_compact` bound (0 for `holding` targets) | 0 (symbolic) | **guard G1 and STOP-D** |
| ENUM-H\*± | ENUM± with exact h\* per target (`backward_bfs`) | 0 | oracle-information reference |
| GNN-ENUM± | ENUM± with the frozen R-04 GNN as ĥ | 0 | existing RCX critic, zero-shot |
| GREEDY | one-step lookahead on w′·φ(s′) − c′, plus the STOP rule; the copy-state floor | 0 | floor |
| STOP / RAND | always stop / uniform over {STOP} ∪ moves (`20261244:rand`) | 0 | floors |
| EXACT-GPI(L) / BEST-SINGLE(L) | GPI over exact library policy tables / best single library policy at s0 | – | mechanism ceiling / reuse reference |
| ORACLE | π\*_{w′} | – | regret 0 |

**ENUM± (F5).** For w′ with positive facts P and negative facts N, targets are G = G⁺ ∪ R with G⁺ ⊆ P and, for each
f ∈ N, "accept" or one falsifier from M(f): M(on(a,b)) = {ontable(a), holding(a), clear(b)}; M(ontable(a)) = {holding(a)}
∪ {on(a,x)}; M(clear(a)) = {holding(a)} ∪ {on(x,a)}. G is kept only if `CompiledGoal(G).consistent` and
`goal_mask(G).any()`. U(G; s′) = Σ_{f∈G⁺} w_f + Σ_{f∈N} w_f·1[f ∈ φ(s′) and G ∪ {f} satisfiable]; score(s′) = max over G
of [U(G; s′) − c′·(1 + ĥ(s′, G))]. Facts false now that might become true are ignored (optimistic; disclosed).

### 5.3 Tuning (DEV only, selection seed 20261245) and the P2 comparator

- **Zero-shot arms.** Epochs {20, 40} at lr 1e-3 (20 is a checkpoint of the 40-epoch run); per arm, by DEV-ID macro
  nreg. **UVFA competence (G-P4b):** if it fails, one extension {80 epochs, lr 3e-4}.
- **PGV_400 grid:** width {64, 256} × steps {500, 2,000}, batch 64 (500 is a checkpoint). **FT grid:** steps
  {100, 400} (100 is a checkpoint). Tuned once on DEV-ID; never per goal.
- **Competence (G-P3).** A candidate is competent if its best config reaches DEV-ID macro nreg ≤ nreg(GREEDY) − 0.20.
- **PG400** = the competent candidate with the lower DEV-ID nreg (ties to FT-UVFA_400). If neither is competent, each
  grid is extended once (PGV: width 128, 4,000 steps, lr 3e-4; FT: 1,000 steps, lr 1e-4). If still neither is
  competent, **P2 is dropped**: the Holm family is {P1} and the best possible outcome is REVISE.
- PGV_400 runs on every primary parent in every case; the non-selected candidate is reported descriptively.

### 5.4 Fairness constraints

- SF, UVFA and UVFA+aux see the identical (state, weight) label set, trunk, schedule, selection budget and seeds. SF's
  richer supervision (a 57-vector per label) is disclosed and controlled by UVFA+aux.
- PGV_400, FT-UVFA_400 and SF-FT_400 see the identical 400 labels per parent.
- Zero-shot arms use joint renaming augmentation of (s, w); per-goal arms use the stabilizer augmentation σ ∈ Stab(w′)
  (exact and label-preserving; |Stab| logged).
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
  4. the macro D1 > 0 for each matched replicate pair (SF sₖ vs UVFA sₖ, k = 20261246, 47, 48).

**P2, non-inferiority at zero vs 400 new labels (F4).** D2 = nreg(SF-GPI) − nreg(PG400) (macro).
- One-sided p_P2 = max(1/R, share of bootstrap replicates with D2\* ≥ δ_NI), δ_NI = 0.05.
- **P2 passes** iff Holm-adjusted p_P2 < 0.05 (equivalently, at the Holm level the one-sided upper bound of D2 is < 0.05).
- If P2 was dropped at the pilot, the Holm family is {P1}.

Holm: `rcx.stats.holm({"P1": p_P1, "P2": p_P2})`.

### 6.3 Guard G1 and STOP-D (outside Holm)

D_G = nreg(SF-GPI) − nreg(ENUM-MIS±) (macro; positive means the zero-label planner is better).
- **G1 holds** iff the point D_G < 0.05.
- **STOP-D condition** iff the point D_G ≥ 0.05 **and** its 95% CI lower bound > 0.

### 6.4 Estimator, intervals and p-values

- **Unit of independence:** the parent (= the support-class cluster). Starts, bank states, renamed variants and
  replicates are within-cluster repeats.
- **Stratified paired cluster bootstrap**, R = 10,000. Strata: H-ID-S, H-ID-U, H-K, H-NEG (and each secondary family
  for its own tables). Within each stratum, parents are resampled with replacement; family means and the macro are
  recomputed. One index stream per stratum, `numpy.random.Generator(PCG64(derive(20261250, "boot:<stratum>")))`,
  shared by every contrast (common random numbers).
- **CI:** 95% percentile interval (2.5th, 97.5th). The "upper bound" in futility rules is its 97.5th percentile.
- **Two-sided p:** `rcx.stats.bootstrap_p_two_sided(boot, floor=1/R, ties='both')`. With 10,000 replicates, p < 0.05
  implies the 95% interval excludes 0; condition 3 of P1 is still checked.
- **Tolerance.** Thresholds are compared with 1e-9: "x ≥ t" means x ≥ t − 1e-9, "x < t" means x < t − 1e-9, "x > t"
  means x > t + 1e-9.
- `analyze_e05.py` runs **once** per attempt and refuses unless every eval file names the committed `TEST_MANIFEST`
  sha256 and the T3 code digest. Outputs are create-only.

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
- SF-GPI-5; GNN-ENUM±, ENUM-H\*± and ENUM-MIS± on every family; rev 1's positive-only ENUM-MIS on H-NEG (the size of
  the F5 fix); SF-shuffled.
- Model quality on TEST states: endpoint-fact accuracy, T MAE, value MAE and the c/2 value-exact rate, against the
  copy-state floor.
- Binary view: per parent, optimal on ≥ half of the starts, by exact McNemar (`rcx.stats.mcnemar_exact`).

## 7. Decision rule

Tested in this order; the first that holds is the decision.

1. **VOID.** An instrument gate fails on TEST: Q3/Q4/Q8; ORACLE nreg ≠ 0 or STOP ≠ 1 exactly; RAND or GREEDY macro not
   strictly between them; a disjointness, support-uniqueness or exclusion assertion; a manifest or code-digest mismatch.
   Fix the fault, redraw TEST from `20261242:test:<attempt+1>` and log the attempt. Replicates are not retrained; per-goal
   fits are redone. At most two redraws; a third VOID gives INCONCLUSIVE and a design review.
2. **INCONCLUSIVE (missing).** A protected-core cell (§10.3) is missing. Report descriptively; apply no other rule.
3. **STOP-D (dominated).** The STOP-D condition of §6.3 holds. A zero-label symbolic planner beats the learned SF; the SF
   line stops on this testbed. P1/P2 are reported as mechanism findings only.
4. **ADVANCE.** P1 passes, P2 passes (not dropped), and G1 holds.
5. **STOP-M (mechanism misfit).** EXACT-GPI(L) primary macro nreg ≥ 0.5.
6. **STOP-F (futile).** P2 did not pass **and** (macro D1 upper bound < 0.05 **or** H-ID D1 upper bound < 0.05). This
   applies even if P1's significance conditions hold: an interval entirely below 0.05 excludes the planned effect.
7. **REVISE.** P1 or P2 passes. The report names the failed ADVANCE condition, e.g. G1 in the middle zone (D_G ≥ 0.05,
   CI includes 0), "label-cost claim only", or "P1-M passes, P1-ID does not, P2 passes" (follow-up: UVFA-wide/SF-wide
   arms trained on weights covering sign and cardinality).
8. **INCONCLUSIVE.** Otherwise. A positive point with a wide interval is not a pass; a nonsignificant loss is not
   equivalence.

**Qualifiers** (change wording, never the decision):
- **Attribution:** if macro nreg(UVFA+aux) ≤ nreg(SF-GPI) + 0.02, the gain is credited to vector consequence
  supervision, not to test-time recomposition.
- **GPI:** if macro nreg(SF-noGPI) ≤ nreg(SF-GPI) + 0.01, the claim says "goal-conditioned endpoint predictor read out
  linearly", not "GPI".
- **New directions (F8):** "new preference directions" needs H-ID-U D1 point ≥ 0.03 with CI lower bound > 0 **and**
  H-ID-U D2 upper bound < 0.05 (unadjusted). Otherwise: "re-weighting of seen fact patterns".
- **Naming (F11):** "endpoint-and-length successor features (exact MC targets)".

**Pre-declared notes** (stated now so they cannot be reinterpreted):
- **By construction.** UVFA never sees negative weights or K > 4 in training, so part of any H-K/H-NEG advantage is
  extrapolation outside its input support. This is why P1-ID is a conjunct. The predicted H-ID gap is about +0.00 to
  +0.04, so **P1-ID is the binding constraint**.
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
   parents; σ̂₂ = SD over DEV-ID of y_SF(p) − y_PG400(p). Selection-seed models at the selected configs (single models,
   so conservative for 3-replicate means).
2. σ_plan = 1.15·σ̂ (about the 80% upper-confidence factor for an SD from 24–32 parents: 1.16 at 23 df, 1.13 at 31).
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
| G-P1 | Q1–Q8 pass on the pilot oracle |
| G-P2 | DEV-ID EXACT-GPI(L) macro nreg ≥ 0.5 → add the 49 single-fact base policies to L for every learned arm |
| G-P3 (F4) | Competence vs GREEDY (margin 0.20) as in §5.3 → PG400 identity, one extension, or "P2 dropped". Also on 8 DEV-ID parents: PGV_6400 nreg ≤ 0.25 and PGV monotone in k; otherwise PGV training is fixed before the freeze |
| G-P4 (F9) | On the DEV-ID bank pairs (32 × 64), selection-seed models: acc½ = share with \|V̂ − V\*\| < c/2 (V̂ = w̄·ψ̂(s, w) for SF, V_θ for UVFA, w·φ(s) for the copy-state floor). **(a)** acc½(SF) − acc½(copy) ≥ 0.40 **and** top-1(SF-noGPI) − top-1(GREEDY) ≥ 0.20. **(b)** acc½(UVFA) − acc½(copy) ≥ 0.20, else one UVFA extension. If (a) fails: 64 × 800 labels for every arm; if again, n = 6 (all n-dependent counts replaced by their n = 6 values; thresholds unchanged) |
| G-P5 (F6) | Measured steps/s per model size and batch (single and batched B ∈ {16, 64}), VI s per weight, evaluation decisions/s → projected confirmation CPU-min at the §8 sizes; §10.3 cuts applied until ≤ 150 |

### 9.2 PILOT FILL (the only text added at T3; copied verbatim from `PILOT_REPORT.json`)

| Field | Value |
|---|---|
| T2 pilot code digest; test-file shas; `PILOT_MANIFEST.json` sha256 | ⟨FILL⟩ |
| n (7, or 6 by G-P4 fallback); TRAIN labels per goal (400/800) | ⟨FILL⟩ |
| Library composition (64, or 64 + 49) and the G-P2 value | ⟨FILL⟩ |
| Selected epochs for SF, UVFA, UVFA+aux; UVFA extension used? G-P4 values (acc½ deltas, top-1 delta) | ⟨FILL⟩ |
| PGV_400 and FT-UVFA_400 selected configs; extensions used?; DEV-ID nreg of each and of GREEDY; **PG400 = PGV_400 / FT-UVFA_400 / P2 dropped** | ⟨FILL⟩ |
| PGV_6400 nreg and monotonicity on the 8 DEV-ID parents | ⟨FILL⟩ |
| DEV-ID and DEV-OOD macro nreg for every pilot arm; DEV D1, D2, D_G | ⟨FILL⟩ |
| σ̂_ID, σ̂_OOD, σ̂₂; σ_plan values | ⟨FILL⟩ |
| n_ID, n_S (H-ID-S), n_U, n_K = n_NEG; available unseen support classes | ⟨FILL⟩ |
| Planned powers: P1-M, P1-ID, P2 at D2 = 0 and +0.02 | ⟨FILL⟩ |
| G-P5 rates; projected confirmation CPU-min; cut depth (items of §10.3 applied) | ⟨FILL⟩ |
| Exclusions sha (re-pinned if the B4 fresh set was added) | ⟨FILL⟩ |
| T3 full code digest (incl. `analyze_e05.py`); test pass log | ⟨FILL⟩ |

## 10. Execution, lane and safety

### 10.1 Order

1. T1: CX D-entry adopting this draft and DESIGN rev 2.
2. Write the code and tests (no lane). All tests pass; record the T2 digest (`run_e05.py code-digest`) in DECISIONS.
3. `pilot-a` (Phase 1a), `pilot-b` (Phase 1b), `pilot-report` → `PILOT_REPORT.json`.
4. T3: PILOT FILL → `PREREG_E05.md`; record its sha256 and the T3 digest in DECISIONS; commit.
5. T4: `draw-test --expect-prereg-sha <T3> --expect-code-digest <T3>`; commit the `TEST_MANIFEST.json` sha256.
6. `oracle-test` (Q3, Q4, Q7, Q8 on TEST).
7. `train` (replicates 20261246–48 of SF, UVFA, UVFA+aux; SF-shuffled). Replicates may be trained between T3 and T4.
8. `pgv` (PGV_400 on all primary parents, then PGV_1600/6400 on the subset), `ft`, `eval`, `extras` (unless cut).
9. `analyze_e05.py --manifest-sha <T4> --code-digest <T3>`, once.
10. T6: R-entry, DECISIONS entry, FYIs.

### 10.2 Lane rules (binding; CX D-43)

- **Party** `rcx-e05`, via `rcx.lane.try_acquire` only. E05 **never writes a waiting marker** (never calls
  `rcx.lane.acquire`).
- **Start condition**, checked before every phase:
  1. the lane is free (no lock after stale-pid reclaim) and **no `rineya-cpu-lane.waiting.*` marker of any party**
     exists (`waiting.rcx`, `waiting.bwm`, `waiting.dsv3wm`, …);
  2. **available RAM < 11.0 GB** while any E01R adapter is untrained or E01R's SEALED evaluations are pending (E01R
     needs ≥ 12.5 GB, so E05 only uses time E01R cannot);
  3. not inside a **night window, 22:50–07:00 on the nights starting 10-12, 10-13 and 10-14**, and the projected phase
     end (G-P5 rate + 20%) before 22:50. These dates are hard-coded in `run_e05.py`; E05 does not call
     `rcx.lane.window_ok`, which tests the opposite condition;
  4. `rcx.guard`: tz01 < 97 °C (or resumed at ≤ 93 °C) and no `.thermal_pause` file;
  5. no stray python process;
  6. projected phase length ≤ 15 min; `expected_end` = start + 15 min.
- **Safe points** after each training run, each batch of ≤ 64 PGV/FT models and every 8 evaluation parents (≤ about 2
  min apart). At each, the job checkpoints, releases the lane and exits if any `waiting.*` marker exists, 15 min have
  elapsed, a night window is reached, or the guard asks to pause. Outputs are keyed and append-only.
- Torch threads ≤ 4; `OPENBLAS_NUM_THREADS=1` for the oracle; `CUDA_VISIBLE_DEVICES=""` and the torch.cuda patch before
  torch import.
- **Only the coordinator runs phases.** Subagents edit code and run < 5 s checks while any lane job runs.

### 10.3 Budget, protected core and cut order

Planned at 19 GFLOP/s (pessimistic) / 48 (optimistic) from R-04 (FlatMLP 158,721 parameters, 20,724 steps at batch 64
in 26.1–66.1 s): pilot ≈ 16.5 / 7.5 CPU-min (cap 30); confirmation ≈ 128–147 / 60–70 CPU-min (cap 150), in about 10–12
slices. G-P5 replaces these with measured rates.

**Protected core (never cut):** the TEST oracle; SF, UVFA, UVFA+aux × 3 replicates; SF-noGPI, SF-L, SF-GPI-5;
SF-shuffled; **PGV_400 on all primary parents**; PG400 on all primary parents (3 replicates if FT); **PGV_1600 and
PGV_6400 on the 48-parent subset**; ENUM-MIS±, GREEDY, STOP, RAND, ORACLE, EXACT-GPI(L), BEST-SINGLE(L) on every
family; ENUM-H\*± and GNN-ENUM± on H-ID and H-K; the one-step banks.

**Cut order** (at G-P5, until the projection is ≤ 150 CPU-min): 1. budget curve; 2. SF-FT_400; 3. H-KN; 4. GNN-ENUM± on
H-NEG; 5. FT-UVFA_400 to replicate 20261246 only, if FT is not PG400; 6. ENUM-H\*± on H-NEG to the first 24 parents in
manifest order; 7. H-DYN. If the protected core alone exceeds 150, sizes shrink by §8 step 5 before TEST is drawn; if
it still exceeds 150, a CX D-entry raises the cap (time reasons only) or postpones. Nothing in the protected core is
cut after TEST is drawn.

### 10.4 Failures

- A crashed phase resumes from its checkpoint under the same fingerprint; completed keyed outputs are never recomputed.
- A replicate or fit that cannot be completed is **missing**; it is never replaced by another seed. A missing
  protected-core cell gives INCONCLUSIVE (§7 item 2).
- Every confirmation attempt, including VOID redraws, is logged in `runs/RCX-E05/ATTEMPTS.jsonl`. A redraw is never
  called a fresh confirmation of the same bank.

## 11. Freeze points and deviations

| Point | What is fixed | Recorded in |
|---|---|---|
| T0 (now) | Nothing; DESIGN rev 2 and this draft | – |
| T1 adoption | Scope, the dsv3wm boundary, seeds 20261242–50, E04 independence | CX D-entry |
| T2 pilot-code freeze | Digest of `soft.py`, `families_e05.py`, `enum_e05.py`, `learners_e05.py`, `run_e05.py` and the pinned files; tests pass. TRAIN and DEV are drawn by this code | DECISIONS ("E05 pilot code digest") |
| T3 prereg freeze | `PREREG_E05.md` = this draft + PILOT FILL; full digest incl. `analyze_e05.py` | DECISIONS (prereg sha + digest) |
| T4 TEST draw | `TEST_MANIFEST.json` from `20261242:test:1`; sha committed before any arm is evaluated on TEST | DECISIONS |
| T5 confirmation | Replicates, per-goal fits, evaluation, **one** analysis run | `runs/RCX-E05/confirm/…`, ATTEMPTS.jsonl |
| T6 report | R-entry; DECISIONS entry; FYI to BWM, RCWM and (relayed) dsv3wm | – |

- After T2, any change to a T2-digest file before T3 re-runs the affected pilot phase and is logged.
- After T3, any change to a bound file is a logged deviation marked "outcomes known: yes/no".
- After T4, a change that could affect TEST outcomes voids the attempt.
- **Voids the confirmation:** evaluating a selection model on TEST; editing or re-creating a committed manifest;
  retraining a replicate after any TEST evaluation; a second analysis run on the same attempt.

## Appendix A. Seeds (block 20261242–20261250, CX D-43; E04 uses 20261206–14)

| Seed | Purpose |
|---|---|
| 20261242 | Family generation and splits. TRAIN/DEV at the pilot; TEST `20261242:test:<attempt>` after T3; subset `20261242:pgv-subset` |
| 20261243 | TRAIN labelled-state sampling; renaming-augmentation stream; SF-shuffled permutation `20261243:shuffle` |
| 20261244 | Starts and one-step banks (per split); RAND (`20261244:rand`) |
| **20261245** | **Selection only:** pilot model init, per-goal grids on DEV, G-P3/G-P4, σ̂. Never a TEST replicate |
| **20261246, 20261247, 20261248** | **Confirmation replicates** for SF, UVFA, UVFA+aux and FT-UVFA_400 from each; SF-shuffled, budget curve and SF-FT_400 use 20261246 |
| 20261249 | Per-goal label sampling (nested 400 ⊂ 1,600 ⊂ 6,400; shared by PGV, FT, SF-FT) and PGV init, derived per parent and k |
| 20261250 | Analysis bootstraps |

derive(seed, tag) = int.from_bytes(sha256(f"{seed}:{tag}")[:8], "big"). Unit tests use literal seeds 0–9.
