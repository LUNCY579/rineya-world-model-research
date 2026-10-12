# RCX-E05 DESIGN (rev 2): goal-conditioned successor features for zero-shot re-ranking under new goal weights (RVM-001)

**Status:** DESIGN rev 2, 2026-10-12, written after CX D-43. It revises rev 1 (sha256 `94d315246ac7…`, the text
critiqued in `CRITIQUE_RVM_DESIGNS.md`, workflow wf_ad10715d-8d6). It resolves critique items F1–F9 and records F10–F12.
- Nothing is frozen, no E05 code exists, nothing has run, and no outcome has been seen.
- This is read-only work. No model or tokenizer was loaded and nothing was trained. The only code run was trivial
  arithmetic (< 10 s, 1 thread): the power and FLOP tables below.

**Companion:** `PREREG_E05_DRAFT.md`. It freezes after the Phase-1 DEV pilot. The only change allowed at the freeze is
the mechanical PILOT FILL (§11). Where the draft is more specific than this file, the draft binds.

**Details pinned while drafting the prereg (2026-10-12, no outcome exists):** starts are hand-empty (§3); the bootstrap
resamples H-ID-S and H-ID-U as separate strata (§6.4); STOP-F applies even when P1's significance conditions hold
(§6.5); at most two VOID redraws (§6.5); the 48-parent subset takes 8 parents per H-ID stratum (§4.4); the n_ID
lookup table in §6.6 was recomputed (one row was an arithmetic slip; the formula is unchanged).

**Bundle mapping.**
- RCI-001 registry **E05 "Goal-factorized value transfer"** (track RVM-001, P0): "Successor features versus scalar value
  at equal labels"; draft endpoint "Goal-selection regret on held-out reward weights, fixed dynamics".
- RCX roadmap **P0-c RCX-E05**: "Goal-conditioned successor features allow zero-shot re-ranking for new goal weights,
  matching per-goal value training at lower label cost. Endpoint: regret on held-out goal weights."
- **Scope.** RVM successor features are an RCX free lane (COLLABORATION.md §2). Neither BWM nor RCWM has
  successor-feature, goal-weight, UVFA or soft-goal work.
- **Adjacent work (F7).** dsv3wm's RDW-001 **E07/E08** ("consequence factorization versus scalar value") is
  conceptually adjacent. It uses held-out reward weights in the ledger domain with native features
  (`WorldModel/Rineya_Research_Bundle_2026-10-11/execution/audits/E07_E08_review.md`; `execution/STATUS.json`:
  `FACTORIZED_NATIVE_FEATURE_STUDY_PREPARED_DEPENDENCIES_PENDING`).
  - Rev 1's grep missed it because RDW uses different terms.
  - The boundary note was sent on 2026-10-12 (COLLABORATION.md, "about 12:55: RCX scope note for dsv3wm"). There is no
    direct channel, so it was relayed through BWM and RCWM.
  - The note says E05 is an exact Blocksworld oracle at tier B, using policy-dependent SF with GPI. It uses no dsv3wm
    data or code, and its results are offered for cross-reading.

**Tier.** Instrument (exact oracle and floors) plus mechanism (small CPU learners). It is an external specialist
(tier B) by construction. **Nothing here is a Rineya capability claim.**

## Changes from rev 1 (critique item → resolution)

| # | Critique finding | Resolution (where) |
|---|---|---|
| F1 | Seed collision with E04; the selection seed was also a replicate seed | E05 keeps 20261242–50 (CX D-43; E04 moved to 20261206–14). 20261245 is now **selection only**. The replicates are 20261246–48, bootstraps use 20261250, and unit tests use literal seeds (App. A) |
| F2 | Power was computed for a rule other than the one used | The rule is now "Holm-adjusted p < 0.05 (so the CI excludes 0) **and** point ≥ 0.03". Power is computed for that rule at the planned effect 0.05, and a pilot-based resizing rule is fixed now (§6.6) |
| F3 | P1 was partly true by construction | **Chosen:** P1 must also hold on H-ID alone (intersection–union test, no extra Holm cost), reported per family. UVFA-wide/SF-wide arms are deferred, with reasons (§6.2) |
| F4 | The per-goal comparator risked being a strawman | PGV gets stabilizer augmentation and a DEV-tuned grid. A new FT-UVFA_400 arm fine-tunes the UVFA on the same 400 labels. Competence gate: DEV-ID nreg ≤ GREEDY − 0.20. P2's comparator is the better competent arm, fixed before the freeze (§5.2, G-P3) |
| F5 | Guard G1 was weak on H-NEG | ENUM± enumerates negated targets (falsifier menus) and charges penalties that persist. A new STOP-D branch is added (§5.3, §6.5) |
| F6 | Compute was under-costed | A FLOP-based budget from R-04 throughput. Batched multi-model PGV/FT is measured in Phase 1 (G-P5). Protected cut: PGV_400 on all primary parents, PGV_1600/6400 on a fixed random 48-parent subset (§9) |
| F7 | The dsv3wm overlap was missed | Cited above and in §1.2/§7. The boundary note was sent and cross-reading offered |
| F8 | H-ID mostly re-weights seen patterns | One parent per support class in every TEST family, so the support class is the cluster. H-ID is stratified into seen (H-ID-S) and unseen (H-ID-U) supports, and only H-ID-U can carry the "new direction" wording (§4.3, §6.5) |
| F9 | The learnability gate could pass by copying the state | G-P4 gates on value error < c/2 and one-step top-1, each relative to the copy-state floor; for ranking that floor is GREEDY (App. B) |
| F10 | Flat MLPs are weak here; GPI's max adds an upward bias | A learnability readout (SF-L and SF-noGPI vs EXACT-GPI(L)) and the GPI optimism bias become named secondaries. The equivariant SF stays optional and descriptive (§5.5, §6.3) |
| F11 | Claim wording | Every claim says "policy-conditional endpoint-and-length predictor learned from exact Monte Carlo targets, no TD" (§1.3, §3) |
| F12 | No consumer yet | Framed as a mechanism and instrument study. G1/STOP-D and k\* are the deliverables that drive decisions (§1.3, §8) |

---

## 0. Summary

On RCX's exact Blocksworld state space for n = 7 (65,990 states, `rcx/bw/space.py`):
- a goal is a **weight vector w over 56 observable fact features** (on, ontable, clear) plus a step cost c;
- an episode ends with a STOP action that collects w·φ(s_final), and each move costs c;
- every goal-weighted optimal value V\*_w is computed **exactly** by max-plus value iteration over the enumerated graph.

A goal-conditioned successor-feature (SF) network ψθ(s, z) is trained once on exact labels from a library of 64
training goal weights. Here ψ is a policy-conditional endpoint-and-length predictor, learned from exact Monte Carlo
targets. For an unseen weight w′ it re-ranks candidate moves zero-shot by generalized policy improvement (GPI):

  Q(s, a) = max over z ∈ L ∪ {w′} of w̄′·[(0, 1) + ψθ(s′, z)],   with w̄′ = (w′, −c′).

**Primary comparisons:**
- **P1 (equal labels).** SF-GPI against a matched-capacity scalar goal-conditioned value V_θ(s, w) (UVFA) trained on
  the identical labels. SF must win on the macro over H-ID, H-K and H-NEG, **and on in-distribution weights (H-ID)
  alone**.
- **P2 (zero vs 400 new labels).** SF-GPI must be non-inferior to the stronger *competent* per-goal scalar critic
  given 400 fresh exact labels of each held-out goal. That critic is either PGV_400, trained from scratch, or the UVFA
  fine-tuned on those labels.

**Guards.** ENUM-MIS± is a zero-label symbolic enumeration planner that now handles penalties. If it dominates SF-GPI,
the outcome is STOP.

**Budget.** About 16 CPU-min for the pilot (cap 30), plus 60–150 CPU-min for the confirmation, at ≤ 4 threads and
< 1.5 GB RSS. It runs in lane slices of ≤ 15 min that yield to every other party (§9).

**Feasibility: FEASIBLE.** Everything comes from an exact oracle already on disk, and no peer data is needed.

**Prior.** P(ADVANCE) is about 0.15. REVISE is the most likely outcome (§1.5).

---

## 1. Hypothesis, complementarity and tiers

### 1.1 Falsifiable hypothesis

**H-E05.** The domain has fixed dynamics, and preferences are linear in observable fact features. A goal-conditioned
successor-feature critic ψθ(s, z) is trained once on exact labels from 64 training goal weights and composed by GPI.
On held-out goal weights it achieves:
- **(P1)** lower normalized closed-loop regret than a matched-capacity UVFA trained on the *identical* (state, weight)
  label set. This must hold **both** for the macro over the three primary families **and** on H-ID alone. Each needs a
  point estimate ≥ 0.03 and an interval that excludes 0; the planned effect is 0.05;
- **(P2)** regret non-inferior, within a margin δ_NI = 0.05, to the stronger competent per-goal scalar critic trained
  with 400 fresh exact labels *of each held-out goal* (PG400; §5.2).

The SF critic needs **zero** labels per new goal.

**Falsifiers.** Any one of these falsifies the hypothesis:
- **P1 futile:** the upper 95% bound of D1 is < 0.05 on the macro or on H-ID;
- **P2 fails:** the one-sided upper bound of D2 is ≥ δ_NI;
- **dominated:** the zero-label ENUM-MIS± beats SF-GPI by ≥ 0.05, with the interval excluding 0 (STOP-D);
- **mechanism misfit:** the exact (learning-free) GPI ceiling over the library has normalized regret ≥ 0.5 on the
  primary families (STOP-M).

### 1.2 Why this complements Rineya and its world models

| Existing component | What it gives | What it lacks, and what E05 adds |
|---|---|---|
| BWM F (LM forward model) plus `plan_beam` (frozen `18a5f22b`) | One-step imagined transitions, ranked by unmet goal facts for a fixed **hard** goal | No notion of preference weights. A new objective needs a new heuristic or retraining. Every step of lookahead is an LM call, and errors compound |
| RCWM consequence models (T2, T2b) | One-step state deltas under a (fixed) rule | No multi-step, policy-conditional consequences |
| RCX R-04/R-07 GNN value and A3 hybrid (B4 arm a) | Distance to a fixed **hard** conjunctive goal | One value per goal definition. Soft priorities, penalties and trade-offs cannot be represented |
| BWM value LoRA (B4 arm b, BWM-owned) | Model-native h\* for hard goals | The same limit: hard goals only |
| dsv3wm RDW-001 E07/E08 (prepared, dependencies pending) | Native-feature factorized vs scalar readouts on Rineya's **ledger** world, with held-out reward weights | A different domain, tier and object. E05 uses an exact Blocksworld oracle at tier B, with a **policy-dependent** SF and GPI and no native features. The two are adjacent, not duplicates: no data or code is shared, and results are offered for cross-reading (F7) |

**What SF adds.** It gives a **policy-conditional, multi-step consequence summary**: "if I pursue goal z from here, which
facts will hold when I stop, and after how many steps". This summary is **re-weightable at zero cost** for any new
preference vector over the same features.

**Why this matters for Rineya.** An assistant's objectives change per user and task (priorities, things to avoid, the
cost of effort) while the world's dynamics stay fixed, which is exactly the SF regime (P04, Barreto et al.). Per-goal
critics do not scale to that.

### 1.3 What a positive result would and would not establish

**Wording constraint (F11).** In this deterministic, undiscounted episodic setting, ψ^π(s) = (φ(endpoint), length) of a
deterministic policy. It is learned by **supervised regression on exact rollout (Monte Carlo) targets, with no TD
learning**. Every claim says "goal-conditioned endpoint-and-length successor features (exact MC targets)". dsv3wm's own
E08 card warns about exactly this naming.

**Framing (F12).** There is no current consumer:
- BWM goals are hard `on` facts;
- a soft-goal planner would need a new BWM prereg after B4/B5;
- a native-A probe is expected at the floor (R-03).

E05 is therefore a **mechanism and instrument study**. Its **decision-driving deliverables** for the program are:
1. **G1/STOP-D:** whether a zero-label symbolic planner already suffices for soft goals in Blocksworld;
2. **k\*:** the label exchange rate (the per-goal budget a scalar critic needs to match zero-shot SF).

P1 and P2 are the scientific tests.

A positive result (ADVANCE) **would establish, at mechanism tier (tier-B specialist, exact dynamics, n = 7):**
- A learned endpoint-and-length SF critic composed by GPI transfers zero-shot to unseen preference weights inside the
  feature span.
- It beats a scalar goal-conditioned critic trained on the same oracle labels, **including on in-distribution weights**,
  so the gain is not only extrapolation outside the UVFA's input support.
- It matches the best competent per-goal scalar critic that gets 400 fresh labels per new goal.
- It is not dominated by a zero-label symbolic planner.
- It yields a measured label exchange rate k\*.
- The words "new preference directions" are allowed only if the unseen-support stratum H-ID-U supports them (§6.5).
  Otherwise the claim is "re-weighting of seen fact patterns".

It **would not establish:**
- any Rineya native ability;
- transfer across object counts or domains (n is fixed at 7);
- robustness to LM-imagined states or LM-proposed candidates;
- value for **hard conjunctive goals** (H-CONJ, outside the linear span) or under changed dynamics (H-DYN);
- that BWM's beam or any LM planner gains;
- the bundle's "+5 pp goal success at matched compute" target (an E06 and native-path claim, §8).

### 1.4 Tier ladder for this line of work

| Rung | E05 content | Claim allowed |
|---|---|---|
| **Instrument** | Exact oracle (soft-goal value iteration, policy SF tables, GPI) qualified against brute force and against `space.py` h\*; floors separated from the oracle; exact-GPI library ceiling | "The battery can separate SF/GPI from its shortcuts" |
| **Mechanism (this experiment)** | CPU learners (MLPs of about 150k–180k parameters) on exact labels; the frozen R-04 GNN as a baseline | "A small external specialist does or does not transfer to new goal weights" (tier B) |
| **Native-A** (follow-up, §8) | Frozen Titan-Sparse probed, inference only, on SF-style questions | "Rineya already has or lacks the capability" |
| **Native-C** (follow-up, §8) | ψ-critic LoRA on frozen FP32 Titan-Sparse (unmerged), A/B/C/D tiers, E33 retention | "Internalized component", still not dense |

### 1.5 Prespecified directional predictions (copied into the prereg; never back-fitted)

1. **Exact-GPI ceiling.** It is low on H-ID (≤ 0.15) and higher on H-K and H-NEG.
2. **SF-GPI vs UVFA.**
   - The advantage is largest on H-K and H-NEG. Part of it is by construction: those weights are outside the UVFA's
     training input support (disclosed per family).
   - On H-ID it is near parity, about +0.00 to +0.04. **The H-ID conjunct is therefore the binding constraint on P1.**
3. **Attribution.** UVFA+aux closes part of the gap on H-ID.
4. **Guard.** ENUM-MIS± is within ±0.05 of SF-GPI on H-ID and H-K, and much better than rev 1's positive-only ENUM-MIS
   on H-NEG.
5. **P2.** FT-UVFA_400 is the likelier P2 comparator, and the stronger one.
6. **Assumption boundary.** On H-CONJ and H-DYN, SF-GPI degrades to ENUM-MIS± level or worse. ENUM-H\* and GNN-ENUM do
   well on H-CONJ.
7. **Confidence** (noted, never used):

   | Event | Probability |
   |---|---|
   | P1, including the H-ID conjunct | about 0.25–0.3 |
   | P2 | about 0.5 |
   | STOP-D | about 0.25 |
   | ADVANCE | about 0.15 |
   | REVISE | about 0.4 |
   | Any STOP | about 0.35 |
   | INCONCLUSIVE | about 0.1 |

---

## 2. Choice of testbed (unchanged from rev 1)

| Option | Exact goal-weighted values? | Fixed dynamics, varying goals? | Peer overlap | Verdict |
|---|---|---|---|---|
| **A. RCX exact Blocksworld space, n = 7** (`rcx/bw/space.py`) | Yes: max-plus value iteration over all 65,990 states and 186,578 edges | Yes: one MDP, and only w varies | None with BWM or RCWM. Adjacent to RDW E07/E08 (a different domain) | **Chosen** |
| A′. Same, n = 6 (7,057 states) | Yes | Yes | None | Unit tests. Fallback only on an engineering failure (G-P4/G-P5) |
| A″. n = 8 (695,417 states) | Yes | A new MDP | None | Deferred to a relational, size-agnostic SF |
| B. RCWM-style ledger decisions | Partly | Weakly | RCWM fixed-rule learners; RDW E07/E08 | Rejected |
| C. Factory RVM data | No: one goal per item | No | Factory-owned | Rejected |
| D. Larger-n A\* labels | Not cheaply for soft goals | Yes | None | Rejected |

**Why A.**
- Every quantity in the claim is exact.
- The domain is BWM's and Rineya's planning domain.
- Blocksworld is the canonical domain of non-serializable subgoals (the Sussman anomaly), which is GPI's known weakness.
  The test is not rigged in SF's favour.

---

## 3. Task definition (exact)

**State space.** Blocksworld with n = 7 under BWM 4-op semantics (`space.successors`, verified equal to `B.step` in
`tests/test_bw.py`):
- |S| = 65,990, of which 37,633 are hand-empty and 28,357 are holding states;
- 186,578 move edges, so on average 2.83 legal moves per state, plus STOP.

Block ids are 0–6; names are attached only for BWM-format calls.

**Features.** φ(s) ∈ {0,1}^56: `on(i,j)` (42), `ontable(i)` (7) and `clear(i)` (7), as derived by `space.to_facts`.
A held block has none of these facts. So STOP in a holding state hides that block's facts. This is a legitimate feature
of the MDP: every arm and the oracle see the same MDP, and the ENUM± menus include `holding` targets (§5.3).

**Reward and termination (episodic, undiscounted).**
- Actions are the legal moves (cost c each) plus **STOP**, which is always legal and collects w·φ(s).
- The goal vector is w̄ = (w, −c). Training goals use c = 0.25.
- Weights are small integers and c ∈ {0.125, 0.25, 0.75} is dyadic, so every value is an exact float64 dyadic rational
  and ties are exact.
- V\*_w(s) = max( w·φ(s), max over legal s′ of [ −c + V\*_w(s′) ] ).
- It is solved by max-plus value iteration over the forward CSR graph, initialized at the STOP values. It converges in
  at most diameter + 1 sweeps, because c > 0 and there are no dead ends.

**Policies and successor features.**
- π_w is greedy on V\*_w. Ties are broken deterministically: STOP first, then the lowest action code.
- ψ^π(s) = (φ(s_end^π(s)), T^π(s)). For a move a to s′: ψ^π(s, a) = (0, 1) + ψ^π(s′).
- Q^π_w(s, a) = w̄·ψ^π(s, a), and V\*_w = w̄·ψ^{π_w} exactly.
- **Renaming and ties.** Values are exactly renaming-invariant. The tie rule depends on block ids, so on tie states
  σψ^{π_w}(s) is the SF of the σ-conjugate optimal policy for σw, not of π_{σw}.
  - Under renaming augmentation, ψ̂ therefore learns the SF of a stochastic mixture of optimal policies for z.
  - That is still a valid GPI input, and w̄_z·ψ is unchanged. Q5 tests both facts.

**GPI.** For a test weight w′ and a policy set C:

  π_GPI(s) = argmax over {STOP} ∪ moves of max over z ∈ C of Q^{π_z}_{w′}(s, a),   with Q(s, STOP) = w′·φ(s).

With exact ψ, Barreto's theorem gives Q^{π_GPI} ≥ max over z of Q^{π_z}.

**Decision protocol (identical for every arm).**
- The candidates are STOP plus all legal moves, so hard legality comes first. Each arm scores the candidates and the
  argmax is taken, with the tie rule above.
- Every arm scores STOP exactly as w′·φ(s). Only continuation values are learned.
- Closed loop: from start s0, act until STOP, H_max = 40 moves, or a revisited state. A revisit forces STOP at the
  current state and is logged as a loop.

**Metrics.**
- Return: J = w′·φ(s_final) − c·T.
- **Normalized regret:** nreg(s0, w′) = clip( (V\*_{w′}(s0) − J) / (V\*_{w′}(s0) − w′·φ(s0)), 0, 1 ).
  - It is 0 when optimal and 1 when no better than stopping immediately.
  - Starts need a gap V\* − w′·φ(s0) ≥ 1.0, which is at least 4 steps of cost at c = 0.25. This filter comes from the
    oracle and is outcome-free.
  - Starts are **hand-empty** states (BWM instances start with an empty hand), drawn uniformly without replacement from
    the qualifying, non-excluded hand-empty states; 16 per parent. A parent with fewer than 16 is rejected and logged.
    H-DYN starts must also satisfy the height limit. (Pinned for the prereg draft, 2026-10-12.)
- **One-step re-ranking** (secondary): regret1(s) = (max over a of Q\*(s, a) − Q\*(s, â)) / c, plus the top-1
  optimal-action rate.
  - It uses a fixed bank of 64 states per parent, drawn uniformly from the states where STOP is not optimal.
  - Every arm faces identical candidate banks.

---

## 4. Data

### 4.1 Sources (all on disk, read-only, sha-pinned at first import)

| Source | Path (relative to `complementary/`) | Use |
|---|---|---|
| Exact state space, h\*, misplaced bound | `rcx/bw/space.py` (sha `29d0b420…`, = HSTAR recipe and B4 pin) | State enumeration, CSR graph, `CompiledGoal` (on/ontable/clear/holding), `goal_mask`, `backward_bfs`, `misplaced_compact` |
| BWM simulator and signatures | via `rcx/bw/bwm_import.py` → BWM `native_v1/bwmn/blocks.py` (read-only, `dont_write_bytecode`) | Exclusion invariants and signatures only |
| Exclusions | `experiments/RCX-E07/EXCLUSIONS.json` (sha `2f211de3…`) plus `rcx/bw/excl.py` (`eb31d9eb…`: `row_exclusion_reason`, `exclusion_reason`, `with_selection_sets`, `load_exclusions`) | Leakage control (§4.5) |
| R-04 GNN value (baseline only) | `runs/RCX-E07/main/models/gnn_s20261276.pt` (`9cc417a3…`), `rcx/bw/models.py` (`4c2e3242…`), `rcx/bw/encode.py` (`32980f14…`) | GNN-ENUM± baseline, frozen, inference only |
| Statistics, lane, guard | `rcx/stats.py` (`06a30c10…`), `rcx/lane.py` (`03f93d59…`), `rcx/guard.py` (`08fa782d…`) | `bootstrap_p_two_sided`, `holm`, `mcnemar_exact`; lane lock; thermal and RAM guard |

**No dataset file is consumed.** All E05 data are generated by the exact oracle at run time from the seeds in App. A.

### 4.2 How labels are produced (oracle, never LM judgement)

**Per training weight w_i:**
- one value-iteration solve gives V\*_{w_i} over all states;
- following π_{w_i} with pointer doubling gives endpoints e_i(s) and lengths T_i(s);
- so ψ^{π_{w_i}}(s) = (φ(states[e_i(s)]), T_i(s)) exactly.

**What a label is.** A label is one (state, weight) oracle query.
- The SF target is the 57-vector ψ (an exact Monte Carlo endpoint-and-length target).
- The UVFA target is V\* = w̄·ψ.
- Both come from the same oracle pass, so their oracle cost is equal. SF's richer supervision is disclosed and
  controlled by UVFA+aux.

**Per-goal labels (PGV and FT).**
- For each held-out parent, **nested** uniform state samples (400 ⊂ 1,600 ⊂ 6,400) come from a derived stream of
  20261249.
- PGV_400, FT-UVFA_400 and SF-FT_400 use **the same 400 labels**.

**Exact policy tables.** The library tables (int32 endpoints and int16 lengths, 64 × 65,990, about 25 MB) serve the
exact-GPI ceiling only. They are an oracle-information reference, not a competitor.

**Modified oracles:**
- H-CONJ: terminal reward R·1[G ⊆ s];
- H-DYN: height-limit-3 dynamics, solved on the masked edge set.

### 4.3 Goal-weight families, classes and support classes

**Fact universe.** F_on (42), F_tab (7), F_clr (7).

**Keys (all canonical under block renaming S_7: the minimum over the 5,040 permutations, vectorized).**
- **Parent key (weighted class):** (w, c, reward type). Renamed copies are the same task.
- **Support class (F8):** the unweighted fact set supp(w). For families with negative weights it is **signed**: the set
  of (fact, sign w_f). For H-CONJ it is the class of G.
- **Seen support:** the support class of at least one library goal in L.

**One parent per support class (F8).** Every TEST family contains at most one parent per (signed) support class. This
is asserted at generation.
- The support class is therefore the unit of independence. Re-weighted copies of one fact pattern can never appear as
  separate clusters.
- Weighted classes stay disjoint across all splits.

| Family | Role | Generator |
|---|---|---|
| **TRAIN (library L)** | training | K ~ {1: 0.1, 2: 0.3, 3: 0.3, 4: 0.3} facts from F_on ∪ F_tab (P(on) = 0.75), weights iid {1, 2, 3}, c = 0.25. Conflicting facts allowed and stratified. Repeated support classes allowed |
| DEV-ID | tuning, gates, σ̂ | As TRAIN with K ∈ {2, 3, 4}: 16 parents with seen supports, 16 with unseen supports |
| DEV-OOD | σ̂ and pipeline check only | 12 parents each of H-K, H-NEG, H-COST, H-SPAN |
| **H-ID** (primary) | goal split | K uniform on {2, 3, 4}, otherwise TRAIN-like. **H-ID-S:** support class ∈ supports(L), min(40, available) parents. **H-ID-U:** support class ∉ supports(L) ∪ supports(DEV-ID); the rest of n_ID. Rejection counts and the realized K mix are logged. K = 2 supports are nearly exhausted by L (only 9 two-fact on/ontable support classes exist up to renaming), so H-ID-U is mostly K = 3–4. This K difference is disclosed and stratified |
| **H-K** (primary) | composition | K ∈ {5, 6}, weights {1, 2, 3}, c = 0.25 |
| **H-NEG** (primary) | sign | K ∈ {2, 3, 4}. One negative fact for K = 2, one or two for K ≥ 3, weights {−1, −2}, at least one positive. Facts from F_on ∪ F_tab |
| H-COST (secondary) | numerical | TRAIN-like (K 2–4) with c ∈ {0.125, 0.75} |
| H-SPAN (secondary) | outside the training span | TRAIN-like (K 1–3) plus 1–2 `clear` facts with weights {1, 2} |
| H-KN (secondary) | combined shift | K ∈ {5, 6} with 1–2 negatives |
| H-CONJ (secondary, assumption boundary) | nonlinear reward | Hard goal G of 2–3 satisfiable `on` facts, terminal reward 3·1[G ⊆ s], c = 0.25. SF and UVFA receive the pre-declared linearization w = (3/\|G\|)·1_G |
| H-DYN (secondary, assumption boundary) | changed dynamics | TRAIN-like weights under the height-limit-3 rule. Learned arms are trained on the unrestricted dynamics; candidates are the restricted legal moves |

### 4.4 Sizes and splits

| Set | Parents | Per parent | Notes |
|---|---|---|---|
| TRAIN library | 64 (+49 single-fact base policies only if G-P2 triggers; then every learned arm gets their labels too) | 400 labelled states (uniform over S) | 25,600 labels |
| DEV-ID | 32 (16 seen + 16 unseen support) | 16 starts, 64 one-step states, 400 per-goal labels; 1,600/6,400 labels on 8 parents | selection, G-P3, G-P4, σ̂ |
| DEV-OOD | 48 | 16 starts | σ̂_OOD (H-K and H-NEG parents), pipeline check |
| **TEST primary** | **H-ID n_ID ∈ [96, 152]; H-K and H-NEG n_K = n_NEG ∈ {56, 72}** (§6.6), so n_P ∈ [208, 296] | 16 starts, 64 one-step states | base plan 96 + 56 + 56 = 208 |
| TEST secondary | H-COST 40, H-SPAN 40, H-KN 24, H-CONJ 40, H-DYN 40 (184) | 16 starts, 64 one-step states | reported per family, never pooled with the primary |
| Per-goal labels | PGV_400 and FT_400: **all** primary TEST parents. PGV_1600 and PGV_6400: a **fixed random subset of 48** primary parents (16 per family; within H-ID, 8 per stratum), drawn with TEST and listed in its manifest | 400 / 1,600 / 6,400 (nested) | label accounting in §5.5 |

**Allocation order.**
- TRAIN and DEV are drawn at the pilot (Phase 1), by the pilot-frozen code.
- TEST is drawn **after the prereg freeze** by the frozen code, from the derived stream `20261242:test:<attempt>`.
- `TEST_MANIFEST.json` lists parents, starts, banks, the 48-parent subset and the label index sets. Its sha256 is
  committed to DECISIONS.md before any arm is evaluated on TEST. The oracle runs used to draw TEST are the only
  exception.
- Rejection counts are logged: duplicate class, duplicate support class, exclusion, too few qualifying starts,
  unsatisfiable, and seen/unseen quota.

### 4.5 Leakage and OOD controls

1. **Parent and support disjointness.**
   - Weighted classes are disjoint across TRAIN, DEV-ID, DEV-OOD and every TEST family (asserted).
   - Within each TEST family, support classes are unique (asserted).
   - The H-ID-U supports are disjoint from supports(L) ∪ supports(DEV-ID) (asserted).
   - H-ID-S re-weights supports seen in training. It is reported separately and can never carry the "new direction"
     wording (F8).
2. **States are shared by design.** Up to renaming, n = 7 has only 26 state orbits, so holding out states is
   meaningless. The unit of novelty is the (state, weight) pair.
   - The zero-shot learned arms get renaming augmentation: a joint random σ on s and w.
   - The per-goal arms (PGV, FT, SF-FT) get the **stabilizer augmentation**: σ ∈ Stab(w′) = {σ : σw′ = w′}. It is
     free, exact and label-preserving (F4). Stab(w′) is enumerated over the 5,040 permutations at parent creation, and
     its size is logged (it is 1 for asymmetric goals).
3. **Peer exclusions (read-only, through the existing file; never evaluated on).**
   - Every TRAIN label row and every DEV/TEST start is read as a BWM instance (state, goal⁺), where goal⁺ is the set of
     positive `on`/`ontable` facts. Each is checked with `excl.row_exclusion_reason` on
     `with_selection_sets(load_exclusions())`. Every H-CONJ (start, G) is checked with `excl.exclusion_reason`.
   - This covers 677 invariants and 349 signatures. At n = 7, 35 instances are reserved: hard110 1, val_large 2,
     mirror 1, b3fresh 1 and distill 30.
   - If BWM's full B4 fresh set (seed 20261410) exists by the freeze, it is added through `excl.EXTRA_RESERVED_FILES`
     and the sha is re-pinned.
   - Matches are dropped and counted, never silently.
4. **RCX's own evaluation sets** (E07 test n ∈ {5, 6, 8, 10, 12}; E07b TEST-B n 10–15; E11a/b n 10–15) contain no n = 7
   instance. This is asserted by block count.
5. **Never opened:**
   - RCWM sealed instruments;
   - PlanBench Logistics;
   - BWM `distill_rows.jsonl` and `mix_*.jsonl`;
   - any dsv3wm data or code (RDW E07/E08 included);
   - factory data;
   - any `*.secret` file.
6. **R-04 GNN training distribution:** n = 4–7 hard on-only goal classes. `ontable`, `clear` and `holding` goal facts
   are out of distribution for it (disclosed). GNN-ENUM± on H-CONJ is split by whether G's class was in its TRAIN
   classes.
7. **OOD matrix.**

   | Component | Family |
   |---|---|
   | Goal split | H-ID (S and U strata) |
   | Composition | H-K |
   | Sign | H-NEG |
   | Numerical | H-COST |
   | Outside span | H-SPAN |
   | Combined | H-KN |
   | Nonlinear reward | H-CONJ |
   | Changed dynamics | H-DYN |

   Only H-ID, H-K and H-NEG form the primary composite.

---

## 5. Models and baselines

All learned models are CPU MLPs with ≤ 200k parameters, run with ≤ 4 torch threads and < 1 GB RSS.

**Inputs.**
- State x_s (70 dims): the on-matrix (49), ontable (7), holding (7) and clear (7).
- Goal or policy code z (57 dims): fact weights / 3, plus c.

**Trunk.** 3 × 256 with LayerNorm and ReLU, AdamW (lr 1e-3, wd 1e-4, constant), batch 256.

| Model | Parameters |
|---|---|
| SF | 180,537 |
| UVFA | 166,145 |
| UVFA+aux | 180,794 |
| PGV width 256 | 151,553 |
| PGV width 64 | 13,313 |

### 5.1 Proposed method

**SF-GPI** (the USFA form):
- Network: ψθ(x_s, z) → (56 sigmoid endpoint-fact probabilities, softplus T̂).
- Loss: BCE on facts plus Huber on T/10.
- At test, each successor s′ is scored as max over z ∈ C of [ w′·ψ̂_φ(s′, z) − c′·(1 + T̂(s′, z)) ], with
  C = L ∪ {w′} (65 codes).

### 5.2 Learned comparators and attribution controls

| Arm | Definition | Role |
|---|---|---|
| **UVFA** | V_θ(x_s, w) → scalar V\*; Huber loss; identical labels, trunk and schedule | **P1 comparator** |
| UVFA+aux | Shared trunk with a scalar head (used at test) and a ψ head (auxiliary, λ = 1) | Separates richer supervision from linear recomposition |
| SF-noGPI | C = {w′} only | Role of GPI |
| SF-L | C = L only | Role of extrapolating ψ to the new policy index |
| SF-GPI-5 | C = {w′} plus the 4 nearest library codes (cosine) | Compute-matched view |
| SF-shuffled | SF trained with ψ targets permuted across states within each weight (replicate 20261246) | Shuffled-label control |
| **PGV_400** | Per-goal V_θ(x_s) with no goal input, trained from scratch for each held-out parent on its 400 labels. Uses stabilizer augmentation and early stopping on 40 of the 400 labels (checked every 100 steps; the best is kept). **Grid, tuned once on DEV-ID, never per goal:** width {64, 256} × steps {500, 2,000} at batch 64. The 500-step setting is the 500-step checkpoint of the 2,000-step run | P2 candidate |
| **FT-UVFA_400** (new, F4) | Each UVFA replicate (20261246–48), fine-tuned on the same 400 labels with the goal input fixed to w̄′. All parameters are trained, with AdamW lr 3e-4, batch 64 and stabilizer augmentation. **Grid:** steps {100, 400} (the 100-step setting is a checkpoint). Early stopping on the same 40 labels. Three fits per parent, averaged within the parent | P2 candidate: the realistic competitor ("a scalar critic adapted with 400 labels") |
| PGV_1600, PGV_6400 | Width 256; batch 128 / 256; 2,000 / 2,500 steps; stabilizer augmentation; early stopping on 10%. Not tuned | Label-cost curve and k\* (48-parent subset) |
| SF-FT_400 (descriptive) | SF replicate 20261246 fine-tuned on the ψ targets of the same 400 labels (FT-UVFA's selected step count) | How much SF gains from labels. Cut-list item 2 |
| GNN-ENUM± | Frozen R-04 RelGNN `gnn_s20261276` as ĥ(s′, G) inside the ENUM± target enumeration (§5.3). Zero new labels | The existing RCX critic, used zero-shot |

**P2 comparator PG400 (fixed at the pilot, before the freeze; G-P3).**
- A candidate is *competent* if its DEV-ID macro nreg ≤ nreg(GREEDY) − 0.20.
  - The 0.20 margin is pre-set. GREEDY is expected to stop early on most starts with a gap ≥ 1.0, so a smaller margin
    would accept near-floor critics.
- PG400 is the competent candidate with the lower DEV-ID nreg; ties go to FT-UVFA_400.
- If neither is competent, each grid is extended **once**:
  - PGV: width 128, 4,000 steps, lr 3e-4;
  - FT: 1,000 steps, lr 1e-4.
- If still neither is competent, **P2 is dropped**. The Holm family is then {P1}, ADVANCE is unavailable and the best
  possible outcome is REVISE. This is recorded before the freeze.
- PGV_400 runs on all primary parents in every case. The non-selected candidate is reported descriptively.

**UVFA-wide and SF-wide arms (F3 alternative): not adopted.**
- They would change the training-weight distribution, the library and the exact-GPI ceiling at once.
- They would cost about 15–20 more pessimistic CPU-min (second library, second selection, 6 runs) in a lane-starved
  week.
- They answer a different question: in-support sample efficiency over a wider weight distribution.
- They are pre-registered as the **follow-up** if P1's macro conjunct passes and its H-ID conjunct does not (§6.5,
  REVISE wording).

### 5.3 Non-learned baselines and oracle references (ENUM± fixes F5)

**ENUM± target enumeration (shared by ENUM-MIS±, ENUM-H\*± and GNN-ENUM±).** Let w′ have positive facts P and
negative facts N.

**Falsifier menus (fixed).**

| Negative fact | Menu M(f) |
|---|---|
| on(a,b) | {ontable(a), holding(a), clear(b)} |
| ontable(a) | {holding(a)} ∪ {on(a,x) : x ≠ a} |
| clear(a) | {holding(a)} ∪ {on(x,a) : x ≠ a} |

**Targets.** G = G⁺ ∪ R, where:
- G⁺ ⊆ P;
- for each f ∈ N, R picks either "accept" or one falsifier from M(f);
- G is kept only if `CompiledGoal(G).consistent` and `goal_mask(G).any()`, an exact satisfiability check computed once
  per parent.

For positive-only goals (H-ID, H-K), ENUM± equals rev 1's ENUM. Target counts per parent:

| Family | Targets |
|---|---|
| H-NEG | ≤ 2³ × 8² = 512 |
| H-K | ≤ 2⁶ = 64 |
| H-KN | ≤ 4,096 |

**Predicted terminal utility.**

  U(G; s′) = Σ_{f∈G⁺} w_f + Σ_{f∈N} w_f · 1[ f ∈ φ(s′) and G ∪ {f} is satisfiable ].

Penalties that hold now and *can persist* under G are charged. Those that G necessarily falsifies are not. Facts that
are false now and might become true are ignored; this is optimistic, and disclosed.

**Score.**
- score(s′) = max over G of [ U(G; s′) − c′·(1 + ĥ(s′, G)) ].
- STOP is scored exactly as w′·φ(s), with the common tie rule.

| Arm | Definition | Information tier |
|---|---|---|
| **ENUM-MIS±** | ĥ = admissible `misplaced_compact` lower bound. It falls back to 0 for `holding` targets, per `space.py` | Symbolic only, zero labels. **Guard G1 and STOP-D** |
| ENUM-H\*± | ĥ = exact h\*(·, G) by one `backward_bfs` per target | Oracle-information reference |
| GNN-ENUM± | ĥ = R-04 GNN (§5.2). Goal facts other than `on` are out of distribution for it | Learned, zero new labels |
| GREEDY | One-step lookahead on w′·φ(s′) − c′, plus the STOP rule. It is also the copy-state floor for ranking: ψ_copy(s′) = (φ(s′), 0) | Floor |
| STOP | Always stop (nreg = 1) | Constant floor |
| RAND | Uniform over {STOP} ∪ moves (stream `20261244:rand`) | Random floor |
| EXACT-GPI(L) | GPI over the exact library policy tables | Mechanism ceiling |
| BEST-SINGLE(L) | The single library policy with the best exact value at s0 | Policy-reuse reference |
| ORACLE | π\*_{w′} | Regret 0 |

### 5.4 Coverage of the RVM-001 required controls

| Required control | Arm(s) used here |
|---|---|
| Random and constant scorers | RAND; STOP |
| Deterministic validator ceiling | ORACLE; EXACT-GPI(L) |
| Frozen confidence | N/A: no LM in the loop (recorded) |
| Scalar head with matched capacity | UVFA |
| Consequence features at equal labels | UVFA+aux |
| Identical candidate banks | The one-step bank and identical starts |
| Shuffled-label critic | SF-shuffled |

Two further card rules also hold:
- **"Do not give only the learned critic the goal":** every arm receives w′.
- **"Apply hard legality first":** the candidates are exact legal moves plus STOP.

**E04 dependency.** E05 never ranks illegal candidates, so it is independent of E04 (CX D-43).

### 5.5 Training, tuning, seeds and accounting

**Zero-shot arms (SF, UVFA, UVFA+aux).**
- Grid: epochs {20, 40} at lr 1e-3. The 20-epoch config is the 20-epoch checkpoint of the 40-epoch run, so each arm needs
  one run.
- Selection is per arm, on DEV-ID closed-loop nreg, with **selection seed 20261245 only**.
- Confirmation trains **fresh replicates 20261246, 20261247 and 20261248** at the selected config. The selection models
  are never evaluated on TEST.
- **UVFA competence (G-P4b).** If the UVFA fails it, its grid is extended once with {80 epochs, lr 3e-4} before the
  freeze, so P1's comparator is not a strawman.

**Per-goal arms.** Their grids are tuned on DEV-ID with selection seed 20261245 and the DEV label streams (§5.2, G-P3).
PGV is never tuned per goal.

**Library composition.** 64 training weights. The 49 single-fact base policies are added only if the DEV-ID exact-GPI
ceiling is ≥ 0.5 (G-P2), and then for every learned arm.

**Batched multi-model training (F6).**
- PGV and FT fits run as B ≤ 64 independent models stacked in one `BatchedMLP`: per-model weights [B, in, out] through
  `torch.baddbmm`, per-model LayerNorm affine parameters, and AdamW over the stacked tensors.
- AdamW is elementwise, so this equals B independent optimizers. There is no global gradient clipping.
- Each model has its own early-stopping snapshot.
- Equivalence to single-model training is a unit test.

**Label accounting** (logged per arm, together with supervised numbers, optimizer steps, parameters and wall time):

| Arm | One-off labels | Labels per new goal | Primary TEST total (n_P = 208 / 296) |
|---|---|---|---|
| SF, UVFA, UVFA+aux | 25,600 | 0 | 0 |
| PGV_400 and FT-UVFA_400 (one shared label set) | – / 25,600 | 400 | 83,200 / 118,400 |
| PGV_1600 (48-parent subset) | – | 1,600 | 76,800 |
| PGV_6400 (48-parent subset) | – | 6,400 | 307,200 |

Break-even: SF's one-off budget equals 400 labels on each of 64 new goals.

**Inference accounting.** Network calls per decision, misplaced or h\* calls, targets per parent, and wall time per
decision.

**Descriptive extras** (cut first):
- the budget curve (1 replicate): SF-GPI and UVFA at 16 × 400 and 64 × 1,600 labels;
- SF-FT_400;
- the optional permutation-equivariant SF (F10; only if lane time remains, 1 replicate): an R-04 RelGNN trunk with
  per-fact readouts.

---

## 6. Endpoints, statistics and decision rule

### 6.1 Primary endpoint

**The macro-average over H-ID, H-K and H-NEG of the family mean of parent-level normalized closed-loop regret.**
- A parent's value for one model is the mean over its 16 fixed starts.
- For learned zero-shot arms and FT, a parent's value is the mean over the 3 replicates. PGV is one fit per parent;
  deterministic arms need no seeds.
- A family mean is the mean over its parents, which are its clusters, since there is one parent per support class.
- Lower is better.

### 6.2 Primary comparison family (Holm, family-wise α = 0.05)

**P1, superiority at equal labels (F2, F3).**
- D1 = nreg(UVFA) − nreg(SF-GPI); positive favours SF.
- It is tested as an intersection–union of two conjuncts:
  - **P1-M:** the macro D1;
  - **P1-ID:** D1 on H-ID alone (H-ID-S and H-ID-U pooled).
- Each conjunct gets a two-sided bootstrap p (§6.4). p_P1 = max(p_M, p_ID). An intersection–union test at level α
  needs no adjustment for its conjuncts, so the H-ID requirement costs no Holm budget.
- **P1 passes** if all of the following hold:
  - the Holm-adjusted p_P1 < 0.05;
  - both points are ≥ 0.03;
  - both 95% CI lower bounds are > 0. With ties = 'both' and 10,000 resamples this follows from p < 0.05, and it is
    checked anyway;
  - the macro D1 is > 0 for each matched replicate pair (SF sₖ vs UVFA sₖ).
- **Why the 0.03 floor.** The planned effect stays 0.05 and is what power is computed for. The 0.03 floor stops a
  significant but trivial effect from passing.
- Per-family D1 is always reported. H-K and H-NEG are labelled "includes extrapolation outside UVFA's training weight
  support".

**P2, non-inferiority at zero vs 400 new labels (F4).**
- D2 = nreg(SF-GPI) − nreg(PG400).
- p_P2 = max(1/R, share of bootstrap replicates with D2\* ≥ δ_NI), with δ_NI = 0.05 and a one-sided test.
- **P2 passes** if the Holm-adjusted p_P2 < 0.05. Equivalently, at the Holm level, the one-sided upper bound of D2 is
  < 0.05.
- If P2 was dropped at the pilot, the Holm family is {P1}.

**Guard G1 (not in Holm; point condition).**
- D_G = nreg(SF-GPI) − nreg(ENUM-MIS±) (macro).
- G1 holds if D_G < +0.05.

### 6.3 Secondary (descriptive, prespecified, no decision weight)

- **Results by family and stratum:**
  - per-family and per-replicate results;
  - the H-ID-S and H-ID-U strata;
  - K strata and conflict strata.
- **Ranking and loops:** one-step re-ranking regret and top-1 rate on the identical bank; loop rate.
- **k\*:** the smallest per-goal budget at which PGV matches SF-GPI, by log-k interpolation on the 48-parent subset,
  with a bootstrap CI and censoring at > 6,400. The same is computed for FT-UVFA where defined.
- **Attribution:**
  - SF-GPI vs UVFA+aux;
  - SF-noGPI and SF-L vs SF-GPI;
  - the **learnability readout (F10):** SF-L vs EXACT-GPI(L) and SF-noGPI vs ORACLE;
  - the **GPI optimism bias (F10):** on the bank, the mean of max_z ŵ·ψ̂ minus the exact Q of the chosen action, by
    |C| ∈ {1, 5, 65}.
- **Other arms:** the compute-matched SF-GPI-5; GNN-ENUM±, ENUM-MIS± and ENUM-H\*± on every family; and rev 1's
  positive-only ENUM-MIS on H-NEG, to show the size of the F5 fix.
- **Secondary families:** H-SPAN, H-COST, H-KN, H-CONJ and H-DYN, labelled "assumption boundary" where predicted.
- **Model quality on TEST states:** endpoint-fact accuracy, T MAE, value MAE and the c/2 value-exact rate for every
  learned arm, against the copy-state floor.
- **Binary view:** per parent, whether the arm is optimal on at least half of the starts, compared by exact McNemar.

### 6.4 Statistics

- **Unit of independence: the parent.** Each TEST family has one parent per (signed) support class, so a parent is a
  support-class cluster (F8). Starts, one-step states, renamed variants and replicates are within-cluster repeats.
- **Bootstrap.**
  - It is a **stratified paired cluster bootstrap**: within each stratum, parents are resampled with replacement, and
    the family means and macro are recomputed. There are R = 10,000 resamples. The strata are the families, except
    that H-ID is resampled within H-ID-S and H-ID-U separately (their sizes are fixed by design); the H-ID mean is
    the parent-weighted mean of the two.
  - The seed is **20261250**. One index stream per family is shared by every contrast (common random numbers).
  - The CI is the 95% percentile interval.
  - It is implemented in `analyze_e05.py`, because `rcx.stats` does not stratify. A unit test checks that it equals
    `rcx.stats.seed_parent_bootstrap_test` on a single family.
- **p-values.**
  - Two-sided p comes from `rcx.stats.bootstrap_p_two_sided(boot, floor=1/R, ties='both')`.
  - The one-sided NI p is as in §6.2.
  - Holm uses `rcx.stats.holm`.
- **Thresholds** are compared with a 1e-9 tolerance.
- **Attempt log.** Every confirmation attempt, including VOID reruns on a fresh TEST draw, is logged in
  `runs/RCX-E05/ATTEMPTS.jsonl`. A rerun is never called a fresh confirmation of the same bank.

### 6.5 Decision rule (tested in this order; the first that holds is the decision)

1. **VOID.** An instrument gate fails. That covers:
   - the App. B checks on the TEST oracle;
   - ORACLE nreg ≠ 0 or STOP ≠ 1 exactly;
   - floors not separated from the oracle;
   - a disjointness, support-uniqueness or exclusion assertion;
   - a TEST manifest or code-digest mismatch.

   Fix the fault, redraw TEST from `20261242:test:<attempt+1>`, and log the attempt. At most two redraws: a third
   VOID ends E05 as INCONCLUSIVE pending a design review. Replicates are not retrained (they never see TEST); per-goal
   fits are redone on the new parents.
2. **INCONCLUSIVE (missing).** A protected-core cell (§9.2) is missing. Report descriptively and apply no other rule.
3. **STOP-D (dominated, F5).** D_G ≥ +0.05 with its 95% CI lower bound > 0. A zero-label symbolic planner beats the
   learned SF, so the SF line stops on this testbed. Any P1/P2 results are reported as mechanism findings only.
4. **ADVANCE.** P1 passes, P2 passes (it must not have been dropped), and G1 holds.
5. **STOP-M (mechanism misfit).** The EXACT-GPI(L) primary macro nreg is ≥ 0.5.
   - Rev 1 placed this before ADVANCE. It now comes after ADVANCE, because a learned SF that passes P1 and P2 through
     the z = w′ index should not be stopped by a library ceiling. The GPI qualifier below then applies.
6. **STOP-F (futile).** P2 did not pass **and** P1 is futile, meaning the macro D1's upper 95% bound is < 0.05 or H-ID
   D1's is. The first case means no advantage at all. The second means the advantage is confined to weights outside
   UVFA's support, which is the by-construction part (F3). STOP-F applies even if P1's significance conditions hold:
   an interval entirely below 0.05 excludes the planned effect (the E01N convention).
7. **REVISE.** P1 or P2 passes, but not ADVANCE. This includes:
   - G1 failing in the middle zone (D_G ≥ 0.05 with a CI that includes 0);
   - "only the label-cost claim holds";
   - "P1-M passes, P1-ID does not, and P2 passes". The follow-up is then the UVFA-wide/SF-wide design (§5.2).
8. **INCONCLUSIVE.** Otherwise. A positive point estimate with a wide interval is not a pass, and a nonsignificant loss
   is not equivalence.

**Qualifiers** change the claim wording, never the decision:
- **Attribution:** if macro nreg(UVFA+aux) ≤ nreg(SF-GPI) + 0.02, the gain is credited to vector consequence
  supervision, not to test-time recomposition.
- **GPI:** if macro nreg(SF-noGPI) ≤ nreg(SF-GPI) + 0.01, the claim says "goal-conditioned endpoint predictor read out
  linearly", not "GPI".
- **New directions (F8):** the words "new preference directions" need H-ID-U D1 point ≥ 0.03 with its CI lower bound
  > 0 and H-ID-U D2 upper bound < 0.05 (unadjusted). Otherwise the wording is "re-weighting of seen fact patterns".
- **Naming (F11):** "endpoint-and-length successor features (exact MC targets)".

### 6.6 Power and sizing (F2)

**Rev 1's rule, stated honestly.** Rev 1 required a point estimate ≥ 0.05. At a true effect of 0.05 its power is ≤ 0.5
whatever n. With 56 parents per family, 80% power needs a true effect of 0.060, 0.063 or 0.066 at σ = 0.15, 0.20 or
0.25.

**Assumptions for the new rule.**
- σ is the parent-level SD of the paired difference.
- The normal approximation is used, at Holm step 1: two-sided z = 2.2414 for P1 and one-sided z = 1.96 for P2.
- The planned effect is 0.05.
- Power is Φ((0.05 − max(z·SE, 0.03)) / SE).
- The macro SE is (σ/3)·√(1/n_ID + 1/n_K + 1/n_NEG).

**P1-M power at a true effect of 0.05.**

| σ | n = (96, 56, 56) | n = (152, 72, 72) |
|---|---|---|
| 0.15 | 0.97 | 0.98 |
| 0.20 | 0.89 | 0.95 |
| 0.25 | 0.71 | 0.84 |

**P1-ID power at a true effect of 0.05.**

| n_ID | σ = 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|
| 96 | 0.97 | 0.85 | 0.58 | 0.39 |
| 120 | 0.99 | 0.92 | 0.69 | 0.48 |
| 152 | 0.99 | 0.95 | **0.80** | 0.59 |

Joint P1 power lies between max(0, a + b − 1) and min(a, b). H-ID is a third of the macro, so the two conjuncts are
positively correlated and the joint power is close to the weaker conjunct.

**P2 NI power** (δ_NI = 0.05):

| σ₂ | True D2 = 0: (96, 56, 56) | True D2 = 0: (152, 72, 72) | True D2 = +0.02: (96, 56, 56) | True D2 = +0.02: (152, 72, 72) |
|---|---|---|---|---|
| 0.15 | 1.00 | 1.00 | 0.80 | 0.90 |
| 0.20 | 0.94 | 0.98 | 0.55 | 0.68 |
| 0.25 | 0.80 | 0.90 | 0.39 | 0.49 |
| 0.30 | 0.64 | 0.77 | 0.29 | 0.37 |

**Resizing rule (fixed now; applied once at the pilot, before TEST is drawn; uses DEV variance only).**
1. **Estimate σ̂** from the selection-seed models at the selected configs:
   - σ̂_ID = SD over the 32 DEV-ID parents of d1(p) = y_UVFA(p) − y_SF(p);
   - σ̂_OOD = the same over the 24 DEV-OOD H-K and H-NEG parents;
   - σ̂₂ = the SD over DEV-ID of y_SF(p) − y_PG400(p).

   These are single-model differences, while the confirmation averages 3 replicates, so σ̂ is conservative.
2. **Plan with σ_plan = 1.15·σ̂.** 1.15 is about the 80% upper-confidence factor for an SD from 24–32 parents (1.16 at
   23 df, 1.13 at 31 df).
3. **Set n_ID** = min(152, max(96, ⌈3,802·σ²_plan,ID⌉)), where 3,802 = ((2.2414 + 0.8416)/0.05)².

   | σ̂_ID | n_ID |
   |---|---|
   | ≤ 0.138 | 96 |
   | 0.15 | 114 |
   | 0.16 | 129 |
   | 0.17 | 146 |
   | ≥ 0.174 | 152 (cap) |

   (Recomputed 2026-10-12 for the prereg draft; the earlier row "≥ 0.155 → 152" was an arithmetic slip. The formula,
   not the table, is binding.)
4. **Set n_K = n_NEG = 56.** Raise both to 72 if, at the chosen n_ID, either is < 0.80:
   - the planned P1-M power with σ_plan = 1.15·max(σ̂_ID, σ̂_OOD);
   - the planned P2 power at a true D2 = 0 with σ_plan,2.
5. **Caps.** If the protected core does not fit 150 CPU-min at the G-P5-measured rates, n_ID is reduced toward 96, then
   n_K and n_NEG toward 56. The same happens if fewer distinct unseen support classes exist than H-ID-U needs. The
   planned powers are recomputed and written into the PILOT FILL. **Power above the cap is stated, not hidden.** For
   example, the P1-ID power is 0.59 at σ = 0.25 with n_ID = 152.

---

## 7. Feasibility check (inspected on disk, 2026-10-12)

| Item | Finding | Status |
|---|---|---|
| `rcx/bw/space.py` | sha `29d0b420…` (matches the HSTAR-recipe and B4 pins). It provides `StateSpace` (CSR forward and reverse), `CompiledGoal` with on/ontable/clear/holding, `goal_mask`, `backward_bfs` and `misplaced_compact` (the `clear` goal handling is in `is_done`; `holding` goals fall back to a bound of 0). **Missing:** max-plus value iteration, policy tables, GPI, the height-limit mask. New code: `rcx/bw/soft.py` | OK (extension needed) |
| State and edge counts | n = 7: 65,990 states (37,633 + 28,357) and 186,578 edges; `space.py` asserts the count | OK |
| Oracle cost | n = 7 build about 0.25 s, plus 9 ms per backward BFS (HSTAR recipe era). Value iteration is estimated at about 0.05–0.2 s per weight and is measured in Phase 1 | OK |
| **Training throughput (R-04, measured)** | `runs/RCX-E07/main/REPORT.md`: FlatMLP (158,721 parameters, batch 64) ran 20,724 steps in 26.1–66.1 s at 4 threads, which is 19–48 GFLOP/s effective (6 × parameters × samples). The RelGNN ran the same steps in about 160 s (about 8.3k samples/s). §9 plans at 19 GFLOP/s (pessimistic) and 48 (optimistic), and G-P5 measures the real rates | OK, used in §9 |
| R-04 GNN | `gnn_s20261276.pt` sha `9cc417a3…` = the B4 pin; `models.py` `4c2e3242…` and `encode.py` `32980f14…` match. The encoder accepts on/ontable/clear/holding goal facts, but training used on-only goals | OK (OOD disclosed) |
| Exclusions | `EXCLUSIONS.json` `2f211de3…`: 677 invariants and 349 signatures. `excl.py` `eb31d9eb…` | OK |
| Statistics, lane, guard | `rcx/stats.py` `06a30c10…` (`seed_parent_bootstrap_test`, `bootstrap_p_two_sided`, `holm`, `mcnemar_exact`; no stratified bootstrap, so `analyze_e05.py` adds one); `rcx/lane.py` `03f93d59…` (`try_acquire`, `read_lock`, `release`, `waiting_path`); `rcx/guard.py` `08fa782d…` | OK |
| **Peer overlap (corrected, F7)** | No BWM or RCWM SF/UVFA/goal-weight work. **dsv3wm RDW E07/E08** is adjacent: ledger domain, native features, 184 parents, factorized vs scalar readouts, status `FACTORIZED_NATIVE_FEATURE_STUDY_PREPARED_DEPENDENCIES_PENDING`. The boundary note was sent on 2026-10-12. E05 reads none of its data or code | OK (boundary declared) |
| **Seeds (F1)** | A grep on 2026-10-12 found 20261242–50 only in DECISIONS.md (D-43), the E05 docs and `experiments/RCX-E04/DESIGN.md`. That E04 file still lists 20261242–48 from before D-43; **E04's revision must apply D-43's move to 20261206–14** | OK for E05; E04 text flagged |
| CPU lane | E01R training and SEALED evaluations have priority. E01R's chain waits for ≥ 12.5 GB free RAM without a marker | Schedule constraint (§9.3) |

**Verdict: FEASIBLE.**
- Every input exists. No peer data, no GPU and no 0.6B load are needed.
- New code: about 1,200–1,500 lines plus tests that run in < 5 s on n = 3–5 spaces (§10).

**Main risks.**
1. **Learnability.** G-P4 is set against the copy-state floor. The fallbacks are 64 × 800 labels for every arm, then
   n = 6.
2. **Per-goal fit cost.** This is the dominant cost (§9). It is measured in G-P5, with the protected cut.
3. **H-ID-U supply.** The count of distinct unseen support classes is checked at the pilot. The estimate is about 20
   at K = 3 and well over 100 at K = 4.
4. **Lane availability:** about 10–12 slices of 15 min (§9.3).

---

## 8. Native follow-up path (only after an E05 ADVANCE; each step gated)

1. **E06 (RCX, CPU).** The SF-GPI critic serves as the heuristic of a budgeted search on soft-goal tasks, against
   ENUM-MIS± and GNN-ENUM± at matched expansions. A relational SF adds the structural split n = 6 → 7 → 8.
2. **Native-A probe** (inference only; frozen Titan-Sparse and Supa).
   - Prompt: "you will pursue goal z: which facts hold when you stop, after how many moves?"
   - Scored exactly against ψ^{π_z}.
   - The expected result is a floor (R-03). It needs ≥ 12 GB free RAM and runs after E01R.
3. **Native-C ψ-critic** (LoRA on frozen FP32 Titan-Sparse, unmerged; the D-03/T2 recipe).
   - It uses exact (state, z) → endpoint-and-length targets, with a matched scalar-target control.
   - Readouts at tiers A/B/C/D.
   - Data are audited by the exact validator. E33 retention runs through RCWM.
4. **Planner integration (BWM's call).** It needs a `value(state, w)` hook and a soft-goal task family in a new BWM
   prereg after B4/B5. For hard goals SF is out of span, and the A3 hybrid and BWM's value LoRA remain the hard-goal
   critics.

**Consumer status (F12).** None today. If E05 ends in STOP-D, the program learns that zero-label symbolic enumeration
already serves soft Blocksworld goals, which is a useful negative for the native path. k\* is reported in every
outcome.

**dsv3wm.** RDW E07/E08 results and E05 results are offered for cross-reading. No shared instruments are created.

---

## 9. Compute and schedule (single CPU lane; E01R has priority)

### 9.1 Budget (F6; FLOPs ≈ 6 × parameters × samples; pessimistic 19 / optimistic 48 GFLOP/s; G-P5 replaces these with measured rates)

| Phase | Content | CPU-min, pessimistic | CPU-min, optimistic | Slices of ≤ 15 min |
|---|---|---|---|---|
| 0 | Code and unit tests on n = 3–5 (each < 5 s). Not a lane job | < 1 | < 1 | – |
| 1a | Pilot A: draw TRAIN/DEV; oracle and library tables; selection training (3 arms × 1 run of 40 epochs = 3.3 TFLOP); DEV evaluation of the zero-shot arms; G-P1, G-P2, G-P4 | ≈ 5 | ≈ 2.5 | 1 |
| 1b | Pilot B: PGV_400 grid (32 parents × 126 GFLOP) and FT grid (32 × 25.5 GFLOP) on DEV-ID; PGV_1600/6400 on 8 DEV-ID parents (6.5 TFLOP); G-P3; throughput timing (G-P5); σ̂ | ≈ 11.5 | ≈ 5 | 1 |
| — | **Prereg freeze** (PILOT FILL; shas); then `draw-test` and the manifest-sha commit | 0 | 0 | – |
| 2 | TEST draw and oracle (n_P + 184 goals), starts, banks, exclusions, label indices | ≈ 3 | ≈ 2 | 1 (shared) |
| 3 | 9 confirmation runs (3 arms × 3 replicates, 1.11 TFLOP each) plus SF-shuffled | ≈ 10 | ≈ 4 | 1 |
| 4 | PGV_400 on all primary parents. Worst case is width 256 with 2,000 steps (116 GFLOP per fit); width 64 is about 3 min | ≈ 21 (n_P 208) – 30 (296) | ≈ 8.5–12 | 2 |
| 5 | FT-UVFA_400 × 3 replicates on all primary parents (worst case 400 steps, 25.5 GFLOP per fit) | ≈ 14–20 | ≈ 5.5–8 | 1–2 |
| 6 | PGV_1600 and PGV_6400 on the 48-parent subset (815 GFLOP per parent) | ≈ 34 | ≈ 14 | 3 |
| 7 | Evaluation. A per-replicate library ψ̂ cache (≤ 1.5 TFLOP, float16, ≤ 0.5 GB) serves SF-GPI, SF-L and SF-GPI-5. Also all other arms and families; ENUM-MIS± (Python, about 5 min), ENUM-H\*± (about 8), GNN-ENUM± (about 10) | ≈ 32–36 | ≈ 20–22 | 3 |
| 8 | Descriptive extras: budget curve (≈ 8), SF-FT_400 (≈ 5), optional equivariant SF | ≈ 13 | ≈ 5.5 | 1 |
| 9 | Analysis (create-only outputs) | ≈ 1 | ≈ 1 | (shared) |
| | **Pilot total** | **≈ 16.5** (cap 30) | ≈ 7.5 | 2 |
| | **Confirmation total** | **≈ 128 (n_P 208) – 147 (n_P 296)** (cap 150) | ≈ 60–70 | ≈ 10–12 |

Peak RSS is < 1.5 GB: the ψ̂ cache is ≤ 0.5 GB, V\* tables are streamed per parent, and the models are small.

### 9.2 Protected core and cut order (pre-declared)

**Protected core (never cut):**
- the TEST oracle;
- SF, UVFA and UVFA+aux × 3 replicates;
- SF-noGPI, SF-L and SF-GPI-5 (inference only);
- SF-shuffled;
- **PGV_400 on all primary parents**;
- PG400 on all primary parents (3 replicates if it is FT);
- **PGV_1600 and PGV_6400 on the fixed 48-parent subset**;
- ENUM-MIS±, GREEDY, STOP, RAND, ORACLE, EXACT-GPI(L) and BEST-SINGLE(L) on every family;
- ENUM-H\*± and GNN-ENUM± on H-ID and H-K;
- the one-step banks.

**Cut order** (applied at G-P5 until the projected confirmation is ≤ 150 CPU-min):
1. the budget curve;
2. SF-FT_400;
3. H-KN (all arms);
4. GNN-ENUM± on H-NEG;
5. FT-UVFA_400 reduced to replicate 20261246, **only if FT is not PG400**;
6. ENUM-H\*± on H-NEG reduced to the first 24 parents in manifest order;
7. H-DYN.

If the protected core alone exceeds 150, sizes shrink as in §6.6 step 5, before TEST is drawn. If it still exceeds
150, the coordinator records a CX D-entry, which either raises the cap (for time reasons only) or postpones. The
protected core is never cut after TEST is drawn.

### 9.3 Lane rules (binding; CX D-43)

**Party and markers.** The job holds the lane as party `rcx-e05` through `rcx.lane.try_acquire` only. It **never writes
a waiting marker**, so it never calls `rcx.lane.acquire`, which writes one.

**Start condition.** `run_e05.py` checks all of the following before each phase:
1. **The lane is free:** no lock, after stale-pid reclaim, and **no `rineya-cpu-lane.waiting.*` marker of any party**
   (`waiting.rcx`, `waiting.bwm`, `waiting.dsv3wm`, …).
2. **E01R protection.** Available RAM must be **< 11.0 GB** while any E01R adapter remains untrained. The same applies
   until E01R's SEALED evaluations are done, since they also need ≥ 12 GB for 0.6B loads and outrank E05. E01R's
   12.5 GB start condition then cannot be met during the phase anyway.
3. **Night windows.** It is not inside a night window: **22:50–07:00 on the nights starting 10-12, 10-13 and 10-14**
   (the last ends 10-15 07:00). The phase's projected end (G-P5 rate + 20%) must also fall before 22:50.
4. **Thermal guard:** `rcx.guard` reports tz01 < 97 °C (or it has resumed at ≤ 93 °C), and no `.thermal_pause` file
   exists.
5. **No stray python process** (`Get-CimInstance Win32_Process`).
6. **Length.** The projected phase length is ≤ 15 min, and `expected_end` = start + 15 min.

**Safe points and yielding.**
- Safe points come after each training run, each PGV/FT batch of ≤ 64 models, and every 8 evaluation parents. They
  are at most about 2 min apart.
- At each safe point the job checkpoints, releases the lane and exits if any of these holds:
  - any `waiting.*` marker exists;
  - 15 min have elapsed;
  - the clock is within a night window;
  - the guard asks it to pause.
- The next phase resumes from the checkpoint. Outputs are keyed and append-only, so nothing is redone.

**Interaction with E01R.** If RAM rises during a phase, the E01R chain finds the lane held, writes `waiting.rcx`, and
E05 yields within about 2 min.

**Threads and devices.**
- Torch threads ≤ 4, and `OPENBLAS_NUM_THREADS=1` for the numpy oracle.
- `CUDA_VISIBLE_DEVICES=""` and the `torch.cuda` patch are applied before torch is imported, as in `rcx/bw/models.py`.

**Who runs what.** **Only the coordinator runs phases.** Subagents edit code and run checks of < 5 s while any lane job
runs.

### 9.4 Calendar (indicative)

| When | Step |
|---|---|
| 10-12 | DESIGN rev 2 and the PREREG draft (this file and `PREREG_E05_DRAFT.md`); coordinator review; CX D-entry adoption |
| 10-12/13, daytime | Code and unit tests (§10), no lane. Pilot code digest recorded |
| 10-13, daytime, when the start condition holds | Phases 1a and 1b, then PILOT FILL, prereg freeze, `draw-test` and the manifest commit |
| 10-13/14 daytime gaps, then after E01R completes | Phases 2–9 in 10–12 slices |

---

## 10. Code to be written (not now) and the tests each needs

| File | Content |
|---|---|
| `rcx/bw/soft.py` | `phi_table(space)` (65,990 × 56 uint8); `soft_values(space, w, c, edge_mask=None)` (max-plus VI); `policy_table(space, V, w, c)` (greedy with the tie rule; pointer doubling gives endpoints and lengths); `sf_table` (ψ for all states); `gpi_scores`; `height_limit_mask(space, h=3)`; `conj_values(space, G, R, c)` via `backward_bfs` |
| `experiments/RCX-E05/families_e05.py` | Generators for every family; `canon_key(w, c, rtype)`; `support_key(w, signed)`; `stabilizer(w)`; `rename_state` / `rename_weight`; seen/unseen quotas; one-parent-per-support assertion; the exclusion adaptor (`excl.row_exclusion_reason`, `exclusion_reason`); derived-stream helper `derive(seed, tag)` (sha256-based) |
| `experiments/RCX-E05/enum_e05.py` | ENUM± target enumeration (falsifier menus, satisfiability, the U(G; s′) persistence rule); scorers for ENUM-MIS±, ENUM-H\*± and GNN-ENUM±; GREEDY, RAND, STOP, EXACT-GPI(L), BEST-SINGLE(L), ORACLE |
| `experiments/RCX-E05/learners_e05.py` | SF, UVFA and UVFA+aux MLPs; `BatchedMLP` for PGV and FT; augmentation (renaming, stabilizer); trainer with checkpoints at 20/40 epochs and 100/500 steps; the library ψ̂ cache |
| `experiments/RCX-E05/run_e05.py` | Subcommands `pilot-a`, `pilot-b`, `pilot-report`, `draw-test --expect-prereg-sha --expect-code-digest`, `oracle-test`, `train`, `pgv`, `ft`, `eval`, `extras`, `code-digest`. Also the lane gate (§9.3), keyed append-only outputs, `ATTEMPTS.jsonl` and a provenance manifest (shas, seeds, versions) |
| `experiments/RCX-E05/analyze_e05.py` | Stratified paired cluster bootstrap; IUT P1; NI P2; Holm; decision rule §6.5 with qualifiers; per-family and strata tables; k\*; create-only report. It refuses unless every eval names `TEST_MANIFEST` sha and the frozen code digest |

**Tests** (all < 5 s on n = 3–5; literal seeds; never reported):

| Test file | Covers |
|---|---|
| `tests/test_e05_oracle.py` | Q1 (VI = brute force over all-pairs BFS on n = 5, 200 random w incl. negatives, `clear`, three c values); Q2 (H-CONJ vs `GoalLabels` at n = 5; the n = 6 version runs in the pilot); Q3 (rollouts = tables, V = w̄·ψ exactly); Q4 (GPI ≥ max over i of V^{π_i}); Q5 (value renaming invariance exact; the conjugate-policy ψ identity w̄_σ·σψ = V\*_{σw}(σs)); Q6 (height-limit mask = a direct rule check, n = 5 all states) |
| `tests/test_e05_families.py` | Canonical keys invariant under random σ; support keys (9 two-fact on/ontable classes; signed keys); stabilizer correctness (σw = w, group closure); split and support disjointness; one-parent-per-support; quotas; the exclusion adaptor on synthetic reserved signatures; derived-stream determinism |
| `tests/test_e05_enum.py` | Falsifier menus; satisfiability pruning; hand cases where a penalty must be avoided (ENUM± avoids it, rev 1's ENUM does not); `holding` targets; ENUM-H\*± = ORACLE on single-target goals; positive-only goals reduce to rev 1's ENUM |
| `tests/test_e05_learners.py` | `BatchedMLP` equals B independent models (same init and data order, 50 steps, max abs diff < 1e-5); per-model early stopping; augmentation preserves labels exactly; the 20-epoch checkpoint equals an independent 20-epoch run; SF-shuffled permutes within weight only; the CUDA-off patch |
| `tests/test_e05_analyze.py` | Stratified bootstrap = `rcx.stats.seed_parent_bootstrap_test` on one family; IUT p = max; NI p; Holm with a dropped P2; every decision branch on synthetic inputs (VOID, missing, STOP-D, ADVANCE, STOP-M, STOP-F, REVISE incl. the P1-M-only case, INCONCLUSIVE); qualifiers; 1e-9 tolerance; refusal on a manifest or digest mismatch |
| `tests/test_e05_lane.py` | With a temporary `RINEYA_LANE_DIR`, an injected clock and an injected RAM function: refuses with any `waiting.*` marker, with RAM ≥ 11 GB while E01R is pending, inside or running into a night window, and with the lane held; never writes a marker; yields at a safe point when a marker appears; caps the phase at 15 min |

---

## 11. What is frozen when

| Point | What is fixed | Recorded in |
|---|---|---|
| **T0 (now)** | Nothing. DESIGN rev 2 and `PREREG_E05_DRAFT.md` are drafts | – |
| **T1 adoption** | Scope, the dsv3wm boundary, seeds 20261242–50 (D-43), and E04 independence | CX D-entry |
| **T2 pilot-code freeze** (before Phase 1) | The sha256 digest of `soft.py`, `families_e05.py`, `enum_e05.py`, `learners_e05.py` and `run_e05.py` plus the pinned existing files (§4.1; digest definition in the prereg draft §2.1), with all tests passing. TRAIN and DEV are drawn by this code | DECISIONS ("E05 pilot code digest") |
| **T3 prereg freeze** (after Phase 1) | `PREREG_E05.md` = the draft plus the PILOT FILL only: selected configs, PG400 identity (or "P2 dropped"), library composition, n_ID/n_K/n_NEG, H-ID-S size, planned powers, measured rates and cut depth, and the DEV results known. Also the full code digest **including `analyze_e05.py`**. All thresholds, the decision rule, the resizing formula and the cut order are already fixed in the draft and do not change | DECISIONS (prereg sha + code digest) |
| **T4 TEST draw** | `TEST_MANIFEST.json` from `20261242:test:1`. Its sha is committed before any arm is evaluated on TEST | DECISIONS |
| **T5 confirmation** | Replicates 20261246–48 are trained after T3. Per-goal fits, evaluation and **one** analysis run | `runs/RCX-E05/…`, ATTEMPTS.jsonl |
| **T6 report** | EXPERIMENT_RESULTS R-nn; DECISIONS entry; FYI to BWM, RCWM and (relayed) dsv3wm | – |

After T3, any change to a bound file is a logged deviation, marked "outcomes known: yes/no". After T4, a change that
could affect TEST outcomes voids the attempt.

---

## 12. Open items before the prereg freeze

1. CX D-entry adoption (T1). E05 runs before E04.
2. The E04 DESIGN.md seed table still shows 20261242–48. E04's own revision applies D-43, and E05 does not edit E04.
3. Whether the optional equivariant SF fits the lane time (descriptive only).
4. Whether BWM's B4 fresh set exists by T3, which would re-pin the exclusions.

---

## Appendix A. Seeds (block 20261242–20261250, CX D-43; E04 uses 20261206–14)

| Seed | Purpose |
|---|---|
| 20261242 | Family generation and splits. TRAIN and DEV are drawn at the pilot. TEST uses `20261242:test:<attempt>` after the freeze, and the 48-parent PGV subset uses `20261242:pgv-subset` |
| 20261243 | TRAIN labelled-state sampling; the renaming-augmentation stream; the SF-shuffled permutation stream `20261243:shuffle` |
| 20261244 | Starts and one-step banks (DEV and TEST, derived per split); RAND's stream |
| **20261245** | **Selection only:** pilot model initialization for config selection, the per-goal grids on DEV, G-P3/G-P4 and σ̂. Never a TEST replicate |
| **20261246, 20261247, 20261248** | **Confirmation replicates** for SF, UVFA and UVFA+aux, and FT-UVFA_400 from each. SF-shuffled, the budget curve and SF-FT_400 use 20261246 |
| 20261249 | Per-goal label sampling (nested 400 ⊂ 1,600 ⊂ 6,400, shared by PGV, FT and SF-FT) and PGV initialization. Derived per parent and k |
| 20261250 | Analysis bootstraps |

**Derived streams.** derive(seed, tag) = int.from_bytes(sha256(f"{seed}:{tag}")[:8], "big"). Unit tests use literal
seeds 0–9.

## Appendix B. Instrument qualification checks and pilot gates

| Check | Content |
|---|---|
| Q1 | Value iteration equals the brute force max over s′ of [ w·φ(s′) − c·d(s, s′) ] on n = 5, for 200 random w (incl. negatives, `clear`, c ∈ {0.125, 0.25, 0.75}) |
| Q2 | H-CONJ: V\*(s) = max( 3·1[G ⊆ s], 3 − c·h\*(s, G) ) on 200 (s, G) pairs at n = 6 (pilot) |
| Q3 | Rollouts on 1,000 (s, w) reproduce the tabled endpoints and lengths, and V\*_w = w̄·ψ^{π_w} exactly |
| Q4 | EXACT-GPI(L) value ≥ the max over i of V^{π_i}_{w′} at every start |
| Q5 | V\*_{σw}(σs) = V\*_w(s) for 500 random σ; canonical keys invariant; w̄_σ·σψ^{π_w}(s) = V\*_{σw}(σs) |
| Q6 | The height-limit successor mask equals a direct rule check (n = 5, all states) |
| Q7 | Floors separated: ORACLE nreg = 0, STOP = 1, RAND and GREEDY strictly between (on DEV and TEST) |
| Q8 | Split hygiene: class disjointness, support uniqueness and H-ID-U unseen-ness, exclusion counts, the n = 7 assertion vs RCX test sets, and the TEST sha committed before evaluation |

**Pilot gates** (DEV only, before the freeze; engineering decisions, recorded with outcomes known on DEV):

| Gate | Condition |
|---|---|
| G-P1 | Q1–Q8 pass |
| G-P2 | DEV-ID exact-GPI(L) ceiling. If it is ≥ 0.5, the 49 single-fact base policies are added to L for every learned arm, before the freeze |
| **G-P3 (F4)** | **Per-goal competence.** Each candidate's best grid config must reach DEV-ID nreg ≤ nreg(GREEDY) − 0.20. PG400 is the better competent candidate (ties to FT), with one grid extension if neither is competent, else P2 is dropped (§5.2). Also, on the 8 DEV-ID parents, PGV_6400 nreg ≤ 0.25 and PGV is monotone in k (if not, PGV training is fixed before the freeze) |
| **G-P4 (F9)** | **Learnability against the copy-state floor**, on the DEV-ID one-step bank pairs (32 × 64), with selection-seed models at the selected configs. The value-exact rate acc½(arm) is the share of pairs with \|V̂ − V\*\| < c/2. V̂ is w̄·ψ̂(s, w) for SF (own index), V_θ(s, w) for UVFA, and w·φ(s) for the copy-state floor, whose acc½ is the share of STOP-optimal pairs. **(a) SF:** acc½(SF) − acc½(copy) ≥ 0.40 **and** top-1(SF-noGPI) − top-1(GREEDY) ≥ 0.20. GREEDY is the copy-state ranking floor. **(b) UVFA:** acc½(UVFA) − acc½(copy) ≥ 0.20, or one grid extension (§5.5). Mean value error is reported for all arms. If (a) fails: 64 × 800 labels for every arm; if it fails again, n = 6. Both fallbacks are engineering decisions recorded before the freeze |
| G-P5 (F6) | **Measured compute.** Steps/s for each model size at its batch size (single and batched B ∈ {16, 64}), VI seconds per weight, and evaluation decisions per second. These give the projected confirmation CPU-min at the sizes from §6.6; the cuts of §9.2 are applied until the projection is ≤ 150. The projection and cut depth go into the PILOT FILL |
