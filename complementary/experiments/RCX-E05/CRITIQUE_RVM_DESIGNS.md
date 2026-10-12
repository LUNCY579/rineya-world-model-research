# Adversarial critique of the RCX-E04 and RCX-E05 designs (workflow wf_ad10715d-8d6, 2026-10-12)

E05 should run first, after a short revision round. Do not draft an E04 prereg until its primary is rescoped and its overlap with dsv3wm is resolved. Both designs share one blocking defect: they claim the same seeds, 20261242–49. Nothing was created, run or committed; I only read the two DESIGN.md files, the RCX records and the peer docs.

=== RCX-E04 (C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary/experiments/RCX-E04/DESIGN.md) ===

Answers to the review questions:
- **Falsifiable:** yes, the bars are explicit.
- **Informative for Rineya:** barely. The primary is a test the design itself expects to pass easily.
- **Labels:** grounded in the oracle (`bw_parts`, `B.step` and an independent precondition table).
- **Leakage:** mostly controlled. There are pre-freeze exposure and peer-data training issues (F7, F8).
- **Decision rule:** can be pre-specified, but one control is broken (F3).
- **Power:** ample for the primary; the producer-identity probe (S2) is underpowered.
- **CPU:** feasible. From R-04 throughput (GNN about 8.3k samples/s) the real training time is about 10–20 min.
- **Duplication:** an undeclared overlap with dsv3wm, plus item-6 boundary risks (F4, F8).

Findings:
1. **HIGH, §1.2/§1.5/§4.6 — the primary sits at ceiling.**
   - Problem: the critic's input is the parsed state plus the action. The label oracle computes legality from that same input at zero cost. That rule is 1–3 facts, and the ORACLE arm beats every learned arm on every metric.
   - The design predicts C\*, L1 and S at precision ≥ 0.99 and recall ≥ 0.97, against bars of 0.95 and 0.90. It predicts a gain over F0's flag of about +0.19 against a bar of +0.10, with power about 1.
   - The fallback compares A\* with SUP. SUP cannot see false-illegal answers (80% of T-KO errors), so the expected Δ of about +0.8 is a strawman.
   - The most likely ADVANCE is S, a precondition table induced from legal rows. That re-derives the PDDL and repeats the R-04 lesson that relational encodings beat flat ones.
   - Fix (pick one):
     - (a) Make the simulator-free S′ the primary: preconditions induced only from Rineya's own F-verified B3 plans. Split the 240 distill problems in two: induce on `solutions.jsonl` plans of half A, test on `fcache` keys of half B. Score recall on Pa_v2's illegal proposals and the false-reject (option-loss) rate. This is uncertain, because induction from positive examples alone can learn spurious preconditions from plan regularities, and it bears directly on BWM's simulator-free claims and Logistics.
     - (b) Demote E04 to an instrument deliverable (battery plus reference numbers for item 6 and dsv3wm) with no ADVANCE claim.
     - (c) A text-channel critic tested under held-out renderings, only with the item-6 owners' consent.
2. **HIGH, §1.3/§1.4 — the complementarity argument and the test bed don't match.**
   - Problem: the case for E04 is that the support check is incomplete in Logistics and the ledger. E04 tests only Blocksworld, where the design shows the check is complete.
   - The incumbent composite INC already has recall 1.0 and a false-reject rate of 0 on T-PA. The only possible planner gain is at most 11.1% of unique F queries, and FREPLAY shows F is not the bottleneck.
   - Fix: state that E04 cannot support any "carries over" claim. Before investing, ask BWM whether a precondition pre-filter would be admissible or wanted in its simulator-free planner.
3. **HIGH, §3 SHUF, §4.4 G6, §4.5(d) — the shuffled-label control will fail by construction.**
   - Problem: permuting labels within (op, n) strata keeps the op×n signal. Under PR-U/PR-L sampling, illegal actions are mostly stack/unstack and legal ones mostly pick-up/unstack, so SHUF's AUROC will match the op×n floor (likely 0.6–0.7), not 0.5.
   - Consequence: G6 and (d), which require AUROC in [0.45, 0.55], trip and the run is VOID or REVISE for a reason unrelated to the hypothesis.
   - Fix: permute labels globally, or gate SHUF to within ±0.03 of the op×n floor.
4. **HIGH, §1.3/§6.1 — an undeclared overlap with dsv3wm (RDW-001).**
   - Problem: dsv3wm's E11–E13 are prepared, with native gates pending (`execution/STATUS.json`, `audits/E11_13_v2_preparation.md`). They include:
     - a Blocksworld critic bank with 64 BWM-family calibration parents and false-acceptance bounds;
     - linear/MLP critic heads;
     - a `FilteredProposer` for learned pruning.
   - That is the same domain and deployment as E04 §6.1. E04 never mentions dsv3wm, and COLLABORATION §1–2 does not list it as a scope party.
   - Fix: add dsv3wm to the scope table, agree a boundary (E04 tier-B parsed-fact baseline vs dsv3wm's native-feature heads), and offer the E04 battery for cross-scoring. Do not run §6.1 as a separate planner filter.
5. **HIGH, §4.1/§6.1 — the qualified operating point differs from the deployed one.**
   - Problem: the primary qualifies τ95 (DEV recall 0.95, precision ≥ 0.95 allowed). §6.1 deploys at DEV false-reject rate ≤ 0.1%.
   - F0flag's false-reject rate is 0/13,360. A recall-only comparison lets C\* win while losing up to about 80 legal options per 13,360 legal keys.
   - Fix: make the primary the recall (F calls saved) at the deployment false-reject point, paired against F0flag, with options lost as a co-endpoint.
6. **HIGH, §2.2 — seed collision with E05.**
   - Problem: both designs assign 20261242–20261249 (E05 also uses 20261250). Each says the seeds are unused.
   - Free 202612xx seeds per a grep of the RCX tree: 06–19, 21–29, 32–35, 37–39, 42–50, 54–55, 60, 82–87.
   - Fix: give E04 its own block, e.g. 20261206–14, and record it in DECISIONS.md.
7. **MEDIUM, §4.4 G3/G5 — the test pool is exposed before the freeze.**
   - Problem: Stage 0 opens the T-PA keys to reproduce counts and to deduplicate training data, before the prereg exists.
   - Fix: freeze the generator and Stage-0 scripts by sha first, and have them emit only aggregates plus opaque canonical hashes for deduplication. Alternatively ask BWM to run G3.
8. **MEDIUM, §3 CR-F0/CR-FK and §6.5 — training on peer outputs; claiming peer pools.**
   - Problem: S2 trains on BWM's F0/FK D-54 evaluation outputs, which were released as readable, not trainable. Declaring T-PA and T-KO as "E04 test pools" also constrains the item-6 owners.
   - Fix: list both explicitly in CX Q-04 as requests, or make S2 purely procedural.
9. **MEDIUM, §4.2 S2 — the bundle's core question is underpowered.**
   - Problem: the producer-identity probe, which is RVM-001's actual concern, has about 23 unique invalid FK half-B keys, so it is descriptive only. The powered part of the experiment is the uninformative one.
   - Fix: power it with procedural producers of swapped error polarity, then confirm on T-KO; or drop it from the claims.
10. **MEDIUM, §7 — lane rules are weaker than E05's.**
    - Problem: E04 runs one 60-min job and "yields to a peer's marker". The E01R chain waits for RAM without writing a marker, and `waiting.rcx` is not a peer marker. There is no RAM gate.
    - Fix: adopt E05 §9.2 verbatim:
      - start only when free RAM is below 11 GB while E01R adapters remain;
      - yield to any `waiting.*` marker, including `waiting.rcx`;
      - run in slices of ≤ 15 min;
      - never start inside the night windows.
11. **LOW, §1.2 — the bars were set with the comparator known.** F0flag's 0.780 recall was public when the +0.10 bar was chosen. This is acceptable once frozen, but disclose it.
12. **LOW, §2.4(2) — renaming dedup thins small-n training.** Deduplicating against T-KO at n = 5, which has only about 12 state orbits, removes most n = 5 (s, a) orbits from training. Report this per orbit.

=== RCX-E05 (C:/Users/clclc/BusinessProject/WorldModel/rineya-complementary-worktree/complementary/experiments/RCX-E05/DESIGN.md) ===

Answers to the review questions:
- **Falsifiable:** yes.
- **Labels:** exact, from value iteration over the full n = 7 graph.
- **Leakage:** weight classes are disjoint across splits, the test set is drawn after the freeze, and the exclusion file is applied.
- **Peer data:** none used.
- **Informative for Rineya:** moderate to low. There is no current consumer (BWM goals are hard `on` facts only), and the one Rineya-like family, H-CONJ, is secondary and predicted to fail.
- **Duplication:** none with BWM or RCWM. There is a conceptual overlap with dsv3wm's E07/E08 (F7).
- **CPU:** feasible, but the per-goal regression phase is under-costed (F6).

Findings:
1. **HIGH, App. A — seed collision with E04.** 20261242–49 are assigned in both designs. Fix: move one block (see E04 F6). Also separate the selection seed from the replicate seeds; 20261245 is used for both.
2. **HIGH, §6.6 — power is computed for the wrong rule.**
   - Problem: P1 passes only if the point estimate D̂1 ≥ 0.05. Power was computed for rejecting 0 at a true effect of 0.05, which gives 0.84. At a true effect of exactly 0.05 the real power is at most 0.5.
   - With σ = 0.20 and n = 168 (SE 0.0154), 80% power needs a true effect of about 0.063; at σ = 0.25, about 0.066.
   - Fix: either keep the rule and state the power honestly with a resizing rule, or switch to "CI lower bound > 0 and point estimate ≥ 0.03" with power at 0.05.
3. **HIGH, §1.1/§6.2 — P1 is partly true by construction.**
   - Problem: UVFA's weight input never sees negative values (H-NEG) or more than 4 nonzero facts (H-K). Two of the three primary families are therefore outside its input support, while SF reads w′ out linearly by design. A P1 pass would mostly restate the linearity assumption.
   - Fix (either):
     - require P1 to also hold on H-ID alone (a conjunctive requirement, no extra Holm cost), or
     - add UVFA-wide and SF-wide arms trained on weights that cover sign and cardinality at equal labels.
   - Report per family in every case.
4. **HIGH, §5.2/App. B G-P3 — the per-goal comparator for P2 risks being a strawman.**
   - Problem: PGV_400 trains a 164k-parameter MLP from scratch on 360 labels. At batch 256 for at least 2,000 steps that is about 1,000 epochs, with early stopping on 40 labels and no augmentation. Correct ranking needs value error below c/2 = 0.125.
   - G-P3 checks only PGV_6400 and monotonicity.
   - Fix:
     - give PGV the free augmentation from permutations that leave w′ unchanged;
     - add a "fine-tune the UVFA (or SF) on 400 labels of the new goal" arm as the realistic competitor;
     - require PGV_400 to beat GREEDY by a pre-set margin on DEV.
5. **HIGH, §5.3 — the non-learned guard G1 is weak on H-NEG.**
   - Problem: ENUM-MIS and GNN-ENUM enumerate only positive and `clear` facts and score w′·1_G, so they ignore penalties while navigating. Only STOP is scored exactly. This makes G1 easy to pass on H-NEG.
   - Fix: give the enumeration negated targets, or penalise successors by the negative facts they force.
   - Also add a STOP branch: if ENUM-MIS beats SF-GPI by ≥ 0.05 with the CI excluding 0, the learned SF is dominated by a zero-label planner. That should be STOP, not REVISE.
6. **MEDIUM, §9.1 — compute is under-costed.**
   - Problem: R-04's flat MLP ran 20,724 steps in 26–66 s at batch 64 (`runs/RCX-E07/main/REPORT.md`). PGV needs 504 fits × ≥ 2,000 steps, about 1.0M steps at batch 256. Unbatched that is roughly 1–3 h, against 10 min budgeted. The pilot and training phases are also about 2× optimistic. A realistic total is 1.5–3 h.
   - Fix: write batched multi-model PGV and measure it in Phase 1 (G-P5). Pre-declare a cut that keeps PGV_400 on all 168 parents and runs PGV_1600/6400 on a fixed random subset, e.g. 48 parents.
7. **MEDIUM, §0/§7 — the dsv3wm overlap was missed.**
   - Problem: RDW-001's E07/E08 ("Consequence factorization versus scalar value", held-out reward weights, ledger domain, native features, hosted under "RCWM consequence features") have status FACTORIZED_NATIVE_FEATURE_STUDY_PREPARED_DEPENDENCIES_PENDING (`execution/audits/E07_E08_review.md`). The grep missed them because they use different terms.
   - Fix: cite them, send an FYI and boundary note to dsv3wm and RCWM (Blocksworld exact oracle, tier B, policy-dependent SF with GPI), and offer cross-reading of results.
8. **MEDIUM, §4.5(1)/§6.4 — H-ID mostly re-weights seen patterns.**
   - Problem: up to renaming there are only 9 two-fact support patterns over on/ontable. With about 19 two-fact library classes, nearly every K = 2 H-ID parent shares its support with a training goal. Clustering by weighted class therefore understates the SE.
   - Fix: cluster H-ID by support class, and make the unseen-support stratum carry the "new direction" claim.
9. **MEDIUM, App. B G-P4 — the learnability gate can pass on a trivial predictor.**
   - Problem: per-fact endpoint accuracy ≥ 0.90 is reachable by just copying the current state (about 0.9, since about 10 of 56 facts are true and policies change only a few).
   - Fix: gate on value error |w̄·(ψ̂ − ψ)| < c/2, or one-step top-1 vs the oracle, relative to the copy-state floor.
10. **MEDIUM, §5 — flat MLPs are a known weak point here.**
    - Problem: R-04's flat MLP reached a value MAE of 0.55 against 0.07 for the GNN. GPI's maximum over 65 noisy ψ estimates adds an upward bias.
    - Fix: report SF-L and SF-noGPI against EXACT-GPI(L) as the learnability readout. Consider making the equivariant SF a reported arm rather than optional.
11. **LOW, §3 — claim wording.** The SF here is a terminal-reward endpoint-and-length predictor learned from exact rollout (Monte Carlo) targets, with no TD learning. Say so; dsv3wm's own E08 card warns about exactly this naming.
12. **LOW, §1.2/§8 — no consumer yet.** The planner route needs a new BWM soft-goal prereg after B4/B5, and native-A is expected to score at the floor. Frame E05 as a mechanism or instrument study, and treat G1 together with the label exchange rate k\* as the deliverables that drive decisions.

=== Recommendation ===
**Run E05 first, after revision.** Revise E04 substantially; do not drop it.

Why E05 first:
- It needs no peer release or seed permission.
- Its outcome is genuinely uncertain (the designer gives ADVANCE about 0.35), unlike E04's predicted pass.
- Its ≤ 15-min phases, start rule of "free RAM < 11 GB" and no-marker policy suit the gaps while E01R waits for RAM.
- Its independence from E04 holds, because every candidate move is exact-legal.

Before an E05 prereg is drafted:
- E05 F1–F5 must be fixed: seeds, the power rule, a fair per-goal comparator for P2, P1 holding on H-ID or the wide control arms, and enumeration that handles negative weights plus the dominated-STOP branch.
- F6–F9 too: measured compute with the protected PGV_400 cut, the dsv3wm FYI, support-class clustering and a G-P4 gate tied to the copy-state floor.

E04 cannot freeze anyway: both its primary and its fallback need BWM replies (CX Q-04). Before any E04 prereg:
- choose between the simulator-free S′ primary (problem-disjoint distill halves, recall plus options lost) and an instrument-only deliverable;
- fix SHUF;
- qualify the deployment operating point;
- resolve the dsv3wm E11–E13 overlap and the item-6 and peer-output training permissions;
- freeze the Stage-0 scripts before opening T-PA;
- adopt E05's lane rules;
- take the new seed block.

The coordinator can send the revised Q-04 and the dsv3wm/RCWM boundary note now at no compute cost. E01R's training and SEALED evaluations keep priority over both.
