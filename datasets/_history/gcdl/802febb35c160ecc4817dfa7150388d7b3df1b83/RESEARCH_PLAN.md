# GCDL-001 research plan

Version 1; 11 October 2026, America/Chicago. This is a dataset-production programme for a future compact Rineya decision model. No fine-tuning, model download, paid inference or GPU experiment is part of this implementation. Measured implementation status and counts belong to `reports/` and the release manifest, not to this plan.

## Research question and claim boundary

Can oracle-verified alternative consequences support goal-conditioned choice, calibrated uncertainty and useful computation allocation, while preserving existing model skills? Dataset correctness is a prerequisite; producing a correct dataset does not answer the model-learning question.

Separate three quantities: the quality/coverage of proposed candidates; the accuracy of their predicted consequences; and the quality of selection given the same candidates and information. In every future experiment, preserve candidate availability across comparison arms. Report omitted solutions separately from poor ranking. Keep model-only behaviour, learned-world-model assistance, exact-tool assistance and trained-student transfer in different inference tiers.

The research sources are pinned in `reports/reference_snapshot.json`. Current main snapshot is `ece566dfa31e1e49f93efb264505c3c622c5dad9` (11 October). The initial local checkout and the 9 October complementary planning bundle were older snapshots; their queued-status text is superseded by the current dated records. `reports/RESEARCH_ARCHITECTURE.md` records the checked numbers and remaining limits.

## Hypotheses and falsification

| ID | Hypothesis | Observable dataset requirement | Future falsification test |
|---|---|---|---|
| H1 | Choice changes with the goal while dynamics stay fixed | Multiple goals on the same state/rules/candidates; identical transition evidence | Matched-candidate choice on held-out goals versus goal-blind, first-option and simple heuristic floors |
| H2 | Explicit rule diversity can break token cues | Minimal rule reversals, token-novel names, semantic twins, legal and illegal cases | Held-out rule wording, new vocabulary, numerical boundaries and combined shifts; report false-apply and false-copy separately |
| H3 | Serial consequences support delayed-benefit decisions | Executed trajectories, intermediate state, cost and constraints | Longer-than-training horizons, delayed harm, no-solution and illegal-prefix cases; compare at matched calls/tokens |
| H4 | Verified search can supply useful compact labels | Direct-success, search-repair, search-failure, cheaper-equal and omitted-solution cases | Matched-extra-training control, independent seeds and fresh sealed problems; search preference is not a global optimality certificate |
| H5 | Abstention reflects the observation's information content | Fully observed, tied, missing-information and no-feasible-candidate cases | Success probability proper scoring under a specified distribution; selective risk/coverage; ambiguity under possible-world changes |
| H6 | Computation has task-specific value | Identical initial information/candidate universe, fixed cheap/expanded budgets, actual execution results and operation costs | Held-out improvement prediction and utility-minus-compute curves; realized hindsight gain is excluded from policy inputs |
| H7 | Training effects transfer and retain old skills | Rule-composition holdouts, control operations and retention-ready evaluation contract | Paired parent-level performance plus correctness-based retention; a renamed instance alone is surface invariance, not domain transfer |

These are hypotheses, not promised gains. A successful finite shortcut battery excludes the enumerated heuristics; it cannot prove absence of every possible shortcut.

## Initial scope and curriculum

Use fresh, self-contained ledger-style and Blocksworld-style environments with independently implemented checkers. Existing RCWM/BWM programmes remain read-only. Do not import their training jobs, model loaders, reserved confirmations or output caches into this worker. No `HSKILL_CONFIRM_V2*` file or RCX E01R secret wording pool is an input. Existing public/used reports may inform failure categories; their test answers and near-duplicate instances may not seed production.

| Family | Minimal meaningful content | Coverage extension / limit to report |
|---|---|---|
| A | Same world and candidates under multiple goals; candidate-set best and complete ties | Held-out goal weights and priorities; no inference-time utility labels |
| B | Counterfactual candidates from the same world, including close calls and delayed cost | Distinguish neutral effect from beneficial goal-relative effect; separate irreversible harms |
| C | Real execution of multi-step plans with intermediate states, legality and costs | Horizon and state-size holdouts; bound search and mark incomplete proofs |
| D | Semantics-preserving renaming, presentation order and equivalent rule renderings; semantics-changing reversals | Track each factor and combined shifts; text paraphrase equivalence is only within a closed verified renderer |
| E | Exact success facts; finite possible worlds with explicitly supplied weights when used | Probability is conditional on the supplied distribution, not an intuitive confidence or real-world uncertainty estimate |
| F | Cheap and expanded selection/search on the same task, with executable outcomes and operation counts | A simulator-expansion cost is not measured LM FLOPs or a hardware latency claim |
| G | Bounded search/direct-choice comparisons and exact verification of selected plans | `best_among_candidates`, `best_found_within_budget`, `optimal_within_horizon` and global proof are distinct |
| H | Fresh compositions of familiar primitives, multi-goal tradeoffs and rule reversals | True environment/domain transfer requires separately designed tasks; permutations do not establish it |

Maintain independent strata for action count, plan horizon, world size, goal/constraint count, objective conflict, rule complexity, numeric magnitude, renderer, utility gap and search difficulty. Track impossible, tied, ambiguous and invalid cases explicitly. Difficulty tags describe generator settings; measured shortest path, candidate gap and expansion cost provide stronger difficulty evidence. Easy/medium/hard/adversarial coverage is assessed by independent semantic parents, not inflated by rendering variants.

Each production cycle has a bounded number of candidates and a deterministic seed/configuration. The generator may rotate among underrepresented validated strata. It may adapt only from TRAIN/DEV coverage and failures. Rare/hard categories retain minimum quotas. When new semantic coverage stalls, the worker should audit, analyze or stop at its configured ceiling; an infinite stream of renamed duplicates is not progress.

## Ground truth and supervision

The oracle executes every supplied candidate and computes legality, constraint violations, trajectory cost, success and utility using a versioned definition. A separate checker reconstructs those results from task inputs. Agreement of repeated LLM judgments is not independent verification. Never repair labels by silently editing the task. Preserve rejected/quarantined original candidates with a reason and the source fingerprint.

`DATA_CONTRACT.md` is authoritative for concrete field names and exact formulas. The canonical record separates `input` from oracle/decision targets and provenance. Pairwise/listwise/pointwise exports use only `input` as features. Rationales and evidence remain targets/metadata. Stable identifiers use canonical semantic content; provenance, split IDs, filenames and outcome-dependent formatting are excluded from model inputs.

Hard constraints take priority before utility. Tied maxima remain a set; deterministic presentation order is only a serialization convention. Illegal/no-feasible candidates need an abstention label, not a forced arbitrary best option. If no candidate succeeds, state `no_candidate_meets_goal`; do not infer that the entire state space is unsolvable. Finite-horizon exhaustive search proves only its registered bound. Budget exhaustion gives `unknown`, never an optimality or impossibility certificate.

For stochastic or partially observed records, define the probability event, observation and possible-world weighting. Record exact numerator/denominator or weighted mass, not a language-model estimate. Model confidence remains a future predicted quantity. Aleatoric outcomes, missing information, ties and insufficient candidate coverage are different causes.

## Protected evaluation and sealing

Assign semantic parent, template, rule-composition and generator-lineage groups before any rendering or goal variants. Treat overlapping groups as connected components: a counterfactual or trajectory suffix bridging two parents must not cross protected splits. Keep a durable group-to-split registry across cycles and generator versions. Renaming-canonical state/task hashes supplement raw hashes. Identical seeds alone provide no leakage protection.

Use TRAIN, DEV, SEALED and named OOD suites. Reserve complete rule/template/renderer combinations when evaluating that dimension; fresh payload seeds with all templates shared only test interpolation. Freeze evaluation generation recipe, salt/seed commitment, source hashes, permitted observations, labels, analysis, primary metric and comparison before optimizing against the set. Preserve create-only seals. Any changed instrument is a new version with an explicit reason.

This host shares one filesystem and account among agents. Separate processes, file permissions and role labels provide procedural custody, not cryptographic or OS isolation from the lead agent. The production/improvement path must never load sealed targets; auditor processes may verify them mechanically and return gate summaries. Reading per-example results spends the instrument for adaptive confirmation. Do not claim the sealed pool is secret from all agents or continue calling it fresh after an adaptive failure analysis. For stronger future claims, use auditor-controlled credentials or an external custodian.

Dedicated suites cover rule reversal, novel words, numeric thresholds on both sides of training range, longer horizons, larger states, input and option order, no-solution/invalid cases, renaming and combined shifts. A separate predicted-world-model-error suite belongs to future experiments; exact-oracle data alone does not establish robustness to learned simulation error.

## Quality and release gates

Every accepted record must pass schema checks, complete oracle evidence, independent-checker agreement, reproducibility and leak/duplicate controls. A known disagreement quarantines the record. Accepted shards are immutable; a correction produces a new version and a supersession receipt. Quarantine never enters training exports.

Before a release, recompute all shard hashes and inventory every split/family/difficulty/perturbation by independent parent. F performs a separate integrity/adversarial review of accepted records, with random sampling seed and reviewed parent count. Report zero disagreements as an observation with its sample size, not proof of zero error. With zero failures in n independent random audited parents, the conservative one-sided 95% binomial upper bound is `1 - 0.05**(1/n)`; 100 parents still permit about 2.95% error. Clustered variants cannot inflate n. Exhaustive deterministic double-checking and sampled human/agent inspection answer different questions.

Release PASS requires 100% schema/evidence coverage, zero unresolved checker disagreements, reproducibility, protected split integrity, known leakage addressed, meaningful category coverage, passing regressions and signed review evidence. Otherwise the release remains experimental. Passing synthetic code tests or family counts does not make model efficacy research-ready.

## Controlled improvement protocol

Freeze a DEV-only comparison before testing a candidate. Record hypothesis, changed source/config hashes, parent-seed list, primary quality/coverage endpoint, minimum effect, permitted regressions, resource budget and all attempted variants. Run baseline/candidate with paired semantic worlds and matched generation budget. Analyze failure causes and balance per stratum. Promote only a measured quality/coverage gain with no critical gate regression; save rejected proposals too.

The initial safe experiment can remove answer-position correlation by semantics-preserving candidate rotation. Count winner positions, entropy, tied/no-feasible exclusions and exact label invariance on held-out DEV parents. Increased position entropy is a generator-quality result; it is not an increase in intelligence. A throughput increase without correctness/coverage improvement is insufficient. Autonomous analysis may propose source changes, but the unattended worker only runs reviewed, hash-pinned code/configuration; it does not rewrite its own checker from its own output.

## Milestones and future model experiments

M0 pins current research, isolates the branch/worktree and records boundaries. M1 delivers a complete small end-to-end oracle pipeline. M2 adds adversarial and split/integrity checks. M3 provides meaningful representatives for A–H, reporting sparse strata honestly. M4 records a controlled DEV improvement and promotion/rejection evidence. M5 starts only after resource/release-operational checks and persists bounded CPU cycles. M6 supplies input-only loaders, evaluation contracts and future experimental designs. `README.md`/status reports must state which milestones actually passed.

Future training requires new authorization. Recommended comparisons are answer-only SFT, explicit predicate/trace SFT with a fixed per-action filler control, consequence-factorized readouts versus scalar evaluators, matched-extra-training search distillation, and a value-of-computation predictor. Require simple copy/first-option/goal-distance floors, matched candidate availability, exact scoring, operation-specific retention controls, held-out semantics/surfaces and at least three independent training seeds for replicated efficacy. Use paired parent-level exact McNemar for binary discordance and clustered paired intervals for regret. Freeze practical effect, multiplicity and stopping rules before confirmation. No significant-loss result is not equivalence; retention needs a prospectively powered margin and correctness measures, not KL alone.
