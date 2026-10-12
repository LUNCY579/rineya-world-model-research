# GCDL-001 research architecture and reference audit

11 October 2026, America/Chicago. Scope: dataset-generation infrastructure, exact executable labels and bounded CPU quality research. No Rineya model was trained, evaluated or downloaded for this audit. Existing native output files were read and selected counts recomputed; that is different from independently executing native experiments or VAL.

## Snapshot and precedence

The isolated worktree started from `8f41180a7f207a359541094871ab7ece293363c1` (9 October). After the coordinator fetched the configured remote and rebased the dedicated branch, this audit uses main `ece566dfa31e1e49f93efb264505c3c622c5dad9` (11 October, 16:08:47 CDT). The primary checkout's older `a223884...` and the 9 October complementary bundle are historical sources. The live complementary programme was read with `git show` from remote-tracking ref `origin/research/rineya-complementary-intelligence-v1`, pinned to `11981b66aba399be927109a74639f7c73f8232e5`. No checkout or programme file was changed to read it.

`reference_snapshot.json` contains SHA-256, size and Git blob identities for the inspected reference files, source commits, exact paired count recomputations and verification scope. These are audit-time pins, not a promise that other programmes will stop evolving. Re-read changed source records before adopting a new generator hypothesis, while keeping already released provenance immutable.

## User-provided findings, checked against current records

| Finding | Current evidence | Scope and limits |
|---|---|---|
| Original BWM search solves 73/110 versus proposer 5/110 | P-08/P-10, B2 raw plan files and summaries; locally recomputed 73/5, 68 search-only wins, 0 direct-only wins, exact two-sided McNemar `6.776263578034403e-21` | Frozen domain-trained specialist, learned F plus beam and parser checks, one training seed; execution feedback absent during search; not generalized decision intelligence |
| Self-distilled proposer reaches 13/110 without search | P-13/H1 and raw H1 plans; recomputed +10/−2 versus original 5, p `0.03857421875` | 12/13 successes within 6–10-block distillation range; 1/74 at 11–15. Token cap may understate capability but was not proven the cause of failure |
| Swapping it into frozen search reaches 88/110 | P-14/H2 and raw H2 plans; recomputed +20/−5 versus 73, p `0.004077315330505371`; all 110 recorded success fields equal recorded VAL fields | 95% interval recorded as 0.713–0.870; one seed. No same-extra-training Pa_v2c beam arm on hard. B3 design followed B2 hard findings. Two recorded leak audits found no data route; pretraining contamination remains unknown |
| Ledger adaptation improves four-step regret and exactness | RCWM PREREG-2 analysis JSON: regret `0.5444576494 -> 0.0560061043`; exact state `0.0291666667 -> 0.6975`; 150 fresh parents, fixed T2 s0 | Same event generator/marginals and frame, horizon ≤4, unmerged FP32 adapter; frame adaptation and targeted rule learning not separated |
| Later ledger OOD exposes the `void` token cue | PREREG-3 analysis verdict TRIGGER; three T2 seeds, 1,500 one-step rows and 80 chains | Unseen non-void statuses 0.01–0.18, trained void 0.99–1.00; numerical and key-order failures. Finite named strata do not characterize every possible rule/string |
| Hidden-state consequence heads do not consistently beat chaining | PREREG-1 and RCWM report | Chaining regret 0.448 across ledger/register, beats equal-label readouts that were near/below answer-free floors. Prerequisite chaining remains poor; abundant symbolic labels beat chaining on ledger. No universal latent-head inferiority claim |

The initial local snapshot could verify only 73/5; it lacked the later 13/88 exports. The fetched 11 October snapshot resolves that apparent contradiction. All three BWM counts are now verified from recorded per-instance outputs. The locally recomputed VAL agreement means agreement with the stored field; no VAL binary or native model was rerun here.

## Latest findings that change dataset design

**BWM search verification contains an executable support check.** F-09 (11 October) is explicitly exploratory pending a registered replay. On queried illegal actions, F's own `legal` flag falsely says legal for 798/4,066 B2-hard keys and 657/2,363 H2-hard keys. The deletion-support parser rejected these outputs because a deleted fact was absent. The scratch replay found zero accepted false-legal outputs and zero wrong next states on those hard forests, but the apparently perfect verified-plan precision belongs to the combined learned F and code check. GCDL must execute all teacher/search plans in its true environment and compare a separate oracle. It must not treat a model's own verification flag as gold. The same parser safeguard may be ineffective in another domain.

**Search-to-proposer distillation has useful controlled evidence, with limits.** B3 fresh S-direct compared Pa_v3 22/110 against matched-extra-training Pa_v2c 8/110 (+18/−4, p≈0.0043), and 211/211 model-verified teaching plans were valid under analysis-time exact execution. The hard H2 contrast lacks that extra-training control under beam. GCDL records search-best versus bounded/exhaustive optimal proof separately, and keeps unsuccessful/omitted-candidate cases to avoid success-only selection bias. Pa_v3 retained original simulator-labelled training; “no simulator labels” covers only the distillation increment.

**BWM surface invariance is still partial.** Key-order retraining P-12 improves held-out reversal from 0.454 to 0.917 but misses its 0.95 bar and drops canonical from 0.992 to 0.963. Illegal-action handling worsened. Track invariance and legality separately; reward-diverse formatting can damage existing operations. Original F remains format-locked and frozen planners use the canonical rendering.

**RCWM planning gain is now measured, not queued.** PREREG-LB on 180 in-range ledger instances reports T2 success 0.706 versus OFF 0.078 and ORACLE 0.889; T2/ORACLE 0.794 fails the 0.80 near-oracle point bar. About 5% of imagined verified plans fail exact execution (precision 0.952), and held-out-surface success is 5/180. Consequently, corpus labels cannot inherit T2's training-surface competence as a universal simulator certificate.

**Positive rule predicates can coexist with failed release gates.** RCWM 5a's bit representation wins a DEV screen and reaches roughly 0.84 exact four-step chains in surface, but conversion from decimal is code-side assistance and held-out surface accuracy is much lower. 5b's register trace learns the conditional predicate while its registered overall verdict is NO-GO: mix remains near zero and swap accuracy collapses from OFF 1.000 to trace 0.000. Narrow training lacked swap controls. Every future adapter dataset needs control-operation coverage and separate acquisition/retention scores. GCDL itself does not train an adapter or certify retention.

**Correctness-based retention is stronger than KL and still scoped.** R25's 450-prompt confirmation gives registered GO to T2 s0 and two original BWM adapters versus Titan-Sparse OFF, over five selected categories under either-grader no-go semantics. Titan-Sparse itself loses unit conversion versus Supa (49→30/75). One BWM proposer answers five unanswerable twins (abstention 0.933) despite pooled positive primary net. These passes do not establish a universal ≤2-point loss, and no Pa_v3 retention pass is inferred. GCDL future experiments require powered margins and operation-specific controls.

## Complementary programme: historical plan versus current evidence

The external bundle `rineya_complementary_research_2026-10-09/` is a private planning package pinned to `a223884...`. Its utility tests and sample scores are synthetic, and its proposed GPU caps are not authorization or measured gains. Its useful reusable principles are semantic-parent custody, separate ability tiers, matched-label/matched-compute controls, fixed per-action filler policy and paired parent-level analysis.

The live remote branch has progressed beyond that package. Its README and experiment-results record say:

- RCX E01N, three native adapter seeds per arm, sealed comparison: REVISE. Counterfactual/paraphrase-diverse data improves held-out wording by about +0.090, below the preregistered +0.10 point threshold. Unseen status behaviour is rule-like on trained wordings, yet restrictive positive paraphrases still fail (about 14% correct no-ops). No retention/dense-model claim follows. The consumed wording battery is spent.
- E07b's relational-value hybrid improves an external exact-planning system; this is assistance evidence, not unaided 0.6B behaviour.
- E11a's mined macros improve external GBFS+A3 from 52 to 77/120 at 200 expansions (+25/−0), with random/shuffled controls. E11b is REVISE and does not beat ready-first ordering. Macro-specificity depends on the heuristic and cost unit; primitive steps and heuristic calls differ.
- RCX E01R has new secret/sealed wording pools and queued diagnostics. GCDL did not read those pools or labels. Existing RCX CPU/GPU coordination files and jobs remain untouched.

These branch findings were read as recorded evidence; native training, macro experiments and all statistics were not rerun. Fresh standalone GCDL tasks reduce direct interference, but do not establish that no semantic overlap with every historical private corpus is possible.

## Contradictions, corrections and chronology

1. RCWM report's top Date remains 6 October, while later status/sections include confirmations through 10–11 October. Older “confirmatory statements limited to PREREG-1” and queued-prose sentences are stale. Use dated confirmation JSON and latest decision entries; preserve the discrepancy rather than editing another programme's records.
2. The 8 October joint roadmap and 6 October BWM interim report describe runs as pending that current P-10/P-14 completed. The complementary bundle's older pinned snapshot does not disprove later results.
3. P-14 corrects earlier source claims: o1-mini “10% zero-shot” was actually verifier-assisted; Pa_v3 was not trained only on ≤6 blocks; the hard references are not aggregate-only; truncation was not proven the cause of the limited H1 gain. GCDL uses these as source errata and makes no new frontier-model comparison.
4. “Frozen settings/compute unchanged” in P-14 means the search configuration stayed fixed. Observed averages differ: ~8.2k proposer tokens/198 unique F queries for H2 versus ~8.0k/187 B2. It does not establish matched actual latency/FLOPs. Search code also gained default-neutral hooks; recorded regression shows original default behaviour reproduced.
5. Mirror query-count erratum E-01 changes the cost from the final-attempt-only summary 183.6 to 215.9 unique F queries per instance across warm-cache restarts. Account for all attempts/cache accesses in future computation labels.
6. BWM's hard-minus-mirror contamination flag (+10.9 points) remains raised; post hoc goal-shape reweighting explains much of the gap but does not retroactively remove the preregistered flag.
7. Numeric RNG seeds containing `202611` or `202614` are seeds, not observations made in future calendar months. All user-facing chronology here uses the client America/Chicago context; source UTC stamps are retained as written.

## Pipeline architecture and verification boundary

The executable design is:

`fresh semantic task -> primary exact execution -> decision labels -> separate executable replay -> schema/integrity/adversarial gates -> immutable accepted shard or quarantine -> manifest/checkpoint -> DEV quality analysis`.

Generation owns task proposals; oracle execution owns truth; label construction owns explicitly defined rankings/probabilities; review owns acceptance; operations owns persistence. Use standard-library CPU code where sufficient. No model call is needed for gold. `DATA_CONTRACT.md` defines exact fields/formulas; `RESEARCH_PLAN.md` defines hypotheses and evaluation; `AGENT_ROLES.md` defines four-slot sequencing and review conflicts. Operational commands, resource ceilings and measured release state are delivered separately by the coordinator.

Candidate availability is equalized within matched decisions. Costs and hard constraints are explicit and observable. Multiple goals never mutate transition rules. Unknown hidden state uses specified possible worlds, not the actual hidden outcome as policy input. No-candidate-success is distinct from proven global impossibility. Value-of-computation targets compare realized matched configurations, while policy features include only information available before that compute. Oracle expansions supply a transparent synthetic cost unit; they are not actual future model latency or FLOPs.

Protected splits are assigned before serialization and maintained across cycles. Group complete counterfactual worlds, goal variants, trajectory suffixes and perturbations. Raw byte hashes supplement semantic/isomorphism and template lineage checks. Independent audits assess oracle agreement and diverse parent coverage; output totals cannot establish generalization. Sealed process separation is procedural on this shared host. Generators/improvement agents do not inspect sealed targets to choose changes, and any adaptive use spends the instrument.

## Remaining programme limits

Finite synthetic ledger/Blocksworld tasks do not cover open-ended language, social goals, real-world causal uncertainty or arbitrary rules. Closed paraphrase templates certify only those renderings. A bounded exhaustive plan search is optimal only within its named horizon/budget semantics. A release needs real counts, source/shard hashes, all acceptance gates, separate review evidence and audit sample uncertainty. It remains experimental if a critical gate is incomplete.

Fresh review contexts and two code implementations reduce correlated errors; shared account, files and model ancestry prevent an absolute independence claim. The persistent runner can generate and mechanically audit reviewed code while the host is awake and resources are healthy. It cannot guarantee uninterrupted 24/7 uptime or perpetual fresh LLM scientist reviews. Controlled source-changing research remains session/service dependent. Storage/failure ceilings terminate or pause safely and preserve accepted evidence.

No completed dataset infrastructure result demonstrates improved Rineya decisions, calibration, reduced planning cost, broad retained skill or general transfer. Those require separately authorized future matched experiments, new sealed instruments, multiple training seeds and correctness-based retention.
