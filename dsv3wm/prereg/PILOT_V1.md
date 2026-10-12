# DSV3WM pilot protocol v1 — frozen before pilot outcomes

Date: 2026-10-11 (America/Chicago). Parent repository baseline
`aeb19ab17128d884c4457ce59db6e926ceb13a7a`. This is a development pilot,
not the independent confirmation study. All historical confirmation banks and
published test plans are excluded from construction, training and tuning.

## G0/G1 prerequisites

Native parent: Titan-Sparse Experimental 1.0.0 devfreeze, SHA256
`a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284`,
596,049,920 parameters. Execute FP32, TF32 off, no autocast or adapter merging.
Reuse BWM's immutable loader, tokenization, chunked CE and lease. A new q/v
rank-4 adapter is trained; this capacity differs from historical BWM adapters.
Historical 73/110 or 88/110 results are contextual, not matched pilot controls.

GPU lease failure allows a bounded **native CPU** smoke/pilot, subject to its
measured cost. CPU execution of the actual parent is native model evidence;
tests with tiny artificial tensors establish implementation correctness only.
No remote work, paid compute, model upload or merge is authorized.

## Mechanisms and budgets

- A0: next-state teacher-forced response CE, plus explicit last-action legality
  loss when beta > 0. Setting beta=0 provides the original CE-only baseline.
- A1: A0 plus sequential positionwise bottleneck auxiliary modules, two total
  future tokens initially (ordinary head plus depth one), weight 0.3. A four-token
  option has three auxiliary depths. These replace V3 transformer modules;
  they are not an exact V3 implementation. Auxiliary modules are absent in inference.
- A2: independent labels at horizons 1/2/4, weights 1/.5/.5, conditioned only on
  s0/rules/actions up to that horizon. No teacher intermediate states are inputs.
- A3: A2 plus plans ranked by learned predictions then independently verified.
- A4: A0 plus the same verified planning examples.
- A5: more A0 steps to control A2's extra decoder forwards and supervision.

Initial operational screen: A0/A1/A2, seed 0, batch 1, 4 optimization steps,
lr 1e-4, rank 4, no dropout, same seed/order and fresh training data. A5 uses
12 steps, three times A0/A2's optimizer steps to match A2 decoder-forward count.
This does NOT exactly match input/output tokens or runtime. All differences,
auxiliary parameters, measured tokens, forwards, duration, memory and a scoped
decoder FLOP proxy are reported. A1 has extra trainable parameters; no superiority
claim can isolate MTP until equal-capacity and matched-compute controls run.

The pilot uses three domains in fresh data; native operational training/evaluation
may select a preregistered register-only subset to bound CPU duration. This subset
is an implementation/feasibility screen and cannot establish cross-domain efficacy.
Full-state is initial representation; delta/mixed modes and legality-off,
shared-head and shuffled-target controls are implemented and unit-tested, but
are run only after a feasible useful pilot. No uncontrolled parameter sweep.

## Data and leakage controls

Freeze cryptographic manifests for training/development/public confirmation
inputs/private confirmation labels before training. Check exact duplicates,
parent-world duplication and renamed Blocksworld isomorphisms across splits.
One primary episode per independent parent. Parent canonicalization, schema,
source hashes and numerical ranges are recorded. Dynamic validation never calls
historical sealed-bank helper functions. Dynamics prompts exclude goal/labels.

Unit gates: causal future-token intervention, prompt and padding masks, empty
mask behavior, CE value/gradient parity, disabled-mechanism numerical parity,
loss weighting gradients, invalid-action semantics, independent rollout without
teacher state, split disjointness, negative-verifier rejection and budget tests.

## Pilot measurements and decision

Report teacher-forced CE separately from greedy direct horizon 1/2/4 prediction
and recursively chained one-step free-running accuracy. Malformed outputs count
as incorrect, with no oracle correction. Report legality, per-step state errors,
predicted-only planning and model calls; exact simulator scoring occurs AFTER
selection. Long/invalid/renamed/key-order cases are development stress tests.

G3 usefulness requires finite stable training, budget feasibility, and a candidate
showing both improved four-step free-running development correctness and useful
planning proposals relative to A0, with no observed retention regression.
Four steps and a very small evaluation are insufficient for a scientific benefit
claim: a positive operational signal only authorizes a larger development study.
No G4/G5 promotion from mere token-loss reduction, tests or native smoke.
Without signal or resources, preserve negative/inconclusive results and defer
G4–G7. H4/H5 remain conditional and are not implemented as a new native attention/MoE.

## Confirmation study specification (pending resource qualification)

Co-primary P1 = exact final state of a **free-running four one-step rollout** on
fresh independent parents. Direct h4 accuracy is secondary. P2 = exact all-legal
goal success of a learned-prediction planner with a fixed decoder-token and
forward-call budget. No exact transition/legality check may influence selection.
Independent verification of selected final plans is scoring only. Define equal
budget as BOTH total input+decoded tokens and model forwards, with truncation
on either limit; wall time is separately reported. Cache policy frozen.

Only one pilot-selected candidate/config is frozen; compare to A0 and A5.
Practical criterion: at least +5 percentage points on BOTH P1 and P2 against
both controls, plus Holm-corrected p<.05 across four paired exact McNemar tests,
and no prespecified retention failure. This proposed threshold is not supported
by a result yet. Seed 0 is predesignated for artifact-level tests; replicate seeds
1/2 must have positive effects and satisfy retention. Do not pool shared test
instances across seeds as independent observations. Report Wilson intervals,
paired effect estimates, crossed seed/parent bootstrap for secondary continuous
metrics, and weak precision with only three training seeds.

After pilot, freeze independent bank size using discordance/power simulation,
hardness/object/rule/goal cohorts, exact budgets, decoder/search settings,
retention margins/cases, statistical code hash and all artifacts BEFORE opening
new confirmation labels. Until those fields and three-seed training are frozen,
this file is not a complete G5 registration and the confirmation CLI must refuse
access. Non-significance is not equivalence. No reuse of exposed pilot parents.
