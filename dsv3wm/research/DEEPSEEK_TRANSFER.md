# DeepSeek mechanisms and a bounded Rineya transfer study

Status: primary-paper review completed; proposed adaptations below are research
hypotheses, not established Rineya improvements. Implementation and execution
status belongs in the package's run records. No historical sealed bank was opened
for this review.

## Primary evidence, with its limits

The reviewed versions are [DeepSeek-V3 v2](https://arxiv.org/html/2412.19437v2)
(Sections 2.1, 2.2, 4.5 and 5) and
[DeepSeek-V2 v5](https://arxiv.org/html/2405.04434v5) (Section 2). The Hugging Face
Markdown endpoints were attempted first; the browser could not consume those
responses, so the original arXiv HTML was read. No secondary summaries support
the mechanism claims.

### V3: verified source summary

Sequential MTP depth `k` combines `h_i^(k-1)` with normalized `Emb(t_(i+k))`,
projects, applies a causal Transformer block, and predicts `t_(i+k+1)` through
the shared vocabulary head. Embeddings/head are shared; depth losses are averaged
and weighted. Modules can be discarded at inference. Section 4.5 tests one added
depth on 15.7B/228.7B MoEs with 1.33T/540B tokens. Most benchmarks improve;
large-model MMLU decreases 67.5 to 66.6. This does not test world states,
frozen-backbone LoRA, or four extra depths.

Bias balancing selects experts using affinity plus bias but weights outputs using
unbiased, normalized sigmoid affinity. Bias updates respond to batch load.
Deployed V3 retains a small sequence-wise auxiliary loss. Section 4.5.3 reports
batch-wise auxiliary loss matching bias balancing validation loss at 1B/3B.

Section 5 uses R1-derived expert data, rejection sampling, SFT and GRPO; rule
rewards verify suitable math/code, while broader tasks use model rewards.
Distillation improves math/code but increases response length. It does not
establish independently verified world-model planning.

Source: [V3 Technical Report](https://arxiv.org/html/2412.19437v2).

### V2: verified source summary

MLA learns a joint low-rank KV latent `c = W_down h`, uses a decoupled RoPE key,
caches latent plus positional key, and can absorb content up-projections into
query/output projections during inference. Query compression primarily reduces
training activation memory. Per-token/layer cache scales as `d_c + d_R` rather
than MHA's `2 * n_heads * d_head`. These are learned attention projections, not
text summaries.

DeepSeekMoE splits FFN experts finely and combines always-active shared experts
with top-k routed experts. V2 uses softmax affinities, device-limited routing and
several auxiliary balance losses. Its full-model throughput/cache comparisons
involve different architecture, size and training; they do not isolate a Rineya
efficiency forecast.

Source: [V2 Technical Report](https://arxiv.org/html/2405.04434v5).

## Existing Rineya assets relevant to transfer

These are repository-record observations, separate from the DeepSeek evidence.
The [BWM entry point](../../bwm/README.md) describes a frozen FP32 0.6B
Titan-Sparse core, goal-blind next-state LoRA, separate proposer, learned-model
beam search, delta-state targets, confidence deferral and search distillation.
The [RCWM report](../../rcwm/experiments/RCWM-T1/REPORT.md) describes chained
four-step self-rollout, rule-coverage LoRA and consequence heads on frozen hidden
states. The [BWM decisions](../../bwm/docs/DECISIONS.md), D-02/D-14, record the
precision requirement and chunked checkpointed cross-entropy.
The [RCWM decisions](../../rcwm/experiments/RCWM-T1/DECISIONS.md), D-28, require
the registered adapter to stay unmerged.

The distinction between earlier search distillation and this assignment matters:
BWM's README explicitly describes learned-world-model verification plus a
domain-agnostic delta-consistency check, without simulator labels. That is useful
prior infrastructure, but it does not itself satisfy the new requirement that an
independent exact simulator establish every retained plan's correctness.

The baseline inventory should pin actual model, adapter, runtime and data hashes
before new training. Read-only reuse of those interfaces must preserve previous
records. This document does not certify checkpoint availability, training-data
licenses, baseline numerical parity or native GPU feasibility.

## Three distinct predictive mechanisms

The following designs are our adaptations. Token position and environment time
are different axes. A four-token objective can concern punctuation or state keys
within one next-state answer; it is not evidence about four environment actions.

| Mechanism | Conditioning and target | Main scientific question | Estimated implementation cost |
|---|---|---|---|
| A0: next-state LM | Current state, rule specification and one action; serialize the next state | Can the inherited forward interface be reproduced on permitted development cases? | Low if validated training/inference utilities can be reused |
| A1-sequential | Causal token-prefix representation plus earlier target-token embeddings at successive auxiliary depths; predict the next token beyond that prefix | Does auxiliary training improve the main adapter when the added modules are discarded? | Medium to high: shifted module inputs, extra causal blocks, gradient and alignment tests |
| A1-independent | The same main-model hidden vector at position `i` feeds small independent projections for targets at `i+2` and/or `i+4` | Does a cheaper shifted-token auxiliary objective improve dynamics? | Medium: low-rank projection modules and shifted response masks; shared vocabulary projection remains expensive |
| A2-direct | Initial state, rule specification and action prefix through horizon `h`; independently serialize `s_h`, for `h` in `{1,2,4}` | Does explicit consequence supervision improve free-running state accuracy? | Medium: trajectory builder, separate causal records, horizon-weighted loss and direct/rollout evaluators |
| A2-rollout-loss | Training rollout uses model predictions for intermediate states; horizon losses compare its outcomes with labels | Does training on its own errors reduce compounding mistakes? | High: generated-text parsing/discrete feedback complicates differentiability, memory and supervision |

Use explicit names in configuration and reports. An independent shifted head
must not be called an exact implementation of sequential DeepSeek MTP. If the
native pilot implements only independent heads, mark sequential MTP as deferred.

Define token horizon `q` as the distance from main hidden position `i` to target
`t_(i+q)`. Then `q=1` is ordinary next-token prediction; total future-token
horizons `{1,2,4}` correspond to auxiliary offsets `{2,4}`. A contiguous
sequential version reaching `q=4` requires auxiliary depths `1,2,3`. Calling
this "four additional predictions" would instead mean total horizon 5 and must
use a different configuration. State horizons always count executed actions.

For a frozen base, all new trainable capacity must live in unmerged adapters or
named auxiliary modules. Shared frozen embedding and output-head weights still
permit gradients to pass back into adapter-produced hidden states. Do not detach
those hidden states for an auxiliary objective intended to improve the main
adapter. Conversely, training a head on detached captures tests readout quality,
not representation learning through auxiliary supervision.

A lightweight independent head can use a residual low-rank transformation of
`h_i`, followed by the existing frozen output projection. Keeping the projection
shared avoids another vocabulary-sized weight matrix; it does not eliminate its
per-head compute or logit activation cost. Sequential blocks add much more
capacity than tiny residual heads. Those are experimentally different arms even
when their target tokens match.

For A2, construct a separate input for each horizon from the same trajectory.
It contains only `s_0`, rules and `a_0 ... a_(h-1)`; target `s_h` is kept in the
loss label. Within-target language teacher forcing is allowed during training.
Gold `s_1` and `s_2` must never be conditioning inputs for an independent
four-step prediction. Ground-truth current states in separate one-step training
examples are legitimate supervision and should be counted as such.

At evaluation, distinguish:

1. **Teacher-forced token loss:** likelihood of target answer tokens given their
   earlier correct answer tokens. This is a training diagnostic.
2. **Direct consequence prediction:** generate `s_h` once from initial state and
   action prefix. No intermediate state labels are consumed.
3. **Free-running rollout:** call the one-step model on its own previous decoded
   state and the next action. Errors accumulate naturally. No oracle reset.

P1's instrument must state which of the last two is primary. Report both if
available. An A2 direct head need not improve the A0-shaped one-step interface;
an inference change must be measured, not hidden behind one primary metric.

## Loss design and causal correctness

Recommended independent-head objective:

`L_A1 = mean_valid(CE_next) + lambda / |Q| * sum(q in Q) mean_valid(CE_q)`

where `Q` contains only auxiliary token horizons. Each mean uses its own valid
response-target count. Mask prompt targets, padding, positions past answer end
and transitions across packed-example boundaries. Prevent empty masks from
producing NaN; reject a completely unsupervised example. Record both raw target
counts and weighted loss components so a longer answer does not silently change
the relative weight.

Recommended explicit-horizon objective:

`L_A2 = L_state1 + lambda2 * L_state2 + lambda4 * L_state4 + beta * L_legality`.

Loss weights, response reduction and legality definition must be declared.
Legality at a state/action boundary must be predicted from information before
the legality value or resulting state is exposed. Count explicit applicability
flags as extra supervision if a representation ablation supplies them.

Start with full-state and delta-state as a small development comparison.
Canonical reconstruction from predicted deltas is necessary for state accuracy:
reject deletions of facts that are absent, duplicate/contradictory edits and
malformed outputs by a declared generic schema policy. A structural check must
not secretly execute the domain transition rule. Report the model's legality
flag and any postprocessing guard separately, because a correct guarded plan
can conceal an incorrect learned legality prediction.

The correctness suite should include these discriminating tests:

| Trap | Intervention/test | Required observation |
|---|---|---|
| Future attention | Replace every token strictly after main position `i` for an independent head | Logits at `i` are invariant in deterministic eval mode |
| Sequential target leak | At auxiliary depth `k`, alter only positions after `i+k` | Logits predicting `i+k+1` are invariant; altering allowed earlier token embeddings is a sensitivity control |
| Wrong target shift | Hand-labelled short sequence with distinct tokens; vary each horizon | CE target is exactly `i+q`; prompt/padding/end/boundary positions are ignored |
| Packed-row contamination | Alter all tokens of a different packed record | The first record's predictions and losses are invariant |
| State-horizon leakage | Change stored gold intermediate/final states without changing the allowed input | Generated predictions are invariant; offline scores change when labels change |
| Oracle fallback | Replace oracle callable with one that raises during learned inference/search | Generation and candidate selection still complete; scoring occurs in a separate phase |
| Unintended backbone update | Inspect `requires_grad`, optimizer groups and before/after parameter hashes | Only named adapters/auxiliary modules change; base stays frozen and unmerged |
| Missing adapter gradients | Backpropagate auxiliary loss alone on nondegenerate fixtures | Main adapter receives finite nonzero gradients; a detached-head control does not |
| Loss normalization error | Compare chunked CE and simple dense CE on small CPU tensors | Values and gradients agree within preregistered FP32 tolerances |
| Baseline drift | Disable experimental modules; compare inherited forward outputs on permitted fixtures | Baseline output is unchanged; no accidental adapter or precision switch |
| Teacher-forcing confusion | Corrupt predicted intermediate states in a free-running chain | Later inputs reflect that corruption and never reset to stored labels |
| Numeric incompatibility | Inspect model/adapter dtypes, autocast and TF32 configuration | FP32 remains enforced end to end, including auxiliary operations |

These are implementation tests, not demonstrations of native Rineya efficacy.
Run deterministic causal interventions with dropout disabled; inference-mode
invariance and training-gradient correctness are separate checks.

## Controlled pilots and confirmation recommendations

Begin with A0, one explicitly named A1 variant, A2 and the A5 extra-one-step
control. Fix the parent model, LoRA targets/rank/initialization, dataset parent
pool, optimizer and development selection rule. A frozen-core native smoke is
required before treating a successful CPU toy run as a feasible pilot.

Match more than optimizer steps. Record input tokens, valid target tokens,
unique parent trajectories, supervised transition labels, activated/trainable
parameters, forward/backward calls, vocabulary projections, elapsed active
compute, duty pauses, peak dedicated VRAM and a credible FLOP proxy. A2's access
to intermediate labels and A1's repeated vocabulary projections are material
advantages/costs. Report unavoidable mismatches.

A5 should spend the candidate's measured compute allowance on additional
one-step updates using the same permitted parent pool. Where A2 supervises
intermediate transitions, include a one-step control with those labels rather
than giving A0 fewer labelled events and attributing all improvement to the
horizon objective. Parameter-matched extra-capacity and randomized-auxiliary
controls are useful only after the primary pilot identifies a signal; do not
expand an uncontrolled sweep. Randomize auxiliary labels within training split
and comparable domain/length strata, never across confirmation records.

Select hyperparameters using development results only. Decide a small fixed
candidate set, practical improvement and retention thresholds, then freeze the
model/data/code/analysis/search configuration before independent confirmation.
Power should use pilot paired discordance and parent clustering; a numerical
sample-size promise without that information is speculative.

P1 should use one preregistered four-step trajectory per independent parent for
exact paired binary testing, or a prespecified cluster-aware alternative for
multiple trajectories. P2 should score a complete candidate plan with a separate
exact evaluator after search has finished. Budget both model calls and generated
tokens or an explicit compute proxy; equal beam width alone is insufficient.
Fix malformed-output, timeout, partial-plan and invalid-action scoring in
advance. Report plan validity, final-state accuracy and goal success separately.

Exact McNemar tests compare matched outcomes of fixed trained artifacts. They
do not make repeated seeds on the same instances independent. One defensible
small study preregisters seed 0 for the artifact-level primary comparisons and
seeds 1/2 for replication gates, reporting all seed effects. Correct the whole
registered claim family: one selected candidate against A0 and A5 on P1 and P2
means four tests, not only two. Promotion should require both meaningful
co-primary gains and retention/budget gates. Broader claims about training-seed
robustness need seed-aware uncertainty; resample shared parent IDs and training
seeds as crossed units, not `seed * case` rows as independent data. Three seeds
remain a weak estimate of seed variability. A nonsignificant result is not an
equivalence result.

Group all trajectories and surface variants of a parent into the same split.
Check exact fingerprints plus domain-appropriate semantic signatures, including
object-renaming equivalence. A Blocksworld canonical signature must include the
relational initial/goal structure; name-normalized strings alone are not a
complete isomorphism test. Sealed benchmark labels and historical frozen plans
must not be visible to model selection, training or failure debugging.

Targeted development counterexamples should cover reordered keys, equivalent
encodings, renamed objects, unseen rule compositions, novel numerical ranges,
invalid/ambiguous actions, longer action sequences and misleading rule cues.
Include copy-input, first-action-only and last-action-only diagnostic baselines.
Count false-copy failures on examples whose true state changes. An accuracy gain
concentrated on no-op/copy cases is not strong evidence of learned rule dynamics.

## Verified trajectory distillation

Reuse the existing proposer/search interfaces, while adding an independent
training-split verification stage. Generate diverse candidate plans, rank them
with the learned forward model, and run the exact simulator on candidate plans
without using confirmation instances. Retain only plans whose independently
executed transitions are legal and whose final state meets the goal. Keep
verifier implementation/version/hash, input parent, selected plan, per-step
legality and final result in the training provenance.

Keep rejected plans as labelled hard negatives only with an explicit training
objective. Verifier-approved final success does not imply every learned
intermediate state was correct; retain state discrepancies as an audit. Empty
plans on already-solved goals, cycles and duplicate trajectories need declared
handling. Verify all retained labels rather than extrapolating a sampled
precision estimate into a claim that every plan is exact.

Compare A3 (A2 plus independently verified distillation) with A2 and A4
(one-step baseline plus the same verified distillation). Match distilled parent
pool and number of accepted labels. Track first-action optimality, complete-plan
validity, pass@k, candidate diversity and learned-model queries per solution.
Teacher-generation and verification compute are part of the training cost.
Evaluation can use the exact simulator to score a submitted plan; if it repairs
or selects the plan online, label that system simulator-assisted and report it
as a separate comparator.

This transfer may fail if search proposes only a narrow easy curriculum,
verification rejects almost everything, labels teach token templates rather
than rule application, or longer plan outputs consume more inference budget.
Verified planning quality and state correctness can move in different
directions; both must be measured.

## Conditional routing and memory investigations

| Follow-up | Rineya starting point | Proposed adaptation and test | Failure assumptions / complexity |
|---|---|---|---|
| H4 adapter routing | Separate world-model/proposer adapters and distinct domain error patterns already exist | First build a per-parent error/complementarity matrix. Then compare universal adapter, best specialist, fixed/random/learned routing and an equal-cost ensemble. Apply any load-bias update outside gradient loss; freeze its deployment behavior. | Domain demand need not be uniform. Forced equal routing can degrade expert suitability; routing metadata can be a shortcut. Small batches make utilization noisy. Medium to high integration cost; no large MoE rebuild. |
| H5 compact history | Existing learned-model planning repeatedly handles serialized states and branching prefixes | Measure actual KV bytes and planner bottleneck first. Test safe prefix reuse, compact state records and selective history independently. Compare equal query/compute budgets and state fidelity. | Text compression does not establish KV compression. A Markov state may make history irrelevant; summaries can omit rule-relevant facts. Prefix-cache sharing can accidentally mix branches. Low to medium for representation/cache reuse; high for learned latent cache or attention conversion. |

For H4, reported oracle-routing performance is an upper bound that uses correct
outcomes, not a deployable router. Specialized adapter weights must remain
separate; a top-2 composition requires numerical and semantic compatibility
tests rather than assuming two individually good adapters combine cleanly.
Measure utilization by domain and OOD strata, and use a declared universal or
adapter-off fallback when routing confidence is low.

For H5, full MLA replacement would change attention projections, cache format
and positional behavior of the frozen core. It would require retraining and
new qualification, so it is outside the primary adapter study. FP8 and unsafe
cache quantization are likewise outside its numerical contract. Start with
measurements that distinguish actual cache allocation, model weights, temporary
activations, serialized record size and shared host-memory paging.

## Current recommendation

Prioritize explicit state horizons and a clearly labelled lightweight shifted
token pilot under the inherited FP32/unmerged contract. Their comparative
value is unknown until native controlled data exist. Preserve the sequential
architecture as a distinct optional arm instead of silently claiming fidelity.
Extend verified planning only after correctness and pilot gates pass. Routing,
compressed predictive memory and changes to the dense core remain conditional.
No mechanism is ready for promotion from literature evidence or toy tests alone.
