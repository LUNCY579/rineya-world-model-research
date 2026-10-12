# Prospective conditional legality calibration diagnostic

Status: instrument preparation. No probability trace, native measurement,
qualified source/artifact freeze or calibration conclusion exists at registration.
The pending H3 student readouts retain priority and their protocols are unchanged.
This diagnostic uses the ordinary CPU11 admission path with its unchanged floor,
CPU4, seed0, FP32/TF32-off, unmerged adapters and shared-lane boundary checks.

## Defined endpoint and fixed inputs

Use the existing recovery24-parent development collection, SHA256
`7fe3c996a8ab4d00fe623edc8ad99f643ebc44f7b36f4a8935bd3c21ea012cc3`.
For each parent, render the unchanged legacy h1 prompt from initial state, rules
and the first action only. Do not supply a goal, intermediate/future state,
gold legality, solver output or confirmation input. Append the fixed assistant
schema prefix `{"legal":` and evaluate both canonical boolean continuations
with a comma delimiter. Exclude EOS and special tokens. Verify that encoding
each concatenated prompt/continuation preserves the exact prompt-token prefix;
refuse nonmatching boundaries, empty targets or overlapping candidate token
paths. Overlap means identical token paths or either path being a prefix of the
other; the two candidates must be prefix-free. No tokenizer-boundary repair is
permitted.

The schema prefix fixes legality as the first response field, before any state
value. It changes the response context relative to ordinary legacy generation.
The endpoint is therefore the conditional boolean diagnostic under that prefix,
not calibrated correctness of naturally generated state replies or plans.
Only initial h1 legality is scored; absorbing later-step legality is outside it.
The fixed cohort has16 true and8 false labels. This permitted DEV inspection
does not change parent selection or any existing evaluation endpoint.

## Probability capture and delayed scoring

Let `l_true` and `l_false` be the autoregressive log likelihoods of the two
complete delimiter-terminated token paths, conditioned on the common prefix.
Capture both raw log likelihoods and token paths before accessing gold labels.
Preserve their raw probability masses, their sum `m`, and the complementary
other-path mass. Keep native model/head computations in FP32, then calculate
log-softmax and the literal likelihood sum in float64 for probability accounting.
The quantity `1-m` includes all alternatives outside the two
canonical comma-terminated prefixes, including valid alternate serialization.
It is not the probability of invalid JSON. Normalize with stable log-sum-exp:

`q = exp(l_true) / (exp(l_true) + exp(l_false))`.

The value `q` is conditional on choosing one of the two declared canonical
paths. Low canonical-path mass must remain visible; neither it nor normalization
establishes the validity of ordinary generated replies or deployment confidence.
Reject impossible
mass, nonfinite values and undefined normalization. Preserve finite log-space
information where direct exponentiation underflows. Record every failed case
and incomplete capture; never fill missing probabilities or scores with zero.

Fix all24 probability traces before scoring `labels.legal[0]`. No exact
transition rule is called to repair or select a prediction. Scoring uses the
already permitted labels only after capture and records that boundary explicitly.

## Descriptive analysis and provenance

Report per-parent raw masses, `q`, true label, Brier score and natural-log binary
loss. For the reported binary log loss, clamp `q` to[1e-12,1-1e-12] before
applying the natural logarithm; record which cases are clipped. Also preserve
the unclipped conditional log-space loss from `l_true-log(m)` or
`l_false-log(m)` for the gold class. Use ten fixed equal-width probability bins
on[0,1], assigning1 to the final bin. Report count, mean probability, empirical
legal frequency and absolute reliability gap for each nonempty bin. Empty bins
remain empty. Report aggregate Brier/log loss and weighted absolute bin gap
for all24 parents and the unchanged12/12 inherited-F-known/unseen-limit cohorts.
The cohort names do not establish unseen dense-backbone exposure.

These are descriptive one-seed development diagnostics with no threshold
selection, promotion criterion, equivalence claim or confirmatory hypothesis
test. Case counts do not establish cross-domain or training-seed calibration.
If a capture fails, summarize successful captures with their denominators and
the full failure count; do not silently report a complete24-case result.

Before any native execution, a separate prospective freeze must bind this
protocol, new driver, all scoring/lineage/loader dependencies, fixed DEV bytes,
actual warm artifact/run/config, parent core and sidecars. Warm A0/A1/A2/A5 are
eligible diagnostic artifacts; current code must validate their actual training
history rather than trust a label or supplied path. Existing freezes cannot be
repurposed as registration of new source. The new implementation needs pure
boundary/probability/causal/delayed-label tests and independent review first.

Preserve per-path actual token work and physical forward counts, elapsed time,
host-memory resources and admission records. Scoring forwards are separate
from autoregressive generation requests and planning queries; no FLOP or GPU
allocation claim is permitted without measurement. Use fresh exclusive output
paths and bounded retries under the unchanged admission rules. Sealed labels,
historical frozen plans, owner assets and published prior outputs remain untouched.
