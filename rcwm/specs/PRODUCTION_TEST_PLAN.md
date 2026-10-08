# Production test plan: not a list of tests already passed

## Mandatory before G0-native

Artifact identity; tokenizer roundtrip; prefix/bulk logits; cache/no-cache capture; padding/position
invariance; selected hidden-layer parity; batch independence; deterministic replay in declared scope;
resume/accumulation; no unintended trainable parameters; base hash immutability; memory admission.

## Data and causality

Counterfactual-family split; source deduplication; goal/paraphrase inheritance; TRAIN-only statistics;
future outcome perturbation; future-action causal invariance; behavior-suffix rejection; timeline
availability; plan provenance; snapshot equivalence; terminal/truncated target handling; no missing-label
as-zero; status failure retention; immutable candidate sets.

## Model and control

Tiny-set overfit; shuffled-action negative control; collapse/variance monitoring; rare-failure decoding;
new-goal readout protocol; candidate regret; OOD and calibration; exploitation under search width;
real-environment re-anchoring; permission checks; invalid preconditions; independent verifier;
end-to-end budget and protected behavior NI.

## Parallel/native extensions

DAG acyclicity; observation-availability masks; asynchronous completions; resource locks; cancellation;
joint outcome consistency; native mode isolation; typed embedding/vocabulary safety; quantized-delta
fidelity; language output preservation; long-context memory; no test data in rehearsal.

The actual executed CPU subset is reported in `validation/pytest_output.txt`. Do not mark the remaining
items passed based on similar names in the reference code.
