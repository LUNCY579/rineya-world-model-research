# Delivered reference mechanics and validation

## What was actually run

The bundle's original CPU package was tested in the recorded environment: Python 3.13.5, PyTorch 2.10.0+cpu and NumPy 2.3.5. **60 tests passed.** They cover finite-horizon consequence accounting, goal revaluation, mask conventions, future-action invariance, parent grouping, train-only codebook fitting, schema/temporal checks, simulator behavior, tensor shapes, finite gradients and checkpoint roundtrip. Full test names and logs are included. No GPU or native Rineya checkpoint was used.

A 40,325-parameter causal action-token Transformer was trained for 250 updates on an original KeyDoor toy dataset. The check-set mean-squared feature error changed from 0.540924 to 0.002186. This is a mechanics demonstration with repeated support in a tiny state space, not an independent generalization benchmark. Its learned bottleneck is supervised through measured features; it does not validate self-supervised semantic discovery or clustering.

## An important observed limitation

The average error alone would be misleading. On the illustrated safe sequence `take_key -> open_door -> collect`, the model predicted a cumulative object-gained feature of approximately **0.064**, although the exact simulator value was **1.000**. Under the damage-averse goal, it scored this route at approximately **-3.46**, although the exact value was **7.00**. The demo therefore does **not** pass a strong per-plan fidelity gate merely because its aggregate error fell.

This failure is deliberately preserved in the delivered evidence. It illustrates why the proposed program requires prerequisite-stratified checks, candidate regret and independent real outcomes, rather than only an average reconstruction score. The code's mechanical tests passing means the selected tested invariants held; it does not mean its learned predictions are accurate enough for deployment. A later coverage or training change should be a separately logged experiment, not a silent rewrite of this run.

## Resource and artifact checks

Three JSON schemas, four illustrative trajectory records, seven proposed configurations and the 43-entry source register were checked by the bundled validation script. Temporal and feature-dimension checks extend the JSON Schema validation for the initial serial format. Production provenance, native memory, stochastic calibration, parallel execution and retention remain unmeasured.

The final archive includes a file manifest with SHA-256 hashes. The manifest excludes itself to avoid a recursive hash definition. The verification utility checks each listed file after extraction. These integrity checks are reproducibility aids, not a security certification or validation of scientific claims.

## Issues found during construction

An initial unit-test run exposed a Python/PyTorch naming collision: a submodule named `type` conflicted with the parent module's dtype-conversion method. Renaming it to `type_embedding` fixed the forward path, after which the suite was rerun. This is recorded as a resolved engineering issue, not hidden behind the final passing result.

The report and machine-readable status files distinguish implemented reference mechanics from planned native work. The most important next implementation task is still the matched scalar/feature baseline and qualified Rineya capture, not an expansion into imagined RL.
