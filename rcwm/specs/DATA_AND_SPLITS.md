# Data and split implementation contract

1. Assign each source problem a stable parent-family key BEFORE generating variants.
2. Generate branches, goal relabels, paraphrases and alternate observations under that key.
3. All descendants inherit the parent's split. Different snapshots from the same source scenario may
   require the same broader family key when a narrow parent key would leak the template.
4. Fit normalization, clustering and representation pretraining on TRAIN only. Fit new-goal readouts
   on a declared calibration subset, never on sealed evaluation parents.
5. Store the policy and information available when actions were selected. Only precommitted plans
   are valid whole-plan interventions without further policy-tree modeling.
6. Check actual timestamps, status, feature validity and termination/truncation semantics.

`schemas/trajectory.schema.json` is a strict initial serial/open-loop envelope. It does not by itself
validate every causal or temporal invariant. `tools/validate_resources.py` adds cross-field checks for
bundled examples. Production data need snapshot replay, source/variant deduplication, executor
allowlisting and observation leakage audits too.

Hidden captures must include checkpoint, tokenizer, template, precision, mask and capture IDs.
Changing any one invalidates dependent caches unless an explicitly evaluated conversion is provided.
Do not store large tensors inline in JSONL; use immutable chunk references with offsets and hashes.

Label budget accounting separates environment interactions, unlabeled transitions, verifier calls,
human comparisons, teacher tokens and accepted training tokens. Retain invalid actions and failed
calls with status labels. Balanced training sampling does not replace a natural-prevalence calibration
set. Avoid using a learned predictor's own scores as ground truth.
