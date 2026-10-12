# Prospective RCWM R/P retention v2 — 2026-10-12 (America/Chicago)

Status: implementation and fixture construction only; no native v2 outcome.
The v1 protocol, source, fixture bank and B0 freeze remain unchanged. No model
outcome motivated this correction: the original v1 B0 freeze is unexecuted.

A read-only audit of SHA-bound `t2_train.jsonl` input fields found that all 500
original modular-register inputs contain `tick`. The twelve v1 register events
omit that field. Their SYSTEM_ONE instructions, chat framing, bare targets and
exact arithmetic are valid, but their event surface differs from original T2.
The prerequisite events already retain all seven original event fields.

The authenticated input-only frame audit finds all 500 original R TRAIN rows
use SYSTEM_ONE with `{before, event}`, whereas all 500 P TRAIN rows use
SYSTEM_MULTI with `{before, events:[one event]}` (recorded frame `m1`). V2 R
therefore restores the actual T2 R event/message surface. P deliberately retains
the paired original Titan SYSTEM_ONE frame with exact seven-field event parity;
it does **not** establish parity with T2's trained `m1` prompt. B0 measures whether
P behavior is available under this original one-step frame. Cases failed by B0
cannot establish subsequent loss of that capability or retention of T2's trained
P interface. Do not create a third fixture revision after observing outcomes.

V2 retains the same twelve register and twelve prerequisite cases. It adds an
integer register `tick` by deterministic cycling through sorted original TRAIN
tick support. It leaves initial states, operative actions, goals, targets,
semantic input IDs and initial-parent IDs unchanged. Tick is an inert input
field under the audited register transition rule. Both the original and added-
tick inputs must be excluded from known semantic TRAIN keys; inserting a neutral
field must never disguise an exposed operative transition.

Record a one-to-one v1-to-v2 mapping with exact input/prompt hashes, tick values,
unchanged target hashes, semantic IDs, parents and shared rule-group IDs. Recheck
the original TRAIN SHA before input-only audit; no original targets or historical
evaluation/sealed rows enter construction. Snapshot the source bytes and hash
every snapshot, fixture and mapping before any native v2 measurement.

Inference remains FP32, unmerged, CPU, greedy, at most 512 input tokens and 64
generated tokens. Tokenize-only construction must confirm all prompt and target
lengths fit; there is no truncation or outcome-driven token-cap adjustment. Use
the original SYSTEM_ONE/message/chat formats and bare register-object or
prerequisite-integer targets. Finish all 24 generations before strict offline
scoring. Preserve raw replies; no output extraction or repair is allowed.

Cases are paired across T2 B0 and Ledger-adapted models. Twelve distinct initial
states in each family do not create twelve independent transition graphs: v2
preserves one modular-97 and one eight-bit shared graph group. Report families,
cases, initial parents and shared groups separately. No broad retention pass or
independent efficacy confirmation follows from this supplement. The separately
required registered six-fixture audit remains required.

The v2 execution freeze includes first-party import closure documented in
`EXECUTION_SOURCE_CLOSURE_V1.md`, including package initializers, `flops.py`,
`blocks.py` and `bwmn/planbench_extract.py`. Source or artifact drift stops
execution. B0 v2 must use unchanged registered T2 weight/configuration pins and
the same frozen parent core. Adapted freezes additionally require completed
Ledger training, full typed configuration/adapter behavior and qualification
lineage. New v2 filenames/freezes are required; never edit or relabel v1 evidence.
