# Pilot amendment 1 — before development generation outcomes

2026-10-11 America/Chicago. Original pretraining freeze/commit remains intact.
During A0 training, independent code review identified that response parsing
accepted numeric booleans/floats and missing termination/failure flags. Python
equality could then falsely mark a numeric state correct. No development
generation result was accessed before this correction.

`native.parse_prediction` now enforces exact output keys, integer state values,
boolean flags, unique JSON keys and finite constants, and pure structural state
validation. Added regression tests. No transition oracle is used in parsing.
All arms use this same corrected parser and rollout instrument. Training loss,
initialization, data, budgets and objectives are unchanged.

Code review also found an existing shared CPU-lane protocol in the newer BWM
runner. The initial 33-second native probe and 14-second A0 training did not
acquire this CPU lock; process/lock inspection found no competing holder. This
coordination omission is recorded rather than concealed. Subsequent heavy native
CPU work acquires the same exclusive token-checked lane, identifies as dsv3wm,
honors rcx/bwm waiters at boundaries, caps threads at four, and requires 12 GiB
free RAM before load. No other programme's lock is deleted.

The pilot bank contains repeated parents within splits (24 rows each, 18/18/19
train/dev/confirmation parents). Cross-split parent disjointness holds. Operational
metrics report evaluated unique parents and never treat all 24 rows as independent.
The sealed bank remains unopened; a future G5 bank must select exactly one primary
episode per parent or use a predeclared clustered instrument.

The predictor also records exact decoder token positions for cache-disabled
generation: n*g + g*(g-1)/2 for n prompt tokens and g decoded tokens. An optional
position budget limits decoding before a model call. Transition-model query
counts are distinct from autoregressive decoder calls (g) and computation.
The pilot report shows all of these; no equal-compute superiority is inferred.

Before A3/A4, re-verification additionally binds canonical parent identities to
the hashes of the supplied approved training rows. Merely relabelling a dev
record as train cannot satisfy this allowlist. The upcoming freeze binds native
loader/CPU-lane/GPU-lease source, and native loads recheck core configuration
and tokenizer hashes. A0/A1 preceded automatic source-freeze enforcement but
their recorded configs/data/core and the initial freeze are preserved; A2/A5
and all development generation use the corrected source freeze.

Git text normalization is added to avoid treating Windows CRLF as trailing
whitespace. Frozen logical records remain unchanged; original hashes remain
the original bytes. New source/analysis freeze is recorded before evaluation.
