# Recovery clarification 1 — before adapted training outcomes

The v1 draft abbreviated the termination field as `term` in several explanatory
passages. The exact frozen implementation and inherited registered interface use
**`terminated`**, with response keys `{x, legal, terminated}`. The shorthand was
an editorial error, not a different model target. h1 preserves the existing
prompt/target/parser; h2/h4 and every scalar label use this same key. Absorbing
termination and last-action legality remain unchanged.

This clarification was recorded while B0 qualification was running and BEFORE
its result was read; no adapted recovery training has started. Preserve the v1
protocol/freeze and original outcomes. The prospective training freeze binds this
clarification and the actual qualified B0 artifact.

The recovery optimizer is fresh AdamW, learning rate2e-5, betas(.9,.999), eps1e-8,
weight decay.01, gradient clip1, constant LR/no scheduler, batch1. This does not
reproduce historical BWM's optimizer schedule. Original adapter dropout.05 and
rank16/alpha32/all projection targets are preserved. Initial adapter/parent
weights are the same across arms; per-step/horizon dropout seeds are recorded
in source and auxiliary initialization uses a separate RNG fork.
