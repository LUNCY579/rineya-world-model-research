# Recovery H3: B0 teacher execution registration

This registration freezes teacher collection only. The student portion of
`RECOVERY_H3_DRAFT_V1.md` remains a draft until the shared corpus qualifies and
actual warm parent artifacts are prospectively bound. No teacher outcomes have
been inspected for this registration.

Use the immutable qualified registered F adapter (B0), the exact existing
one-step prompt/parser, FP32 with TF32 disabled, eval mode, greedy generation,
and the shared four-thread CPU lane. Limits: beam width 1, depth 6, 30 learned
queries and 120,000 conservative processed positions per training task; prompt
length 512 and generated length 32. Record broad-batch padding costs and qualify
decoder-forward counts as lower bounds because the reused decoder splits batches.

The input pool is the 32-parent prospective TRAIN allowlist with task-file hash
`86f9daab2ed2df48fce3c04fcc0f541f2c4f73cdd32ccedbb25ee10b15a359bc`.
It retains the original 16 tasks and adds the next 16 eligible parents in the
frozen 128-row TRAIN order, using seed 202610123 for fresh desired goals at exact
distance 2..4. Collection used distances but emitted no solver plans. All task
inputs, parent identities, source TRAIN hashes and provenance are frozen before
native teacher outcomes. No additional tasks or goals may be selected afterward.

Complete learned searches and fixed candidate rankings for all 32 tasks before
exact verification starts. Write rankings and raw traces to disk at that boundary.
Reuse computed learned nodes, with learned goal distance/length/lexicographic
ranking. A candidate whose stored model prediction reaches the desired goal may
receive deterministic final `finish` completion before verification. Keep its
original source plan and completion provenance. Reject interior finish or more
than six completed actions. A learned nongoal candidate receives no completion.

Independent exact verification then checks legality, execution, final termination
and desired final state. Keep at most the first eligible verified positive in
the fixed learned order per parent. Verified nongoal cases and extra positives
are diagnostics; hard negatives never become positive CE targets. Use byte-exact
`rw2.proposer_prompt` and `rw2.target_actions` with `{"actions":[...]}` targets.
Verification curates TRAIN labels; it supplies no development/test selection.

The corpus gate requires at least 16 distinct accepted TRAIN parents, at least
two limits and four distinct action sequences. Failure preserves the whole
pool, rejected candidates, costs and provenance, forbids this student launch,
and does not justify an outcome-driven pool expansion. Any alternative teacher
or larger budget needs a separately dated prospective study.

This is a native feasibility/development experiment. It is not independent
confirmation, a learned planning success rate at test time, an H2 promotion,
or authorization to modify the dense core or other programmes' artifacts.
