# CPU10.5 remaining H3 training — prospective operational extension

Before any retry of the refused A4 run or any C3/C4 update under this profile,
preserve every original loader, CLI, configuration, corpus, freeze, completed
A3 artifact and failed A4 attempt. A3 completed its fixed 32 CPU11 updates after
physical admission at 11.032722 GiB. Its sampled process-tree peak was
3,806,597,120 bytes (3.545 GiB), with an observed training health reading near
7.73 GiB free RAM and 94.9 C. A4_s0 refused before core load at 10.869400 GiB
physical / 18.301304 GiB commit and performed no updates; C3/C4 are unstarted.
These are resource observations, not H3 performance or retention results.

The current approximately 11.18 GiB before-import availability can fall below
the previous 11 GiB after-import floor. For the remaining fixed batch-one,
no-cache training only, add `native_cpu105_training.py` and the copied
`train_recovery_student_cpu105.py`. Require both physical and commit availability
at least 10.5 GiB after BWM/torch imports and before core hashing/loading. Using
the measured A3 peak and a conservative prior warm-guard background deficit of
1.337 GiB gives 10.5 - 3.545 - 1.337 = 5.618 GiB, about 1.618 GiB above the
unchanged four-GiB work boundary. This empirical margin supports a bounded
attempt; it does not guarantee future concurrent usage or an unseen peak.

The new profile accepts only the complete SHA-bound existing A4/C3/C4
StudentConfig recipes: seed 0, four threads, batch one, 32 updates, identical
draws/corpus, precision, cache setting, optimizer, CE chunk and tensor shapes.
Ordinary A3 repeats, altered recipes, inference-style adapter loading and calls
without an explicit training-only declaration refuse before BWM/torch work.
The warm unmerged adapter is loaded trainably by the unchanged trainer after
the core load. Core/sidecar hashes, FP32/TF32-off, deterministic settings, MTP/
multihorizon losses, slot weighting, dropout seeds, fresh AdamW and checkpoints
are unchanged. Exact-copy/AST tests protect those algorithm boundaries.

Retain the original exclusive CPU lane, priority-waiter check, thermal guard and
four-GiB memory boundary. Never change a peer's lease, terminate an owner,
lower precision, merge adapters, alter shapes or modify sampler geometry to fit.
Sampler environment still requires `BWM_CPU_BATCH_TOKENS` unset or8000 and
`BWM_RAM_FLOOR_GB` unset or equivalent to8 GiB. Record the profile and actual
physical/commit admission, including refusal. Preserve failed/partial artifacts
when existing priority, thermal or memory conditions refuse further work.

The new `freeze_recovery_student_cpu105_training.py --freeze OLD --out NEW`
verifies the original CPU11 H3 training freeze and preserves all metadata/member
digests and its actual bytes. It accepts only the warm verified H3 student/control
training instrument, adds the new loader/trainer/helper and this declaration,
and retains all original source/corpus/warm/qualification pins. Existing outputs
and an already extended freeze cannot be overwritten. The parent must review
and create that new prospective freeze before native execution.

Only a single bounded A4 retry at the fresh `A4_s0_retry01` path is proposed
after registration and CPU-lane admission. C3/C4 remain their prescribed fixed
training runs; this profile does not authorize additional training, seed changes,
checkpoints chosen by correctness or corpus expansion. An existing completed
A4/C3/C4 run must not be repeated merely because this loader admits it.

All inference, teacher search, warm-parent/proposer/guard readouts, Ledger and
Blocksworld retain their existing CPU11 or historical 12 GiB admission paths.
No confirmation access, scientific gate relaxation or claim follows from this
operational amendment. Implementation tests create no actual phase freeze or
native outcome; completed training later requires its own output/readout freeze.
