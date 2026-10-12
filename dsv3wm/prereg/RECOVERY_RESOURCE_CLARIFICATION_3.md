# CPU11 teacher admission — prospective resource recovery

2026-10-12 UTC, before any H3 teacher prediction. RCX's live CPU owner exited
and released the shared lane. A new BWM GPU training job uses host memory;
available physical RAM is near11.9 GiB before torch imports, below this study's
conservative12 GiB admission check. Its GPU lease and all owner files remain
untouched. This is an operational resource amendment, not a model/data or
scientific gate change.

Use a NEW CPU-only `native_cpu11.py` loader and an explicit copied
`collect_recovery_teacher_cpu11.py` entrypoint. Preserve the old loader,
collector, failed attempt and all historical freezes. The inherited loader's
core/sidecar hashes, determinism, four threads, FP32/TF32-off, unmerged adapter
semantics and sampler are unchanged. New admission requires at least11 GiB
physical **and commit** availability after torch/BWM imports. The original
four-GiB work-boundary check and CPU lease remain active.

The observed maximum planning process-tree peak is5.21 GB (4.853 GiB), and
dynamics peaks4.508 GiB. Conservatively subtracting the whole planning peak
from11 GiB leaves6.147 GiB, approximately2.15 GiB above the boundary floor.
This supports admission; it is not a guarantee against later concurrent memory
growth. Stop/preserve failure when the existing boundary guard yields.

The fixed teacher's initial160 prompts are117–119 tokens. The existing sampler
chunks them into48/47/47/18 rows under its default8000-token parameter. The
largest padded prompt-plus32-token envelope is7248 positions: approximately
1.548 GiB resident KV or3.097 GiB under a conservative doubled-cache estimate.
Previous12-parent planning already filled similar internal chunks.
`logits_to_keep=1` and no-grad inference remain active.

The8000 parameter is a heuristic grouping budget, not a strict padded-token
ceiling. Reject a nondefault `BWM_CPU_BATCH_TOKENS`; do not change grouping to
fit memory. `BWM_RAM_FLOOR_GB` must remain unset/default8 for the inherited
loader's additional wait. Record the exact environment and actual admission
values. No speculative low precision, dense merge, owner-process termination,
GPU lock reclamation or oracle fallback is permitted.

The32 TRAIN parents, goals, configurations, immutable B0 adapter, generation,
learned search/ranking and delayed exact verification stay fixed. The same
bounded retry uses the fresh `teacher_B0_retry01` path with a NEW CPU11/source
freeze. All actual source pins include both old and new collectors/loaders.
Resource admission cannot convert a failed corpus-quality gate into a pass.
Subsequent phases retain their existing12 GiB paths unless separately registered.
