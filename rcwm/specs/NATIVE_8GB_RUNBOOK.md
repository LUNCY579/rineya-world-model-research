# Native Rineya / 8 GB runbook

Status: NOT EXECUTED. Reconcile the current repository and hardware before use. Historical report
[I01] is not a current admission grant. Source identity, allowed host RAM and maximum steps are unset.

## Read-only preflight

Record repo/branch/revision/dirty files, checkpoint name and SHA256, tokenizer/config hashes,
parameter count, loader, GPU model/free memory, host available RAM, disk space and power mode.
Do not substitute stock Qwen3 or an older Rineya student when the expected artifact is missing.

## Numerical preflight

At a small context, compare prefix and bulk logits/hidden captures at the same positions. Check
padding/positions, cache on/off, precision, dropout/eval mode and fused attention. Store absolute and
relative errors and declared tolerance. Test save/resume and base immutability. A failed comparison
blocks capture/training; do not merely widen tolerance after seeing an error.

## Memory ladder

Start at 512 text tokens, horizon 1, one candidate microbatch, no native backward pass. Measure
allocated/reserved GPU bytes and process-tree RAM. Cache action-boundary summaries in shards.
Unload the base before CPU/small-sidecar training when possible. Increase one dimension at a time.
For native adapters, remeasure backward/optimizer peaks; forward fitting is not training qualification.

Proposed guard: do not admit a pilot expected to exceed 6.5 GiB allocated on the 8 GB device without
a new explicit budget. This is a conservative starting guard, not a proven safe peak. Also check reserved
memory and OS headroom. A null host-RAM budget or max-step limit blocks native training.

## Runtime stop conditions

Nonfinite loss/gradients, source hash mutation, leakage assertion failure, uncontrolled paging,
resource limit, unexpected network action, changed tokenizer/capture version, invalid verifier or
sealed-data access. Save a diagnostic report rather than repeatedly retrying an OOM run.

No paid compute, weight downloads, background schedules, repository writes or remote pushes are
implicitly authorized by this runbook. A later scoped instruction may authorize specific steps.
