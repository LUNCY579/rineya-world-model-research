# Recovery host-memory instrumentation

Recorded before adapted recovery training or planning/behavioral readouts.
The existing training/evaluation mathematics, data, generation and budgets are
unchanged. `scripts/run_measured.py` optionally wraps one local command in the
current authorized session, preserving child standard output and exit status.

It records wall time, maximum sampled subprocess-tree RSS every 0.5 seconds,
and each observed Windows process's kernel peak working set. The sampled tree
maximum may miss short peaks; separate process peaks are not simultaneous and
must not be summed as a concurrent maximum. Native trainer peak RSS remains a
separate measurement. Parent wrapper memory is excluded. GPU memory is unknown
for CPU execution; no CUDA discovery is invoked. B0's already completed dynamics
qualification has no such record and its memory cost remains unmeasured.

These are resource diagnostics, not training FLOPs or new efficacy outcomes.
Preserve failure/partial records and use new output paths rather than retrying
an existing native artifact. The shared CPU lease and model-load RAM floors
remain mandatory within each native child.
