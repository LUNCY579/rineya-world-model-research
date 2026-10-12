# H3 singleton readout: partial native qualification

Evidence cutoff: 2026-10-12 11:45 UTC. This checkpoint records completed warm
A0 output parity and the first warm A2 admission refusal. No H3 student efficacy
result, confirmation result, compression benefit or integration qualification
follows from it. All earlier broad readouts and input freezes remain unchanged.

## Completed warm A0

The new, prospectively frozen CPU10.5 singleton path used the actual warm A0
artifact, FP32, TF32 disabled, unmerged rank16 LoRA, CPU4 and seed0. It completed
the unchanged24-parent DEV contract and six behavior fixtures. Generation
finished before scoring. Its admission measured10.619370GiB physical and
17.606789GiB commit availability after BWM/torch imports, before core load.

The entire old/new trace files are byte-identical, including raw replies,
conditioning, token counts, EOS information, source tags and ordering. SHA256:
`427ae31e00aa86a18786bb7851179f0fb5c9961b70246bd74371396d3937bcb5`.
The registered `raw_guard_parity` check passed168 dynamics and six behavior
rows. An independent audit also matched full outcomes, configurations, actual
artifact/input identities and90 current source/freeze pins. No metric-based
selection or simulator repair was used.

| Measurement | Preserved batch A0 | Singleton A0 |
|---|---:|---:|
| Logical generation requests |174|174|
| Input/output tokens |24,374 /2,530|24,374 /2,530|
| Physical decoder forwards |124|2,530|
| Padded decoder positions |26,910|26,730|
| Script duration, seconds |200.014667|550.536555|
| Sampled process-tree peak, bytes |4,816,850,944|3,710,717,952|
| Sampled process-tree peak, GiB |4.486042|3.455875|

The singleton run preserved rollout h1/h2/h4 at24/24 each, direct h1/h2/h4
at24/2/0 of24 and behavior at3/6. It uses substantially more physical forward
calls and takes2.75 times the elapsed script time. The lower working set is a
batch-geometry tradeoff. Neither cache compression nor useful H5 efficiency
has been demonstrated. Dedicated GPU memory and physical FLOPs are unmeasured.
Elapsed times are descriptive runs under a shared host, not controlled latency
replications.

The resource wrapper exited0 after554.209966s. Its after-launch diagnostic
observer completed49 samples, with minimum available RAM6.833179GiB and maximum
reported thermal-zone temperature94.9C. These samples cover part of the run;
they cannot establish unsampled extrema. The observer changed no process state.

Actual records are
`results/recovery-h3-v1/warm_A0_guards_cpu105_singleton.json`, its `.traces.json`,
`.resources.json`, `.health.jsonl` and `.parity.json`. The result SHA256 is
`f60dce3fba0165a379996e3294ce1819482ff8fbcaf3e93004ba9d392ce0ce6d`.
The paired older result SHA256 is
`b04674adf5c34cd969d22494d4fa28db856dfb5dc47dfddcae52324b83a2fbb1`.

## Preserved warm A2 refusal

`warm_A2_guards_cpu105_singleton.json` failed before core hashing/loading.
Physical availability10.067127GiB was below the unchanged10.5GiB floor;
commit availability15.808804GiB qualified. The result has empty reserved trace
arrays and no outcomes or cost record. Source control flow and the admission
record establish that no learned generation occurred. Missing result cost fields
remain absent rather than being filled with purported measured zeros.
Its resource wrapper exited1 and sampled a394,760,192-byte process-tree peak.
The failed attempt remains immutable. This is a resource refusal, not an A2
predictive or parity failure.

The A0/A2 sequence stopped at that refusal. No student guard extension or native
student readout was executed. The original broad A2 result is still valid within
its registered scope; it does not substitute for required singleton raw parity.

## Remaining execution gate

The next permissible native step is one same-input warm A2 retry in a fresh
`warm_A2_guards_cpu105_singleton_retry01.json` destination, after actual
post-import physical and commit availability both reach10.5GiB and the shared
CPU lane admits it. Keep the same frozen sources, configuration, weights and
data. Do not lower the floor or overwrite the failed path.

An exact A2 raw-parity pass is required before creating the actual student guard
CPU10.5 extension. That extension must bind both warm references to each
selected student's frozen provenance. The student proposer extension is already
prospectively registered, but any native proposer also needs resource admission.
The four student/control effects remain unknown.

At11:45UTC a read-only host inventory showed9.81GiB available, external BWM GPU
worker153532, external RCWM orchestrator164060 and external pytest173032 live.
No own native model was active. Owner assets, processes and GPU lease were not
modified. The inventory is a dated observation, not a predicted release time.

Full research completion additionally requires qualified domain experiments,
causal ablations/stress evidence, retention/calibration, successful-pilot seed
replication, fresh sealed paired co-primary confirmation, and justified
conditional follow-ups. This partial memory/parity checkpoint closes none of
those scientific gates.
