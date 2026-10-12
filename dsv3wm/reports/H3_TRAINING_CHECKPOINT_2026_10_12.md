# Native H3 training checkpoint — 2026-10-12 UTC

All four registered continuations completed32 optimizer updates: A3/A4 use
verified-plan supervision; C3/C4 replace that slot with real same-row one-step
CE. All use the shared27-positive TRAIN corpus, identical deterministic source/
plan ordering, seed0 and10,092,544 trainable unmerged LoRA parameters. A3/C3
warm from final A2; A4/C4 warm from final A0. No student planning, dynamics or
retention readout is included at this cutoff, so efficacy remains unknown.

| Actual run | Warm parent | Updates | Nominal objective forwards | Input / supervised tokens | Dynamics / plan / extra-h1 examples | Script seconds | Sampled process-tree peak, bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| A3_s0 | A2 | 32 | 128 | 21,884 / 1,938 | 96 / 32 / 0 | 503.8473842 | 3,806,597,120 |
| A4_s0_retry01 | A0 | 32 | 64 | 8,420 / 1,000 | 32 / 32 / 0 | 197.0455431 | 3,774,160,896 |
| C3_s0 | A2 | 32 | 128 | 21,968 / 1,870 | 96 / 0 / 32 | 449.7709994 | 3,797,176,320 |
| C4_s0 | A0 | 32 | 64 | 8,504 / 932 | 32 / 0 / 32 | 176.4720912 | 3,772,571,648 |

Each resource wrapper completed with exit0. Wrapper wall durations were
505.3925247,198.7681332,451.4100455 and178.2906431 seconds in table order.
Forward counts are nominal loss-pair calls, not physical decoder passes:
checkpoint recomputation/head work is excluded. Token allocation differs despite
matched nominal forwards and updates. Composite training losses and elapsed
times across these different objectives do not establish planning improvement.
GPU memory is unmeasured.

The initial A4_s0 attempt refused before core load at10.869400GiB physical /
18.301304GiB commit under CPU11. It has zero recorded forward/token/example
counts and an empty loss curve; no optimizer update occurred. Its failed run and
exit1 resource record remain preserved. A4's bounded retry and C3/C4 then used
the separately preregistered CPU10.5 **training-only** profile; objectives,
precision, row order, optimizer and tensor shapes were unchanged. A3 used CPU11.
The new training profile did not silently alter inference or unrelated domain
floors. Its13 targeted tests, source review and prospective freeze passed
before use; checkpoint13 later passed287 package tests and compilation.

Actual successful after-import physical/commit admission values, in table order,
were11.032722/18.422142,11.055122/19.387333,10.854969/18.393539 and
10.870762/18.428185GiB. See
[resource clarification5](../prereg/RECOVERY_RESOURCE_CLARIFICATION_5.md).
The intervening peer slot003 attempted and exited1; this is not a completed
E06 training claim.

Completed adapter weight SHA256 values:

- A3: `79a0d09dc9e8834a953eb777ec9be746a72cbbfc2700486e3a5c5d1c5c08b29e`.
- A4 retry01: `79cf22d21673b9ea2792be9faf6a9e93bf3f0ce781638536ca52936bf0c0b72d`.
- C3: `9d0bbb10bcfbb8c91744188441a7d81efa0a080d823046996d990778387b7145`.
- C4: `d6e5fc3c5fe92c26d7488462fe7a1bc4e4f4b8d7023a66d8855b5d1d81ba7d38`.

The shared verified corpus SHA remains
`3a2a25a33b7dc983b422b67fc1a1977d7315896e42c72132a454012c27f1841b`.
A3 binds CPU11 training freeze
`47b520b2c6ddd474c5ddce2f7de6b255dce6e83c72873fe9dae89a596dedbad7`;
the other three bind CPU10.5 training freeze
`493ef2ba76210f6dd5c71d3b17724db909bbc7aebe1a94acc60e204c154763ef`.
Actual completed weights/configurations now have output freezes:
`RECOVERY_H3_STUDENT_READOUT_FREEZE_V3_CPU11.json`
(`10a88c0bb31cab3e413e7af95c98968a6524dd5efff77a68e938f22404250546`)
and `RECOVERY_H3_STUDENT_GUARDS_FREEZE_V3_CPU11.json`
(`18323febfab44c1f78e13699fd43b7d00c4df12f2efdb332784a9a1423b5b4d3`).
These preserve their V1/V2 predecessors and actual model lineage; a training
freeze is not being substituted for output weights.

Primary records are the four `.run.json` and `.resources.json` files under
`results/recovery-h3-v1/` named in the table, plus
`A4_s0.failed_preload.run.json` and `A4_s0.resources.json`. Large weights remain
private. The earlier [warm-reference report](H3_WARM_REFERENCES_2026_10_12.md)
retains its original cutoff and measured failures.

At the current preparation cutoff, a no-model CPU11 admission probe measured
10.917744GiB physical /18.396648GiB commit and refused before core load.
No student readout followed. Singleton readout preparation is separately
declared in [resource clarification6](../prereg/RECOVERY_RESOURCE_CLARIFICATION_6.md);
it needs prospective source/output binding and exact raw-output parity against
new warm guard references before student guards. Existing broad warm results
cannot be silently replaced. Peer slot004 subsequently exited1 after a bounded
wait; helper session14808 exited/released. This is not a completed peer-training
claim. No dsv3wm native model is active; singleton source review/parity is pending.

Training completion qualifies these artifacts for the registered readout
workflow only. The operational student gain/grammar, fixed-B0 hybrid,
dynamics-preservation and behavior guards remain unevaluated. The earlier
RegisterWorld predictive gain gate remains failed; no larger-study,
confirmation, efficiency or integration benefit is established.
