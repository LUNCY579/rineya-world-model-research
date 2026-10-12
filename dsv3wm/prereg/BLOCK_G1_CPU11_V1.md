# CPU11 admission for the existing constructed Block G1 qualification

Prospective resource preparation, before any Block G1 native outcome. This
extends `BLOCK_G1_EXECUTION_V1.md` and its existing input/source freezes only.
It supplies no new cases, training, scientific gate, or efficacy result.
Keep the original evaluator, native12 loader and every prior freeze unchanged.

Use the separately implemented `dsv3wm.native_cpu11` loader through the NEW
`scripts/evaluate_block_g1_cpu11.py`. Its changes are the loader import, old/new
source pins, constant resource profile and actual admission-check metadata.
The inherited loader's computation after admission is unchanged: core and
sidecar SHA checks, deterministic seed0, four CPU threads, FP32/TF32-off,
unmerged private BW_Fdelta_s0 adapter and the original greedy sampler. Require
both physical and commit availability >=11 GiB after BWM/torch imports. Retain
the original CPU lease, thermal behavior and sampler RAM wait. Record admission
values even on a refused/failed attempt. No owner process or lease is changed
by this amendment.

The inherited Block evaluator performs one generation batch and never invokes
`native.boundary`. This copy preserves that behavior. The shared CPU11 profile
still names the unchanged four-GiB boundary function, but that field does not
establish an evaluator-level or continuous four-GiB check in this instrument.
Admission and the unchanged sampler guards remain its resource checks; the
fixed single-batch envelope below is only an estimate.

`BWM_CPU_BATCH_TOKENS` must remain unset or exactly8000;
`BWM_RAM_FLOOR_GB` must remain unset or equivalent to8 GiB. The inherited8000
parameter is a heuristic grouping budget, not a strict padded-token ceiling.
Do not change grouping, caps, precision or parser policy to fit resources.
The fixed sixteen prompts previously measured at most149 tokens, with48
generated tokens allowed. Their single-batch conservative envelope is
16*(149+48)=3152 cache positions: about0.673 GiB resident FP32 KV, or1.347 GiB
under the doubled-cache estimate. This is a tensor estimate, not measured
Block G1 peak memory or a continuous available-RAM guarantee.

The completed CPU11 H3 teacher provides narrow resource evidence for the same
parent/runtime: admitted with11.766 GiB physical and19.535 GiB commit, sampled
process-tree peak4.794 GiB and successful exit. This does not establish Block
prediction behavior. Preserve failure artifacts if admission, lease or sampler
guards refuse work. No additional evaluator-level RAM/priority boundary is
introduced by this copy.

All Block inputs and evaluation semantics stay fixed: twelve primary canonical
parents, six legal changes/six illegal no-ops, four dependent name/atom-order
diagnostics, max256 input and greedy48 output tokens. Preserve the registered
initializer SHA, original prompt/target/parser bytes, typed configuration,
raw-before-filter diagnostics, support/structural refusals and malformed
abstention. Persist every generation before exact scoring; no oracle repairs
or selection enter prediction. The original gate remains >=11/12 strict
format, >=10/12 accepted tuples and >=5/6 accepted legal changes. Its sole
scope is constructed original-functionality qualification, with known prior
exposure; it is not unseen-state generalization, training qualification,
confirmation, broad retention or promotion.

Before execution, run the NEW `scripts/freeze_block_g1_cpu11.py` against the
verified existing `BLOCK_G1_FREEZE_V2.json`, choosing a fresh output. The helper
verifies every original pin, retains all original metadata/digests and the
original freeze bytes, and adds the copied evaluator, old evaluator, CPU11
loader, this declaration, helper and first-party execution closure. Use that
new freeze and a fresh result/raw path. This declaration prepares execution;
no extension freeze or native Block output is generated merely by adding it.
