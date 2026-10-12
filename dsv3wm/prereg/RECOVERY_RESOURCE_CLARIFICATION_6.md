# CPU10.5 singleton H3 readout — prospective resource extension

Before any H3 student proposer/guard output under this profile, preserve all old
loaders, evaluators, freezes, trained adapters and old warm guard outputs. All four
fixed students/controls are trained, and their V3 CPU11 output freezes are created.
A fresh after-torch check was 10.917744 GiB physical / 18.396648 GiB commit, below
the unchanged CPU11 floor; no model loaded in that check. This amendment changes
resource admission/guard batching only, not a metric, configuration or artifact
selected by correctness, and makes no compression/H5 claim.

Use NEW `native_cpu105_readout.py`, `evaluate_recovery_student_cpu105_readout.py`
and `evaluate_recovery_student_guards_cpu105_readout.py`. The loader requires exact
SHA-bound A3/A4/C3/C4 StudentConfig recipes, four threads, seed0, CPU, core-only
load and explicit `readout_only=True`. Both physical and commit availability
must be at least10.5 GiB after BWM/torch imports and before core hashing/loading.
Training, teacher search, Ledger and Blocksworld are outside this profile.

The prior singleton warm proposer peak is3.455 GiB. A conservative592/593 cache
position envelope for512 input plus80 output tokens adds approximately0.253 GiB
under a doubled-KV estimate. With the prior1.337 GiB background deficit, the
margin calculation is10.5 -3.455 -.253 -1.337 =5.455 GiB, above the unchanged4 GiB
work boundary. This is empirical admission support, not a guarantee against
future concurrency or an unseen peak. The original CPU lease/priority/thermal
guards, FP32/TF32-off, sidecar/core hashes, unmerged adapters and sampler settings
remain active. Sampler environment stays unset/default8000 and RAM floor8 GiB.

Proposer generation and fixed-B0 forward reranking already make one sampler
request per call; their numerical/generation/scoring loops are unchanged. Guard
dynamics use a new LegacyPredictor subclass that invokes the unchanged parent
`predict_batch` once per input, in source order. Behavior generation invokes the
same sampler once per fixture. All24 DEV parents, six behavior fixtures, four
rollout steps, direct h1/h2/h4, raw tags, greedy32 limits and scoring boundaries
stay fixed: the nominal complete schedule remains174 logical queries. Existing
malformed/absorbing rollout stops still apply. Physical hook counters remain
measured; their values and padding costs can differ from broad batching. No
oracle, label, behavior target or guard callback enters generation or changes
selection. Proposer K4/80, seeds, budgets and fixed-B0 adapter switching stay fixed.

Before student guards, run new warm A0/A2 singleton references under their own
prospective output freeze. Require exact raw-output parity against preserved old
warm guards: same artifact/config/data/fixtures, conditioning, tags/order and
raw replies including token/EOS fields. Compare no correctness metric to choose
a reference. A mismatch preserves both outputs and blocks student guards pending
a separately registered investigation. Never replace or relabel old guard results.

The NEW `freeze_recovery_student_cpu105_readout.py --freeze OLD --out NEW` accepts
only actual-output H3 proposer or guard CPU11 instruments. It verifies and retains
every old pin/metadata field plus original freeze bytes, and appends the new
loader/evaluators/helper, this declaration and dependency closure. Warm guard
freezes need no comparator. Student guard extensions additionally require paired
`--old-warm-guard OLD --new-warm-guard NEW` arguments for each actual warm arm;
the helper validates the raw parity and binds both results/traces/execution
freezes and the newly qualified references. The new guard refuses old broad-batch
references. New exclusive destinations are mandatory; no actual freeze or native
result is created during implementation/testing.

All earlier training admission profiles and unrelated inference floors remain
unchanged. This operational extension does not relax any H3 planning/dynamics/
behavior gate or authorize confirmation access, retries chosen by correctness,
new data, additional training or changed scoring. Parent review and prospective
freeze registration precede any native readout.
