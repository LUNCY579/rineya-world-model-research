# Mask and causality contract

| Query | Known observed prefix | Proposed actions through t | Later proposed actions | Actual future outcomes |
|---|---:|---:|---:|---:|
| causal consequence at t | yes | yes | no | no |
| whole-plan value | yes | yes | yes, precommitted only | no |
| retrospective target encoder | yes | yes | yes | yes, targets only |
| live DAG decision at clock c | only available by c | legal metadata | according to planning mode | never before availability |

A future recorded action chosen after an unseen observation can itself leak that observation.
Use intervention-generated open-loop plans, causal prefixes, or an explicitly modeled conditional
policy tree. Rename a mode rather than relabeling a leak as “world understanding.”

## Required adversarial tests

Change all future labels without changing prospective input bytes or outputs. Change a future
candidate action and ensure earlier causal predictions remain invariant. Move an observation's
availability beyond decision time and require rejection. Change sibling completion order in a DAG
and ensure no result arrives early. Keep parent/counterfactual variants in a single split. Test padding,
batch independence and incremental-versus-bulk capture numerics.

`nn.Transformer` Boolean True blocks attention. Other APIs may use True to allow attention.
Use explicit conversion and behavior tests before switching kernels [S41]. A mask-shape assertion
alone is insufficient. For a row with no valid keys, either supply a known context token or reject the
batch; do not allow silent NaNs from fully masked attention.

The toy causal model has only observed initial state plus candidate action tokens. Its forward
signature cannot receive future labels. The provided availability/DAG utilities are not an asynchronous
scheduler; full DAG execution is planned G4 work.
