# Native H3 warm-parent references — 2026-10-12 UTC

Both unchanged final32-step warm parents preserve24/24 own-state four-step
dynamics but produce no syntactically valid complete plans in the registered
H3 proposer interface. A0 and A2 each score0/12 greedy top1,0/12 offline pass@4
and0/12 learned-reranked success. These are completed native CPU references for
future student comparisons; they establish no student or distillation effect.

Cutoff: the four warm readouts completed by2026-10-12 09:53:51.760277UTC,
according to the final resource record. Sequential A3/A4/C3/C4 continuation
training has since started in session85910, with32 updates per recipe. This
report includes no student training or readout outcome. The prospective fixed
warm guards, prompts, cases, thresholds and inference budgets remain unchanged.

## Dynamics and narrow behavior guards

| Actual warm parent | Own-state rollout h1/h2/h4 | Direct h1 | Direct h2 | Direct h4 | Direct h2/h4 malformed | Six behavior cases |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 24/24 at each | 24/24 | 2/24 | 0/24 | 5/24;10/24 | 3/6 |
| A2 | 24/24 at each | 24/24 | 3/24 | 0/24 | 0/24;0/24 | 2/6 |

Both have24/24 rollout legality and termination agreement at h1/h2/h4,
zero malformed rollouts and no correct-final/wrong-or-unknown-intermediate
case. Perfect rollout states coexist with failed direct composition. A2's
complete direct grammar does not establish direct state learning: its direct
h4 legality agreement is10/24 and direct state accuracy0/24. A0's corresponding
legality agreement is9/24. These records do not measure probability calibration.

The behavior references are the same three RegisterWorld and three Ledger
fixtures. A0 scores3/3 Register and0/3 Ledger; A2 scores2/3 Register and0/3
Ledger. A2's failed boundary reply remains
`{"legal":true,"terminated":false,"x":32}` for the boundary fixture. Thus its
warm baseline already lacks that retained legality behavior. A later absence of
additional losses versus A2 cannot establish restoration of the earlier skill.
Cases that both warm parents fail cannot certify retained Ledger capability.
Both outputs explicitly leave `retention_pass` null.

## Complete-plan proposer references

Both readouts use the actual A0/A2 weights, with the A4/A3 StudentConfig only
specifying the registered inference interface. Their recorded artifact status
is `pilot-trained`; neither warm parent is an H3-trained student. Each uses12
fixed DEV tasks, greedy candidate0 plus three fixed CRN samples, K=4, at most80
output tokens and six actions including final `finish`.

| Actual warm parent | Greedy top1 | Offline pass@4 | Fixed-B0 learned reranking | Strict plan syntax | Proposal queries | Forward queries |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 0/12 | 0/12 | 0/12 | 0/48 | 48 | 0 |
| A2 | 0/12 | 0/12 | 0/12 | 0/48 | 48 | 0 |

All48 replies per parent reached EOS: A0 used14–15 tokens per reply and A2
14–18, below the80-token cap. Every A0 reply is a JSON object with `x`, `legal`
and `terminated`, instead of the registered `actions` object. A2 emits that
same key set47 times and adds `budget` once. This is an observed output-contract
failure, without a causal claim about the model's internal planning ability.
No valid plan diversity or first-action optimality is observed.

The optional B0 reranking phase was enabled, with the original adapter restored,
but every proposal was malformed. Consequently it made zero learned forward
queries and selected no plan. The nominal hybrid result does not measure B0
selection among valid candidates. No verifier fallback, syntax repair or oracle
deployment selection occurred. All generation/ranking was fixed before scoring.

## Measured costs and their scopes

| Readout | Script duration, seconds | Wrapper wall duration, seconds | Logical model queries | Input/output tokens | Generation positions | Generation decoder forwards | Sampled process-tree peak, bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 guards | 200.0146674 | 203.5414583 | 174 | 24,374 / 2,530 | 26,910 measured | 124 measured | 4,816,850,944 |
| A2 guards | 181.9014980 | 185.8209951 | 174 | 24,374 / 2,602 | 26,910 measured | 124 measured | 4,817,731,584 |
| A0 proposer | 161.7778898 | 164.9661543 | 48 | 5,468 / 716 | 6,136 formula estimate | 716 formula estimate | 3,709,796,352 |
| A2 proposer | 162.5966019 | 165.6449668 | 48 | 5,468 / 720 | 6,140 formula estimate | 720 formula estimate | 3,709,386,752 |

Each guard's physical forward-pre-hook ledger separates dynamics105 forwards /
26,088 padded positions from behavior19 forwards /822 padded positions. Its
174 logical queries comprise168 dynamics and six behavior requests. Actual
generation durations are187.9748101s and169.9460477s; script durations also
include loading and scoring. The guard counters are generation-only and are
not measured FLOPs.

The proposer counters use the registered single-query cached-decoder formula;
their physical forwards/positions and FLOPs remain unmeasured. Its cost reducer
charges all four proposals on every failed task and reports queries-per-solved
as null because zero tasks were solved. There is no planning-efficiency claim
from these failures or from the zero reranking queries. Host-RAM peaks are
sampled process-tree RSS; GPU memory is unmeasured.

CPU11 admission, after imports, observed physical/commit availability:
A0 guards11.160515/19.002197GiB, A2 guards11.229160/19.094471GiB,
A0 proposer11.212086/19.076103GiB and A2 proposer11.209160/19.060791GiB.
All four passed the registered11GiB physical and commit floor.

After-launch health observations were recorded for A0 guards17 samples,
A2 guards8 samples and A2 proposer13 samples. Their minimum observed available
RAM was5.337482,7.057846 and8.512234GiB respectively; maximum observed thermal-zone
temperature was96.9,96.9 and95.9C. These are intermittent observations and did
not change process state or owner health files. The A0 proposer health observer
missed the child after exit; its health/temperature observation is unmeasured.
The separate resource-wrapper RAM record remains available for that readout.

## Binding and next interpretation

Guard source freeze SHA256:
`13b2fcbbe8890a335197eacf02e39746cfd8b3b832474a9502d7579551ec1b2b`.
Proposer source freeze SHA256:
`4170f6ad5ffbabcf526109fe3c9a7e69991ad8cf268363afb166576fe0142f2f`.
The same actual adapter is bound across each parent's two readouts:

- A0 weights: `9b6ff87157ba7418fd328e1325c1461a7bbf305e01d13381d6225c4f8332cfe9`.
- A2 weights: `905d52126ed6f4774f935642ccda2c3edd0d00c38eb2a9bad03628698f7dcd97`.

Result SHA256 values, in A0 guards/A2 guards/A0 proposer/A2 proposer order:
`b04674adf5c34cd969d22494d4fa28db856dfb5dc47dfddcae52324b83a2fbb1`,
`f30b25bbf25ee9e0d3073c284926120e451eb5a2fa85324467e43855d5cb0712`,
`65a1d72b4aaeb80be003e721d9b2d07f206faf542473de34dae6c118edd685e6`,
`c849070cab244931be35d24d40df7f651258d5ae3f4808d72ce95bdc5d6eeb31`.

Primary sources are the four `warm_A0/A2_guards/proposer.json` files under
`results/recovery-h3-v1/`, their two proposer `.costs.json` files, four
`.resources.json` files, two guard `.traces.json` files and the three available
`.health.jsonl` files. All resource wrappers completed with exit0.

Future students must be compared with their own frozen warm parent, including
these existing failures and behavior denominators, and with the registered
C3/C4 continuation controls. Any later plan-format improvement must be separated
from complete-plan success, dynamics preservation and retained behavior.
Twelve planning cases/one training seed remain development feasibility evidence.
No larger-study, confirmation or integration qualification follows from this
warm-reference checkpoint; the earlier predictive gain gate remains failed.
