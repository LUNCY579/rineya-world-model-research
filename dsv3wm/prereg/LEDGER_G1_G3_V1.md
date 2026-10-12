# Tick-compatible Ledger G1/G3 registration

Freeze before Ledger native outcomes. This is a separate development pilot,
using unchanged registered T2 seed0, not a redraw of the RegisterWorld cohort.
Parent: verified Titan-Sparse FP32, TF32 disabled. Unmerged T2 initializer SHA:
`92bc7c7497c3cb2fc0c7c917b4f3b6bb931f0ed108144d969b26b18b42670b8a`.
Rank16/alpha32/dropout.05, all seven original projection types. No owner files
or dense core are modified. Record full weight/config/source/data lineage.

The frozen collection is `datasets/ledger-recovery-v1`:128 TRAIN,24 DEV,
4 separate invalid-input diagnostics, no confirmation. Known prior501 T2
training parents/operative keys and24 old public Ledger parents were excluded.
Amounts and bounded balances use proven TRAIN marginals. Every primary
four-step final state changes; each trajectory has two posted changes and
two nonposted no-ops. Native parent exposure outside this support is unknown.

DEV cohorts:12 canonical,4 held-out status words,4 reversed-key presentation,
4 shifted inert ticks. Different cohorts have different parents; cross-cohort
effects are descriptive. Within-arm comparisons are paired on these fixed rows.
No data/step/budget resampling after outcomes. Input states are A-D balances;
events have exactly account/amount/kind/other/status/tick. Tick is inert.
The frozen generic five-field interface is preserved; the new dedicated
adapter calls the authoritative `env.step` for label construction/replay only.

Canonical h1 prompt is byte-identical to registered nonthinking chat.
Direct h2/h4 uses the existing multi-step prompt on initial balances and ordered
events, without intermediate states or goals. Replies are bare four-integer
balance objects, strict complete JSON and duplicate/type checks. Rollouts use
only preceding predicted balances and abstain after malformed replies. Structural
legal/termination flags are contract-derived, never learned-legality metrics.

B0 reads the entire fixed24 DEV bank: own-state h1/h2/h4, direct h1, raw trace
IDs and separately identified teacher-forced losses. Its qualification gate
uses the12 canonical h1 cases only:>=10 exact states and>=11 syntactically valid.
Gate failure stops this training recipe and preserves stress outcomes. Never
lower the gate after failure. Freeze its completed result/traces afterward,
with a training-freeze reference to the original qualification freeze.

If qualified, A0/A1/A2 have32 updates, A5 has96. All use seed0, batch1, LR2e-5,
fresh AdamW(.9,.999), eps1e-8, weight decay.01, clip1, constant LR; no scheduler.
Shared source-row/dropout seeds and warm initializer. A1 MTP total2 tokens,
rank16, weight.1; causal auxiliary heads omitted at inference. A2 state horizons
1/2/4 weighted1/.5/.5. All full-state labels; beta0 means no extra legality head.
A5 matches96 nominal forwards; actual tokens, supervised labels and compute
remain unequal and must be reported. No longer training selected by correctness.

All native runs use CPU<=4 threads, shared lease/RAM/thermal protocol, prompt+
training sequence cap512, greedy output cap48, standard FP32 KV. Count actual
padded decoder positions/forwards with nonmutating hooks and raw trace provenance,
logical queries, input/output tokens, duration and host memory. GPU memory is
unmeasured under CPU execution. No truncation, clipping or oracle repair.

Adapted readouts measure fixed own-state and direct h1/h2/h4. Supplement all
steps' correctness, wrong-intermediate/correct-final cases and conditional no-op
errors. False copying after prior rollout error alone is not shortcut proof.
Calibration is unknown without probability captures. Require B0 and each arm's
supplementary R/P behavior audit on the same fixed fixtures before any promotion.

Exploratory continuation requires paired h4 net gain>=2/24 over A0, canonical
h4 net loss<=2/12, no unexplained retained-skill regression and feasible runtime.
This is an engineering gate, not efficacy or significance. Report paired effect
sizes and descriptive uncertainty; no single-seed/binary nonsignificance is
equivalence. Planning, three-seed replication and independent sealed co-primary
confirmation require separate qualification and registration. H4/H5 architecture
changes and model integration remain conditional. No public weight upload/merge.
