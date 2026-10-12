# Native recovery research checkpoint — 2026-10-12 UTC

**No DeepSeek-inspired dynamics gain is qualified by the completed warm pilot.**
MTP loses two four-step rollout cases; multi-horizon supervision preserves the
inherited state ceiling without improving it. Both lose a retained legality
case. Planning success does not remove those regressions. This is an interim
consolidation of actual evidence, not completion of the research assignment:
H3 teacher/student outcomes, Ledger and Blocksworld native studies, qualifying
seed replication, independent confirmation and integration remain incomplete.

The initial cold rank-4 report remains unchanged in `REPORT.md`. Current raw
records, failed attempts and freezes remain separate. No confirmation labels,
historical frozen plans or owner weights were used to revise these outcomes.

## Paper mechanism and actual experiment

The executed A1 uses sequential local residual bottlenecks, shared frozen token
embeddings/output head and response-only causal future-token supervision. It
has one auxiliary depth: two total predicted token positions, rank16,
weight0.1 and51,200 auxiliary parameters. It has no auxiliary Transformer
attention, and its auxiliary module is absent at inference. It is an adaptation
of the sequential supervision idea, not an implementation of V3's complete
module, MLA or MoE. See the source-checked
[implementation addendum](../research/IMPLEMENTATION_EVIDENCE_2026_10_12.md) and
the preserved [technical review](../research/DEEPSEEK_TRANSFER.md).

A2 supervises separate full legacy responses at action horizons1,2,4 with
weights1,0.5,0.5. Boolean legality/termination tokens receive CE, but there is
no additional legality head. Direct future responses and independent own-state
rollouts are different measurements. Future-state labels and hidden goals do
not enter independent dynamics inputs.

## Parent, initialization and exposure

All native jobs use the real frozen596,049,920-parameter Titan-Sparse dense
Qwen3 parent, FP32/TF32-off, and an unmerged rank16 registered-F adapter:
`c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466`.
It has10,092,544 trainable LoRA parameters across q/k/v/o/gate/up/down.
Every dynamics/planning inference arm activates606,142,464 parameters.
The parent/core and BWM/RCWM owner files remain unchanged.

The fresh recovery TRAIN collection offers128 unique parents, with limits
24,32,40. Actual sampling is with replacement. DEV's teacher-known cohort uses
limits32,48,64; its unseen-limit cohort uses80,96. Thus the term `ID` in these
registered records means **known to the inherited F adapter**, not identical
to every warm adaptation's training distribution. The inherited training audit
and unknown parent-backbone exposure are in `RECOVERY_TRAINING_EXPOSURE.md`.

| Arm | Updates / nominal forwards | Parent draws / distinct parents | Unique parent-horizon targets | Input / supervised / auxiliary tokens | Training seconds |
|---|---:|---:|---:|---:|---:|
| A0 | 32 / 32 | 32 / 29 | 29 | 4,252 / 466 / 0 | 63.745 |
| A1 | 32 / 32 | 32 / 29 | 29 | 4,252 / 466 / 434 | 71.395 |
| A2 | 32 / 96 | 32 / 29 | 87 | 17,716 / 1,404 / 0 | 238.580 |
| A5 | 96 / 96 | 96 / 64 | 64 | 12,768 / 1,410 / 0 | 185.035 |

The `(parent,horizon)` target identity is not independent transition-law evidence
or a count of distinct response strings. Recorded `counts.transitions` counts
supervised horizon responses/nominal objective forwards, not unique parents or
simulator action executions. A2 and A5 match nominal forwards but differ in
updates, parent exposure, prompt length and label allocation. A2 uses38.8% more
input positions and28.9% more training time. A1 uses12.0% more measured training
time than A0 and has extra auxiliary capacity/supervision.

A0/A5 share the same first32 row draws, loss values and step32 adapter hash.
This is native training-prefix parity, not independent-seed replication.
Measured training process peaks are3.77–3.79GB. The6*parent-parameters*input-token
estimate excludes recomputation/head work and is a rough proxy, not FLOPs.
Sources: `TRAINING_SUMMARY_01.json` and the new
`EXPOSURE_PAIRED_SUPPLEMENT_01.json` under `results/recovery-v1/`.

## Fixed development dynamics and uncertainty

| Arm | h4 own-state rollout | Descriptive Wilson95 interval | Direct h2 | Direct h4 | Direct h4 valid schema |
|---|---:|---:|---:|---:|---:|
| B0 | 24/24 | 86.2–100% | not measured | not measured | not measured |
| A0 | 24/24 | 86.2–100% | 2/24 | 0/24 | 14/24 |
| A1 | 22/24 | 74.2–97.7% | 3/24 | 0/24 | 24/24 |
| A2 | 24/24 | 86.2–100% | 3/24 | 0/24 | 24/24 |
| A5 | 23/24 | 79.8–99.3% | 1/24 | 0/24 | 7/24 |

Every arm scores24/24 at h1/h2 rollout; adapted direct h1 is24/24. All direct
h4 state intervals are0–13.8%. These intervals describe case outcomes on this
fixed bank; one seed and shared rule families do not establish training-seed
or broad environment uncertainty. B0 direct h2/h4 is unknown, not zero.

A1 loses two unseen-limit h4 rollouts; A5 loses one. Versus A0, A1 has0 gains/2
losses (exact two-sided McNemar p0.5); A2 has0/0. Both planning contrasts versus
A0 are0/0. The original descriptive four-test Holm family rejects none. Neither
predictive arm passes the fixed>=2/24 paired rollout-gain gate. Nonsignificance
does not prove equivalence, and better preservation than A5 would not improve
on inherited B0's ceiling.

The additive A5 comparisons are posthoc: A1 h4 gains1/loses2; A2 gains1/loses0.
Both planning comparisons gain1/lose0. All two-sided p-values are1 and the
separate descriptive Holm family rejects none. Original P1/P2 and gates are
unchanged; these comparisons do not qualify a mechanism claim.

Malformed-only deferral accepts14/24,24/24,24/24,7/24 direct h4 replies in
A0/A1/A2/A5, respectively. **Every accepted direct h4 reply is still wrong.**
For own-state h4, A1 accepts23 cases with1 wrong state and1 abstention; A5 accepts
24 with1 wrong and no abstention. Grammar validity detects only some benchmark
failures. The exact operating-point counts are in
`SELECTIVE_RISK_DESCRIPTIVE_01.json`. No probabilities were captured: calibration,
Brier/ECE/log-loss and calibrated confidence remain unmeasured.

## Planning, selection and realized cost

| Arm | Goal-only success | Optimal move count / first action | Conditional normalized excess cost | All-search queries per solved | Output tokens | Position upper ledger | Seconds / sampled tree GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 12/12 | 7/12 / 8/12 | 0.173611 | 30.000 | 5,386 | 48,240 | 250.754 / 5.211 |
| A0 | 12/12 | 7/12 / 8/12 | 0.173611 | 30.000 | 5,386 | 48,240 | 251.069 / 5.208 |
| A1 | 12/12 | 7/12 / 8/12 | 0.173611 | 30.000 | 5,394 | 48,480 | 253.773 / 5.206 |
| A2 | 12/12 | 7/12 / 8/12 | 0.173611 | 30.000 | 5,386 | 48,240 | 249.932 / 5.210 |
| A5 | 11/12 | 6/12 / 8/12 | 0.189394 | 32.727 | 5,386 | 48,240 | 250.099 / 5.207 |

Every arm spends360 learned transition queries and42,681 prompt tokens. The
query-per-solved ratio includes failed-search expense; average cost among only
successful searches is a different statistic. Normalized excess cost is
`(successful_move_cost-optimal_moves)/optimal_moves`; failed cases keep null
cost/regret and remain planning failures. It is success-conditional path
inefficiency, not an all-case decision-regret score. Paired cost comparisons in
`PLANNING_COSTS_DESCRIPTIVE_01.json` use matched common-success parents, with
one-seed parent-only descriptive bootstrap intervals. They do not impute zero
regret to failures or infer equivalence.

The position ledger sums per-parent conservative bounds. A separate original
batch estimator is47,880 for B0/A0/A2/A5 and48,120 for A1; the analysis keeps
both scopes separate. Neither is measured physical decoder FLOPs. GPU memory
is unmeasured; process-tree host peaks and elapsed times are observed CPU costs.

All selected successful paths have correct intermediate states. Unused-branch
own-input state/legality checks are358/360,356/360,349/360,358/360,349/360 for
B0/A0/A1/A2/A5; A1 includes4 malformed/unknown queries. There are245 unique
conditionings in the first four arms and240 in A5, across12 parents. Queries
are dependent branch observations, not360 independent trials.

The registered P2 success is nonterminating goal-state achievement. Every
selected success here omits terminal finish. H3's legal complete plan ending
in `finish` is a separate contract; these counts are not H3 student success.

## Retention and failure interpretation

On the six fixed original-format behavior cases, B0/A0/A5 score3/6 and A1/A2
score2/6. The three Ledger cases fail in every arm, including B0; they certify
no inherited Ledger behavior to retain. Register behavior is3/3 versus2/3:
A1/A2 predict legal=true for add1 at x32/limit32 despite returning unchangedx32.
State-only accuracy masks this known legality regression. No broad retention
pass, noninferiority result or catastrophic-forgetting exclusion is supported.

Direct response grammar improves without accepted state improvement. The raw
failure audit reports state copying and matches to first/last-action
counterfactuals, but those are observational signatures, not causal attribution.
Token-weighted teacher-forced DEV loss was measured in the original tiny cold
screen; it is not a measured warm DEV loss. Warm composite training losses
cannot be compared across different objective sums as evidence of state learning.

## Unexecuted work and integration decision

H3 A3/A4 and real C3/C4 continuation controls are implemented. The fixed32-parent
teacher pool is still unexecuted; its first launch failed lane admission before
model loading. Student continuation requires a qualified independently verified
shared corpus, its own warm-parent direct/reranked comparison, and the new fixed
dynamics/behavior guards. Those guards are implemented/tested, not native results.
The additive proposal/reranking cost reducer is implemented and tested, but
there is no genuine H3 output to analyze yet. Calibration remains unmeasured;
text-only outputs cannot recover predictive probabilities.

Ledger T2 and constructed Block G1 native qualification are pending. R/P
retention V2 restores the R training event surface while P deliberately probes
the source's original one-step framing, not T2's trained single-event multi-step
frame. Renamed Block inputs and Ledger key/tick/status stresses are frozen
inputs without native outcomes. Current observed generalization is confined
to the registered RegisterWorld compositions and numerical/rule-limit cohorts.

| Mechanism | Current classification | World-model adapters | Dense core | Composable skills | Test-time planning |
|---|---|---|---|---|---|
| H1 local sequential MTP | Inconclusive efficacy; tested recipe not recommended | No promotion; retained-legality loss | No transfer | Isolated research module only | No demonstrated gain; aux module discarded |
| H2 full-response horizons | Inconclusive efficacy; gain gate failed | No promotion; retained-legality loss | No transfer | Isolated unmerged prototype | No gain over A0/B0 demonstrated |
| H3 verified plans | Inconclusive; native efficacy unknown | Require corpus/student/guard evidence | No transfer | Separate research proposer only | Pure and learned-rerank outcomes pending |
| H4 routing | Not recommended under current evidence | No complementary-specialist evidence | No MoE rebuild | No router qualification | No routing claim |
| H5 latent/compressed memory | Inconclusive; conversion deferred | No conversion | No attention rewrite | No memory qualification | No compression claim |
| Standard KV reuse | Ready for further engineering qualification | Preserve validated numerical behavior | No architectural change | Existing interface reuse | Six-token CPU parity only; broader latency/success pending |

There is no supported model/adapter/skill promotion. A negative operational
pilot is not proof that the mechanism cannot work elsewhere. Keep fixed failed
gates, frozen cases and the sealed vault intact. Successful feasible pilots
would still need at least three independent training seeds and complete fresh
sealed co-primary dynamics/hard-planning registration before confirmation.
See `GATE_STATUS.md` and `NATIVE_QUEUE.md` for exact remaining dependencies and
new output paths. The full research objective remains open.
