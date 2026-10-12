# Native H3 teacher corpus qualification — 2026-10-12 UTC

The unchanged32-parent TRAIN pool produced27 accepted independently verified
plans,20 distinct action sequences and all three rule limits. The fixed corpus
gate passed (minimum16 parents/two limits/four plans). No student, DEV or
confirmation performance is established by this TRAIN collection.

| Rule limit | Accepted / frozen parents |
|---|---:|
| 24 | 9/11 |
| 32 | 6/7 |
| 40 | 12/14 |

Every retained candidate is the first independently exact-positive eligible
candidate at learned rank0. All27 use the registered syntactic final `finish`
completion, made after a learned goal prediction and before verification.
Each plan is legal, actually terminates at the requested goal and has at most
six actions. No oracle plan was inserted, search restarted or task redrawn.

Two failed parents never predicted the goal, two had only overlength goal
candidates, and one predicted the goal through an actually illegal plan.
The whole32-parent pool and its failures remain. There are41 eligible candidates
across28 parents and61 predicted-goal candidates across30 parents. The920
negative/refused candidates comprise514 nonterminating,183 wrong-goal,203
illegal and20 overlength grammar refusals. Another13 exact-positive candidates
are unselected diagnostics. These sets are not positive plan-CE labels.

Independent audit reproduced all940 candidates reaching exact verification;
stored outcomes and learned rankings match. All960 causal inputs follow
model-produced states. The96 learned transitions in accepted source plans
have no state/legality/termination discrepancy against independent execution.
Original32 task bytes and128 TRAIN rows, source/model pins and corpus admission
were reverified without a model or DEV/vault access.

Costs:960 logical learned transition queries,113,651 input and14,165 output
tokens;841.880 seconds including loading/hash/collection/verification. The
manifest's broad legacy position bound is127,680, while summed per-task padded
bounds are128,640. The difference is960 (one per query); preserve both scopes.
The recorded90 decoder forwards are a lower-bound estimate that does not
enumerate internal chunk splitting, not a physical-call/FLOPs measurement.

CPU11 admitted with11.766 GiB physical and19.535 GiB commit availability after
imports. The sampled process-tree peak was5,147,717,632 bytes (4.794 GiB).
Fifteen after-launch PDH/RAM observations show available RAM at least7.301 GiB
and maximum temperature95.9 C; the observer changed no process state. The
inherited guard emitted a stale-health-file warning. Fresh observations were
recorded separately without writing BWM owner files or changing the sampler.
GPU memory is unmeasured. The native child and health observer exited; CPU
lease release preceded the explicitly coordinated E06 peer handoff.

The corpus SHA256 is
`3a2a25a33b7dc983b422b67fc1a1977d7315896e42c72132a454012c27f1841b`.
The teacher source freeze is
`aa49f0e28654192a88c76739a4f8c15e6d4d097dbb0596ec8b6ed1db7b4a85c1`.
Primary evidence: `results/recovery-v1/H3_teacher_B0_retry01.manifest.json`,
resource/health records and `H3_CORPUS_QUALIFICATION_01.json`. Private corpus,
candidate/search traces and negatives remain excluded with file hashes.

Actual final32-step A0/A2 weights and the shared corpus now have a prospective
student-training freeze. A3/A4 and C3/C4 recipes are unchanged. Separately
registered CPU11 student/proposer/guard entrypoints preserve all old loaders
and historical freezes. Their native outcomes remain pending; no larger-study
or integration gain gate is relaxed by teacher corpus qualification.
