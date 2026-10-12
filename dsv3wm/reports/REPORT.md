# DSV3WM research report — 2026-10-11 (America/Chicago)

This is the initial native-pilot checkpoint. The full study remains open; see
[remaining work](NEXT_STAGE.md) for the stronger baseline qualification required
before larger training or confirmation.

**Conclusion: native implementation feasibility is established on CPU; useful DeepSeek-inspired world-model improvement is not established.** A0/A1/A2/A5 trained on the actual 596M-parameter parent. On two new register development parents, every accepted exact-state metric was 0/2; all four planning screens were 0/1. The short runs are underpowered and undertrained, so these are operational failures and inconclusive efficacy findings, not proof that MTP or multi-horizon learning cannot work. No independent confirmation result is reported.

The GPU smoke was blocked by the existing shared lease. CPU execution of the real checkpoint is native evidence; small tensor and mock-planner tests establish code properties only. Historical BWM/RCWM results were audited, preserved, and never used as tuning banks.

## Baseline and primary-source findings

The parent is `Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze`, a dense Qwen3ForCausalLM with 596,049,920 parameters, 28 layers, hidden size 1024, 16 query heads, 8 KV heads and tied 151,936-token embeddings. “Sparse” describes its sparse update, not a DeepSeek MoE. All 310 stored tensors are F32. SHA256: `a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284`. Weights stayed frozen, FP32/TF32-off; every new LoRA stayed unmerged.

Historical BWM B2 scored 73/110 on October 9 and H2 scored 88/110 on October 11. Both are single-seed results; the newer replay requires wording “learned F plus deletion-consistency check.” That check rejected 19.6%/27.8% of queried illegal actions. They are not pure raw-legality baselines, nor matched comparators for these rank-4 adapters. RCWM PREREG-2’s ledger final-state accuracy 0.029→0.698 and regret 0.544→0.056 are qualified in-distribution, seed-0 findings with known status/range/serialization and prerequisite-copy limitations. See [baseline inventory](BASELINE_INVENTORY.md) for exact dates, hashes, configurations, uncertainty, licenses and replication gaps.

DeepSeek’s sequential future-token supervision differs from state-horizon prediction. This implementation uses lightweight positionwise bottlenecks in place of V3 Transformer auxiliary modules; it is an adaptation. MLA compresses learned attention KV representations; ordinary state summaries or standard cache reuse are different mechanisms. Verified planning labels require independent execution beyond a learned success claim. See [technical review](../research/DEEPSEEK_TRANSFER.md), based on [V3 v2](https://arxiv.org/html/2412.19437v2) and [V2 v5](https://arxiv.org/html/2405.04434v5).

## Dataset, preregistration and integrity

The fresh seed 202610111 collection contains 24 training, 24 development and 24 sealed confirmation trajectories across RegisterWorld, ledger and Blocksworld. Independent parent counts are 18/18/19. Repeated within-split parents are preserved as clusters, never advertised as 24 independent units. Exact/semantic cross-split duplication and renamed object isomorphisms through five objects are checked. Labels are deterministic exact-executor outputs. Prompts contain only domain/rules/s0/action prefix; future states, solver plans and hidden goals are excluded.

The confirmation vault is external and unopened after sealing: `../dsv3wm-vault-v1/confirmation.labels.jsonl`, SHA256 `b6c414bc84662adf70b265abf72bcd06f2d7eea8e0e97f9e1c58357c7d442076`. No old sealed rows/plans were read or re-evaluated. Disjointness against historical banks was not checked by opening them; “fresh” does not assert universal historical nonoverlap. Exact new-bank provenance is in [manifest](../data_manifests/pilot-v1.json).

The [pilot protocol](../prereg/PILOT_V1.md) and initial source/data/config freeze preceded training. [Amendment 1](../prereg/PILOT_AMENDMENT_1.md) records parser corrections, CPU-lane coordination and clustered-parent caveats before dev generation. [Amendment 2](../prereg/PILOT_AMENDMENT_2_CACHE.md) was made after the uncached A0 outcome and explicitly marks cache/chat follow-ups exploratory. The v3 freeze pins cache source, config, data and trained artifacts. Original freezes/results remain append-only. The early probe/A0 training lacked the newer CPU-lane lock; no competing holder was found, and all later heavy CPU work used the shared lock.

Confirmation P1 is proposed as four-step own-prediction rollout accuracy; direct h4 is secondary. Proposed P2 is goal success at fixed query/token budgets. One selected candidate versus A0/A5, Holm across four co-primary tests, +5pp practical thresholds, seed 0 designation and seeds 1/2 replication are specified prospectively. Bank size/power, complete hard-planning cohorts, retention margins and three-seed artifacts are NOT frozen. G5 is incomplete and the confirmation CLI refuses access. No pilot p-value is substituted for confirmation.

## Implemented and executed arms

| Arm | Native training status | Steps / forwards | Trainable params | Input / response / auxiliary tokens | Training s | Parameter-token proxy |
|---|---|---:|---:|---:|---:|---:|
| A0 | pilot-trained | 4 / 4 | 573,440 | 710 / 81 / 0 | 13.66 | 2.539T |
| A1 | pilot-trained | 4 / 4 | 624,640 | 710 / 81 / 77 | 15.44 | 2.539T |
| A2 | pilot-trained | 4 / 12 | 573,440 | 2168 / 245 / 0 | 40.83 | 7.753T |
| A5 | pilot-trained | 12 / 12 | 573,440 | 2142 / 249 / 0 | 40.93 | 7.660T |
| A3 | implemented/unit-tested; native training deferred | — | — | — | — | — |
| A4 | implemented/unit-tested; native training deferred | — | — | — | — | — |

All actual arms use the same frozen parent, q/v rank-4 alpha 8 adapters, seed 0, batch 1, AdamW lr 1e-4, default weight decay 0.01, dropout 0 and gradient clipping 1. No WSD schedule is used in this operational screen. The same seeded row order is used; A2 reuses each chosen parent for h1/h2/h4 while A5 consumes 12 one-step draws. Eight register training rows are available. The pilot adapter capacity/recipe differs from historical BWM rank-16 all-projection F and must not be treated as its reproduction.

A1 adds 51,200 parameters and 77 auxiliary token labels, two total future tokens including the ordinary head. A2 weights h1/h2/h4 losses 1/.5/.5. A5 approximately controls A2 cost:12 decoder forwards,2142 versus 2168 input tokens and 249 versus 245 response labels, with different label allocation and optimizer steps. A1 has no exact equal-capacity/compute control here. No superiority is inferred from these unequal budgets.

The “decoder_flops_proxy” field is a crude 6×total-parent-parameters×input-tokens proxy, not measured FLOPs. It includes parameters such as embeddings and omits checkpoint recomputation/auxiliary work; early run scope text was too narrow and is corrected here without rewriting those records. Actual token counts, forwards and wall times are more defensible comparisons. Activated parent computation remains dense; discarded MTP heads add no inference work.

## Dynamics and planning outcomes

| Arm | Dev one-step teacher-forced CE | h1 exact direct / rollout | h2 exact direct / rollout | h4 exact direct / rollout | Planning | Dynamics s | Planning s |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 1.770085 | 0/2 / 0/2 | 0/2 / 0/2 | 0/2 / 0/2 | 0/1 | 123.36 | 54.97 |
| A1 | 1.811802 | 0/2 / 0/2 | 0/2 / 0/2 | 0/2 / 0/2 | 0/1 | 121.42 | 54.93 |
| A2 | 1.762756 | 0/2 / 0/2 | 0/2 / 0/2 | 0/2 / 0/2 | 0/1 | 124.17 | 55.14 |
| A5 | 1.241176 | 0/2 / 0/2 | 0/2 / 0/2 | 0/2 / 0/2 | 0/1 | 124.48 | 55.14 |

The two development parents are one ordinary case and one invalid-action cohort; one-step copy accuracy is 2/2, while h2/h4 copy accuracy is 0/2. This is a tiny biased operational cohort, not broad dynamics validation. Strict output parsing requires one complete JSON response, integer states, boolean flags, unique keys and structural state consistency. Domain state-invariant checks are explicit parser refusals, not hidden transition simulation or oracle legality correction. Every direct/rollout response here was malformed. Rollouts stopped at the first malformed prediction; this does not measure meaningful four-transition compounding-error behavior.

A0’s first plain decoded object predicted x=55 and legal=true after halving x−11, followed by prose. The operative rule requires illegal odd halving to preserve −11. Thus failure includes wrong dynamics/legality, beyond trailing-text rejection. Raw [A0 traces](../results/A0_cached_dev.traces.json) and the corresponding other-arm traces are retained. A5 lowers token loss substantially without meeting the declared state interface.

Every planning arm used the same independent register parent, optimal distance 3, beam width 1, depth 4, upper cap 20 queries and 120,000 standard-cache processed-token positions. All actually consumed 5 queries,788 input tokens and 320 output tokens, then produced no accepted child/plan. Exact scoring occurred after selection; no simulator feedback re-entered search. This is one register feasibility screen, not the requested broad hard-planning confirmation bank. No model-selected plan means no native verified-distillation corpus or A3/A4 training is justified yet.

For each 0/2 binary state estimate the 95% Wilson interval is[0, 0.658]; for 0/1 planning it is[0, 0.793]. Every paired h4 comparison has 0 gains/0 losses and exact McNemar p1; exploratory Holm-adjusted p1. These tiny samples cannot establish equivalence or reject useful effects. No shared cases are pooled across seeds. [Machine-readable summary](../results/SUMMARY_V1.json) contains paired counts and intervals.

## Retention, representation and efficiency diagnostics

| Model | Six-fixture token-weighted old-format CE | Difference from core | Behavioral retention pass |
|---|---:|---:|---|
| Frozen core | 1.273320 | — | unmeasured |
| A0 | 1.243929 | -0.029391 | null / unmeasured |
| A1 | 1.244203 | -0.029117 | null / unmeasured |
| A2 | 1.244860 | -0.028460 | null / unmeasured |
| A5 | 1.190558 | -0.082762 | null / unmeasured |

Three fresh RegisterWorld and three ledger fixtures use original prompt formats. Exact fixture/tokenization hashes and per-case CE are preserved. Lower CE on six cases is not retention of behavior, the 375-item guard, catastrophic-forgetting exclusion or noninferiority evidence. The registered original BWM F, SHA `c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466`, passed 6/6 original-format constructed behavior cases in 17.81 s. It was loaded unmerged and untouched. This demonstrates working inherited functionality rather than an inability of the parent ecosystem to predict transitions.

The post-hoc A0 chat-prefix diagnostic still scored 0/2 at all horizons, but decoded 179 rather than 768 tokens and had CE 1.181236. It emitted old-schema/extra-key responses, including `{"x":-11,"y":11}`. This is evidence of representation sensitivity; extracting/repairing answers after seeing outcomes would change the benchmark and is not done. The frozen confirmation remains sealed.

The standard-KV native parity probe passed, maximum logit error 4.7e-05, identical HF greedy raw IDs and BWM text/length/EOS. Its actual 6-token greedy completion took 3.86 s full-prefix versus 1.18 s cached. The original no-cache A0 dynamics screen took 635.75 s versus 123.36 s standard-cache. That comparison also changes last-logit materialization and CPU idle policy; it does NOT isolate caching alone or establish a DeepSeek memory effect. The matched cached arms all use the same instrument.

The resident KV estimate peaks at 52,985,856 bytes on these plain screens,224 KiB per cache token for 28 layers×2×8 KV heads×128×4 bytes. This is an analytic tensor estimate, not measured dedicated GPU peak or allocation-transient memory. Standard KV reuse is not compression or MLA. Prefix sharing, quantization, low-rank history, specialist routing and native attention conversion were not evaluated.

## Tests, execution environment and gates

56 unit/integration tests passed; compileall passed. Tests cover full/delta semantics, renamed-parent duplication, sealing/overwrite refusal, future-label interventions, causal MTP2/4, masked CE value/gradient parity, loss weights, strict parsing, independent own-state rollout, late verification/hard negatives/training-parent allowlists, cache budgets/EOS, McNemar/Holm/bootstrap and direct-student top 1/pass@k separation. Small mock cases are not native efficacy measurements.

Native CPU G1/G2 probe: baseline and disabled-experimental CE 2.521045207977295; initialized LoRA hidden-state difference 0; causal future-label prefix difference 0; finite gradients on 624,640 trainable LoRA/MTP parameters. Native two-token MTP and h1/h2/h4 training completed. Four-token MTP, shared-head, shuffled-target, delta/mixed and legality-head options have implementation/unit coverage where specified but no native ablation outcome. No claim of native shuffled-target or legality-head training is made.

Host: RTX 4070 Laptop 8188 MiB, Ryzen 9 7940HX, 16 physical/32 logical cores, approximately 32 GB RAM and 31 GiB initial free disk. Runtime: Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0. CPU native runs used four threads and 12 GiB load/4 GiB boundary floors. One in-flight uncached working-set sample was 3,736,174,592 bytes, not peak RSS. GPU peak allocation for CPU runs is 0; native GPU memory/latency is unmeasured. Environment/version evidence is in [snapshot](../results/ENVIRONMENT_V1.json).

GPU smoke failed before model loading with LeaseError: shared Temp/rineya-gpu-0.lock, label bwm:bundle-bundleI, PID 480900, worker 590308. The recorded worker list prevents reclamation even when those PIDs are not observed live. No lock was overridden and no OOM/retry loop was launched. Two user-authorized RDW native slots were coordinated using our token-checked CPU reservation; that bundle remained untouched and our confirmation vault remained protected. Pending peer work is separate from these experiment outcomes.

| Gate | Actual status |
|---|---|
| G0 | audited: current baseline, parent/adapter hashes, boundaries, resources and license/reproducibility gaps |
| G1 | native CPU loss/causal/init parity passed before pilot; inherited registered-F behavior smoke 6/6 completed as later qualification |
| G2 | CPU/unit/native-CPU implementation checks passed; native GPU smoke blocked by lease |
| G3 | operational A0/A1/A2/A5 pilot executed; usefulness gate NOT met; schema/dynamics failures preserved |
| G4 | deferred: no useful G3 signal; no larger matched study or three-seed training |
| G5 | deferred: incomplete registration, no confirmation labels opened |
| G6 | specialists/compressed-memory deferred; existing standard-KV engineering diagnostic only |
| G7 | no promotion qualified |

Failures led to explicit parser/coordination corrections and a standard-cache diagnostic, not altered primary confirmation outcomes. G4–G7 were not silently bypassed. Complete hyperparameter selection, hard-bank construction, behavioral retention, original-capacity baseline and seed replication remain required before efficacy claims.

## Integration recommendation and next experiment

| Mechanism | Classification | World-model adapters / skill architecture | Dense core / test-time planning |
|---|---|---|---|
| Sequential bottleneck MTP | inconclusive efficacy; implementation ready for further qualification | keep isolated/unmerged, retain disable switch | no dense promotion; heads discarded at inference |
| Direct multi-horizon state loss | inconclusive; tiny CE gain below longer-training control | further development after baseline/format qualification | no primary planning benefit demonstrated |
| Independently verified planning distillation | implemented/unit-tested; native efficacy unmeasured | approved training-parent allowlist, exact re-verification, separate student APIs | no student/core promotion until useful verified corpus and native A3/A4 study |
| Lightweight experts | deferred / not justified by these errors | no router/adapters trained | no new MoE recommended |
| MLA/compressed memory | deferred / unmeasured | no native conversion | standard KV reuse qualified narrowly on CPU; GPU and broader branches need qualification |

The next useful study should reproduce the validated BWM forward interface/capacity as the matched A0 baseline, qualify the chosen response schema on development, then use substantially more fresh training parents and an information-rich larger pilot. Existing registered adapters may provide a controlled warm start, with explicit old-training exposure audit and unmerged provenance. Compare A1/A2 to equal-capacity and matched-token/compute controls; run shuffled targets and legality/full-vs-delta ablations only after the schema is learned. Reuse standard KV; add validated resumable optimizer checkpoints before long runs. Obtain a legitimately available GPU lease or a separately agreed CPU budget; do not override another programme.

Only after useful development signals: create exact-verified TRAIN-only plans, train/evaluate A3/A4 direct proposer and with search, freeze a complete fresh dynamics/hard-planning confirmation registration and behavioral retention guard, and run at least three independent seeds. Expand goal/rule compositions, renamed encodings, numerical ranges, object counts and longer action sequences. Current generator stresses are implemented, but native generalization outside the two register parents is not measured. No transferable compact-world-model efficacy claim is made.

## GitHub and reproduction record

Final code review added prospective v4 replay gates: CLI-selected config, data,
adapter weights/config and qualified probe must belong to the freeze; recorded
training lineage and reused exact-semantics sources are checked. Student tasks
require canonical parent IDs. These are safeguards added after the original
outcomes, as recorded in [replay amendment](../prereg/REPLAY_AMENDMENT_4.md).
Original v1/v2/v3 files and numerical results are unchanged. Archived v3 code is
in checkpoint58ba757; current reproduction uses `freeze_replica.py` and v4.

The chat diagnostic changed role formatting and strengthened return-only/boolean
instructions, so its shorter outputs cannot be attributed to the role prefix
alone. Delta inference additionally uses explicit deletion-support/state
consistency refusal; no delta-native result is reported here.

Branch: `research/rineya-deepseek-v3-worldmodel-v1`, repository [LUNCY 579/rineya-world-model-research](https://github.com/LUNCY579/rineya-world-model-research/tree/research/rineya-deepseek-v3-worldmodel-v1). Baseline: `aeb19ab17128d884c4457ce59db6e926ceb13a7a`. Validated checkpoints 4be0803,108631d, 58ba757; final commit is reported in the chat/Git log because embedding a report’s own commit SHA would be recursive. No merge, force-push or model publication.

Changes are confined to dsv3wm. Package includes source, CLIs, configs, preregistrations, technical review, baseline inventory, tests, raw development generations, native records and this report. Large/sensitive weights and generated datasets stay local; [exclusion manifest](../results/EXCLUDED_ARTIFACTS_V1.json) records locations/sizes/SHA256, and sealed-label hashes are reused without reading labels.

See [README](../README.md) for exact commands. Original v1/v2 replay requires its pinned source commit; newer cache/native audits use v3 source and their hash gates. Fresh reproduction must use new output paths, regenerate seed-controlled data and freeze new trained artifacts before outcomes, rather than overwrite previous records. Private weights must exist locally at their documented paths and match hashes. Default native runtime is Windows-specific; no claim of cross-platform/native-GPU qualification is made.
