# Rineya world-model baseline inventory

Audit date: 2026-10-11, America/Chicago. Repository baseline: `aeb19ab17128d884c4457ce59db6e926ceb13a7a`, read in the dedicated `rineya-dsv3wm-worktree`. This inventory supersedes an initial audit of the older research checkout because the verified baseline adds BWM P-15 and P-16. Historical results below are evidence from records, not experiments re-run by DSv3WM.

## Audit scope and preservation

Read the required repository README, BWM README, DECISIONS, PREREG_PLANBENCH, PREREG_ITEM7_SEEDS and SOLUTIONS_AND_BENCHMARK_PLAN; RCWM REPORT and DECISIONS; relevant configuration, freeze, aggregate summary, license and source files. Read the operator's attached assignment. Inspected native core metadata and safetensors headers without loading model tensors or importing torch. No sealed raw test rows, submitted plans or reference-plan payloads were inspected. No BWM/RCWM file, branch, GPU lease or model weight was changed. The sole audit output is this DSv3WM inventory.

Paths below are relative to the verified repository unless an absolute native-artifact path is given. Exported BWM `results/` corresponds to native bundle `runs/`; references inside historical records retain native names. Their path strings must not be blindly resolved relative to the export.

## Frozen parent model and numerics

Native core: `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze`.

| Property | Evidence and value |
|---|---|
| Name and status | Titan-Sparse-Experimental 1.0.0-devfreeze; experimental research core, not release qualified. Native `CORE.json` and README. |
| Weight identity | Actual `Get-FileHash` matches registered SHA-256 `a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284`; 2,384,234,992 bytes. |
| Header verification | `safetensors.safe_open(framework="np")`, shapes/dtypes only: 310 tensors, exactly 596,049,920 serialized parameters, all F32. Metadata `format=pt`, `sparse001_keep=0.25`. No tensors materialized. |
| Architecture | Dense `Qwen3ForCausalLM`; 28 decoder layers; hidden size 1024; intermediate size 3072; 16 attention heads, 8 KV heads, head dimension 128; vocabulary 151,936; tied embeddings; embedding shape `[151936,1024]`; RoPE theta 1,000,000; maximum positions 40,960. This is the existing architecture, without MLA or MoE. Native config and header. |
| Lineage | Supa 1.7 plus per-tensor largest 25% by magnitude of the RRP-015 control update, selection seed 17. Source Supa weight SHA `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b`; consolidated trained FP32 tensors SHA `07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d`; checkpoint manifest SHA `3d5cef318a578c71e3d0fe10bdeb27657a31c21d4826f26a6a276e07cacd50cc`. Native CORE.source/construction. |
| Required execution | FP32 end to end; CUDA and cuDNN TF32 disabled; no BF16/autocast substitution, no FP8 primary experiment. Never silently dense-merge registered adapters. BWM D-02, `bwm/code/bwmn/lm.py` module contract/load_core; RCWM D-01/D-28, `rcwmt/runtime.py:set_numerics`. |
| Why FP32 matters | Native CORE records BF16 would erase 28.49% of retained delta elements; measured results belong to FP32 bytes. A cast core is a different model requiring separate validation. |
| Sidecar hashes, measured | config `25ace0268b891881bf1d5a95d18611a74ce1896922107d10adc526257a1962e8`; tokenizer.json `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`; tokenizer_config.json `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`. |

The native CORE carries one training draw for this weight artifact; its README contains a later pruning-direction replication amendment. Neither is independent seed replication of the BWM/RCWM adapters studied here. Native long-context 164/256 is report-only on a SPENT instrument, explicitly barred for TRAIN/SELECTION/CONFIRMATION, with source-answer churn and no equivalence conclusion.

## Adapter and training inventory

All four adapters use unmerged PEFT causal-LM LoRA, rank 16, alpha 32, dropout 0.05, bias none, on q/k/v/o/gate/up/down projection modules. Exported `adapter_config.json` files and BWM D-03 establish the recipe. The T2 safetensors header verifies 392 LoRA tensors and 10,092,544 parameters; the BWM adapters use the same dimensions/target modules (about 10.09M each, about 20.19M for a pair). Backbone parameters remain frozen. BWM training uses target+EOS supervision, AdamW betas 0.9/0.95, WSD 10/70/20, gradient clip 1 and checkpointed/chunked supervised cross-entropy (D-03/D-14).

| Artifact | Registered weight SHA-256 | Dataset and recipe | Measured training cost |
|---|---|---|---|
| BW_Fdelta_s0 | `2ed0bb0064bc6b11ec9c661840a38094e468462cea2ff833cda3bf234ead65ae` | 12,000 generated 4-6-block one-step transitions; delta targets; 600 steps, batch16, lr1e-4, seed0, max length256 | 1,551,649 processed tokens; 175,624 supervised tokens; 5,940.8s; 2.645GiB peak |
| BW_Pa_v2_s1 | `ee1c5aec5e4a55e30f6df0e30a3cd68242c928cd158f9e5cf06184f409108110` | 53,682 generated 3-6-block action-proposer rows, including whole-configuration goals; 1,000 steps, batch16, lr1e-4, seed1, max length256 | 2,781,649 processed; 476,002 supervised tokens; 14,804.9s; 2.896GiB peak |
| BW_Pa_v3_s1 | `48c9d279bd8813311f3232bc867a88d96604a93347270a16ac57829662e4c678` | Continues Pa_v2; mix_v3 contains 8,508 rows, half new rows from generated 6-10-block F-verified search plans; 400 extra steps, batch16, lr1e-4, seed1, max length640 | 1,272,786 processed; 267,786 supervised tokens; 5,099.3s; 3.015GiB peak |
| RCWM T2 s0 | `92bc7c7497c3cb2fc0c7c917b4f3b6bb931f0ed108144d969b26b18b42670b8a` | 1,501 oracle-labelled one-step rule-coverage rows across ledger/prerequisite/register; 190 steps, batch16, lr1e-4, seed0, max length512 | 464,409 processed; 41,666 supervised tokens; 1,858.8s; 2.608GiB peak |

Sources: `bwm/results/{BW_Fdelta_s0,BW_Pa_v2_s1,BW_Pa_v3_s1}/metrics.json`, `bwm/EXCLUDED_LARGE_FILES.json`; `rcwm/experiments/RCWM-T1/runs/t2/t2_s0/metrics.json`, `PREREG_2_FREEZE.json`. T2 weight SHA was independently hashed in this audit and matches. BWM weights are absent from the git export by policy; their full hashes above come from freezes/exclusion manifest, not a new audit of native BWM weight bytes.

Measured BWM adapter config SHAs: Fdelta `3038c77961f33461e1bef736959765434c0be473d991b0ef92c4813d269b3b8c`; Pa_v2 `f6a2c4efb381678ac44bebef1165d84d587a81e72c8a8ea1c27062f2bc7494bf`; Pa_v3 `d9d8a9c6556d602d3497772eabf0d2d6751d90284bfb27a9a558e9329be7a4a1`. Registered T2 config SHA: `50d9a776e093a952ac440a77c9cd7cd8e51eb22a85eb93b383aee1eacc185d9d`.

Dataset hashes from exclusion/freeze metadata, without opening historical test payloads:

| Native training dataset | SHA-256 |
|---|---|
| BWM runs/BW_data/forward_delta_train.jsonl | `9108e25b20651941754a8d284748e06c6206e68250f79ef35025d5b2c83e8c2c` |
| BWM runs/BW_data_v2/proposer_actions_train.jsonl | `6b46743a2837c511902493adf90b231e795b4849d2df81963d4494421277709c` |
| BWM runs/BW_DISTILL_r1/distill_rows.jsonl | `e0d1c72822a3cee7151afec4a2503429552e18f061e0f17ad197264c0a41928b` |
| BWM runs/BW_DISTILL_r1/mix_v3.jsonl | `7dadf5c9ab271383489d91c3394d44fac580bc574ffd72a6eec430035af0c656` |
| RCWM data/t2_train.jsonl | `bf5a22765757856517496b64e54d4d7b307dbb50af8f4e30540ff69937a1d744` |

## Historical Blocksworld measurements

### B2, 73/110

`bwm/docs/DECISIONS.md` R-05/D-48/P-08 and `bwm/results/FREEZE_B2_beam.json`: freeze written **2026-10-08 13:55:02 CDT**, after selection on generated val_v2 and val_large only. Eight-shard once-only evaluation completed **2026-10-09 ~06:17 CDT**. P-08 lacks an explicit completion timestamp, so this date/time is inferred from the original native artifact metadata: hard_beam/shard_7of8/summary.json LastWriteTimeUtc `2026-10-09T11:17:42.5155015Z`, merged summary `2026-10-09T11:17:45.0337881Z`, corroborated by the exported last progress entry `06:17:42 round 8: active 2 -> 0`. This uses native artifacts, not git checkout mtimes.

Frozen system: Titan-Sparse + BW_Fdelta_s0 + BW_Pa_v2_s1; delta F; beam `K=8,W=4,E=24,R=12,M=80,P=1,max_depth=80,eval_seed=0`. S-direct is the same Pa_v2, one greedy plan, M256, no world-model queries. The direct freeze has `fwd=exact` as a harness argument, but the direct planner does not use F; this is not oracle-assisted planning. Prereg SHA `adfc4354adb0262ad5175c517ff186e7a17e4efe95ddd33ba6ae4ad1fc3bf4d2`.

On PlanBench hard, n110, 6-15 blocks, optimal plan lengths20-40: **73/110 = 66.36%**, Clopper-Pearson 95% CI **56.73%-75.09%**; direct **5/110**, CI1.49%-10.29%; paired +68/-0, exact McNemar p6.776e-21. VAL/executor disagreements0. Aggregate summary records 186.9 new F queries and 7,991.9 proposer tokens/instance, summed shard wall34,770.6s. F caching means new model queries and all queried unique keys are different cost quantities. Neither wall time nor token count alone establishes compute matching.

Training was limited to generated problems <=6 blocks. Plans are assembled without exact execution feedback, submitted once, then scored by exact executor/VAL. Validity of all73 verified plans now has the mandatory **learned F plus delta-consistency check** qualifier from P-15 below.

Other B2 frozen results: mirror61/110 vs direct6/110; PlanBench nominal600 set actually602 instances,602/602 vs584/602. Hard-minus-mirror +10.9pp crosses the preregistered >10pp contamination flag, Fisher p0.128. The post-hoc taller-goal-tower explanation and reweighting (-3.5pp) are descriptive, not removal of the raised flag. See P-10 and `bwm/results/BW_TEST_B2/MIRROR_GAP_ANALYSIS.json`.

### H2, 88/110, and the distillation controls

`bwm/docs/DECISIONS.md` P-14: H2 finished **2026-10-11 13:07 CDT**; H2 freeze predates both hard runs (2026-10-10 20:29). `bwm/results/FREEZE_B3_H2.json` preserves every B2 beam argument and swaps only `pa` to BW_Pa_v3_s1. Eight shards, each summary matches freeze.

H2 **88/110 =80.0%**, CP95% CI **71.30%-87.02%**, vs73/110; paired +20/-5, exact McNemar p**0.0040773153**; both68, neither17. VAL disagreements0; all88 F+check verified plans valid; all22 failures unverified. Mean proposer tokens8,206.1 and new F queries198.0; summed shard wall31,996.0s. Search settings stayed fixed but realized token/F-query costs increased slightly; do not label all compute identical.

Pa_v3's additional training includes generated6-10-block problems. The <=6-block-only B2 claim **does not apply to H2**. B3's increment uses no simulator labels: its learned-F+check verified searches supply targets, while an exact simulator audits label precision211/211. The baseline data were simulator-generated, so neither the whole student nor whole system is simulator-free. DSv3WM's proposed independent-verifier distillation is a stricter new intervention and must be distinguished from B3.

P-11 B3 registered fresh-set direct comparison: Pa_v3 22/110 vs equal-extra-training Pa_v2c8/110, paired +18/-4,p0.0043, practical +12.7pp bar met. H1 hard direct13/110 vs5/110,+10/-2,p0.03857;12 of13 successes lie in the6-10-block distillation range. H2 did **not** include a Pa_v2c beam arm on hard, so the H2 contrast does not isolate search-plan content from extra training. B3 design was made after B2 hard results were known, although training-instance exclusions and fixed settings were audited. Single training seed and once-only registered result; no broad planning superiority claim.

### New baseline P-15: model and parser are separate

`bwm/docs/DECISIONS.md` P-15, registered CPU replay, independent implementation reproduced40,365 compared fields. Across logged F queries there were zero false-legal answers accepted by the parser and zero wrong accepted next states. But **19.6%** of illegal B2-hard queried actions (CI18.4%-20.9%) and **27.8%** H2-hard (CI26.0%-29.7%) were rejected **only** because predicted deletions were not present in the current state. Mirror24%, PB60213%, distillation22%. This support check is domain-agnostic symbolic consistency; it is not evidence that the model's raw legal flag learned all preconditions.

Every historical validity/precision statement must be scoped to F+check. The claim that rules were known only to the learned F is withdrawn. Exact-F ceiling bounds from registered replay: B2-hard[72,73], H2-hard[87,88], mirror[61,62]; PB602 and distillation fully oracle-equivalent. F was not the demonstrated bottleneck on hard. Better world-model loss alone need not raise this particular historical score. New DSv3WM evaluations must separately report raw model legality, parser refusal, wrong accepted states and end-to-end planning.

### P-16 published comparisons and errata

P-16 is **POST HOC, DIFFERENT REGIME**, domain-trained specialist+F+search versus zero-shot generalists. Upstream step0 direct answers at LLMs-Planning commit `fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284` were VAL-rescored, agreement330/330; direct counts R1=59, o1-preview=26, o1-mini=1 of110. Paired H2/R1+36/-7,p9.0e-6,Holm5.4e-5 (12 contrasts); B2/R1+30/-16,p0.054,Holm0.108, not significant. Never phrase these as unqualified "beats". Source: P-16, `bwm/bench/planbench/SOURCE.md` and `bwm/results/HARD_PUBLISHED_PAIRED/REPORT.json`.

Frozen PREREG_PLANBENCH retains o1-mini10%, aggregate-only and R1/TMLR attribution errors; preserve its hash and use P-14/P-16 errata. o1-mini direct is0.9% (1/110);10% is verifier-in-loop and its P-08 comparison is withdrawn. Hard whole-goal on-pair range is4-13, correcting original5-12. Mutable SOLUTIONS doc fixes its principal o1-mini line but still has stale older status/citation language; prioritize dated DECISIONS appendices and current summaries.

## RCWM ledger baseline and limits

`rcwm/experiments/RCWM-T1/REPORT.md` §3: PREREG-1 confirmed2026-10-07,run18:08-18:32. On300 fresh ledger+register parents,8 candidate four-event plans,256 held-out goals, Titan-Sparse one-step chaining reaches normalized regret0.448 (ledger0.539,register0.356). Five Holm contrasts pass. C0 is driven by register; C1/C2 beat comparators at/below answer-free floor, not latent learners in general. Ledger exact final state at4 steps remains0.024. Abundant symbolic learner ledger regret0.273 is better than rollout. Horizon4,L/R only; prerequisite exact0.006 and copy failures. This is a decision result, not exact universal simulation.

§3b: PREREG-2 freeze **2026-10-08 15:34:35 CDT**, run started15:34:36 and ended18:38; analysis once18:38. Fixed T2 s0 ON versus OFF in same process;150 fresh ledger parents:

- Normalized four-step decision regret **0.544→0.056**, ON-OFF **-0.488**,95% CI[-0.535,-0.443], Holm p0.0002.
- Four-step exact final state **0.029→0.698**, gain0.668,95% CI[0.642,0.695].
- Forgetting guard PASS on248 prompts, first-token OFF/ON KL0.50 times Titan's own delta drift; cumulative Supa/ON drift1.94 times, and64-token Supa agreement0.50→0.37. Guard pass is not preserved general ability.
- Same generator/marginals as training; payload-fresh, not OOD; chain_one frame; one fixed confirmatory adapter seed; register/prerequisite not fixed. Prerequisite false copying increases and applicable-changing accuracy0.59→0.35. No untargeted-LoRA/self-distillation control separates rule learning from frame adaptation.

§3c PREREG-3 confirmed **TRIGGER**,freeze2026-10-09 02:38,run06:34-07:19, all3 T2 seeds. Unseen non-posted statuses without "void" score0.01-0.18 versus trained void0.99-1.0; copy cue extends to Void/VOID/voided, not stated status rule. Magnitudes, renamed accounts and reordered keys degrade; key-order accuracy0.37-0.47. Out-of-range four-step exact0 in all arms.

§3d joint PREREG-LB freeze2026-10-09 23:12,run2026-10-10 02:06-05:19. BWM beam **enumerated model-free proposer,W2,E12**, T2 as F: **127/180=0.706**, OFF14/180,ORACLE160/180. Paired+113/-0, Holm p3.9e-34; needs-lookahead23/52 vs3/52,+20/-0,Holm1.9e-6. T2/ORACLE **0.794 [0.730,0.855]**, below registered near-oracle point bar0.80. Imagined-verification precision0.952; held-out surface only5/180. Training surface,in-range,one seed/config,model-free proposer; not a learned-proposer or general rule-reading result. BWM D-61 independently rescored every registered number. D-75 follow-up confirms ledger F has no deletion support check; LB's residual errors come from its state predictions.

## Replication and retention status

BWM item7 prereg written2026-10-10 ~15:00. R0=(F s0,Pa s1), R1=(F s1,Pa s2), R2=(F s2,Pa s3), same B2 recipes/settings. Fresh n40 mirror-style set,seed20261014, exclusion-controlled and hashed before training. Both independent replicates must have beam-direct gain>=0.30 and paired exact p<0.01. Historical hard/mirror/602 remain single-seed results regardless; item7 does not establish Pa_v3 or key-order FK robustness. At audited git baseline D-71 calls seeds pending; D-77 has addendum/freeze work still to do and no exported FREEZE_I7 result. Live lease label bundleI suggests activity, not a completed or inspected seed result.

RCWM T2 ledger improvement direction passed its preconfirmation three-seed pilot gate: drops about+0.47/+0.41/+0.48; PREREG-2 confirms only the fixed s0 artifact. PREREG-3's shortcut replicated in all3 seeds. Distinguish pilot seed replication from three independent sealed confirmations.

RCWM R25 (2026-10-11 04:27) and BWM D-71: T2s0, Fdelta_s0, Pa_v2_s1 all **RETAINED/GO**,5/5 binding selected categories,75 items/category, both graders. Draw contains450 prompts but selected scored comparisons use375=5x75; do not confuse these counts. BWM's independent rescore matches1,077 fields. Low-power guard pass **is not non-inferiority** and says nothing yet about Pa_v3 or a new adapter.

Mandatory caveats: Fdelta unit conversion30→23/75,8 losses/1 gain,Holm0.195; one gain makes pass fragile. Pa_v2 answers5/75 unanswerable questions with invented values,abstention0.933 vsOFF1.0; pooled net+9 primary but-10 lenient. Titan-Sparse itself loses unit conversion49→30 vsSupa,Holm0.0008; adapters add further nonsignificant drops. Each adapter tested alone,short templated answers,one seed,5 of8 categories. Registered results bind through DONE→provenance→adapter hashes, not RET_RESULTS alone.

Shared joint rule RCWM D-49/BWM X-04: include control-skill rows for every operation an adapter must keep; report retention of untouched operations. RCWM register trace screen R23 learned conditional predicates but failed swap retention and mix generalization (NO-GO); BWM key-order augmentation P-12 improves unseen orders to0.917/0.950 yet harms illegal-action preconditions and canonical accuracy (PARTIAL). These are concrete design hazards for auxiliary objectives.

## Reusable interfaces and isolation points

| Surface | Existing interface and restriction |
|---|---|
| Native model/training | `bwm/code/bwmn/lm.py`: load_core,attach_lora,load_lora,encode_pair,chunked_ce,train_sft,generate,score_targets. Separate prompt/target tokenization,-100 prompt/pad labels,target+EOS only; reject overlength rather than truncate; unmerged adapter switching. Do not edit pinned native utilities. |
| BWM RegisterWorld | `bwm/code/bwmn/rw2.py`: exact_step(x,action,limit)->(next_x,legal,terminated), dist_to/optimal_next_actions/executor tools. Goal information is separate from dynamics; exact instruments may label training or score, not silently enter learned inference. This differs from RCWM's modulo97 register_machine. |
| Blocksworld | `bwm/code/bwmn/blocks.py`: step/applicable/execute,random_towers/generate,signature,state_json/state_from_json,forward_prompt/target,delta variants,parse_forward_delta,VAL. State=frozenset facts,action=tuple; delta={add,del,legal}; reject unsupported deletions. Avoid test_signatures/hard_signatures in an unreviewed new pipeline because they open historical test inputs. Reuse precomputed exclusions/provenance rather than labels/plans. |
| Planner | `bwm/code/bwmn/blocks_planner.py`: LMForwardBW.predict(keys)->[(state,legal) or None], LMProposerBW.propose(items,eval_seed),rollout_many,plan_direct,plan_beam(...,goal_met=None,unmet=None). Actual pinned SHA verified `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960`. Caching and CRN sampling preserve frozen function/instance seeds. |
| Domain adapter | `bwm/code/bwmn/domain_planner.py`: LMForwardD/LMProposerD,ExactD,score_d; reexports frozen planner; refuses import when planner hash differs. Additive domain integration with existing checks, not edits to BWM. |
| RCWM environments | `rcwm/experiments/RCWM-T1/rcwmt/env.py`: step(family,state,event),rollout->s0..sk,one_step_messages,multi_step_messages,parse_state,cross_check; ledger/prerequisite/modulo97 register. The step return is actual state (not a universal legality tuple); use family-aware semantics. External marginal/source-path dependencies mean blindly importing every generator can open historical source pools. |
| RCWM generation/decisions | rcwmt/generation.py chain/e2e; rcwmt/decide.py and fidelity/decisions; scripts/lbeam_f.py LedgerF; confirm2_gen.py freeze verification/registered runtime identity. Preserve existing frozen evaluation artifacts and access boundaries. |

## Dataset access and licenses

| Material | Evidence and reuse boundary |
|---|---|
| Titan core | Native README YAML declares Apache-2.0. This is a declared model-card license; this audit does not establish a full training-corpus license chain. Native CORE says private archival backup,release_permitted=false. Keep weights private. |
| BWM adapters/package | `bwm/code/scripts/build_devfreeze.py` generates Apache-2.0 model-card metadata. Repository lacks a separate blanket BWM license grant. Generated training datasets have **no explicit dataset license located**. Internal authorized research reuse is the scope here; public release rights remain unresolved. |
| RCWM original software | `rcwm/LICENSE_REFERENCE_CODE.txt`: MIT, explicitly excludes cited third-party papers,implementations,datasets,trademarks and model weights. Preserve notice when vendoring original reference software. |
| RCWM training data/adapters | Synthetic/local T2 rows and private weights: **no explicit dataset/adapter license located**. RCWM software MIT must not be treated as a weight or dataset license. |
| PlanBench | `bwm/bench/planbench/SOURCE.md` plus LICENSE_LLMs-Planning_MIT.txt,upstream commitfc638a1aff7df3fe7a1a1d289fa2c04cc24dc284. Vendored evaluation assets described as MIT; TEST ONLY,never training,selection or prompt construction. This also applies to published result files and Logistics prompt/reference-plan assets. |
| VAL | SOURCE.md describes LGPL-3 upstream external binary used unmodified; no relicensing inferred. |
| Parent pretraining/fine-tuning source corpora | Complete content/license inventory **unknown in this audit**; native CORE provides lineage hashes but not a full rights manifest. New DSv3WM generated datasets must document their own provenance and license instead of implying upstream licenses cover them. |

Historical PREREG_1/2/3,LB,retention and BWM hard/mirror/602 are frozen/spent evidence; never debug or select a new system against their raw labels,plans or model outputs. Metadata may identify excluded parent signatures without exposing solutions. New training/dev/confirmation must use separate parent identities,traceable seeds,semantic duplicate controls,frozen hash manifests and no-peeking confirmation analysis. Keep unknown freshness/license facts explicit.

## Resource and runtime constraints at audit

Root session's read-only host inventory: NVIDIA RTX4070 Laptop,8188MiB GPU memory; AMD Ryzen9 7940HX,16 cores/32 logical processors;32GB host RAM; about31.3GiB free workspace disk. Free RAM and temperature vary with other programmes; these are not guaranteed available capacities. Native core alone uses2.38GB on disk, FP32 weights about2.22GiB, before activations/KV/optimizer. Recorded LoRA peaks about2.6-3.0GiB used throttled historical recipes; larger MTP heads/sequence lengths require new bounded memory measurements.

Validated historical interpreter: `C:/Users/clclc/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe`; PATH points there. Receipt runtime Python3.11.15,torch2.6.0+cu124,transformers5.14.1,PEFT0.20.0,safetensors0.8.0 (RCWM D-01,confirm2_gen.REGISTERED_RUNTIME,docs/understanding/runtime-env.md, scripts/master_chain.sh). Audit used only safetensors header access; it did not validate current torch imports,CUDA kernels or anchor generations. Secondary Chat-Bubble venv has different torch/PEFT and requires explicitly recorded parity before matched comparisons.

Read-only lease snapshot: `C:/Users/clclc/AppData/Local/Temp/rineya-gpu-0.lock`,label `bwm:bundle-bundleI`,holderPID480900,worker_pids[590308]. Neither PID was present in Get-Process at observation, but the worker list is nonempty. `bwm/code/bwmn/gpu_lease.py:_try_reclaim` refuses a nonempty worker list regardless of holder liveness. No reclaim/lock deletion or third GPU waiter was attempted. This is a concrete resource-coordination restriction, not evidence that item7 completed or failed. Owner coordination is needed before native GPU work can acquire its authorized turn safely.

Existing GPU safeguards: shared lease/strict alternation,4.5h cap,thermal watchdog,synchronization/duty-cycle gaps,70% allocator cap,foreign-VRAM checks and RAM floors. The GPU drives the display and historical jobs triggered TDR/display failures/reboots. Preserve these operational rules for actual native training. CPU has a shared exclusive heavy-lane protocol (BWM D-76); use existing owner coordination and do not start a competing heavy CPU job just because GPU is unavailable.

## Baseline interpretation for DSv3WM

The defensible starting claim is narrow: a frozen dense FP32 ~0.6B parent with small unmerged adapters supports useful trained-surface transitions, four-step ledger decisions and Blocksworld imagination search/distillation. Results have measurable serialization/status shortcuts, parser contributions, retention failures and limited seed confirmation. P-15 shows improving F alone may not improve this historical hard set; new sealed dynamics and planning tests are necessary. No result establishes that DeepSeek MTP transfers,that future-token loss equals future-state prediction,or that routing/cache compression will help. Those remain new,controlled hypotheses.
