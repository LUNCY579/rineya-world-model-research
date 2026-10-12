# Recovery audit: exposure of the registered RegisterWorld forward adapter

Date: 2026-10-11, America/Chicago. Purpose: qualify an original-format/warm-start recovery experiment without claiming old training keys are unseen. This is provenance and training-support reconstruction, not model evaluation or retraining.

## Immediate conclusion

The registered `F_titan_main` adapter contains the **complete 1,455-key one-step table** for limits **32, 48 and 64**, every integer state **-L through +L**, and each action **add1, sub1, double, halve, finish**. Every valid state/action/limit combination in that support must be labelled **seen by the adapter**, including illegal-action no-op labels. A new goal, random seed, trajectory or parent-world ID does not make those constituent one-step keys unseen.

Training examples and the seeded exposure order are reconstructible without historical evaluation files. A pure tokenizers-only reconstruction reproduced both recorded totals exactly: **639,400 processed token positions and 71,061 supervised positions**. This supports exact reconstruction of the input/target inventory and order; it does not establish byte-identical retraining with today's amended trainer.

## Artifact identity and recorded training metadata

Native adapter directory:
`C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/F_titan_main/adapter`.

| Artifact | Actual SHA-256 measured in this audit |
|---|---|
| adapter_model.safetensors | `c465356f85ead24b609af2b8b1cb0a14799e6d2d16116944c9c607c933e66466` |
| adapter_config.json | `55b8fb922c8957573cfd36910ad64bbf4e328f23f00a58dcb889b899b64ca065` |
| RUN_META.json | `ef67edaf30d8babb5157bd82acb29a6adb2178ee6b53dad36322b16861b02bdc` |
| ../log.txt | `acf15b2d8a2af8df3abe14c0136f5a710dce297374de34f6b2256d0504e17311` |

The weight hash matches `bwm/EXCLUDED_LARGE_FILES.json`; weights remain local and unmodified. Config declares rank16, alpha32, dropout0.05, q/k/v/o/gate/up/down, no bias, PEFT0.20.0, frozen Titan-Sparse parent. It is a different capacity/initialization from the earlier DSv3WM q/v rank4 adapters. Warm-start controls must all start from this same registered unmerged artifact if that becomes the recovery baseline.

`adapter/RUN_META.json` records: Titan-Sparse; rows1455; families3000; seed0; steps300; batch16; lr1e-4; zero_shot_only=false. Training records show 1455 available rows,639400 processed tokens,71061 supervised tokens,574.6s and peak3.135GiB. These are historical training measurements, not new native runs.

The first training-only log line is:
`unique transitions available 1455; training on 1455; illegal 294; finish 291`.
No `TRAIN_KEYS.json` is present in the registered adapter directory. The currently exported trainer writes that file, so its absence indicates incomplete archived key metadata, not lack of a reconstructible support set. Current source includes later flags/checkpoint/microbatch amendments; the old RUN_META lacks those new flags.

## Source evidence for support

- `bwm/code/bwmn/rw2.py:145`: `ID_LIMITS=(32,48,64)`; `ACTIONS` is add1/sub1/double/halve/finish.
- `rw2.py:327`, `build_training_rows`: executed transitions come from TRAIN families only; states/action/limit and exact next-state/legal/termination labels are retained. No goal appears in forward prompts.
- `bwm/code/scripts/train_forward.py`, main training selection: collect unique `(x,action,limit)` keys from generated training transitions, sort, shuffle with `random.Random(seed)`, truncate to `--rows`, render via `rw2.forward_sft`, then train the frozen-parent LoRA.
- `bwm/docs/DECISIONS.md` D-05 explicitly calls the full ID result **in-sample: training covers every ID key**. `scripts/f_factorized.py`'s training-construction comment independently says F_titan_main saw every key at limits32/48/64; this audit read code only, not its cached labels/outputs.
- Native and exported `rw2.py` hash identically: `1195ea10044b28cecb7f30244aa2e51628222748c1e15d6bebb68ad5d11d3da9`.
- Native and exported `train_forward.py` hash identically: `1ee4c75382342a1fb4bac513bcdb3e994c17f6c44041d5b9bdeb7623e26d8d86`.

The complete support has:

| Limit | Distinct integer states | Keys, five actions each |
|---:|---:|---:|
| 32 | 65 | 325 |
| 48 | 97 | 485 |
| 64 | 129 | 645 |
| Total | 291 state-limit pairs | 1,455 |

There are291 keys for each action. Exact-law enumeration gives294 illegal keys:144 odd halving,144 doubles beyond the limit,6 add/sub boundary violations. Finish is legal and terminating on all291 states. This agrees with the recorded training-only log. The support concerns the original valid-action contract; unknown/malformed-action wrapper semantics are an extension.

## Reconstruction performed without a model load

In memory only:

1. Enumerate and lexicographically sort all `(x,a,L)` above.
2. Shuffle the full1455 keys with `random.Random(0)`; render each using the original `forward_prompt` and `forward_target(exact_step(...))`.
3. Follow today's `lm.train_sft` row-order construction: repeatedly shuffle index range1455 with `random.Random(0)`, then take4800 exposures=300steps×16.
4. Use `tokenizers.Tokenizer.from_file` on the verified core tokenizer, with add_special_tokens=false and one EOS per response. No torch import, model tensors, GPU or CPU-model execution.

Reconstructed counts are **639400/71061**, exactly the RUN_META values; max encoded prompt+response+EOS length134. All1455 training rows occur in that reconstructed exposure order. This also corroborates the current prompt/target serialization against the recorded token totals.

Hashes of **newly reconstructed canonical JSON**, using `json.dumps(value,sort_keys=True,separators=(',',':'))` then UTF-8 SHA-256:

| Reconstructed material | SHA-256 |
|---|---|
| Sorted1455 support keys | `34cad2ffd91958e361bf092302a66f02afb2ba57f9e97ab20017a3e2d80337a8` |
| Seed0 initial key ordering | `bc4ba67666c01ae1c808994f484c1e9e126eae15f862a11fe378a49fccb814be` |
| Ordered current original-format SFT rows | `7989a091e0b667c19558100d27990b64b0b20adb4d68f52fb952a848f7042f7e` |
| First4800 current epoch-shuffle row indices | `b3673e780262f1f9554edb8c54a0e0f2a5f14a86147c669d35eae992e09c1fa1` |

These are reconstruction hashes, **not hashes of a recovered historical dataset file**. They do not freeze today's training algorithm as the exact old executable. Historical optimizer/dropout RNG state, dependency/kernel behavior and amended batch implementation remain necessary to reproduce final weight bytes. Existing D-03 describes AdamW0.9/0.95,WSD10/70/20 and target+EOS supervision; do not silently substitute the earlier DSv3WM screen's default AdamW/no-WSD recipe and call that exact replication.

## Conservative recovery cohorts and wording

**ID composition cohort:** generated initial states with L in32/48/64 and |x|<=L; valid action strings. Use new parent-disjoint train/dev trajectories and independent seeds, fixed original one-step serialization, and own-state rollouts. Call this **new trajectories over adapter-seen one-step transitions**. It measures composition, compounding error and preservation of inherited skill; it cannot establish unseen one-step state/action learning. Every legal/illegal key in this cohort was exposed to the inherited adapter.

**Limit-parameter OOD cohort:** choose limits outside32/48/64, with states still inside the selected limit; record all original adapter keys as seen/unseen by exact membership. Limits24 and40 already used by DSv3WM are limit-OOD relative to this adapter; limit32 is ID. Reusing those now-exposed DSv3WM dev parents is development recovery, not a new independent confirmation. New limit values establish adapter-label OOD only, not novel rule laws or absence from the parent model's history.

**State-range OOD cohort:** use |x|>64 with L large enough to admit the initial state, ensuring every scored one-step key is explicitly classified. A rollout may move from OOD into the seen support; report that transition and do not label an entire chain uniformly unseen solely from its start. Invalid outputs outside state/rule constraints stay wrong; do not repair them with exact dynamics during learned inference.

**Computed-legality stress:** preregister, before outputs, balanced legal/illegal odd-halving, boundary moves and parameter-dependent doubling cohorts, plus false-apply/false-copy rates. Sample new cases from law-defined strata and fresh seeds; avoid choosing cases from historical error lists. New limits can reveal legal doubles at magnitudes that were illegal under every old training limit. This is a parameter-rule extrapolation test, not proof of arbitrary new world laws.

**Presentation stress:** unchanged original chat serialization is the recovery control. Reordered fields or changed rule prose are separately labelled representation stresses. A future multi-action/direct-horizon prompt changes the task interface; qualify that interface before interpreting a failure as a dynamics-objective failure. Primary four-step own-state rollout can reuse the verified original one-step interface without teacher intermediate states.

**Unknown/malformed actions:** the original adapter has no training support for teleport-like strings, missing fields or the new wrapper's illegal-no-op conventions. Keep these in an explicit extension cohort. They must not dominate an inherited-skill reproduction result.

**Planning goals:** F training is goal-blind and its state/action support is exhaustive in ID. New goal-held-out families can evaluate planning generalization, but never make the forward transitions unseen. With an inherited F adapter, claims should say **warm-start adaptation/composition relative to its documented complete ID support**. Fresh DSv3WM training/dev disjointness cannot erase prior exposure inherited from this adapter or from the frozen parent.

Historical96/128 range probes and earlier calibration/selection settings are already described in BWM records. Do not reopen their rows/plans or tune recovery on them. A newly generated cohort may be adapter-OOD even if its parameter family was studied historically; say so explicitly. Independent confirmation still needs a new, complete preregistration and held-out parent/goal design after development recovery, with untouched label access boundaries.

## Access and limitations

F_TABLE.json, historical dataset/bank files and test-plan payloads were not opened, and no historical model result was recomputed. Only new training-support labels were reconstructed from allowed exact-law code. An initial full metrics.json read inadvertently returned its bundled old error-sample section; that avoidable access was immediately reported to the lead, those samples were not used for cohort selection or reconstruction, and subsequent reads selected training metadata only. No sealed confirmation labels were opened. No weight, adapter, BWM/RCWM record, Git state or native runtime was changed. This new report is the sole written artifact of this audit.
