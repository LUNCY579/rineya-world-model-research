# Blocksworld-hard: BWM vs published per-instance results (POST HOC, DIFFERENT REGIME)

**POST HOC, DIFFERENT REGIME.** POST HOC: not pre-registered; written after every BWM hard result was known. Different regime: the published systems are general zero-shot reasoning models given PDDL text, one answer per instance, no domain training, no learned simulator, no search; the BWM systems are a 0.6B core with LoRA adapters trained on generated Blocksworld problems, planning without execution feedback. Never to be worded 'beats' / 'outperforms o1/R1 at planning'.

- Decision: D-75.3 (user decision 2026-10-11); closes P-14 correction (d). Hard-set exposure: none added: stored outputs only, no model run on the hard set.
- Published side: LLMs-Planning results_backprompting/blocksworld_hard, task zero_shot_pddl with the VAL backprompting loop; ONLY step 0 (first answer, before any verifier feedback) is used. Prompt: PDDL domain + problem, 'Provide your answer as a sequence of actions in PDDL format', no examples; o1 and R1 called without a temperature argument (API defaults).
- BWM side: B2 S-beam: frozen B2: learned F + beam search over Pa_v2 proposals (P-08/P-10); H2 Pa_v3 S-beam: frozen B2 search with the self-distilled proposer Pa_v3 (P-14); H1 Pa_v3 S-direct: Pa_v3 proposer alone, greedy, M 256, no F (P-13); B2 S-direct: Pa_v2 proposer alone, greedy, M 256, no F (P-08/P-10).
- Pairing: the step-0 prompt is identical across the three published files on 110/110 instances and contains the local problem file's text on 110/110; the local instances are byte-identical to upstream's on 110/110.

## Success on the same 110 instances (local VAL)

| System | Regime | Solved | Rate | Clopper-Pearson 95% |
|---|---|---|---|---|
| B2 S-beam | BWM, frozen | 73/110 | 0.664 | 0.567-0.751 |
| H2 Pa_v3 S-beam | BWM, frozen | 88/110 | 0.800 | 0.713-0.870 |
| H1 Pa_v3 S-direct | BWM, frozen | 13/110 | 0.118 | 0.065-0.194 |
| B2 S-direct | BWM, frozen | 5/110 | 0.045 | 0.015-0.103 |
| DeepSeek-R1 step 0 (pb_pddl) | published, zero-shot | 59/110 | 0.536 | 0.439-0.632 |
| DeepSeek-R1 step 0 (bwm_lenient) | published, zero-shot | 59/110 | 0.536 | 0.439-0.632 |
| o1-preview step 0 (pb_pddl) | published, zero-shot | 26/110 | 0.236 | 0.161-0.327 |
| o1-preview step 0 (bwm_lenient) | published, zero-shot | 26/110 | 0.236 | 0.161-0.327 |
| o1-mini step 0 (pb_pddl) | published, zero-shot | 1/110 | 0.009 | 0.000-0.050 |
| o1-mini step 0 (bwm_lenient) | published, zero-shot | 1/110 | 0.009 | 0.000-0.050 |

Upstream's own recorded step-0 verdicts: DeepSeek-R1 59/110; o1-preview 26/110; o1-mini 1/110.
Verifier-in-loop final results (later backprompting steps; NOT compared, a different regime again): DeepSeek-R1 59/110 (steps recorded per instance: {1: 110}); o1-preview 108/110 (steps recorded per instance: {2: 33, 1: 26, 3: 18, 4: 9, 7: 5, 6: 4, 5: 13, 9: 2}); o1-mini 11/110 (steps recorded per instance: {4: 101, 3: 5, 2: 3, 1: 1}). The R1 file records only step 0 for every instance, so its 'final' equals its step 0; o1-mini's 11/110 is the 10% LRM-Modulo figure of P-14 correction (a), its direct step 0 is 1/110.
Step-0 finish reasons: DeepSeek-R1 {'stop': 110}; o1-preview {'stop': 110}; o1-mini {'stop': 110} (no truncated published answer).

## Reading (POST HOC, DIFFERENT REGIME)

- H2 Pa_v3 S-beam vs DeepSeek-R1 step 0: 88 vs 59/110, paired +36/-7, exact p 9e-06, Holm p 5.4e-05.
- B2 S-beam vs DeepSeek-R1 step 0: 73 vs 59/110, paired +30/-16, exact p 0.054, Holm p 0.108 (NOT significant after Holm); 'B2 above R1' is not established (as P-08's unpaired test).
- Both S-beam systems vs o1-preview and o1-mini step 0: Holm p <= 4.5e-10.
- The proposers alone are below R1: H1 13 vs 59/110, paired +2/-48, exact p 2.3e-12, Holm p 1.8e-11; B2 S-direct 5 vs 59/110, paired +0/-54, exact p 1.1e-16, Holm p 1e-15.
- vs o1-preview: H1 13 vs 26/110, paired +9/-22, exact p 0.029, Holm p 0.088 (NOT significant after Holm); B2 S-direct 5 vs 26/110, paired +4/-25, exact p 0.0001, Holm p 0.00052. P-14 correction (b)'s unpaired 'H1 below o1-preview (Fisher p = 0.033)' becomes paired p 0.029 unadjusted, 0.088 after Holm (m = 12).
- R1 solves 7 instances H2 does not (instance-21, instance-22, instance-39, instance-52, instance-55, instance-58, instance-96); all have >= 10 blocks; H2's outcome there: {'budget': 5, 'dead_end': 2}, none verified (0 verified); R1's plan is optimal-length on 4 of them.
- Extraction does not move any verdict: upstream's own PDDL extractor and the lenient bwm extractor give the same VAL verdict on every published answer (330/330), and the local VAL reproduces upstream's recorded step-0 verdict on 330/330.

## Paired exact McNemar, Holm over 12 contrasts (POST HOC, DIFFERENT REGIME)

Primary extractor pb_pddl (upstream's own). bwm_only / published_only are the discordant pairs.

| Contrast | BWM | Published | Both | BWM only | Published only | Neither | Diff (pp) | p (exact) | Holm p | Lenient: BWM only / Pub only, Holm p |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 S-beam vs DeepSeek-R1 | 73 | 59 | 43 | 30 | 16 | 21 | +12.7 | 0.054 | 0.108 | 30 / 16, 0.108 |
| B2 S-beam vs o1-preview | 73 | 26 | 21 | 52 | 5 | 32 | +42.7 | 6.4e-11 | 4.5e-10 | 52 / 5, 4.5e-10 |
| B2 S-beam vs o1-mini | 73 | 1 | 1 | 72 | 0 | 37 | +65.5 | 4.2e-22 | 4.7e-21 | 72 / 0, 4.7e-21 |
| H2 Pa_v3 S-beam vs DeepSeek-R1 | 88 | 59 | 52 | 36 | 7 | 15 | +26.4 | 9e-06 | 5.4e-05 | 36 / 7, 5.4e-05 |
| H2 Pa_v3 S-beam vs o1-preview | 88 | 26 | 24 | 64 | 2 | 20 | +56.4 | 6e-17 | 6e-16 | 64 / 2, 6e-16 |
| H2 Pa_v3 S-beam vs o1-mini | 88 | 1 | 1 | 87 | 0 | 22 | +79.1 | 1.3e-26 | 1.6e-25 | 87 / 0, 1.6e-25 |
| H1 Pa_v3 S-direct vs DeepSeek-R1 | 13 | 59 | 11 | 2 | 48 | 49 | -41.8 | 2.3e-12 | 1.8e-11 | 2 / 48, 1.8e-11 |
| H1 Pa_v3 S-direct vs o1-preview | 13 | 26 | 4 | 9 | 22 | 75 | -11.8 | 0.029 | 0.088 | 9 / 22, 0.088 |
| H1 Pa_v3 S-direct vs o1-mini | 13 | 1 | 0 | 13 | 1 | 96 | +10.9 | 0.002 | 0.007 | 13 / 1, 0.007 |
| B2 S-direct vs DeepSeek-R1 | 5 | 59 | 5 | 0 | 54 | 51 | -49.1 | 1.1e-16 | 1e-15 | 0 / 54, 1e-15 |
| B2 S-direct vs o1-preview | 5 | 26 | 1 | 4 | 25 | 80 | -19.1 | 0.0001 | 0.00052 | 4 / 25, 0.00052 |
| B2 S-direct vs o1-mini | 5 | 1 | 0 | 5 | 1 | 104 | +3.6 | 0.219 | 0.219 | 5 / 1, 0.219 |

Secondary (panel item 4 family: R1 and o1-preview only, Holm m = 8, pb_pddl): B2 S-beam vs DeepSeek-R1 Holm 0.059; B2 S-beam vs o1-preview Holm 3.2e-10; H2 Pa_v3 S-beam vs DeepSeek-R1 Holm 3.6e-05; H2 Pa_v3 S-beam vs o1-preview Holm 4.8e-16; H1 Pa_v3 S-direct vs DeepSeek-R1 Holm 1.4e-11; H1 Pa_v3 S-direct vs o1-preview Holm 0.059; B2 S-direct vs DeepSeek-R1 Holm 7.8e-16; B2 S-direct vs o1-preview Holm 0.00031.

## Extractors and VAL

- **pb_pddl**: PRIMARY. Port of upstream llm_utils.save_gpt3_response (fc638a1a; blob 62f36232), the extractor back_prompting_parallel.get_repeated_verification_pddl uses for PDDL tasks; reproduces every published step-0 llm_plan; scored by bwmn.blocks.val_check and verbatim.
- **bwm_lenient**: SENSITIVITY. bwmn/planbench_extract.py helpers + PDDL/call-notation action regex; list markers stripped; all actions per line; last block of action lines; no repair.
- **nl_probe**: bwmn/planbench_extract.py extract_plan and text_to_plan_blocksworld (colour-based natural-language parsers) are not applicable to PDDL-letter answers; probe counts below.

| Check | DeepSeek-R1 | o1-preview | o1-mini |
|---|---|---|---|
| bwm_answer_source_full | 110 | 110 | 110 |
| bwm_multi_block_answers | 1 | 0 | 0 |
| extractors_same_plan | 109 | 109 | 109 |
| extractors_same_verdict | 110 | 110 | 110 |
| local_val_pb_pddl_agrees_with_published_verdict | 110 | 110 | 110 |
| nl_extract_plan_nonempty | 0 | 0 | 0 |
| nl_text_to_plan_blocksworld_nonempty | 0 | 0 | 0 |
| pb_pddl_reproduces_llm_plan | 110 | 110 | 110 |
| pb_pddl_tuple_vs_verbatim_agree | 110 | 110 | 110 |
| published_verdict_from_single_step_act_correct | 0 | 26 | 0 |
| published_verdict_from_sound_verifier_json | 110 | 84 | 110 |

VAL calls: {'verbatim': 325, 'val_check': 731} (cached per instance and plan). BWM plans re-scored with the same val_check: B2 S-beam 110/110 agree with the recorded verdict; H2 Pa_v3 S-beam 110/110 agree with the recorded verdict; H1 Pa_v3 S-direct 110/110 agree with the recorded verdict; B2 S-direct 110/110 agree with the recorded verdict.

Earlier audit (simulator over upstream llm_plan, H2 leak/claims audit): DeepSeek-R1 59/110, per-instance agreement with local VAL 110/110; o1-preview 26/110, per-instance agreement with local VAL 110/110; o1-mini 1/110, per-instance agreement with local VAL 110/110.

## Instances DeepSeek-R1 solves (step 0) but H2 does not (7)

| Instance | Blocks | Optimal len | R1 plan len | H2 fail | H2 verified | H2 rounds | B2 S-beam | R1 under bwm_lenient |
|---|---|---|---|---|---|---|---|---|
| instance-21 | 12 | 38 | 38 | budget | False | 7 | failed | valid |
| instance-22 | 15 | 40 | 46 | budget | False | 8 | failed | valid |
| instance-39 | 12 | 30 | 32 | budget | False | 8 | failed | valid |
| instance-52 | 15 | 34 | 34 | budget | False | 7 | failed | valid |
| instance-55 | 13 | 40 | 44 | budget | False | 7 | failed | valid |
| instance-58 | 10 | 24 | 24 | dead_end | False | 5 | solved | valid |
| instance-96 | 12 | 34 | 34 | dead_end | False | 8 | failed | valid |

R1 solves but B2 S-beam does not (16): instance-8, instance-21, instance-22, instance-23, instance-39, instance-40, instance-49, instance-52, instance-55, instance-62, instance-74, instance-85, instance-91, instance-96, instance-106, instance-110.

## By block count (descriptive, post hoc)

| Blocks | n | B2 S-beam | H2 Pa_v3 S-beam | H1 Pa_v3 S-direct | B2 S-direct | DeepSeek-R1 | o1-preview | o1-mini |
|---|---|---|---|---|---|---|---|---|
| 6 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| 7 | 1 | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| 8 | 7 | 6 | 7 | 1 | 2 | 6 | 4 | 0 |
| 9 | 14 | 14 | 14 | 7 | 3 | 11 | 7 | 1 |
| 10 | 13 | 10 | 11 | 3 | 0 | 8 | 7 | 0 |
| 11 | 12 | 9 | 12 | 1 | 0 | 7 | 4 | 0 |
| 12 | 13 | 8 | 9 | 0 | 0 | 9 | 2 | 0 |
| 13 | 18 | 11 | 13 | 0 | 0 | 7 | 1 | 0 |
| 14 | 21 | 11 | 16 | 0 | 0 | 5 | 1 | 0 |
| 15 | 10 | 2 | 4 | 0 | 0 | 4 | 0 | 0 |
- 6-10 blocks (n 36): H2 34, B2 S-beam 32, R1 27, o1-preview 18, o1-mini 1. H2 vs R1 +8/-1 (unadjusted p 0.039); B2 S-beam vs R1 +7/-2 (unadjusted p 0.180).
- 11-15 blocks (n 74): H2 54, B2 S-beam 41, R1 32, o1-preview 8, o1-mini 0. H2 vs R1 +28/-6 (unadjusted p 0.0002); B2 S-beam vs R1 +23/-14 (unadjusted p 0.188).

## Compute (report-only; tokens are not comparable units)

- DeepSeek-R1 step 0: prompt median 612.5 (mean 608.5); completion median 7814.0 (mean 7796.5); reasoning median 7645.0 (mean 7640.9); time_s median 181.8 (mean 182.9).
  - ESTIMATED FLOPs per instance: median 6.2e+14, mean 6.2e+14 (2 x 37e9 activated parameters x (prompt + completion tokens); dense matmuls only, no attention term, no MoE routing; ESTIMATED, the provider's real serving compute is unknown).
- o1-preview step 0: prompt median 670.5 (mean 663.6); completion median 7804.5 (mean 7732.1); reasoning median 7584.0 (mean 7439.1); time_s median 75.2 (mean 74.6).
- o1-mini step 0: prompt median 670.5 (mean 663.6); completion median 12395.5 (mean 12188.3); reasoning median 12256.0 (mean 12051.2); time_s median 77.3 (mean 79.2).
- B2 S-beam: proposer generated tokens median 7455.5 (mean 7991.9); F queries per instance 186.9.
- H2 Pa_v3 S-beam: proposer generated tokens median 7084.0 (mean 8206.1); F queries per instance 198.0.
- H1 Pa_v3 S-direct: proposer generated tokens median 174.5 (mean 181.3); F queries per instance 0.0.
- B2 S-direct: proposer generated tokens median 130.0 (mean 129.5); F queries per instance 0.0.
- Tokens are not comparable units across a 0.6B dense model and API models. R1 FLOPs are an ESTIMATE (dense-only, activated parameters). o1 FLOPs are undisclosed: tokens only. BWM FLOPs (proposer + F queries, with the attention term) are left to the item-3 ledger; no FLOPs ratio and no iso-FLOP claim is made here.

## Wording

- Allowed (proposed; the user decides): "POST HOC, different regime: on the same 110 PlanBench Blocksworld-hard instances, re-scored by the same local VAL, the frozen BWM S-beam with the self-distilled proposer (H2) solved 88 and DeepSeek-R1's zero-shot direct answer (published per-instance file, step 0) 59 (paired +36/-7, exact McNemar Holm p 5.4e-05, 12 contrasts); the original B2 S-beam solved 73 (+30/-16, Holm p 0.108, not significant). The regimes differ: R1 is a general model with no domain training, no learned simulator and no search; BWM is trained on generated Blocksworld problems." Always with the regime sentence and the extractor note.
- Not allowed: 'beats' / 'outperforms o1/R1 at planning', any iso-compute claim, or citing the verifier-loop numbers as zero-shot.

## Inputs

- DeepSeek-R1: `bench/planbench/results/blocksworld_hard/deepseek-r1-api/task_1_plan_generation_backprompting_zero_shot_pddl_sound_verifier_sound_critique_zero_shot_val_form_1_history_feedback_type_1_tv0.0_tg1.0.json` sha256 332b0f8ec03f337b4f320e4986b3d758c0fbb6efd223c7cfef05534e8202740e (git blob 22ca8ced06058f04532cfe4a74f57459e1014046; llm_planning_analysis/results_backprompting/blocksworld_hard/deepseek-r1-api/task_1_plan_generation_backprompting_zero_shot_pddl_sound_verifier_sound_critique_zero_shot_val_form_1_history_feedback_type_1_tv0.0_tg1.0.json @ fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284).
- o1-preview: `bench/planbench/results/blocksworld_hard/o1-preview_chat/task_1_plan_generation_backprompting_zero_shot_pddl_sound_verifier_sound_critique_zero_shot_val_form_1_history_feedback_type_1_tv0.0_tg1.0.json` sha256 b8fdc9ba2b7cbd3f1f26a3e4aabe2be5ac108a1499916ef91809a88b1a51478c (git blob 87ccabe011fabdcb752af3a1142f96b83b24cc79; llm_planning_analysis/results_backprompting/blocksworld_hard/o1-preview_chat/task_1_plan_generation_backprompting_zero_shot_pddl_sound_verifier_sound_critique_zero_shot_val_form_1_history_feedback_type_1_tv0.0_tg1.0.json @ fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284).
- o1-mini: `bench/planbench/results/blocksworld_hard/o1-mini_chat/task_1_plan_generation_backprompting_zero_shot_pddl_sound_verifier_sound_critique_zero_shot_val_form_1_history_feedback_type_1_tv0.0_tg1.0.json` sha256 928497901fc3e4be5b78fe26a5afdfa2939dec6c5b6e291c5cd67211fd71bc44 (git blob 7f2e7e2bf825e1dc4a1e43ab0dd4f6e25f6eb208; llm_planning_analysis/results_backprompting/blocksworld_hard/o1-mini_chat/task_1_plan_generation_backprompting_zero_shot_pddl_sound_verifier_sound_critique_zero_shot_val_form_1_history_feedback_type_1_tv0.0_tg1.0.json @ fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284).
- B2 S-beam: `runs/BW_TEST_B2/hard_beam/plans.jsonl` sha256 85812972a2e5e8bcedee245673dee98a0f0fdb93f02e2e852e44fa7c037fb3ee (P-08/P-10).
- H2 Pa_v3 S-beam: `runs/BW_B3_HARD/h2_beam/plans.jsonl` sha256 3cf3cd01b8a73f3e036fda126a5d6b562b38f63e458197a188639f2df6e2b91b (P-14).
- H1 Pa_v3 S-direct: `runs/BW_B3_HARD/h1_direct/plans.jsonl` sha256 fb5ab1d0e0a8808c4b882a6e440d4c888f17ce52c7738120e5142525c7f4fc42 (P-13).
- B2 S-direct: `runs/BW_TEST_B2/hard_direct/plans.jsonl` sha256 67b7bc33991e7bb571b128ef72681bc0a5b131a591fd4d0345c9e59dcc82c034 (P-08/P-10).
- Domain `bench/planbench/blocksworld_hard/generated_domain.pddl` sha256 7bc3adedadbc3165c0c4372633305dcfb3238c205a5af0c18b8bc0ec2399a468; VAL `bench/planbench/VAL/bin/validate/validate.exe` sha256 9a59116222dd2cc1b61f57f62f9c1152afc1c81df9f8d1d9e6c5a6bbe9876947.
- Code `scripts/hard_published_paired.py` sha256 4e1364d4d47e58d599fb449e183ed021558d3ad3e2456f622bf4ebc8d4f9af32.
- Code `bwmn/blocks.py` sha256 b1e28ed55e34d2621007f8b22a45213769da316f8957a628b279f872f9c22d83.
- Code `bwmn/planbench_extract.py` sha256 9d7bc546d5efba8eb5643b6fe4ecb7bd560597c249781ae996049ed5fab6867d.
- Code `bwmn/stats.py` sha256 ede4454e83f32b3dc6f638b44d16cd953332bc9e5f987183f2b8b3dc6c98bbdc.
