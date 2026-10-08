# Block-localised Titan delta: off-task drift vs the transition skill (exploratory)

Run `C:\Users\clclc\BusinessProject\WorldModel\Rineya_Consequence_World_Model_Bundle_2026-10-05\experiments\RCWM-T1\runs\bank_pilot\localized` (DONE 2026-10-08 08:53:41); analysis `scripts/localized_analysis.py`; 248 off-task prompts, 640 pilot TEST chains (chain_one); 10000 bootstrap resamples. Exploratory: pilot bank, TEST seen in development, one greedy decode per prompt.

Arms (all built in place on one Supa load; build gates G1-G5 passed, see run_localized.py):

| arm | Titan blocks | differs from Supa | fingerprint | run_patch text reproduction (first 4/family) | states == Supa below first Titan block |
|---|---|---|---|---|---|
| supa | – | none | a9952c55f5a1 | exact (12/12; batch 48/48) | yes |
| loc8_15 | [8, 15] | block8..block15 | 749f209a1504 | exact (12/12; batch 48/48) | yes |
| sparse | [8, 27] | block8..block27 | 818be9c6b6a3 | exact (12/12; batch 48/48) | yes |
| up16_27 | [16, 27] | block16..block27 | d6e183df8a2e | exact (12/12; batch 48/48) | yes |

loc8_15 states equal sparse's exactly at layers 0, 4, 8, 12, 16: yes.

## Pre-stated reading (DECISIONS D-23)

Rule (verbatim from D-23): "localisation reduces interference" iff the KL ratio loc8_15/sparse has an upper 95% CI < 0.8 AND regret(loc8_15) − regret(sparse) ≤ +0.05 in every family. "No reduction" if the ratio's lower CI > 0.9. Otherwise inconclusive.

Operationalisation, fixed in docs/localized/DESIGN.md before the run: KL ratio = mean KL(loc8_15‖supa) / mean KL(sparse‖supa) over the off-task prompts with the paired percentile bootstrap CI (primary direction; the same rule on KL(supa‖arm) is shown as secondary); regret = chain_one k=4 normalised regret, difference = paired per-parent mean loc8_15 − sparse; families L, P, R.

**Verdict (primary, KL(arm‖supa)): localisation reduces interference.** Secondary, KL(supa‖arm): localisation reduces interference (agrees).

| leg | value | 95% CI | criterion met |
|---|---|---|---|
| KL ratio loc8_15/sparse (mean KL(arm\|\|supa), nats; primary) | 0.680 | [0.617, 0.735] | upper CI < 0.8: yes; lower CI > 0.9: no |
| KL ratio loc8_15/sparse (mean KL(supa\|\|arm), nats; secondary) | 0.650 | [0.594, 0.716] | upper CI < 0.8: yes; lower CI > 0.9: no |
| L: k=4 regret loc8_15 − sparse (n 26 parents) | 0.017 | [-0.127, 0.158] | ≤ +0.05: yes |
| P: k=4 regret loc8_15 − sparse (n 26 parents) | 0.047 | [-0.031, 0.124] | ≤ +0.05: yes |
| R: k=4 regret loc8_15 − sparse (n 26 parents) | -0.005 | [-0.041, 0.029] | ≤ +0.05: yes |

Regret leg in context (the +0.05 leg is informative only where the copy chain is well above sparse's regret; a family with headroom < 0.10 is marked low-headroom and its skill evidence is the one-step FR on changed first steps):

| family | regret sparse | regret loc8_15 | copy chain | headroom (copy − sparse) | FR loc8_15 (changed first steps) [CI] | one-step acc supa / sparse / loc8_15 (changed) |
|---|---|---|---|---|---|---|
| L | 0.507 | 0.525 | 1.011 | 0.503 | 0.85 [0.68, 1.03] | 0.034 / 0.433 / 0.371 |
| P | 0.879 | 0.925 | 0.955 | 0.077 **low** | 1.00 [0.76, 1.30] | 0.000 / 0.529 / 0.529 |
| R | 0.386 | 0.380 | 1.007 | 0.622 | 0.97 [0.91, 1.03] | 0.080 / 0.595 / 0.580 |

Caveats (reported, they do not change the pre-stated verdict):

- mean KL(supa||arm), nats: CAUTION: concentrated: the top 10% of prompts carry 0.54 of sparse's mean KL
- P: low headroom (copy floor 0.955 − sparse regret 0.879 = 0.077 < 0.10): the +0.05 regret leg can barely fail; read its one-step FR

## Key ratios: drift of the localised arms relative to the full delta

Paired bootstrap over prompts; a ratio < 1 whose CI excludes 1 = less drift than the full delta. The two ratios need not add to 1 (KL and disagreement are not additive over blocks). KL in nats.

Denominator (sparse vs Supa, 248 prompts): top-1 token flipped on 52 prompts; greedy reply differs on 123 (0.496); top 10% of prompts carry 0.49 of mean KL(sparse‖supa) and 0.54 of mean KL(supa‖sparse).

| drift measure | sparse (full delta) | loc8_15 / sparse [95% CI] | up16_27 / sparse [95% CI] | loc8_15 − sparse | up16_27 − sparse | denominator flag |
|---|---|---|---|---|---|---|
| mean KL(arm\|\|supa), nats | 0.1068 | 0.680 [0.617, 0.735] | 0.170 [0.123, 0.219] | -0.0341 [-0.0436, -0.0259] | -0.0887 [-0.1100, -0.0713] | ok |
| mean KL(supa\|\|arm), nats | 0.1285 | 0.650 [0.594, 0.716] | 0.142 [0.104, 0.189] | -0.0450 [-0.0644, -0.0305] | -0.1103 [-0.1456, -0.0835] | **CAUTION**: concentrated: the top 10% of prompts carry 0.54 of sparse's mean KL |
| greedy 64-token exact disagreement with Supa | 0.4960 | 0.854 [0.772, 0.935] | 0.350 [0.265, 0.441] | -0.0726 [-0.1169, -0.0323] | -0.3226 [-0.3871, -0.2621] | ok |
| greedy 64-token prefix divergence from Supa (1 - common-prefix chars / longer reply) | 0.3901 | 0.861 [0.778, 0.944] | 0.365 [0.274, 0.462] | -0.0541 [-0.0886, -0.0212] | -0.2477 [-0.3002, -0.1960] | ok |
| top-1 first-token disagreement | 0.2097 | 0.788 [0.651, 0.929] | 0.327 [0.196, 0.476] | -0.0444 [-0.0766, -0.0121] | -0.1411 [-0.1895, -0.0968] | ok |

## Skill: k=4 chain_one normalised regret (lower is better; decisions.g_only, pilot TEST goals)

| family | parents | supa | loc8_15 | sparse | up16_27 | copy chain | loc8_15 − sparse [CI] | up16_27 − sparse [CI] |
|---|---|---|---|---|---|---|---|---|
| L | 26 | 0.821 [0.757, 0.883] | 0.525 [0.426, 0.631] | 0.507 [0.392, 0.630] | 0.824 [0.747, 0.913] | 1.011 | 0.017 [-0.127, 0.158] | 0.317 [0.176, 0.455] |
| P | 28 | 0.965 [0.874, 1.027] | 0.925 [0.819, 1.016] | 0.879 [0.761, 0.992] | 0.928 [0.827, 1.010] | 0.955 | 0.047 [-0.031, 0.124] | 0.049 [-0.075, 0.166] |
| R | 26 | 0.898 [0.847, 0.948] | 0.380 [0.348, 0.413] | 0.386 [0.349, 0.424] | 0.886 [0.828, 0.947] | 1.007 | -0.005 [-0.041, 0.029] | 0.501 [0.430, 0.572] |

## Skill: one-step accuracy from the first chain step (= run_patch's one-frame items)

| family | changed steps | supa acc (changed) | loc8_15 acc (changed) | sparse acc (changed) | up16_27 acc (changed) | FR loc8_15 | FR up16_27 |
|---|---|---|---|---|---|---|---|
| L | 178 | 0.034 [0.006, 0.075] | 0.371 [0.303, 0.437] | 0.433 [0.360, 0.511] | 0.067 [0.037, 0.102] | 0.85 [0.68, 1.03] | 0.08 [-0.05, 0.19] |
| P | 87 | 0.000 [0.000, 0.000] | 0.529 [0.358, 0.712] | 0.529 [0.365, 0.701] | 0.000 [0.000, 0.000] | 1.00 [0.76, 1.30] | 0.00 [0.00, 0.00] |
| R | 200 | 0.080 [0.050, 0.112] | 0.580 [0.513, 0.647] | 0.595 [0.532, 0.654] | 0.080 [0.045, 0.119] | 0.97 [0.91, 1.03] | 0.00 [-0.07, 0.07] |

All first steps and chain exact-state accuracy:

| family | arm | first step acc (all) | − sparse | k=1 | k=2 | k=4 | chains complete | steps |
|---|---|---|---|---|---|---|---|---|
| L | supa | 0.072 [0.029, 0.120] | -0.327 [-0.404, -0.255] | 0.072 | 0.000 | 0.000 | 0.808 | 775 |
| L | loc8_15 | 0.351 [0.288, 0.409] | -0.048 [-0.115, 0.014] | 0.351 | 0.139 | 0.014 | 0.995 | 829 |
| L | sparse | 0.399 [0.341, 0.462] | – | 0.399 | 0.202 | 0.024 | 0.995 | 829 |
| L | up16_27 | 0.135 [0.087, 0.188] | -0.264 [-0.337, -0.202] | 0.135 | 0.010 | 0.000 | 1.000 | 832 |
| P | supa | 0.429 [0.330, 0.527] | 0.219 [0.071, 0.366] | 0.429 | 0.241 | 0.107 | 0.987 | 887 |
| P | loc8_15 | 0.210 [0.134, 0.290] | 0.000 [-0.058, 0.049] | 0.210 | 0.040 | 0.004 | 0.509 | 735 |
| P | sparse | 0.210 [0.134, 0.290] | – | 0.210 | 0.049 | 0.000 | 0.594 | 784 |
| P | up16_27 | 0.384 [0.259, 0.513] | 0.174 [0.000, 0.353] | 0.384 | 0.254 | 0.161 | 1.000 | 896 |
| R | supa | 0.101 [0.067, 0.135] | -0.505 [-0.572, -0.438] | 0.101 | 0.014 | 0.000 | 0.495 | 650 |
| R | loc8_15 | 0.591 [0.524, 0.659] | -0.014 [-0.048, 0.014] | 0.591 | 0.341 | 0.082 | 0.957 | 820 |
| R | sparse | 0.606 [0.543, 0.663] | – | 0.606 | 0.361 | 0.087 | 1.000 | 832 |
| R | up16_27 | 0.115 [0.072, 0.159] | -0.490 [-0.558, -0.423] | 0.115 | 0.038 | 0.000 | 1.000 | 832 |

## Off-task drift per arm (vs Supa)

| arm | KL(arm‖supa) mean, nats | median | KL(supa‖arm) mean, nats | median | top-1 agree | greedy exact agree | greedy prefix divergence | JSON bleed (non-numeric) | JSON-state (all) |
|---|---|---|---|---|---|---|---|---|---|
| supa | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.000 | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.005 (n 204) | 0.000 |
| loc8_15 | 0.0727 [0.0559, 0.0943] | 0.0398 [0.0343, 0.0437] | 0.0835 [0.0642, 0.1071] | 0.0406 [0.0346, 0.0454] | 0.835 | 0.577 [0.516, 0.637] | 0.336 [0.284, 0.388] | 0.010 (n 204) | 0.000 |
| sparse | 0.1068 [0.0853, 0.1336] | 0.0592 [0.0535, 0.0663] | 0.1285 [0.0980, 0.1687] | 0.0583 [0.0529, 0.0692] | 0.790 | 0.504 [0.440, 0.569] | 0.390 [0.337, 0.444] | 0.010 (n 204) | 0.000 |
| up16_27 | 0.0181 [0.0114, 0.0263] | 0.0050 [0.0043, 0.0057] | 0.0182 [0.0117, 0.0259] | 0.0049 [0.0044, 0.0057] | 0.931 | 0.827 [0.778, 0.871] | 0.142 [0.104, 0.185] | 0.005 (n 204) | 0.000 |

KL of each arm to the full delta, KL(arm‖sparse) mean, nats: supa 0.1285, loc8_15 0.0129, up16_27 0.0757.
Noise floor: KL(stored Titan-Sparse ‖ this run's sparse arm) mean 0.000000, top-1 agreement 1.000.

### Answer-slot state shift per layer (RMS over dims of (h_arm − h_supa)/sd_supa; mean over prompts)

| layer | supa RMS z | loc8_15 RMS z | sparse RMS z | up16_27 RMS z | supa rel L2 | loc8_15 rel L2 | sparse rel L2 | up16_27 rel L2 | loc8_15/sparse | up16_27/sparse |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | raw only | raw only | raw only | raw only | 0.000 | 0.000 | 0.000 | 0.000 | – | – |
| 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – |
| 8 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | – | – |
| 12 | 0.000 | 0.367 | 0.367 | 0.000 | 0.000 | 0.112 | 0.112 | 0.000 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| 16 | 0.000 | 0.389 | 0.389 | 0.000 | 0.000 | 0.182 | 0.182 | 0.000 | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] |
| 20 | 0.000 | 0.248 | 0.303 | 0.089 | 0.000 | 0.158 | 0.193 | 0.057 | 0.82 [0.81, 0.83] | 0.29 [0.28, 0.30] |
| 24 | 0.000 | 0.215 | 0.276 | 0.102 | 0.000 | 0.155 | 0.198 | 0.072 | 0.78 [0.76, 0.79] | 0.37 [0.35, 0.39] |
| 28 | 0.000 | 0.195 | 0.256 | 0.107 | 0.000 | 0.130 | 0.171 | 0.072 | 0.76 [0.75, 0.78] | 0.42 [0.40, 0.44] |

### Greedy exact agreement with Supa by reply class

| reply class | n | loc8_15 | sparse | up16_27 |
|---|---|---|---|---|
| comparative_recommendation | 4 | 0.00 | 0.00 | 0.50 |
| comparison_ranking | 4 | 0.50 | 0.00 | 0.75 |
| conversation_closure | 4 | 0.50 | 0.00 | 0.75 |
| deescalation | 4 | 0.00 | 0.00 | 0.25 |
| natural_follow_up | 4 | 0.25 | 0.00 | 0.75 |
| plan_revision | 4 | 0.00 | 0.00 | 0.50 |
| tactful_disagreement | 4 | 0.25 | 0.00 | 1.00 |
| uncertainty | 4 | 0.00 | 0.00 | 0.75 |
| arithmetic_chain | 4 | 0.75 | 0.25 | 0.50 |
| arithmetic_short_answer | 4 | 0.50 | 0.25 | 0.50 |
| comparison_recommendation | 4 | 0.00 | 0.25 | 1.00 |
| complex_escalation | 4 | 0.50 | 0.25 | 1.00 |
| confirmation | 4 | 0.25 | 0.25 | 1.00 |
| contradiction_detection | 4 | 0.25 | 0.25 | 0.50 |
| decision_explanation | 4 | 0.25 | 0.25 | 1.00 |
| estimation_sanity | 4 | 0.50 | 0.25 | 0.75 |
| evidence_grounded_dialogue | 4 | 0.25 | 0.25 | 0.50 |
| informal_texting | 4 | 0.25 | 0.25 | 0.75 |
| negotiation_compromise | 4 | 0.50 | 0.25 | 0.50 |
| roleplay_short | 4 | 0.25 | 0.25 | 0.50 |
| spatial_relations | 4 | 0.50 | 0.25 | 0.75 |
| unit_conversion | 4 | 0.50 | 0.25 | 0.50 |
| acknowledgment_update | 4 | 0.50 | 0.50 | 1.00 |
| apology_repair | 4 | 0.50 | 0.50 | 1.00 |
| boundary_setting | 4 | 0.50 | 0.50 | 0.75 |
| celebration | 4 | 0.50 | 0.50 | 0.50 |
| clarification | 4 | 0.50 | 0.50 | 0.75 |
| counting_sets | 4 | 0.75 | 0.50 | 0.75 |
| empathy | 4 | 0.50 | 0.50 | 1.00 |
| factual_compression | 4 | 0.50 | 0.50 | 0.75 |
| insufficient_information | 4 | 0.75 | 0.50 | 0.75 |
| knowledge_boundary | 4 | 0.50 | 0.50 | 1.00 |
| physical_commonsense | 4 | 0.50 | 0.50 | 1.00 |
| practical_next_step | 4 | 0.75 | 0.50 | 1.00 |
| practical_one_step | 4 | 0.75 | 0.50 | 1.00 |
| ratio_proportion | 4 | 1.00 | 0.50 | 1.00 |
| safety_boundary | 4 | 0.75 | 0.50 | 1.00 |
| time_duration | 4 | 0.50 | 0.50 | 0.75 |
| troubleshooting_diagnosis | 4 | 0.50 | 0.50 | 1.00 |
| acknowledgment | 4 | 0.75 | 0.75 | 1.00 |
| causal_reasoning | 4 | 0.75 | 0.75 | 1.00 |
| clarification_request | 4 | 0.75 | 0.75 | 1.00 |
| concise_explanation | 4 | 0.75 | 0.75 | 0.75 |
| conditional_rules | 4 | 1.00 | 0.75 | 0.75 |
| conflict_deescalation | 4 | 0.75 | 0.75 | 0.75 |
| empathy_validation | 4 | 0.75 | 0.75 | 1.00 |
| gentle_disagreement | 4 | 0.75 | 0.75 | 1.00 |
| logical_implication | 4 | 1.00 | 0.75 | 0.75 |
| multi_turn_context_update | 4 | 0.75 | 0.75 | 1.00 |
| output_only | 4 | 0.75 | 0.75 | 1.00 |
| percentage_reasoning | 4 | 0.50 | 0.75 | 0.75 |
| preference_opinion | 4 | 0.50 | 0.75 | 1.00 |
| pronoun_resolution | 4 | 0.75 | 0.75 | 0.75 |
| roleplay_reasoning | 4 | 0.75 | 0.75 | 0.75 |
| communication_rewrite | 4 | 1.00 | 1.00 | 1.00 |
| constraint_selection | 4 | 1.00 | 1.00 | 1.00 |
| context_recall | 4 | 1.00 | 1.00 | 1.00 |
| context_update | 4 | 0.75 | 1.00 | 1.00 |
| direct_preference | 4 | 1.00 | 1.00 | 1.00 |
| evidence_inference | 4 | 0.75 | 1.00 | 1.00 |
| schedule_ordering | 4 | 1.00 | 1.00 | 1.00 |
| sequence_pattern | 4 | 1.00 | 1.00 | 1.00 |

## Consistency checks (descriptive)

- supa: first-step text == run_patch `gen_base_supa.json` on 1.000 of 640 items; identical to the stored pilot chains: 0.317 of 640 chains, 0.947 of step texts
- loc8_15: first-step text == run_patch `gen_HlowT_16.json` on 1.000 of 640 items
- sparse: first-step text == run_patch `gen_base_sparse.json` on 1.000 of 640 items; identical to the stored pilot chains: 1.000 of 640 chains, 1.000 of step texts
- up16_27: first-step text == run_patch `gen_HupT_16.json` on 1.000 of 640 items

The stored pilot chains predate the 160-token cap and the D-14 generation batch budget (Supa's include truncated texts), so identical_chains < 1 is expected there, mostly for supa. It is not a splice check: the build gates G1-G5 and the run_patch text reproduction are.

## Reading guide and limits

- Pre-stated reading (DECISIONS D-23, verbatim): "localisation reduces interference" iff the KL ratio loc8_15/sparse has an upper 95% CI < 0.8 AND regret(loc8_15) − regret(sparse) ≤ +0.05 in every family. "No reduction" if the ratio's lower CI > 0.9. Otherwise inconclusive. The KL ratio is mean KL(arm‖supa) (primary; fixed in docs/localized/DESIGN.md before the run), the regret difference the paired per-parent mean of chain_one k=4 normalised regret. The verdict above applies this rule mechanically; the denominator flags and low-headroom marks are caveats, not part of the rule.
- up16_27 is descriptive: whether the upper delta (blocks 16-27, no measurable skill in R12) carries drift on its own. Both localised arms are measured against Supa, so loc8_15's remaining drift is the price of the skill-carrying blocks.
- Off-task drift here is first-token and 64-token greedy behaviour on 248 neutral prompts plus answer-slot states; it is not a retention benchmark. One greedy decode per prompt; replies compared by exact text.
- Skill is chain_one on the pilot TEST plans (one-step frame, self-rollout) and its first step; P's primary chain frame in T2 was chain_m1, which this run does not generate.
- Pilot TEST was seen in development; exploratory: the D-23 thresholds were stated in advance but are not a preregistration. Bootstrap CIs resample prompts (off-task) or parents (skill) and ignore the selection of the 8/16 cut, which R12 fixed beforehand.
