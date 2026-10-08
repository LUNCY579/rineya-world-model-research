# Titan-series report: Rineya_Published_Cores

## 1. What each Titan model is

The three Titans share the same base architecture and lineage:
- **Architecture:** Qwen3ForCausalLM (Qwen3-0.6B shape). 28 layers, hidden 1024, 16 attention heads with 8 KV heads, head_dim 128, MLP 3072, vocab 151936, tied embeddings, rope_theta 1e6, max_pos 40960. 596,049,920 params, 310 tensors.
- **Lineage:** Qwen3-0.6B → Nimbus 1.6 (α=0.40 soup toward Halo-Uncensored) → **Supa 1.7** = Nimbus + long-context "step200" LoRA at scale 0.8125, merged dense and saved BF16 (weight sha `962abc5b…d24b`) → Titans.
- **Shared sidecars:** tokenizer and generation_config are byte-identical across all three and identical to Supa 1.7.

| Core | Construction | Stored dtype / bytes | Weight sha |
|---|---|---|---|
| **Titan-Experimental** (RRP-015 control, seed 17) | Supa 1.7 fine-tuned on 384 transition rows. FP32 training, then the trained tensors were rounded to BF16 on export. | BF16, 1,192,135,064 | `53720167…31b7` |
| **Titan-Sparse-Experimental** (SPARSE-001, k=0.25) | `θ_supa + M_k ⊙ δ_T`, where `δ_T = θ_ctrl_fp32 − θ_supa` over the 220 trained tensors. M_k keeps the top 25% of \|δ\| **per tensor** (stable descending argsort; ties go to the lower flat index). No new training. | **FP32**, 2,384,234,992. config `torch_dtype: float32` (the only sidecar that differs) | `a28629c9…1284` |
| **Titan-Transition** (CONFIRM-002 TH25) | `Titan-Sparse + 0.25·P16(δ_H)`. Summed in float64, cast to FP32. P16 = per-tensor float64 SVD truncated to rank 16 (1-D tensors kept as is). δ_H = COMPOSE-001 boundary delta (trained sha `a1103060…`). No new training. | **FP32**, 2,384,235,008 | `a4d73353…9567` |

**Training recipe.** Applies to RRP-015 control and to the REP-001 seeds 23 and 29.
- **Trainable scope:** blocks 8–27 (the last 20 of 28). 220 tensors, 314,618,880 elements. Embeddings, blocks 0–7 and the final norm are frozen.
- **Objective:** `assistant_response_plus_eos`. Labels cover only the answer tokens plus `<|im_end|>`; the trailing `\n` is masked.
- **Prompt rendering:** Qwen3 chat template with an empty think block (`<|im_start|>assistant\n<think>\n\n</think>\n\n`), i.e. `enable_thinking=False`.
- **Optimizer:** MQ CpuAdamW, peak LR 1e-5, WSD 20/60/20 decaying to 0.1×peak, betas (0.9, 0.95), eps 1e-8, wd 0, clip 1.0. **All 28 steps were clipped** (grad norm 33.4 at step 1, 4.6–17.1 later).
- **Precision and length:** FP32, max_sequence 2048.
- **Data volume:** 384 rows, 1 exposure, 28 updates (about 14 rows per update). 54,172 processed tokens, of which only **5,254 are supervised**. Max rendered length 162 tokens.
- **Train CE:** 0.575 at step 1, about 0.12–0.17 late, 0.508 at the last step.
- **In-run eval:** microfit 24-row set went 0/24 → 11/24 at midpoint → 16/24 final.
- **Cost:** 266 s wall on the RTX 4070 Laptop. GPU peak 4.57 GiB, host RSS 3.99 GiB.

**Training data.** 384 Forge RL v2 rows (`process/train_transitions.jsonl.gz`, positives only, train split), 128 per family.
- Gold answers come from `oracle.replay_transition`.
- The 768 scored parents are excluded: the historical probe's 384 and SELECT-VIEWS-001's 384.
- Parents still available after exclusion: ledger 1744, prereq 899, register 1744.
- **Human data review: NOT PERFORMED.**

The three families, with real rows from SELECT-VIEWS-001 (system prompts are verbatim):
- **ledger_replay.**
  - System: "Apply one event to the supplied account balances. If status is not posted, leave all balances unchanged. add: add amount to account. scale: multiply account by amount. transfer: subtract amount from account and add it to other, simultaneously. Return only the complete resulting balances as a JSON object."
  - User: `{"before":{"A":-9,"B":16,"C":3,"D":-3},"event":{"account":"C","amount":-6,"kind":"transfer","other":"A","status":"posted","tick":1}}`
  - Answer: `{"A":-15,"B":16,"C":9,"D":-3}`
- **prerequisite_plan.**
  - System: "The state is an 8-bit flag mask. A supplied applicable action requires all pre_on bits set and all pre_off bits clear. Apply delete bits first, then add bits. Return only the resulting state as a JSON integer. Names and costs do not change the state."
  - User: `{"before":56,"event":{"add":2,"cost":3,"delete":0,"kind":"action","name":"A3","pre_off":0,"pre_on":32}}`
  - Answer: `58`
- **register_machine.**
  - System: "Apply one instruction to the supplied registers. Registers are integers modulo 97. swap exchanges register and other simultaneously. add adds value to register. mix adds (2 times other plus value) to register. conditional adds value if other is greater than threshold, otherwise subtracts value. Reduce updated numeric registers modulo 97; preserve other registers. Return only the complete resulting registers as a JSON object."
  - User: `{"before":{"X":71,"Y":94,"Z":45},"event":{"kind":"swap","other":"Z","register":"Y","threshold":83,"tick":9,"value":10}}`
  - Answer: `{"X":71,"Y":45,"Z":94}`

**δ_H (boundary donor).** 648 epistemic/safety-boundary rows from the v3 preservation slice, 54,217 tokens, 28 updates (budget-matched to δ_T). Seven classes: boundary_setting, clarification, clarification_request, insufficient_information, knowledge_boundary, safety_boundary, uncertainty. Example: "Please remove that from the reservation." → "What exactly should I remove from the reservation?"

**Norms.**
- ‖δ_T‖ = 0.8298 (relative update 0.0558%), ‖dSparse‖ = 0.6941, ‖P16(δ_T)‖ = 0.6740.
- ‖δ_H‖ = 1.0382, ‖P16(δ_H)‖ = 0.7099, so ‖0.25·P16(δ_H)‖ = 0.1775.

## 2. Measured endpoints

### Instruments
- **SELECT-VIEWS-001.** 384 rows (128 per family), selection-admissible, sha `7dd53f2a…`.
  - Floors were published before any model call: ledger 0.2344 (copy_before), prereq 0.0781 (constant_majority), register 0.0703 (copy_before).
  - A family is cleared only if its two-sided Clopper–Pearson 95% lower bound exceeds the floor.
- **Arithmetic panel.** 150 rows covering 3 of the 39 v3 classes (percentage_reasoning, ratio_proportion, time_duration). Supa scores 47 permissive / 41 strict. "Retained /47" counts how many of Supa's 47 correct rows are kept.
- **Long-Context Demo Suite.** 256 rows (192 answerable + 64 unanswerable). **SPENT, report-only.**
  - Run conditions: implementation digest `a329eb7bb0`, transformers 4.51.3, max_context 4096, max_new_tokens 24, TF32 off, greedy, FP32.
- **Gate v2** = arithmetic ≥ 62 AND retained ≥ 37/47 AND all 3 family floors cleared.

### Main endpoint table

| Model | Acq /384 | Floors | Arith perm/strict | Retained /47 | LC /256 (lost/gained, McNemar p) | UNKNOWN /64 | Gate v2 / legacy |
|---|---|---|---|---|---|---|---|
| Supa 1.7 | 14 | 0/3 | 47/41 | 47 | 161 | 13 | — |
| Opallo 1.0 | 8 | 0/3 | — | 41 | 135 | 0 | — |
| Titan-Exp (BF16) | 234 (L .4219 / P .7422 / R .6641) | 3/3 | 69/62 | 39 (8 lost, 30 new) | 162 (11/12, p=1.0) | 10 | PASS / FAIL |
| Same ckpt, unrounded FP32 (α=1) | 235 | 3/3 | 71/64 | 40 | **157** (p=0.5716) | 9 | PASS |
| **Titan-Sparse** | 232 (L .4141 / P .7500 / R .6484) | 3/3 | **72/63** | **41** (6 lost, 31 new) | **164** (7/10, p=0.6291) | 12 | PASS / FAIL |

Titan-Experimental's abstention detail: 0 false abstentions on answerable rows; the literal and substring counters agree.

**RAB-001 demo path (transformers 4.51.3).** Do not compare these with the vault's 40/64 for Supa.
- Titan-Exp: 42/64 showcase, 120/192 stability, 162/256 all.
- Titan-Sparse: 43/64, 121/192, 164/256.
- Supa on the same path: 41/64, 161/256.

### INT-001: FP32 interpolation of δ_T, θ(α) = θ_supa + α·δ_T

| α | Acq | Floors | Arith perm/strict | Retained | LC | UNK |
|---|---|---|---|---|---|---|
| 0.25 | 145 | 2/3 | 59/52 | 43 | 163 | 13 |
| 0.50 | 219 | 3/3 | 67/59 | 42 | 164 | 12 |
| 0.75 | 234 | 3/3 | 68/61 | 39 | 163 | 10 |

The α=0.50 arm (sha `ced9d0a8…`) won INT-001's registered rule but was not promoted.

### SPARSE-001: magnitude-pruned arms (Titan-Sparse is k=0.25)

| k | Kept elements | Norm retained | Acq | Floors | Arith | Retained | LC | UNK |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 0.10 | 0.6383 | 197 | 2/3 (ledger fails) | 61/54 | 42 | 163 | 12 |
| 0.25 | 0.25 | 0.8364 | 232 | 3/3 | 72/63 | 41 | 164 | 12 |
| 0.50 | 0.50 | 0.9595 | 235 | 3/3 | 68/61 | 39 | 159 | 9 |

### REP-001: control (ctrl) vs k25 within each seed

| Seed | Acq | Arith | Strict | Retained | LC | p vs Supa | UNK |
|---|---|---|---|---|---|---|---|
| s17 ctrl / k25 | 235 / 232 | 71 / 72 | 64 / 63 | 40 / 41 | 157 / 164 | — | 9 / 12 |
| s23 ctrl / k25 | 229 / 228 | 71 / 66 | 56 / 55 | 38 / 39 | 152 / 152 | — | 8 / 8 |
| s29 ctrl / k25 | 228 / 230 | 65 / 68 | 51 / 57 | 36 / 40 | 144 / 159 | 0.0115 / 0.8450 | 4 / 12 |

- Spread across the three controls: acquisition 7 rows, long context 13, strict arithmetic 13.
- No floor-clearing arm has reached retained ≥ 43 on any line (α, k, or seeds).

### RRP-015 rehearsal arm
The rehearsal arm (+648 rows, 108,389 tokens, 55 updates) scored acquisition 225, arithmetic 63/50, retained 35, LC **134** (lost 27, p=2.53e-5), UNK 8.

### CONFIRM-002 (fresh banks, read once)
- Runtime: Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1, FP32, greedy, max_new_tokens 48, thinking off.
- Banks: T_BANK_V4 (864 rows, 288 per family, sha `82f0a642…`) and HSKILL_SOUP_V3 (336 boundary rows + 109 answerable, sha `686df45b…`).

| Arm | T primary | T strict | Boundary /336 | Attempts /109 | Correct /109 |
|---|---|---|---|---|---|
| BASE | 5 | 5 | 185 | 109 | 41 |
| SPARSE | 430 | 430 | 183 | 109 | 37 |
| H25 | — | — | 263 | 109 | 41 |
| **TH25 (Titan-Transition)** | 434 | 434 | 250 | 109 | 35 |
| THFULL | 423 | **260** (342 rows gained a trailing period) | 249 | 109 | 32 |

Verdicts (Holm-adjusted):
- **C1 PASS:** TH25 − SPARSE = +4 (9 gained / 5 lost), lower bound −2.0 against a margin of −21.5.
- **C2 PASS:** +67 boundary (78 gained / 11 lost), p = 1.37e-13.
- **C3 FAIL:** share of H's solo effect kept P_H = 0.859, lower bound 0.718 against a required 0.75.
- **C4 PASS:** 0 discordant answerable rows.
- **C5 FAIL:** THFULL loses only 7 T rows (p = 0.49) against the required 43.
- Gate G_H: H25 − BASE = +78, p = 5.8e-20.
- Per-class TH25 − SPARSE: clarification_request +20, uncertainty +18, boundary_setting +13, insufficient_information +7, safety_boundary +7, clarification +5, knowledge_boundary −3. The −3 is a predicate artefact: "I need your calendar details." is not credited by the regex.

## 3. Key research discoveries

1. **The update is heavy-tailed.** The top 25% of elements carry 83.6% of ‖δ‖; the top 10% carry 63.8%.
   - At nominal 0.5: k=0.5 pruning scores 235 acquisition, while α=0.5 scaling scores 219. Pruning keeps the skill; scaling dilutes it.
   - Working interpretation: the skill lives in the large-magnitude head of δ, the long-context/abstention damage in the small tail.
   - Long context is non-monotone along both lines and peaks at 163–164 in the interior.
2. **Pruning works as a repair.** In REP-001 it never made retained or LC worse (3/3 draws), but the LC gain ranges +0 to +15.
   - The gain is largest in the most damaged draw: s29 went 144 → 159 and was better on every endpoint.
   - Acquisition is draw-stable (span 7). Retention is not (LC span 13), so single-draw retention numbers carry little information.
3. **BF16 export erases much of the update.**
   - 60.61% of δ elements (190.7M of 314.6M) become zero, and 24/220 tensors are erased entirely (mostly LayerNorms and q_norms).
   - The effect resembles α-shrinkage: Titan-Exp's 162 sits with the interpolants (163–164), not with its FP32 parent (157). This is AMENDMENT_001.
   - BF16 zeroing is **not** a magnitude mask: Jaccard 0.615 against a size-matched magnitude mask, because it depends on each weight's ULP.
   - Storing Titan-Sparse in BF16 would zero 28.49% of the kept elements and distort its norm to 1.0347.
4. **Composition is an allocation problem, not interference** (COMPOSE-001 → INDEP-001/002/004; synthesis in `COMPOSITION_SYNTHESIS_20260921.md`).
   - Summing both deltas at 1.0 failed: P16(δ_T)+P16(δ_H) gave acquisition 175 and LC 127; δ_T+δ_H unrounded gave 139.
   - Giving each skill its own coefficient works: 1.0·T + 0.25·H gives acquisition 235 = 235, 0 answerable losses, boundary +20/−4 (p=0.0015), LC 164. It replicated on seed 23.
   - The two coefficients are separable (interaction ≤ 3 rows on every endpoint).
   - δ_H's usable band is [0.10, 0.50]; at α=0.5 it costs 12 LC rows (p=0.0042).
   - Magnitude, not rank or direction, is the operative variable: r16 ≈ N16 (full-rank δ_T norm-matched to r16) ≈ α=0.75.
   - Geometry: cos(δ_T, δ_H) = 0.0027. Mean rank-16 subspace cosine is 0.151 for δ_T vs δ_H, against 0.60 for same-skill seeds and 0.0885 for a random null. The same skill converges on the same subspace across seeds.
   - CONFIRM-002 confirmed this on fresh banks. CONFIRM-001 found that averaging costs 26% and adding dOpallo at full strength keeps 413/447 (C3 failed there too).
5. **Instrument caveats.**
   - The arithmetic panel measures **transfer, not retention**: Supa scores 47, and every trained arm gains new rows (`PANEL_IS_TRANSFER_NOT_RETENTION_20260920.md`).
   - High-dose LC "damage" is **entirely abstention collapse**: answerable rows stay at 147–149/192 while UNKNOWN emissions fall from 13 to 0 (`HIGHDOSE_DAMAGE_IS_ABSTENTION_20260920.md`).
   - `ledger_replay` is the binding family in every arm.
   - The legacy per-class monitor fails every arm and does not discriminate between candidates.
6. **Shortcut warning for SELECT-VIEWS-001** (`FAMILY_DECOMPOSITION.json`, `STRUCTURAL_FLOOR_PROBE.json`):
   - prerequisite_plan's named constraints are dead (precondition 0/128, ordering 0/128). The structural predictor `ignore_preconditions` scores 128/128.
   - ledger_replay's `correct_kind_ignore_status` scores 117/128.
   - About 43% of the acquisition gain comes from the dead-constraint family.
   - So the "skill" is essentially executing one JSON arithmetic or bitmask operation. The published floors do not bound structural shortcuts.

## 4. Where code and data live

| Item | Path |
|---|---|
| Training controller | `C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/reasoning_posttraining_v1/` (`controller.py`, `engine.py`, `prepare.py`, `process_dose.py`, `select_views.py`, `sources.py`). Branch `codex/dataset-retention-audit-v1`. Original command: `python -m experiments.reasoning_posttraining_v1.controller train --source … --prepared … --plan … --grant … --journal … --max-sequence 2048 --experiment RRP-008-PROCESS-DOSE` |
| Training venv | `C:/Users/clclc/BusinessProject/.venv-mq001` (py3.11.15, torch 2.10.0+cu126, transformers 4.57.1) |
| Clean clone used for runs | `ChatBubble-retention-pilot-20260917` (**deleted**; outputs rescued under `rineya-dataset-retention-records/rescued-pilot-outputs-20260922`) |
| RRP-015 prereg, summaries, raw eval JSONs | `…/ChatBubble-ced-native-loader-qualification/docs/model-quality/audit-2026-09-17/` (`RRP015_PREREG.md`, `supervised_probe.py`, `rrp015-evidence/{PROCESS,RETENTION_PROBE,LONGCTX}_rrp015_*.json`) |
| Probes | `…/experiments/rab001_exploration/{process_probe.py, retention_probe.py, long_context_probe.py}` (manual greedy, FP32 DenseReference) |
| Long-context suite | `Chat-Bubble/Rineya_Long_Context_Demo_v1/data/all_prompts.jsonl`, `all_answer_key.jsonl` |
| Retention panel data | `rab001` `data/monitor_probe.jsonl` |
| Forge RL v2 source | `…/ChatBubble-ced-native-loader-qualification/rineya_reasoning_forge_rl_v2/` (`process/train_transitions.jsonl.gz`; oracle in `forge_rl/oracle.py`, tasks in `forge_rl/tasks.py`) |
| Records root | `C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/` |
| RRP-015 run | `rrp015-20260917/` (`control-prepared/microfit-train.jsonl` = the 384 tokenized training rows with ids and labels; `DATA_CONTRACT.json`; `control-journal/runs/train-001/microfit/steps.jsonl`; `summary-001/SUMMARY.md`) |
| SELECT-VIEWS-001 | `select-views-001/{SELECT_VIEWS.jsonl, FLOORS.json, BUILD.json}` |
| INT-001 | `int001-20260917/` (`build_arm.py`, `run_battery.sh`, `INT001_RESULTS.md`, `BF16_IS_IT_A_MAGNITUDE_MASK.json`) |
| SPARSE-001 | `sparse001-20260917/`; builder also copied to `Rineya-Titan-Sparse-Experimental…/evidence/build_sparse.py` |
| REP-001 | `rep001-20260917/` (`rep_build.py`, `prepare_rep001.py`, `s23/`, `s29/`) |
| COMPOSE-001 | `compose001-20260918/` (`build_compose.py`, `subspace_null.py`) |
| INDEP-* | `indep00{1..5}-20260921/` (`build_indep.py` defines P16) |
| HSKILL banks | `hskill-views-001/`, `hskill-views-002/` (`boundary_rule.py` sha `416cfb6a…`, `hskill_probe_v2.py`; sealed `HSKILL_CONFIRM_V2.jsonl`, 224 rows, still **UNREAD** and operator-reserved) |
| CONFIRM-002 | `experiments/composition_confirmation_v2/` in Chat-Bubble worktrees, e.g. `C:/Users/clclc/BusinessProject/ChatBubble-efe-func001/experiments/composition_confirmation_v2/` (`PREREG.md`, `build_arms_c2.py`, `score_boundary.py`, `run_stage_b.py`, `audit_confirm002.py`, `T_BANK_V4.json`, `HSKILL_SOUP_V3.jsonl`). The T scorer is `../fresh_dual_skill_instrument_v1/score_arms.py`. |

**Trained FP32 tensors.** Local copies were deleted 2026-09-26 (`storage-delete-20260926.json`).
- They are in the private HF vault `Luncy1978/rineya-published-cores`, path `research-records/delta-weight-screens-20260918/_trained_tensors/`, revision `33b5f9a46f4c49f22f1a305fdb21373f15401eb8`.
- Files: `delta_T_seed17` (`07156fb5…`), `delta_T_seed23` (`e7f17175…`), `delta_T_seed29` (`4f4e1c25…`), `delta_H_boundary` (`a1103060…`), `delta_MIX_rehearsal` (`f0ec2968…`). Each is about 1.26 GB.
- Despite the "delta" names, the content is the **trained tensors**, so δ = file − Supa.
- One third-skill delta is still local: `triskill001-20260926/backup/trained_C_fp32.safetensors`.

**Recoverable locally without downloads:**
- dSparse = Titan-Sparse − FP32(Supa).
- 0.25·P16(δ_H) ≈ Titan-Transition − Titan-Sparse.
- BF16-rounded δ_T = Titan-Exp − Supa.
- Supa 1.7 is at `Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze`.

**Follow-up lines already explored or in progress** (check before designing new work). These live under `ChatBubble-efe-func001/experiments/`:
- composition_confirmation_v1, titan_opallo_skill_vector_v1, titan_model_soup_mini_v1
- delta_subspace_geometry_v1 (elementwise cosine is the wrong statistic; overlap should be weighted by tr(dSᵀ dO Σ))
- functional_additivity_v1, module_disjoint_composition_v1, o_side_module_vs_magnitude_v1 (verdict NULL_MODULE_ROUTE_BUYS_NOTHING)
- clare_sparse_composition_v1, jumplora_rineya_skill_isolation_v1, rineya_composable_subspace_stack_v1, maxrl_composable_skills_v2 (data only)

Elsewhere under `C:/Users/clclc/BusinessProject`: Titan-Joint-Quality-Pilot-v1–v3, Titan-Quality-Patterns-v1, titan-sparse-quality-records, and triskill001.

## 5. Which Titan is strongest for what

- **Transition skill with the best preservation profile:** Titan-Sparse. It is the preregistered winner and essentially dominates Titan-Exp (+3 arithmetic, +2 retained, +2 LC, +2 UNK, acquisition within noise). It is FP32 only, and its retention numbers are single-draw.
- **Transition plus boundary/clarification abstention:** Titan-Transition. It is the only Titan with fresh-bank confirmation, and the only one carrying two confirmed skills (+67 boundary, T preserved, no over-refusal).
  - Caveat: answerable correctness is 35 vs Sparse 37 vs Base 41 (descriptive only).
  - LC, arithmetic panel and retained-47 were **not measured** for TH25. The closest analog, INDEP-001 (P16(δ_T)+0.25·P16(δ_H)), scored LC 164.
- **Titan-Experimental:** the historical control. Smallest file (BF16), but the worst preservation. Useful mainly as the "BF16-rounded δ_T" reference.
- **None of the three** carries a general-quality claim or release qualification. For long-context Q&A the family's dense best remains Supa 1.7 (161/256) or Supa-A1 (168/256 with remote code).

## 6. Loading recipe (the one the evaluations used)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, low_cpu_mem_usage=True,
                                             local_files_only=True, device_map={"": "cuda"}).eval()
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
out = model.generate(**enc, do_sample=False, max_new_tokens=48, eos_token_id=[151645, 151643], pad_token_id=151643)
reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```

- **dtype:**
  - **Always FP32 compute.** Titan-Exp is stored BF16 but was evaluated in FP32; native BF16 serving is not qualified.
  - **Never cast Sparse or Transition to BF16.** Doing so silently destroys the update. Pass `torch_dtype=torch.float32` explicitly rather than relying on "auto".
- **Template:** Qwen3 template; the tokenizer_config template supports `enable_thinking`, which must be False. Use system+user messages and `add_special_tokens=False`.
- **Generation:** greedy. The shipped `generation_config.json` (do_sample=True, temperature 0.6, top_k 20, top_p 0.95) is **overridden** in every evaluation.
  - max_new_tokens: 48 for CONFIRM-002, 24 for the LC suite (max_context 4096).
  - EOS ids: 151645 `<|im_end|>` and 151643. Pad 151643.
- **transformers version:**
  - 4.51.3 for RRP-015/INT/SPARSE/REP/COMPOSE (digest `a329eb7bb0`). Its venv `.venv-rab001-cu126` no longer exists.
  - 5.14.1 with torch 2.6.0+cu124 for CONFIRM-002.
  - Closest local env: `C:/Users/clclc/BusinessProject/Chat-Bubble/.venv` (transformers 5.14.1, torch 2.12.1+cu126). Config files record 4.51.0.
- **Resources:**
  - FP32 weights are 2.38 GB. Inference fits in 8 GB VRAM; CONFIRM-002 used `torch.cuda.set_per_process_memory_fraction(0.78)`.
  - With about 2 GB of free host RAM, load one model at a time (safetensors mmap with low_cpu_mem_usage).
  - Retraining the RRP-015 recipe needs about 4 GiB of host RSS, so free RAM first.
- **Spent instruments:** the LC suite, SELECT-VIEWS-001 (selection-only, never confirmation), T_BANK_V4 and HSKILL_SOUP_V3 are all spent. Any new confirmatory claim needs fresh banks.