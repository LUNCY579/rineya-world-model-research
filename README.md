# Rineya world-model research: records and results

Private research record for two sibling programmes run on the Rineya Titan models (Supa 1.7 + a sparse transition
update, "Titan-Sparse"). Each programme keeps its own folder. This repository is for material reuse and for
validating results; it is not a model release.

| Folder | Programme | Contents |
|---|---|---|
| `rcwm/` | RCWM-T1: consequence world models on frozen Titan-Sparse | Mirror of the RCWM bundle: code, preregistrations, freeze records, decisions log, results, generations, logs, registered LoRA adapters |
| `bwm/` | BWM-001: forward world models and planners (RegisterWorld, Blocksworld) | Maintained by the BWM-001 session |
| `tools/` | Shared | Export and sync scripts |

## Where to start (RCWM)

- `rcwm/experiments/RCWM-T1/REPORT.md`: the report. Confirmed results are in §3 and §3b.
- `rcwm/experiments/RCWM-T1/DECISIONS.md`: every decision, incident, amendment and freeze, in order.
- `rcwm/experiments/RCWM-T1/RESULTS_PILOT.md`: exploratory findings R1–R14.
- `rcwm/experiments/RCWM-T1/RESULTS_CONFIRM.md`: PREREG-1, mechanically rendered.
- `rcwm/experiments/RCWM-T1/runs/bank_confirm2/analysis/PREREG_2_RESULTS.md`: PREREG-2, mechanically rendered.

## Confirmed results (preregistered, sealed fresh banks)

1. **PREREG-1, confirmed 2026-10-07.** Titan-Sparse chains its own one-step predictions into 4-step consequences for plan selection. Normalised regret is 0.448 (0 = oracle, 1 = random).
   - This beats each of the following by at least 0.10 regret: an answer-free rule, consequence heads on its own hidden states, a symbolic learner at equal labels, Supa 1.7's best generation, and its own single-shot answer.
   - Qualifiers: the C0 effect is driven by register, and the C1/C2 comparators sit at or below the answer-free floor.
2. **PREREG-2, confirmed 2026-10-08.** A LoRA trained on 1,501 oracle-labelled rule-coverage rows (unmerged; the T2 seed-0 artifact) changes ledger 4-step decisions as follows:
   - decision regret 0.544 → 0.056;
   - exact final state 0.029 → 0.698;
   - forgetting guard PASS.
   - Prerequisite learns a copy shortcut, and register is unchanged.

## Validation

- **Large binaries** (hidden-state captures, log-prob arrays, full model weights) are not stored here. `rcwm/EXCLUDED_LARGE_FILES.json` lists each one with its size and sha256, so a local copy can be checked.
- **Registered inputs:**
  - PREREG-1: `PREREG_1_FREEZE.json` and `scripts/verify_freeze.py`.
  - PREREG-2: `PREREG_2_FREEZE.json` and `scripts/confirm2_gen.py --verify`.
  - Both list every input with its sha256.
- **The registered PREREG-2 adapter:** `rcwm/experiments/RCWM-T1/runs/t2/t2_s0/adapter/adapter_model.safetensors` (sha256 92bc7c74…). Load it unmerged on FP32 Titan-Sparse; never dense-merge it (DECISIONS D-28).

## Rules for contributors

- Files over 25 MB stay out; list them in an exclusion manifest instead.
- No secrets, tokens or personal data.
- Run `git pull --rebase` before every push. Each session commits only inside its own folder.
- Nothing here is a release. Promotion of any core into the private vault, or upload to Hugging Face, requires the operator's explicit go-ahead.
