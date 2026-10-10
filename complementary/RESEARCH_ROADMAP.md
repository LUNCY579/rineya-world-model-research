# RESEARCH_ROADMAP: RCX complementary intelligence

**Spec:** bundle RCI-001 v1.0, `rineya_complementary_research_2026-10-09/`.
**Program label:** RCX. Status as of 2026-10-09 is CPU only.

## Guiding question

Which *specialized mechanisms* give Rineya capabilities that its forward/consequence world models (BWM, RCWM) do not already
provide?

"Complementary" means one of two things:
- it measurably fixes a failure the world models have; or
- it adds a capability they lack.

It is *not* complementary if it only re-implements what the world models already do.

Each mechanism is judged on the same ladder of evidence:

1. **Instrument.** A qualified exact oracle, parent-grouped splits, shortcut floors, and an OOD battery. Floor validation
   must show that the battery can separate the mechanism from its shortcuts.
2. **Small-model mechanism study** (CPU, tier "mechanism", *not native*). Does the mechanism or data factor change
   behavior on fresh OOD strata against matched baselines? Report results by seed and by parent cluster.
3. **Native tier-A probe** (inference only). Does frozen Titan-Sparse or Supa already have the capability?
4. **Native student** (LoRA on frozen FP32 Titan-Sparse, unmerged). Report A/B/C/D with matched SFT, plus E33 retention.
   This requires a GPU lease and operator approval.
5. **Dense transfer.** A fresh 0.6B student without the specialist (bundle INTEGRATION_AND_TRANSFER).

**What can be claimed at each stage.** Stage 2 results are never described as Rineya intelligence gains. Only stage 4 or 5
results, with retention, may be described as native improvements.

## Priorities and hypotheses

### P0-a RCM-001: rule-conditioned causal execution (started)

| ID | Hypothesis (falsifiable) | Primary endpoint | Depends on |
|---|---|---|---|
| **RCX-E01** (bundle E01) | Training on *counterfactually varied, in-context rules* (opposite meanings, nonce statuses, minimal twins, surface diversity) yields RULE-level behavior on fresh vocabularies and inverted rules. Fixed-rule training, even surface-diversified, yields TRIGGER or fixed-rule behavior that cannot follow a new supplied rule. A predicate trace (`cmp → calc → after`) adds to this beyond a constant filler. | Parent-macro exact transition accuracy on the predeclared held-out semantic+surface composite | qualified interpreter; floor validation |
| RCX-E02 (bundle E02) | Computed predicates (thresholds, compositions, larger arithmetic ranges) extrapolate only when the trace exposes the comparison. | Exact accuracy on held-out thresholds and compositions | E01 instrument |
| RCX-E03 (bundle E03) | Rule handling learned in the ledger transfers to unseen natural-language constraints and to the factory's `{x,y}` causal-AST domain. | Exact accuracy on the domain split | E01 positive |

### P0-b RGM-001: relational / neuro-symbolic

| ID | Hypothesis | Endpoint |
|---|---|---|
| RCX-E07 | On Blocksworld/graph transitions with more objects, held-out names and orders, a graph-structured learner (GNN over entity-relation facts) generalizes structurally better than a flat MLP or a sequence model at matched labels. A canonicalized-text model closes part of the gap. | Exact next-state and validity accuracy at the larger object counts |
| RCX-E08 | A learned heuristic (cost-to-go) with exact dynamics solves more hard Blocksworld instances within a fixed expansion budget than the hand-coded unmet-goal heuristic (BWM `plan_beam`, used read-only). | Solved within budget on fresh instances |

### P0-c RVM-001: value / process critics

| ID | Hypothesis | Endpoint |
|---|---|---|
| RCX-E04 | A held-out-error-trained legality critic (logistic first, then small net) detects invalid steps produced by a *different* error source with calibrated precision. | AUROC and precision at fixed recall on a held-out producer |
| RCX-E05 | Goal-conditioned successor features allow zero-shot re-ranking for new goal weights, matching per-goal value training at lower label cost. | Regret on held-out goal weights |
| RCX-E06 | A dead-end / value critic reduces beam failures (dead_end, budget) at equal expansions. | Success at a fixed budget |

### P0-d RSM-001: skills (after BWM B3)

- RCX-E11: option composition (AB / BA / conditional).
- RCX-E12: DAgger on top of the B3 distilled policy.

These start only after BWM's B3 is analysed: **no B3 reads before `runs/BW_B3_EVAL/B3_REPORT.json` exists**. E10 is
BWM's B3.

## Dependency graph

```
instrument(E01) ──► floors validated ──► E01 small-model factorial ──► E01 native probe (tier A, GPU or 12 GB RAM)
                                              │                              │
                                              ▼                              ▼
                                         E02 / E03                    E01 native LoRA cells (GPU lease + approval) ──► E33 retention
Blocksworld sim (BWM, read-only) ──► E07 GNN vs MLP vs seq ──► E08 learned heuristic ──► E06 critic in beam
                                                        └──► E04 legality critic ──► E05 successor features
B3 outputs (BWM) ──► E11 / E12
```

## Compute envelope

- CPU only: ≤ 4 torch threads, thermal back-off at tz01 ≥ 97 °C, < 1.5 GB RSS for small models.
- GPU: none requested yet. The proposed RCM pilot cap from the bundle is 4 GPU-h, and it is not an allocation.
