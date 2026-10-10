# RCX-E01: rule-semantics factorial on the rule-conditioned ledger (CPU mechanism tier)

| | |
|---|---|
| Bundle mapping | E01 "Rule semantics factorial" (RCM-001), plus the PREREG-3 anti-shortcut battery |
| Tier | **mechanism**: small character-level transformers trained from scratch on CPU. This is *not* a Rineya-core result. |
| Status | ENGINEERING PILOT (DEV only). The confirmation preregistration (`PREREG_E01_CPU.md`) is frozen before any SEALED battery is minted. |

## 1. Question

Which training-data factors make a learner **execute a rule supplied in its input**? The alternative is that it learns
a lexical trigger, or only the single fixed rule it was trained on.

PREREG-3 showed the failure mode on Titan-Sparse with T2: a `void` trigger, and "neither base model reads the rule
text". The CPU tier asks the question in a fully controlled world where everything is learned from scratch, so the
data factor can be isolated:

| Factor | Levels |
|---|---|
| Rule variation (R) | **fixed** (always `S+{posted}`, the T2 analog), **varied iid**, or **varied with counterfactual twins** |
| Surface diversification (SD) | canonical, or diversified (train paraphrases, key-order permutations, 3–4 accounts, nonce words) |
| Target format (F) | answer-only (`ans`), predicate trace (`cmp → calc → after`), or constant filler (`fill`) |

## 2. Arms (matched label budget: identical number of training examples, steps and batch size)

| Arm | Regime | Format | Role |
|---|---|---|---|
| A1 `FX-ans` | FX | ans | T2 analog. Expected TRIGGER. |
| A2 `FXSD-ans` | FX_SD | ans | T2b analog (RCWM's fix, fixed rule with surface diversity) |
| A3 `VR-ans` | VR | ans | bundle 2x2 cell: ordinary × answer-only |
| A4 `VR-trace` | VR | trace | bundle 2x2 cell: ordinary × trace |
| A5 `CFSD-ans` | CF_SD | ans | bundle 2x2 cell: counterfactual/surface-diversified × answer-only |
| A6 `CFSD-trace` | CF_SD | trace | bundle 2x2 cell: counterfactual/surface-diversified × trace |
| A7 `CFSD-fill` | CF_SD | fill | trace-length control (constant filler, D-36 4b analog) |
| A8 `VRSD-ans` | VR_SD | ans | decomposition: SD without twins |
| A9 `CF-ans` | CF | ans | decomposition: twins without SD |

**Eval-time control** (no extra training): `+canon`, the canonicalizer tool (tier B). It parses any rendering and
re-renders it in the canonical surface, applied to A3 and A5. If an arm's gains vanish under canonicalization, the
credit goes to interface repair (bundle RCM-001 failure interpretation).

**Floors and ceiling** are scored on the same items: COPY, ALWAYS, TRIGGER, CONTAINS_POSTED, STATUS_LOOKUP,
RULE_WORD_MATCH, MAJORITY, and ORACLE (the exact parser plus interpreter).

## 3. Data

- **Generator.** `rcx/rcl/gen.py`, following `SPEC_RCL.md`.
- **Qualification.** The interpreter is qualified by `rcx/rcl/qualify.py`:
  - Q1: 20,000 posted-rule cases equal to RCWM `env.step` and `p3_common.rule_gold`;
  - Q2: all generated records equal to the independent oracle `oracle_indep.py`;
  - Q3: round trip.
- **TRAIN.** Per regime, `make_train(regime, N_TRAIN, seed)`. The training seed is shared across formats of the same
  regime, so `VR-ans` and `VR-trace` train on identical parents and differ only in the target.
- **DEV battery.** `make_battery("dev", N_DEV_PARENTS, 20261220)`:
  - 13 strata × 6 surfaces;
  - each parent contains at least one *apply* variant and one *no-op* variant, paired across surfaces.
- **SEALED battery.** Minted only after the preregistration freeze. The seed is derived from a committed salt.
- **Split hygiene** (tested): zero semantic-hash overlap and zero (state, event-without-status) skeleton overlap across
  TRAIN, DEV and SEALED. DEV and SEALED status words are disjoint from TRAIN and from each other, with edit distance ≥ 2
  across splits. PREREG-3 and factory vocabularies are excluded everywhere.

## 4. Endpoints (pilot reporting; frozen for confirmation in the prereg)

**Primary candidates** (one to be chosen in the prereg, using DEV pilot evidence about floors and ceilings only, never
arm-vs-arm DEV differences):
- **P-ho:** parent-macro exact transition accuracy on the semantic composite {F2, F3, F5, V1, V2, V3, V5} under the
  combined held-out surface `ho` (bundle: "held-out semantic/surface composite");
- **P-in:** the same composite under the canonical surface `in`, which isolates semantic OOD.

**Secondary:**
- each stratum and each surface component (`ho_tpl`, `ho_key`, `ho_name`, `ho_many`);
- decision accuracy;
- error direction (applied_when_noop, false_copy, arith, format);
- V3 parent-all-exact (both inverted twins);
- V6 held-out compositions;
- V7 big sets;
- M1 and M2 magnitudes (report only).

**Descriptive per-arm verdict** (PREREG-3 definitions, on probe variants):
- RULE: F2, F3 and F5 probes ≥ 0.80, plus the twin guard.
- TRIGGER: F2, F3 and F5 probes ≤ 0.30, F1 ≥ 0.80, and most errors applied-when-noop.
- MIXED: otherwise.

## 5. Statistics

- **Unit.** The semantic parent. Variants are averaged within a parent, and parents are resampled with a paired cluster
  bootstrap (`rcx/stats.py`).
- **Seeds.** One seed for the engineering pilot. At least 3 seeds for the preregistered comparison, with the
  seed-by-parent bootstrap.
- **Multiplicity.** Holm over the small primary family. Everything else is descriptive.
- **Decision vocabulary** (bundle COMMON_PROTOCOL): ADVANCE / REVISE / STOP / INCONCLUSIVE.

## 6. Compute and safety

- CPU only, ≤ 4 threads, using `rcx/guard.py` (pause at tz01 ≥ 97 °C, resume at ≤ 93 °C, honor `.thermal_pause`).
- No GPU and no Rineya model.
- Checkpoints are kept under `runs/` and are git-ignored.

## 7. What a positive CPU result would *not* establish

- Any change in Rineya/Titan behavior.
- That the same data recipe works in a LoRA on a pretrained 0.6B core.
- Transfer beyond the ledger world.

A positive result would only motivate the native kit (§8), whose claims require the A/B/C/D tiers, matched SFT and E33
retention.

## 8. Native follow-up (not executed; needs GPU lease plus operator approval)

1. **Tier-A probe.** Frozen Titan-Sparse and Supa on the RCL battery rendered in Titan's chat frame, with the rule in
   the system text. This measures whether the cores read supplied rules (PREREG-3 suggests not).
2. **LoRA cells.** On frozen FP32 Titan-Sparse, unmerged, using the T2 recipe (r16/α32, 190 steps × 16). The winning
   CPU recipe is compared with the matched non-counterfactual recipe, against OFF and T2, plus E33 retention.
