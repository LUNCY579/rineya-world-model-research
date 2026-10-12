# complementary/: RCX, Rineya complementary intelligence research

This folder is maintained by the complementary research session on branch `research/rineya-complementary-intelligence-v1`.
It is a peer of `bwm/` (BWM-001) and `rcwm/` (RCWM-T1), and it never writes outside `complementary/`.

**Spec:** bundle RCI-001 v1.0 (`rineya_complementary_research_2026-10-09/`), outside the repository.

## Status (2026-10-12, about 06:40 CDT)

| Track | State | Key evidence |
|---|---|---|
| **RCM-001** (causal and counterfactual rules) | The instrument is qualified (R-01). The char-model tier is closed (R-02). **Frozen Titan-Sparse and T2 ignore in-context rules** (R-03). DEV pilots: R-05, R-06. **SEALED confirmation (PREREG_E01N, 3 seeds per arm): REVISE** (R-08). Counterfactual and paraphrase data beats rule-varied data on both co-primaries in every seed: training wordings +0.018 [+0.006, +0.030]; held-out wordings +0.090 [+0.069, +0.113], Holm p 0.0002, short of the preregistered +0.10. N-CFSD is RULE-like on unseen statuses in 3/3 seeds, but unseen *positive* rule phrasings are still read as "apply" (14% correct no-ops). **E01R:** the blind wording pools are fixed (DEV P2 45, sealed P3 48), and the verifiers are calibrated (12/12 ambiguous plants rejected). The **step A diagnostic** (R-10) found that N-CFSD decides 0.50 of the positive-rule no-ops on fresh wordings, with complement-first wordings the weakest. The frozen branch rule selects **B4: train R2 + R3** against N-CFSD on sealed P3. The prereg is being finalized. | R-01 to R-03, R-05, R-06, R-08, R-10 |
| **RGM-001** (relational) × **RVM-001** (value) | A relational GNN value generalizes to 8–12 blocks: policy accuracy 0.998 vs MLP 0.563 (R-04). The **preregistered E07b reached CLOSED**: the GNN + misplaced hybrid doubles low-budget success and matches classical coverage (R-07). Joint B4 with BWM (inside its frozen beam planner) is agreed. | R-04, R-07 |
| **RSM-001** (skills) | **E11 (frozen prereg) done** (R-09). **E11a ADVANCE:** 8 macros mined from the 211 Rineya-verified B3 plans raise GBFS + A3 from 52 to 77 of 120 hard instances at 200 expansions (+25/−0, p = 6e-8). They beat random and shuffled-plan macro controls (Holm). This is tier B (an external planner), and the specificity holds with the A3 heuristic, not misplaced_blocks. **E11b REVISE:** a learned subgoal order transfers (−7% length vs bottom-up, CI touches 0) but is no shorter than ready-first, and is bounded by its achiever labels. | R-09, CX D-31, D-35 |

Every claim in `EXPERIMENT_RESULTS.md` states its tier. **No retained or dense-model claim has been made.** Native cells
are component adapters on a frozen core. E33 retention through RCWM is tied to a confirmation ADVANCE; R-08 was REVISE, so
none has been requested.

## Documents

| File | Purpose |
|---|---|
| `RESEARCH_ROADMAP.md` | Priorities, hypotheses and dependencies |
| `EXPERIMENT_RESULTS.md` | Measured results only, with evidence and limitations |
| `COLLABORATION.md` | Peer map, scope split, resource policy, shared findings and agreements |
| `DECISIONS.md` | Decisions, incidents and revisions (`CX D-nn`, `CX I-nn`) |
| `NEXT_STEPS.md` | Most promising next experiments |
| `docs/REQUESTS.md` | Requests to peers (`CX Q-nn`) and their replies |
| `docs/understanding/` | Read-only surveys of the bundle and the peer programmes |

## Code map

| Path | Content |
|---|---|
| `rcx/rcl/` | Rule-conditioned ledger (RCL): spec-driven generator, exact interpreter, independent oracle, qualification, scoring and floors |
| `rcx/models/` | CPU char LM and symbol-token LM (SymLM) on a shared core |
| `rcx/native/lora_cpu.py` | CPU LoRA SFT with the exact D-03/T2 recipe; parity with the vendored trainer is verified |
| `rcx/bw/` | Blocksworld E07/E08: pinned read-only BWM imports, exclusions, exact h*, encodings, GNN/MLP, heuristics, search |
| `rcx/guard.py`, `rcx/stats.py` | Thermal/RAM courtesy guard; parent-clustered statistics |
| `experiments/RCX-E01/` | RCL spec, design, prereg draft, CPU runner and analysis |
| `experiments/RCX-E01N/` | Native probe and cells (Titan-Sparse), prereg draft, analysis |
| `experiments/RCX-E07/` | Relational value learning and search: design, runner, exclusions |

## Running (from `complementary/`, CPU only, one job at a time)

```bash
python -B -m unittest discover -s tests -t .
```

```bash
python -B -m rcx.rcl.qualify experiments/RCX-E01/qualification/QUALIFICATION.json
```

The native probe needs at least 12 GB of free RAM and runs for hours:

```bash
python -B experiments/RCX-E01N/native_probe.py --out runs/RCX-E01N/probe1 --arms OFF,T2,SUPA --n-parents 40
```

The native cells are built, audited and trained with these four commands. Training needs at least 12 GB of free RAM and
takes about 2.5 h per cell on CPU:

```bash
python -B experiments/RCX-E01N/native_e01.py build --out runs/RCX-E01N/cells
```

```bash
python -B experiments/RCX-E01N/native_e01.py audit --out runs/RCX-E01N/cells
```

```bash
python -B experiments/RCX-E01N/native_e01.py train --out runs/RCX-E01N/cells --arm N-VR --seed 0
```

```bash
python -B experiments/RCX-E01N/native_e01.py eval --out runs/RCX-E01N/cells --arm N-VR --seed 0
```

Compare the native arms:

```bash
python -B experiments/RCX-E01N/analyze_native.py --out runs/RCX-E01N/ANALYSIS_DEV.md
```

E07/E08: rebuild the exclusions first, then run the experiment into a new output directory:

```bash
python -B -m rcx.bw.excl
```

```bash
python -B experiments/RCX-E07/run_e07.py --out runs/RCX-E07/main --phase all
```

All jobs honor `rcx/guard.py`, which pauses on the shared thermal flag, at tz01 ≥ 97 °C and on low RAM. Never run two
heavy jobs at once (CX I-02).
