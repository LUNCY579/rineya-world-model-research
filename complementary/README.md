# complementary/: RCX, Rineya complementary intelligence research

This folder is maintained by the complementary research session on branch `research/rineya-complementary-intelligence-v1`.
It is a peer of `bwm/` (BWM-001) and `rcwm/` (RCWM-T1), and it never writes outside `complementary/`.

**Spec:** bundle RCI-001 v1.0 (`rineya_complementary_research_2026-10-09/`), outside the repository.

## Status (2026-10-10, about 15:00 CDT)

| Track | State | Key evidence |
|---|---|---|
| **RCM-001** (causal and counterfactual rules) | The instrument is qualified (R-01). The char-model CPU tier is closed as non-informative (R-02). **Native probe: frozen Titan-Sparse and T2 s0 ignore in-context rules** (R-03). The native LoRA cells N-VR, N-CFSD and N-FX are in progress on CPU. | `EXPERIMENT_RESULTS.md` R-01 to R-03 |
| **RGM-001** (relational) × **RVM-001** (value) | E07/E08 infrastructure is built and reviewed: exact h*, the BWM exclusion recipe, GNN, MLP, heuristics and search. The full run is queued. | `experiments/RCX-E07/` |
| **RSM-001** (skills) | Waiting: no B3 reads before BWM's `B3_REPORT.json` exists. | `COLLABORATION.md` |

RCX has trained no Rineya-core adapter yet. Every claim in `EXPERIMENT_RESULTS.md` states its tier.

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
