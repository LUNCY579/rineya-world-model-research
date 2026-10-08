# RCWM-T1 — latent consequence readout from frozen Titan

The first native execution of the RCWM-001 plan on Rineya Titan models. It asks four things.

- Does frozen Titan encode the multi-step consequences of a precommitted action plan in its residual stream?
- Is that encoding caused by the 0.056%-norm transition delta?
- Does a goal-agnostic consequence readout make better goal-conditioned decisions than a matched scalar evaluator?
- Can such a readout improve Titan's actual multi-step outputs with no weight change? This last part is probe-verified rollout (PVR).

Start with `DESIGN.md` for hypotheses, `DECISIONS.md` for the decision and incident log, and
`docs/design_critique/` for the pre-analysis critique and the amendments that were adopted. The
reader reports that grounded the design are in `docs/understanding/`.

## Layout

| Path | Contents |
|---|---|
| `rcwmt/env.py` | Families, FDS-oracle dynamics (cross-checked), multi-step prompts, features |
| `rcwmt/bank.py` | Parents × K candidate plans; historical-exclusion checks; filters; order twins |
| `rcwmt/runtime.py` | FP32 arms (cores and on-device delta arms), GPU lease, resource guards |
| `rcwmt/capture.py` | Answer-slot and action-boundary hidden states; G0 parity checks |
| `rcwmt/generation.py` | Greedy e2e, chained self-rollouts, batched sampling with states |
| `rcwmt/fidelity.py` | Exact change-mask metrics, floor ladder, input decodability, P strata |
| `rcwmt/decisions.py` | Goal-conditioned selection: B3 vs B2 (mlp and bilinear), S, G, F |
| `rcwmt/pvr.py` | Logistic and Gaussian next-state verifiers for probe-verified rollout |
| `scripts/run_arm.py` | Capture (plus optional generation) for one arm, under the lease |
| `scripts/fidelity_report.py` | H1/H2 fidelity per arm and cross-arm contrasts; optional decisions |
| `scripts/run_pvr.py` | PVR stage1 / hstates / analyze / stage2 |
| `scripts/pilot_queue.sh`, `launch_pilot_queue.cmd`, `wait_then_queue.sh` | Detached, resumable GPU queue |
| `data/bank_pilot.json` | Exploratory bank (200 parents per family) |
| `data/bank_confirm.json` | **Sealed** confirmation bank (150 test parents per family); untouched until PREREG |

## Reproduce

Use the hermes venv: Python 3.11.15, torch 2.6.0+cu124, transformers 5.14.1. Run everything from
this directory.

```bash
python -m rcwmt.bank --parents 200 --seed 20261005 --out data/bank_pilot.json
python scripts/run_arm.py --bank data/bank_pilot.json --arm sparse --parity --gen
python scripts/fidelity_report.py --bank bank_pilot --arms sparse,supa --decision-ks 4
python scripts/fidelity_report.py --bank bank_pilot --compare sparse:supa
python scripts/run_pvr.py stage1 --bank bank_pilot --arm sparse --splits dev,test
python scripts/run_pvr.py analyze --bank bank_pilot --arm sparse --verifier-arms sparse,supa
```

GPU jobs always take the shared lease `%TEMP%/rineya-gpu-0.lock`, and only one model is loaded per
process.
