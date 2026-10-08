# Implementation status

| Capability | Status | Evidence / next gate |
|---|---|---|
| Research synthesis and 43-source register | Delivered | Report and research directory |
| Architecture, action, feature, data, mask and evaluation contracts | Delivered proposal | specs, schemas, configs |
| Tiny causal action-token Transformer | Implemented CPU reference | reference/rcwm/model.py |
| Learned bottleneck with supervised measured features | Implemented toy only | demo results; not self-supervised discovery |
| Linear goal revaluation and finite-return accounting | Implemented/tested | reference tests |
| Parent-stable split and precommitted-plan checks | Implemented subset/tested | data module; production provenance still required |
| Train-only k-means | Implemented utility/tested | not the clustering research experiment |
| DAG validation and availability masks | Implemented utilities/tested | no production scheduler |
| Reference package tests | 60 passed | validation/pytest_results.xml |
| Synthetic training demonstration | Executed with known fidelity limitation | validation/demo; no model-quality promotion |
| Native Rineya loader/capture integration | Not implemented or executed | G0-native |
| Matched scalar native/sidecar experiment | Planned | G1/G2 |
| Self-supervised outcome target encoder | Planned | G2 |
| Learned goal-language mapping / new-goal transfer | Planned | G3 |
| VQ and learned consequence clustering training | Planned | G3 |
| Policy-conditioned successor learner / distributional planner | Planned | G4 |
| Calibrated uncertainty / OOD support | Planned | G4 |
| Long-context and parallel-DAG execution | Planned | G4 |
| Native typed-slot adapter and retention evaluation | Planned | G5 |
| Distillation and imagined RL | Deferred | G6 |
| Recurrence controller and visual world model | Deferred | G7 |
| Repository branch/commit/push | Not performed | Separate authorization and actual repository review |

This is an implementation-ready research specification with a small tested mechanics aid, not a
completed implementation of every proposed component or a certified production system.
