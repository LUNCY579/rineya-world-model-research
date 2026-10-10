# PREREG-LB results

Rendered mechanically by `scripts/lbeam_analysis.py` from the results JSON.

## T2 ENABLES PLANNING, BELOW THE NEAR-ORACLE BAR (on its training surface)

Reading: **enables planning** (branch YES).

- Co-primary 1 (all 180 instances): T2 127/180 (0.706) vs OFF 14/180 (0.078); T2 - OFF +0.628 (T2 only 113, OFF only 0); exact McNemar p <0.0001, Holm-adjusted p <0.0001: passes the pre-set bar (T2 - OFF >= +0.20 and Holm-adjusted p < 0.01).
- Co-primary 2 (the 52 needs-lookahead instances): T2 23/52 (0.442) vs OFF 3/52 (0.058); T2 - OFF +0.385 (T2 only 20, OFF only 0); exact McNemar p <0.0001, Holm-adjusted p <0.0001: passes the pre-set bar (T2 - OFF >= +0.20 and Holm-adjusted p < 0.01).
- Key secondary: T2/ORACLE success ratio 0.794 (95% paired bootstrap CI [0.73, 0.855], 10000 resamples, seed 17); T2 0.706, ORACLE 0.889; near-oracle bar (>= 0.80, lower bound >= 0.70) not met.

T2 enables planning, including on the instances greedy descent on the true hook cannot solve, but it stays below the near-oracle bar (T2/ORACLE >= 0.80 with lower bound >= 0.70).

Qualifiers (report-only; they never change the branch):

- Report-only: T2/ORACLE on the 52 needs-lookahead instances 0.719 (95% paired bootstrap CI [0.552, 0.868], 10000 resamples, seed 17); T2 0.442, ORACLE 0.615. It never changes the branch.
- Report-only: T2 vs IGNORE_STATUS (exact arithmetic without the status rule; enum W2 E12): T2 127/180 vs IGNORE_STATUS 63/180 (difference +0.356, T2 only 80, IGNORE_STATUS only 16, exact McNemar p <0.0001); this bears on whether T2's status-rule competence, not its arithmetic alone, carries its planning.
- D-35 audit (report-only): T2 under the held-out combined rendering (reversed keys, word account names, 'void' rendered 'cancelled') 5/180 vs the canonical frame 127/180 (difference -0.678, 95% paired bootstrap CI [-0.744, -0.611], exact McNemar p <0.0001): surface-dependent (a significant drop off the training surface).
- Report-only: enum W4 E24 point, T2 0.744 vs ORACLE 0.928 (T2/ORACLE 0.802).
- Report-only: select_fixed (PREREG-2-style selection among the K = 8 stored candidates) success T2 0.867, OFF 0.233, ORACLE 1.0, IGNORE_STATUS 0.494; IGNORE_STATUS beam (enum W2 E12) 0.35.

Limits that always accompany the result:

- Training surface only (trigger-level, PREREG-3): every prompt F answers is Titan's canonical one-step ledger frame (accounts A-D, statuses posted / void, compact sorted-key JSON). PREREG-3 found T2's void no-op to be a token cue and its accuracy to drop off this surface, so the result says nothing about held-out renderings (the SurfaceAuditF heldout_combined arm is report-only).
- Problem distribution: generator v2 in-range instances only. Every state reachable within D_CLOSE = 8 vocabulary steps has |b| <= 98, every amount is one of T2's 21 trained values (-9..12) and every status is posted or void, so scale actions have factor -1 or 0 only; optimal depth d* 2-6; 8 vocabulary actions per instance, including 1-2 void twins. This is a subset of PREREG-2's event distribution, fresh at payload level, not out of distribution.
- Planner and proposer: BWM's plan_beam (bwmn/blocks_planner.py, sha 18a5f22b...) with the model-free enumerating proposer (every vocabulary action as a one-step candidate, 0 proposer tokens), in ONE registered configuration as the primary (enum W2 E12, max_depth 7). The W4 E24 point, select_fixed and IGNORE_STATUS are report-only. Single-plan setting: no execution feedback while planning.
- One adapter: T2 seed 0 (PREREG-2's registered artifact), unmerged on frozen FP32 Titan-Sparse; OFF is the same PeftModel with the adapter disabled. No other seed, base model, proposer or planner configuration is tested.
- F is the first computed answer per (state, action) key, cached per arm, behind the registered input-range guard (|b| <= 98, trained amounts, posted / void), identical for every arm; a refused key ends its candidate. Every truly reachable state is in range, so a refusal also prunes a branch whose imagined state is impossible (dev simulation: about 0.02% of a T2-like F's keys and 0.12% of an OFF-like F's).
- Success is exact goal attainment of the one submitted plan (the verified plan, else the best node's prefix) under env.step. The needs-lookahead stratum is defined by greedy one-step descent on the true unmet hook; it is a property of the instance, not of any arm.
- 'Enables planning' is measured against the unadapted base model (OFF), whose F is close to useless on this task (dev simulation: success 0.12, needs-lookahead 0.04). On its own it does not show that near-oracle prediction gives near-oracle planning; that reading rests on the T2/ORACLE ratio.

Instances: 180 (52 needs-lookahead, 128 greedy-solvable).

## Arms (success of the submitted plan under env.step)

| arm / config | success | needs-lookahead | greedy | verified | verified-plan precision | optimal length | void-twin exposure: applied as posted (failures) | void step not a no-op |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eps:T2:W2E12 | 0.706 | 0.442 | 0.812 | 0.7 | 0.952 | 0.672 | 0.0 (0/53) | 0.0 |
| eps:OFF:W2E12 | 0.078 | 0.058 | 0.086 | 0.278 | 0.22 | 0.072 | 0.301 (50/166) | 0.506 |
| fixed:T2 | 0.867 | 0.827 | 0.883 | 0.778 | 0.979 | 0.867 | 0.0 (0/24) (incl. stored distractors) | 0.0 |
| fixed:OFF | 0.233 | 0.308 | 0.203 | 0.167 | 0.533 | 0.233 | 0.254 (35/138) (incl. stored distractors) | 0.58 |
| eps:T2:W4E24 | 0.744 | 0.558 | 0.82 | 0.772 | 0.935 | 0.717 | 0.0 (0/46) | 0.0 |
| eps:T2@heldout_combined:W2E12 | 0.028 | 0.019 | 0.031 | 0.083 | 0.133 | 0.028 | 0.114 (20/175) | 0.337 |
| eps:ORACLE:W2E12 | 0.889 | 0.615 | 1.0 | 0.889 | 1.0 | 0.861 | 0.0 (0/20) | 0.0 |
| eps:ORACLE:W4E24 | 0.928 | 0.75 | 1.0 | 0.928 | 1.0 | 0.906 | 0.0 (0/13) | 0.0 |
| eps:IGNORE_STATUS:W2E12 | 0.35 | 0.231 | 0.398 | 0.889 | 0.394 | 0.339 | 0.906 (106/117) | 0.906 |
| fixed:ORACLE | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | None (0/0) (incl. stored distractors) | None |
| fixed:IGNORE_STATUS | 0.494 | 0.673 | 0.422 | 1.0 | 0.494 | 0.494 | 1.0 (91/91) (incl. stored distractors) | 1.0 |

## Success by optimal depth d*

| arm / config | d*=2 | d*=3 | d*=4 | d*=5 | d*=6 |
|---|---:|---:|---:|---:|---:|
| eps:T2:W2E12 | 0.833 | 0.778 | 0.667 | 0.667 | 0.583 |
| eps:OFF:W2E12 | 0.139 | 0.111 | 0.056 | 0.028 | 0.056 |
| fixed:T2 | 0.917 | 0.889 | 0.806 | 0.917 | 0.806 |
| fixed:OFF | 0.222 | 0.306 | 0.194 | 0.139 | 0.306 |
| eps:T2:W4E24 | 0.889 | 0.833 | 0.667 | 0.694 | 0.639 |
| eps:T2@heldout_combined:W2E12 | 0.056 | 0.056 | 0.028 | 0.0 | 0.0 |
| eps:ORACLE:W2E12 | 0.944 | 0.972 | 0.833 | 0.833 | 0.861 |
| eps:ORACLE:W4E24 | 0.972 | 1.0 | 0.889 | 0.889 | 0.889 |
| eps:IGNORE_STATUS:W2E12 | 0.333 | 0.389 | 0.306 | 0.361 | 0.361 |
| fixed:ORACLE | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| fixed:IGNORE_STATUS | 0.361 | 0.444 | 0.611 | 0.583 | 0.472 |

## Co-primary discordant pairs by d* (report-only)

| contrast | d*=2 (n; T2 only / OFF only) | d*=3 (n; T2 only / OFF only) | d*=4 (n; T2 only / OFF only) | d*=5 (n; T2 only / OFF only) | d*=6 (n; T2 only / OFF only) |
|---|---|---|---|---|---|
| co-primary 1 (all) | 36; 25 / 0 | 36; 24 / 0 | 36; 22 / 0 | 36; 23 / 0 | 36; 19 / 0 |
| co-primary 2 (needs lookahead) | 3; 0 / 0 | 5; 4 / 0 | 16; 6 / 0 | 11; 5 / 0 | 17; 5 / 0 |

## Cost (distinct F keys per instance; tokens re-encoded from the answer texts; refused keys)

| config | keys mean | keys p90 | keys max | tokens mean | refused unique | refused share | instances with a refused key |
|---|---:|---:|---:|---:|---:|---:|---:|
| T2:W2E12 | 69.333 | 104 | 104 | 1371.922 | 0 | 0.0 | 0 |
| OFF:W2E12 | 90.489 | 104 | 104 | 1794.439 | 0 | 0.0 | 0 |
| T2:fixed | 22.578 | 33 | 42 | 447.628 | 0 | 0.0 | 0 |
| OFF:fixed | 22.667 | 33 | 41 | 449.772 | 0 | 0.0 | 0 |
| T2:W4E24 | 124.444 | 200 | 200 | 2465.039 | 0 | 0.0 | 0 |
| T2@heldout_combined:W2E12 | 100.356 | 104 | 104 | 2067.633 | 0 | 0.0 | 0 |
| ORACLE:W2E12 | 61.067 | 104 | 104 | 0.0 | 0 | 0.0 | 0 |
| ORACLE:W4E24 | 110.222 | 168 | 200 | 0.0 | 0 | 0.0 | 0 |
| IGNORE_STATUS:W2E12 | 61.067 | 104 | 104 | 0.0 | 0 | 0.0 | 0 |
| ORACLE:fixed | 22.65 | 33 | 42 | 0.0 | 0 | 0.0 | 0 |
| IGNORE_STATUS:fixed | 22.578 | 32 | 39 | 0.0 | 0 | 0.0 | 0 |

## Instrument and deviations (report-only)

- Terminal record: done.
- A0 anchor: {"pass": true, "off_states": 1.0, "on_states": 1.0, "off_texts": 1.0, "on_texts": 1.0}.
- A0 batch invariance: {"pass": true, "T2_states": 1.0, "OFF_states": 1.0, "T2_texts": 1.0, "OFF_texts": 1.0}.
- Turn 0: stages run ['A0', 'G1', 'G2', 'G3', 'G4'], deferred [], no progress False, cap 4.5 h, device cuda; A0 re-check null; batch-invariance re-check null.
- Orphaned stages (moved to _orphans/, regenerated in full): [].
- Stage attempts: {"A0": [0], "G1": [0], "G2": [0], "G3": [0], "G4": [0], "C1": ["cpu"]}; abandoned: {}.
- Failed A0 re-check records: [].
- BWM modules differing from the freeze record (report-only; only blocks_planner.py binds): [].

Decision record: status 1 PASS, status 2 PASS, near False.

Inputs: {"code_sha256": {"scripts/lbeam_analysis.py": "e3b0e9e0baaf23f183218fda57c1e4855d6db80deb9ca9db5a51d3753dafbf7e", "scripts/lbeam_run.py": "55a0bd0fb8e2af0112b8485ae5f8fc858c56508f4293d456bc24802d0c1efa5c", "scripts/lbeam_problems.py": "2353e2fc846fac5ced11cfc5420bd60b6e9e50cf3399e9299974b3921a09465e", "scripts/lbeam_f.py": "9a76562c2739df3254a650154acde41fe2052173b2f35556e5b75a7728680b6f"}, "numpy": "2.4.6", "torch_imported": false, "seconds": 2.5, "freeze_prereg_sha256": "0f414cbc59dfa45af4e1a42f3284ba1ebe87110c147a59229e1c96ec32cf0286"}

