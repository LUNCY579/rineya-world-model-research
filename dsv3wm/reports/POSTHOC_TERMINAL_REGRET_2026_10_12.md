# Posthoc terminal-reward audit of preserved native planning episodes

All60 published episodes have complete recorded execution labels. Under the
preexisting terminal source reward, none terminates at its goal. The original
goal-only P2 result remains59/60 across the five arm panels. This is a posthoc
contract diagnostic of previously executed native episodes, not new model
execution, a replacement endpoint or confirmation.

| Arm | Original goal-only success | Terminal goal success | Executed actions | Mean source return | Mean bound-normalized regret |
|---|---:|---:|---:|---:|---:|
| B0 |12/12|0/12|50|-0.041667|0.616940|
| A0 |12/12|0/12|50|-0.041667|0.616940|
| A1 |12/12|0/12|50|-0.041667|0.616940|
| A2 |12/12|0/12|50|-0.041667|0.616940|
| A5 |11/12|0/12|49|-0.040833|0.616422|

The source reward is
`1[terminated at goal] - .01*executed actions - .1*illegal executed actions`.
Finish counts as an action; an absorbing suffix is ignored by this source
episode contract. None of the60 observed episodes has an ignored or illegal
executed action. Their already verified optimal nonfinish distances are3–5,
so the optimal terminal return is `1-.01*(distance+1)` within the six-action cap.
The new normalization divides regret by `optimal_return+.66`; -.66 is a
conservative universal lower bound, not each parent's attainable worst policy.

The planner originally selected for reaching a goal state, without requiring
finish. Most selected successes stop at the goal without terminating. A5's
single failed choice ends with finish at the wrong goal and uses one fewer
action. Its slightly higher return is the resulting lower step penalty; it is
not an improved planning mechanism. That episode's original null move cost and
success-conditional regret remain null. The new calculation measures its three
recorded executed actions rather than imputing either original nullable field.

For A5 minus A0, the all-case mean return difference is+0.000833 and normalized
regret difference is-0.000518. All12 matched parents, including the original
failure, enter the descriptive2000-repetition parent bootstrap. The95% intervals
are[0,+0.002500] for return and[-0.001553,0] for normalized regret. This one-seed
development audit estimates no training-seed uncertainty, proves no equivalence
and supplies no confirmatory hypothesis family. H3's legal complete plan ending
at final finish remains a stricter contract than source episodes with ignored
suffixes, and no H3 student outcome follows from this audit.

## Reproducible evidence

`analyze_recovery_terminal_regret.py` verifies original task/config/adapter/run/
source/freeze/trace identities, learned-trace ancestry and recorded verifier
flags. It uses recorded labels only, calls no model/transition/solver and never
appends or repairs a finish. Null/malformed/incomplete choices are refused;
recorded empty execution is handled separately. Nine targeted pure tests passed.

The explicitly posthoc55-member source/input manifest is
`prereg/POSTHOC_TERMINAL_REWARD_REGRET_FREEZE_V1.json`, SHA256
`68f55ae0bdb05ff08499406a5b99f44abe483677a280821bec4f235cf67671b9`.
The analyst had already inspected the outcomes before declaring this new
normalization; it is not preregistered native confirmation. The protocol is
[POSTHOC_TERMINAL_REWARD_REGRET_V1.md](../prereg/POSTHOC_TERMINAL_REWARD_REGRET_V1.md).
Result `results/recovery-v1/POSTHOC_TERMINAL_REGRET_01.json` SHA256:
`a542405e38e35d27228f06c6a98950797a9a6ffdc97ff4bc94e25f3a03dd469f`.
`PREPARATION_CHECKPOINT_18.json` binds the passing test log, preserved inputs,
60/60 coverage and independent algebra cross-check. No original native output,
choice, success predicate, cost field, gate or sealed bank was changed.

From the isolated checkout, reproduce only at a fresh output path:

```powershell
$nativePython='C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
& $nativePython dsv3wm/scripts/analyze_recovery_terminal_regret.py --tasks dsv3wm/datasets/recovery-v1/planning_dev.jsonl --arm B0=dsv3wm/results/recovery-v1/B0_planning.json --arm A0=dsv3wm/results/recovery-v1/A0_planning.json --arm A1=dsv3wm/results/recovery-v1/A1_planning.json --arm A2=dsv3wm/results/recovery-v1/A2_planning.json --arm A5=dsv3wm/results/recovery-v1/A5_planning.json --out dsv3wm/results/recovery-v1/POSTHOC_TERMINAL_REGRET_REPLICA_01.json
```

The broader research remains incomplete: student readouts, domain qualification,
native selected controls/calibration, qualifying seeds and fresh co-primary
confirmation still need their registered execution conditions.
