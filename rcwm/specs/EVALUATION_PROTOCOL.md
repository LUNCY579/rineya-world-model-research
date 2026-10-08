# Evaluation protocol

Primary comparison: learned consequence evaluator B4 versus a matched scalar trajectory evaluator B2.
Both see the same information and candidate sets; both have comparable parameter, tuning, pretraining
and inference budgets. Include measured-feature B3 and exact candidate oracle O1 to localize failures.

Primary independent unit: parent problem family. Report actual task success/utility and candidate
regret. Group correlated branches/goals within the parent. Distinguish conditional-on-fixed-model
parent bootstrap uncertainty from training-seed uncertainty. Do not treat tokens as independent trials.
The supplied bootstrap helper is a primitive, not a complete scientific certification routine.

Secondary endpoints: feature accuracy, success-probability Brier/log score, calibration by operator,
OOD rejection coverage, label budget curves, long-distance/horizon sweeps, DAG makespan/conflicts,
end-to-end latency/cost, protected skill retention and capability acquisition.

Before opening FINAL, fill `configs/evaluation_gates.yaml`: endpoint, meaningful effect, retention NI
margin, power-derived parent count, final seeds and maximum cost. Null fields BLOCK promotion.
Predeclare one primary contrast and label exploratory multiple comparisons. Keep every selected
candidate and actual outcome, including failures. Do not retrain on sealed failures and call the split
unchanged.

Stress search width: as candidates increase, measure predicted utility and REAL utility separately.
A widening gap indicates exploitation [S26]. Test action-shuffled and future-leak controls. Preserve
rare constraint violations separately; aggregate reward cannot cancel them.

A no-harm claim needs a noninferiority interval, not a nonsignificant ordinary difference. A frozen
weight hash alone certifies weight preservation, not preservation of behavior under a new planner.
