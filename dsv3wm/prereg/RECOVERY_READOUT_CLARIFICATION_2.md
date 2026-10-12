# Recovery readout provenance clarification 2

Recorded before any adapted recovery training, B0 planning, behavioral retention,
or teacher collection. `RECOVERY_B0_READOUT_FREEZE_V1.json` was generated but
never executed. Review found its freezer compared only adapter rank, alpha,
dropout and targets; an altered `use_rslora` field could therefore change scaling
while retaining the trained weight hash. A pure tamper test reproduced this gap.

The corrected freezer requires equality of every behavior field with the initial
registered F configuration. Only `base_model_name_or_path`, `revision`,
`peft_version` and `inference_mode` are treated as metadata: the parent checkpoint
is independently SHA-bound, the actual runtime is recorded, and evaluation always
uses eval mode. Unknown behavior fields, scaling/DoRA changes, rank/alpha patterns,
bias changes and target changes are refused. Target-module list order is normalized
because PEFT stores this collection as an unordered set. Other values use strict
canonical JSON comparison, including boolean versus integer types. No training, generation or metric
mathematics is changed. The existing V2 training freeze remains valid.

Keep the unexecuted V1 readout freeze and original freezer bytes under its
SHA-named source snapshot. Create new prospective readout freezes after this
clarification, starting with B0 V2. The TRAIN-only teacher freeze is unaffected.
The archived V1 is provenance, not a current executable qualification.
