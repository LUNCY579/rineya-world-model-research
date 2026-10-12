# Conditional legality calibration: prepared, no native result

The additive diagnostic is implemented, independently reviewed, tested and
prospectively frozen. No learned probabilities, Brier score, reliability estimate
or native calibration outcome have been measured. The pending H3 student
readouts and independent domain qualification retain their scientific gates.

## Qualified preparation

- `scripts/evaluate_recovery_legal_calibration.py` uses the original24 DEV
  parents and causal initial h1 prompt plus a fixed legal-first JSON prefix.
  It scores two comma-terminated boolean token paths with EOS excluded. Native
  model/head computations remain FP32; probability reductions use float64.
- It preserves raw path likelihoods, canonical-path mass, conditional legality
  probability, other-path mass and underflow information. Other mass includes
  valid alternative serialization; it is not invalid-JSON probability.
  All captures precede label scoring. Ten fixed bins, Brier loss, clipped
  reporting loss and separate unclipped conditional log loss are preregistered.
- Partial path scores and actual counter deltas are persisted incrementally.
  A later operational interruption flushes the attempted parent before
  rethrowing. Failed cases retain null metrics and explicit denominators;
  completed attempts and successful probability captures are separate.
- Final18 semantic tests passed in1.867s; evidence is
  `results/recovery-v1/TESTS_LEGAL_CALIBRATION_05.txt`. Earlier logs01–04 remain
  preserved. Logs01 and04 contain test-only assertion/error failures, corrected before
  the final pass; neither records native model execution. The earlier full
  checkpoint14's290 tests remains a separate historical suite result. No new
  full-suite count is inferred by adding these counts.
- Tokenizer-only audit02 checked48 paths using the actual SHA-bound tokenizer
  and unchanged native `Encoded`/`encode_pair` AST nodes, without importing
  torch. All boundaries, masks, special-token exclusion and prefix-free paths
  passed; maximum full scored sequence length is124 tokens. This is input
  geometry evidence, not a probability or model measurement.
- Final lineage audit02 verified actual warm A0/A1/A2 at32 updates and A5 at96,
  each against its real source/config/weight/run/qualification history. The
  registered freeze binds72 source/data/artifact/core/test/evidence files.
  Audits01 remain historical before the partial-path persistence repair.

Driver SHA256:
`aa31f01862809aeb1975eaa0d1a1c943db70f785c39ba0050a71072af09476c2`.
`prereg/RECOVERY_LEGAL_CALIBRATION_FREEZE_V1_CPU11.json` SHA256:
`6d6fb87188b50a2ccbad5ada16cc1b3f24954d530f3765fd90eaec2690148abd`.
`PREPARATION_CHECKPOINT_16.json` records the final evidence bindings and absence
of model/probability/confirmation access. The registered contract is
[RECOVERY_LEGAL_CALIBRATION_V1.md](../prereg/RECOVERY_LEGAL_CALIBRATION_V1.md).

## Native queue and interpretation

This measures legality conditional on a forced canonical response prefix.
It changes response context relative to ordinary generation and cannot certify
calibrated state correctness, planning confidence, deployment confidence or
broad retention. The fixed cohort has16 legal/8 illegal initial actions and
12 inherited-F-known/12 unseen-limit parents. No thresholds or parents will be
selected from diagnostic probabilities.

Ordinary CPU11 admission requires both physical and commit availability of at
least11GiB after imports; the four-GiB work boundary and shared lane are unchanged.
The H3 training/singleton CPU10.5 profiles do not authorize this diagnostic.
Do not attempt loads while the actual floor is unmet or overlap native jobs.

After resource admission, each artifact can be evaluated once at its new
exclusive destination. The registered freeze already binds the original
selected source/artifact/input bytes; replay requires those verified locations
and refuses changed members. Example from the isolated checkout:

```powershell
$env:BWM_DEVICE='cpu'
$env:PYTHONPATH='dsv3wm/src'
$nativePython='C:\Users\clclc\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe'
$taskCore='C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze'
foreach ($arm in @('A0','A1','A2','A5')) {
  & $nativePython dsv3wm/scripts/run_measured.py --out "dsv3wm/results/recovery-v1/warm_${arm}_legal_calibration_cpu11.resources.json" -- dsv3wm/scripts/evaluate_recovery_legal_calibration.py --config "dsv3wm/configs/recovery_${arm}.json" --adapter "dsv3wm/artifacts/recovery-v1/${arm}_s0/adapter" --run "dsv3wm/artifacts/recovery-v1/${arm}_s0/run.json" --data dsv3wm/datasets/recovery-v1/dev.jsonl --train dsv3wm/datasets/recovery-v1/train.jsonl --core $taskCore --freeze dsv3wm/prereg/RECOVERY_LEGAL_CALIBRATION_FREEZE_V1_CPU11.json --out "dsv3wm/results/recovery-v1/warm_${arm}_legal_calibration_cpu11.json"
  if ($LASTEXITCODE -ne 0) { throw 'Stop and preserve the failed calibration attempt' }
}
```

These are pending commands, not executed results. Complete capture would attempt
48 literal-scoring paths per artifact (192 across the four-arm example); actual
forward/token/memory/time work must be measured.
Scoring forwards are separate from generation requests and planning queries.
GPU allocation and physical FLOPs remain unmeasured. No confirmation access,
model release, merge, dense-core change or promotion follows from preparation.
