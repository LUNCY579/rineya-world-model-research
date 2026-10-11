# RCX-E01N native analysis (DEV battery sha `0f4b263c008dcbde…`, 2240 records)

**Tiers:** OFF and SUPA are tier A (frozen cores). T2 and N-* are trained students (component adapters on frozen FP32 Titan-Sparse, unmerged). **DEV, descriptive.** Confirmation requires the frozen PREREG_E01N on a SEALED battery, 3 seeds, and E33 retention.

## Summary

| arm | composite in | composite ho_tpl | decision in | no-op when required (varied, in) | V3 sensitivity | V3 both-twins | verdict (F strata, in) |
|---|---|---|---|---|---|---|---|
| OFF | 0.325 | 0.308 | 0.500 | 1/280 [0.001, 0.020] | 0.000 | 0.000 | never no-op (ignores the rule) |
| T2 | 0.400 | 0.393 | 0.502 | 1/280 [0.001, 0.020] | 0.000 | 0.000 | TRIGGER (trigger widened: near-misses) |
| N-VR_s0 | 0.964 | 0.594 | 0.966 | 254/280 [0.867, 0.936] | 1.000 | 0.975 | partial |

## Per-stratum exact / decision (surface `in`)

| arm | F1 | F2 | F3 | F4 | F5 | V1 | V2 | V3 | V5 | V6 | V7 | M1 | M2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OFF | 0.31/0.50 | 0.35/0.50 | 0.36/0.50 | 0.29/0.50 | 0.31/0.50 | 0.40/0.50 | 0.26/0.50 | 0.35/0.50 | 0.24/0.50 | 0.29/0.50 | 0.33/0.50 | 0.31/0.49 | 0.12/0.50 |
| T2 | 0.94/1.00 | 0.39/0.50 | 0.40/0.50 | 0.85/0.99 | 0.42/0.50 | 0.39/0.50 | 0.41/0.51 | 0.39/0.50 | 0.40/0.50 | 0.34/0.49 | 0.42/0.50 | 0.33/0.50 | 0.25/0.53 |
| N-VR_s0 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 0.99/1.00 | 0.79/0.79 | 0.99/0.99 | 1.00/1.00 | 0.99/1.00 | 0.99/0.99 | 0.68/0.68 | 0.93/0.93 | 0.85/0.97 | 0.72/0.91 |

## Paired contrasts on the primary composite (parent-clustered bootstrap; descriptive on DEV)

| A | B | contrast | surface | A | B | A−B | 95% CI | n parents |
|---|---|---|---|---|---|---|---|---|
| N-VR | OFF | training effect (C - A), ordinary | in | 0.964 | 0.325 | +0.639 | [+0.608, +0.671] | 280 |
| N-VR | OFF | training effect (C - A), ordinary | ho_tpl | 0.594 | 0.308 | +0.286 | [+0.251, +0.321] | 280 |
