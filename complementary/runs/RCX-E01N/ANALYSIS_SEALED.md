# RCX-E01N SEALED confirmation (PREREG_E01N)

**DECISION: REVISE** (applied mechanically by `analyze_native.decide`; PREREG_E01N section 5)

## Inputs

- SEALED battery: seed 88527345, sha256 `c87765eaf275554946bc92513359cecfcc04c5eb16bf0b30d7ec80f33256df48`, 2240 records (surfaces in, ho_tpl; 40 parents per stratum). Dropped train-overlap parents: 0; records analysed: 2240.
- Commitment sha256(salt) `8e5e08f1b7dc7a4fb39e33a874c1b5deba28b6074389f3de3be52c6d50943765`, minted 2026-10-11 01:54:24 Central Daylight Time, opened 2026-10-11 08:40:45 Central Daylight Time.
- Cells (each `eval_sealed_<tag>.json` names this battery's sha256; N-* adapter shas equal open-sealed's record; every score re-derived from the raw output): N-VR_s0, N-VR_s1, N-VR_s2, N-CFSD_s0, N-CFSD_s1, N-CFSD_s2, OFF, T2.
- Method: seed × parent paired cluster bootstrap, 10000 resamples, seed 20261236 (variants averaged within parent per seed, seeds averaged, parents resampled; inference conditional on these adapters). Two-sided p = 2·min(share ≤ 0, share ≥ 0) of the replicate differences, floored at 0.0001, capped at 1; Holm over C1, C2 at α = 0.05. CIs are 95% percentile intervals, not multiplicity-adjusted.

## Confirmatory: N-CFSD − N-VR

| endpoint | surface | N-VR | N-CFSD | diff | 95% CI | p | p (Holm) | per-seed diff | parents | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | in | 0.963 | 0.981 | +0.018 | [+0.006, +0.030] | 0.0020 | 0.0020 | s0 +0.015, s1 +0.002, s2 +0.036 | 280 | 3 / 3 |
| C2 | ho_tpl | 0.591 | 0.681 | +0.090 | [+0.069, +0.113] | 0.0001 | 0.0002 | s0 +0.095, s1 +0.121, s2 +0.055 | 280 | 3 / 3 |

## Protected strata: N-CFSD − N-VR (basis: ci)

| stratum | N-VR | N-CFSD | diff | 95% CI | parents | rule | within margin |
|---|---|---|---|---|---|---|---|
| F0 posted twins, exact (`in`) | 0.980 | 0.973 | -0.007 | [-0.025, +0.013] | 200 | 95% CI lower bound > -0.05 | yes |
| format-error rate (all records) | 0.001 | 0.000 | -0.000 | [-0.001, +0.000] | 520 | 95% CI upper bound <= +0.02 | yes |

## Decision trace (outcomes tested in the order ADVANCE, STOP, REVISE; the first that holds decides)

| endpoint | Holm p < 0.05 | point ≥ +0.10 | CI lo > 0 | every per-seed diff > 0 | other CI lo > -0.05 | point > 0 | CI hi < +0.05 |
|---|---|---|---|---|---|---|---|
| C1 | yes | no | yes | yes | yes | yes | yes |
| C2 | yes | no | yes | yes | yes | yes | no |

- All confirmatory cells analysed: yes.
- Protected strata within margin: yes.
- ADVANCE: no.
- STOP: no.
- REVISE: yes.
- **Decision: REVISE**

## Secondary (descriptive; no test)

### Arms (seed means; per-seed values; spread)

| arm | cells | composite `in` | per seed | composite `ho_tpl` | per seed | SD in / ho_tpl | no-op when required, `in` (pooled, Wilson) | no-op when required, `ho_tpl` |
|---|---|---|---|---|---|---|---|---|
| OFF | 1 | 0.336 | 0.336 | 0.319 | 0.319 | — / — | 4/280 [0.006, 0.036] | 4/280 [0.006, 0.036] |
| T2 | 1 | 0.367 | 0.367 | 0.364 | 0.364 | — / — | 0/280 [0.000, 0.014] | 1/280 [0.001, 0.020] |
| N-VR | 3 | 0.963 | 0.968, 0.971, 0.952 | 0.591 | 0.591, 0.581, 0.601 | 0.010 / 0.010 | 782/840 [0.912, 0.946] | 260/840 [0.279, 0.342] |
| N-CFSD | 3 | 0.981 | 0.983, 0.972, 0.987 | 0.681 | 0.686, 0.702, 0.656 | 0.008 / 0.023 | 817/840 [0.959, 0.982] | 535/840 [0.604, 0.669] |

### Cells

| cell | composite in / ho_tpl | decision in / ho_tpl | no-op when required, `in` | V3 sensitivity in / ho_tpl | V3 both-twins in / ho_tpl | PREREG-3-style verdict `in` | verdict `ho_tpl` |
|---|---|---|---|---|---|---|---|
| OFF | 0.336 / 0.319 | 0.499 / 0.499 | 4/280 [0.006, 0.036] | 0.000 / 0.000 | 0.000 / 0.000 | never no-op (ignores the rule) | never no-op (ignores the rule) |
| T2 | 0.367 / 0.364 | 0.500 / 0.500 | 0/280 [0.000, 0.014] | 0.000 / 0.000 | 0.000 / 0.000 | TRIGGER (trigger widened: near-misses) | TRIGGER (trigger widened: near-misses) |
| N-VR_s0 | 0.968 / 0.591 | 0.971 / 0.595 | 256/280 [0.876, 0.942] | 1.000 / 0.525 | 0.975 / 0.525 | partial | never no-op (ignores the rule) |
| N-VR_s1 | 0.971 / 0.581 | 0.976 / 0.587 | 266/280 [0.918, 0.970] | 1.000 / 0.500 | 1.000 / 0.500 | partial | never no-op (ignores the rule) |
| N-VR_s2 | 0.952 / 0.601 | 0.959 / 0.606 | 260/280 [0.892, 0.953] | 1.000 / 0.550 | 0.975 / 0.550 | partial | never no-op (ignores the rule) |
| N-CFSD_s0 | 0.983 / 0.686 | 0.990 / 0.696 | 275/280 [0.959, 0.992] | 0.975 / 0.950 | 0.950 / 0.900 | generalises to unseen statuses (RULE-like) | partial |
| N-CFSD_s1 | 0.972 / 0.702 | 0.979 / 0.709 | 268/280 [0.927, 0.975] | 0.950 / 0.975 | 0.900 / 0.925 | generalises to unseen statuses (RULE-like) | never no-op (ignores the rule) |
| N-CFSD_s2 | 0.987 / 0.656 | 0.989 / 0.662 | 274/280 [0.954, 0.990] | 1.000 / 0.950 | 1.000 / 0.900 | generalises to unseen statuses (RULE-like) | never no-op (ignores the rule) |

### Matched seeds: N-CFSD_s − N-VR_s (composite exact, parent macro)

| seed | in | ho_tpl |
|---|---|---|
| s0 | +0.015 | +0.095 |
| s1 | +0.002 | +0.121 |
| s2 | +0.036 | +0.055 |
| t interval (df = 2) | +0.018 [-0.025, +0.060] | +0.090 [+0.009, +0.172] |

### Seed-0 sensitivity: C1/C2 on seeds 1, 2 only (descriptive; p unadjusted)

| endpoint | surface | N-VR | N-CFSD | diff | 95% CI | p | seeds | point ≥ +0.10 and CI lo > 0 |
|---|---|---|---|---|---|---|---|---|
| C1 | in | 0.961 | 0.980 | +0.019 | [+0.006, +0.033] | 0.0048 | 2 / 2 | no |
| C2 | ho_tpl | 0.591 | 0.679 | +0.088 | [+0.066, +0.111] | 0.0001 | 2 / 2 | no |

### Reference contrasts (composite exact; seed × parent bootstrap; descriptive CI)

| contrast | surface | A | B | B − A | 95% CI | parents |
|---|---|---|---|---|---|---|
| N-CFSD - T2 | in | 0.367 | 0.981 | +0.614 | [+0.588, +0.641] | 280 |
| N-CFSD - T2 | ho_tpl | 0.364 | 0.681 | +0.317 | [+0.280, +0.354] | 280 |
| N-VR - OFF | in | 0.336 | 0.963 | +0.628 | [+0.599, +0.656] | 280 |
| N-VR - OFF | ho_tpl | 0.319 | 0.591 | +0.272 | [+0.237, +0.307] | 280 |

### Per stratum, exact / decision (seed means), surface `in`

| arm | F1 | F2 | F3 | F4 | F5 | V1 | V2 | V3 | V5 | V6 | V7 | M1 | M2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OFF | 0.35/0.51 | 0.31/0.50 | 0.38/0.50 | 0.39/0.51 | 0.26/0.50 | 0.36/0.50 | 0.35/0.50 | 0.35/0.50 | 0.34/0.49 | 0.29/0.49 | 0.30/0.50 | 0.23/0.50 | 0.05/0.47 |
| T2 | 0.93/1.00 | 0.38/0.50 | 0.40/0.50 | 0.91/0.99 | 0.33/0.50 | 0.39/0.50 | 0.31/0.50 | 0.38/0.50 | 0.39/0.50 | 0.38/0.50 | 0.44/0.50 | 0.33/0.50 | 0.09/0.50 |
| N-VR | 1.00/1.00 | 0.98/0.98 | 0.97/0.98 | 0.99/1.00 | 0.83/0.84 | 0.99/0.99 | 0.99/0.99 | 0.99/1.00 | 0.99/0.99 | 0.75/0.75 | 1.00/1.00 | 0.85/0.97 | 0.64/0.92 |
| N-CFSD | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 0.98/0.99 | 0.93/0.95 | 0.99/1.00 | 0.98/0.98 | 0.97/0.99 | 1.00/1.00 | 0.81/0.81 | 0.98/0.99 | 0.83/0.99 | 0.62/0.91 |

### Per stratum, exact / decision (seed means), surface `ho_tpl`

| arm | F1 | F2 | F3 | F4 | F5 | V1 | V2 | V3 | V5 | V6 | V7 | M1 | M2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| OFF | 0.30/0.50 | 0.31/0.50 | 0.36/0.50 | 0.35/0.51 | 0.24/0.50 | 0.34/0.50 | 0.33/0.50 | 0.34/0.50 | 0.32/0.49 | 0.33/0.50 | 0.34/0.50 | 0.25/0.50 | 0.04/0.49 |
| T2 | 0.91/0.99 | 0.38/0.50 | 0.40/0.50 | 0.88/0.96 | 0.33/0.50 | 0.38/0.50 | 0.30/0.50 | 0.40/0.50 | 0.38/0.50 | 0.31/0.50 | 0.45/0.51 | 0.29/0.50 | 0.10/0.50 |
| N-VR | 0.51/0.51 | 0.50/0.50 | 0.50/0.50 | 0.50/0.50 | 0.48/0.50 | 0.50/0.50 | 0.77/0.77 | 0.75/0.76 | 0.64/0.64 | 0.70/0.70 | 0.55/0.55 | 0.48/0.61 | 0.32/0.58 |
| N-CFSD | 0.58/0.58 | 0.52/0.52 | 0.51/0.52 | 0.56/0.58 | 0.52/0.53 | 0.52/0.53 | 0.97/0.97 | 0.95/0.98 | 0.78/0.78 | 0.84/0.85 | 0.66/0.67 | 0.53/0.71 | 0.40/0.67 |

### Error directions (counts per cell)

| cell | surface | applied_when_noop | false_copy | arith | format |
|---|---|---|---|---|---|
| OFF | in | 549 | 7 | 215 | 6 |
| OFF | ho_tpl | 550 | 8 | 227 | 3 |
| T2 | in | 481 | 0 | 157 | 0 |
| T2 | ho_tpl | 483 | 0 | 168 | 0 |
| N-VR_s0 | in | 36 | 13 | 42 | 1 |
| N-VR_s0 | ho_tpl | 448 | 16 | 42 | 1 |
| N-VR_s1 | in | 26 | 9 | 33 | 1 |
| N-VR_s1 | ho_tpl | 447 | 12 | 30 | 3 |
| N-VR_s2 | in | 44 | 10 | 33 | 0 |
| N-VR_s2 | ho_tpl | 436 | 14 | 31 | 0 |
| N-CFSD_s0 | in | 5 | 22 | 38 | 0 |
| N-CFSD_s0 | ho_tpl | 302 | 21 | 43 | 0 |
| N-CFSD_s1 | in | 14 | 30 | 41 | 0 |
| N-CFSD_s1 | ho_tpl | 286 | 46 | 41 | 0 |
| N-CFSD_s2 | in | 10 | 15 | 43 | 1 |
| N-CFSD_s2 | ho_tpl | 350 | 35 | 46 | 1 |

### Held-out wording (`ho_tpl`, all strata, pooled over seeds): no-op items decided no-op · apply items decided apply

AND/OR compositions render both h0 and h1 with the h0 frame (`templates.render_rule`).

| wording | OFF | T2 | N-VR | N-CFSD |
|---|---|---|---|---|
| AND h0 | 0/4 · 4/4 | 0/4 · 4/4 | 8/12 · 12/12 | 12/12 · 12/12 |
| AND h1 | 0/7 · 7/7 | 0/7 · 7/7 | 17/21 · 21/21 | 21/21 · 20/21 |
| OR h0 | 0/10 · 9/10 | 0/10 · 10/10 | 5/30 · 30/30 | 27/30 · 22/30 |
| OR h1 | 2/19 · 18/19 | 0/19 · 19/19 | 22/57 · 54/57 | 46/57 · 44/57 |
| S+ h0 | 2/178 · 197/198 | 42/178 · 198/198 | 15/534 · 576/594 | 96/534 · 553/594 |
| S+ h1 | 3/173 · 189/193 | 34/173 · 193/193 | 9/519 · 573/579 | 50/519 · 567/579 |
| S- h0 | 0/84 · 64/64 | 0/84 · 64/64 | 228/252 · 177/192 | 243/252 · 171/192 |
| S- h1 | 1/85 · 63/65 | 1/85 · 65/65 | 44/255 · 192/195 | 247/255 · 188/195 |

### M1 / M2 magnitudes (seed means, exact / decision)

| arm | M1 `in` | M2 `in` | M1 `ho_tpl` | M2 `ho_tpl` |
|---|---|---|---|---|
| OFF | 0.23 / 0.50 | 0.05 / 0.47 | 0.25 / 0.50 | 0.04 / 0.49 |
| T2 | 0.33 / 0.50 | 0.09 / 0.50 | 0.29 / 0.50 | 0.10 / 0.50 |
| N-VR | 0.85 / 0.97 | 0.64 / 0.92 | 0.48 / 0.61 | 0.32 / 0.58 |
| N-CFSD | 0.83 / 0.99 | 0.62 / 0.91 | 0.53 / 0.71 | 0.40 / 0.67 |
