# RCX-E01R step A: DEV2 diagnostic on fresh blind-authored wordings (descriptive)

Battery: seed 20261261, sha256 `38362338b7716af4acd647c0cd426f8bca28cfe8aca4e8de7ed69ebf94a1da4f`, 1320 records ({'in': 660, 'p2': 660}), 45 P2 templates used; P2 sha256 `7e625a590004798e7ddd3b2828cd8f9009744abf14c996774a2ad890ccb7f64e`.
Arms: N-CFSD_s0, N-CFSD_s1, N-CFSD_s2, N-VR_s0. Readings fixed in DESIGN.md section 3 before the run.

| arm | surface | polarity | form | no-op items decided no-op | apply items decided apply | exact |
|---|---|---|---|---|---|---|
| N-CFSD (s0-s2 pooled) | in | POS | - | 668/672 = 0.99 [0.98, 1.00] | 748/762 = 0.98 [0.97, 0.99] | 1397/1434 = 0.97 [0.96, 0.98] |
| N-CFSD (s0-s2 pooled) | in | NEG | - | 318/318 = 1.00 [0.99, 1.00] | 218/228 = 0.96 [0.92, 0.98] | 534/546 = 0.98 [0.96, 0.99] |
| N-CFSD (s0-s2 pooled) | p2 | POS | EXPLICIT_LISTED_FIRST | 113/222 = 0.51 [0.44, 0.57] | 245/252 = 0.97 [0.94, 0.99] | 350/474 = 0.74 [0.70, 0.78] |
| N-CFSD (s0-s2 pooled) | p2 | POS | EXPLICIT_COMPLEMENT_FIRST | 80/231 = 0.35 [0.29, 0.41] | 214/261 = 0.82 [0.77, 0.86] | 293/492 = 0.60 [0.55, 0.64] |
| N-CFSD (s0-s2 pooled) | p2 | POS | IMPLICIT | 140/219 = 0.64 [0.57, 0.70] | 230/249 = 0.92 [0.88, 0.95] | 366/468 = 0.78 [0.74, 0.82] |
| N-CFSD (s0-s2 pooled) | p2 | NEG | EXPLICIT_LISTED_FIRST | 99/108 = 0.92 [0.85, 0.96] | 58/78 = 0.74 [0.64, 0.83] | 157/186 = 0.84 [0.79, 0.89] |
| N-CFSD (s0-s2 pooled) | p2 | NEG | EXPLICIT_COMPLEMENT_FIRST | 74/99 = 0.75 [0.65, 0.82] | 43/69 = 0.62 [0.51, 0.73] | 117/168 = 0.70 [0.62, 0.76] |
| N-CFSD (s0-s2 pooled) | p2 | NEG | IMPLICIT | 74/111 = 0.67 [0.57, 0.75] | 54/81 = 0.67 [0.56, 0.76] | 128/192 = 0.67 [0.60, 0.73] |
| N-VR s0 | in | POS | - | 210/224 = 0.94 [0.90, 0.96] | 254/254 = 1.00 [0.99, 1.00] | 463/478 = 0.97 [0.95, 0.98] |
| N-VR s0 | in | NEG | - | 106/106 = 1.00 [0.97, 1.00] | 75/76 = 0.99 [0.93, 1.00] | 181/182 = 0.99 [0.97, 1.00] |
| N-VR s0 | p2 | POS | EXPLICIT_LISTED_FIRST | 20/74 = 0.27 [0.18, 0.38] | 84/84 = 1.00 [0.96, 1.00] | 102/158 = 0.65 [0.57, 0.72] |
| N-VR s0 | p2 | POS | EXPLICIT_COMPLEMENT_FIRST | 5/77 = 0.06 [0.03, 0.14] | 87/87 = 1.00 [0.96, 1.00] | 92/164 = 0.56 [0.48, 0.63] |
| N-VR s0 | p2 | POS | IMPLICIT | 15/73 = 0.21 [0.13, 0.31] | 83/83 = 1.00 [0.96, 1.00] | 97/156 = 0.62 [0.54, 0.69] |
| N-VR s0 | p2 | NEG | EXPLICIT_LISTED_FIRST | 21/36 = 0.58 [0.42, 0.73] | 24/26 = 0.92 [0.76, 0.98] | 45/62 = 0.73 [0.60, 0.82] |
| N-VR s0 | p2 | NEG | EXPLICIT_COMPLEMENT_FIRST | 12/33 = 0.36 [0.22, 0.53] | 16/23 = 0.70 [0.49, 0.84] | 28/56 = 0.50 [0.37, 0.63] |
| N-VR s0 | p2 | NEG | IMPLICIT | 10/37 = 0.27 [0.15, 0.43] | 21/27 = 0.78 [0.59, 0.89] | 31/64 = 0.48 [0.37, 0.60] |

## POS - NEG no-op accuracy within form (`p2`; unpaired parent bootstrap, descriptive)

| arm | form | diff | 95% CI | parents POS / NEG |
|---|---|---|---|---|
| N-CFSD (s0-s2 pooled) | EXPLICIT_LISTED_FIRST | -0.408 | [-0.525, -0.285] | 74 / 36 |
| N-CFSD (s0-s2 pooled) | EXPLICIT_COMPLEMENT_FIRST | -0.401 | [-0.545, -0.253] | 77 / 33 |
| N-CFSD (s0-s2 pooled) | IMPLICIT | -0.027 | [-0.181, +0.127] | 73 / 37 |
| N-VR s0 | EXPLICIT_LISTED_FIRST | -0.313 | [-0.493, -0.120] | 74 / 36 |
| N-VR s0 | EXPLICIT_COMPLEMENT_FIRST | -0.299 | [-0.481, -0.130] | 77 / 33 |
| N-VR s0 | IMPLICIT | -0.065 | [-0.241, +0.098] | 73 / 37 |

**Pre-stated reading (DESIGN.md section 3), N-CFSD pooled:** intermediate. POS no-op by form: EXPLICIT_LISTED_FIRST 0.51, EXPLICIT_COMPLEMENT_FIRST 0.35, IMPLICIT 0.64; NEG no-op by form: EXPLICIT_LISTED_FIRST 0.92, EXPLICIT_COMPLEMENT_FIRST 0.75, IMPLICIT 0.67.
