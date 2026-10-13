# Item 7 addendum A: report-only metrics

Nothing here is a test; no number changes item 7's outcome.

| system | beam solved | verified-but-invalid | dead_end | budget | stalled at 1 | FREPLAY-R2 | R1 [L, U] |
|---|---|---|---|---|---|---|---|
| R0 | 27 | 0 | 8 | 5 | 12 | 0.21826 | [27, 27] |
| R1 | 27 | 0 | 2 | 11 | 10 | 0.156002 | [26, 27] |
| R2 | 24 | 0 | 11 | 5 | 5 | 0.001969 | [11, 39] |

| system | beam FLOPs/instance (F exact + proposer ESTIMATE) | direct FLOPs/instance | ratio |
|---|---|---|---|
| R0 | 4.795e+13 | 3.19e+11 | 150 |
| R1 | 5.455e+13 | 3.097e+11 | 176 |
| R2 | 4.139e+13 | 3.061e+11 | 135 |
