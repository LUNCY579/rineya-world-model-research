# PREREG-LB analysis dry run

ok: True

Draft sha256 `26c62144d89dde15e7d18ad119f9361ce6f93562bd3d358f26808e7aefbd788d`; hook draft sha256 `098cb1311bd9d5352b042a4e84dcabc606df6209fd9429a60ee4b543d27102ff`; registered sentences missing from the draft: 0.

## Wording branches

| case | want (branch, s1, s2, near) | got | truthful | ok |
|---|---|---|---|---|
| YES_NEAR | ['YES_NEAR', 'PASS', 'PASS', True] | ['YES_NEAR', 'PASS', 'PASS', True] | True | True |
| YES (ratio below 0.80) | ['YES', 'PASS', 'PASS', False] | ['YES', 'PASS', 'PASS', False] | True | True |
| GREEDY_ONLY_NEAR (lookahead NS) | ['GREEDY_ONLY_NEAR', 'PASS', 'NS', True] | ['GREEDY_ONLY_NEAR', 'PASS', 'NS', True] | True | True |
| GREEDY_ONLY (lookahead NS) | ['GREEDY_ONLY', 'PASS', 'NS', False] | ['GREEDY_ONLY', 'PASS', 'NS', False] | True | True |
| GREEDY_ONLY (lookahead SHORT) | ['GREEDY_ONLY', 'PASS', 'SHORT', False] | ['GREEDY_ONLY', 'PASS', 'SHORT', False] | True | True |
| LOOKAHEAD_ONLY (all SHORT) | ['LOOKAHEAD_ONLY', 'SHORT', 'PASS', False] | ['LOOKAHEAD_ONLY', 'SHORT', 'PASS', False] | True | True |
| LOOKAHEAD_ONLY (all NS, near) | ['LOOKAHEAD_ONLY', 'NS', 'PASS', True] | ['LOOKAHEAD_ONLY', 'NS', 'PASS', True] | True | True |
| DIRECTIONAL (both SHORT) | ['DIRECTIONAL', 'SHORT', 'SHORT', False] | ['DIRECTIONAL', 'SHORT', 'SHORT', False] | True | True |
| DIRECTIONAL (all SHORT, lookahead NS) | ['DIRECTIONAL', 'SHORT', 'NS', False] | ['DIRECTIONAL', 'SHORT', 'NS', False] | True | True |
| DIRECTIONAL (all NS, lookahead SHORT) | ['DIRECTIONAL', 'NS', 'SHORT', False] | ['DIRECTIONAL', 'NS', 'SHORT', False] | True | True |
| NO (both NS) | ['NO', 'NS', 'NS', False] | ['NO', 'NS', 'NS', False] | True | True |
| NO (both NS, ratio near: OFF also near the oracle) | ['NO', 'NS', 'NS', True] | ['NO', 'NS', 'NS', True] | True | True |
| REVERSED (all instances) | ['REVERSED', 'REVERSED', 'NS', False] | ['REVERSED', 'REVERSED', 'NS', False] | True | True |
| REVERSED_LOOKAHEAD (lookahead only) | ['REVERSED_LOOKAHEAD', 'NS', 'REVERSED', False] | ['REVERSED_LOOKAHEAD', 'NS', 'REVERSED', False] | True | True |
| REVERSED (both) | ['REVERSED', 'REVERSED', 'REVERSED', False] | ['REVERSED', 'REVERSED', 'REVERSED', False] | True | True |
| MIXED_PASS (reviewer's case: PASS overall, REVERSED where lookahead is needed, NEAR) | ['MIXED_PASS', 'PASS', 'REVERSED', True] | ['MIXED_PASS', 'PASS', 'REVERSED', True] | True | True |
| combination NS/NS/not near -> NO | ['NO', 'NS', 'NS', False] | ['NO', 'NS', 'NS', False] | True | True |
| combination NS/NS/NEAR -> NO | ['NO', 'NS', 'NS', True] | ['NO', 'NS', 'NS', True] | True | True |
| combination NS/PASS/not near -> LOOKAHEAD_ONLY | ['LOOKAHEAD_ONLY', 'NS', 'PASS', False] | ['LOOKAHEAD_ONLY', 'NS', 'PASS', False] | True | True |
| combination NS/PASS/NEAR -> LOOKAHEAD_ONLY | ['LOOKAHEAD_ONLY', 'NS', 'PASS', True] | ['LOOKAHEAD_ONLY', 'NS', 'PASS', True] | True | True |
| combination NS/REVERSED/not near -> REVERSED_LOOKAHEAD | ['REVERSED_LOOKAHEAD', 'NS', 'REVERSED', False] | ['REVERSED_LOOKAHEAD', 'NS', 'REVERSED', False] | True | True |
| combination NS/REVERSED/NEAR -> REVERSED_LOOKAHEAD | ['REVERSED_LOOKAHEAD', 'NS', 'REVERSED', True] | ['REVERSED_LOOKAHEAD', 'NS', 'REVERSED', True] | True | True |
| combination NS/SHORT/not near -> DIRECTIONAL | ['DIRECTIONAL', 'NS', 'SHORT', False] | ['DIRECTIONAL', 'NS', 'SHORT', False] | True | True |
| combination NS/SHORT/NEAR -> DIRECTIONAL | ['DIRECTIONAL', 'NS', 'SHORT', True] | ['DIRECTIONAL', 'NS', 'SHORT', True] | True | True |
| combination PASS/NS/not near -> GREEDY_ONLY | ['GREEDY_ONLY', 'PASS', 'NS', False] | ['GREEDY_ONLY', 'PASS', 'NS', False] | True | True |
| combination PASS/NS/NEAR -> GREEDY_ONLY_NEAR | ['GREEDY_ONLY_NEAR', 'PASS', 'NS', True] | ['GREEDY_ONLY_NEAR', 'PASS', 'NS', True] | True | True |
| combination PASS/PASS/not near -> YES | ['YES', 'PASS', 'PASS', False] | ['YES', 'PASS', 'PASS', False] | True | True |
| combination PASS/PASS/NEAR -> YES_NEAR | ['YES_NEAR', 'PASS', 'PASS', True] | ['YES_NEAR', 'PASS', 'PASS', True] | True | True |
| combination PASS/REVERSED/not near -> MIXED_PASS | ['MIXED_PASS', 'PASS', 'REVERSED', False] | ['MIXED_PASS', 'PASS', 'REVERSED', False] | True | True |
| combination PASS/REVERSED/NEAR -> MIXED_PASS | ['MIXED_PASS', 'PASS', 'REVERSED', True] | ['MIXED_PASS', 'PASS', 'REVERSED', True] | True | True |
| combination PASS/SHORT/not near -> GREEDY_ONLY | ['GREEDY_ONLY', 'PASS', 'SHORT', False] | ['GREEDY_ONLY', 'PASS', 'SHORT', False] | True | True |
| combination PASS/SHORT/NEAR -> GREEDY_ONLY_NEAR | ['GREEDY_ONLY_NEAR', 'PASS', 'SHORT', True] | ['GREEDY_ONLY_NEAR', 'PASS', 'SHORT', True] | True | True |
| combination REVERSED/NS/not near -> REVERSED | ['REVERSED', 'REVERSED', 'NS', False] | ['REVERSED', 'REVERSED', 'NS', False] | True | True |
| combination REVERSED/NS/NEAR -> REVERSED | ['REVERSED', 'REVERSED', 'NS', True] | ['REVERSED', 'REVERSED', 'NS', True] | True | True |
| combination REVERSED/PASS/not near -> MIXED_LOOKAHEAD_BETTER | ['MIXED_LOOKAHEAD_BETTER', 'REVERSED', 'PASS', False] | ['MIXED_LOOKAHEAD_BETTER', 'REVERSED', 'PASS', False] | True | True |
| combination REVERSED/PASS/NEAR -> MIXED_LOOKAHEAD_BETTER | ['MIXED_LOOKAHEAD_BETTER', 'REVERSED', 'PASS', True] | ['MIXED_LOOKAHEAD_BETTER', 'REVERSED', 'PASS', True] | True | True |
| combination REVERSED/REVERSED/not near -> REVERSED | ['REVERSED', 'REVERSED', 'REVERSED', False] | ['REVERSED', 'REVERSED', 'REVERSED', False] | True | True |
| combination REVERSED/REVERSED/NEAR -> REVERSED | ['REVERSED', 'REVERSED', 'REVERSED', True] | ['REVERSED', 'REVERSED', 'REVERSED', True] | True | True |
| combination REVERSED/SHORT/not near -> MIXED_LOOKAHEAD_BETTER | ['MIXED_LOOKAHEAD_BETTER', 'REVERSED', 'SHORT', False] | ['MIXED_LOOKAHEAD_BETTER', 'REVERSED', 'SHORT', False] | True | True |
| combination REVERSED/SHORT/NEAR -> MIXED_LOOKAHEAD_BETTER | ['MIXED_LOOKAHEAD_BETTER', 'REVERSED', 'SHORT', True] | ['MIXED_LOOKAHEAD_BETTER', 'REVERSED', 'SHORT', True] | True | True |
| combination SHORT/NS/not near -> DIRECTIONAL | ['DIRECTIONAL', 'SHORT', 'NS', False] | ['DIRECTIONAL', 'SHORT', 'NS', False] | True | True |
| combination SHORT/NS/NEAR -> DIRECTIONAL | ['DIRECTIONAL', 'SHORT', 'NS', True] | ['DIRECTIONAL', 'SHORT', 'NS', True] | True | True |
| combination SHORT/PASS/not near -> LOOKAHEAD_ONLY | ['LOOKAHEAD_ONLY', 'SHORT', 'PASS', False] | ['LOOKAHEAD_ONLY', 'SHORT', 'PASS', False] | True | True |
| combination SHORT/PASS/NEAR -> LOOKAHEAD_ONLY | ['LOOKAHEAD_ONLY', 'SHORT', 'PASS', True] | ['LOOKAHEAD_ONLY', 'SHORT', 'PASS', True] | True | True |
| combination SHORT/REVERSED/not near -> MIXED_SHORT | ['MIXED_SHORT', 'SHORT', 'REVERSED', False] | ['MIXED_SHORT', 'SHORT', 'REVERSED', False] | True | True |
| combination SHORT/REVERSED/NEAR -> MIXED_SHORT | ['MIXED_SHORT', 'SHORT', 'REVERSED', True] | ['MIXED_SHORT', 'SHORT', 'REVERSED', True] | True | True |
| combination SHORT/SHORT/not near -> DIRECTIONAL | ['DIRECTIONAL', 'SHORT', 'SHORT', False] | ['DIRECTIONAL', 'SHORT', 'SHORT', False] | True | True |
| combination SHORT/SHORT/NEAR -> DIRECTIONAL | ['DIRECTIONAL', 'SHORT', 'SHORT', True] | ['DIRECTIONAL', 'SHORT', 'SHORT', True] | True | True |
| refused-key flag above 1% only |  |  | True | True |
| ratio not evaluable when ORACLE success is 0 (NEAR false) |  |  | True | True |
| not-evaluable report-only stages are rendered mechanically (abandoned / stopped) and never change the branch |  |  | True | True |
| exhaustive: every (status 1, status 2, NEAR) combination maps to the registered table (32) and is realised through decide() / wording() with truthful wording |  |  | True | True |

## Statistics

- exact McNemar == the two-sided binomial test on the discordant pairs (9 cases): True
- Holm (m=2): 0.004, 0.03 -> 0.008, 0.03; 0.03, 0.004 -> 0.03, 0.008; 0.02, 0.02 -> 0.04, 0.04; 0.6, 0.7 -> 1.0, 1.0: True
- bootstrap ratio CI is deterministic for the fixed seed and brackets the point ratio: True
- delta boundary: exactly +0.20 with Holm p < 0.01 is PASS; +0.1944 is SHORT: True
- Holm boundary: adjusted p 0.01 is NS; 0.0099 passes: True
- audit reading: b 12 / c 18 of 180 (difference -0.033, p 0.36, CI lower bound < -0.05) is 'inconclusive': True
- audit reading: b 2 / c 3 of 180 (CI lower bound >= -0.05) is 'surface-invariant within 0.05': True
- audit reading: a significant drop (b 2 / c 40) is 'surface-dependent'; the CI is deterministic: True

## Metrics

- a failed plan with a void step: IGNORE_STATUS applies it as posted (exposure 1), ORACLE does not (0): True
- the stored optimal plan scores success 1, optimal length 1, precision 1: True
- exposure read from a cache whose answer applied the void step: True
- a void step applied with an arithmetic error: not 'applied as posted' (0) but 'not a no-op' (1): True
- a guard-refused key (absent from the cache) is counted, and select_fixed exposures are labelled: True

