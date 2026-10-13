# B4 futility screen (arm a vs R0, seeds_fresh)

**Status: GO.** B4 screen GO: arm (a), the RCX hybrid hook, passed the pre-set futility rule on seeds_fresh-40, paired vs item 7's R0 beam. This unlocks the co-signed B4 primary (fresh seed-20261410 set, n = 110). It is not a claim that the hook improves planning, and the screen's net and p are not effect estimates.

Rule: GO iff (a_only - R0_only) >= 2; otherwise NO-GO (net <= 1 is always NO-GO). The exact one-sided sign-test p and the two-sided McNemar p are reported only and do not gate (D-77). If more than 5% of hook calls return a penalty-band value (invalid state or critic error; value >= PENALTY = 1000), a GO is not credited to the learned critic alone and c+pen is mandatory in the full B4 (D-76a). The screen is VOID (no GO/NO-GO) on any hook model/encode/nonfinite/failsafe error, model_evals = 0, a canary failure or R0 outputs that do not match ADDENDUM_A_INPUTS; it is an INSTRUMENT FAULT (no decision; a and c re-run in one turn) if the round-1 identity check or the R0 shard-argument check fails.

| | value |
|---|---|
| a_only | 7 |
| R0_only | 1 |
| both | 26 |
| neither | 6 |
| net | 6 |
| one-sided sign p (reported only) | 0.03516 |
| two-sided exact McNemar p (reported only) | 0.07031 |
| n paired | 40 |
| round-1 identity holds | True |

| arm | success | CP 95% CI | failures | unverified VAL successes | mean tokens | mean rounds |
|---|---|---|---|---|---|---|
| R0 | 27/40 | [0.5087, 0.8143] | {'budget': 5, 'dead_end': 8, 'solved': 27, 'verified_but_invalid': 0} | 0 | 7120.2 | 4.67 |
| a | 33/40 | [0.6722, 0.9266] | {'budget': 5, 'dead_end': 2, 'solved': 33, 'verified_but_invalid': 0} | 0 | 5903.5 | 3.9 |

Hook: 24583 calls; penalty band (invalid state or critic error) share 0.0 (max 0.05; split {'invalid_state': 0, 'critic_error': 0, 'failsafe': 0}); GO credited to the critic: True.

Canary per hook shard (the torch versions are report-only; the binding check is |delta| <= 1e-5 per probe):

| shard | max abs delta | canary torch | host torch | match |
|---|---|---|---|---|
| shard_0of4 | 0.0 | 2.6.0+cu124 | 2.6.0+cu124 | True |
| shard_1of4 | 0.0 | 2.6.0+cu124 | 2.6.0+cu124 | True |
| shard_2of4 | 0.0 | 2.6.0+cu124 | 2.6.0+cu124 | True |
| shard_3of4 | 0.0 | 2.6.0+cu124 | 2.6.0+cu124 | True |
