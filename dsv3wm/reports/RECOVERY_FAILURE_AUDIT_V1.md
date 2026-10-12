# Recovery dynamics failure audit v1 — post hoc

This audit reads the fixed 24-parent recovery DEV labels, preserved native
outputs and supplementary trace reconstructions. Exact transitions are used
only for the offline first/last-action counterfactuals below. No model weights
were loaded, no native inference was run, no prompt/data/protocol was changed,
and no confirmation label was accessed. These are one-seed development findings.

## Observed primary outcomes

The registered P1 remains four-step **free-running exact final state**. It is
different from a single direct request containing four actions.

| Arm | h4 free-running P1 | ID / OOD successes | Correct final despite a wrong earlier state |
|---|---:|---:|---:|
| B0, inherited registered F | 24/24 | 12/12 / 12/12 | 0 |
| A0, 32 new one-step updates | 24/24 | 12/12 / 12/12 | 0 |
| A1, auxiliary token training | 22/24 | 12/12 / 10/12 | 0 |
| A2, explicit horizon training | 24/24 | 12/12 / 12/12 | 0 |
| A5, 96 one-step updates | 23/24 | 12/12 / 11/12 | 0 |

All adapted arms predict the first one-step state correctly on 24/24. A1 loses
one parent at step 3 and a second at step 4; A5 loses one at step 4. All three
failures are OOD rule-limit parents. B0 and A0 are already at the P1 ceiling;
A2 matching them cannot establish a gain beyond the inherited F. The fixed
cohort is retained, not redrawn to create room for improvement.

The zero count in the last column comes from checking the intermediate
predicted states, not merely observing final goal success. It does not imply
that other worlds or longer trajectories cannot contain compensating errors.

## Direct horizon readouts and flags

Counts below use all 24 parents. A reply must contain exactly integer `x` and
boolean `legal`/`terminated` to be accepted. Copy and flag counts use accepted
replies only; malformed replies remain failures/abstentions in the original
metrics and are not repaired for this audit.

| Arm | h2 direct state correct | h2 copies initial state | h2 malformed | h4 direct state correct | h4 copies initial state | h4 malformed |
|---|---:|---:|---:|---:|---:|---:|
| A0 | 2/24 | 5 | 5 | 0/24 | 2 | 10 |
| A1 | 3/24 | 11 | 0 | 0/24 | 19 | 0 |
| A2 | 3/24 | 16 | 0 | 0/24 | 19 | 0 |
| A5 | 1/24 | 6 | 9 | 0/24 | 0 | 17 |

B0's qualification collected direct h1 only; no B0 direct-h4 result is invented.
For every DEV parent, the gold h4 state differs from its initial state. Thus
copying the input cannot be a correct h4 answer on this cohort. A1/A2 produce
valid h4 JSON on 24/24 but copy the initial state on 19/24 and are correct on
0/24. Better response completeness is visible; improved direct dynamics is not.

| Arm | Accepted h4 replies | Wrong last-action `legal` among accepted | Correct last-action `legal` / all 24 | Wrong `terminated` among accepted |
|---|---:|---:|---:|---:|
| A0 | 14 | 5 | 9/24 | 0 |
| A1 | 24 | 12 | 12/24 | 0 |
| A2 | 24 | 14 | 10/24 | 0 |
| A5 | 7 | 0 | 7/24 | 0 |

Sixteen gold h4 last actions are legal and eight are illegal. Every gold
termination flag is false. All accepted replies also emit false termination,
so the absence of a termination error here is a constant-label result, not a
test of positive termination detection or absorbing dynamics.

Every malformed direct-h4 A0 response (10) and A5 response (17) is the same
complete JSON text, missing `x`:

```json
{"legal":false,"terminated":false}
```

Each ends at EOS after 10 output tokens. These particular failures are missing
state fields, not output-cap truncation or broken JSON syntax. For a separate
content-only diagnostic, the emitted raw legality boolean is wrong on 11/24
A0 replies and 9/24 A5 replies: respectively six and nine of these errors occur
inside the rejected missing-state objects. This does not rescue those objects
or replace the accepted-schema flag metric above.

## First/last-action counterfactuals

For each parent, the **first-only** counterfactual executes just action 1 from
the original state; the **last-only** counterfactual executes just action 4 from
that same original state. Neither executes intervening actions. The audited
exact RegisterWorld rule implementation was invoked only after the preserved
outputs existed. A match requires an accepted reply and exact state equality;
the full-reply columns additionally compare that single action's legality and
termination flags.

| Arm | Accepted h4 replies | First-only state match | First-only full reply match | Last-only state match | Last-only full reply match |
|---|---:|---:|---:|---:|---:|
| A0 | 14 | 9 | 9 | 4 | 4 |
| A1 | 24 | 13 | 13 | 9 | 9 |
| A2 | 24 | 6 | 6 | 9 | 7 |
| A5 | 7 | 5 | 5 | 2 | 2 |

These matches overlap with one another and with copying, especially when a
single action is illegal and leaves the state unchanged. They are diagnostic
agreements, not a mutually exclusive classifier or proof of a causal shortcut.
A0/A1 accepted outputs frequently agree with a first-action-only answer;
A2's 19 copies and lower first-only agreement have a different descriptive
pattern. Neither pattern supplies a correct four-action state on this bank.

## The three free-running failures

All three parents' earlier raw replies are correct up to the step identified
below. The one-step inputs use each preceding model-produced state, and no
oracle reset or feedback follows the failure.

### A1: illegal odd halving accepted as legal, final step

Parent `b2f45d7b298aad02b3f3f922a7776f038bbcee1752655ba9aa932ccf221c0f0a`,
OOD limit 96, initial state −64, actions `add1, add1, add1, halve`.
The first three predictions are −63, −62 and −61, all correct and legal.
At step 4, halving odd −61 must be illegal and preserve −61. The raw reply is:

```json
{"legal":true,"terminated":false,"x":-30}
```

This is valid output syntax with wrong state and wrong legality. The accumulated
predicted attempted-failure flag remains false; the gold accumulated flag becomes
true. The integer −30 resembles a rounded half, but this trace alone does not
identify the rounding mechanism or explain why the legality rule was lost.

### A1: odd halving returns a fractional state, then abstention

Parent `434b5933452acf3180ef67d8b2f4e358b5f77fc8ef5cb9a23fa88a40162682a0`,
OOD limit 96, initial state −51, actions `add1, halve, halve, sub1`.
Steps 1/2 correctly produce −50/−25. At step 3, halving odd −25 must be illegal
and preserve −25. The raw reply is:

```json
{"legal":true,"terminated":false,"x":-12.5}
```

The fractional state violates the integer-state contract, and the raw legality
claim also contradicts the rule. Parsing rejects the entire reply. Step 4 is
not queried; the free-running chain abstains, instead of converting −12.5 to an
integer or substituting the correct −25. Cumulative attempted-failure becomes
unknown after the malformed step, not a falsely inferred `not last_legal` value.
The gold final state would be −26, but that label never repairs inference.

### A5: legal doubling rejected as illegal, final step

Parent `e5624fb3f257b1bf5a52354cfcd5d5cf5b6e53dc010094bd5a0a090f4303bcc7`,
OOD limit 80, initial state 66, actions `halve, sub1, sub1, double`.
Steps 1–3 correctly produce 33, 32 and 31. At step 4, doubling 31 to 62 remains
within limit 80 and is legal. The raw reply is:

```json
{"legal":false,"terminated":false,"x":31}
```

This is valid output syntax with a false illegality/no-op prediction. The
accumulated predicted attempted-failure flag becomes true; the gold flag stays
false. A retained magnitude/limit shortcut after training at limits 24/32/40
is one possible explanation. No rule-token or limit intervention was executed
to establish it.

## Token loss and competing explanations

The target is a short repeated JSON object. A tokenizer-only, no-model audit of
the 24 gold DEV h4 labels uses the source-verified tokenizer
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` and the
same separate-target tokenization plus one supervised EOS as `lm.encode_pair`.
Offsets classify tokens overlapping the numeric state value or boolean values;
remaining positions are JSON keys/punctuation/EOS.

| Gold target horizon | Total supervised positions | Numeric-state value overlap | Boolean-value overlap | JSON structure/EOS |
|---|---:|---:|---:|---:|
| h1 | 357 | 58 | 48 | 251 |
| h2 | 359 | 60 | 48 | 251 |
| h4 | 358 | 59 | 48 | 251 |

About 70% of h4 target positions are structure/EOS, 16.5% overlap the state value
and 13.4% overlap flags. A merged token touching a value is conservatively counted
in the value category. This is an audit of this target format on DEV labels,
not a reconstruction of the training loss. It does **not** establish that 70%
of loss magnitude or gradient comes from structure; per-position losses were
not captured. Mean token CE mixes these categories and cannot prove transition
learning or successful free-running composition. Complete JSON can coexist
with every direct h4 state being wrong.

The multi-action system prompt appends an ordered-sequence instruction to the
old `SYS_FORWARD`, which still contains the phrase **one action**. Its user
payload also changes the scalar `action` field to an `actions` list. That is an
observed interface mismatch. Plausible explanations include a learned scalar
action-field shortcut, treating an unfamiliar list as an illegal/no-op command,
first-action processing, and insufficient adaptation of the inherited prompt
representation. The copy/missing-state/counterfactual counts are consistent
with parts of these explanations but do not distinguish them causally.

The inherited transition skill still works well when called serially through
its registered one-step interface. Therefore failure of one-shot list handling
must not be presented as failure to compose four one-step transitions in general.
Conversely, A2's correct serial rollouts do not show that direct horizon training
learned list-conditioned consequences: direct h4 remains 0/24.

The A1/A5 cases show additional rule-sensitive failures in serial inference,
beyond the action-list framing problem. One seed, three errors and a small
selected development surface cannot establish that auxiliary token training or
extra one-step training generally harms Rineya. They are preserved regressions
under these specific artifacts, limits, data and budgets.

## Consequences for this fixed study

No prompt edit, parser rescue, new native intervention or easier redraw is part
of this audit. P1, schemas, parent cohorts, budgets and previous numeric records
remain unchanged. The H1/H2 continuation rules are not met by these P1 results;
no larger dynamics benefit or integration claim follows. Any investigation that
removes the inherited one-action phrase or varies action-list serialization
needs a separately dated development protocol before its new outcomes.

Calibration is unknown because no predictive probabilities/logit captures were
recorded. Zero positive termination labels, known ID one-step exposure, limited
OOD rule ranges, one training seed and the B0 ceiling constrain interpretation.
Keep those limits visible alongside any independently qualified planning-stage
work; planning gains must be measured on their own frozen contracts.

Evidence: [A0 native result](../results/recovery-v1/A0_dynamics.json),
[A1 native result](../results/recovery-v1/A1_dynamics.json),
[A2 native result](../results/recovery-v1/A2_dynamics.json),
[A5 native result](../results/recovery-v1/A5_dynamics.json).
Their `.traces.json` files contain the original replies; corresponding
`.posthoc.json` files contain strict causal reconstructions. The cohort is
[`dev.jsonl`](../datasets/recovery-v1/dev.jsonl), the prompt/parser source is
[`recovery.py`](../src/dsv3wm/recovery.py), and original executor/tokenizer
contracts are in [`rw2.py`](../../bwm/code/bwmn/rw2.py) and
[`lm.py`](../../bwm/code/bwmn/lm.py). This supplementary report adds interpretation
and diagnostics without replacing any registered result.
