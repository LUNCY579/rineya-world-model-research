# T2b-register screen: intermediates in the answer, with a filler control

RCWM roadmap item 3 (`docs/roadmap/RCWM_ROADMAP_2026-10-08.md`); joint roadmap item 5b with BWM-001
(`JOINT_ROADMAP_2026-10-08.md`, BWM D-52).

**Status:** exploratory screen. Code and data built 2026-10-09 on CPU only, then revised the same day after review
(section 12). Not frozen; no GPU run; no model output exists. The GPU job refuses to run until
`python -B scripts/t2br_run.py --freeze --approval "..."` has recorded the operator's go-ahead (a screen lock, not a
preregistration). It is queued after PREREG-3, whose verdict is decision point 1 below.

## 1. Question

T2 trained 175 mix rows and got nothing: mix one-step accuracy was 0.022 before and 0.022 after. The conditional branch
choice is at chance (0.51) before and after T2, in all 3 seeds, while the in-branch arithmetic is right. OFF outputs the
unreduced sum. The joint finding (with BWM's RW2 `|2x| <= L` legality shortcut) is that computed-predicate rules are
not learned from bare-answer rows.

This screen asks two things:

- Does supervising the intermediates in the answer (compare, then compute, then answer) teach these rules where bare
  answers did not?
- Is any gain supervision, or only extra decode compute? A filler arm, matched token for token, is the control. Its
  limits are stated in section 3.

## 2. Environment semantics (`rcwmt/env.py`, frozen; quoted, not changed)

State `{"X","Y","Z"}`: integers modulo 97. One instruction per step (`env.step`):

| kind | update |
|---|---|
| swap | `reg[r], reg[o] = reg[o], reg[r]` |
| add | `reg[r] = (reg[r] + v) % 97` |
| mix | `reg[r] = (reg[r] + 2*reg[o] + v) % 97` |
| conditional | `reg[r] = (reg[r] + (v if reg[o] > threshold else -v)) % 97` |

- **The conditional predicate is STRICT.** If `reg[other] == threshold`, the instruction subtracts. The frame text says
  so: "adds value if other is greater than threshold, otherwise subtracts value".
- Python's `%` is non-negative, so an underflow wraps up: 3 - 5 gives 95.
- **wrap** (as `scripts/make_t2_data.stratum`): the raw value of the updated register lies outside [0, 96]. The raw value
  is r + v for add, r + 2o + v for mix, and r +/- v on the chosen branch for conditional.
- **distance** of a conditional: d = reg[other] - threshold (ok iff d > 0).

Ranges in the recorded CANDIDATE_VIEWS marginals (asserted at build): value -8..11; threshold 0..96; registers 0..96.

## 3. Joint trace format, register instance

The target is one JSON object. Its keys come in LITERAL order cmp, calc, after. It is serialized with an explicit order:
never `sort_keys`, never a sorting canonicaliser.

- **cmp:** the branch predicate's inputs, plus its boolean.
  - conditional: `{"x": reg[other], "thr": threshold, "ok": reg[other] > threshold}`. The joint template's `v >= t` is
    replaced by the env's strict `>`, as the template itself instructs.
  - swap, add and mix have no branch predicate, so cmp is `null` (as BWM's `finish`).
- **calc:** the chosen branch's arithmetic, `{"raw": <unreduced value of the updated register>, "mod": raw % 97}`.
  - swap has no arithmetic, so calc is `null`.
  - raw exposes exactly the intermediate that OFF stops at (the unreduced sum).
- **after:** the family's existing answer, the register state JSON exactly as `env.dumps` writes it.

Examples:

```
trace  {"cmp":{"x":35,"thr":34,"ok":true},"calc":{"raw":64,"mod":64},"after":{"X":35,"Y":64,"Z":49}}
plain  {"X":35,"Y":64,"Z":49}
filler {"fill":" . . . . . . . . . . . . . . . . . . . . . . . . .","after":{"X":35,"Y":64,"Z":49}}
```

**Arms.** All three use the same rows and the same prompt: Titan's verbatim one-step frame, `env.one_step_messages`.

- **plain:** the after state only.
- **trace:** cmp, then calc, then after.
- **filler:** the fill string is a run of the one-token unit `" ."`. Its length is chosen so that the WHOLE filler target
  has exactly as many Titan tokens as that row's trace target (binding amendment 4). This is checked on every row: with
  the `tokenizers` library at build and verify, and with transformers' AutoTokenizer in the tiny test.

**The filler control is not answer-free (amendment 4 as written).** The trace's length depends on the answer: the digit
counts and signs of raw and mod, and the branch. So the matched fill length encodes them, and the filler arm must
compute that information before it can close the fill string. Measured on the data alone (`AUDIT.json` `fill_leakage`):

| cell | fill length alone predicts | best single cut | majority | fill length differs under the other branch |
|---|---|---|---|---|
| training mix | wrap | 0.957 | 0.50 | - |
| dev mix_oor (primary) | wrap | 0.970 | 0.50 | - |
| training conditional | branch | 0.527 | 0.50 | 0.580 |
| dev cond_oor_val / cond_oor_thr / cond_far | branch | 0.57 / 0.85 / 0.72 | 0.50 | 0.40 / 0.10 / 0.10 |

On the conditional strata part of the best-cut value is prompt-derived (the digit count of x, the sign of thr); the last
column isolates the answer-dependent part. The pre-stated rule therefore words decision point 3 as "trace ~ filler, not
supervision of the intermediates (extra decode compute and/or the length-coded information in the amendment-4 fill)",
and pre-states the filler diagnostics (section 7). An answer-independent fill length is proposed to BWM as an open
question (section 12); it is not adopted here, because amendment 4 is binding.

**Parsing.**

- `parse_trace` rejects any key order other than cmp, calc, after. It also rejects extra, missing or duplicate keys.
  Duplicate keys are stricter than BWM's `traces.parse_trace`, which keeps the last duplicate.
- `parse_filler` applies the same rules to `["fill","after"]`.
- The decision parser reads ONLY after: `after_text` re-serializes it (sorted, compact; BWM's convention) and
  `rcwmt.env.parse_state(lenient=True)` reads it, the parser of every existing chain and regret pipeline. A malformed
  trace or filler answer has no after, so it counts as a wrong answer.
- cmp_ok, ok_ok, calc_ok, after_follows_calc and calc_follows_cmp are scored separately, as mechanism. cmp and calc are
  compared as ordered JSON, so `true` and `1` differ.

## 4. Training rows: `data/t2br_train_{plain,trace,filler}.jsonl`

750 register rows, rendered three ways, in the same order (plain `49067515…`, trace `1bad8116…`, filler
`3102361d…`).

| block | rows | design |
|---|---:|---|
| mix | 300 | 150 wrap / 150 no-wrap; the state always changes |
| wrap | 150 | add whose raw sum leaves [0, 96]: 75 over (v > 0) / 75 under (v < 0) |
| conditional, core | 150 | the roadmap's rows: other = threshold - 1 / threshold / threshold + 1, 50 each; value != 0 |
| conditional, twins | 150 | one minimal-pair twin per core row, on the OTHER side of the threshold at distance 2..15 (uniform) |

- Each twin differs from its core row in ONE number: per core offset 25 twins move the threshold and 25 move the other
  register. Moving only the threshold would make the threshold alone predictive of the branch (in a design simulation, threshold-only twins at distance 2..30 gave "threshold < 48" 0.62-0.63).
- Twins: 100 ok-true at d = +2..+15 (twins of the ok-false cores), 50 ok-false at d = -15..-2.
- Add and subtract are exactly 150 / 150. Pair wrap patterns are balanced: neither member wraps / the add member wraps /
  the subtract member wraps, 50 pairs each.
- Swap and add/no-wrap are absent on purpose. They are the untouched-skill controls.
- Mean target tokens: plain 15.55, trace 38.06, filler 38.06.

**Audit** (`AUDIT.json`; exact counts). Exact-after accuracy of each shortcut on the training rows.

| heuristic | all | mix | wrap | conditional | gate |
|---|---|---|---|---|---|
| copy | 0.000 | 0.000 | 0.000 | 0.000 | <= 0.55 |
| unreduced sum | 0.467 | 0.500 | 0.000 | 0.667 (reported) | <= 0.55 (all, mix, wrap) |
| always add / always subtract | - | - | - | 0.500 / 0.500 | <= 0.55 |
| wrong register; other > 48; threshold < 48; value > 0 | - | - | - | 0.527; 0.527; 0.540; 0.500 | <= 0.55 |
| rule variants: `>=`; other == thr + 1; units digit == (thr + 1) units; units digit `>`; tens digit `>` | - | - | - | 0.833; 0.667; 0.680; 0.717; 0.757 | contradicted by >= 1/6 (acc <= 5/6) |
| mix with 1x other; mix without value | - | 0.030; 0.110 | - | - | reported |
| add => wrap by 97 | - | - | 1.000 (by design) | - | reported |

- **Rule variants cannot meet a 0.55 bar.** Each coincides with `other > threshold` on a structural subset, so its
  accuracy is at least that subset's share: acc(other == thr + 1) = P(ok false) + P(other = thr + 1) >= 0.5 + 50/N with
  the roadmap's 50 rows at thr + 1 (0.55 would need N >= 1,000 conditional rows), and acc(`>=`) = 1 - P(other = thr).
  They are gated by identification instead: every variant is contradicted by at least 50 of the 300 conditional rows
  (`>=` by the 50 rows at other = thr; the near-only variants by the far twins; coarse comparisons by the core rows).
- The best single-feature stump for the branch (X, Y, Z, the register's and the other's value, threshold, value, tick,
  register and other id) reaches 0.547; the permutation null's 95th percentile is 0.593 (reported).
- "add => wrap by 97" is 1.0 on the training adds by design (every training add wraps). A G4 loss on add_nowrap shared
  by every trained arm is therefore a property of the data; the analysis reports trace - plain beside trace - OFF.

**Surface diversity (decision point 1).** OFF by default.

- `python -B scripts/t2br_make_data.py --surface-diversity on` re-renders the SAME rows into
  `data/t2br_train_sd_<arm>.jsonl`. It uses 4 rule-text paraphrases (paraphrase 0 with the canonical names is the
  verbatim frame) and field-name variants: register / target / dst; other / source / src; value / amount / delta;
  threshold / limit / cutoff; kind / op; before / registers / state; event / instruction.
- `python -B scripts/t2br_run.py --train-variant sd` trains on it (outputs `runs/t2b_register/s0_sd/`; results
  `T2BR_RESULTS_s0_sd`). The queue hook draft has one variable, `T2BR_VARIANT`. No code change is needed.
- Golds and targets are unchanged. The dev bank stays canonical.

## 5. Dev screen bank: `data/t2br_dev.jsonl` (`2c0250a2…`) + `data/t2br_chains.jsonl` (`fe15c2b7…`)

| stratum | n | range | role | definition |
|---|---:|---|---|---|
| mix | 100 | in | non-inferiority (mix_in) | 50 wrap / 50 no-wrap |
| wrap | 80 | in | reported | add, raw outside [0, 96]: 40 over / 40 under |
| cond_xm1 | 60 | in | non-inferiority | other = threshold - 1 (ok false) |
| cond_x0 | 60 | in | non-inferiority | other = threshold (ok false: strict) |
| cond_xp1 | 60 | in | non-inferiority | other = threshold + 1 (ok true) |
| cond_mid | 100 | in | non-inferiority | 50 pairs at \|d\| 2..15 (the twins' band); 25 move the threshold, 25 the other register |
| swap | 80 | in | control | swap of unequal registers (not trained) |
| add_nowrap | 80 | in | control | add, value != 0, no wrap (not trained) |
| mix_oor | 100 | OUT | PRIMARY (mix) | \|value\| 12..40; 50 wrap / 50 no-wrap |
| wrap_oor | 80 | OUT | reported | add, \|value\| 12..40, wraps: 40 over / 40 under |
| cond_oor_val | 100 | OUT | PRIMARY (conditional) | 50 boundary pairs: other = threshold (false) / threshold - 1 (true); \|value\| 12..40 |
| cond_oor_thr | 100 | OUT | PRIMARY (conditional) | 50 pairs: threshold -40..-1 (true) / 97..136 (false) |
| cond_far | 100 | OUT | PRIMARY (conditional) | 50 pairs at \|d\| 16..96, beyond every training row's 0..15; in-range values and thresholds; each pair moves the other register |

- The three in-range near strata are 60 TRIPLETS: the same row, with threshold = x + 1 / x / x - 1.
- Every pair stratum is branch-balanced: one ok-true and one ok-false member per pair.
- cond_far pairs move the other register because a far pair that moves the threshold is separated by the threshold
  alone ("threshold < 48" would score about 0.96 on it).
- What each shortcut would score on each stratum is in `AUDIT.json` `dev_heuristics`. On the pooled primary
  conditional: other == thr + 1 0.667; threshold < 48 0.667; wrong register 0.670; other > 48 0.650; units-digit `>`
  0.677; tens-digit `>` (a coarse comparison) 0.843; `>=` 0.833. On cond_far, other == thr + 1 scores 0.50 (the
  reviewed failure mode).
- **Chains.** k = 4, `chain_one` self-rollout, the existing decision pipeline. CH_IN: 32 parents x 8 candidates from
  `env.gen_event`'s recorded marginals. CH_X: 16 parents x 8 candidates, the same except that every add / mix /
  conditional value is replaced by |v| 12..40. Each parent needs at least 4 distinct final states. (Unchanged by the
  revision: the same 384 chains.)

**Freshness.** Zero payload and zero FDS operative-content overlap with PREREG-3's exclusion union: 308,726
payloads and 301,783 operative contents from 17 sources (the records tree harvested by
`l1_make_ood.harvest_historical`, which skips the 4 `HSKILL_CONFIRM_V2*` files by NAME, so they are never opened or
hashed, and `rcwmt.bank.load_exclusions()` is NOT called; bank_pilot, bank_confirm, bank_confirm2 and bank_s4000; the
T2 data files; SELECT-VIEWS-001; the L1 OOD bank; the FDS-era T_BANK* files; 280 rolled-out transitions of multi-event
register prompts). The PREREG-3 bank is ledger-only, so it was not opened. Training, dev and chain units are mutually
disjoint. `--verify` rebuilds every data file in memory with the CURRENT builder against a fresh harvest and requires
byte equality, so the data's provenance never rests on a stale builder hash.

## 6. Arms, recipe, runner (`scripts/t2br_run.py`)

**Recipe.** R-only LoRA per arm, with the T2 recipe: the vendored BWM trainer (D-15), `lm.load_core('titan-sparse')`,
FP32; r16 / alpha 32 / dropout 0.05 on q, k, v, o, gate, up, down; 190 steps x batch 16, lr 1e-4 with WSD; max_len 512,
micro 4; `train_generic`'s row order; seed 0. All three arms see identical rows in identical order. 190 x 16 over 750
rows is 4.05 epochs; T2 was 2.03 epochs.

**Runner.** One lease, `rcwm-t1:t2b-register`, with D-22 pairing inside the 4.5 h turn cap (a larger `--cap-h` is
refused): no stage starts within 0.5 h of the cap, and a stage starts only if 1.25 x its estimate fits. The gate is
checked before AND after the resource / thermal wait that precedes every model load, so a long wait defers the stage
instead of pushing it past the cap. Stages, in order:

1. T_plain, T_trace, T_filler. Each is written to `.partial`, then renamed.
2. Load Sparse plus the 3 adapters, unmerged (D-28). Then A0: the OFF chain_one on the 208 pilot TEST register chains
   must reproduce `runs/t2/t2_s0/gen_chain_one_base.jsonl` on at least 99% of chains. A resumed turn re-anchors.
3. E1: 1,100 one-step rows x OFF / plain / trace / filler.
4. E2: 384 chains x the same 4 sides. Each side keeps its own parser's after state.

Outputs are create-only, under `runs/t2b_register/s<seed>[_<variant>]/`. Every stage is recorded with sha256; partial
stages are orphaned and redone; DONE.json at the end. One-waiter check (R14), run lock, runtime pin (torch 2.6.0+cu124,
transformers 5.14.1, peft 0.20.0), Sparse fingerprint (818be9c6b6a3) at both loads. Exit codes: 0 done; 3 deterministic
stop (persisted when it concerns the run); 5 resumable deferral; 1 crash.

## 7. Pre-stated go rule (verbatim; `scripts/t2br_common.GO_RULE_TEXT`, applied by `scripts/t2br_analysis.verdict`)

> PRE-STATED SCREEN GO RULE (T2b-register, seed by seed; point estimates on the fresh dev screen bank, decided EXACTLY on integer counts as rationals (no floating-point rounding: a difference equal to a bar meets a '>=' bar and does not meet a '<' bar); 95% cluster-bootstrap CIs reported, not gating; every condition component decided within one bootstrap SE of its bar is flagged in the report). Outcome = exact 'after' state (the decision parser reads only 'after'; a malformed trace/filler answer is wrong). PRIMARY outcomes are out of range (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr + cond_far pooled (branch-balanced pairs; cond_far = in-range values and thresholds with |other - threshold| 16..96, beyond every training row's 0..15); 'mix' = mix_oor. VALIDITY: format_ok = the strict joint parser accepts the answer; the trace (filler) side is valid iff its format_ok >= 0.95 on each primary. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH primaries, STOP: 'trace ~ filler, not supervision of the intermediates (extra decode compute and/or the length-coded magnitude/branch information in the amendment-4 fill; not attributable to decode compute alone)', with the sub-label 'trace format failed' if the trace side is not valid, else 'filler control invalid' if the filler side is not valid, else 'neither format helped' if filler - plain < 0.10 on both primaries. Otherwise GO iff ALL of: (G1) trace - plain >= +0.25 on conditional; (G2) trace - plain >= +0.30 on mix; (G3) filler below trace: trace - filler >= +0.10 on EACH primary; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean over cond_xm1, cond_x0, cond_xp1, cond_mid) and on mix_in; and the filler side is valid. If G1-G5 all hold but the filler side is not valid: INCONCLUSIVE: filler control invalid. Otherwise NO-GO, naming every failed condition. Pre-stated filler diagnostics (reported, not gating): per primary stratum the filler side's format_ok, fill_exact and its after-accuracy split by fill_exact, and the bank's fill-length leakage (how well the gold fill length alone predicts wrap / branch). The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

Interpretation choices, fixed before any output:

- **PRIMARY outcomes are out of range** (binding joint amendment 5). "conditional" pools cond_oor_val (values), cond_oor_thr
  (thresholds) and cond_far (distance beyond every training row); "mix" is mix_oor. The in-range strata are reported and
  act as the non-inferiority controls (G5); swap and add_nowrap are the untouched-skill controls (G4).
- **"Filler below trace"** means trace - filler >= +0.10 on EACH primary (G3); decision point 3 (both below 0.10) is
  checked first. A GO therefore separates each primary's gain from the filler by the same margin that decision point 3
  uses.
- **Exactness.** Every condition is decided on integer counts as rationals (`fractions.Fraction`); `verdict()` refuses
  floats. The dry run checks every exact-boundary count pair on the bank's own denominators (1,287 pairs at a bar, each with its one-count-past partner), and an
  engineered run whose every component sits exactly on its bar must say GO (a float implementation fails G2 there:
  0.6 - 0.3 < 0.3).
- **Validity.** A filler side that does not parse (for example a greedy " . . ." loop that never closes) would pass G3
  trivially; it now yields INCONCLUSIVE instead of GO. A trace side that does not parse gives the decision-point-3
  sub-label "trace format failed" instead of a supervision reading.

## 8. Reported, not gating

- Per stratum and side: accuracy (Wilson CI) and the parse rate; cluster-bootstrap contrasts for every pair of sides.
- Every rule component with its exact value, bootstrap SE and a flag when it lies within one SE of its bar.
- Trace fields per stratum: format_ok, cmp_ok, ok_ok, calc_ok, after_follows_calc and calc_follows_cmp.
- Filler diagnostics per primary stratum: format_ok, fill_exact, and after-accuracy split by fill_exact.
- Controls: trace - OFF (gating), trace - plain, plain - OFF and filler - OFF.
- Every side's branch choice, read from its after (correct / wrong / neither), on all conditional strata; unreduced-sum
  answers on mix and wrap.
- Chains, per stratum (CH_IN, CH_X, ALL) and side: normalised regret at k = 1, 2, 4 (`rcwmt.decisions.g_only`, split dev)
  and exact state, with parent-paired contrasts; the copy-s0 floor; the per-step to k4 product law (p(class) from each
  side's gold-input chain steps, R8 convention; predicted k4 exact = mean over chains of the product of p over the
  chain's 4 gold step classes, against the observed k4 exact and the pooled p^4).

## 9. GPU plan (`docs/t2b/register/GEN_PLAN.json`)

One turn: about 3.84 h at the registered planning rate (2.7 short gens/s) and 3.27 h at the observed rate
(3.8). With a 3 h cap the job takes 2 turns (the second re-anchors, then E2).

| stage | estimate (planning / observed) |
|---|---|
| T_plain | 34.2 min |
| T_trace / T_filler | 38.8 min each |
| A0 | 7.1 / 5.6 min (832 generations) |
| E1 | 46.4 / 33.0 min (4,400 generations) |
| E2 | 64.8 / 46.1 min (6,144 generations) |

Training is T2's measured 1,858.8 s scaled by processed tokens (483k / 551k / 551k against 464k). Trace and filler
answers decode 2.42x longer than plain. 11,376 generations at most.

## 10. Tests and records (`docs/t2b/register/`)

- **AUDIT.json** (`t2br_make_data.py --verify`): golds, targets and token equality re-derived; stratum, pair and twin
  definitions; the rebuild check; freshness with a fresh name-skipping harvest; the audit, fill-length leakage and dev
  heuristic floors.
- **SCORER_SELFTEST.json** (`t2br_analysis.py --selftest`): golds 100% for every side and field; malformed or wrong
  answers (including an unterminated fill and a fill one unit short) never accepted; floors exact; exact metric
  identities; chain-pipeline identities.
- **DRYRUN.json / DRYRUN.md** (`t2br_analysis.py --dryrun`): 8 synthetic scenarios (GO; two decision-point-3 stops;
  lost controls; in-distribution loss; G3 alone; invalid filler; malformed traces), 3 engineered exact-boundary runs,
  the exhaustive boundary checks, float refusal and the results stem.
- **TINY_TEST.json** (`t2br_test_tiny.py`): tiny random Qwen3 and the real vendored trainer: formats; surface diversity;
  chain equality; a full run; adapter switching; crash and resume in eval and in training; deferral, including the
  re-check after the resource wait and the cap clamp; deterministic stops; the one-waiter rule and the run lock; an
  oracle run on the full bank through the analysis; the freeze tool; the plan.
- **GEN_PLAN.json** (`t2br_run.py --plan`, CPU, no torch).
- **queue_hook_t2br.sh.draft:** NOT installed. It is a frozen input. The runner refuses (exit 3, not persisted) if an
  installed `scripts/queue_hooks/*t2br*.sh` is not the draft byte for byte.

## 11. Decision points (roadmap sequencing)

- **DP1:** PREREG-3 says TRIGGER. Build with `--surface-diversity on` and set `T2BR_VARIANT=sd` in the hook draft, both
  before the freeze.
- **DP3:** trace - filler < 0.10 on both primaries. Stop, with the section 7 wording.
- **GO:** seeds 1-2 (`--seed 1/2`). The roadmap then requires the same rule in all 3 seeds, a sealed R bank, and k4 regret
  <= -0.10.

## 12. Revision after review (2026-10-09)

1. **Identification (must-fix).** The first build's 300 conditional training rows all had d in {-1, 0, +1}, so
   "other == thr + 1" fitted them perfectly and would also pass every conditional dev stratum except cond_oor_thr. Now:
   the twins sit at distance 2..15 on the other side of the threshold; cond_far (primary) and cond_mid (in-distribution)
   were added; rule variants are audited (gated by identification; 0.55 is unreachable for them, section 4); the single
   surface cues of the branch are now gated at 0.55.
2. **Filler leakage (must-fix).** Amendment 4 is binding and kept. The leakage is measured, the filler diagnostics are
   pre-stated, and decision point 3 is worded so that it does not attribute trace ~ filler to decode compute alone.
3. **Exact decisions (must-fix).** Rationals throughout, float refusal, exhaustive boundary checks, engineered
   boundary runs.
4. **Validity gates and G3 at +0.10 per primary (should; both adopted).**
5. **Notes adopted:** trace - plain on the controls and the "add => wrap" diagnostic; SE flags; the cap clamp and the
   post-wait gate re-check; the rebuild check; the variant in the results stem and one variable in the hook draft.
