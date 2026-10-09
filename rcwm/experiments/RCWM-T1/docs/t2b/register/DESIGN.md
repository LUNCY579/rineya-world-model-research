# T2b-register screen: intermediates in the answer, with a filler control

RCWM roadmap item 3 (`docs/roadmap/RCWM_ROADMAP_2026-10-08.md`); joint roadmap item 5b with BWM-001
(`JOINT_ROADMAP_2026-10-08.md`, BWM D-52).

**Status:** exploratory screen. Code and data built 2026-10-09 on CPU only. Not frozen; no GPU run; no model output
exists. The GPU job refuses to run until `python -B scripts/t2br_run.py --freeze --approval "..."` has recorded the
operator's go-ahead (a screen lock, not a preregistration). It is queued after PREREG-3, whose verdict is decision
point 1 below.

## 1. Question

T2 trained 175 mix rows and got nothing: mix one-step accuracy was 0.022 before and 0.022 after. The conditional branch
choice is at chance (0.51) before and after T2, in all 3 seeds, while the in-branch arithmetic is right. OFF outputs the
unreduced sum. The joint finding (with BWM's RW2 `|2x| <= L` legality shortcut) is that computed-predicate rules are
not learned from bare-answer rows.

This screen asks two things:

- Does supervising the intermediates in the answer (compare, then compute, then answer) teach these rules where bare
  answers did not?
- Is any gain supervision, or only extra decode compute? A filler arm, matched token for token, separates the two.

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

Ranges in the recorded CANDIDATE_VIEWS marginals (asserted at build):

- value: -8..11;
- threshold: 0..96;
- registers: 0..96.

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

Examples, from training rows:

```
trace  {"cmp":{"x":35,"thr":34,"ok":true},"calc":{"raw":64,"mod":64},"after":{"X":35,"Y":64,"Z":49}}
plain  {"X":35,"Y":64,"Z":49}
filler {"fill":" . . . . . . . . . . . . . . . . . . . . . . . . .","after":{"X":35,"Y":64,"Z":49}}
```

**Arms.** All three use the same rows and the same prompt: Titan's verbatim one-step frame, `env.one_step_messages`.

- **plain:** the after state only.
- **trace:** cmp, then calc, then after.
- **filler:** the fill string is a run of the one-token unit `" ."`. Its length is chosen so that the WHOLE filler target
  has exactly as many Titan tokens as that row's trace target.
  - This was checked on every row: with the `tokenizers` library at build and verify, and with transformers'
    AutoTokenizer in the tiny test.

**Parsing.**

- `parse_trace` rejects any key order other than cmp, calc, after. It also rejects extra, missing or duplicate keys.
  - Duplicate keys are stricter than BWM's `traces.parse_trace`, which keeps the last duplicate.
- `parse_filler` applies the same rules to `["fill","after"]`.
- The decision parser reads ONLY after.
  - `after_text` re-serializes after, sorted and compact, as BWM's convention does. `rcwmt.env.parse_state(lenient=True)`
    then reads it. That is the parser of every existing chain and regret pipeline.
  - A malformed trace or filler answer has no after, so it counts as a wrong answer.
- cmp_ok, ok_ok, calc_ok, after_follows_calc and calc_follows_cmp are scored separately, as mechanism.
  - cmp and calc are compared as ordered JSON, so `true` and `1` differ.

## 4. Training rows: `data/t2br_train_{plain,trace,filler}.jsonl`

750 register rows, rendered three ways, in the same order. The row files have these SHA-256 hashes:

- plain `7344677d…`;
- trace `6c034182…`;
- filler `a3c8ada6…`.

| block | rows | design |
|---|---:|---|
| mix | 300 | 150 wrap / 150 no-wrap; the state always changes |
| wrap | 150 | add whose raw sum leaves [0, 96]: 75 over (v > 0) / 75 under (v < 0) |
| conditional, core | 150 | other = threshold - 1 / threshold / threshold + 1, 50 each; value != 0 |
| conditional, twins | 150 | minimal pairs: the core row with ONLY the threshold moved so the branch flips (thr-2 / thr-1 / thr+1) |

- Add and subtract are exactly 150 / 150.
- Pair wrap patterns are balanced: neither twin wraps / the add twin wraps / the subtract twin wraps, 50 pairs each.
- Swap and add/no-wrap are absent on purpose. They are the untouched-skill controls.
- Values, thresholds and registers come from the recorded marginals, so these are in-range rows.
- Mean target tokens: plain 15.55, trace 38.08, filler 38.08.

**Surface-heuristic audit.** Exact-after accuracy of each shortcut on the training rows. The bar is <= 0.55: globally,
and inside the block where each shortcut is the candidate.

| heuristic | all | mix | wrap | conditional |
|---|---|---|---|---|
| copy (gated) | 0.000 | 0.000 | 0.000 | 0.000 |
| unreduced sum (gated: all, mix, wrap) | 0.467 | 0.500 | 0.000 | 0.667 (reported) |
| always add / always subtract (gated) | - | - | - | 0.500 / 0.500 |
| compare the wrong register; other > 48; threshold < 48; value > 0 (diagnostic) | - | - | - | 0.510; 0.500; 0.507; 0.500 |
| mix with 1x other; mix without value (diagnostic) | - | 0.030; 0.110 | - | - |
| non-strict `>=` (rule variant, not a surface cue; diagnostic) | - | - | - | 0.667 |

- The best single-feature stump for the branch reaches 0.533. The permutation null's 95th percentile is 0.587. Features:
  X, Y, Z, the register's value, the other's value, threshold, value, tick, register id and other id.
- `>=` differs from the env only on the x = threshold rows, so it scores 1 - share(x0). It is reported, not gated.

**Surface diversity (decision point 1).** OFF by default.

- `python -B scripts/t2br_make_data.py --surface-diversity on` re-renders the SAME rows into
  `data/t2br_train_sd_<arm>.jsonl`. It uses 4 rule-text paraphrases (paraphrase 0 with the canonical names is the
  verbatim frame) and field-name variants:
  - register / target / dst;
  - other / source / src;
  - value / amount / delta;
  - threshold / limit / cutoff;
  - kind / op;
  - before / registers / state;
  - event / instruction.
- `python -B scripts/t2br_run.py --train-variant sd` trains on it. No code change is needed.
- Golds and targets are unchanged. The dev bank stays canonical.

## 5. Dev screen bank: `data/t2br_dev.jsonl` (`8b01899c…`) + `data/t2br_chains.jsonl` (`fe15c2b7…`)

| stratum | n | range | role | definition |
|---|---:|---|---|---|
| mix | 100 | in | non-inferiority (mix_in) | 50 wrap / 50 no-wrap |
| wrap | 80 | in | reported | add, raw outside [0, 96]: 40 over / 40 under |
| cond_xm1 | 60 | in | non-inferiority | other = threshold - 1 (ok false) |
| cond_x0 | 60 | in | non-inferiority | other = threshold (ok false: strict) |
| cond_xp1 | 60 | in | non-inferiority | other = threshold + 1 (ok true) |
| swap | 80 | in | control | swap of unequal registers (not trained) |
| add_nowrap | 80 | in | control | add, value != 0, no wrap (not trained) |
| mix_oor | 100 | OUT | PRIMARY (mix) | \|value\| 12..40; 50 wrap / 50 no-wrap |
| wrap_oor | 80 | OUT | reported | add, \|value\| 12..40, wraps: 40 over / 40 under |
| cond_oor_val | 100 | OUT | PRIMARY (conditional) | 50 boundary pairs: other = threshold (false) / threshold - 1 (true); \|value\| 12..40 |
| cond_oor_thr | 100 | OUT | PRIMARY (conditional) | 50 pairs: threshold -40..-1 (true) / 97..136 (false) |

- The three in-range conditional strata are 60 TRIPLETS: the same row, with threshold = x + 1 / x / x - 1.
- **Chains.** k = 4, `chain_one` self-rollout, the existing decision pipeline.
  - CH_IN: 32 parents x 8 candidates, from `env.gen_event`'s recorded marginals.
  - CH_X: 16 parents x 8 candidates, the same except that every add / mix / conditional value is replaced by
    |v| 12..40.
  - Each parent needs at least 4 distinct final states.

**Freshness.** Zero payload and zero FDS operative-content overlap with PREREG-3's exclusion union:

- 308,726 payloads and 301,783 operative contents, from 17 sources;
- the records tree, harvested by `l1_make_ood.harvest_historical`. It skips the 4 `HSKILL_CONFIRM_V2*` files by NAME,
  so they are never opened or hashed. `rcwmt.bank.load_exclusions()` is NOT called;
- bank_pilot, bank_confirm, bank_confirm2 and bank_s4000;
- the T2 data files;
- SELECT-VIEWS-001;
- the L1 OOD bank (which has R_wrap and R_boundary rows);
- the FDS-era T_BANK* files;
- 280 rolled-out transitions of multi-event register prompts.

The PREREG-3 bank is ledger-only, so it was not opened. Training, dev and chain units are mutually disjoint.

## 6. Arms, recipe, runner (`scripts/t2br_run.py`)

**Recipe.** R-only LoRA per arm, with the T2 recipe:

- the vendored BWM trainer (D-15): `lm.load_core('titan-sparse')`, FP32;
- r16 / alpha 32 / dropout 0.05 on q, k, v, o, gate, up, down;
- 190 steps x batch 16, lr 1e-4 with WSD;
- max_len 512, micro 4;
- `train_generic`'s row order;
- seed 0.

All three arms see identical rows in identical order. 190 x 16 over 750 rows is 4.05 epochs; T2 was 2.03 epochs.

**Runner.** One lease, `rcwm-t1:t2b-register`, with D-22 pairing inside the 4.5 h turn cap: no stage starts within
0.5 h of the cap, and a stage starts only if 1.25 x its estimate fits. Stages, in order:

1. T_plain, T_trace, T_filler. Each is written to `.partial`, then renamed.
2. Load Sparse plus the 3 adapters, unmerged (D-28). Then A0: the OFF chain_one on the 208 pilot TEST register chains
   must reproduce `runs/t2/t2_s0/gen_chain_one_base.jsonl` on at least 99% of chains. A resumed turn re-anchors.
3. E1: 900 one-step rows x OFF / plain / trace / filler.
4. E2: 384 chains x the same 4 sides. Each side keeps its own parser's after state.

**Mechanics.**

- Outputs are create-only, under `runs/t2b_register/s<seed>/`.
- Every stage is recorded with sha256. Partial stages are orphaned and redone. DONE.json is written at the end.
- The job runs the one-waiter check (R14) and takes a run lock.
- It checks the runtime pin (torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0) and the Sparse fingerprint
  (818be9c6b6a3) at both the training and the eval loads.
- Exit codes:
  - 0: done;
  - 3: deterministic stop (persisted when it concerns the run);
  - 5: resumable deferral;
  - 1: crash.

## 7. Pre-stated go rule (verbatim; `scripts/t2br_common.GO_RULE_TEXT`, applied by `scripts/t2br_analysis.verdict`)

> PRE-STATED SCREEN GO RULE (T2b-register, seed by seed; point estimates on the fresh dev screen bank; 95% cluster-bootstrap CIs reported, not gating). Outcome = exact 'after' state (decision parser reads only 'after'; a malformed trace/filler answer is wrong). PRIMARY outcomes are the out-of-range strata (joint amendment 5): 'conditional' = cond_oor_val + cond_oor_thr pooled (branch-balanced by construction); 'mix' = mix_oor. GO iff ALL of: (G1) trace - plain >= +0.25 on conditional; (G2) trace - plain >= +0.30 on mix; (G3) filler below trace: trace - filler > 0 on both primaries; (G4) controls: trace ON - OFF >= -0.05 on swap and on add_nowrap (untouched skills); (G5) in-distribution non-inferiority: trace - plain >= -0.05 on conditional_in (branch-balanced mean of cond_xm1/cond_x0/cond_xp1) and on mix_in. DECISION POINT 3 (checked first): if trace - filler < 0.10 on BOTH primaries, STOP and report 'extra decode compute, not supervision' (sub-reason 'neither format helped' when filler - plain < 0.10 on both primaries as well). Otherwise NO-GO, naming every failed condition. The screen is seed 0; the roadmap's confirmation requires the same rule in all 3 seeds before a sealed R bank is built.

Interpretation choices, fixed before any output:

- **PRIMARY outcomes are out of range** (binding joint amendment 5). "conditional" means cond_oor_val + cond_oor_thr;
  "mix" means mix_oor. The in-range strata are reported. They also act as the non-inferiority controls G5, and swap and
  add_nowrap are the untouched-skill controls G4.
- **"Filler below trace"** means trace - filler > 0 on both primaries (G3). Decision point 3 is checked first.

## 8. Reported, not gating

- Per stratum and side: accuracy (Wilson CI) and the parse rate.
- Cluster-bootstrap contrasts for every pair of sides.
- Trace fields per stratum: format_ok, cmp_ok, ok_ok, calc_ok, after_follows_calc and calc_follows_cmp. For filler:
  format_ok and fill_exact.
- Every side's branch choice, read from its after (correct / wrong / neither), on all conditional strata.
- Unreduced-sum answers on mix and wrap.
- Chains, per stratum (CH_IN, CH_X, ALL) and side:
  - normalised regret at k = 1, 2, 4 (`rcwmt.decisions.g_only`, split dev) and exact state, with parent-paired
    contrasts;
  - the copy-s0 floor;
  - the per-step to k4 product law: p(class) from each side's gold-input chain steps (R8 convention). The predicted k4
    exact is the mean over chains of the product of p over the chain's 4 gold step classes. It is compared with the
    observed k4 exact and with the pooled p^4.

## 9. GPU plan (`docs/t2b/register/GEN_PLAN.json`)

One turn: about 3.66 h at the registered planning rate (2.7 short gens/s) and 3.15 h at the observed rate (3.8).

| stage | estimate |
|---|---|
| T_plain | 34.2 min |
| T_trace / T_filler | 38.8 min each |
| A0 | 7.1 / 5.6 min |
| E1 | 37.1 / 26.4 min |
| E2 | 63.3 / 45.0 min |

- Training is T2's measured 1,858.8 s, scaled by each arm's processed tokens (483k / 552k / 552k against 464k).
- Trace and filler answers decode 2.34x longer than plain.
- 10,576 generations at most.

## 10. Tests and records (`docs/t2b/register/`)

- **AUDIT.json** (`t2br_make_data.py --verify`): golds, targets and token equality are re-derived, with a fresh
  name-skipping harvest, and the audit is rerun.
- **SCORER_SELFTEST.json** (`t2br_analysis.py --selftest`) checks:
  - golds 100% for every arm;
  - 14,100 malformed or wrong answers, none accepted;
  - floors exact;
  - metric identities;
  - chain pipeline identities.
- **DRYRUN.json / DRYRUN.md** (`t2br_analysis.py --dryrun`): 6 synthetic scenarios with known verdicts. These are:
  - GO;
  - the two decision-point-3 stops;
  - lost controls;
  - an in-distribution loss;
  - malformed traces.
- **TINY_TEST.json** (`t2br_test_tiny.py`): tiny random Qwen3 and the real vendored trainer. It covers:
  - formats;
  - surface diversity;
  - chain equality;
  - a full run;
  - adapter switching;
  - crash and resume in eval and in training;
  - deferral;
  - deterministic stops;
  - the one-waiter rule and the run lock;
  - an oracle run on the full bank through the analysis;
  - the freeze tool;
  - the plan.
- **GEN_PLAN.json** (`t2br_run.py --plan`, CPU, no torch).
- **queue_hook_t2br.sh.draft:** NOT installed. It is a frozen input. The runner refuses (exit 3, not persisted) if an
  installed `scripts/queue_hooks/*t2br*.sh` is not the draft byte for byte.

## 11. Decision points (roadmap sequencing)

- **DP1:** PREREG-3 says TRIGGER. Rebuild with `--surface-diversity on` and train with `--train-variant sd`.
- **DP3:** trace - filler < 0.10 on both primaries. Stop. Report "extra decode compute, not supervision".
- **GO:** seeds 1-2 of the winning arms (`--seed 1/2`). The roadmap then requires the same rule in all 3 seeds, a sealed
  R bank, and k4 regret <= -0.10.
