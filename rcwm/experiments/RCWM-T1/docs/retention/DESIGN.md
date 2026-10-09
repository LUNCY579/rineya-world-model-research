# Correctness-based retention suite (RCWM roadmap item 4, joint item 5c)

**Status: DRAFT kit (CPU-built, not frozen, no GPU run).** Nothing here has seen a model output. The kit freezes with
`python -B scripts/ret_gen.py freeze --approval "<who, when, DECISIONS entry>"`. The freeze:
- re-checks that this file states `RULE_DEV` and `RULE_CONFIRM` verbatim, and that the tokenizer on disk equals the pins;
- hashes every kit file, this file and the six pre-freeze records into `docs/retention/RET_FREEZE.json` (create-only),
  each record bound to the code (and this file) it was made with;
- records the tokenizer pins, the chat-template pin and the Python interpreter.

Any later change needs the freeze moved aside and logged, and the dev draws re-minted (allowed only while no
confirmation draw of that setting exists). Before the freeze the six records are working records and are rewritten
by every test run; after it they are refused.

## 1. Why

- The PREREG-2 forgetting guard passed at 0.50x, yet cumulative drift from Supa is 1.94x KL(Supa||Sparse). Supa/ON
  64-token agreement is 0.37.
- First-token KL never flagged Titan's measured abstention loss (p 6e-5; DECISIONS D-32 iv). Retention must therefore
  be measured by **correctness**.
- BWM's adapters have no retention evidence at all.

The suite is 8 code-graded categories x 75 items. Categories and difficulty levels are chosen once per
surface-diversity setting, on a Supa-only dev draw. **Every confirmation mints its own sealed draw**, so no instrument
is reused.

Templated tasks are narrow. A RETAINED verdict is a **floor**: it shows these skills survive. It does not prove that
general skill is retained.

## 2. Files

| file | role |
|---|---|
| `scripts/ret_common.py` | constants (categories, levels, dev seeds 20261121 / 20261125, dev thresholds 0.70 / 0.40, gate constants, 48 tokens, tokenizer and chat-template pins), paths, create-only IO, freshness normal form, freeze verifier |
| `scripts/ret_grade.py` | primary typed whole-answer grader and lenient value-extraction grader (both decision-bearing, section 7) |
| `scripts/ret_gen.py` | frozen generator, `freeze`, `preview`, `mint`, `verify`, `stats`, freshness harvest, token audit |
| `scripts/ret_run.py` | GPU runner: `--plan`, dev mode (Supa only), confirm mode (Titan-Sparse + T2 s0 OFF/ON in one process, then Supa; optional extra adapters) |
| `scripts/ret_analysis.py` | `dev-select`, `analyze` (registered, once per confirmation run), `oc` (operating characteristics), `dryrun` |
| `scripts/ret_test_graders.py` | grader / independent gold re-derivation / tense / strata / generator / harvest / mint / freeze / preview / plan tests (no torch) -> `GRADER_TEST.json` |
| `scripts/ret_test_tiny.py` | tiny random Qwen3 runner test incl. resume, cap, stops, tokenizer pins, analysis hand-off -> `TINY_TEST.json` |
| `docs/retention/DRYRUN.json`, `RET_PLAN.json` | analysis dry-run record (with the operating characteristics); GPU plan record |
| `docs/retention/DEV_DRAW_PREVIEW.json`, `DEV_DRAW_PREVIEW_SD.json` | pre-mint audit of both pre-registered dev draws (bytes, stats, golds, tokens, freshness, their mutual overlap), one harvest pass |
| `docs/retention/draws/` | sealed draws (`<purpose>_<seed>.jsonl` + `.SEAL.json`), written only by `mint` |

## 3. Instrument

**Prompt format.** Each item has two turns:
- a system turn: "Answer the user's question. Give only the final answer in the requested format, with no
  explanation.";
- a user turn that ends with an explicit answer-format instruction.

The prompt is rendered with the cores' chat template, `enable_thinking=False`, like the programme's retention probes
(architecture_benchmark_001 monitor probe: system + user turns). Decoding:
- greedy, at most 48 new tokens;
- EOS [151645, 151643];
- FP32.

**Categories.** Each has three difficulty levels. The dev draw holds every level; a confirmation draw holds one level
per category, chosen by the dev rule in section 7. People are named by a full name wherever a template is short
(unit, date, percentage/ratio and format items), so independent draws practically never share a user turn.

| category | L1 | L2 | L3 | gold kind |
|---|---|---|---|---|
| arith_word | 2 ops (+/-), start 5-30 | 3 ops, one x2/x3 | 4 ops, one xk and one /k (exact) | number (+ object noun) |
| unit_conv | forward, integer source (12 factors) | reverse; a .5 source with integer result | two-step chains (h-min-s, yd-ft-in, ...), forward; reverse | number (+ unit aliases) |
| date_time | clock + 5-55 min; weekday + 1-13 days | clock + h:mm across noon; weekday + 15-60; minutes between two times | date + 5-45 days across a month (2027); clock across midnight; weekday 8-40 days earlier | clock / weekday / date / number |
| count_sort | 5 elements: count a word, count > t, sort asc | 7: count a word, count > t, sort desc, sort words | 9: negatives, a duplicate, words sharing first letters | number / num_list / word_list |
| pct_ratio | p% of N (10/20/25/50); unit-price scaling | % change; what %; simplify a:b | p% of N (12.5..62.5); two successive changes; split T in a:b | number / ratio |
| kv_recall | 150-300-token passage of records (6 themes), ask one value | first names repeat (full name needed) | + 2 later corrections; half the questions target a corrected record (gold = latest) | kv (code or word) + distractors |
| unknown_twin | absent person | same person, attribute never mentioned | near-miss name (first name of one record + last name of another) | **pair**: twin a = kv value, twin u = literal UNKNOWN |
| format_follow | word in capitals; yes/no | n-th word; first letter (lowercase); last word | reverse a word; JSON object with two keys; hyphen-spelled capitals | exact (case declared) / json |

**Strata.** Each cell's 75 items are dealt round-robin over the cell's subtypes (`ret_gen.STRATA`; kv_recall: the six
themes, at L3 x {uncorrected, corrected}; unknown_twin: the six themes) and shuffled. Every draw of a cell, dev or
confirmation, therefore has the same subtype mix (e.g. count_sort L3: 19 / 19 / 19 / 18).

**Passage questions are in the present tense.** Each question asks for the record's CURRENT value ("In which freezer
is X's sample tray stored?", "Which instrument is X's room booked for?"); the corrections are past events ("Later, ...
was moved to ..."). A corrected record therefore has one reading, its latest value. `ret_test_graders` checks the tense
of every template and of every generated question about a corrected record, and re-derives every gold from the prompt
text with an independent solver (section 4).

**unknown_twin.** Both twins share one passage and one instruction ("If the passage does not contain the answer,
reply with exactly UNKNOWN."). An item is correct only if **both** twins are. Always answering UNKNOWN (over-abstention)
and never abstaining both score 0. The components are reported separately:
- literal UNKNOWN on unanswerable twins;
- correct answers on answerable twins;
- UNKNOWN on answerable twins (over-abstention).

**Item counts.** A dev draw is 1,800 items (2,025 prompts). A confirmation draw is 600 items (675 prompts) if all
8 categories are selected.

**Passages.** Records are added until a conservative character-based bound reaches the level's token band. `mint`
re-checks every passage with the pinned Titan tokenizer and gates on 150 <= tokens <= 300.

## 4. Grading (normalisation, in order; `ret_grade.py`)

The **primary** grader parses the WHOLE answer as the item's declared kind. It follows the programme's typed retention
contract (TYPED_SCORING_CONTRACT.md):

- N0: EOS within the 48-token cap is required. A truncated answer is wrong.
- N1: Unicode NFKC. Typographic quotes, the Unicode minus and en/em dashes become ASCII; NBSP becomes a space.
- N2: Outer whitespace is stripped, and inner whitespace (including newlines) is collapsed to single spaces.
- N3: Case is folded, except for case-sensitive kinds:
  - the literal UNKNOWN;
  - exact items that demand a case (capitals, a lowercase letter, a reversed lowercase word, a hyphen spelling);
  - JSON.
- N4: At most one closing period is removed. Then at most one wrapping pair is removed: `** **`, `__ __`, `* *`,
  `` ` ` ``, `" "`, `' '`. Then a second period is removed, but only if none was removed yet. For JSON, a single
  surrounding code fence is removed first.
- N5: At most one prefix is removed: `final answer:`, `answer:`, `the answer is`, `it is`, `it will be`. Clock answers
  may instead start with `at`; weekday and date answers with `on`.
- N6: The remainder must be consumed entirely by the kind's parser:
  - **Numbers** are exact rationals: integers, finite decimals, or one a/b fraction. A fraction must be in lowest
    terms and is wrong for a whole-number answer (`48/2` or `24/1` for 24). Commas are allowed only in valid groups
    of three. Scientific notation and number words are not accepted. A declared unit alias may follow the number.
  - **Clock**: without am/pm it is a 24-hour time; with am/pm it is a 12-hour time; `noon` and `midnight` are accepted.
  - **Weekday**: the full English name.
  - **Date**: the full month name and the day, in either order, with an optional ordinal suffix, an optional "of", and
    an optional year (which must equal the item's).
  - **Lists**: comma-separated, with an optional "and" before the last element; order and multiplicity must be exact.
  - **Ratio**: `a:b` or `a to b`, exactly in the requested form.
  - **kv**: exactly the value.
  - **unknown**: exactly `UNKNOWN`.
  - **json**: one object equal to the gold, with equal value types and no duplicate keys.

Hedges, ranges, explanations, second answers, wrong units, abbreviations and brackets are all wrong.

The **lenient** grader extracts a value from free text. It is the go rule's **second grader** (section 7), so a format
change is neutral and a strict-format gain cannot offset a skill loss:
- the last number, time, weekday, date or ratio;
- the in-vocabulary list order;
- a kv value with no other passage value present;
- an abstention phrase with no passage value.

Alternatives are wrong under it: two or more distinct candidate values of the kind together with "or" / "either"
("350 or 351", "Thursday or Friday"); "yes" together with "no"; an exact answer next to "or" / "either"; two
different JSON objects. This is deliberately conservative: an equivalent restatement joined by "or"
("150 minutes, or 2.5 hours") is also wrong.

**Tests.**
- Grader tests: 207 adversarial primary cases, 82 adversarial lenient cases and 9 malformed golds.
- Every draw's golds pass a self-test before sealing. Canonical answers must score 100%, and negatives 0%. The
  negatives are: empty answers, unterminated answers, the answer plus an explanation, the answer plus "or ...",
  another item's answer, UNKNOWN on answerable twins, and passage values, `unknown` or `Unknown.` on unanswerable
  twins.
- An **independent solver** in `ret_test_graders` re-derives every gold of four full dev-plan draws from the prompt
  text alone: both pre-registered dev seeds, and seed 7 with surface diversity off and on. It uses its own regexes,
  unit table, `datetime` calendar, sorting, latest-value record semantics, and absence checks for the unanswerable
  twins. Expected: 0 mismatches.

## 5. Draws, seeds, sealing, freshness

**Determinism.** A draw is a pure function of:
- the seed;
- the plan;
- the surface-diversity flag;
- the frozen code;
- the interpreter's `random` module.

Each (category, level) cell has its own `random.Random(f"{seed}|{cat}|L{level}|sd{0|1}")`. CPython hashes string
seeds with SHA-512, so the draw does not depend on PYTHONHASHSEED; this is tested in a child process. The seal
records the Python version, and `verify` reports a different interpreter next to any mismatch.

**Seeds.**
- Dev, surface diversity OFF: the fixed `DEV_SEED = 20261121`.
- Dev, surface diversity ON: the fixed, pre-registered `DEV_SEED_SD = 20261125`.
- No earlier RCWM-T1 builder used either seed. Their two dev draws share no user turn and no passage sentence (checked
  in the preview and in the tests), so both can be minted.
- Each confirmation: a fresh `secrets.randbits(62)`, never reused. Both dev seeds are refused for a confirmation.

**Seal.** `<draw>.SEAL.json` records:
- the seed and its source;
- the plan and the file sha256;
- the generator code sha256;
- the kit-freeze sha256;
- the dev-selection sha256 (confirmation draws);
- the interpreter;
- stats, the gold self-test, the token audit and every freshness result.

`verify` regenerates the draw byte for byte. The runner and the analysis refuse any draw that does not hash to its
seal.

**Freshness.** Minting refuses on any gated hit (F0, F1, F2); a refusal means minting again (a confirmation draws a
new seed).
- F0 (gate): no prompt contains a `{before, event}` transition payload. The programme's payload harvest is therefore
  checked, not assumed, to be disjoint.
- F1 (gate): no normalised user turn equals:
  - any harvested string;
  - any user turn of an earlier draw.
- F2 (gate): no item-unique sentence of 8 or more words equals a sentence of a harvested string. No passage record
  sentence (one that names a person) of 8 or more words is shared with an earlier draw.
- F3 (report only):
  - 6-7-word sentences and template sentences found in the corpora;
  - operative-content overlaps with earlier draws.

**Sources.** Every string value of 16 or more characters is harvested from:
- the retention-records tree, walked exactly as `l1_make_ood.harvest_historical` walks it: `*.jsonl` then `*.json`;
  **files named `HSKILL_CONFIRM_V2*` are skipped by name before any stat or open (DECISIONS I-28)**; files over 60 MB
  are skipped;
- RCWM-T1 `data/`, `docs/l1`, `docs/prereg3`;
- the Rineya corpora: General Reasoning v2.0 and v3.0, Short Response Behavior v1.0, short-response-model data,
  Long Context Demo, Puzzle-Reasoning-51k, LCR SFT, Invisible Limit Lab;
- the **BWM research bundle** (`Rineya_BWM_001_Research_Bundle`: its data, banks, run outputs and exports, about
  0.65 GB of JSON), so a draw scored on BWM adapters is also fresh against BWM's training data. The same name rule
  applies.

That is about 5 GB, read in one pass; the preview records the measured time and peak memory. Nothing is selected on
harvested content: a hit only refuses the mint. Stored model outputs are read like any other string, and only to
refuse.

**Surface diversity** (DECISION POINT 1 uniformity) is OFF by default and switched on with `--surface-diversity`. It
changes:
- paraphrased system prompts and answer instructions;
- the kv question template;
- the passage field names (door/entry/keypad code, depot/base/yard, ...).

The setting is part of every cell seed, so OFF and ON draws share no items. **Each setting has its own pre-registered
dev seed, its own dev draw and its own dev selection** (`DEV_SELECTION.json` / `DEV_SELECTION_SD.json`). A
confirmation draw uses the selection of its own setting. If PREREG-3 returns TRIGGER, switching the suite to diversified
forms needs no code change: mint the SD dev draw, run it, select, then mint SD confirmations.

## 6. Runner (`scripts/ret_run.py`)

**Dev mode.** Supa 1.7 only, on a dev draw, with no log-probs.

**Confirm mode.** Two stages under ONE lease, `rcwm-t1:retention`:
- SPARSE: Titan-Sparse (fingerprint 818be9c6b6a3) with the T2 s0 LoRA **unmerged** (PEFT adapter `t2s0`, sha-pinned
  to PREREG-2) and any `--extra-adapter NAME=DIR`, all in one process. Side `off` = `disable_adapter()`; each adapter
  side = `set_adapter`.
- SUPA: Titan-Sparse is unloaded (VRAM checked empty), then Supa 1.7 is loaded (fingerprint a9952c55f5a1), with any
  `--extra-adapter-supa`.

**Per prompt** the runner stores:
- the text;
- EOS termination;
- the token count;
- the first generated token;
- in confirm mode, the full first-token log-prob vector from a prefill forward of the same batch (fp16, clamped at
  -60, as R14 stores them) -> `lp_<side>.npy`.

**Guards.** The lease is `rcwmt.runtime.gpu_lease`: alternation, the thermal flag, the duty cycle, the RAM floor and
the dedicated-memory holder check. The D-22 stage rule applies with the 4.5 h cap. Before the lease, the runner checks:
- the kit freeze, including its tokenizer and chat-template pins;
- the draw seal;
- the freshness record;
- the dev selection of the draw's setting;
- the **tokenizer files** the runtime loads (`tokenizer.json`, `tokenizer_config.json` with the chat template,
  `vocab.json`, `merges.txt`; byte-identical across Titan-Sparse and Supa 1.7) against `TOKENIZER_PINS`;
- the adapter pins and hashes, which are hashed again at load;
- the one-waiter rule (R14);
- the run lock.

After each load it checks:
- the parameter fingerprint;
- torch / transformers / peft against the registered runtime;
- the sha256 of the loaded tokenizer's **chat template** against `CHAT_TEMPLATE_SHA`.

A mismatch stops the run before anything is generated (persisted). The tokenizer hashes and the chat-template hash of
every load are recorded in `provenance.json`, and the registered analysis requires them.

**Outputs.** Outputs are create-only. A stage is written only when all its sides are done, and it is resumable (a
partial stage is moved to `_orphans/` and regenerated).

**Exit codes.**
- 0: complete.
- 3: deterministic stop, persisted when it concerns the run.
- 5: resumable (cap).
- 1: crash.

## 7. Pre-stated rules (verbatim; `RULE_DEV` and `RULE_CONFIRM` in `scripts/ret_analysis.py`)

**Dev selection rule.**

> For each category c and level l in {1, 2, 3}, a(c, l) is Supa 1.7's primary-grader accuracy on the 75 items of that cell of the dev draw of a surface-diversity setting (dev seed 20261121 with surface diversity off, 20261125 with it on; an unknown_twin item counts as correct only if both twins are). The selected level of c is the highest l with a(c, l) >= 0.70; if no level reaches 0.70, it is the highest l with a(c, l) >= 0.40. A category with no level at 0.40 or above is EXCLUDED from every confirmation draw of that setting (reported, never gated). The selection is written once per setting (DEV_SELECTION.json for off, DEV_SELECTION_SD.json for on; create-only) before any confirmation draw of that setting is minted, and every confirmation draw uses the selection of its own setting.

**Confirmation decision / go rule.**

> For each gating comparison of an arm A with its reference R (T2 s0 ON vs adapter OFF = Titan-Sparse; each extra Titan-Sparse adapter vs OFF; each extra Supa adapter vs Supa 1.7) on a sealed confirmation draw minted after the dev selection of its surface-diversity setting, for each of the two graders g (the primary typed grader and the lenient extraction grader) and each selected category c (75 items): lost = items R answers correctly and A does not; gained = items A answers correctly and R does not; net = gained - lost (an unknown_twin item is correct only if both twins are). p = two-sided exact McNemar p-value (binomial on the lost + gained discordant items, 1/2); p_Holm = Holm-Bonferroni adjustment of p over the selected categories of that comparison and grader. Category c has a DETECTED LOSS under g iff lost > gained and p_Holm < 0.05, and EXCEEDS ITS MARGIN under g iff net < -11 (more than 11 of its 75 items lost net). The pooled net under g is the sum of net over the selected categories; it EXCEEDS ITS MARGIN iff it is below -0.04 x the number of selected items (-24 for 600 items). Comparison verdict: LOSS DETECTED iff some category has a detected loss under either grader; otherwise MARGIN EXCEEDED iff some category or the pooled net exceeds its margin under either grader; otherwise RETAINED. GO (the retention guard passes for arm A) iff RETAINED, reported as GO (k/m binding), where k of the m selected categories are binding, i.e. R answers at least the fewest items whose loss with no gain would fail the category alone (9 when 8 categories are selected); LOSS DETECTED and MARGIN EXCEEDED are NO-GO. Report-only, never gating: the per-category flag net < -5 of 75 (a point estimate, not a non-inferiority test), Titan-Sparse (OFF) vs Supa 1.7 and T2 s0 vs Supa 1.7 (core and cumulative drift, verdicts computed the same way), excluded categories, accuracies with Wilson 95% intervals, EOS-termination rates, abstention components, the operating characteristics, and the AUROC of the per-item first-token KL(R || A) for predicting correctness flips. The suite is templated and narrow: a RETAINED verdict is a floor, not a proof of general retention.

**Why the draft rule was replaced.** The draft gated each category on net >= -5 plus a Holm-McNemar clause. Two
problems:
- No (lost, gained) pair with net >= -5 has a raw two-sided p < 0.05 (checked exhaustively over 0..75 in DRYRUN). So
  the McNemar/Holm clause could never change GO, and the gate was a per-category point estimate.
- With no true effect, the draft returned NO-GO 11-12% of the time at 10% churn and 32-50% at 20-30% churn.

The roadmap's net >= -5 of 75 is kept as a reported flag.

**How the gate was set.** The gate has three clauses per grader:
- Holm-McNemar **detection**, which adapts to the observed churn and catches concentrated losses when churn is low;
- a **per-category margin**, net < -11, for concentrated losses at moderate churn;
- a **pooled margin**, below -4% of the selected items, for diffuse losses.

The two margins were set by the simulation below, before any model output. The target was a single-grader false
NO-GO of at most 0.025 at every simulated churn up to 30% and reference accuracy 0.30-0.80. Two graders together then
stay at or below 0.05 by the union bound.

**Operating characteristics** (`ret_analysis.py oc`; recorded in DRYRUN.json). One grader; 8 selected categories of 75
items; 40,000 seeded replicates per row. In the null rows, churn is balanced: expected lost = expected gained per
category, so there is no true effect. In the loss rows, a real loss is added on top of that churn. "Draft" is the
replaced rule.

| background discordance per category | ref acc | P(NO-GO) go rule | via detection | via category margin | via pooled margin | superseded draft rule |
|---|---|---|---|---|---|---|
| 3.75 (5%) | 0.30 | 0.000 | 0.000 | 0.000 | 0.000 | 0.015 |
| 7.5 (10%) | 0.30 | 0.001 | 0.001 | 0.000 | 0.000 | 0.109 |
| 11.25 (15%) | 0.30 | 0.004 | 0.002 | 0.000 | 0.002 | 0.227 |
| 15 (20%) | 0.30 | 0.006 | 0.002 | 0.001 | 0.004 | 0.324 |
| 22.5 (30%) | 0.30 | 0.015 | 0.001 | 0.005 | 0.011 | 0.443 |
| 3.75 (5%) | 0.45 | 0.000 | 0.000 | 0.000 | 0.000 | 0.018 |
| 7.5 (10%) | 0.45 | 0.002 | 0.001 | 0.000 | 0.000 | 0.122 |
| 11.25 (15%) | 0.45 | 0.005 | 0.002 | 0.001 | 0.002 | 0.253 |
| 15 (20%) | 0.45 | 0.009 | 0.002 | 0.003 | 0.006 | 0.360 |
| 22.5 (30%) | 0.45 | 0.025 | 0.002 | 0.013 | 0.013 | 0.495 |
| 3.75 (5%) | 0.80 | 0.000 | 0.000 | 0.000 | 0.000 | 0.019 |
| 7.5 (10%) | 0.80 | 0.002 | 0.001 | 0.000 | 0.000 | 0.122 |
| 11.25 (15%) | 0.80 | 0.003 | 0.001 | 0.001 | 0.002 | 0.235 |
| 15 (20%) | 0.80 | 0.007 | 0.001 | 0.003 | 0.004 | 0.316 |
| 22.5 (30%) | 0.80 | 0.014 | 0.000 | 0.008 | 0.007 | 0.389 |

| real loss (ref acc 0.80) | background 7.5 | background 15 |
|---|---|---|
| 6 items in ONE category | 0.026 (draft 0.567) | 0.043 (draft 0.608) |
| 9 items in ONE category | 0.128 (draft 0.902) | 0.141 (draft 0.841) |
| 12 items in ONE category | 0.464 (draft 0.994) | 0.384 (draft 0.969) |
| 15 items in ONE category | 0.868 (draft 1.000) | 0.716 (draft 0.999) |
| 2 items in EVERY category (16 pooled) | 0.094 (draft 0.458) | 0.130 (draft 0.641) |
| 3 items in EVERY category (24 pooled) | 0.390 (draft 0.693) | 0.351 (draft 0.796) |
| 4 items in EVERY category (32 pooled) | 0.781 (draft 0.882) | 0.651 (draft 0.908) |
| 5 items in EVERY category (40 pooled) | 0.971 (draft 0.973) | 0.883 (draft 0.970) |

How to read it:
- **False NO-GO.** The worst single-grader rate is 0.025, at 30% churn. Two graders together are at most
  0.050 (union bound), and equal the single-grader rate when the graders agree item by item. At 20% churn or
  less the single-grader rate is at most 0.010.
- **Power.** The rule reliably catches a loss of 15 items in one category (0.72-0.87) and 4-5 items in every
  category (0.65-0.97). It catches 9-12 items in one category only sometimes (0.13-0.46), and 6 rarely.
- **Main limit.** 75 items per category cannot resolve losses below about 12% of one skill. A RETAINED verdict
  therefore bounds what was lost; it does not show that nothing was.

**Consequences of the rule** (checked in DRYRUN.json):
- With 8 categories, detection needs at least 9 losses with no gain (Holm 8 x 2^-8 = 0.031). 8 losses (0.0625) are
  not detected.
- A category net of -11 is within its margin; -12 fails it.
- A uniform net of -3 in every category (pooled -24) is RETAINED; a pooled -25 is MARGIN EXCEEDED.
- Format gains cannot hide skill losses. Under the primary grader, 12 strict-format gains offset 12 skill losses (net
  0). The lenient grader sees 12 lost and 0 gained: LOSS DETECTED.
- Binding: a category whose reference answers fewer than 9 items cannot fail alone. GO is reported as "GO (k/m
  binding)" so a partly vacuous pass is visible.
- Titan-Sparse vs Supa 1.7 and T2 s0 vs Supa 1.7 are report-only. The core is frozen, and its loss relative to Supa
  (e.g. abstention, D-32 iv) is a known property; it is not a decision about an adapter.

**KL calibration (report-only).** For each item, KL(R || A) is computed over the renormalised stored first-token
distributions; for unknown_twin items it is the mean of the two prompts. Reported:
- the AUROC for any correctness flip (primary grader), with a 2,000-resample bootstrap CI (seed 20261122);
- the AUROC for losses among the items R answers correctly.

## 8. Workflow and GPU estimate

1. CPU, before the freeze:
   - `python -B scripts/ret_test_graders.py`
   - `python -B scripts/ret_test_tiny.py`
   - `python -B scripts/ret_analysis.py dryrun --record`
   - `python -B scripts/ret_run.py --plan --record`
   - `python -B scripts/ret_gen.py preview` (about 4-5 min, one harvest pass, both dev draws)
2. `python -B scripts/ret_gen.py freeze --approval "..."` (after operator sign-off on section 7).
3. `python -B scripts/ret_gen.py mint --purpose dev` (CPU, about 4-5 min including the harvest).
4. GPU: `python -B scripts/ret_run.py --mode dev --draw docs/retention/draws/dev_20261121.jsonl`
5. `python -B scripts/ret_analysis.py dev-select --run runs/retention/dev_20261121`
6. Per confirmation, the following steps. A new adapter means a new confirmation, which mints a new draw.
   - `python -B scripts/ret_gen.py mint --purpose confirm`
   - GPU: `python -B scripts/ret_run.py --mode confirm --draw docs/retention/draws/confirm_<seed>.jsonl [--extra-adapter NAME=DIR]`
   - `python -B scripts/ret_analysis.py analyze --run runs/retention/confirm_<seed>`

For the surface-diversity variant, steps 3-6 are the same with `--surface-diversity` at each `mint` (dev seed 20261125,
`DEV_SELECTION_SD.json`).

**GPU estimate** (`RET_PLAN.json`). The planning rate is a conservative 1.0 generation/s. The optimistic rate is
PREREG-2's registered 2.7/s; PREREG-2 observed 3.8/s on short answers. Each model load costs 120 s, and the
first-token prefill adds 10%.

| job | turns | GPU-h at 1.0/s | GPU-h at 2.7/s |
|---|---|---|---|
| dev (Supa, 2,025 prompts), per setting | 1 | 0.60 | 0.24 |
| confirm (off + t2s0 + supa, 675 prompts each) | 1 | 0.69 | 0.30 |
| each extra arm (675 prompts) | same turn | +0.21 (+10 s adapter load) | +0.08 |

Disk: about 0.205 GB of log-probs per confirm arm (675 x 151,936 fp16).

## 9. Limits and open points

- The instrument is templated and English-only. It probes short-answer arithmetic, conversion, time, counting,
  percentages, recall, abstention and format compliance, not open-ended generation.
- Power is limited (section 7): a loss of 6-12 of 75 items in a single category is often missed.
- The dev rule takes the highest level at which Supa reaches 0.70, so that most items are at risk and churn near the
  capability boundary is lower. If no level reaches 0.70, it falls back to the highest level at 0.40. The reference for
  ON-vs-OFF is Titan-Sparse, whose accuracy may be lower; the binding count makes that visible.
- **No anchor reproduction** of stored outputs is run, unlike PREREG-3's A0. Arm identity rests on:
  - parameter fingerprints;
  - adapter pins;
  - the registered runtime;
  - the tokenizer-file and chat-template pins.

  An optional anchor was declined. It would add about 10 GPU-minutes per confirmation, and it would test a different
  code path (`rcwmt.offtask` batching), not this runner.
