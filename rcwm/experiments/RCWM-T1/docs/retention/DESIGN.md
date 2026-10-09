# Correctness-based retention suite (RCWM roadmap item 4, joint item 5c)

**Status: DRAFT kit (CPU-built, not frozen, no GPU run).** Nothing here has seen a model output. The kit freezes with
`python -B scripts/ret_gen.py freeze --approval "<who, when, DECISIONS entry>"`, which hashes every kit file and the
four test records into `docs/retention/RET_FREEZE.json` (create-only). Any later change needs the freeze moved aside
and logged, and the dev draw re-minted (allowed only while no confirmation draw exists).

## 1. Why

- The PREREG-2 forgetting guard passed at 0.50x, yet cumulative drift from Supa is 1.94x KL(Supa||Sparse). Supa/ON
  64-token agreement is 0.37.
- First-token KL never flagged Titan's measured abstention loss (p 6e-5; DECISIONS D-32 iv). Retention must therefore
  be measured by **correctness**.
- BWM's adapters have no retention evidence at all.

The suite is 8 code-graded categories x 75 items. Categories and difficulty levels are chosen once, on a Supa-only
dev draw. **Every confirmation mints its own sealed draw**, so no instrument is reused.

Templated tasks are narrow. A RETAINED verdict is a **floor**: it shows these skills survive. It does not prove that
general skill is retained.

## 2. Files

| file | role |
|---|---|
| `scripts/ret_common.py` | constants (categories, levels, DEV_SEED 20261121, margin 5, alpha 0.05, 48 tokens), paths, create-only IO, freshness normal form, freeze verifier |
| `scripts/ret_grade.py` | primary typed whole-answer graders (decision-bearing) and lenient value-extraction graders (report-only) |
| `scripts/ret_gen.py` | frozen generator, `freeze`, `mint`, `verify`, `stats`, freshness harvest, token audit |
| `scripts/ret_run.py` | GPU runner: `--plan`, dev mode (Supa only), confirm mode (Titan-Sparse + T2 s0 OFF/ON in one process, then Supa; optional extra adapters) |
| `scripts/ret_analysis.py` | `dev-select`, `analyze` (registered, once per confirmation run), `dryrun` |
| `scripts/ret_test_graders.py` | grader / generator / harvest / mint / plan tests (no torch) -> `GRADER_TEST.json` |
| `scripts/ret_test_tiny.py` | tiny random Qwen3 runner test incl. resume, cap, stops, analysis hand-off -> `TINY_TEST.json` |
| `docs/retention/DRYRUN.json`, `RET_PLAN.json` | analysis dry-run record; GPU plan record |
| `docs/retention/DEV_DRAW_PREVIEW.json` | pre-mint audit of the dev draw bytes (stats, golds, tokens, freshness) |
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
per category, chosen by the dev rule in section 7.

| category | L1 | L2 | L3 | gold kind |
|---|---|---|---|---|
| arith_word | 2 ops (+/-), start 5-30 | 3 ops, one x2/x3 | 4 ops, one xk and one /k (exact) | number (+ object noun) |
| unit_conv | forward, integer source (12 factors) | reverse, or a .5 source with integer result | two-step chains (h-min-s, yd-ft-in, ...) | number (+ unit aliases) |
| date_time | clock + 5-55 min; weekday + 1-13 days | clock + h:mm across noon; weekday + 15-60; minutes between two times | date + 5-45 days across a month (2027); clock across midnight; weekday 8-40 days earlier | clock / weekday / date / number |
| count_sort | 5 elements: count a word, count > t, sort asc | 7: + sort desc, sort words | 9: negatives, a duplicate, words sharing first letters | number / num_list / word_list |
| pct_ratio | p% of N (10/20/25/50); unit-price scaling | % change; what %; simplify a:b | p% of N (12.5..62.5); two successive changes; split T in a:b | number / ratio |
| kv_recall | 150-300-token passage of records (6 themes), ask one value | first names repeat (full name needed) | + 2 later corrections; half the questions target a corrected record (gold = latest) | kv (code or word) + distractors |
| unknown_twin | absent person | same person, attribute never mentioned | near-miss name (first name of one record + last name of another) | **pair**: twin a = kv value, twin u = literal UNKNOWN |
| format_follow | word in capitals; yes/no | n-th word; first letter (lowercase); last word | reverse a word; JSON object with two keys; hyphen-spelled capitals | exact (case declared) / json |

**unknown_twin.** Both twins share one passage and one instruction ("If the passage does not contain the answer,
reply with exactly UNKNOWN."). An item is correct only if **both** twins are. Always answering UNKNOWN (over-abstention)
and never abstaining both score 0. The components are reported separately:
- literal UNKNOWN on unanswerable twins;
- correct answers on answerable twins;
- UNKNOWN on answerable twins (over-abstention).

**Item counts.** The dev draw is 1,800 items (2,025 prompts). A confirmation draw is 600 items (675 prompts) if all
8 categories are selected.

**Passages.** Records are added until a conservative character-based bound reaches the level's token band. `mint`
re-checks every passage with the Titan tokenizer and gates on 150 <= tokens <= 300.

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
  - **Numbers** are exact rationals: integers, finite decimals, or one a/b fraction. Commas are allowed only in valid
    groups of three. Scientific notation and number words are not accepted. A declared unit alias may follow the
    number.
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

The **lenient** grader extracts a value from free text and is **report-only**:
- the last number, time, weekday, date or ratio;
- the in-vocabulary list order;
- a kv value with no other passage value present;
- abstention phrases.

Comparing the two graders separates format drift from skill loss. The grader tests cover 201 adversarial primary
cases, 19 lenient cases and 9 malformed golds. Every draw's golds also pass a self-test before sealing:
- canonical answers score 100%;
- negatives score 0%. The negatives are: empty answers, unterminated answers, the answer plus an explanation, the
  answer plus "or ...", another item's answer, UNKNOWN on answerable twins, and passage values, `unknown` or
  `Unknown.` on unanswerable twins.

## 5. Draws, seeds, sealing, freshness

**Determinism.** A draw is a pure function of:
- the seed;
- the plan;
- the surface-diversity flag;
- the frozen code.

Each (category, level) cell has its own `random.Random(f"{seed}|{cat}|L{level}|sd{0|1}")`. CPython hashes string
seeds with SHA-512, so the draw does not depend on PYTHONHASHSEED. This is tested in a child process.

**Seeds.**
- Dev: the fixed `DEV_SEED = 20261121`, which no earlier RCWM-T1 builder used.
- Each confirmation: a fresh `secrets.randbits(62)`, never reused. The dev seed is refused for a confirmation.

**Seal.** `<draw>.SEAL.json` records:
- the seed and its source;
- the plan and the file sha256;
- the generator code sha256;
- the kit-freeze sha256;
- the DEV_SELECTION sha256 (confirmation draws);
- stats, the gold self-test, the token audit and every freshness result.

`verify` regenerates the draw byte for byte. The runner and the analysis refuse any draw that does not hash to its
seal.

**Freshness.** Minting refuses on any gated hit (F0, F1, F2); a refusal means minting again with a new seed.
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
  Long Context Demo, Puzzle-Reasoning-51k, LCR SFT, Invisible Limit Lab.

That is about 4.3 GB, scanned in about 190 s with a peak of about 180 MiB. Nothing is selected on harvested content:
a hit only refuses the mint.

**Surface diversity** (DECISION POINT 1 uniformity) is OFF by default and switched on with `--surface-diversity`. It
changes:
- paraphrased system prompts and answer instructions;
- the kv question template;
- the passage field names (door/entry/keypad code, depot/base/yard, ...).

A confirmation draw must use the dev selection's setting.

## 6. Runner (`scripts/ret_run.py`)

**Dev mode.** Supa 1.7 only, on the dev draw, with no log-probs.

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
- the kit freeze;
- the draw seal;
- the freshness record;
- DEV_SELECTION;
- the adapter pins and hashes, which are hashed again at load;
- the one-waiter rule (R14);
- the run lock.

After each load it checks the fingerprint and torch / transformers / peft against the registered runtime.

**Outputs.** Outputs are create-only. A stage is written only when all its sides are done, and it is resumable (a
partial stage is moved to `_orphans/` and regenerated).

**Exit codes.**
- 0: complete.
- 3: deterministic stop, persisted when it concerns the run.
- 5: resumable (cap).
- 1: crash.

## 7. Pre-stated rules (verbatim; `RULE_DEV` and `RULE_CONFIRM` in `scripts/ret_analysis.py`)

**Dev selection rule.**

> For each category c and level l in {1, 2, 3}, a(c, l) is Supa 1.7's primary-grader accuracy on the 75 items of that cell of the dev draw (dev seed 20261121; an unknown_twin item counts as correct only if both twins are). The selected level of c is the highest l with a(c, l) >= 0.40. A category with no such level is EXCLUDED from every confirmation draw (reported, never gated). The selection is written once (DEV_SELECTION.json, create-only) and before any confirmation draw is minted.

**Confirmation decision / go rule.**

> For each comparison of an arm A with its reference R (Titan-Sparse = adapter OFF vs Supa 1.7; T2 s0 ON vs OFF; each extra Titan-Sparse adapter vs OFF; each extra Supa adapter vs Supa 1.7) on a sealed confirmation draw minted after DEV_SELECTION.json, and for each selected category c (75 items): lost = items R answers correctly and A does not; gained = items A answers correctly and R does not; net = gained - lost (primary grader; an unknown_twin item is correct only if both twins are). p = two-sided exact McNemar p-value (binomial on the lost + gained discordant items, 1/2); p_Holm = Holm-Bonferroni adjustment of p over the selected categories of that comparison. Category c is RETAINED iff net >= -5 and NOT (lost > gained and p_Holm < 0.05). Comparison verdict: RETAINED iff every selected category is RETAINED; LOSS DETECTED iff some category has lost > gained with p_Holm < 0.05; otherwise MARGIN EXCEEDED (some category has net < -5 without a detected loss). GO (the retention guard passes for arm A) iff RETAINED; LOSS DETECTED and MARGIN EXCEEDED are NO-GO. Report-only, never gating: T2 s0 vs Supa 1.7 (cumulative drift), excluded categories, accuracies with Wilson 95% intervals, the lenient grader, EOS-termination rates, abstention components, the can-fail binding note, and the AUROC of the per-item first-token KL(R || A) for predicting correctness flips. The suite is templated and narrow: a RETAINED verdict is a floor, not a proof of general retention.

**Consequences of the rule** (checked in DRYRUN.json):
- With 8 categories, a category fails at 6 losses and 0 gains, through the margin.
- 12 losses and 0 gains are a detected loss (Holm p 8 x 2^-11 = 0.0039 with 8 categories).
- A uniform net of -4 in every category (pooled -32) is RETAINED. The rule is per category, and the pooled net is
  reported.
- Binding note: a category whose reference gets fewer than 6 items right cannot fail. This is flagged, and never
  changes the verdict.

**KL calibration (report-only).** For each item, KL(R || A) is computed over the renormalised stored first-token
distributions; for unknown_twin items it is the mean of the two prompts. Reported:
- the AUROC for any correctness flip, with a 2,000-resample bootstrap CI (seed 20261122);
- the AUROC for losses among the items R answers correctly.

## 8. Workflow and GPU estimate

1. `python -B scripts/ret_test_graders.py`, `python -B scripts/ret_test_tiny.py`,
   `python -B scripts/ret_analysis.py dryrun --record`, `python -B scripts/ret_run.py --plan --record`
2. `python -B scripts/ret_gen.py freeze --approval "..."`
3. `python -B scripts/ret_gen.py mint --purpose dev` (CPU, about 4 min including the harvest).
4. GPU: `python -B scripts/ret_run.py --mode dev --draw docs/retention/draws/dev_20261121.jsonl`
5. `python -B scripts/ret_analysis.py dev-select --run runs/retention/dev_20261121`
6. Per confirmation, the following steps. A new adapter means a new confirmation, which mints a new draw.
   - `python -B scripts/ret_gen.py mint --purpose confirm`
   - GPU: `python -B scripts/ret_run.py --mode confirm --draw docs/retention/draws/confirm_<seed>.jsonl [--extra-adapter NAME=DIR]`
   - `python -B scripts/ret_analysis.py analyze --run runs/retention/confirm_<seed>`

**GPU estimate** (`RET_PLAN.json`). The planning rate is a conservative 1.0 generation/s. The optimistic rate is
PREREG-2's registered 2.7/s; PREREG-2 observed 3.8/s on short answers. Each model load costs 120 s, and the
first-token prefill adds 10%.

| job | turns | GPU-h at 1.0/s | GPU-h at 2.7/s |
|---|---|---|---|
| dev (Supa, 2,025 prompts) | 1 | 0.60 | 0.24 |
| confirm (off + t2s0 + supa, 675 prompts each) | 1 | 0.69 | 0.30 |
| each extra arm (675 prompts) | same turn | +0.21 (+10 s adapter load) | +0.08 |

Disk: about 0.205 GB of log-probs per confirm arm (675 x 151,936 fp16).

## 9. Limits and open points

- The instrument is templated and English-only. It probes short-answer arithmetic, conversion, time, counting,
  percentages, recall, abstention and format compliance, not open-ended generation.
- The strict grammar counts format drift as a loss. The lenient grader is reported beside it, so the cause can be read.
- The dev rule selects the hardest level at which Supa reaches 0.40. That may regress toward 0.40 on a fresh draw. The
  reference for ON-vs-OFF is Titan-Sparse, whose accuracy may be lower, and the can-fail note flags categories where
  the reference is too weak.
- No anchor reproduction of stored outputs is run, unlike PREREG-3's A0. Identity rests on parameter fingerprints,
  adapter pins and the registered runtime.
