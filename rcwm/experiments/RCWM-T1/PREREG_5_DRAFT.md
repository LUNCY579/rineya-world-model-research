# RCWM-T1 PREREG-5 (DRAFT): does the surface-diversified ledger adapter (T2b-ledger) follow the frame's rule on unseen surfaces, and does it plan there? (one-step map on a fresh sealed bank, LB-style planning on the held-out surface, the 5c retention guard)

**Status:** DRAFT revision 1, not frozen. Nothing in this kit has used the GPU, trained an adapter, or generated a model output. The freeze (`scripts/t2bl_confirm_run.py --freeze`) is possible only after the T2b-ledger dev screen has run under its frozen gate rule and returned PROCEED, and after BWM has co-signed this text with a salt commitment.

**Sources and templates.** `PREREG_3.md` (the out-of-range map, its scorer and strata), `PREREG_LB.md` (the joint planning test, its commit-reveal draw, runner and deviation policy), `PREREG_2.md` (the A0 anchor), `docs/retention/DESIGN.md` (the 5c guard), `docs/t2b/ledger/DESIGN.md` (the T2b-ledger kit, its partitions and its dev gate), DECISIONS D-34/D-35/D-45 and BWM D-54 (the shared wording). Where this document is silent, PREREG-LB's and PREREG-3's conventions apply (runtime, decoding, parsing, lease and turn rules, create-only outputs).

**What exists now.** The training set `data/t2bl_train.jsonl` (sha256 8faa6a5d24e545ccf7273a042631fb9d77a6a3a7f4d25545a58799f4fda3f94b) and the dev bank `data/t2bl_dev.jsonl` (sha256 befcd7132ef84a71cc354b9209bd5c549bc31ae1dce968490f3efe9bd36752bd), both built without any model call; the dev runner, the pre-stated dev gate (`docs/t2b/ledger/GATE_RULE.md`, not yet frozen), this draft, the confirmation runner and analysis, all tested on the CPU with tiny random stand-ins. **The map bank and the planning draw do not exist**: they are minted at the freeze from a recorded rule (section 3). The retention draw is minted at run time.

## 1. Question

PREREG-3 found that T2 s0's ledger no-op is the token 'void' (STATUS: TRIGGER; unseen statuses 0.01-0.18) and that it fails out of range on magnitudes, reordered keys and word account names; PREREG-LB found that T2 s0 plans at 0.706 on its training surface but 5/180 on the held-out combined surface. T2b-ledger trains the same recipe (D-15/D-16: 1,500 oracle-labelled ledger one-step rows, LoRA r16, 190 steps, the vendored BWM trainer) on a SURFACE-DIVERSIFIED set whose training and held-out partitions were fixed at build time (`docs/t2b/ledger/DESIGN.md` section 2). The question, in the shared wording agreed with BWM (D-54):

- **(a)** For each surface factor X (status words, key order, account names, account count, rule paraphrase, magnitude, and all of them combined), does T2b-ledger **generalise to unseen X** (on held-out items never in training: the absolute bar, non-inferiority to its own canonical accuracy, and a significant gain over the unaugmented T2 s0), or is the **trigger widened** (it learned only the trained surface variants and stays near T2 s0 on held-out ones), or is it **partial**?
- **(b)** Used as the world model inside BWM's beam planner, does T2b-ledger **plan on the unseen surface** (PREREG-LB's held-out combined rendering), where T2 s0 solved 5/180?
- **(c)** Does T2b-ledger seed 0 retain Titan-Sparse's general competence on the 5c guard?

## 2. Registered instruments (pinned by SHA-256 at freeze)

- **Adapters.** T2b-ledger seeds 0, 1, 2 (`runs/t2bl/dev/s<k>/train/adapter`), trained by the dev screen (`scripts/t2bl_run.py`) on `data/t2bl_train.jsonl`; T2b-ledger seed 0 is the registered artifact for parts (b) and (c), seeds 1 and 2 replicate in part (a); no seed is chosen by its outcome. T2 s0 (`runs/t2/t2_s0/adapter`, PREREG-2's pins `92bc7c74...` / `50d9a776...`). All unmerged on frozen FP32 Titan-Sparse (fingerprint `818be9c6b6a3`); never dense-merged (D-28).
- **One model per turn.** `PeftModel.from_pretrained(Titan-Sparse, T2 s0, adapter_name='t2')` plus `load_adapter` for 't2bl0', 't2bl1', 't2bl2'; OFF = `disable_adapter()`; every side is read back from PEFT's own LoRA layer state before any generation, and every generation call asserts that no LoRA layer is merged.
- **Decoding.** Greedy, 160 new tokens, EOS [151645, 151643], thinking off, FP32, TF32 off, `rcwmt.generation.generate_texts` batching; runtime torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0, checked after each load.
- **Frame and scoring (part a).** Prompts `t2bl_common.messages(before, event, surface)`: a system text (paraphrase 0 = Titan's verbatim rule text, 1-3 training paraphrases, 4 held out) and a compact JSON user turn whose objects are serialised in the row's recorded key orders; on the canonical surface the prompt is byte-identical to `env.one_step_messages`. Scorer `p3_common.score_onestep` (PREREG-3's: exact balances, the row's account set, any key order).
- **Planner (part b).** BWM's `plan_beam` (`native_v1/bwmn/blocks_planner.py`, sha256 `18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960`), imported read-only at its pinned sha (no bytecode written into the BWM bundle); the lbeam kit (`lbeam_f.py`, `lbeam_problems.py`, `lbeam_proposer.py`, `lbeam_run.py`, `lbeam_analysis.py`, frozen by PREREG-LB) imported read-only; F = `lbeam_f.LedgerF` (canonical frame) or `lbeam_f.SurfaceAuditF(..., 'heldout_combined')` (keys reversed at every level, accounts rendered bank / card / cash / fees, 'void' rendered 'cancelled'), every F behind `lbeam_f.guarded` (the registered range guard), `EnumerateProposer`, width 2, 12 expansions, `beam_kwargs`, eval_seed 0. Success: `lbeam_problems.execute` of the one submitted plan.
- **Retention (part c).** The frozen 5c suite (`docs/retention/RET_FREEZE.json`, `DEV_SELECTION.json`, `scripts/ret_*.py`).
- **Kit.** `scripts/t2bl_common.py`, `t2bl_make_data.py`, `t2bl_audit.py`, `t2bl_run.py`, `t2bl_gate.py`, `t2bl_confirm_bank.py`, `t2bl_confirm_run.py`, `t2bl_confirm_analysis.py`, `t2bl_test.py`, `p3_common.py`, `confirm2_gen.py`, `rcwmt/*.py`, the training set, the dev bank, the dev screen's records (GATE_RULE.json/.md, DONE.json, provenance.json, GATE.json, the three seeds' dev answers) and this draft: all hashed at the freeze.

**Training partitions (what "unseen" means).** Fixed at build time in `t2bl_common.py` and recorded in `data/t2bl_train.SOURCE.json`:

| factor | in training | held out |
|---|---|---|
| status words | 40 non-posted words in five kinds (12 not-applied, 10 neutral, 8 nonce, 6 case / spelling variants of void, 6 posted look-alikes), plus posted and void | 20 words: rescinded refused stalled nullified / verified valid signed routine / toma zib fep wux / voidd re_void voidy xvoid / postedq yposted posted_zero posted9; PREREG-3's 16 strings are never trained (secondary stratum) |
| key order | every permutation except the held-out ones (the 2-key top object never varied) | top reversed; event and before: the reversal + 2 fixed random permutations each (seed 20261163) |
| account names | A-D, 14 letters I-W (not O), 16 words | E-H and 16 words (bank card cash fees fund loan rent wage barn dock lamp mill nest pier silo yard) |
| account count | 4-7 | 8 |
| rule paraphrase | 0-3 | 4 |
| magnitudes | balances up to 4 digits; add / transfer amounts up to 3 digits; scale factors T2's marginal | 5-digit balances; 4-digit add / transfer amounts |

## 3. Instruments minted at the freeze, and the retention draw

- **Map bank (part a).** `t2bl_make_data.build_bank(seed_map, t2bl_common.CONFIRM_STRATA, union, 'p5', 'p5-map')`: 100 fresh bases (accounts A-D, |b| <= 98, a trained amount, an effective posted event, kinds 45/33/22) rendered in 24 strata, each the same base with ONE factor changed or one combined surface (2,367 rows; H_amt on the 67 add / transfer bases): the dev bank's 16 strata without its pooled P3_words, plus PREREG-3's eight unseen-status strata (U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_voidcase, N_postedlike; their strings by base index exactly as PREREG-3 assigned them). The stratum audit (`t2bl_audit.audit_bank`: every row is its declared change of its base, classified from the rendered text; golds recomputed independently) must pass.
- **Planning draw (part b).** `lbeam_problems.make_set` (generator v2, primary spec): 36 instances per optimal depth 2-6, N = 180, V = 8, K = 8, closure depth 8, |b| <= 98, statuses posted / void, trained amounts, no out-of-range instances, no enrichment; excluded against the union AND every map-bank payload.
- **Seed rule (recorded).** `seed = int(sha256('<tag>|<sha256 of this draft>|<freshness-union digest>|<BWM salt>')[:12], 16)` with tags 'PREREG-5 map bank v1' and 'PREREG-5 planning draw v1' (`t2bl_confirm_bank.seed_from_rule`). **Commit-reveal:** BWM chooses a salt of at least 16 characters, puts its sha256 into its co-sign as `commit:<64 hex>`, and reveals the salt only for the freeze; the freeze refuses a salt that does not match. Nobody can compute either set before the freeze. Spent seeds (20261161-20261164, 20261141-20261143) are refused.
- **Freshness union, rebuilt at the freeze** (`t2bl_make_data.build_union(skip_prefix=None)`): PREREG-LB's name-skipping historical harvest (`HSKILL_CONFIRM_V2*` never opened or hashed; `rcwmt.bank.load_exclusions()` never called), every pinned source, PREREG-3's bank, every payload and rolled-out multi-event transition of every `data/**/*.json[l]` file INCLUDING T2b-ledger's training set and dev bank, the binding transitions of PREREG-LB's spent dev sets and its spent confirmation draw, and every {before, event} payload under docs/ (except docs/t2b/ledger/). data/ is hashed before and after the union build; any change, or any source that cannot be read, refuses the freeze.
- **Seal.** Both sets regenerate byte for byte from (seed, recorded rejections), have 0 payload and 0 operative overlaps with the union, and are written create-only as `data/t2bl_p5_map.jsonl` and `data/t2bl_p5_plan.jsonl` with `.SOURCE.json` records (spec, seed rule and inputs, rejections, overlap counts; no content), sealed by sha256 in `PREREG_5_FREEZE.json` and section 10. **Never inspected before the run:** the freeze prints only seals and counts; until the registered analysis the sets are read only by `t2bl_confirm_run.py` and `t2bl_confirm_analysis.py`.
- **Retention draw (part c).** Minted ONCE at run time, after the GPU job is terminal, by `t2bl_confirm_run.py --ret-mint` (`ret_gen.mint('confirm')`: a fresh OS-entropy seed, the registered dev selection, the 5c freshness gate); recorded in `runs/prereg5/RET_DRAW.json` and never re-minted (an interrupted mint is recovered from its intent record: exactly one new confirmation draw is adopted, anything else stops).

## 4. Arms and stages

| stage | what | GPU |
|---|---|---|
| A0 | (a) PREREG-2's anchor: 208 pilot TEST ledger chain_one chains, OFF and T2 s0 against `runs/t2/t2_s0/gen_chain_one_base.jsonl` / `gen_chain_one.jsonl`: identical states on >= 99% per side; (b) adapter identity: per T2b-ledger seed, the first 50 ID_posted and the first 50 HS_words dev rows regenerated ON vs the dev screen's recorded answers: identical parsed states on >= 97% per seed; (c) batch invariance of the planning F: 64 keys of the spent dev set dev2_primary (never a confirmation key) for T2BL@canon and T2BL@HS, one predict list vs one key per call: identical parsed states on >= 99% per arm | yes |
| M1 | part (a): the map bank, sides OFF, T2 s0, T2b-ledger s0, s1, s2 (one process; written together) | yes |
| P1 | part (b) PRIMARY arm: plan_beam on the planning draw, T2BL@HS (adapter 't2bl0', held-out combined rendering) | yes |
| P2 | part (b) PRIMARY comparator: T2s0@HS (adapter 't2', held-out combined rendering) | yes |
| P3 | part (b): T2BL@canon (adapter 't2bl0', canonical frame) | yes |
| P4 | part (b): T2s0@canon (adapter 't2', canonical frame) | yes |
| C1 | part (b): ORACLE (exact env.step) and IGNORE_STATUS (every status applied as posted) at the same configuration | **no**: CPU, never under the lease, only at the start of an invocation after the one that recorded A0, in a process that has not initialised CUDA (recorded) |
| retention | part (c): `ret_run.py --mode confirm --draw <the recorded draw> --extra-adapter t2bl_s0=runs/t2bl/dev/s0/train/adapter` (sides OFF, T2 s0, Supa and x_t2bl_s0), then `ret_analysis.py analyze` once | yes, its own lease `rcwm-t1:retention`, after the PREREG-5 job |

A0 runs before any output and again at the start of every later turn that loads the model, before any stage. Each planning arm has its own F cache (one stage per arm), so F is the first computed answer of a recorded stage.

## 5. Decision rules (registered verbatim; applied mechanically by `scripts/t2bl_confirm_analysis.py`)

### 5.1 Part (a), the one-step map

> PREREG-5 part (a), the one-step map. Bank: the sealed map bank minted at the freeze (t2bl_common.CONFIRM_STRATA: 24 strata on 100 fresh bases; H_amt on the 67 add / transfer bases). Sides, all generated in one process: OFF (Titan-Sparse, adapters disabled), T2 s0 (PREREG-2's adapter) and T2b-ledger seeds 0, 1, 2, unmerged. Scoring: p3_common.score_onestep (exact balances, any key order). For every primary cell c (words: H_words; key order: H_order; names: H_names; count: H_count; paraphrase: H_para; magnitude: H_bal on the add / transfer bases and H_amt; combined held-out surface: HS_words and HS_posted) and every T2b-ledger seed s: BAR(c, s) iff accuracy >= 0.80 and Wilson 95% lower bound >= 0.70; GAIN(c, s) iff the exact two-sided McNemar test of seed s against T2 s0 on c's rows is significant at 0.05 after Holm adjustment over the 27 (cell, seed) tests, with seed s ahead (b > c); NI(c) iff every seed's difference between its accuracy on c and its accuracy on the same bases' canonical twins (ID_posted for apply rows, ID_void for no-op rows) is >= -0.10 and the paired base-bootstrap one-sided 95% lower bound of that difference averaged over the three seeds (bases resampled with all their rows and seeds; 10,000 resamples, seed 17) is >= -0.10. Per factor X: GENERALISES TO UNSEEN X iff BAR and GAIN hold for every cell of X in all three seeds and NI holds for every cell of X; TRIGGER WIDENED iff the factor's trained-surface control (words: S_words; every other factor: S_surface_posted, plus S_surface_words for the paraphrase and the combined surface) has accuracy >= 0.80 in all three seeds while, on every cell of X and in every seed, the seed's accuracy is at most 0.10 above T2 s0's and at least one cell of X misses BAR in every seed; PARTIAL otherwise, reported with the per-cell, per-variant, per-seed table and its named patterns (below_bar, not_noninferior, no_gain, no_headroom where T2 s0 itself meets BAR on a cell without GAIN, seed_disagreement). In-distribution control: ID_NI iff the paired base-bootstrap one-sided 95% lower bound of T2b-ledger's accuracy minus T2 s0's on ID_posted and ID_void (200 rows), averaged over the three seeds, is >= -0.05 (each seed's difference is reported); if ID_NI fails, every part (a) headline carries 'at an in-distribution cost relative to T2 s0'.

Factor labels used in the wording: words: 'unseen status words'; order: 'unseen key orders'; names: 'unseen account names'; count: 'an unseen account count (8 accounts)'; para: 'an unseen rule paraphrase'; magnitude: 'unseen magnitudes (5-digit balances and 4-digit amounts, add / transfer)'; combined: 'the combined unseen surface'.

### 5.2 Part (a), secondary: PREREG-3's 16 strings

> PREREG-5 part (a), secondary (comparable with PREREG-3's V1; never changes a factor verdict): PREREG-3's V1 rule applied to each T2b-ledger seed on this bank's eight PREREG-3 strata (U_pending, U_draft, U_cancelled, U_reversed, U_held, U_neutral, N_voidcase, N_postedlike: PREREG-3's 16 non-posted strings, none of them in T2b-ledger's training), with ID_posted as the posted twins and ID_void as C_void: RULE iff in every seed every one of the eight strata meets BAR and the twin guard holds (ID_posted accuracy >= 0.80 and copied <= 0.10); TRIGGER iff in every seed the seven strata without a form of 'void' are each <= 0.30, ID_void >= 0.80 and more than half of the errors on those seven strata apply the event as if posted; PARTIAL otherwise. The same rule is applied to T2 s0 in this run as the comparator. Gains over T2 s0 per stratum and seed: exact two-sided McNemar, Holm over the 24 at 0.05.

### 5.3 Part (b), planning on the unseen surface

> PREREG-5 part (b), planning. Instances: the sealed planning draw minted at the freeze (lbeam_problems generator v2, 36 per optimal depth 2-6, N = 180, in range and closed under the vocabulary). Planner: BWM's plan_beam (native_v1/bwmn/blocks_planner.py, sha256 18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960) with lbeam_proposer.EnumerateProposer at width 2 and 12 expansions (lbeam_proposer.beam_kwargs), eval_seed 0, every F wrapped in lbeam_f.guarded (the registered range guard). Arms (one GPU stage each; every turn loads one PeftModel holding both adapters unmerged and re-runs A0 before any stage): T2s0@HS = lbeam_f.SurfaceAuditF('T2', 'heldout_combined') with adapter 't2' (PREREG-LB's held-out combined rendering: keys reversed, accounts bank / card / cash / fees, 'void' rendered 'cancelled'); T2BL@HS = the same with T2b-ledger seed 0's adapter; T2s0@canon and T2BL@canon = lbeam_f.LedgerF('T2') with each adapter; ORACLE = exact env.step (CPU only). Success: the one submitted plan reaches the goal exactly under env.step. PRIMARY: T2BL@HS vs T2s0@HS paired over all 180 instances, exact two-sided McNemar on the discordant pairs (b = T2BL only, c = T2s0 only), delta = (b - c) / 180: PASS iff p < 0.01 and delta >= +0.20; SHORT iff p < 0.01, b > c and delta < +0.20; REVERSED iff p < 0.01 and c > b; NS otherwise. Reading: GENERALISES (T2b-ledger plans on the unseen surface) iff PASS, the absolute bar success(T2BL@HS) / success(ORACLE) >= 0.70 holds, and T2BL@HS is non-inferior to T2BL@canon (paired instance-bootstrap one-sided 95% lower bound of success(T2BL@HS) - success(T2BL@canon) >= -0.10; 10,000 resamples, seed 17); TRIGGER WIDENED iff success(T2BL@canon) - success(T2s0@canon) >= -0.05 and the primary is NS with delta < +0.10; PARTIAL otherwise, with its named reason (short, below_bar, not_noninferior, reversed, canonical_cost, ns_with_gain). Report-only: NEAR-ORACLE on the unseen surface iff the T2BL@HS / ORACLE ratio is >= 0.80 with paired bootstrap 95% lower bound >= 0.70; T2BL@canon vs T2s0@canon (exact McNemar, the difference with its interval); success by optimal depth and on the needs-lookahead stratum, verified-plan precision, refused keys and void-twin exposure for every arm; IGNORE_STATUS (CPU).

### 5.4 Part (c), the retention guard

> PREREG-5 part (c), the retention guard: the frozen 5c retention suite (docs/retention/RET_FREEZE.json) on a confirmation draw minted at run time (ret_gen.py mint --purpose confirm: a fresh OS-entropy seed, the registered dev selection), run by ret_run.py --mode confirm with T2b-ledger seed 0 as the extra Titan-Sparse adapter 'x_t2bl_s0' and scored once by ret_analysis.py analyze under its registered RULE_CONFIRM. The guard is the comparison 'x_t2bl_s0_vs_off': GO (k/m binding) iff RETAINED; NO-GO iff LOSS DETECTED or MARGIN EXCEEDED. It is reported with every headline and never changes a part (a) or (b) verdict; a NO-GO adds 'T2b-ledger s0 loses retention against Titan-Sparse on the 5c guard; it is not a packaging candidate without retention repair'. If the retention run is not completed, the guard is 'not evaluable' and every headline says so.

### 5.5 How the readings map to the shared wording (D-54)

- "generalises to unseen X" is read exactly when GENERALISES holds for X: on held-out surfaces never in training, the absolute bar (0.80, Wilson lower bound 0.70), non-inferiority to its own canonical accuracy (margin 0.10), and a Holm-significant gain over T2 s0, in all three seeds. Where T2 s0 itself already meets the bar on a cell (no headroom; PREREG-3 predicts this for the 8-account count), a gain cannot be shown and the factor is PARTIAL with the 'no shown gain where T2 s0 already meets the bar' clause: the joint definition requires a significant gain over the unaugmented arm.
- "trigger widened" is read exactly when TRIGGER WIDENED holds: the trained surface variants pass and the held-out ones stay within +0.10 of T2 s0 with a cell below the bar in every seed.
- "partial" otherwise, always with the per-cell, per-variant, per-seed table.
- Part (b) uses the same three words for planning (GENERALISES = plans on the unseen surface).

**Why these thresholds.** 0.80 with a Wilson lower bound of 0.70 is the programme's bar (PREREG-3). The own-canonical margin 0.10 and the in-distribution margin 0.05 (against T2 s0) follow the task's dev gate; both are tested on the seed mean with a paired base bootstrap (each seed's own difference must also clear the cell margin), which is what makes them testable at 100 bases (section 5b). Holm at 0.05 over the 27 (cell, seed) gain tests controls the family of 'significantly above T2 s0' claims. Planning keeps PREREG-LB's +0.20 at p < 0.01 for its primary; the absolute bar is 0.70 of ORACLE's success (T2 s0 reached 0.794 of ORACLE on its own surface in PREREG-LB), and the own-canonical margin 0.10 mirrors part (a).

## 5b. Power (computed by `t2bl_confirm_analysis.py --power`; recomputed from the dev screen's outcomes before the co-sign)

Method: simulation through the registered rules; per cell, P(BAR and GAIN in all three seeds and NI), with the cell-versus-twin correlation bracketed by two regimes (nested: the cell's successes inside its twins'; independent), Holm bounded by alpha / 27 and alpha; P(ID_NI) under the same two regimes; planning through the registered `decide_plan` with nested success sets (T2BL@HS inside T2BL@canon inside ORACLE), T2 s0 at PREREG-LB's 5/180 (held-out) and 0.706 (canonical), ORACLE 0.889, plus the exact P(primary PASS) under independence (`lbeam_analysis.exact_pass_prob`). Scenario figures (`docs/t2b/ledger/P5_POWER.json`):

| factor | cell | n | true accuracy (seeds) | twin | T2 s0 | P(GENERALISES) nested / independent (Holm worst case) |
|---|---|---:|---:|---:|---:|---:|
| words | H_words | 100 | 0.85 | 0.95 | 0.05 | 0.06 / 0.06 |
| words | H_words | 100 | 0.90 | 0.95 | 0.05 | 0.95 / 0.66 |
| words | H_words | 100 | 0.95 | 0.95 | 0.05 | 1.00 / 1.00 |
| order | H_order | 100 | 0.85 | 0.95 | 0.40 | 0.06 / 0.06 |
| order | H_order | 100 | 0.90 | 0.95 | 0.40 | 0.95 / 0.66 |
| order | H_order | 100 | 0.95 | 0.95 | 0.40 | 1.00 / 1.00 |
| names | H_names | 100 | 0.85 | 0.95 | 0.68 | 0.01 / 0.02 |
| names | H_names | 100 | 0.90 | 0.95 | 0.68 | 0.52 / 0.41 |
| names | H_names | 100 | 0.95 | 0.95 | 0.68 | 0.93 / 0.95 |
| count | H_count | 100 | 0.85 | 0.95 | 0.87 | 0.00 / 0.00 |
| count | H_count | 100 | 0.90 | 0.95 | 0.87 | 0.00 / 0.00 |
| count | H_count | 100 | 0.95 | 0.95 | 0.87 | 0.01 / 0.01 |
| para | H_para | 100 | 0.85 | 0.95 | 0.60 | 0.05 / 0.05 |
| para | H_para | 100 | 0.90 | 0.95 | 0.60 | 0.90 / 0.64 |
| para | H_para | 100 | 0.95 | 0.95 | 0.60 | 1.00 / 1.00 |
| magnitude | H_bal | 67 | 0.85 | 0.95 | 0.50 | 0.05 / 0.04 |
| magnitude | H_bal | 67 | 0.90 | 0.95 | 0.50 | 0.78 / 0.49 |
| magnitude | H_bal | 67 | 0.95 | 0.95 | 0.50 | 1.00 / 0.98 |
| magnitude | H_amt | 67 | 0.85 | 0.95 | 0.15 | 0.05 / 0.04 |
| magnitude | H_amt | 67 | 0.90 | 0.95 | 0.15 | 0.81 / 0.52 |
| magnitude | H_amt | 67 | 0.95 | 0.95 | 0.15 | 1.00 / 0.98 |
| combined | HS_words | 100 | 0.85 | 0.95 | 0.02 | 0.06 / 0.06 |
| combined | HS_words | 100 | 0.90 | 0.95 | 0.02 | 0.95 / 0.66 |
| combined | HS_words | 100 | 0.95 | 0.95 | 0.02 | 1.00 / 1.00 |
| combined | HS_posted | 100 | 0.85 | 0.95 | 0.20 | 0.06 / 0.06 |
| combined | HS_posted | 100 | 0.90 | 0.95 | 0.20 | 0.95 / 0.66 |
| combined | HS_posted | 100 | 0.95 | 0.95 | 0.20 | 1.00 / 1.00 |

| in-distribution control (each seed vs T2 s0) | P(ID_NI) nested / independent |
|---|---:|
| seed_acc_0.90_vs_t2s0_0.93 | 0.84 / 0.23 |
| seed_acc_0.93_vs_t2s0_0.93 | 1.00 / 0.83 |
| seed_acc_0.96_vs_t2s0_0.93 | 1.00 / 1.00 |

| planning: T2BL@HS / T2BL@canon success | P(GENERALISES) | P(primary PASS), simulated / exact under independence |
|---|---:|---:|
| 0.30 / 0.70 | 0.00 | 0.98 / 0.982 |
| 0.50 / 0.70 | 0.00 | 1.00 / 1.000 |
| 0.60 / 0.70 | 0.04 | 1.00 / 1.000 |
| 0.65 / 0.70 | 0.77 | 1.00 / 1.000 |
| 0.70 / 0.75 | 0.90 | 1.00 / 1.000 |

Assumption source: scenarios. T2 s0's held-out-surface planning success is taken as 5/180 (PREREG-LB's registered D-35 audit, REPORT 3d) and its canonical success as 0.706 (PREREG-LB); ORACLE 0.889 (PREREG-LB); N = 180 is PREREG-LB's and was not chosen from it.

Reading (scenario figures; refreshed from the dev screen before the co-sign): where T2 s0 is far below the bar (status words, key order, paraphrase, the combined surface), a true held-out accuracy of 0.90 with twins at 0.95 reads GENERALISES with probability 0.64-0.95 per cell and a true 0.95 about 1.0; for account names (T2 s0 about 0.68, so the Holm gain is harder) 0.41-0.52 at 0.90 and 0.93-0.95 at 0.95; for the 67-row magnitude cells 0.49-0.81 at 0.90 and 0.98-1.0 at 0.95; a true 0.85 sits at the own-canonical margin and reads GENERALISES at most about 0.06 of the time (PARTIAL not_noninferior otherwise). The 8-account cell has no headroom over T2 s0 (PREREG-3: 0.84-0.90), so 'generalises to an unseen account count' is not reachable under the joint definition; it is registered and worded as such. The in-distribution control is non-inferior with probability 0.83-1.0 when T2b-ledger equals T2 s0 in distribution (0.23-0.84 if it is 0.03 lower). The planning primary passes with probability >= 0.98 whenever T2b-ledger's held-out planning success is >= 0.30; the full GENERALISES reading needs held-out success near 0.65 or above (0.70 of ORACLE and within 0.10 of its own canonical success): 0.77 at 0.65 / 0.70, 0.90 at 0.70 / 0.75, 0.04 at 0.60 / 0.70. **Before the co-sign** the operator reruns `t2bl_confirm_analysis.py --power --from-gate runs/t2bl/dev/analysis/GATE.json --record` (`P5_POWER_DEV.json`, required by the freeze) and replaces this section's numbers with the dev-based ones; the decision rules do not change.

## 6. Report-only analyses (no decision depends on them)

- Part (a): every stratum per side (OFF, T2 s0, T2b-ledger s0-s2) with Wilson intervals; variants (word kind, name kind, held-out event order, paraphrase status); H_bal on scale bases (multi-digit multiplication); LB_void / LB_posted (PREREG-LB's rendering); S_words, S_surface_posted, S_surface_words (trained surfaces); error classes (applied as posted, other change, copied, wrong value, unparsed).
- Part (b): per arm (T2BL@HS, T2s0@HS, T2BL@canon, T2s0@canon, ORACLE, IGNORE_STATUS) success by optimal depth 2-6, on the needs-lookahead stratum (`lbeam_problems.greedy_solvable` false) and on greedy-solvable instances, verified rate, verified-plan precision, optimal-length rate, void-twin exposure (PREREG-LB's definition, read from each arm's own F cache), the range guard's refused keys; T2BL@canon vs T2s0@canon (exact McNemar, the difference with its paired bootstrap interval).
- Part (c): everything `ret_analysis.py` reports (both graders, per category, Wilson intervals, KL calibration).
- Instrument record: the terminal record, A0 (anchor, adapter identity, batch invariance) and every re-check, deferrals, attempts, orphaned stages, BWM modules loaded per turn.

## 7. Procedure and integrity

0. **Path to the freeze (operator steps, each logged in DECISIONS.md; none is taken by this kit build):** (i) freeze the dev gate rule (`t2bl_gate.py --freeze-rule --approval "..."`; it also pins PREREG-5's rules digest); (ii) install `docs/t2b/ledger/queue_hook_t2bl.sh.draft` as `scripts/queue_hooks/88_t2bl.sh` in a fresh chain (joint turn order, D-45); (iii) the dev screen runs and the registered gate writes `runs/t2bl/dev/analysis/GATE.json`; only PROCEED continues (otherwise the pre-stated revision path applies and this draft is not frozen); (iv) `t2bl_confirm_analysis.py --power --from-gate runs/t2bl/dev/analysis/GATE.json --record` and `--render-draft` (section 5b refreshed; the rules are unchanged); (v) `t2bl_test.py --part all` (records bound to the final draft and hook draft); (vi) BWM's co-sign with its salt commitment; (vii) `--freeze`; (viii) install the PREREG-5 hook in a fresh chain.
1. **Pre-freeze records** (rewritten only before the freeze): `docs/t2b/ledger/P5_DRYRUN.{json,md}` (every wording branch on synthetic outcomes through the real `decide_map` / `decide_plan` / `wording`), `P5_POWER.json` and `P5_POWER_DEV.json` (section 5b), `P5_GEN_PLAN.json` (`--plan`), `TEST_{P5ANALYSIS,P5TINY,P5FREEZE,PLAN,HOOK}.json` (`scripts/t2bl_test.py`; each records the code it ran with and the sha256 of this draft and of the hook draft), `queue_hook_prereg5.sh.draft`.
2. **Freeze:** `python -B scripts/t2bl_confirm_run.py --freeze --approval "<who, when, DECISIONS D-<n>>" --bwm-cosign "<BWM's reference> <this draft's full sha256> commit:<sha256 of BWM's salt>" --bwm-salt "<salt>"` (CPU). It refuses (never persisted; nothing minted or renamed) if: the approval cites no DECISIONS entry; the co-sign does not quote this draft's sha256 or carries no commitment; the salt is shorter than 16 characters or does not match; PREREG-5 is already frozen or a freeze was interrupted; any PREREG-5 output exists; a frozen input is missing; a pre-freeze check fails (a test record not produced by the current code, draft and hook draft; a registered string of the analysis missing from this draft; no dev-based power record; the dev gate rule not frozen, the dev screen not complete or the dev gate not PROCEED; a missing T2b-ledger adapter; a pin, the BWM planner or the retention freeze differing); a source of the union cannot be read or data/ changes while it is built. Otherwise it mints and seals both sets (section 3), hashes every input, records whether PREREG-5's rules are byte-identical to the digest pinned when the dev gate rule was frozen, and writes `PREREG_5.md` (this draft with the status line and section 10 filled) and `PREREG_5_FREEZE.json`, all as `.partial` first and renamed in order (sets, PREREG_5.md, freeze record), create-only.
3. **Verification:** `t2bl_confirm_run.py --verify` (also run by the job, the retention binding and the analysis): PREREG_5.md's hash; every input against the freeze with the `PREREG_5_AMENDMENT_*.json` chain applied; the pins; BWM's planner on disk; both sealed sets (hash = SOURCE = freeze, byte-for-byte regeneration from seed and recorded rejections, the seed recomputed from the rule, the salt against the commitment in the co-sign, the stratum audit, `verify_instances`); the co-sign quotes the frozen draft's sha256; the dev gate's PROCEED; the registered runtime.
4. **GPU job:** `python -B scripts/t2bl_confirm_run.py`, one lease `rcwm-t1:prereg5` (`rcwmt.runtime.gpu_lease`: alternation, thermal flag, duty cycle, RAM floor, the dedicated-memory holder check), 4.5 h cap from lock acquisition, D-22 stage rule (no stage within 0.5 h of the cap; a stage starts only if 1.25 x its estimate fits; the start-of-turn check adds the model load and the longest A0 measured so far; if the first remaining stage does not fit, the model is not loaded).
   - **Before the lease:** the freeze verifies; an installed `scripts/queue_hooks/90_prereg5.sh` is the frozen hook draft byte for byte; no other RCWM job holds or waits for the lease (R14); the run lock; every adapter, the sealed sets, the anchor references, dev2_primary, the dev bank and the dev answers against the freeze; a failed A0 record or re-check refuses (persisted).
   - **After each load, before generating:** fingerprint, no merged LoRA layer, adapter hashes, runtime versions.
   - **GPU plan per turn** (`P5_GEN_PLAN.json`; planning F 2.7 / observed 3.8 generations per second, map 2.7 / 4.88 scaled by answer length, batch-1 calls 0.3/s, 120 s per load):

| stage | what | generations expected (ceiling) | h planning rate | h observed rate |
|---|---|---:|---:|---:|
| A0 | PREREG-2 anchor (208 chains x 4 x OFF/T2s0) + adapter identity (3 x 100 dev rows) + batch invariance (64 keys x T2BL canonical / held-out x batched and batch 1) | 2,220 (2,220) | 0.34 | 0.27 |
| M1 | part (a): the map bank (2367 rows) x OFF, T2s0, T2BL s0-s2 | 11,835 (11,835) | 1.46 | 0.80 |
| P1 | part (b) PRIMARY arm: plan_beam enum W2 E12, T2BL@HS (held-out combined rendering) | 12,584 (18,720) | 1.29 | 0.92 |
| P2 | part (b) PRIMARY comparator: plan_beam enum W2 E12, T2s0@HS (held-out combined rendering) | 14,537 (18,720) | 1.50 | 1.06 |
| P3 | part (b): plan_beam enum W2 E12, T2BL@canon (canonical frame) | 12,584 (18,720) | 1.29 | 0.92 |
| P4 | part (b): plan_beam enum W2 E12, T2s0@canon (canonical frame) | 12,584 (18,720) | 1.29 | 0.92 |

   - expected, observed rate: turn 1: A0, M1, P1, P2, P3 (4.01 h); turn 2: A0 re-check, P4 (1.22 h); 5.23 GPU-h
   - expected, planning rate: turn 1: A0, M1, P1 (3.12 h); turn 2: A0 re-check, P2, P3 (3.16 h); turn 3: A0 re-check, P4 (1.67 h); 7.95 GPU-h
   - ceiling, observed rate: turn 1: A0, M1, P1, P2 (3.84 h); turn 2: A0 re-check, P3, P4 (3.04 h); 6.88 GPU-h
   - ceiling, planning rate: turn 1: A0, M1, P1 (3.75 h); turn 2: A0 re-check, P2 (2.3 h); turn 3: A0 re-check, P3 (2.3 h); turn 4: A0 re-check, P4 (2.3 h); 10.65 GPU-h
   - part (c): one more RCWM turn under its own lease (about 0.5-1.1 GPU-h).

   - **Completion and exit codes:** as PREREG-LB: 0 complete (`DONE.json` lists every output hash); 3 deterministic stop, never retried (persisted as `STOPPED.json` with its `eligible_parts` when it concerns the run); 5 deferred at a stage boundary or C1 pending; 1 crash (the hook retries once).
5. **Part (c):** after the GPU job ends complete or verdict-eligible: `--ret-mint` (once), the retention run (its own lease; deferral: sleep 60 s, then call again), `ret_analysis.py analyze` once, then `--ret-record` binds the result (or the retention run's persisted stop) to PREREG-5 in `runs/prereg5/RETENTION.json` after checking that the run used the recorded draw and that side x_t2bl_s0 is T2b-ledger s0's frozen adapter.
6. **Queue.** `docs/t2b/ledger/queue_hook_prereg5.sh.draft` is installed byte for byte as `scripts/queue_hooks/90_prereg5.sh` only after the freeze, on the user's go-ahead, logged in DECISIONS.md, and runs only in a fresh chain. It adds no second GPU waiter; never retries after exit 3; on exit 5 it **sleeps 60 s before the re-call** (agreed with BWM, so BWM's yield loop sees the release); retries a crash once; runs the analysis once, after the GPU job is complete or verdict-eligible and part (c) is bound.
7. **No peeking.** Between the freeze and the registered analysis nobody opens the sealed sets or reads `runs/prereg5/{map,plan,cpu}/*`; only `provenance.json`, `anchor/*`, `STOPPED.json` / `DONE.json` and the job log may be read, as instrument checks. No analysis code changes after the first output except a logged non-statistical crash fix (an amendment).
8. **Analysis:** `python -B scripts/t2bl_confirm_analysis.py`, once, CPU, numpy only. It verifies the freeze; reads the terminal record (DONE.json, or a STOPPED.json naming its verdict-eligible parts; never both) and checks provenance against it; re-checks the instrument (every part's stages recorded and unchanged; A0's anchor, adapter identity and batch invariance passed; every turn that ran a stage after A0's turn passed its A0 re-check; runtime, fingerprint, adapter hashes at load, BWM planner and code hashes per turn against the freeze; C1 CUDA-free and outside the lease); scores; writes `runs/prereg5/analysis/PREREG_5_RESULTS.{json,md}` create-only.

**Deviation and stop policy.**
- A0 failure in the first A0 (anchor, adapter identity or batch invariance): stop before any output (persisted); no verdict; investigate and log; a software or environment cause is restored and the stop records moved aside with a DECISIONS entry; any other remedy is an amendment before any output exists.
- A0 re-check failure on a later turn: stop before any stage of that turn; parts whose stages were all recorded under a passing A0 stay verdict-eligible (STOPPED.json `eligible_parts`); the other part is rendered 'not evaluable'.
- Any other deterministic stop (runtime drift, fingerprint, adapter hash, BWM planner sha, a changed recorded output, a failed freeze verification, two consecutive turns without a recorded stage, a stage failing three times): persisted; per-part eligibility as above (a changed output makes nothing eligible).
- GPU crash or reset: a stage that is not done is moved to `runs/prereg5/_orphans/` (never deleted) and regenerated in full; the resumed turn re-runs A0 first.
- Retention: a retention run that stops (persisted) is bound as stopped and part (c) reads 'not evaluable'; parts (a) and (b) are unaffected.
- Analysis crash before the results exist: an identical rerun is allowed; a crash fix only as `PREREG_5_AMENDMENT_*.json` (file, old and new sha256, reason, `approved_by`), logged.

## 8. Wording (rendered mechanically by `t2bl_confirm_analysis.wording()`; every branch exercised by the dry run)

These are the analysis code's own templates, verbatim (placeholders in braces are filled from the results; `{qual}` is empty or the in-distribution qualifier; ranges are rendered '0.88 to 0.95'). The freeze refuses if any of them is missing from this file.

- `factor_generalises`: T2b-ledger GENERALISES TO {LABEL}{qual}: in all three seeds, accuracy on {cells} {accs} (each >= 0.80, Wilson lower bound >= 0.70) and above T2 s0 ({t2}) in every Holm-adjusted McNemar test, within the margin of its own canonical accuracy (difference {nis} per seed, seed-mean lower bound {lbs}, each >= -0.10).
- `factor_trigger`: TRIGGER WIDENED for {label}{qual}: on trained surface variants ({controls}) the adapter reached {ctrl} (each >= 0.80), but on {cells} it stayed near T2 s0 ({accs} vs T2 s0 {t2}; at most +0.10 above it in every seed).
- `factor_partial`: PARTIAL for {label}{qual}: accuracy {accs} (T2 s0 {t2}); {bits}. Per-cell and per-variant table below.
- `bit_below_bar`: below the bar in {list}
- `bit_not_noninferior`: not non-inferior to its own canonical accuracy in {list}
- `bit_no_gain`: no Holm-significant gain over T2 s0 in {list}
- `bit_no_headroom`: no shown gain where T2 s0 already meets the bar ({list})
- `bit_seed_disagreement`: the seeds disagree
- `id_cost_qual`:  (at an in-distribution cost relative to T2 s0)
- `headline_a`: PREREG-5 (a): GENERALISES TO {g}; TRIGGER WIDENED for {t}; PARTIAL for {p}{cost}
- `headline_a_cost`:  (every part (a) result at an in-distribution cost relative to T2 s0)
- `none`: none
- `id_sentence`: In distribution (ID_posted + ID_void), T2b-ledger minus T2 s0: {per_seed}; seed mean {mean} (one-sided 95% lower bound {lo}){tail}
- `id_seed`: {s} {diff}
- `id_tail_ok`: : non-inferior within 0.05.
- `id_tail_cost`: : not shown non-inferior within 0.05.
- `p3_sentence`: Secondary, comparable with PREREG-3 (PREREG-3's 16 strings, never trained): STATUS {status} for T2b-ledger (T2 s0 in this run: {t2status}). Accuracy on the eight PREREG-3 strata {allp} across seeds (the seven without a form of 'void': {trig}); T2 s0 {t2}; Holm-significant gains over T2 s0 in {n_gain} of 24 stratum x seed tests.
- `plan_base`: {hs}/{n} vs T2 s0's {t2hs}/{n} on PREREG-LB's held-out combined rendering (delta {delta}, discordant {b}/{c}, exact McNemar p {p}; status {status})
- `plan_generalises`: T2b-ledger PLANS ON THE UNSEEN SURFACE: {base}; {ratio} of ORACLE's success (>= 0.70) and within the margin of its own canonical success ({can}/{n}; difference {nid}, lower bound {nilb}).
- `plan_trigger`: TRIGGER WIDENED (planning): T2b-ledger planned on the canonical surface ({can}/{n} vs T2 s0 {t2can}/{n}) but stayed near T2 s0 on the held-out surface: {base}.
- `plan_partial`: PARTIAL (planning): {base}; {why}.
- `why_short`: the gain is significant but below +0.20
- `why_reversed`: T2 s0 planned significantly better on the held-out surface
- `why_below_bar`: its held-out success is below 0.70 of ORACLE's ({ratio})
- `why_not_noninferior`: its held-out success is not non-inferior to its own canonical success (lower bound {nilb} < -0.10)
- `why_canonical_cost`: it plans worse than T2 s0 on the canonical surface ({can}/{n} vs {t2can}/{n})
- `why_ns_with_gain`: the held-out difference is not significant but not near T2 s0 either
- `plan_near`:  Report-only: near-oracle on the held-out surface (ratio {ratio}, 95% interval {lo}-{hi}).
- `plan_canon_line`: Report-only: on the canonical surface T2b-ledger {can}/{n} vs T2 s0 {t2can}/{n} (difference {d}, 95% paired bootstrap CI [{lo}, {hi}], exact McNemar p {p}); ORACLE {orc}/{n}; IGNORE_STATUS {ign}/{n}.
- `headline_b_GENERALISES`: PREREG-5 (b): PLANS ON THE UNSEEN SURFACE
- `headline_b_TRIGGER_WIDENED`: PREREG-5 (b): TRIGGER WIDENED
- `headline_b_PARTIAL`: PREREG-5 (b): PARTIAL
- `ret_go`: Retention guard (5c, draw {draw}): {label}: T2b-ledger s0 retains against Titan-Sparse.
- `ret_nogo`: Retention guard (5c, draw {draw}): NO-GO ({verdict}): T2b-ledger s0 loses retention against Titan-Sparse on the 5c guard; it is not a packaging candidate without retention repair.
- `ret_ne`: Retention guard (5c): not evaluable ({reason}); every headline is read without it.
- `headline_c`: PREREG-5 (c): retention guard {label}

**Parts that are not evaluable:**

- PREREG-5 (a): not evaluable ({reason}).
- PREREG-5 (b): not evaluable ({reason}).
- reason `stopped`: the run ended by a deterministic stop before this part's stages were recorded
- reason `missing`: this part's stages are not recorded

**Rules since the dev gate (always rendered; `{when}` = when the dev gate rule was frozen):**

- unchanged: The decision rules of parts (a)-(c), the wording templates and the limits are byte-identical to those recorded when the dev gate rule was frozen ({when}), before any T2b-ledger adapter existed.
- changed: The decision rules, wording templates or limits differ from those recorded when the dev gate rule was frozen ({when}); the changes were made after the dev screen's outputs existed and are listed in PREREG_5.md Appendix A.

**Limits that always accompany the result (rendered verbatim from `LIMITS`):**

- Ledger only: Titan's one-step ledger frame family; T2b-ledger seeds 0-2 on frozen FP32 Titan-Sparse, unmerged; T2 s0 (PREREG-2's artifact) is the unaugmented comparator.
- T2 s0 differs from T2b-ledger in more than surface diversity: it trained on 501 ledger rows inside 1,501 rows of three families; T2b-ledger trains on 1,500 ledger rows with widened magnitudes, twin pairs and 600 rows of non-posted words. A gain over T2 s0 is a gain of the T2b-ledger recipe, not of surface augmentation alone.
- 'Unseen' means outside T2b-ledger's training partitions, fixed at build time: 20 held-out status words (five kinds), letters E-H and 16 held-out word names, 8 accounts, the reversed top-level order and 2 + 1 held-out orders per object, rule paraphrase 4, 5-digit balances and 4-digit add / transfer amounts. A pass is transfer to these items, not to every surface.
- The top-level object (before, event) is never varied in training, so every held-out-order row also reverses it.
- Field names (before, event, account, amount, kind, other, status, tick) and the kind words are never varied; scale factors stay in the recorded marginal everywhere, so magnitude claims cover add / transfer only (scale is report-only).
- Planning: in-range instances only, one planner configuration (enum W2 E12), one adapter seed (s0), a model-free proposer, and PREREG-LB's single held-out combined rendering.
- Retention: the five categories the 5c dev selection kept (R17); losses of 6-12 items in one category are often missed.
- Thresholds were fixed before any T2b-ledger output: bar 0.80 with Wilson 0.70, own-canonical margin 0.10, in-distribution margin 0.05, Holm at 0.05 over 27 tests; planning +0.20 at p < 0.01, ratio 0.70, margin 0.10.
- Fixed rule: every prompt states the same rule (paraphrased); a pass shows that T2b-ledger follows this fixed rule on unseen surfaces, not that it reads a rule supplied in the prompt (RCX's replication, R19: neither OFF nor T2 s0 reads a supplied rule; that needs rule-varied training).
- T2b-ledger's training contains case and spelling variants of 'void' (VOId, vOid, voids, voiding, void_, voidz), so PREREG-3's N_voidcase strings (Void, VOID, voided) are near-trained for it; the secondary PREREG-3 reading counts them as PREREG-3 did, and the seven strata without a form of 'void' carry its TRIGGER clause.
- The dev screen chose nothing that this test reads: its gate only decided whether PREREG-5 runs; the map bank, the planning draw and the retention draw are fresh, sealed, and were minted after the dev gate's PROCEED.

## 9. Roles and co-signing

- RCWM writes and runs everything (GPU and CPU stages) and holds the answer key; BWM's code enters only as the read-only, sha-pinned planner, which BWM keeps unchanged from the freeze until the analysis has run.
- **Before the freeze:** BWM co-signs THIS draft at its stated sha256 (part (b) uses BWM's planner and PREREG-LB's joint design; the shared wording is D-54's), quoting the draft's full sha256 and `commit:<sha256 of BWM's salt>`, and gives the salt to the operator for the freeze only.
- **Amendments** touching part (b)'s rules, wording, planner call or instances need BWM's co-sign before they are recorded; operational amendments are RCWM's, with notice to BWM.
- **After the analysis:** BWM reviews `PREREG_5_RESULTS.md` and co-signs the planning section; RCWM gives BWM the episodes, caches and per-instance success arrays for an independent re-score. A co-sign never changes a reading.

## 10. Freeze record

Filled by `--freeze`.

## Appendix A. Choices made in this draft, with reasons

- **The dev gate precedes the freeze.** A sealed confirmation is drafted from a kit whose dev screen passed a pre-stated gate (HS_words >= 0.80 in all seeds, in-distribution within 0.05 of T2 s0, a copy guard on the held-out posted twins). The gate decides only whether PREREG-5 runs; the confirmation reads only fresh sealed sets, and its rules were pinned (by digest) when the gate rule was frozen, before any T2b-ledger output.
- **Seed 0 is the registered artifact** for planning and retention (fixed before training); seeds 1 and 2 replicate the map.
- **Non-inferiority on the seed mean.** Per-seed one-sided intervals at 100 bases would fail a true non-inferior adapter most of the time (section 5b); the seed mean with a paired base bootstrap keeps the interval honest while every seed's own difference must also clear the margin.
- **One planning arm per stage.** Each arm's stage fits a turn even at the ceiling of its generations at the planning rate; the primary arm runs first.
- **The held-out rendering of part (b) is PREREG-LB's**, so 5/180 is the comparable number; it is not one of T2b-ledger's own held-out surfaces (its 'cancelled' is one of PREREG-3's reserved strings, its names are held-out words, its key orders are held out).
- **Retention binding by pointer.** The 5c suite is frozen separately and its analysis writes its own create-only result; PREREG-5 binds that result (or the stop) after checking the draw and the adapter, and never re-scores it.

## Appendix B. Open questions (defaults in force unless changed before the co-sign)

- Q1 (BWM): keep T2s0@canon (P4) as part of part (b)? It re-runs PREREG-LB's T2 arm on a new draw (about 1.3 h at the planning rate) and is used only for the trigger-widened reading and the canonical-cost reason. Default: keep.
- Q2 (BWM): the absolute planning bar 0.70 of ORACLE. Default: keep (PREREG-LB's near-oracle bar was 0.80 with a 0.70 lower bound; T2 s0 reached 0.794 on its own surface).
- Q3 (operator): run part (c) in the same chain pass as parts (a)/(b) (one more RCWM turn) or leave it for the following pass. Default: the hook runs it right after the GPU job is terminal.
- Q4 (operator): if the dev gate stops (TRIGGER WIDENED or PARTIAL), the pre-stated revision path applies and this draft is not frozen; a revised kit gets a new dev seed and a new draft revision.

## Appendix C. How this draft was checked (CPU only; no GPU, no lease, no real model)

`scripts/t2bl_test.py` parts data, gate, p5analysis, devtiny, p5tiny, p5freeze, plan and hook (records `docs/t2b/ledger/TEST_*.json`): the data audit and its perturbation refusals; the scorer self-test; every dev-gate branch; every PREREG-5 wording branch and the verbatim presence of every registered string in this file; tiny random Qwen3 + LoRA runs of the dev runner (the real vendored trainer, crash / resume / deferral / stops) and of the PREREG-5 job (BWM's pinned planner, A0 failures, C1 outside the lease, crash / resume / deferral / stops); a fake-root freeze (refusals, mint, verify, tamper, run, retention binding, the registered analysis, create-only); both `--plan` paths without torch; both hook drafts under bash with a stub step (every exit-code path, the 60 s sleep before a deferral re-call, idempotency).
