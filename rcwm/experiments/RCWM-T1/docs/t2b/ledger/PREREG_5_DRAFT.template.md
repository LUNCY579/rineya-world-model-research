# RCWM-T1 PREREG-5 (DRAFT): does the surface-diversified ledger adapter (T2b-ledger) follow the frame's rule on unseen surfaces, and does it plan there? (one-step map on a fresh sealed bank, LB-style planning on the held-out surface, the 5c retention guard)

**Status:** DRAFT revision 1, not frozen. Nothing in this kit has used the GPU, trained an adapter, or generated a model output. The freeze (`scripts/t2bl_confirm_run.py --freeze`) is possible only after the T2b-ledger dev screen has run under its frozen gate rule and returned PROCEED, and after BWM has co-signed this text with a salt commitment.

**Sources and templates.** `PREREG_3.md` (the out-of-range map, its scorer and strata), `PREREG_LB.md` (the joint planning test, its commit-reveal draw, runner and deviation policy), `PREREG_2.md` (the A0 anchor), `docs/retention/DESIGN.md` (the 5c guard), `docs/t2b/ledger/DESIGN.md` (the T2b-ledger kit, its partitions and its dev gate), DECISIONS D-34/D-35/D-45 and BWM D-54 (the shared wording). Where this document is silent, PREREG-LB's and PREREG-3's conventions apply (runtime, decoding, parsing, lease and turn rules, create-only outputs).

**What exists now.** The training set `data/t2bl_train.jsonl` (sha256 @@TRAIN_SHA@@) and the dev bank `data/t2bl_dev.jsonl` (sha256 @@DEV_SHA@@), both built without any model call; the dev runner, the pre-stated dev gate (`docs/t2b/ledger/GATE_RULE.md`, not yet frozen), this draft, the confirmation runner and analysis, all tested on the CPU with tiny random stand-ins. **The map bank and the planning draw do not exist**: they are minted at the freeze from a recorded rule (section 3). The retention draw is minted at run time.

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

> @@RULE_MAP@@

Factor labels used in the wording: @@FACTOR_LABELS@@.

### 5.2 Part (a), secondary: PREREG-3's 16 strings

> @@RULE_P3@@

### 5.3 Part (b), planning on the unseen surface

> @@RULE_PLAN@@

### 5.4 Part (c), the retention guard

> @@RULE_RET@@

### 5.5 How the readings map to the shared wording (D-54)

- "generalises to unseen X" is read exactly when GENERALISES holds for X: on held-out surfaces never in training, the absolute bar (0.80, Wilson lower bound 0.70), non-inferiority to its own canonical accuracy (margin 0.10), and a Holm-significant gain over T2 s0, in all three seeds. Where T2 s0 itself already meets the bar on a cell (no headroom; PREREG-3 predicts this for the 8-account count), a gain cannot be shown and the factor is PARTIAL with the 'no shown gain where T2 s0 already meets the bar' clause: the joint definition requires a significant gain over the unaugmented arm.
- "trigger widened" is read exactly when TRIGGER WIDENED holds: the trained surface variants pass and the held-out ones stay within +0.10 of T2 s0 with a cell below the bar in every seed.
- "partial" otherwise, always with the per-cell, per-variant, per-seed table.
- Part (b) uses the same three words for planning (GENERALISES = plans on the unseen surface).

**Why these thresholds.** 0.80 with a Wilson lower bound of 0.70 is the programme's bar (PREREG-3). The own-canonical margin 0.10 and the in-distribution margin 0.05 (against T2 s0) follow the task's dev gate; both are tested on the seed mean with a paired base bootstrap (each seed's own difference must also clear the cell margin), which is what makes them testable at 100 bases (section 5b). Holm at 0.05 over the 27 (cell, seed) gain tests controls the family of 'significantly above T2 s0' claims. Planning keeps PREREG-LB's +0.20 at p < 0.01 for its primary; the absolute bar is 0.70 of ORACLE's success (T2 s0 reached 0.794 of ORACLE on its own surface in PREREG-LB), and the own-canonical margin 0.10 mirrors part (a).

## 5b. Power (computed by `t2bl_confirm_analysis.py --power`; recomputed from the dev screen's outcomes before the co-sign)

Method: simulation through the registered rules; per cell, P(BAR and GAIN in all three seeds and NI), with the cell-versus-twin correlation bracketed by two regimes (nested: the cell's successes inside its twins'; independent), Holm bounded by alpha / 27 and alpha; P(ID_NI) under the same two regimes; planning through the registered `decide_plan` with nested success sets (T2BL@HS inside T2BL@canon inside ORACLE), T2 s0 at PREREG-LB's 5/180 (held-out) and 0.706 (canonical), ORACLE 0.889, plus the exact P(primary PASS) under independence (`lbeam_analysis.exact_pass_prob`). Scenario figures (`docs/t2b/ledger/P5_POWER.json`):

@@POWER_TABLE@@

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

@@PLAN_TABLE@@

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

@@TEMPLATES@@

**Parts that are not evaluable:**

@@NOT_EVALUABLE@@

**Rules since the dev gate (always rendered; `{when}` = when the dev gate rule was frozen):**

@@RULES_SENTENCES@@

**Limits that always accompany the result (rendered verbatim from `LIMITS`):**

@@LIMITS@@

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
