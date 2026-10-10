# RCWM-T1 PREREG-LB (DRAFT): does T2's near-oracle ledger prediction become planning? (BWM's beam planner with the T2 seed-0 adapter vs the same model with the adapter off as its world model; joint roadmap item 3)

**Status:** DRAFT revision 2, RCWM version of the joint preregistration, for BWM review and co-signature. Not frozen. `scripts/lbeam_run.py --freeze` fills this line and section 10.

**Sources.** This draft includes BWM-001's draft **faithfully and verbatim**: `Rineya_BWM_001_Research_Bundle/native_v1/research/PREREG_LEDGER_BEAM_DRAFT.md`, draft v1, sha256 `b2896ba39301083a71db490c37c68d5826240faa4c10017830537d0b3e44c34f`. Its text appears below as quoted blocks marked **[BWM v1]**, in BWM's order. Everything RCWM adds or changes is marked **[RCWM amendment]** with a reason; Appendix A lists all of them. Where the two texts differ, the amendment states what changes and why. Templates: `PREREG_2.md` and `PREREG_3.md`. Where this document is silent, their conventions apply (runtime, decoding, parsing, lease and turn rules, create-only outputs, deviation policy). Interface: `docs/ledger_beam/INTERFACE.md` revision 3 (DECISIONS D-37).

**Revision 2** applies RCWM's two internal reviews of revision 1 (statistics, engineering). The main changes: every mixed-sign outcome of the two co-primaries now has its own branch and wording, and the dry run checks the rendered text of all 32 (status 1, status 2, NEAR) combinations for truthfulness; the verdict needs only A0, G1 and C1, so report-only stages that fail or are not run are rendered "not evaluable" by a pre-registered rule instead of blocking the verdict; the draw's seed rule includes a salt that BWM commits to in its co-sign and reveals at the freeze, so RCWM cannot compute the draw in advance; the freeze checks BWM's co-sign (this draft's sha256 and the salt commitment) and refuses if data/ changes while the freshness union is built; a turn that cannot make progress after A0 stops instead of deferring forever; C1 never runs in a process that used the GPU; the D-35 audit reading is an equivalence reading. Appendix D lists every review item and what was done with it.

**What exists now.** No planning generation has been run on any confirmation instance, and **the confirmation draw does not exist**: it is minted at the freeze from a recorded rule (section 3). Everything below was built and tested on the CPU with tiny random stand-ins and model-free simulated F (no GPU, no lease, no real model).

## 1. Question

> **[BWM v1]** T2 s0, an unmerged rule-coverage LoRA on Titan-Sparse, makes ledger one-step and 4-step rollouts near-oracle on its training surface (PREREG-2). Used as the world model F inside BWM's beam planner (planning in imagination, one plan submitted per problem), does it turn that accuracy into goal-reaching plans? The comparison arms are the same backbone with the adapter off, and the exact simulator.

**Binding context (RCWM).** PREREG-3 returned **STATUS: TRIGGER** (REPORT section 3c): T2 is reliable only inside its trained surface (accounts A-D, statuses posted / void, |amount| <= 12, |balance| <= 98, canonical sorted keys), and its void no-op is a token cue. D-34(c) keeps joint item 3 inside that range; D-35 requires a surface-form audit next to every claim. So the claim tested here is scoped to T2's training surface from the start (section 8.6 limits).

## 2. Registered instruments (pinned by SHA-256 at freeze)

> **[BWM v1]**
> - **Planner:** `native_v1/bwmn/blocks_planner.py` (BWM; `plan_beam` with domain hooks), imported read-only. Current SHA-256 18a5f22b94e5304f9dfe117d3afa7d89e5f21596b86ec0412b25d0b24b9ab960.
> - **Kit (RCWM):** `lbeam_f.py`, `lbeam_problems.py` (generator v2), `lbeam_proposer.py`, `lbeam_run.py` (to be written), at their SHAs at freeze.
> - **Adapter:** the PREREG-2 registered T2 s0 adapter (unmerged).
> - **Call:** `plan_beam(P.planner_instances(insts), EnumerateProposer, LF.guarded(LedgerF(arm), insts), eval_seed, **PR.beam_kwargs(prop, W=2, E=12))`. Instances are blind (id, init, goal, vocabulary).
> - **Guard:** D_CLOSE = 8, |b| <= 98, trained amounts, statuses posted/void, and `max_depth = D_CLOSE - sim_len`. The same guard applies to every arm; a refused key is `None` (the candidate stops) and refused keys are reported per arm.
> - **Decoding:** 160 new tokens, greedy. F is the first computed (cached) answer, with per-arm caches; batch invariance is measured at the anchor (batch 1 against batched, about 500 keys).
> - **Instrument identity:** PREREG-2's A0 anchor reruns before any planning generation and on every resumed turn.

Concretely (RCWM, no change of substance):
- **Planner:** BWM's `plan_beam`, imported read-only from the BWM bundle at the pinned sha256; the runner refuses if the file's sha256 differs (at import, at every turn, in `--verify`, in the analysis). No bytecode is written into the BWM bundle. `bwmn/__init__.py`, `blocks.py` and `lm.py` (imported by the planner module, not used by `plan_beam` when the hooks are given) are hashed into the freeze record as information, not as binding pins (Appendix B, Q3); **[RCWM amendment]** the sha256 of every BWM module loaded is also recorded for every turn and every CPU run, and the analysis reports any drift from the freeze record (report-only). *Reason: they run at import time inside RCWM's GPU process; a change must at least be visible.*
- **Kit:** `scripts/lbeam_f.py` 9a76562c..., `lbeam_problems.py` 2353e2fc..., `lbeam_proposer.py` 6c555d2e..., `lbeam_test.py` 1f21b65f... (D-37), plus `lbeam_run.py`, `lbeam_analysis.py` and `lbeam_test_run.py` (this preregistration), all hashed at the freeze. No kit file was changed for this draft.
- **Model:** Titan-Sparse FP32 (fingerprint `818be9c6b6a3`) plus `runs/t2/t2_s0/adapter` (adapter_model.safetensors sha256 `92bc7c74...`, adapter_config.json `50d9a776...`), loaded unmerged by `lbeam_f.load_registered_t2` exactly as `confirm2_gen.py` loads it (adapter name `t2`; D-28: never dense-merged; every generation call asserts no merged LoRA layer). **T2** = adapter `t2` active; **OFF** = the same PeftModel inside `model.disable_adapter()`, in the same process.
- **Call:** exactly BWM's, with `EnumerateProposer()` (depth 1: the 8 vocabulary actions as one-step candidates, 0 proposer tokens), `eval_seed = 0`, `PR.beam_kwargs(prop, 2, 12)` = width 2, max_expansions 12, max_rollout 0, **max_depth 7** (= D_CLOSE 8 minus the candidate length 1), hooks `goal_met` and `unmet`.
- **Instance ids:** `P.planner_instances` passes the kit's ids (`lb-lbc-in-d<d*>-<slot>`), which encode the optimal depth. No registered component reads them: `EnumerateProposer` ignores ids, `plan_beam` passes them through, and `select_fixed` uses the id only to seed a random tie order. The vocabulary is shuffled at generation, and the hooks see only the goal and the imagined state.
- **F:** `LF.guarded(LF.LedgerF(arm, ...), insts)` for every arm. LedgerF renders only the canonical frame and asserts every prompt byte for byte before generation. Decoding: greedy, EOS [151645, 151643], thinking off, FP32, TF32 off, 160 new tokens, `rcwmt.generation.generate_texts` batching (<= 32 prompts, batch x (prompt + 160) <= 8,000 tokens). Parser `env.parse_state(..., lenient=True)`; an answer whose account set differs is `None`.
- **Runtime:** torch 2.6.0+cu124, transformers 5.14.1, peft 0.20.0, checked after the model load and before anything is generated; a drift stops the job (persisted).

**[RCWM amendment] A0, exactly.** *Reason: "reruns PREREG-2's A0 anchor" and "batch invariance at the anchor" need an operational definition that the runner can check mechanically.*
- **(a) Anchor:** PREREG-2's A0 exactly as `confirm2_gen.py` defines it: the 208 pilot TEST ledger `chain_one` chains (`data/bank_pilot.json`, split test; 208 chains x 4 events), generated with `rcwmt.generation.chain` first OFF (inside `disable_adapter()`) then ON (adapter `t2`), each compared with its T2 s0 reference (`runs/t2/t2_s0/gen_chain_one_base.jsonl` for OFF, `gen_chain_one.jsonl` for ON; both PREREG-2 pins). Pass iff identical states on **>= 99%** of chains on each side; identical texts are recorded as a diagnostic. Two neutral differences from `confirm2_gen.py`, stated so the definition is complete: the ON side calls `model.set_adapter('t2')` and asserts that PEFT reports the `t2` adapter active before generating (confirm2 relied on the adapter being active after the `disable_adapter()` context exits); and a chain whose `cand_id` is missing from the reference counts as a mismatch instead of raising an error.
- **(b) Batch invariance:** **256 keys per arm** (512 in all, BWM's "about 500"): init x every vocabulary action of the first 32 instances of the DEV set `dev2_primary.jsonl` (never a confirmation key, so no confirmation output exists before A0 passes). For T2 and for OFF, the 256 keys are sent once as ONE `LedgerF.predict` list (one generation call, the registered sub-batches) and once as one key per predict call (batch 1). Pass iff the parsed states are identical on **>= 99%** of keys per arm, the batched path made exactly one generation call and the batch-1 path one call per key. Identical texts are recorded.
- A0 runs before any planning output and again at the start of **every** later turn that loads the model, before any planning stage.

**[RCWM amendment] F as "the first computed answer" across stages.** *Reason: the arms are run in several stages and possibly several turns; F must be one function per arm, not one per stage.* Each stage loads the cumulative cache of the latest earlier RECORDED stage for that arm (a hash-verified output; the cache of a stage that is not recorded never counts) through the kit's own cache loader (class, arm, frame and identity checked), never recomputes a cached key, and writes its own cumulative cache (create-only) when it completes. The analysis verifies that every later cache extends the earlier one entry for entry and that no key appears twice. Per-arm caches: T2 canonical, OFF canonical, T2 under the audit surface; ORACLE and IGNORE_STATUS are exact functions.

## 3. Instances

> **[BWM v1]**
> - **Primary:** N = 180 fresh instances. The generator-v2 seed is minted at freeze; dev seeds 20261141-43 are spent. The freshness union is rebuilt at freeze. There is no enrichment: the needs-lookahead share is expected to be about 36% (about 65 instances).
> - **Out-of-range set:** descriptive only, and not part of the confirmation turn.

**[RCWM amendment] (d) The fresh confirmation draw.** *Reason: "minted at freeze" must exclude choosing a seed by looking at instances, and the stratum that carries co-primary 2 must be defined before any instance exists.*
- **Generator and spec:** generator v2 (`lbeam_problems.py`, GEN_VERSION `lbeam-problems-v2`), primary spec only: **36 instances per optimal depth d* = 2, 3, 4, 5, 6 (N = 180)**, V = 8, K = 8 fixed candidates, closure depth D_CLOSE = 8, bound |b| <= 98, no out-of-range instances, no enrichment. Set name `lbc`.
- **Seed rule (recorded):** `seed = int(sha256("PREREG-LB confirmation draw v2|<sha256 of this draft file>|<freshness-union digest>|<BWM's salt>")[:12], 16)`. The union digest is sha256 over the sorted payloads and the sorted FDS operative contents of the union rebuilt at the freeze. **Commit-reveal:** BWM chooses a random salt of at least 16 characters, puts its sha256 into its co-sign as `commit:<64 hex>`, and reveals the salt to the operator only for the freeze (`--bwm-salt`); the freeze refuses a salt whose sha256 is not the committed one. No input can be chosen by looking at instances (this draft is final and co-signed before the freeze, the union is rebuilt from the data on disk at the freeze, the salt is fixed by BWM's commitment before RCWM knows it), and nobody can compute the draw before the freeze (RCWM lacks the salt; BWM lacks nothing it could use, since it never sees the draw and RCWM holds the answer key). The freeze refuses a seed equal to a spent seed (20261141, 20261142, 20261143). `--verify` recomputes the seed from the recorded draft sha256, union digest and salt, and checks the salt against the commitment in the co-sign.
- **Freshness union, rebuilt at the freeze:** `lbeam_problems.load_exclusions(historical=True)`: the name-skipping historical harvest (`HSKILL_CONFIRM_V2*` never opened or hashed; `rcwmt.bank.load_exclusions()` never called), every pinned RCWM source, PREREG-3's bank, every payload and rolled-out multi-event transition of every other top-level `data/*.json[l]` file; **plus [RCWM amendment] every payload of the data/ subdirectories** (today `data/t2br_superseded/rev3/`), *because superseded training files moved there by other sessions would otherwise silently leave the union*; **plus [RCWM amendment] every binding transition of the three spent dev sets** (`dev_set.jsonl`, `dev2_primary.jsonl`, `dev2_oor.jsonl`), *because the dev sets were used for harness and budget tuning and are not in `data/`*. A draw whose binding transitions (init x vocabulary, every optimal-plan step, every fixed-candidate step) overlap the union in payload or operative content is redrawn and recorded.
- **[RCWM amendment] Concurrent writers.** *Reason: other sessions write data/ while RCWM works.* The freeze hashes every `data/**/*.json[l]` file (`HSKILL_CONFIRM_V2*` skipped by name) before and after building the union and refuses if anything changed; it also refuses if any source of the union could not be read. The freeze is run while no agent writes to data/ (the operator checks this before running it). The snapshot's file count and digest are recorded.
- **Mint and seal:** inside `--freeze`, after every pre-freeze check and before anything is renamed into place: build, then require 0 payload and 0 operative overlaps, `verify_instances` ok, byte-for-byte regeneration from (seed, spec, recorded rejections), and 36 per d*. The draw is written create-only as `docs/ledger_beam/LB_CONFIRM_DRAW.jsonl` with `LB_CONFIRM_DRAW.SOURCE.json` (spec, seed rule and its inputs, union sources, rejection counts, overlap counts, data/ snapshot; no instance content), and sealed by its sha256 in the freeze record and in section 10.
- **Never inspected before the run:** the freeze prints only the seal and the count (a failed verification reports only the number of problems). Until the registered analysis, the draw file is read only by `lbeam_run.py` (the job) and `lbeam_analysis.py`, and hashed by `--verify`. Nobody opens it, prints an instance, or computes anything from it (including its needs-lookahead count). A draw found without a freeze (an interrupted freeze) is moved aside unread and logged; the rule re-mints the same draw if the union and the salt are unchanged.
- **Needs-lookahead stratum (instance property, computed only by the analysis):** an instance needs lookahead iff `lbeam_problems.greedy_solvable(inst)` is false. That procedure, registered exactly: start at init; repeat at most D_MAX = 6 times: stop if the goal is met; otherwise move to the successor state, under the TRUE dynamics (`env.step`; void twins are no-ops), that minimises the true `unmet` hook over the instance's full vocabulary, the first such action in vocabulary order. Because every instance has at least one void twin (a true no-op), the minimum is never above the current value: greedy never moves uphill; it moves to a lower state, moves sideways (or stays put) at a local minimum, and fails if the goal is not met after 6 steps. On the dev set this solves 77/120 (by d* = 2..6: 0.96, 0.71, 0.71, 0.46, 0.38), so about 36% (about 64 of 180) are expected to need lookahead; the realised count is reported, not chosen. Since D_MAX is fixed, greedy has 4 slack steps at d* = 2 and none at d* = 6, so the stratum is partly a depth stratum; the co-primaries' discordant pairs are therefore also reported by d* (section 6).
- **Out-of-range set:** as BWM: descriptive only, not in this preregistration's run.

## 4. Arms

> **[BWM v1]** T2 (adapter on), OFF (`disable_adapter`), ORACLE (env.step), all at the identical planner configuration. Secondary and report-only arms:
>
> - IGNORE_STATUS (a diagnostic);
> - the `select_fixed` replication (T2 / OFF / ORACLE / IGNORE_STATUS);
> - the enum W4E24 curve point;
> - the `SurfaceAuditF` held-out-combined surface for T2 only (option i of §12.7). T2 s0's planning claim is registered as **training-surface (trigger-level)**; a surface-diversified T2b confirmation is a separate, later preregistration.

**[RCWM amendment] (e) Which arms need the GPU.** *Reason: only a model F needs the lease; exact and rule F must never hold it, and must not run in a process that holds a CUDA context.*
- **GPU (model F, one lease `rcwm-t1:lbeam`):** T2 and OFF at enum W2 E12 (the primary pair); `select_fixed` T2 and OFF (they need model F, so they cannot be CPU stages); T2 at enum W4 E24; T2 under `SurfaceAuditF('T2', 'heldout_combined')` at enum W2 E12.
- **CPU only, no GPU, never under the lease (stage C1):** ORACLE (exact `env.step`) at enum W2 E12 and W4 E24; IGNORE_STATUS (rule F: every status applied as posted) at enum W2 E12; `select_fixed` ORACLE and IGNORE_STATUS. They use exact or rule F and load no model. C1 runs only after A0 has passed (so an A0 failure leaves no planning output of any arm), and only at the START of a later invocation of the job than the one that recorded A0: before that invocation's lease, in a process that has not initialised CUDA (checked and recorded before and after; the analysis refuses a C1 run not recorded CUDA-free). The invocation that records A0 therefore ends with exit 5 while C1 is pending, even if every GPU stage is done, and the hook's one resume call runs C1 first.
- The W4 E24 point is run for T2 and ORACLE (the T2/ORACLE ratio at that point is report-only); OFF is not run at W4 E24.

## 5. Outcomes, statistics and decision rules (registered verbatim; applied mechanically by `scripts/lbeam_analysis.py`)

> **[BWM v1]**
> - **Success:** the submitted plan reaches the goal exactly under env.step. The submission is the verified plan if one exists, otherwise the best node's prefix; unverified submissions count by the same rule.
> - **Co-primaries** (Holm across the two):
>   1. T2 against OFF, paired per instance, over all instances. Exact McNemar.
>   2. The same contrast on the needs-lookahead stratum (instances greedy descent on the true hook does not solve).
> - **Key secondary:** T2/ORACLE success ratio, with a paired bootstrap 95% CI (10,000 resamples over instances).
> - **Pre-set reading:**
>   - "T2 enables planning" requires T2 - OFF >= +0.20 with Holm-adjusted p < 0.01 on co-primary 1.
>   - "Near-oracle planning" additionally requires the T2/ORACLE ratio to be >= 0.80 with lower CI >= 0.70.
>   - Anything else is reported as "partial" or "no".
> - **Mechanism and cost (report only):**
>   - verified-plan precision per arm;
>   - optimal-length rate;
>   - unique F keys and tokens per instance;
>   - refused keys;
>   - success by optimal depth d* 2-6;
>   - **void-twin exposure**: the share of failures whose submitted plan contains a void twin applied as posted, which links back to PREREG-3's trigger finding.
> - **Matched calls:** the primary contrasts compare F quality at an identical planner configuration, so they are not call-capped (realised calls are reported). Only the "beam vs one-shot" secondary uses hard `BudgetF` caps, set from ORACLE.

**Registered rules (verbatim):**
- **Success:** the ONE submitted plan (`plan_beam`'s `plan`: the verified plan, else the best node's prefix) reaches the goal exactly when executed from init under `env.step` (`lbeam_problems.execute`). Unverified submissions count by the same rule. Every arm and configuration is scored this way.
- **Co-primary 1:** T2 vs OFF at enum W2 E12, paired per instance over ALL N instances: exact two-sided McNemar on the discordant pairs (b = T2 only, c = OFF only), p = min(1, 2 P(X <= min(b, c))), X ~ Bin(b + c, 1/2).
- **Co-primary 2:** the same contrast on the needs-lookahead stratum (section 3).
- **Holm** across the two co-primaries (m = 2, monotone step-down).
- **Status per co-primary**, with delta = (b - c) / n on that co-primary's instances:

| status | condition |
|---|---|
| PASS | Holm-adjusted p < 0.01 and delta >= +0.20 |
| SHORT | Holm-adjusted p < 0.01, b > c, delta < +0.20 |
| REVERSED | Holm-adjusted p < 0.01, c > b |
| NS | Holm-adjusted p >= 0.01 |

- **Key secondary:** T2/ORACLE = success(T2) / success(ORACLE) at enum W2 E12 over all instances, with a paired percentile bootstrap 95% CI over instances (unstratified: instances are resampled over all 180, not within d*, which makes the interval slightly wide and the bar conservative; 10,000 resamples, `numpy.random.default_rng(17)`, numpy's default linear percentile; resamples whose ORACLE mean is 0 are dropped and counted). **NEAR** iff ratio >= 0.80 and lower bound >= 0.70. If ORACLE succeeds on no instance the ratio is not evaluable and NEAR is false.
- **Branch (first match; s1 / s2 = the statuses of co-primary 1 / 2):**

| branch | condition | reading |
|---|---|---|
| MIXED_PASS | s1 PASS, s2 REVERSED | enables planning, qualified: significantly worse than OFF where lookahead is needed |
| MIXED_SHORT | s1 SHORT, s2 REVERSED | partial (mixed) |
| MIXED_LOOKAHEAD_BETTER | s1 REVERSED, s2 PASS or SHORT | partial (mixed) |
| REVERSED | s1 REVERSED, s2 NS or REVERSED | no |
| REVERSED_LOOKAHEAD | s1 NS, s2 REVERSED | no |
| YES_NEAR | s1 PASS, s2 PASS, NEAR | near-oracle planning |
| YES | s1 PASS, s2 PASS, not NEAR | enables planning |
| GREEDY_ONLY_NEAR | s1 PASS, s2 SHORT or NS, NEAR | near-oracle planning, qualified: not shown where lookahead is needed |
| GREEDY_ONLY | s1 PASS, s2 SHORT or NS, not NEAR | enables planning, qualified: not shown where lookahead is needed |
| LOOKAHEAD_ONLY | s1 SHORT or NS, s2 PASS | partial |
| DIRECTIONAL | s1 and s2 in SHORT / NS, at least one SHORT | partial |
| NO | s1 NS, s2 NS | no |

- **What the verdict needs:** the branch is computed from A0 (the first run passed), G1 (T2 and OFF at enum W2 E12) and C1 (ORACLE at enum W2 E12). It is rendered when those three are recorded and hash-verified and the run ended either complete (`DONE.json`) or by a deterministic stop that the runner marked verdict-eligible (section 7, amendment (b)). Report-only stages that are not recorded are rendered "not evaluable" by the registered sentence of section 8.5; they never block or change the branch.

**[RCWM amendment] (a) A pass bar for co-primary 2, and a reading for every combination.** *Reason: BWM's reading sets a bar for co-primary 1 only, so without an amendment the outcome "co-primary 1 passes, co-primary 2 fails" (and the reverse) has no registered wording. The planning signal lives in the needs-lookahead stratum (INTERFACE finding 3: 64% of instances are solved by greedy descent on the true hook), so a gain carried by greedy-solvable instances must be worded as such. Revision 2 adds the mixed-sign outcomes (one co-primary significant in T2's favour, the other in OFF's): revision 1 sent them all to REVERSED, whose sentences then contradicted a passing co-primary 1.* Co-primary 2 gets the same bar as co-primary 1 (delta >= +0.20, Holm-adjusted p < 0.01); the ORACLE ceiling there is about 0.58 (dev), which leaves room for a +0.20 gain. "T2 enables planning" (BWM: co-primary 1) keeps its meaning: every branch with s1 PASS reads "enables planning" or "near-oracle planning", qualified when co-primary 2 does not pass ("not shown where lookahead is needed") or reverses ("significantly worse than OFF where lookahead is needed"). "Near-oracle planning" (BWM: co-primary 1 plus the ratio bar) is read only when s1 is PASS and s2 is not REVERSED; NEAR without that is reported by a qualifier. Co-primary 2 alone never yields "enables planning". All 4 x 4 x 2 combinations of (status 1, status 2, NEAR) map to exactly one branch; the dry run realises every one of them with synthetic paired outcomes through the real `decide()` and `wording()` and checks the rendered text mechanically (section 8.7).

**Why these thresholds** (BWM's, kept): +0.20 and Holm p < 0.01 make "enables planning" a large and unambiguous effect at N = 180; 0.80 with lower bound 0.70 is the programme's bar (PREREG-3). The ratio is against the exact simulator at the identical planner configuration, so it measures how much of the planner's ceiling T2's F realises.

**[RCWM amendment] What "enables planning" does and does not show.** *Reason (statistics review): OFF's F is close to useless on this task, so co-primary 1 passes for almost any moderately better F (section 5b: probability 1.000 even for the pessimistic T2).* A pass of co-primary 1 therefore does not distinguish near-oracle from mediocre prediction; the construct "near-oracle prediction becomes near-oracle planning" rests on the T2/ORACLE ratio. This is an always-rendered limit (section 8.6), and two report-only contrasts bear on the construct: T2/ORACLE on the needs-lookahead stratum and T2 vs IGNORE_STATUS (section 6).

## 5b. Power (computed, not guessed) [RCWM amendment (c)]

*Reason: BWM's draft fixes N = 180 without a power statement; RCWM's sizing data allow one.* `docs/ledger_beam/LB_POWER.json` (`scripts/lbeam_test_run.py --part power`; CPU, model-free, about 40 s).

**Method.** On the 120 DEV instances (`dev2_primary`, the registered call: enum W2 E12, guarded, registered max_depth), per-instance success was computed for ORACLE and for model-free SimulatedF stand-ins of each arm (INTERFACE section 4.4), each with 5 noise seeds: **SIM_T2** (posted answers wrong 0.09: PREREG-2's four-step exact 0.698 = 0.914 per step; void 1.00), **SIM_T2_P3** (posted wrong 0.16: PREREG-3's in-distribution posted accuracy 0.84), a **pessimistic** T2 (posted wrong 0.25, void applied as posted 0.05) and **SIM_OFF** (void applied 0.90, posted wrong 0.55: PREREG-2's OFF step accuracy). A confirmation-sized set (36 per d*) is resampled with replacement within d* from the paired (T2, OFF, ORACLE) triples and the registered `decide()` is applied (Holm across both co-primaries; the ratio bootstrap with B = 2,000 here, computed whenever the point ratio is >= 0.80); 400 replicates per seed x 5 seeds = 2,000 per scenario. **Pairing.** SimulatedF draws its errors from sha256(seed, key) uniforms shared by every preset, so at one seed T2's errors are nested in OFF's: the T2-OFF correlation of the "as simulated" figures is a simulation artifact, not an estimate of the real instance-level correlation. The "OFF decorrelated" figures take OFF from seed + 100; the truth for the real models is unknown and the two columns bracket it.

**Dev success rates (mean over 5 seeds; all / needs-lookahead / greedy-solvable):** ORACLE 0.850 / 0.581 / 1.000 (W4 E24: 0.900); SIM_T2 0.735 / 0.433 / 0.904; SIM_T2_P3 0.647 / 0.349 / 0.813; pessimistic 0.528 / 0.284 / 0.665; SIM_OFF 0.123 / 0.037 / 0.171 (decorrelated: 0.115 / 0.023 / 0.166). T2/ORACLE over all instances / on the needs-lookahead stratum: SIM_T2 0.865 / 0.744; SIM_T2_P3 0.761 / 0.600; pessimistic 0.622 / 0.488.

| scenario (T2 stand-in vs SIM_OFF) | P(co-primary 1 PASS) | P(co-primary 2 PASS), as simulated / OFF decorrelated | P(near-oracle reading) | branch distribution (as simulated) |
|---|---:|---:|---:|---|
| optimistic: SIM_T2 | 1.000 | 0.989 / 0.998 | 0.954 | YES_NEAR 0.947, YES 0.043, GREEDY_ONLY_NEAR 0.008, GREEDY_ONLY 0.004 |
| central: SIM_T2_P3 | 1.000 | 0.865 / 0.920 | 0.268 | YES 0.602, YES_NEAR 0.263, GREEDY_ONLY 0.130, GREEDY_ONLY_NEAR 0.005 |
| pessimistic | 1.000 | 0.740 / 0.830 | 0.000 | YES 0.740, GREEDY_ONLY 0.260 |

The needs-lookahead stratum averaged 64.5 instances in the resamples. **Seed sensitivity:** co-primary 2's power rests on T2's success on only 43 dev needs-lookahead instances, which varies across the 5 noise seeds at the same per-step accuracy (SIM_T2_P3: 0.256-0.488); per seed, P(co-primary 2 PASS) ranges 0.548-1.000 central and 0.545-0.960 pessimistic as simulated (0.733-1.000 and 0.715-0.938 decorrelated). Read co-primary 2's power as about 0.74-0.92 from pessimistic to central, not as a point value.

**Exact power under independence** (T2 and OFF outcomes independent within an instance; exact multinomial enumeration; P(delta >= +0.20 and exact McNemar p < alpha), with alpha 0.005 a lower bound and 0.01 an upper bound on the Holm pass):
- **Co-primary 1 (n = 180):** about 0.98 (>= 0.978) whenever the true T2 - OFF >= 0.30 (e.g. T2 0.55 / OFF 0.25: 0.981; T2 0.65 / OFF 0.35: 0.979; T2 0.50 / OFF 0.16: 0.999); about 0.52 at a true difference of exactly 0.20 (T2 0.55 / OFF 0.35 and T2 0.65 / OFF 0.45), where the +0.20 bar binds.
- **Co-primary 2 (n = 64):** T2 0.40 / OFF 0.05: 0.98-0.99; T2 0.35 / OFF 0.05: 0.93-0.94; T2 0.40 / OFF 0.10: 0.84-0.89; T2 0.30 / OFF 0.05: 0.78-0.80; T2 0.25 / OFF 0.05: 0.51-0.52. At n = 55 / 75 the T2 0.35 / OFF 0.05 cell is 0.87-0.90 / 0.96.

**Reading.** If T2's planning quality is anywhere near its simulated one-step quality, co-primary 1 passes essentially always (and, as section 5 states, a pass says little on its own); co-primary 2 passes with probability about 0.74-0.92 from the pessimistic to the central scenario (0.99 optimistic), so a GREEDY_ONLY outcome is a realistic possibility (0.13 central and 0.26 pessimistic as simulated; 0.08 and 0.17 decorrelated), which is why amendment (a) words it. The near-oracle reading is likely only if T2's posted accuracy is near PREREG-2's level (0.95) and a coin flip or worse at PREREG-3's (0.27 central). The mixed-sign branches have probability about 0 in every scenario; they are registered because the wording is frozen. These figures rest on simulated F, not on any model output.

## 6. Report-only analyses (no decision depends on them)

- **Per arm and configuration:** success, success by d* 2-6, success on the needs-lookahead and greedy-solvable instances, verified rate, **verified-plan precision** (share of verified submissions that succeed), **optimal-length rate** (success with a plan of length d*), failure kinds. Arms and configurations: T2, OFF, ORACLE, IGNORE_STATUS at enum W2 E12; T2 and ORACLE at W4 E24; T2 under the audit surface at W2 E12; `select_fixed` for T2, OFF, ORACLE and IGNORE_STATUS.
- **Cost:** distinct F keys per instance (requested through the guard, cache-independent; refused keys are free), mean / median / p90 / max; answer tokens per instance (each requested key's answer text re-encoded with the registered tokenizer) and the exact generated-token count of each stage; **refused keys per arm** (distinct, share of guarded keys, instances with a refused key).
- **Void-twin exposure** per arm (BWM's definition, made operational): among FAILED instances, the share whose submitted plan contains a void action that this arm's F applied as if posted, i.e. F's first computed answer for that step of the plan's own imagined trajectory (read from the stage's cumulative cache: exactly what the planner used) equals the posted application and differs from the input state. **[RCWM amendment]** Also the share whose plan contains a void step that F did not treat as a no-op at all (predicted state differs from the input state, whether applied as posted or with an arithmetic error), failures containing any void step, void steps in failed plans and those applied as posted, and keys the guard refused (absent from the cache; a select_fixed candidate can be selected although a step was refused). `select_fixed` exposures are labelled as including the stored `void_swap` distractors. *Reason: the applied-as-posted count alone undercounts a void step applied with an arithmetic error.*
- **[RCWM amendment] Construct contrasts.** *Reason (statistics review): see section 5.* T2/ORACLE on the needs-lookahead stratum with its paired bootstrap CI (same B and seed rule); T2 vs IGNORE_STATUS at enum W2 E12 (paired, exact McNemar; IGNORE_STATUS is exact arithmetic without the status rule, CPU, no extra GPU), which bears on whether T2's status-rule competence, not its arithmetic alone, carries its planning; the co-primaries' discordant pairs by d* (so "where lookahead is needed" can be read apart from "deep").
- **D-35 audit (T2 only):** T2 under `SurfaceAuditF('T2', 'heldout_combined')` (keys reversed, accounts rendered bank / card / cash / fees, 'void' rendered 'cancelled'), paired with canonical T2: difference, paired percentile bootstrap 95% CI of the difference (unstratified, 10,000 resamples, seed 17) and exact McNemar. **[RCWM amendment] Three-way reading:** surface-dependent iff the drop is significant (McNemar p < 0.05 and the difference < 0); otherwise surface-invariant within 0.05 iff the lower 95% bound of the difference is >= -0.05 (an equivalence reading); otherwise inconclusive. *Reason (statistics review): revision 1 read non-significance as invariance, and one held-out rendering cannot establish "rule-level" (PREREG-3's criterion is generalisation to unseen statuses), so that label is dropped.* PREREG-3 predicts surface-dependent. It never changes the branch.
- **W4 E24 point** (T2, ORACLE), **select_fixed** replication, **IGNORE_STATUS** floor, answer-free floors (greedy rate), and the per-instance success arrays of every arm (for BWM's independent re-scoring).
- **[RCWM amendment] Instrument and deviation record:** the terminal record (complete or stopped, with the stop reason), A0's anchor and batch-invariance rates (OFF / ON, T2 / OFF; states and texts), every turn's re-checks, deferrals, no-progress flags and cap, orphaned stages and their directories, stage attempts and abandoned stages, failed re-check records, and BWM modules that differ from the freeze record. *Reason (engineering review): the results must show the instrument checks and every deviation, not only the outcomes.*
- **[RCWM amendment] (b) refused-key flag:** if the guard refuses more than **1%** of T2's distinct guarded keys at enum W2 E12, the results carry the flag sentence of section 8.5 (report-only; never a stop, never a branch change). *Reason for 1%:* on the dev sizing the guard refused 0.018% of keys for SIM_T2_P3 (max 0.09% over seeds), 0% for SIM_T2 and 0.12% for SIM_OFF (max 0.34%); with an exact F it is inert (0 in 47 runs). A T2 share above 1% (about 3 x the worst simulated OFF) means T2's own arithmetic errors carry its imagined states past |b| <= 98 often enough that the guard, not F, ends part of its search; readers must know that. Since every truly reachable state is in range, a refusal also reveals an F error and prunes that branch, a signal an F-only planner would not have (small, and it favours OFF, which is refused more often: conservative for T2 - OFF).
- **Beam vs one-shot at matched calls** (BWM: "only the beam vs one-shot secondary uses hard BudgetF caps"): **[RCWM amendment]** not run in this preregistration. *Reason:* it needs further GPU configurations (one-shot proposers under T2 / OFF with ORACLE-set caps) that BWM's GPU estimate does not include; INTERFACE finding 1 shows that matched-call contrasts mostly measure the candidate set. Proposed for a later, separate preregistration (Appendix B, Q2).

## 7. Procedure and integrity

1. **Pre-freeze records** (exist now; rewritten only before the freeze):
   - `docs/ledger_beam/RUN_TEST_{PLAN,ANALYSIS,TINY,FREEZE,POWER,REHEARSAL,HOOK}.json` (`scripts/lbeam_test_run.py`; tiny random Qwen3 + LoRA stand-ins on the CPU; each records the code sha256s it ran with, and the sha256 of this draft and of the hook draft);
   - `docs/ledger_beam/LB_DRYRUN.{json,md}` (`lbeam_analysis.py --dry-run`: statistics unit tests, every wording branch and all 32 combinations with the truthfulness rules, metrics; records the draft's and the hook draft's sha256);
   - `docs/ledger_beam/LB_POWER.json` (section 5b and the sizing used by the turn rule);
   - `docs/ledger_beam/GEN_PLAN_LB.json` (`lbeam_run.py --plan`);
   - `docs/ledger_beam/queue_hook_lbeam.sh.draft`;
   - the kit records `TEST_ORACLE.json`, `TEST_TINY.json`, `TEST_FRESH.json` (D-37).
2. **Freeze:** `python -B scripts/lbeam_run.py --freeze --approval "<who approved, when, DECISIONS entry D-<n>>" --bwm-cosign "<BWM's co-sign reference> <this draft's full sha256> commit:<sha256 of BWM's salt>" --bwm-salt "<BWM's salt>"` (CPU). It refuses (never persisted; nothing minted or renamed) if:
   - the approval is missing or cites no DECISIONS entry (`D-<n>`);
   - BWM's co-sign is missing, does not quote this draft's full sha256, or carries no `commit:<64 hex>`; the salt is shorter than 16 characters or its sha256 is not the commitment;
   - BWM's draft is not the cited version (sha256 `b2896ba3...`): this draft must then be revised to cite BWM's current draft and co-signed again;
   - PREREG-LB is already frozen, or a freeze was interrupted (`.partial` files, a lone `PREREG_LB.md`, or a draw without a freeze; the message names the recovery step);
   - any PREREG-LB output exists (`runs/lbeam*`, any `PREREG_LB_RESULTS*` under runs/);
   - a frozen input is missing;
   - a pre-freeze check fails: a record was not produced by the current code, `LB_DRYRUN.json`, `RUN_TEST_ANALYSIS.json` or `RUN_TEST_HOOK.json` was not produced with the current draft and hook draft, a sentence the analysis can render is missing from this draft, the kit records do not match the current kit and the pinned BWM planner, or a pin (adapter, anchor references, pilot bank, dev set, BWM planner) differs;
   - a source of the freshness union could not be read, or data/ changed while the union was built.

   Otherwise it rebuilds the freshness union, mints and seals the draw (section 3), hashes every input (this draft, the lbeam scripts and kit, `p3_common.py`, `p3_make_bank.py`, `l1_make_ood.py`, `confirm2_gen.py`, `rcwmt/*.py`, the pilot bank, the adapter, the anchor references, the dev set and its record, INTERFACE.md, every pre-freeze record, the hook draft, the draw and its SOURCE record) and writes `PREREG_LB.md` (this draft with the status line and section 10 filled; the draft is kept) and `PREREG_LB_FREEZE.json` (approval, co-sign, inputs, pre-freeze checks, registered runtime, draw summary, BWM planner sha, BWM draft sha at freeze, salt commitment). The draft is read once, and its bytes seed the draw, are hashed and become `PREREG_LB.md`. All four files are written as `.partial` first and then renamed in order (draw, draw SOURCE, PREREG_LB.md, freeze record), create-only. **Recovery from an interrupted freeze:** if `PREREG_LB.md` exists and `PREREG_LB_FREEZE.json.partial` records its sha256, rename the `.partial` into place and run `--verify`; otherwise move every `.partial` (and a lone `PREREG_LB.md` or draw) aside UNREAD and rerun `--freeze`. Either way, log it in DECISIONS.md.
3. **Verification:** `lbeam_run.py --verify`. The job and the analysis call the same verifier, and neither runs unless it passes. It checks `PREREG_LB.md`'s hash; every required input against the freeze, with the `PREREG_LB_AMENDMENT_*.json` chain applied (old -> new hash; every amendment must state its reason and who approved it, `approved_by`); the registered pins; BWM's planner sha (recorded and on disk) and BWM's draft sha at the freeze (= the cited one); the draw (freeze record = SOURCE record = input hash), its seed against the rule and the recorded inputs, the salt against the commitment in the co-sign, and its spec; that the co-sign quotes the frozen draft's sha256; the registered runtime; the approval and the co-sign.
4. **GPU job:** `python -B scripts/lbeam_run.py`, one lease `rcwm-t1:lbeam`, under every lease, alternation, thermal and RAM rule (`rcwmt.runtime.gpu_lease`).
   - **Before the lease:** the freeze verifies (a failed verification of an existing freeze is a persisted stop; "not frozen" is never persisted); an installed `scripts/queue_hooks/70_lbeam.sh` is the frozen hook draft byte for byte (not persisted); no other RCWM job holds or waits for the GPU lease (one-waiter rule, R14: a live `rineya-gpu-0.waiting.rcwm.json` of another process, or a live holder whose label starts with `rcwm-t1:`; dead or reused pids do not count; BWM's jobs are waited for in the lease as usual; not persisted); the one-runner lock `runs/lbeam/.run.lock` (a live lock refuses, not persisted; a stale one is reclaimed); the draw, adapter, anchor-reference and dev-set hashes against the freeze (persisted); a failed A0 record or re-check of an earlier turn refuses (persisted). C1 runs here if A0 passed in an earlier invocation (section 4).
   - **After the model load, before generating:** fingerprint, no merged LoRA layer, adapter hashes against the freeze, torch / transformers / peft equal to the registered runtime (any failure: stop, persisted).
   - **Stages** (create-only; every arm of a stage in one process, written together only after all its arms are done):

| stage | what | GPU | expected generations (ceiling) | h at 2.7 / 3.8 gens/s |
|---|---|---|---:|---:|
| A0 | anchor (208 chains x 4 steps x OFF/ON) + batch invariance (256 keys x T2/OFF x batched and batch 1) | yes | 2,688 | 0.70 / 0.63 |
| G1 | **PRIMARY**: plan_beam enum W2 E12, T2 then OFF | yes | 27,121 (37,440) | 2.79 / 1.98 |
| G2 | select_fixed T2 and OFF (new keys only) | yes | 5,681 (17,280) | 0.58 / 0.42 |
| G3 | plan_beam enum W4 E24, T2 (new keys only) | yes | 10,001 (38,880) | 1.03 / 0.73 |
| G4 | SurfaceAuditF T2 heldout_combined, enum W2 E12 | yes | 14,537 (18,720) | 1.50 / 1.06 |
| C1 | ORACLE W2 E12 and W4 E24, IGNORE_STATUS W2 E12, select_fixed ORACLE and IGNORE_STATUS | **no** (CPU, outside the lease, at the start of a later invocation than A0's) | 0 | about 1-2 CPU-min |

   Expected generations use the sizing measured on the dev set with simulated F and the runner's own stage order and shared caches (SIM_T2_P3 for T2, SIM_OFF for OFF and for the audit arm; `LB_POWER.json`); ceilings are the configurations' hard per-instance bounds ((E + W - 1) x V for enum; K x d* for select_fixed). The A0 time assumes 0.3 generations/s for batch-1 calls (duty cycle and throttle per call).
   - **Turn cap** (D-22): 4.5 h from lock acquisition; no stage starts within 0.5 h of the cap; a stage starts only if 1.25 x its estimate fits (the running rate is re-measured after each planning stage). **[RCWM amendment]** The start-of-turn check adds the model load (120 s) and, on a resumed turn, the longest A0 time MEASURED so far in this run (the first A0 or a re-check) instead of A0's estimate; if the first remaining stage does not fit, the model is not loaded at all. Every turn after the first re-runs A0 (anchor + batch invariance) before any planning stage. *Reason (engineering review): with the estimate, a resumed turn whose A0 runs longer than estimated would load the model, re-run A0 and defer again, every turn.*
   - **GPU plan per turn** (`--plan`, `GEN_PLAN_LB.json`):
     - expected sizing at the planning rate 2.7/s: turn 1 = A0, G1, G2 (4.11 h); turn 2 = A0 re-check, G3, G4 (3.26 h); **7.37 GPU-h**; C1 runs on the CPU at the start of the second call, before its lease;
     - expected sizing at PREREG-2's observed ledger rate 3.8/s: turn 1 = A0, G1, G2, G3 (3.80 h); turn 2 = A0 re-check, G4 (1.73 h); **5.53 GPU-h**;
     - ceilings at 3.8/s: four turns (A0 + G1 3.40 h; then each of G2, G3, G4 in its own turn after an A0 re-check: 1.93, 3.51, 2.03 h); 10.87 GPU-h;
     - ceilings at 2.7/s: G1 alone (3.85 h) exceeds the 1.25 x rule; the job starts G1 on its expected size, and at its ceiling it would end about 0.1 h past the cap (disclosed; the cap is a stage-start rule);
     - the primary alone (A0 + G1, plus C1 on the CPU): 3.52 h at 2.7/s, 2.65 h at 3.8/s, one turn;
     - risk: G1 needs 1.25 x 2.79 = 3.49 h, so it can follow an A0 of at most about 1.0 h within the cap at 2.7/s; if A0 takes longer (its batch-1 rate is an assumption), G1 is deferred, the next turn defers on the measured A0 without loading the model, and the run stops at the second turn without progress (no verdict; an operational amendment such as a smaller batch-invariance re-check is then needed, Appendix B, Q1).
   - **Completion:** a stage is done only if its files exist and provenance records them with matching sha256; `DONE.json` lists every output hash and the report-only stages that are not evaluable (abandoned). Provenance and the job log record only planned generation counts for planning stages (realised calls are outputs).
   - **Exit codes:**

| code | meaning |
|---|---|
| 0 | done (or already done) |
| 3 | deterministic stop: never retried; persisted as `STOPPED.json` when it concerns the run, with `verdict_eligible` true or false (not persisted: "not frozen", "hook not the frozen draft", "another RCWM job holds or waits for the lease", "run lock held", "STOPPED.json exists") |
| 5 | deferred by the cap at a stage boundary, or C1 still to run; resumes next call (C1 first if pending, then the A0 re-check) |
| 1 | crash (the lease is released); the hook retries once |

5. **Queue.** The hook draft `docs/ledger_beam/queue_hook_lbeam.sh.draft` is installed byte for byte as `scripts/queue_hooks/70_lbeam.sh` only after the freeze, by the RCWM-T1 operator on the user's go-ahead, logged in DECISIONS.md. `master_chain.sh` expands its hook glob once, when execution enters the hook loop, so a chain already past that point never sources a new hook; it then runs only in a fresh chain started when no chain process is alive. The hook is idempotent; adds no second GPU waiter (and the runner refuses to be one); makes **no retry after exit 3** or a persisted `STOPPED.json`; makes **exactly one resume call after exit 5** and one retry after a crash (at most two calls per chain pass); runs the analysis once, after `DONE.json` or after a `STOPPED.json` marked `"verdict_eligible": true`.
6. **No peeking.** Between the freeze and the registered analysis nobody opens the draw. Between the first planning output and the registered analysis nobody reads `runs/lbeam/{primary,fixed,w4e24,audit,cpu}/*` or computes anything from them; only `provenance.json`, `anchor/*`, `STOPPED.json` / `DONE.json` and the job log may be read, as instrument checks. No analysis code changes after the first planning output, except a logged non-statistical crash fix (an amendment).
7. **Analysis:** `python -B scripts/lbeam_analysis.py`, once, CPU, numpy only, torch never imported. It verifies the freeze, then loads and validates the run, and dies on any failure: a terminal record (`DONE.json`, or a verdict-eligible `STOPPED.json`, never both), and provenance unchanged since it; A0, G1 and C1 recorded; every recorded stage's files against provenance (and `DONE.json`); a stage both recorded and abandoned, or a core stage abandoned; the draw's hash (freeze = provenance = file); A0 anchor and batch invariance passed; a failed re-check only with a stop; every turn that ran a planning stage after A0's turn passed its A0 re-check (anchor and batch invariance); no planning stage before A0; C1 after A0, outside the lease and CUDA-free; T2 and OFF of G1 and G2 in one process; per-turn code hashes against the freeze and amendments; runtime versions, peft and device (cuda) for every turn that loaded the model; adapter hashes at load; fingerprint; BWM planner sha per turn and for C1; episode files list the draw's ids in order with plans inside each instance's vocabulary; cache headers (class, arm, frame), no duplicate key, every later cache extends the earlier one, every key owned by a draw instance. It writes `runs/lbeam/analysis/PREREG_LB_RESULTS.json` and `.md` as one create-only set (`.partial` first, then renamed).
8. **Report** exactly as computed: a REPORT section and a RESULTS file rendered from the JSON; BWM co-signs the report (section 9).

**[RCWM amendment] (b) Deviation and stop policy.** *Reason: BWM's draft names the instrument checks but not what happens when they fail, nor crash and resume rules. Revision 2 makes every case decidable by a registered rule, so no discretion remains once outputs exist.*
- **A0 anchor or batch-invariance failure in the first A0:** the job stops (exit 3) BEFORE ANY PLANNING OUTPUT of any arm (C1 also waits for A0), and the failure is persisted (`anchor.json` or `batch_invariance.json` with pass false, plus `STOPPED.json` with `verdict_eligible` false). **No verdict.** The cause is investigated and logged: if it is a software or environment change, restore the registered environment, move the stop record and the failed A0 records aside with a DECISIONS entry, and rerun (A0 runs again before anything); any other remedy is an amendment, recorded before any planning output exists.
- **A0 re-check failure on a resumed turn:** the job stops before any planning stage of that turn (persisted: `RECHECK_FAILED_*.json` and `STOPPED.json`). Outputs recorded in earlier turns were made under a passing A0 and stay valid: if A0, G1 and C1 are recorded and unchanged, the stop is **verdict-eligible**, and the analysis renders the verdict with the report-only stages that did not run "not evaluable" (not run: stopped). Otherwise no verdict.
- **Any other deterministic stop** (runtime drift, fingerprint, adapter hash at load, BWM planner sha, a changed recorded output, a failed freeze verification, no progress, a core stage failing three times): persisted; verdict-eligible iff A0's first run passed and A0, G1 and C1 are recorded with every recorded output unchanged (a changed output is never eligible); the analysis re-verifies the freeze and every provenance check independently.
- **T2 refused-key share above 1%:** report flag only (section 6).
- **GPU crash or reset:** a stage that is not done (half-written files, stray `.partial` files, files without a provenance record) is moved to `runs/lbeam/_orphans/` (never deleted; recorded in provenance) and regenerated in full, all arms together; no stage is stitched across processes; caches of a stage that is not done are orphaned with it, so F stays "the first computed answer" of a recorded stage. A recorded output whose hash changes stops the job. The hook retries once after a crash; the resumed turn re-runs A0 first.
- **Attempts:** every stage start is recorded; a started stage that is not recorded failed. A report-only stage (G2, G3, G4) that failed twice is abandoned: never retried, rendered "not evaluable" (abandoned after its failed attempts), and the job continues. A core stage (A0, G1, C1) that failed three times stops the job (no verdict).
- **Deferral:** exit 5 at a stage boundary; the next turn re-runs A0 (anchor and batch invariance) and continues; one resume call per chain pass. **No progress:** two consecutive turns without a recorded stage stop the job (exit 3, persisted), since the start-of-turn check uses the measured A0 time and a stall would repeat; the remedy is an operational amendment (cap or A0 re-check size), co-signed by BWM if it touches A0.
- **Analysis crash:** before the results set is written, an identical rerun is allowed; a non-statistical crash fix only as `PREREG_LB_AMENDMENT_*.json` (file, old and new sha256, diff, reason, approved_by), logged in DECISIONS.md.
- **Operational amendments** (batching, pauses, lease handling) are recorded before any affected output exists; the verifier applies the chain.

## 8. Wording (rendered mechanically by `lbeam_analysis.wording()`; every branch exercised by the dry run)

The headline, reading and conclusion are selected by the branch (section 5); the result lines, the ratio line and the qualifiers are always rendered. Placeholders in braces are filled from the results. These are the analysis code's own strings, verbatim; the pre-freeze check refuses the freeze if any of them is missing from this file.

**8.1 Headline, reading and conclusion, per branch.**

- MIXED_PASS. Headline: MIXED: T2 ENABLES PLANNING OVER ALL INSTANCES BUT PLANS SIGNIFICANTLY WORSE THAN OFF WHERE LOOKAHEAD IS NEEDED (on its training surface)
  - Reading: enables planning, qualified: significantly worse than OFF where lookahead is needed
  - Conclusion: Over all instances T2 passes the pre-set bar (co-primary 1: 'T2 enables planning'), but where lookahead is needed OFF plans significantly better than T2: the gain over OFF is carried by the instances greedy descent already solves and reverses where lookahead is needed.
- MIXED_SHORT. Headline: MIXED: T2 PLANS BETTER THAN OFF OVER ALL INSTANCES (BELOW THE PRE-SET BAR) BUT SIGNIFICANTLY WORSE WHERE LOOKAHEAD IS NEEDED
  - Reading: partial (mixed)
  - Conclusion: Over all instances T2 plans significantly better than OFF but below the pre-set +0.20, and where lookahead is needed OFF plans significantly better than T2: 'T2 enables planning' is not shown, and the gain reverses where lookahead is needed.
- MIXED_LOOKAHEAD_BETTER. Headline: MIXED: T2 PLANS SIGNIFICANTLY WORSE THAN OFF OVER ALL INSTANCES BUT BETTER WHERE LOOKAHEAD IS NEEDED
  - Reading: partial (mixed)
  - Conclusion: Over all instances OFF plans significantly better than T2, but where lookahead is needed T2 plans better than OFF (co-primary 2: {status2}): 'T2 enables planning' is not shown, and T2's advantage where lookahead is needed is outweighed by its losses on the instances greedy descent already solves.
- REVERSED. Headline: NO (REVERSED): T2 PLANS SIGNIFICANTLY WORSE THAN OFF OVER ALL INSTANCES
  - Reading: no
  - Conclusion: OFF plans significantly better than T2 {where}: the adapter's one-step gain does not turn into goal-reaching plans here, and 'T2 enables planning' is not shown.
- REVERSED_LOOKAHEAD. Headline: NO (REVERSED WHERE LOOKAHEAD IS NEEDED): T2 PLANS SIGNIFICANTLY WORSE THAN OFF WHERE LOOKAHEAD IS NEEDED
  - Reading: no
  - Conclusion: Where lookahead is needed OFF plans significantly better than T2, and over all instances T2's gain is {status1}: 'T2 enables planning' is not shown.
- YES_NEAR. Headline: T2 ENABLES NEAR-ORACLE PLANNING (T2/ORACLE over all instances; on its training surface)
  - Reading: near-oracle planning
  - Conclusion: T2's one-step prediction turns into goal-reaching plans: T2 enables planning, including on the instances greedy descent on the true hook cannot solve, and reaches the near-oracle bar against the exact simulator over all instances (the T2/ORACLE ratio on the needs-lookahead instances is reported separately).
- YES. Headline: T2 ENABLES PLANNING, BELOW THE NEAR-ORACLE BAR (on its training surface)
  - Reading: enables planning
  - Conclusion: T2 enables planning, including on the instances greedy descent on the true hook cannot solve, but it stays below the near-oracle bar (T2/ORACLE >= 0.80 with lower bound >= 0.70).
- GREEDY_ONLY_NEAR. Headline: T2 ENABLES NEAR-ORACLE PLANNING OVERALL (T2/ORACLE over all instances), BUT ITS GAIN OVER OFF IS NOT SHOWN WHERE LOOKAHEAD IS NEEDED (on its training surface)
  - Reading: near-oracle planning, qualified: not shown where lookahead is needed
  - Conclusion: T2 enables planning and reaches the near-oracle bar over all instances (T2/ORACLE over all instances), but on the instances that need lookahead its gain over OFF is {status2}, so the planning gain at the pre-set bar is not shown where lookahead is needed.
- GREEDY_ONLY. Headline: T2 ENABLES PLANNING ON INSTANCES GREEDY DESCENT ALREADY SOLVES; NOT SHOWN WHERE LOOKAHEAD IS NEEDED (on its training surface)
  - Reading: enables planning, qualified: not shown where lookahead is needed
  - Conclusion: T2 enables planning on instances greedy descent already solves; not shown where lookahead is needed (the needs-lookahead contrast is {status2}).
- LOOKAHEAD_ONLY. Headline: PARTIAL: T2 ENABLES PLANNING WHERE LOOKAHEAD IS NEEDED, BUT ITS GAIN OVER ALL INSTANCES IS NOT SHOWN AT THE PRE-SET BAR
  - Reading: partial
  - Conclusion: On the instances that need lookahead T2 passes the pre-set bar, but over all instances its gain is {status1}, so 'T2 enables planning' (co-primary 1) is not shown.
- DIRECTIONAL. Headline: PARTIAL: T2 PLANS BETTER THAN OFF, BELOW THE PRE-SET BAR
  - Reading: partial
  - Conclusion: T2 plans better than OFF ({which} significant at Holm-adjusted p < 0.01), but no co-primary reaches the pre-set +0.20, so 'T2 enables planning' is not shown.
- NO. Headline: NO: T2 IS NOT SHOWN TO PLAN BETTER THAN OFF
  - Reading: no
  - Conclusion: Neither co-primary is significant at Holm-adjusted p < 0.01: T2 is not shown to plan better than OFF.

**8.2 Placeholders.** `{status1}` / `{status2}` are the status sentences of 8.3 for co-primary 1 / 2. `{where}` (REVERSED only) is "over all instances" (status 2 NS) or "over all instances and where lookahead is needed" (status 2 REVERSED). `{which}` names the co-primaries with status SHORT ("co-primary 1", "co-primary 2", or both joined by "and").

**8.3 Status sentences.**

- PASS: passes the pre-set bar (T2 - OFF >= +0.20 and Holm-adjusted p < 0.01)
- SHORT: significant in T2's favour (Holm-adjusted p < 0.01) but below the pre-set +0.20
- NS: not significant (Holm-adjusted p >= 0.01)
- REVERSED: significant in OFF's favour (Holm-adjusted p < 0.01)

**8.4 Result lines (always rendered, in this order).**

- Template: {name}: T2 {k_t2}/{n} ({r_t2}) vs OFF {k_off}/{n} ({r_off}); T2 - OFF {delta} (T2 only {b}, OFF only {c}); exact McNemar p {p}, Holm-adjusted p {p_holm}: {status}.
- Names: Co-primary 1 (all {n} instances) / Co-primary 2 (the {n} needs-lookahead instances). Rates to 3 decimals, delta signed to 3 decimals, p values to 4 decimals (below 0.0001: "<0.0001").
- Key secondary: Key secondary: T2/ORACLE success ratio {ratio} ({ci}); T2 {t2}, ORACLE {orc}; near-oracle bar (>= 0.80, lower bound >= 0.70) {met}.
- `{ci}`: 95% paired bootstrap CI [{lo}, {hi}], {B} resamples, seed {seed} (or: interval not computed); `{met}`: met / not met.
- If ORACLE never succeeds: The T2/ORACLE ratio is not evaluable (ORACLE success 0).

**8.5 Qualifiers (report-only; they never change the branch).**

- When NEAR holds and status 1 is not PASS: T2/ORACLE {ratio} (95% CI [{lo}, {hi}]) meets the near-oracle bar, but 'near-oracle planning' also requires co-primary 1 ('T2 enables planning'), which is not met.
- When NEAR holds, status 1 is PASS and status 2 is REVERSED (branch MIXED_PASS): T2/ORACLE {ratio} (95% CI [{lo}, {hi}]) meets the near-oracle bar over all instances, but 'near-oracle planning' is not read because OFF plans significantly better than T2 where lookahead is needed.
- When the guard refused more than 1% of T2's guarded keys at enum W2 E12: Flag (report-only): the input-range guard refused {share} of T2's distinct guarded keys at enum W2 E12 (> {thr}); T2's own arithmetic errors carried its imagined states beyond |b| <= 98 often enough that the guard, not F, ended part of its search ({n_inst} instances with a refused key). Every truly reachable state is in range, so each refusal also pruned a branch whose imagined state is impossible under the true dynamics, a signal an F-only planner would not have.
- When report-only stages are not recorded: Not evaluable (report-only stages that are not recorded; the branch needs only A0, G1 and C1): {stages}.
  - `{stages}` lists, separated by "; ", each stage name followed by ", " and its reason. Names: G2 (select_fixed T2 and OFF) / G3 (T2 at enum W4 E24) / G4 (the D-35 audit, T2 under heldout_combined). Reasons: abandoned after {n} failed attempts (registered rule) / not run: the run ended by a deterministic stop after A0, G1 and C1 were recorded.
  - In the arm tables a missing stage is shown as: not evaluable.
- Always: Report-only: T2/ORACLE on the {n} needs-lookahead instances {ratio} (95% paired bootstrap CI [{lo}, {hi}], {B} resamples, seed {seed}); T2 {t2}, ORACLE {orc}. It never changes the branch.
  - If ORACLE never succeeds there: Report-only: T2/ORACLE on the needs-lookahead instances is not evaluable (ORACLE success 0 there).
- Always: Report-only: T2 vs IGNORE_STATUS (exact arithmetic without the status rule; enum W2 E12): T2 {a}/{n} vs IGNORE_STATUS {b}/{n} (difference {d}, T2 only {bb}, IGNORE_STATUS only {cc}, exact McNemar p {p}); this bears on whether T2's status-rule competence, not its arithmetic alone, carries its planning.
- When G4 is recorded: D-35 audit (report-only): T2 under the held-out combined rendering (reversed keys, word account names, 'void' rendered 'cancelled') {a}/{n} vs the canonical frame {b}/{n} (difference {d}, 95% paired bootstrap CI [{lo}, {hi}], exact McNemar p {p}): {reading}.
  - `{reading}`, first match: surface-dependent (a significant drop off the training surface) / surface-invariant within 0.05 (the 95% CI lower bound of the difference is >= -0.05) / inconclusive (no significant drop, but a drop larger than 0.05 is not excluded)
- When G3 is recorded: Report-only: enum W4 E24 point, T2 {t2} vs ORACLE {orc} (T2/ORACLE {ratio}).
- Always: Report-only: select_fixed (PREREG-2-style selection among the K = 8 stored candidates) success T2 {t2}, OFF {off}, ORACLE {orc}, IGNORE_STATUS {ign}; IGNORE_STATUS beam (enum W2 E12) {ignb}.

**8.6 Limits that always accompany the result (rendered verbatim from `LIMITS`).**

- Training surface only (trigger-level, PREREG-3): every prompt F answers is Titan's canonical one-step ledger frame (accounts A-D, statuses posted / void, compact sorted-key JSON). PREREG-3 found T2's void no-op to be a token cue and its accuracy to drop off this surface, so the result says nothing about held-out renderings (the SurfaceAuditF heldout_combined arm is report-only).
- Problem distribution: generator v2 in-range instances only. Every state reachable within D_CLOSE = 8 vocabulary steps has |b| <= 98, every amount is one of T2's 21 trained values (-9..12) and every status is posted or void, so scale actions have factor -1 or 0 only; optimal depth d* 2-6; 8 vocabulary actions per instance, including 1-2 void twins. This is a subset of PREREG-2's event distribution, fresh at payload level, not out of distribution.
- Planner and proposer: BWM's plan_beam (bwmn/blocks_planner.py, sha 18a5f22b...) with the model-free enumerating proposer (every vocabulary action as a one-step candidate, 0 proposer tokens), in ONE registered configuration as the primary (enum W2 E12, max_depth 7). The W4 E24 point, select_fixed and IGNORE_STATUS are report-only. Single-plan setting: no execution feedback while planning.
- One adapter: T2 seed 0 (PREREG-2's registered artifact), unmerged on frozen FP32 Titan-Sparse; OFF is the same PeftModel with the adapter disabled. No other seed, base model, proposer or planner configuration is tested.
- F is the first computed answer per (state, action) key, cached per arm, behind the registered input-range guard (|b| <= 98, trained amounts, posted / void), identical for every arm; a refused key ends its candidate. Every truly reachable state is in range, so a refusal also prunes a branch whose imagined state is impossible (dev simulation: about 0.02% of a T2-like F's keys and 0.12% of an OFF-like F's).
- Success is exact goal attainment of the one submitted plan (the verified plan, else the best node's prefix) under env.step. The needs-lookahead stratum is defined by greedy one-step descent on the true unmet hook; it is a property of the instance, not of any arm.
- 'Enables planning' is measured against the unadapted base model (OFF), whose F is close to useless on this task (dev simulation: success 0.12, needs-lookahead 0.04). On its own it does not show that near-oracle prediction gives near-oracle planning; that reading rests on the T2/ORACLE ratio.

**8.7 Truthfulness rules (checked by the dry run on every realised combination).** For each of the 32 (status 1, status 2, NEAR) combinations, realised by synthetic paired outcomes through the real `decide()` and `wording()`, the dry run fails if: a sentence denies co-primary 1 while it passes; the reading is not "enables planning" / "near-oracle planning" exactly when status 1 is PASS; a headline or conclusion says T2 is worse than OFF over all instances (or where lookahead is needed) without the matching REVERSED status, or omits a REVERSED co-primary; a "NO" headline appears while a co-primary is significant in T2's favour; a headline claims "enables planning" (where lookahead is needed) without the matching PASS; "not shown where lookahead is needed" appears while status 2 passes; "near-oracle" is read other than exactly when NEAR, status 1 PASS and status 2 not REVERSED; the near-only or near-withheld qualifier appears other than exactly in its case; a status sentence is missing from its result line or spliced ungrammatically; a placeholder is left unfilled; or the headline, reading or limits differ from the branch's registered strings.

## 9. Roles and co-signing

> **[BWM v1]**
> - RCWM writes and runs `lbeam_run.py` (GPU, lease `rcwm-t1:lbeam`, stage/resume logic as in `p3_run`). RCWM holds the answer key and does the scoring.
> - BWM reviews the outputs and co-signs the report.
> - GPU estimate: about 2.5 GPU-h for 180 instances × (T2, OFF), plus about 0.8 h for the audit surface. If needed, this is split over two RCWM turns at a stage boundary.

**[RCWM amendment] (f) What BWM co-signs, and when.** *Reason: the freeze must bind both sessions to the same text, and the co-sign must be checkable.*
1. **Before the freeze, the text and the salt commitment:** BWM co-signs THIS draft at its stated sha256: the instruments and pins (including its planner sha), N and the seed rule, the decision rules, every wording template, the limits, and the RCWM amendments. The co-sign (BWM's DECISIONS entry or message) quotes this draft's full sha256 and carries `commit:<sha256 of BWM's salt>`; BWM gives the salt to the operator for the freeze only. The freeze refuses a co-sign without either, a salt that does not match, and BWM's draft at any sha256 other than the cited one (`b2896ba3...`); the co-sign is written into `PREREG_LB.md`'s status line and the freeze record, and `--verify` re-checks it. If BWM revises its own draft, this draft is revised to cite the new sha256 before the co-sign.
2. **Before any affected output, amendments:** any amendment touching the decision rules, the wording, the planner call, the instances, the arms or A0 needs BWM's co-sign before it is recorded. Operational amendments (batching, pauses, lease handling, a non-statistical crash fix) are RCWM's, with notice to BWM; every amendment names who approved it.
3. **After the analysis, the report:** BWM reviews the mechanically rendered `PREREG_LB_RESULTS.md` / `.json` and co-signs the report section. A co-sign never changes a branch or a sentence. After the results exist (never before), RCWM gives BWM the episodes, the per-arm caches and the per-instance success arrays for an independent re-score with BWM's own tools.
4. **Throughout:** RCWM runs everything (GPU and CPU stages) and holds the answer key; BWM's code enters only as the read-only, sha-pinned planner. BWM keeps `bwmn/blocks_planner.py` unchanged from the freeze until the registered analysis has run (the shared verifier checks it: a change stops the job and the analysis).

**GPU budget** (section 7): BWM's estimate of about 2.5 + 0.8 GPU-h covered T2 / OFF at W2 E12 with select_fixed and the audit surface. RCWM's computed plan is 5.5-7.4 GPU-h in two turns, because it also prices the 512-key batch-invariance check (512 batch-1 calls, about 0.5 h per turn, run every turn), the T2 W4 E24 point (0.7-1.0 h) and the audit arm at a simulated-OFF call rate (1.1-1.5 h). The primary alone (A0 + G1) fits one turn (2.7-3.5 h). Appendix B, Q1 asks BWM whether to keep all report-only stages.

## 10. Freeze record

(Filled in by `scripts/lbeam_run.py --freeze`: the time, the approval and BWM's co-sign, the draw's seed, its rule inputs (draft sha256, union digest, salt commitment) and its seal, the data/ snapshot, and the SHA-256 of every frozen input.)

## Appendix A. RCWM amendments (each marked in the text, with its reason)

- **(a) Wording for every outcome combination** (sections 5, 8): co-primary 2 gets BWM's co-primary-1 bar (+0.20, Holm p < 0.01); four statuses per co-primary (PASS / SHORT / NS / REVERSED); twelve branches covering every (status 1, status 2, NEAR) combination, including "enables planning on instances greedy descent already solves; not shown where lookahead is needed" (co-primary 1 passes, 2 does not), its reverse (co-primary 2 passes, 1 does not: "partial"), and the mixed-sign outcomes (one co-primary significant for T2, the other for OFF); truthfulness rules checked on all 32 combinations; the registered limits always accompany the result, including that "enables planning" is measured against a near-useless OFF.
- **(b) Deviation and stop policy** (section 7): an A0 failure stops before any planning output, persisted, no verdict; a later re-check failure or other stop is verdict-eligible iff A0, G1 and C1 are recorded and unchanged; report-only stages abandoned after failing twice and core stages stopping after failing three times; the start-of-turn check on the measured A0 and the stop after two consecutive turns without progress; crash, orphan and resume rules; T2 refused-key share > 1% is a report flag (threshold justified by the dev sizing).
- **(c) Power** (section 5b): computed from simulated F on the dev set with the registered decision function, with T2 and OFF paired as simulated and decorrelated, the seed sensitivity, plus an exact independence grid.
- **(d) The fresh confirmation draw** (section 3): 36 per d*; the seed rule with BWM's committed salt (commit-reveal); the freshness union rebuilt at the freeze plus the data/ subdirectories and the spent dev sets; the data/ snapshot and source-error refusals; mint and seal inside the freeze; never inspected before the run; the needs-lookahead stratum defined by the exact greedy procedure.
- **(e) GPU vs CPU arms** (section 4): ORACLE and IGNORE_STATUS (beam and select_fixed) are CPU-only, outside the lease, after A0, at the start of a later invocation and CUDA-free; select_fixed T2 / OFF need the model and run on the GPU.
- **(f) Co-signing** (section 9): BWM co-signs the draft at a sha256 with a salt commitment before the freeze (enforced and verified), amendments of substance, and the report after the analysis.
- **Operational definitions** (sections 2, 5, 6): A0 exactly (anchor and the 256-key-per-arm batch-invariance check, both >= 99%, with the two neutral differences from confirm2); F as the first computed answer across stages (cumulative per-arm caches of recorded stages); BWM module hashes per turn; void-twin exposure (applied as posted, and not a no-op) from the plan's own imagined trajectory; the unstratified bootstrap; the three-way audit reading; the construct contrasts; the instrument and deviation record.
- **Not run here** (section 6): BWM's "beam vs one-shot at matched calls" secondary (no GPU budget; INTERFACE finding 1).

## Appendix B. Open questions for BWM (defaults in force unless changed before the co-sign)

- **Q1. GPU budget and A0's length.** The full plan is 5.5-7.4 GPU-h in two turns, against BWM's 3.3. Default: keep all stages (the primary runs first and alone fits one turn). Alternatives: drop G3 (T2 W4 E24, 0.7-1.0 h) to a later preregistration; run the batch-invariance part of the A0 re-check with 128 keys per arm on resumed turns (saves about 0.2 h per turn and lowers the risk, section 7, that a slow A0 leaves no room for G1; the first A0 keeps 256).
- **Q2. Beam vs one-shot at matched calls.** Default: not in this preregistration (section 6). If BWM wants it here, it needs its own GPU stage (one-shot T2 / OFF with ORACLE-set BudgetF caps) and power.
- **Q3. Binding pins of BWM code.** Default: only `bwmn/blocks_planner.py` is binding (it contains all of `plan_beam`; the hooks replace the Blocksworld defaults); `__init__.py`, `blocks.py` and `lm.py` are recorded at the freeze and per turn, not binding, so that BWM's ongoing work on `lm.py` cannot stop the run (a drift is reported). Alternative: make them binding (then BWM freezes them too).
- **Q4. Co-primary 2's bar.** Default: the same +0.20 / Holm p < 0.01 as co-primary 1. Alternative: a lower delta (for example +0.15) for the smaller stratum (about 64 instances; ORACLE ceiling there about 0.58).
- **Q5. Batch-invariance tolerance.** Default: >= 99% identical parsed states per arm (the anchor's own bar). PREREG-2's anchor reproduced 100% with identical composition; batch 1 vs batched is a different composition, so a few rounding-level flips are possible.
- **Q6. The audit reading.** Default: the three-way reading of section 6 with an equivalence margin of 0.05 (report-only). Alternative: another margin.
- **Q7. N = 180 vs the needs-lookahead power.** Co-primary 2 passes with probability about 0.87-0.92 central and 0.74-0.83 pessimistic. Default: keep N = 180 and no enrichment (BWM). Alternative: N = 240 (48 per d*), about +1.0 GPU-h, raising the stratum to about 86 instances.
- **Q8. The salt procedure.** Default: BWM draws a random salt of at least 16 characters, records only its sha256 in the co-sign (`commit:<64 hex>`), and gives the salt to the RCWM operator when the freeze is run (it is then recorded in the draw's SOURCE record). Alternative: a public, later-determined value instead of a salt (none is available offline).
- **Q9. data/ subdirectories in the union.** Default: included (today `data/t2br_superseded/rev3/`, superseded T2BR training and dev files). Including more files only makes the draw fresher. Alternative: top-level data/ only, as the kit's `load_exclusions` does.
- **Q10. Verdict after a verdict-eligible stop.** Default: the hook runs the registered analysis automatically after a stop that leaves A0, G1 and C1 recorded and unchanged (the report-only stages are "not evaluable"); the operator still investigates and logs the stop. Alternative: require a BWM-co-signed DECISIONS entry before that analysis (this reintroduces a decision after outputs exist, which revision 2 removed).

## Appendix C. How this draft was checked (CPU only; no GPU, no lease, no real model)

- `lbeam_test_run.py --part plan`: the `--plan` path never imports torch; the turn rule; A0 re-checks on later turns.
- `lbeam_test_run.py --part analysis`: the analysis dry run: exact McNemar against the binomial test, Holm, the bootstraps, both thresholds' boundaries, the three-way audit reading; EVERY branch, every status and all 32 (status 1, status 2, NEAR) combinations through the real `decide()` / `wording()` on synthetic paired outcomes, with the truthfulness rules of 8.7; void-twin exposure (both measures, guard-refused keys, distractor label); every sentence of section 8 and every limit appears in this file verbatim; the dry-run record binds this draft's and the hook draft's sha256.
- `lbeam_test_run.py --part power`: section 5b; the runner's sizing constants equal the measured sizing.
- `lbeam_test_run.py --part tiny`: tiny random Qwen3 + LoRA end to end (one lease; A0 anchor and batch invariance; every GPU stage; C1 in the next call, before its lease, CUDA-free); a rule-stub run on the 120 dev instances through the analysis (T2 trigger-level = ORACLE, OFF status-blind = IGNORE_STATUS, the audit arm = IGNORE_STATUS; branch YES_NEAR; the construct contrasts and the instrument record); crash and resume (orphaned stage, A0 re-check, byte-identical outputs); deferral (nothing fits; mid-run at a stage boundary; the stall of an A0 measured at 1.2 h: one model load, then a stop after two turns without progress); anchor failure and batch-invariance failure (stop before any planning output, no verdict); re-check failure (a verdict-eligible stop; the analysis renders G2-G4 "not evaluable"); a report-only stage abandoned after two crashes (DONE.json, "not evaluable", same branch); a core stage failing three times (stop, no verdict); tamper, fingerprint, runtime, lock and one-waiter stops with their exit codes and persistence; `main()` on the real, unfrozen root (exit 3, nothing written); analysis refusals.
- `lbeam_test_run.py --part rehearsal`: the REAL freeze-time union and mint path at full size (the real freshness union with the name-skipping harvest, the data/ subdirectories and the spent dev sets; N = 180) under a DIFFERENT seed tag and a stand-in salt, so it can never produce the confirmation draw; the rehearsal draw is held in memory, never written or printed (only counts are recorded); no union source errs.
- `lbeam_test_run.py --part freeze`: a fake root with the real code, this draft and the hook draft: every freeze refusal of section 7.2; the freeze (draw minted from the rule with the stand-in salt, never printed, re-mintable, another salt gives another seed); verify; an edited co-sign record; interrupted rename; hook identity; a registered run; the registered analysis (create-only, refuses twice, refuses a non-GPU run); tamper of an input, the draw and BWM's planner (a patched copy); an amendment without `approved_by`.
- `lbeam_test_run.py --part hook`: the hook draft under bash with a stub `step()`: every exit-code path (0, 3 eligible or not, 5 then 5 / 0 / 3, crash then crash / 3 / 0), the analysis trigger (DONE.json or a verdict-eligible STOPPED.json), idempotency over two chain passes.

## Appendix D. Review dispositions (revision 1 -> revision 2)

Statistics review:
- MUST-FIX, mixed-sign wording: applied. Branches MIXED_PASS, MIXED_SHORT, MIXED_LOOKAHEAD_BETTER and REVERSED_LOOKAHEAD; the near-only qualifier only when status 1 is not PASS; a near-withheld qualifier for MIXED_PASS; all 32 combinations realised through `decide()` / `wording()` with the truthfulness rules of 8.7.
- SHOULD, construct validity: applied (limit 7; T2 vs IGNORE_STATUS and T2/ORACLE on the needs-lookahead stratum, report-only).
- SHOULD, pooled near-oracle: applied ("T2/ORACLE over all instances" in the near-oracle headlines and conclusions; the needs-lookahead ratio with its CI always rendered).
- SHOULD, audit reading: applied (three-way equivalence reading; "rule-level" dropped).
- SHOULD, decidability: applied (the verdict needs A0, G1, C1; "not evaluable" by rule; implemented and tested).
- SHOULD, draw computable before the freeze: applied (commit-reveal salt; the co-sign must quote the full draft sha256; BWM's draft sha compared at the freeze and in `--verify`).
- NOTE, greedy description: applied (section 3 reworded; discordant pairs by d*).
- NOTE, answer-key leakage: documented (ids encode d*; no registered component reads them; the refused-key flag sentence names the pruning signal). Declined: opaque ids, because that needs a change to `lbeam_problems.py` that is not strictly required (no component reads the ids) and would invalidate the kit records of D-37.
- NOTE, power statements: applied (">= 0.978"; the shared-hash pairing artifact; decorrelated figures; per-seed ranges).
- NOTE, void-twin exposure: applied (the not-a-no-op measure; the select_fixed label; guard-refused keys counted).
- NOTE, unstratified bootstrap: stated (conservative). Declined: stratifying it by d*, to keep BWM's "resamples over instances" verbatim.

Engineering review:
- SHOULD, deferral stall: applied (measured A0 in the start check; stop after two consecutive turns without progress; tested).
- SHOULD, union robustness: applied (data/ snapshot before and after; refusal on a source error; data/ subdirectories included, Q9).
- SHOULD, draw computable before the freeze: applied (salt).
- SHOULD, pre-freeze records not bound to the draft: applied (`LB_DRYRUN.json`, `RUN_TEST_ANALYSIS.json`, `RUN_TEST_HOOK.json` record the draft and hook-draft sha256 and the freeze checks them; the verbatim check also runs inside the freeze).
- SHOULD, co-sign not checkable: applied (draft sha256 and salt commitment enforced; BWM draft sha compared; `--approval` must cite a `D-<n>` entry).
- SHOULD, report-only stages block the verdict: applied (with the runner's abandon rule and verdict-eligible stops).
- NOTE, BWM modules per turn: applied (recorded per turn and CPU run; drift reported).
- NOTE, C1 in a process with a CUDA context: applied (C1 only at the start of a later invocation, checked CUDA-free; the analysis refuses otherwise).
- NOTE, draft wording on persisted verification failures: applied (section 7.4).
- NOTE, results without the instrument record: applied (section 6).
- NOTE, robustness gaps: applied (inputs checked before minting; only problem counts printed; `cap_h` per turn; `approved_by` required; the draft read once).
- NOTE, test coverage: applied (`run_main` exit codes and persistence, `main()` on the real root, a BWM planner tamper on a patched copy, the stall, the hook simulation as a registered test part).
- NOTE, A0 differences from confirm2: stated (section 2).
