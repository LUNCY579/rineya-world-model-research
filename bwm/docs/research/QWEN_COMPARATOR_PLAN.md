# Same-harness Qwen3-4B comparator on Blocksworld-hard (and PB-600): run plan (D-75.4)

Status 2026-10-11: **ready, not run.** No model was loaded to prepare this plan. Only `--selftest`, `--dry-run`,
`--iso-ledger` and the tests were run. This is a plan, not a frozen preregistration. Before the first Qwen output on
any set listed here, BWM writes and hash-freezes `PREREG_LOCAL_COMPARATORS.md` from §§2-5, after the throughput probe
(§6) and with the open decisions in §9 settled. Source: panel `research/NEXT_ANALYSIS_PANEL_2026-10-11.json`
(props[2].proposals[1]; final.top_now_cpu item 9a); `PREREG_PLANBENCH.md` Comparators 1 and allowed headline (b).

## 1. What is compared, and how

- **BWM side (fixed, already run once):**
  - H2 = frozen B2 S-beam with Pa_v3 (P-14, 88/110);
  - B2 S-beam (P-08, 73/110);
  - B2 S-direct (5/110) and H1 (13/110);
  - per-instance results from `runs/BW_B3_HARD/*` and `runs/BW_TEST_B2/*`, never re-run.
- **Comparator:** Qwen3-4B, HF snapshot `1cfa9a7208912126459214e8b04321603b3df60c`, BF16, N_nonemb 3,633,511,936
  (config formula = safetensors headers). Same machine, CPU only, <= 4 threads.
- **Instances:** PlanBench Blocksworld-hard, the 110 local problem files `bench/planbench/blocksworld_hard/generated/`.
  Ids are `hard/instance-N`, as in `scripts/blocks_eval.py`, so pairing with H2/B2 is by id.
- **Prompt:** the exact zero-shot PDDL query that DeepSeek-R1, o1-preview and o1-mini were sent, read from the
  sha-pinned upstream per-instance files:
  - `llm_planning_analysis/results_backprompting/blocksworld_hard` at fc638a1a;
  - vendored copy `bench/planbench/results/blocksworld_hard/<model>/task_1_..._tg1.0.json`, byte-identical to the D-74
    audit copy;
  - sha256: R1 `332b0f8e…`, o1-preview `b8fdc9ba…`, o1-mini `92849790…`.

  The loader refuses to run unless all three files' step-0 user messages are byte-identical to upstream's query
  builder (`hard_query`, a port of `per_instance_output_pddl`) applied to the local problem file. Verified for 110/110
  instances. The prompt is rendered with the Qwen3 chat template as a single user turn. **No system message:**
  upstream's `send_query_with_feedback` comments the system message out for o1-*/deepseek-r1, and every hard file
  records one user turn. The panel's text said "with PlanBench's system message"; see decision D1 in §9. Prompt
  length is 541-636 tokens (median 601) with thinking on, +4 with thinking off.
- **Scoring:** every plan is checked by the same local VAL binary and the same problem/domain files as BWM's runs.
  - **Primary:** `extract_pddl_plan` (rule-based; s-expressions or bare actions; plan region = the later of the last
    code fence and the last plan section; verification sections excluded; wrong-arity actions dropped and counted),
    then VAL.
  - **Second scorer:** upstream's own `save_gpt3_response` (lines starting with '('), with VAL run on that exact text.
  - Both scorers are applied to the text after `</think>` only.
  - Validation on the 330 published step-0 responses (`--selftest`):
    - the second scorer reproduces every published `llm_plan` (330/330) and every published VAL verdict (330/330:
      R1 59, o1-preview 26, o1-mini 1);
    - the primary scorer agrees with the published verdict on 330/330 (same action sequence 327/330);
    - the simulator agrees with VAL 330/330.

  Scorer disagreements on Qwen outputs are reported, never resolved by hand.

## 2. Arms

| Arm | Decoding | Budget per instance | Instances | Role |
|---|---|---|---|---|
| **Q4N** | thinking off (`enable_thinking=False`), greedy | max-new 1024 | all 110 | report-only, paired vs B2/H2 |
| **Q4T-iso** | thinking on; T 0.6, top-p 0.95, top-k 20; fixed per-instance seed; budget forcing | FLOP cap 5.2988e13 = H2's mean (§3); total <= T_i = 5,521-5,527 tokens; thinking budget 3,864-3,953 (median 3,900); answer reserve 1024 | systematic 40 (§4) | **primary** |
| Q4T x0.5 | as Q4T-iso | cap 2.6494e13; total ~3,017; thinking ~1,392 | same 40 | descriptive budget curve |
| Q4T x2 | as Q4T-iso | cap 1.0598e14; total ~9,747; thinking ~8,122 | same 40 | descriptive; close to R1/o1-preview's median 7.8k completion tokens on hard |

The panel's "iso/4" point is replaced by x0.5. At x0.25 the cap is 1.32e13 and T = 1,609 tokens. That leaves no
thinking budget after a 601-token prompt and the 1024-token answer reserve, and only ~480 tokens with a 512 reserve,
which Q4N already covers. The x0.5 point is at the panel's intended iso/4 token level (~2.7k). The optional Qwen3-8B
point (panel) is not prepared; the 16 GB snapshot needs >= 20 GB free.

**Commands.** Defaults are `--threads 4`, `--seed 0`, and `--system-prompt auto`, which means none on hard.

```
python -B scripts/planbench_qwen.py --model qwen3-4b --set hard --thinking off                         # Q4N
python -B scripts/planbench_qwen.py --model qwen3-4b --set hard --thinking on --iso H2_hard --subset 40  # Q4T-iso
python -B scripts/planbench_qwen.py --model qwen3-4b --set hard --thinking on --iso H2_hard --iso-fraction 0.5 --subset 40
python -B scripts/planbench_qwen.py --model qwen3-4b --set hard --thinking on --iso H2_hard --iso-fraction 2 --subset 40
```

Outputs go to `runs/planbench_qwen/qwen3-4b_hard[_sub40]_<budget>_max1024_s0.jsonl`, one fsynced row per instance.
Each row carries:
- the raw text;
- both scorers' plans and VAL verdicts, plus the simulator verdict;
- token accounting: prompt, think, forced, answer, answer cap, `hit_eos`, forced think end;
- FLOPs by term and `within_flop_cap`;
- `prompt_render_match`, i.e. `apply_chat_template` equals the pure renderer used for the dry-run budgets (checked
  20/20 against transformers' tokenizer, ids identical);
- the full config.

Adding `--dry-run` to any command resolves everything without loading a model or importing torch/transformers.

## 3. The iso-FLOP budget (registered in `scripts/planbench_qwen.py` `ISO_REGISTRY["H2_hard"]`)

`python -B scripts/planbench_qwen.py --iso-ledger` recomputes it from H2's frozen artifacts only (`beam_ledger`; no
model; reproduced, rc 0). It uses the same `bwmn/flops.py` formula as the comparator:
- dense 2·N per processed token;
- causal attention 4·L·H·d_head per (query, key) pair, prefill and decode;
- LM head 2·d·V per generated token;
- active LoRA parameters (10,092,544 each for F and Pa) added to Titan's N_nonemb of 440,467,456.

| H2, per instance (n = 110, 88 solved) | upper (registered) | lower |
|---|---:|---:|
| F: 22,910 unique (state, action) keys summed over shards = 208.3/instance (22,895 across shards = F-09); prompt 173.4 + output 23.6 tokens | 3.944e13 | 2.120e13 |
| proposer: 8,206.1 generated tokens/instance (K = 8 samples per prefilled prompt), 13.3-17.2 expansions | 1.355e13 | 1.280e13 |
| **total** | **5.2988e13** (52,988,071,221,360) | 3.3998e13 |

- **Upper (registered):**
  - every F query is counted as a full sequence, with no credit for the 96-token system-prompt KV cache that
    `greedy_shared_prefix` reuses;
  - expansions are counted at their maximum given the logged rounds (`min(1 + W(rounds-1), E-1+W)`);
  - proposer prompts are counted at the longest state on the submitted plan's trajectory;
  - within-sample attention uses the convexity bound.

  It credits Qwen with at least H2's compute, which is conservative for any "higher at matched compute" reading.
- **Lower:** prefix-shared F, the minimum expansions given the logged tokens, and the shortest prompts. Report-only.
- **Input sha256s (registered):**
  - `plans.jsonl` 3cf3cd01…;
  - `FREEZE_B3_H2.json` a433e499…;
  - Titan `tokenizer.json` aeb13307…;
  - eight `fcache.jsonl` (c7b9dfb6…, 28c5e6ee…, 13ccb217…, f1f2bf09…, 52b312b8…, 39c361ee…, 5b5418bd…, 2c354964…).

  `--iso-ledger` fails (rc 1) if any input sha or the value differs.
- **FR-02-type note:** `summary.json`'s 198 F queries/instance counts only final-attempt cache misses. The fcache gives
  208.3. The same ledger reproduces B2's E-01(b) figure of 186.9 keys/instance and F-09's 20,530 unique keys.
- **Conversion for Qwen3-4B, per instance:**
  - T_i = the largest total length (prompt + thinking + forced + answer) whose cost, with every non-prompt token
    counted as generated, is <= the cap;
  - thinking budget = T_i − prompt − 3 forced tokens (`"\n</think>\n\n"`) − 1024;
  - after the thinking phase the answer may use the rest of T_i, so every completion is <= the cap by construction;
  - a KV re-prefill fallback, if any, is counted and flagged `within_flop_cap = false`.
  - At T ≈ 5.5k, attention is 17% of Qwen's cost (dense 4.02e13, attention 9.0e12, LM head 3.8e12).
- **Why not the panel's 0.9e14 / 10.8k tokens:**
  - the panel's scratch figure assumed ~3.2e11 per F query, but the measured cost is 1.9e11 (prompt 173 + output 24
    tokens);
  - the panel's figure apparently omitted Qwen's attention term;
  - the x2 point (9.7k tokens) is where the panel's figure lands.
- **Report-only ledgers:**
  - B2 hard: 4.846e13 upper / 3.115e13 lower;
  - B2 PB-600: 9.48e11 upper. This is below the cost of Qwen3-4B merely reading a 300-token prompt (2.2e12), so an
    iso-FLOP arm is impossible on PB-600, and PB-600 gets Q4N only (§7).
- **Scope of the budget:**
  - inference only: training FLOPs of the adapters and of Qwen are not counted, and the prereg states this;
  - no iso-FLOP statement against API models (R1's FLOPs are only estimated; o1's are unknown).

## 4. Instances for Q4T: systematic 40

- **Rule:** `systematic_subset(load_hard_instances(), 40)`. Sort by (blocks, optimal length from `info.json`,
  instance id) and take sorted positions floor((2k+1)·110/80), k = 0..39.
- **Determinism and independence:** deterministic, no seed, independent of any result.
- **Registered ids sha256:** `f952f532da5cba077729abea3894ca04c8b9ed9ee4999e2040638881b46ff237`
  (`HARD_SUBSET_40_SHA`; ids one per line).
- **Ids:** 2, 6, 12, 14, 20, 21, 23, 25, 29, 34, 37, 38, 43, 44, 47, 51, 53, 54, 61, 65, 68, 71, 72, 73, 75, 78, 79,
  80, 81, 84, 85, 87, 94, 100, 102, 103, 104, 107, 109, 110 (each `hard/instance-N`).
- **Blocks (drawn/size):** 6: 0/1, 7: 1/1, 8: 2/7, 9: 5/14, 10: 5/13, 11: 4/12, 12: 5/13, 13: 7/18, 14: 7/21, 15: 4/10.
- **Optimal length bins (E-01(c) bins):** 20-24: 13, 25-29: 6, 30-34: 10, 35-40: 11.
- **Fixed results on these 40** (for power only; not used to choose anything):
  - H2 33/40, B2 26/40, B2 S-direct 3/40, H1 4/40;
  - published step 0 (post hoc, different regime): R1 18/40, o1-preview 8/40, o1-mini 1/40.

## 5. Statistics (to be frozen in PREREG_LOCAL_COMPARATORS.md)

- **Primary:** H2 vs Q4T-iso on the 40, exact McNemar on the discordant pairs (two-sided). Readings:
  - "Rineya system higher at matched compute" iff b > c and p < 0.05;
  - "non-inferior" iff the Newcombe paired 95% lower bound of H2 − Q4T-iso > −0.05 (PREREG_PLANBENCH headline (b),
    delta 5 pp);
  - otherwise "not shown".
- **Scorer rule:** the primary scorer decides. The second scorer's success count and the disagreement count are
  reported, and the readings are also stated under the second scorer.
- **Report-only:**
  - B2 vs Q4T-iso;
  - Q4N vs B2 and H2 on all 110 (paired);
  - Q4T and Q4N vs the R1/o1-preview step-0 files, labelled POST HOC and different regime, never "beats";
  - results by block count and optimal-length bin;
  - FLOPs (measured, per term) and wall time;
  - forced-think-end (truncation) rate and answers cut at the cap;
  - the x0.5 / x1 / x2 budget curve, if run.
- **Claim template, if the primary reading is "higher":** "At matched inference FLOPs (Qwen given at least H2's
  measured per-instance compute), on identical PlanBench Blocksworld-hard instances and the same validator, a frozen
  0.6B Rineya world model with search solves X/40 vs Qwen3-4B-thinking Y/40 (paired, p = …)."
- **Mandatory caveats:**
  - domain-trained specialist vs zero-shot generalist;
  - BWM reads code-parsed JSON while Qwen reads the PDDL text;
  - one seed per system;
  - Qwen's pretraining may contain PlanBench hard, which inflates Qwen and is conservative for BWM;
  - n = 40 detects only large differences.

## 6. Compute, RAM, threads

- **Throughput probe first** (the panel's "measured on 3 PB-600 instances before n is fixed"):
  `--set bw3 --limit 3 --thinking off` and `--set bw3 --limit 3 --thinking on --think-budget 2048`, each about 0.2-0.5
  CPU-h. These rows are PB-600 report-only material and decide nothing about hard except n: if the measured decode
  rate puts Q4T-iso above ~30 CPU-h, the prereg fixes a smaller n (same systematic rule) before the first hard
  output. n never changes after that.
- **Generated tokens, worst case** (every completion uses its full budget), at an assumed decode rate. Prefill of
  ~600 tokens/instance and the context-length slowdown come on top:

| Arm | Generated tokens (worst case) | CPU-h at 6 / 4 / 3 tok/s |
|---|---:|---|
| Q4N, 110 x <= 1024 | 112,640 | 5.2 / 7.8 / 10.4 |
| Q4T-iso, 40 x <= 4,924 | 196,960 | 9.1 / 13.7 / 18.2 |
| Q4T x0.5, 40 x <= 2,416 | 96,640 | 4.5 / 6.7 / 8.9 |
| Q4T x2, 40 x <= 9,146 | 365,840 | 16.9 / 25.4 / 33.9 |

  - Registered core (Q4N + Q4T-iso): 14-29 CPU-h.
  - With both descriptive points: 36-71 CPU-h, over 3-6 days at the ~50% duty the lane rules allow.
- **RAM:**
  - weights 8.04 GB (BF16 safetensors);
  - KV cache 147,456 B/token, i.e. 0.78 GB at 5.5k tokens and 1.37 GB at 9.7k;
  - resident estimate 10-11 GB;
  - gates: >= 12,000 MB available before any model load (12 GB floor), >= 4,000 MB before every instance,
    >= 2,000 MB re-checked every 256 generated tokens. On a failure the instance is abandoned unwritten and the run
    stops; it is resumable.
- **Threads:** `--threads` <= 4 (enforced), with OMP/MKL/OpenBLAS pinned to the same value and inter-op threads set to 1.

## 7. Scheduling, lane and gates (all enforced by the script)

- **CPU lane (D-76, RCX `rcx/lane.py` protocol, verified against a copy of RCX's code in the tests):**
  - **Lock:** `%TEMP%/rineya-cpu-lane.json` {holder 'bwm', job, pid, started, expected_end, token}, created with
    O_CREAT|O_EXCL before the model loads and deleted at exit (token-checked).
  - **Stale lock:** a dead holder pid (psutil.pid_exists / OpenProcess, never `os.kill(pid, 0)` on Windows; a pid
    created after the lock = reused) moves the lock to `rineya-cpu-lane.json.stale-<pid>.bak`, and the lane is
    reclaimed.
  - **No acquisition** while RCX's lock or `rineya-cpu-lane.waiting.rcx` exists.
  - **Waiting:** `rineya-cpu-lane.waiting.bwm` is written while waiting.
  - **Yield:** when `waiting.rcx` appears, the run yields at the next instance boundary: the lane is released and
    the model freed. It then waits and resumes, or exits 75 with `--lane-no-wait`.
  - **Lock lost:** a lost lock stops the run.
- **Priority (D-76):** RCX keeps the lane during its E01R training blocks; the comparator gets the gaps and overnight
  when no RCX block is queued.
- **Only while no BWM model is loaded (D-75.4):**
  - qwen3-4b refuses to start, and stops before any instance or within 256 tokens, if another native_v1 python job
    runs;
  - the always-on `health_log.py`, `orchestrate.py` and `gpu_run.py` wrappers are excepted, so BWM's GPU job (gpu_run's
    child) is what blocks;
  - in practice the comparator runs during RCWM's GPU turns.
- **Other gates:**
  - D-35: `bwmn.lm` is imported before transformers, CUDA is patched off, and the run refuses otherwise;
  - CPU thermal guard between instances;
  - one planbench_qwen run at a time (`runs/planbench_qwen/run.lock`).
- **Resumable:** rows carry the full config (code shas, threads, snapshot and weight-header sha, library versions,
  prompt/template/query shas, upstream file shas, subset sha, budget rule).
  - A file with another config is refused.
  - A torn last line is truncated.
  - The first instance of a pass aborts if the KV-cache continuation had to fall back.
- **Order:**
  1. freeze the prereg;
  2. throughput probe;
  3. Q4N on 110;
  4. Q4T-iso on 40;
  5. x0.5, then x2 if CPU time allows;
  6. analysis.
- **PB-600:** ready for Q4N (`--set all|bw3|bw`, NL prompts with PlanBench's chat system message, three scorers;
  scorer selftest 600/600 as before), report-only. No iso arm (§3).

## 8. Not prepared in this track

- **Contamination control on item 7's seeds_fresh-40** (panel): it needs a renderer of generated instances in
  upstream's problem-file format, and item 7's R0/R1/R2 results to pair with. Follow-up.
- **The optional Qwen3-8B point.**
- **Published-file vendoring/SOURCE.md and the post hoc paired McNemar vs R1/o1:** these belong to item 4
  (`scripts/hard_published_paired.py`). This harness only reads the pinned files.

## 9. Open decisions (main session / user)

- **D1 system message on hard:** none, as recorded for R1/o1 (default `auto`), or PlanBench's chat system message
  (`--system-prompt planbench`), as the panel's text suggested. Upstream sends that message to non-reasoning chat
  engines only.
- **D2 budget estimate:** register the upper 5.2988e13 (conservative, as implemented) or the lower 3.3998e13.
- **D3 n for Q4T** after the throughput probe: 40 as registered, or smaller under the same rule.
- **D4** whether to spend ~17-34 CPU-h on the x2 point.

## 10. Verification and files

- `python -B scripts/planbench_qwen.py --selftest`: rc 0, 29.5 s.
- `python -B scripts/planbench_qwen.py --iso-ledger`: REPRODUCED, rc 0.
- `python -B -m pytest tests/test_planbench_qwen.py -q -p no:cacheprovider`: 78 passed (75 + 3 verifier regression cases for the bold-title extractor fix).
- `--dry-run` on hard Q4N/Q4T-iso/x2 and PB-600: no torch/transformers import, no files written, lane untouched.
- Code: `scripts/planbench_qwen.py`, `bwmn/planbench_extract.py`, `bwmn/flops.py`; tests: `tests/test_planbench_qwen.py`.
  Their sha256s are recorded in each output row's config (`code_sha`).
