# Solutions and benchmark plan (2026-10-07)

**Sources:**
- A four-question literature sweep by research agents (web search plus primary sources).
- Adversarial fact-checks of 56 numeric claims: 50 confirmed, 6 corrected. The corrections are listed at the end.
- Raw reports: `research/_raw_research_2026-10-07.json`.

The synthesis below is mine. It is reconciled with what was run today; see DECISIONS.md F-04, F-05 and D-41 to D-44.

## 1. Executive summary

1. **Rule-extrapolation shortcut (F-01).**
   - This is textbook underspecification: two hypotheses fit training equally well. The literature includes Geirhos
     et al. 2020 (shortcut learning), D'Amour et al. 2020 (underspecification) and Hermann et al. ICLR 2024.
   - It sits close to the incoherent-world-model results of Vafa et al. 2024/2025.
   - What our case adds is a clean split inside one model: arithmetic extrapolates (99.5%) while the parameterized
     legality rule does not (36.5%), and the training data provably forces this.
2. **The repair (F-03) is selective prediction / learning to defer.**
   - Related work: Chow's reject option; confidence cascades such as FrugalGPT
     (https://arxiv.org/abs/2305.05176) and Jitkrittum et al. 2023; robustness recovered from the pretrained model,
     e.g. WiSE-FT.
   - The sweep found no prior instance of gating a LoRA world model's own backbone, with the adapter off, as its
     verifier, but the novelty is empirical.
   - Our ladder (F-04) shows what matters:
     - Confidence-based deferral works. Random deferral at the same call rate hurts (0.847 < 0.886).
     - A standard symmetric rule ties the asymmetric band overall (0.979 vs 0.979).
     - The verifier is right on all 26 confident errors, so the remaining errors come from gating, not the verifier.
3. **Proposer coverage (P-03).**
   - Coverage rises roughly log-linearly with samples (Large Language Monkeys, https://arxiv.org/html/2407.21787).
   - Step-level search guided by a prior beats i.i.d. best-of-N at small budgets: Snell et al. 2024 report > 4x;
     the Hugging Face test-time compute blog reports beam N=4 ≈ best-of-16.
   - Our K=12 read on 50 tasks agrees: A01 rises 0.66 → 0.82 (8 wins, 0 losses).
   - Expert iteration and search distillation reliably amortize search (ExIt, ReST-EM, Stream of Search,
     Searchformer, Dualformer), but they tend to narrow pass@k. A K-candidate system needs diversity-preserving
     targets.
4. **The PlanBench 600 set is not where a "rivals larger models" claim can come from.**
   - The only published fine-tuned entry on the identical 600 IDs is GPT-3 davinci at 122/600, with no search
     (Valmeekam et al., NeurIPS 2023, https://arxiv.org/abs/2305.15771).
   - Plain SFT on generated small Blocksworld saturates in distribution: Plansformer (~220M) 90%, Qwen2-7B 98.5%,
     Gemma2-9B LoRA ~99-100%.
   - So a high 600 score says nothing about the world model unless it is paired against our own no-F arm.
5. **The credible headline axis is size generalization: PlanBench Blocksworld-hard.**
   - The set: 110 instances, 6-15 blocks, 20-40-step plans.
   - Zero-shot scores: o1-preview 23.6% (26/110), o1-mini 10% (TMLR 2025, https://arxiv.org/abs/2410.02162).
     DeepSeek-R1 53.6% (PlanBench leaderboard, https://github.com/karthikv792/LLMs-Planning).
   - SFT planners trained on small problems collapse there. Examples: Gemma2-9B LoRA trained on 3-5 blocks falls to
     53% at 6 blocks and 0% at 10; Qwen2-7B SFT gets 0% on obfuscated names.
   - Our world model, trained on 4-6 blocks, is 1.000 exact at 8 and 10 blocks and 0.967 at 12 and 15 (F-05). This is
     the regime where a learned world model plus search can do what plain SFT cannot.

## 2. Benchmark plan (pre-registered: PREREG_PLANBENCH.md, addendum B2)

- **Systems.** One frozen configuration of each:
  - S-beam: Pa_v2 + BW_F, proposer-guided beam search in imagination.
  - S-direct: the same proposer, one greedy plan, no F (the internal SFT arm the literature says we must have).
  - S-replan.
- **Selection.** Settings are chosen only on generated sets: val_v2 (3-6 blocks) and val_large (6-15 blocks,
  hard-style goals, hard instances excluded).
- **Test.** The 600 set and the 110 hard instances, each run once and VAL-scored. Statistics:
  - paired McNemar of S-beam vs S-direct;
  - Clopper-Pearson CIs;
  - block-count strata;
  - compute accounting: proposer tokens, unique F queries, FLOPs via `bwmn/flops.py`.
- **Controls.**
  - Hard-mirror set (110 generated, same block counts). A gap above 10 pp is flagged.
  - Renaming exclusion: exact signatures for ≤7 blocks, plus an invariant filter.
  - Still to do: randomized object names, and the Mystery Regime B (rules given in the prompt, no code mapping).
- **Thresholds that would mean something.** On hard:
  - above 26/110 exceeds o1-preview zero-shot;
  - above ~59/110 exceeds R1;
  - either result only with a significant S-beam > S-direct gain, and stated with the regime caveat (domain-trained
    specialist with a learned simulator vs. zero-shot generalists).
  - VAL-in-the-loop numbers (o1-preview + VAL loop 98.2%) are a different regime and are never compared.
- **Same-harness open comparator.** Qwen3-4B, zero-shot and thinking at fixed budgets, on a stratified subset and
  on hard. It is still pending because `scripts/planbench_qwen.py` must-fix items are only partly done (see
  TODO_BEFORE_RUNS.md), and it needs ~15-30 CPU-h.

## 3. Ranked experiments

| # | Experiment | Status | Cost here | What would be substantial |
|---|---|---|---|---|
| 1 | BW_F size generalization (state format) | **done (F-05)** | 22 CPU-min | — |
| 2 | Delta-format F (predict only the changed facts) | data built; training in bundleA | ~1.3 GPU-h | ≥0.995 exact at 12-15 blocks |
| 3 | Pa_v2 (hard-style goals, ≤6 blocks) | training in bundleA | ~3 GPU-h | — |
| 4 | Planner selection on val_v2 + val_large (`bw_select.py`) | ready | ~2-4 GPU-h | S-beam ≫ S-direct on val_large |
| 5 | Frozen runs on the 600 set, hard-110 and the mirror set | gated on 4 | ~3-6 GPU-h | hard > 59/110 with paired gain over S-direct |
| 6 | RW2 coverage: A01 at K=12 (done for 50 tasks), then policy-guided search with Gumbel-top-k distinct first actions | 2nd slice running | CPU | success vs compute above BFS-under-F at matched LM calls |
| 7 | F-03 follow-ups: a below-range calibration limit for a two-sided gate; an input-novelty signal | not started | CPU | repair of confident below-range errors |
| 8 | RW2 four-arm pilot on GPU (T effect) | pilotG queued | ~3 GPU-h | — |

## 4. Corrections from the fact-check (claims as first reported vs. sources)

- **Gemma2-9B LoRA.** The 53% → 0% curve by block count belongs to the "Random walk++" variant, not plain random walk.
- **DeepSeek-R1 hard-set number.** It is also in the TMLR 2025 paper, not only the README leaderboard.
- **Plansformer.** It was fine-tuned on 14,400 plans (80% of 18,000), with the remaining 3,600 used as the test set.
- **PlanGPT.** The number of IPC Blocksworld problems is not stated in the paper.
- **Qwen2-7B.** The 41.5% after RL is an 8-domain average, not a Blocksworld figure.
- **Llama-3-8B.** The 99.3% at 4,000 samples is on 4-block, not 5-block, tasks.
- **Open discrepancy.** Our VAL re-score of LLaMA-3.1-405B is 386/600, against 376 in the paper. Still to be resolved
  before any paired claim against it.
