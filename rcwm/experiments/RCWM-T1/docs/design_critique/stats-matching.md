# RCWM-T1 design review: matched scalar baseline and statistical design

I checked the bank on CPU with `data/bank_pilot.json` (200 parents per family). The pilot test split has L=26, P=28 and R=26 parents. There is no readout, head or analysis code in `rcwmt/` yet, so B2, B3, S, the seeds and the grids exist only as text in DESIGN.md.

---

### 1. BLOCKER: B2 gets at least as much information as B3, so H3 compares inductive bias, not representation

- I drew 16 Gaussian train goals over φ for each family. The goal matrix has rank 12 for L, 9 for P and 9 for R, with σ_min of 0.71, 1.54 and 1.08.
- So B2's 16 utilities per candidate are U = Wφ, an injective linear map. φ = W⁺U can be recovered exactly. B2 gets 16 scalars per candidate and B3 gets 9–12, with identical information.
- **Failure scenario:**
  - A B3 win over the concat-MLP B2 is predictable. With [h; w_g], 16 distinct goal inputs and a ReLU MLP, the network cannot learn the w×h interaction. The goal is also 12 inputs out of 1036, next to unstandardised residual dims that include Qwen massive-activation outliers.
  - A bilinear B2, U = (M w)·g(h) with r = |φ|, is algebraically B3. Its MSE equals ‖W(φ̂−φ)‖² with WᵀW ≈ 16·I, so it is B3 with plain MSE instead of BCE/Huber. Against it a null is predictable.
  - Either result tells you about loss and architecture, not about consequence representations.
- **Fix:**
  - Use a fixed B2 ladder, all with the same trunk, both readout layers available, and h z-scored on train:
    ```python
    class B2(nn.Module):  # kind in {"concat","film","uvfa","bilinear"}
        def __init__(s, kind, d_phi, r=32, hid=256):
            s.g = nn.Sequential(nn.Linear(1024, hid), nn.GELU(), nn.Linear(hid, r))
            s.e = nn.Linear(d_phi, r, bias=False) if kind == "bilinear" else \
                  nn.Sequential(nn.Linear(d_phi, 64), nn.GELU(), nn.Linear(64, r))  # uvfa
            s.b = nn.Linear(d_phi, 1)
        def forward(s, h, w): return (s.g(h) * s.e(w)).sum(-1) + s.b(w).squeeze(-1)
    ```
    - The primary comparator B2★ is chosen on DEV from {concat, FiLM, UVFA, bilinear r∈{|φ|,32}} × n_train_goals ∈ {|φ|, 16, 64}. Goals are free once φ is known.
    - Standardise U per goal on train.
    - Offer a within-parent listwise softmax loss as a grid option to both B2 and B3.
  - Add a matched linear pair: ridge φ̂ = Ah (B3-lin) versus ridge on w⊗h (B2-lin).
  - Account for labels per labelled train candidate:

    | Arm | Scalars | Information |
    |---|---|---|
    | B3 | \|φ\| | φ |
    | B2@16 | 16 | φ |
    | B2@\|φ\| (primary matched) | \|φ\| | φ |
    | B2@r<\|φ\| | r | Pφ (a projection) |

  - Pre-state the interpretation. B3 beating B2★ means "feature-level supervision and the w-linear prior". B2-concat losing to B2-bilinear means "architecture".

### 2. BLOCKER: every held-out goal is in-span, so H3 does not test goal transfer

- Because the 16 train goals span R^|φ|, every test goal satisfies w_test = Σc_i w_i, and its out-of-span energy is exactly 0. "Disjoint" means nothing for continuous draws.
- **Failure scenario:** a B3 win on "held-out goals" is just in-span interpolation, which is the weaker claim. The parent plan's H2 is explicitly about in-span goals, while the transfer claim needs an out-of-span test.
- **Fix:** add two conditions.
  - **(a) Restricted span.** Draw the train goals inside a random r-dimensional subspace, r ∈ {2, 4}, and draw test goals from the full space. Report regret against ‖P⊥w‖²/‖w‖². The expected out-of-span fraction is 0.56–0.78. Say plainly that B3 has more information here; this is the RCWM claim itself.
  - **(b) Out-of-φ goals.** Use goals on features outside φ:
    - P: n_applied, bit_i∧bit_j
    - L: final>50, n_posted
    - R: final mod 3, X>Y

    Then run few-shot readouts with n_new ∈ {16, 64, 256} labelled train candidates, comparing ridge on φ̂, on B3's hidden layer, on B2★'s penultimate layer, and on raw h.

### 3. MAJOR: the pilot is underpowered and 16 test goals add uncounted variance

- I simulated noisy φ̂ predictors with 256 test goals: two similar methods (ε 0.15 vs 0.18–0.20, error correlation ρ 0.7–0.8). The SD of the per-parent paired regret difference was L 0.07–0.09, P 0.13–0.16 and R 0.10–0.12. With 16 test goals the per-parent SD rises by roughly 1.3–1.7×.
- Inflating the SD by 1.5× for head-training noise gives a pilot minimum detectable effect (80% power, α=.05) of L 0.074, R 0.099 and P 0.12 in normalised regret. That is only enough for very large effects.
- With 16 test goals, the goal-sampling SE is about equal to the parent SE for P (0.020 vs 0.021 at 200 parents). A parent-only bootstrap is anti-conservative.
- **Fix:**
  - Use 256 test goals (no GPU cost) with a two-way bootstrap, resampling parents within family together with goals:
    `D[np.ix_(ip, ig)].mean()`
  - Confirm bank: **150 test parents per family**. This gives a pooled-primary MDE of about 0.028 at α=.0167, and per-family MDEs of P 0.061, R 0.047 and L 0.036. A standalone P claim needs about 225.
  - The same n covers H1 against a 10pp margin (about 65/family needed) and H2 at 10pp (about 42/family).
  - Cheapest route: a fresh test-only bank with a new seed and salt, deduplicated against the pilot keys, with heads trained on the pilot train/dev captures under identical arm fingerprints.
    - It has 450 parents. Capture costs about 0.4× the pilot per arm if non-H1 arms capture only k=4 and "one" items.
    - Generation is about 40k calls per generative arm, versus about 7.7k in the pilot.

### 4. MAJOR: normalised regret is degenerate for P

- In P, 3% of parents have all 8 candidates with identical φ. 2.5% of (parent, goal) pairs have U* − mean U = 0, 5.7% have a denominator below 0.1, and 30% have tied optima. The ratio is bounded at 7 but skewed.
- **Fix:**
  - In `bank.build`, reject P parents with fewer than 4 distinct φ vectors.
  - Use a per-parent ratio of sums, Σ_g(U*−U_sel) / Σ_g(U*−Ū).
  - Score top-1 as U_sel ≥ U* − 1e-9.
  - Use a seeded random tie-break for both argmax and unparseable G outputs.

### 5. MAJOR: H1 can pass on a plan-blind shortcut

- In P at k=4, copying s0 gives 85.2% bit accuracy and 29.2% exact match. An oracle parent-majority predictor gives 90.4% and 45.4%.
- **Failure scenario:** e2e generation sits near 3%. A probe that only decodes s0 then clears "≥20pp" with no consequence computation.
- **Fix:**
  - The H1 comparator should be max(e2e, copy-s0, a probe on an s0-only prompt).
  - Alternatively, score only the bits that change across the plan.
  - Regret is a within-parent comparison and is immune to this, which is another reason to make H3 primary.

### 6. MAJOR: the success criteria have about 50% power at their own thresholds, plus forks

- "≥20pp with CI excluding 0" and "≥+10pp" pass only half the time when the true effect equals the threshold.
- "Exact-P or per-feature accuracy" lets either metric pass, which is a fork.
- "randdelta ≈ Supa" has no equivalence margin.
- "Monotone or peaked" is nearly unfalsifiable.
- **Fix:**
  - H1: require the lower 95% CI bound to exceed 10pp, using one preregistered metric per family.
  - H2: require the lower CI bound above 0, and power the study for 15pp.
  - randdelta: use TOST with a ±5pp margin. This needs about 137 parents for exact match (per-parent SD about 0.2) or about 22 for per-feature accuracy.
  - Dose: use a Jonckheere trend test over α ∈ {0, 0.5, 1}, and treat α=2 as descriptive.

### 7. MAJOR: the multiplicity is very large

- Families (3) × k (3) × arms (8) × layers (8) × readout positions (2) × budgets (≥4) × metrics (≥4) is about 18k cells. "At ≤25% of train labels" also means any of several budgets.
- **Fix:** three primary contrasts, with Holm across them:
  1. **H3:** B3 − B2★ regret, Titan-Sparse, answer slot, k=4, a single 25% budget, family-equal-weighted and pooled.
  2. **H1:** Sparse latent minus max(e2e, copy-s0), k=4.
  3. **H2:** Sparse − Supa, k=4, on the same metric.
- Everything else is secondary under BH q=0.10 and labelled exploratory: per-family results, other k, layer curves, the dose line, Experimental, qwen3, chain modes, S, G and other budgets.

### 8. MAJOR: DEV selection protocol and matched tuning are not specified

- DEV has only 28–41 parents per family, and choosing among 8 layers × λ × head hyperparameters on it is noisy.
- "Same LR grid" does not match budgets once B2 has extra knobs (architecture, r, number of goals).
- Selecting on test goals leaks.
- **Fix:**
  - Select layer, λ and hyperparameters by GroupKFold(5, groups=parent_id) over train+dev, then refit and evaluate once on test.
  - Both B2 and B3 use the same criterion: DEV regret on train goals only.
  - Give both arms the same number of configurations, for example 24 each, and the same 8 layers.
  - Fix the answer slot as the primary position.
  - For H2, select per arm under the identical grid.
  - Choose the layer once at full budget with one seed, then freeze it for other budgets and seeds. This keeps CPU training tractable: one layer is about 39 MB from the memmap.

### 9. MAJOR: seed and subset variance is not specified

- At 25% the budget is 33–36 train parents, or 264–288 candidates, for a head of about 262k parameters. Variation across seeds and subsets can match the effect size.
- **Fix:**
  - Use 5 initialisation seeds (0–4) × 3 nested subset draws per budget below 100%.
  - B2, B3 and S share the subsets and shuffling generator.
  - Average per-parent regret over seeds within each subset, then average over subsets, and then bootstrap.
  - Report seed SD and subset SD separately, because the parent bootstrap does not capture training-set variance.

### 10. MINOR: implementation details that change H3

- B3 must use sigmoid probabilities, not thresholded φ̂. Because U is linear in φ, w·E[φ] is exact expected utility.
- B2 should train on per-goal standardised U. Without it, MSE overweights large-norm goals.
- The design does not say which arm H3 uses; it should name Titan-Sparse.
- Freeze `heads.py` and `analysis.py`, with their hashes, in PREREG.md before any test-split head evaluation.

---

The CPU check scripts are in `C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-Consequence-World-Model-Bundle-2026-10-05/4c2a936e-32bf-4e01-aa62-961d6b162be5/scratchpad/`: `phi.py` computes φ from the bank, `power.py` and `power2.py` run the regret and power simulations.