**Verification against code and pilot bank (rerun on CPU)**

Confirmed:
- P, k=4, test split. Copy-s0 scores 0.852 per-bit and 0.308 exact. First-only scores 0.906 / 0.496. Ignore-pre scores 0.910 / 0.478.
- 38 of 224 P items (17 parents) are contested, meaning the final state differs from every heuristic. At k=2 there are 0. Ignore-pre is wrong on 117/224.
- Share of plans whose final state changes when the events are reordered: P 0.10, L 0.28, R 0.73.
- 98 of 200 P parents have fewer than 4 distinct final states.
- `SYSTEM_ONE[P]` does not state the no-op rule.
- At the answer slot, layer 0 is the embedding of a constant token.
- Generation runs only on `--gen-splits test`.

Dropped or moot:
- Cost leaking into G's φ: `data.PHI_SPEC` gives P 8 bits and no cost. DESIGN.md is out of date.
- "Unstandardized ridge/massive dims dominate probes": `heads.Standardizer` z-scores the inputs to every probe and head.
- "No head code / add a bilinear B2": `ScalarHead("bilinear")` already exists. It computes U = w·g(h), which is B3's functional form with a scalar Huber loss.
- "B3 thresholds φ̂": it uses sigmoid probabilities.
- "Choose the generative mode on DEV": there are no DEV generations.
- "P goal rank 9": φ_P is 8-dimensional.
- The boundary-token count cue: `tracking()` already fits a separate probe per j. It stays as a caveat only.

---

## (1) Amendments

### MUST (before analyzing the pilot)

**M1. Make DESIGN.md match the code.**
- φ_P is 8 bits, with no cost term and no cost-weight constraint.
- An unparseable generation counts as wrong for fidelity (as `gen_fidelity` does). For decisions it gets the parent-mean fill (as `decisions` does). Remove "random tie-break".

**M2. Floor ladder in `analyze.floors()`.** Computed from `s0` and `events`:
- copy-s0
- first-event-only: `env.step(s0, e1)`
- last-event-only: `env.step(s0, e_k)`
- ignore-pre (P): `s = (s & ~e["delete"]) | e["add"]` for every event
- ignore-status (L): treat every event as posted
- touched-rule (L/R `changed`): changed iff the account or register is named by a posted event
- majority
- S-ridge and S-MLP on `decide.symbolic`

Per family × k, report balanced accuracy / MCC per feature, exact binary φ, and skill = (acc − F)/(1 − F), where F is the maximum over the ladder.

**M3. Change-mask target.**
- Δ = φ_bin(s_k) XOR φ_bin(s0).
- Fidelity probes are trained on Δ. Heuristics and generators are scored by computing Δ from their predicted state.
- For any state predictor, exact-Δ equals exact-φ_bin. For probes, it removes the credit they would otherwise get just for copying s0.
- Exact-Δ is the H1/H2 metric.

**M4. Ridge is the confirmatory fidelity readout.** The MLP result is reported as secondary.

**M5. Input-decodability probes.** Same arm, position and layer as the main probe. Targets:
- P: s0 bits
- L: sign(s0) and s0/100
- R: s0 > 48 and s0 parity
- event-1 kind and target slot

Report Δinput next to every H2 delta.

**M6. P chain comparator.**
- Use `chain_m1`, which states the no-op rule, as P's chained comparator.
- Report `chain_one` for P stratified by `env.applicable(s_t, e_t)`, and score it against both `env.step` and the ignore-pre step.

**M7. Generative fidelity.** Report the parse rate. Add a sensitivity analysis that imputes unparseable outputs as s0.

**M8. H1 comparator.** Per family, the maximum on test over {ladder ∪ e2e, `chain_one`, `chain_m1`}. Selecting on test inflates the comparator, so this is conservative for H1.

**M9. Fixed P strata, reported as secondary:**
- contested: final ∉ {copy, first, last, ignore-pre}
- discriminating: ignore-pre ≠ truth

### SHOULD (before the confirm bank and PREREG)

**S1. Confirm bank.**
- New salt, test-only, deduplicated against pilot keys.
- 150 test parents per family. This should give about 90 P parents with contested items.
- Heads are trained on pilot train/dev. Use identical arm fingerprints.
- Arms not used for H1 capture only k=4 and the "one" format.
- Generate on `dev,test`.

**S2. `bank.build` filters and twins.**
- Reject P parents with fewer than 4 distinct final states and resample. About 49% of pilot P parents would be rejected; document the resulting distribution shift.
- Emit at least one order-twin per P and L parent: same s0, same event multiset, permuted, ticks reassigned, kept only if the final state differs. Report probe exact-Δ on twin pairs.

**S3. Goals.**
- 256 test goals (no GPU cost).
- Restricted-span training goals: 16 goals inside a seeded random r ∈ {2, 4} subspace. Test goals come from the full space. Report regret against ‖P⊥w‖²/‖w‖².
- Out-of-φ goals:
  - P: `n_applied`, bit_i ∧ bit_j
  - L: final > 50, `n_posted`
  - R: final mod 3, X > Y
- Fit few-shot ridge with n_new ∈ {16, 64, 256} on four inputs: φ̂, B3's hidden layer, B2★'s penultimate layer, and raw h.
- Add B0 decision floors: selectors based on ignore-pre, ignore-status and first-only.

**S4. B2 ladder in `ScalarHead`.**
- Architectures: concat, FiLM, UVFA, bilinear with r ∈ {|φ|, 32}.
- U standardized per goal on train.
- B2★ and B3 are both selected by DEV regret on train goals, with an equal number of configurations (24 each) and the same layers.

**S5. Model selection.**
- GroupKFold(5, groups = parent) over train+dev.
- Choose the layer once at full budget, then freeze it across budgets and seeds.
- H2 arms use an identical grid, with selection done separately per arm.

**S6. Variance.**
- Replace `tr_pids[:n]` with 3 nested subset draws × 5 initialization seeds for every budget below 100%. B2, B3 and S share subsets.
- Bootstrap two ways: parents within family, and goals.
- Report seed SD and subset SD separately.

**S7. Regret definition.**
- Per parent: Σ_g(U* − U_sel) / Σ_g(U* − Ū).
- Parents with a zero denominator are excluded; preregister this.
- Top-1 counts as correct when U_sel ≥ U* − 1e-9.

**S8. Control labels (Hewitt–Liang style).** CPU only, using existing captures.
- Add `env.step_cf`:
  - P: swap `pre_on` and `pre_off`
  - L: reverse the transfer direction
  - R: mix = 3·other (keep mod 97)
- Report selectivity = skill(true) − skill(cf).

**S9. H2b, readout vs world model.** Uses the k=1 "-one" items, which sparse and supa have already captured and generated.
- Compute Δprobe = probe(Titan) − probe(Supa) and Δgen.
- Classify as "readout" if Δprobe ≤ 0.25·Δgen and Supa's probe exact score is at least 0.8× Titan's.
- Add cross-arm probe transfer: train on one arm's states, test on the other.

**S10. PREREG.md.**
- Hashes of `heads.py` and `analyze.py`.
- Holm across the primaries; BH q = 0.10 across secondaries.
- randdelta ≈ Supa tested by TOST with ±5 pp margins.
- Dose line: Jonckheere test over α ∈ {0, 0.5, 1}; α = 2 is descriptive only.

### COULD

- **C1.** Random-init arm, one GPU capture: `AutoModelForCausalLM.from_config(...)` with `torch.manual_seed(17)`.
- **C2.** Few-shot e2e generation with 3 solved train chains in the prompt.
- **C3.** Parity metric normalized by per-dimension SD. Log per-layer max|h| and the top-5 dimensions.
- **C4.** Exclude layer 0 from selection. Assert that the layer-0 probe equals the majority predictor, as a bug check.

---

## (2) Primary confirmatory contrasts (exactly 3, Holm-adjusted)

Common settings:
- Titan-Sparse, answer slot, k = 4, confirm-bank test parents.
- Unit: the parent, using the mean over its 8 candidates (and over goals for H3).
- Pooled across P, L and R with equal family weights; per-family results are secondary.
- One-sided tests, 10k-resample bootstrap.
- Expected precision (estimate): per-parent paired SD ≈ 0.25, so pooled SE ≈ 0.012.

| # | Contrast | Metric | Test / pass rule | Minimum effect |
|---|---|---|---|---|
| P1 (H3-transfer, RCWM primary) | B3-MLP (sigmoid φ̂, U = w·φ̂) − B2★ | Normalized regret (ratio of sums) on 256 full-space test goals. Training goals restricted to r = 4. Budget 35 train parents (25%), averaged over 5 seeds × 3 subsets | Two-way bootstrap. Pass if the Holm-adjusted upper bound < 0 | −0.05 regret (pooled MDE ≈ 0.03 at 150 parents per family) |
| P2 (H1) | Ridge probe − per-family max comparator (M8) | Exact Δ-mask | Parent bootstrap. Pass if the Holm-adjusted lower bound > +5 pp | Powered for +15 pp. Key secondary: the P contested stratum |
| P3 (H2) | Sparse − Supa, ridge probe, per-arm GroupKFold layer | Exact Δ-mask | Parent bootstrap. Pass if the Holm-adjusted lower bound > 0 | +10 pp |

Gate on P3: call the effect "consequence-specific" only if |Δinput| ≤ 3 pp, or if the lower bound of (Δconsequence − Δinput) is above 0. Otherwise report it as a general change in representation.

In-span H3, which effectively compares losses, is secondary.

---

## (3) Extension: probe-verified self-rollout (PVR)

**Claim.** A linear probe on frozen Titan, trained only on one-step labels, can raise 4-step exact-state accuracy and give calibrated abstention, with no weight changes.

**Probe**
- Ridge/logistic on Titan-Sparse answer-slot states.
- Training items:
  - for `chain_one`: train-split `-one` items
  - for P with `chain_m1`: multi-format k=1 train items
- Targets φ(s_{t+1}):
  - P: 8 bits
  - L: sign / changed, plus ridge on final/100
  - R: hi / parity / changed, plus ridge on final values (Gaussian log-likelihood, σ set on DEV)
- Label budgets: {64, 256, 1024} rows.

**Rollout, at each step t**
- One forward pass on the step prompt gives the probe's posterior.
- Decode greedy plus N = 8 samples at T = 0.7. This needs a `do_sample=True, num_return_sequences=8` path in `generation.generate_texts`.
- Score each parsed candidate by its probe log-likelihood:
  `lp = (y*log p + (1-y)*log(1-p)).sum()` plus Gaussian terms.
- Pick the argmax. Abstain if `lp < τ`, with τ set on DEV.
- PVR-decode (P only): if no sample clears τ, output the probe's argmax state.

**Baselines (all at matched N)**
- greedy
- per-step self-consistency (mode)
- mean token-logprob reranking
- an S-verifier (symbolic MLP on the same labels)
- a Supa-probe reranking the same Titan samples
- oracle pick (per-step pass@8)

**Gate.** Run a 30-chain DEV pilot first. Proceed only if:
- per-step pass@8 − greedy ≥ 5 pp, and
- the probe's AUROC for step correctness ≥ 0.70.

If pass@8 is not above greedy, run only PVR-decode.

**Cost (estimate, not measured)**
- Pilot test set: about 15–25 GPU-minutes.
- Confirm bank: 3,600 chains × 4 steps × 9 decodes ≈ 130k short generations (12–56 tokens) with a shared prefill, about 1.5–3 GPU-hours.
- Probe training is on CPU.

**Primary (its own Holm family of 2)**
- (a) k=4 exact state, P, L and R pooled: PVR − per-step self-consistency. Lower bound > 0, minimum effect +10 pp.
- (b) Titan-probe PVR − Supa-probe PVR. Lower bound > 0, minimum effect +5 pp. This connects the result to H2.

**Secondary**
- Gap closure (PVR − greedy)/(oracle − greedy) ≥ 0.5.
- AURC lower than token-logprob's.
- Exact accuracy at 50% coverage at least 2× the full-coverage rate.
- Titan PVR beats the S-verifier PVR at 256 labels.

**What a null means.** If oracle-pick is high but PVR is no better than self-consistency, that falsifies the claim that the latent readout is useful.

---

My verification script is `C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-Consequence-World-Model-Bundle-2026-10-05/4c2a936e-32bf-4e01-aa62-961d6b162be5/scratchpad/lead_verify.py`. I made no changes to the project.