# RCWM-T1 design review: confounds, leakage, measurement validity

I read DESIGN.md, `rcwmt/*.py` and `scripts/run_arm.py`. I ran CPU-only checks on `data/bank_pilot.json` and the Titan tokenizer. The test split has 28 P, 26 L and 26 R parents, which is 224/208/208 candidates. All floor numbers below are on that test split. My check scripts are in the session scratchpad (`floors.py`, `h3.py`, `slearn.py`). Nothing in the experiment was modified.

**What I verified as clean.** Neither capture position can see the answer or later events.
- The answer slot is always the token `\n\n` after `</think>`, in all 14,400 multi-format items.
- Boundary tokens are `},{"` for j<k (6,400 per family) and `}` for the last event, followed by a separate `]}`. The next token after `},{"` is a fixed key name (`add`, `account` or `kind`), never a value.
- Digits are single tokens, so no value merges into a boundary token.
- Leakage between splits is negligible: 1 of 224 P k=1 test items matches a train item semantically; L and R have none.

---

### 1. BLOCKER: Copying the start state alone can pass H1

**Failure scenario.** P k=4 floors on test:

| Floor | Per-bit accuracy | Exact 8-bit |
|---|---|---|
| copy-s0 | 0.852 | 0.308 |
| first-event-only | 0.906 | 0.496 |
| ignore-preconditions | 0.910 | 0.478 |
| last-event-only | 0.898 | 0.464 |
| majority | 0.545 | 0.000 |

If e2e generation sits near its DMRF floor (~3%), a probe that only decodes s0 clears the "≥20 pp" bar by about 28 pp. That would make H1 pass with zero consequence information. At k=2 every test final state equals one of {copy, first-only, last-only, ignore-pre}, so exact-P there amounts to sorting items into those cases. At k=4 only 38/224 items, from 17 parents, are shortcut-resistant. Only 38/224 chains apply every step.

**Fix.**
- Restate H1 as latent − max(best floor, best generative mode chosen on DEV) ≥ 20 pp.
- Also report skill = (acc − floor_max)/(1 − floor_max) per feature.
- Freeze a contested subset in PREREG (final not in any heuristic set) and make it a co-primary.
- In analysis, compute floors directly from `gold_prefix_states` and events, using the same `rollout` variants as `floors.py`.
- Size the confirmatory bank so the contested subset has at least 60 test parents.

### 2. BLOCKER (for H2): Sparse − Supa ≥ +10 pp could come entirely from better encoding of the start state

**Failure scenario.** Titan's one-step training largely teaches it to copy unchanged bits. At P k=4, per-bit headroom above copy-s0 is only about 15 pp. A Titan answer-slot state that represents s0 more linearly would give "+10 pp latent" with no k-step dynamics. L sign_neg is similar: copy-s0 scores 0.84–0.89 there.

**Fix.**
- Use residualized targets for H2: P `changed` (final XOR s0, already in `env.features`), and per-bit accuracy on contested bits only (425/1792 bits at k=4).
- Add input-decoding control probes per arm at the same positions: s0 bits, event-j `add`/`delete` masks, L `status`.
- Report the H2 effect as Δ(consequence skill) minus Δ(input decodability). If Δinput ≈ Δconsequence, the delta is improved input encoding, not consequence.

### 3. MAJOR: No control separates "the LLM represents the consequence" from "the head computes it"

**Failure scenarios.**
- **Layer 0 is a dead baseline.** At both capture positions, `hidden_states[0]` is the embedding of a constant token (`\n\n`, `},{"` or `}`). Qwen uses RoPE, so no position signal is added. A layer-0 probe is therefore just the majority predictor, not a surface-encoding control.
- **S is a weak bar.** I trained the no-LLM learner S on CPU (140-d symbolic input, 1024→256 head shape, about 1150 train rows):

  | Family, k=4 | S MLP per-bit / exact | Comparison |
  |---|---|---|
  | P | 0.876 / 0.406 | below ignore-pre (0.910 / 0.478) and first-only (0.906 / 0.496) |
  | R | 0.669 | ≈ majority (0.626) |
  | L | 0.840 | ignore-status 0.963 |

  At 25% of labels, S on P drops to 0.771 / 0.170.

  So "B3 beats S" can't be read as "the LLM adds dynamics knowledge". The positive side: at this label budget, learning the dynamics from scratch fails. A latent readout well above the heuristic floors is therefore unlikely to be computed by the head alone.
- **Linear results are near the floor.** S-linear already reaches 0.872 per-bit on P k=4, close to copy-s0's 0.852, because copy-and-set is mostly linear.

**Fix.**
- Make the linear probe the confirmatory readout for H1 and H2; report the MLP only as secondary. The parent plan (`report/04_representation_masks.md`) already asks for linear and nonlinear probes reported separately.
- Add a random-init arm, so the architecture plus head is controlled. In `runtime.load_arm`: `torch.manual_seed(17); m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(PATHS["supa"]), torch_dtype=torch.float32).cuda().eval()`.
- Add control labels in the style of Hewitt and Liang: same prompts and same captures, but φ computed under a rule of matched complexity. Add `env.step_cf`:
  - P: swap the roles of `pre_on` and `pre_off`. Do not use "add before delete": `delete` never overlaps `add`, so that rule is identical to the real one.
  - L: reverse the transfer direction.
  - R: mix = 3·other, mod 89.

  Report selectivity = skill(true) − skill(cf). If the head computes from surface features, it learns both rules equally well.
- Add the shuffled-action and order-twin control that the parent plan already requires (`04_representation_masks.md`: "Compare prediction under real actions against shuffled action labels"). See finding 9.

### 4. MAJOR: Comparing latent and generative scores is not like-for-like

**Failure scenarios.**
- The probe is supervised on about 1150 in-distribution labels in the multi format. Generation is zero-shot in a format Titan never saw (the `"events"` list and the changed system rule).
- "Unparseable → random tie-break" covers selection only. The fidelity rule for unparseable outputs is unstated; counting them as wrong punishes formatting, not dynamics.
- For P, the G arm's φ includes `cost_applied`, which cannot be derived from an e2e final state. Taking it from the gold trajectory leaks oracle information to G; taking it from ignore-pre penalizes G.

**Fix.**
- For fidelity, impute unparseable outputs as s0 (`pred = pred if pred is not None else s0`) and report the parse rate separately.
- Use the best generative mode (e2e, chain_one or chain_m1), chosen on DEV, as the H1 comparator.
- Measure the format penalty from data that already exists: generation accuracy on `-one` vs `multi` k=1 test items.
- Add a few-shot e2e generation (3 solved train chains in the prompt), so generation also gets some in-distribution adaptation.
- For the G arm's P cost: in chain modes use `applied_t = gen_s[t+1] != gen_s[t]`. For e2e, run H3 both with and without the cost dimension.

### 5. MAJOR: The chain_one prompt does not state P's precondition rule

**Failure scenario.** SYSTEM_ONE for P says "a supplied applicable action…" and never states the no-op rule. Yet 44–47% of P test steps are inapplicable. When Titan applies an inapplicable action, it is following its trained frame, and that gets scored as a capability failure. It also mixes H4 (the shortcut question) into H1.

**Fix.** Score every chain_one step against both `env.step` and the ignore-preconditions step. Report per-step accuracy split by `applicable(s_t, e_t)`. For P, run a sensitivity analysis on the 38 all-applicable chains.

### 6. MINOR: The boundary token's identity encodes j==k

**Failure scenario.** Events are captured only for k=4 items, so boundaries j<4 are always `},{"` and j=4 is always `}`. A probe pooled across j, or a comparison of boundary-4 against boundary 1–3, picks up token identity. `},{"` also tells the model that more events follow. This is not content leakage, but it is a count cue.

**Fix.** Fit separate probes per j. Never pool j=4 with j<4. When comparing boundary-j with answer-slot-k=j items, note the token difference.

### 7. MINOR: fp16 storage and massive activations

**Failure scenarios.**
- `parity_check` normalizes by `np.abs(a).max()` over all layers and dims. Massive-activation dims, typical of Qwen at delimiter and sink tokens, dominate that maximum and hide errors in ordinary dims.
- Unstandardized ridge or logistic regression on raw residuals lets those dims dominate the Gram matrix.
- Layer 28 is post-final-norm, so it lives on a different scale from layers 0–24.

**Fix.**
- Log per-layer `max|h|` and the top-5 dims at capture.
- In parity, report `d / (per-dim std across items)`, or cosine distance after train-fit standardization.
- Standardize each layer with train mean and std (or apply that layer's own RMSNorm·γ) before probing.
- Keep the nonfinite check. fp16 range is fine.

### 8. MINOR: Per-feature reporting and the unit of analysis

**Failure scenarios.**
- Near-constant features inflate mean accuracy. The k=4 majority floors are R `changed` 0.84–0.88 and L `changed` 0.66–0.79.
- s0-determined features are effectively measured on only about 26 independent values. Base rates drift between splits (L neg_A: train 0.38, test 0.51).
- One P test parent has 8 candidates with identical φ, so 16 of 448 (parent, goal) pairs have U* = mean U and normalized regret is 0/0.

**Fix.**
- Report balanced accuracy or MCC plus skill over the floor ladder, with parent-clustered CIs.
- Define regret for degenerate parents (exclude them, pre-registered).
- Report decision floors. With simulated Gaussian goals (no goal code exists yet), normalized regret is:

  | Selector | P | L | R |
  |---|---|---|---|
  | random / copy-s0 | 1.0 | 1.0 | 1.0 |
  | first-event-only | 0.59 | 0.62 | 0.83 |
  | ignore-preconditions | 0.31 | 0.10 (ignore-status) | n/a |

  B3, B2 and G should be compared against these heuristic planners, not only against random selection.

### 9. MINOR (MAJOR for any claim about chaining on P): P consequences are mostly order-invariant

**Failure scenario.** Randomly reordering the same 4 events changes the final state for only 11% of P plans (L 29%, R 75%). A bag-of-events readout is nearly sufficient for P, so "multi-step" claims on P barely test sequential composition.

**Fix.** In the confirmatory bank, have `bank.build` emit order-twins: the same s0 and the same event multiset, permuted, with ticks re-assigned and kept only when the final differs. Report probe accuracy on the twin pairs. A surface or bag encoding scores at chance on them.

---

**Priority before spending GPU on the H2 dose-line and randdelta arms:** fix findings 1–3 in the analysis plan first. The random-init arm, the counterfactual labels and the input-decoding probes reuse the same captures, so they cost at most one extra arm.