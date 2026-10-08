# RCWM-T1 design review: novelty and significance

I ran these checks on CPU against `data/bank_pilot.json` (600 parents; the test split has 28 P, 26 L and 26 R parents). I modified no files and did not use the GPU. The check scripts are `chk1.py`–`chk3.py` in `C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-Consequence-World-Model-Bundle-2026-10-05/4c2a936e-32bf-4e01-aa62-961d6b162be5/scratchpad/`.

## Novelty verdict

| H | Closest prior work (2021–2026) | New? | Value to the programme |
|---|---|---|---|
| H1 latent ≥ generative | Li, Nye & Andreas 2021 (state probes in BART/T5); Othello-GPT (Li et al. 2023; Nanda et al. 2023); chess (Toshniwal 2022; Karvonen 2024); Karel program-state probes (Jin & Rinard 2024); "(How) do LMs track state?" (Li, Guo & Andreas 2025); "LLMs know more than they show" (Orgad et al. 2025; Gekhman et al. 2025); latent multi-hop (Yang et al. 2024; Biran et al. 2024) | **No, as written.** A probe beating a generator that is already at floor is the expected result. | Low, unless the readout is used to improve outputs |
| H2 attribution to the delta | Prakash et al. 2024 (fine-tuning strengthens existing entity-tracking circuits); model-diffing/crosscoders; task arithmetic, TIES, DARE | **Yes.** No prior work I know of combines a dose line with a support- and norm-matched random null on a 0.056%-norm delta, scored on both latent and generative accuracy. | High: it bears directly on forgetting |
| H3 B3 vs B2 | Successor features and USFA (Barreto 2017; Borsa 2019); SF-GEN (Cao 2023) | **No.** It holds by construction when goals are linear in φ. | Low: it never changes an output |
| H4 shortcut | Shortcuts to automata (Liu et al. 2023); precondition probing in Physics of LMs 2.1 (Ye et al. 2024) | Partly | Medium, if a verifier exploits it |

## Findings

**1. BLOCKER: H1 can pass without the LM computing anything.**
- **Failure scenario:** e2e generation is at floor (DMRF 1/32). Trivial predictors on the pilot bank already clear the "+20 pp" bar:
  - P, k=4 exact match: copying s0 scores 29.2%, ridge on the symbolic input encoding scores 33.5% on test, and a crude numpy MLP (my stand-in for S) scores 41.1%.
  - P per-bit accuracy from copying s0 is 0.85–0.95.
  - L/R `changed`: the rule "touched by a (posted) event" is right 95.9% and 94.1% of the time.
  - So any probe passes H1, and the "or per-feature accuracy" clause makes it automatic. The 1024→256 MLP head can also learn the dynamics itself from linearly encoded inputs.
- **Fix:**
  - Change H1's comparator from e2e to max(S at matched labels, linear-on-input, chain_one).
  - Make the "the LM computed it" claim only with linear probes.
  - Score P on the discriminating subset, where ignoring preconditions gives a different answer from the truth. That is 53.6% of k=4 chains, or 117 test candidates.
  - Drop `changed` from the H1/H2 metrics, or residualise it against the touched rule.
  - Report decodability by event boundary j=1..4 × layer. That decay curve is what can be compared with Li et al. 2025.

**2. MAJOR: H3's outcome is fixed in advance, and the B2 contrast is unfair.**
- **Failure scenario:**
  - Goals are defined as U = w_g·φ, so B3 is the correctly specified model, and successor-feature theory already predicts that it transfers to in-span goals.
  - B2 is an MLP on [h; w_g], which has to approximate a bilinear interaction it represents poorly.
  - The "label" counts are not comparable: B3 gets one φ vector per candidate, B2 gets 16 scalars.
  - Selectors that only read the input already reach normalised regret 0.38 for P (ignore preconditions), 0.55–0.57 for L and 0.70 for R, against 1.0 for random selection.
  - 5 P parents (80 goal–parent pairs) have the same φ for all 8 candidates, so regret is 0/0. 98 of 200 P parents have at most 3 distinct final states.
- **Fix:**
  - Make B2 bilinear (hᵀ M w_g, or FiLM) at matched parameter count.
  - Count labels as labelled rollouts.
  - Add out-of-span test goals, for example:
    - P: "bits i AND j set", "n_applied ≥ 3"
    - R: "X > Y", "X ≡ Y mod 2"
    - L: "A ≥ B", "Σ balances"
  - Report the input-only heuristics as B0 floors.
  - In `bank.build`, resample degenerate parents:
    ```python
    if len({env.dumps(phi_design(fam, s0, c["events"])) for c in cands}) < 4: continue
    ```

**3. MAJOR: H2 is the most novel part, but only one outcome is preregistered.**
- **Failure scenario:** The more likely result is that Supa already encodes the one-step consequence and the delta mostly installs a readout. Supa's 14/384 may be a failure to verbalise, not a missing skill. The current criterion would record that as an H2 fail. For a programme fighting forgetting, though, "the 0.056% delta fixes the verbaliser, not the world model" is the more useful finding. It would explain why tiny sparse deltas work and predicts low interference.
- **Fix:** Preregister an H2b on the k=1 "one"-format items, which are already captured, so it needs no extra GPU:
  - Compare Δprobe = probe(Titan) − probe(Supa) with Δgen = gen(Titan) − gen(Supa).
  - Call it "readout" if Δprobe ≤ 0.25·Δgen and Supa's probe exact-match is at least 0.8× Titan's.
  - Add cross-arm probe transfer on CPU: train on Supa states and test on Titan, and the reverse. This shows whether the delta rotates the representation.
  - Report the layer at which decodability appears along the α dose line.

**4. MAJOR: no hypothesis uses the readout to change what the model outputs.**
- **Failure scenario:** All four hypotheses could pass and still leave multi-step accuracy and abstention unchanged. Showing latent > generative only helps the programme if that gap can be turned into better outputs.
- **Fix:** add the extra arm described below.

**5. MINOR: the chain_one baseline is mismatched for P.**
- **Failure scenario:** SYSTEM_ONE for P presumes the supplied action is applicable, but 48% of pilot steps are inapplicable and their gold answer is a no-op. Some chain_one P failures, and some H4 "ignores preconditions" evidence, would then just be the model following the prompt.
- **Fix:** use chain_m1 as the generator for H4, and stratify every P generation metric by applicability.

**6. MINOR: power.**
- **Failure scenario:** With 26–28 test parents per family, and assuming a parent-level paired SD of about 0.25, the 95% CI is roughly ±10 pp per family and ±5.5 pp pooled.
- **Fix:** give the confirmatory bank at least 150 test parents per family. Building the bank is CPU-only, and GPU cost scales only with the split you generate on.

**7. MINOR: positioning.**
- **Failure scenario:** Without scoping, results read as a general "LLMs have latent world models" claim that the literature in the table already covers.
- **Fix:** scope claims to "a 0.6B model plus a 384-example sparse delta, on a 3-family simulator". Cite the table above plus:
  - Kadavath 2022 (P(IK) value head)
  - Cobbe 2021 (trained verifiers)
  - Kossen 2024 (semantic-entropy probes)
  - Li et al. 2023 (inference-time intervention, ITI)
  - Sun, Stolfo & Sachan 2025: probes that detect arithmetic errors and guide self-correction. This is the closest precedent for the arm below.

## Highest-value extra arm: probe-verified self-rollout (PVR)

**Claim.** A linear consequence probe on frozen Titan, trained only on one-step labels, raises 4-step end-to-end accuracy and gives calibrated abstention, without changing any weights.

**Protocol** (Titan-Sparse; test parents in P and R; L optional):

1. **Probe.**
   - Fit a linear/logistic probe on the answer-slot states of the train-split `one` items. These are already captured.
   - The target is φ(s_{t+1}): the 8 bits for P; for R, hi/parity/changed plus ridge regression on final values.
   - Use label budgets of {64, 256, 1024}.
2. **Rollout.** At each step t:
   - Run one forward pass on the one-step prompt (chain_m1 for P) to get h_t and the probe's posterior.
   - Sample N=8 answers at T=0.7, plus the greedy answer. This needs a `do_sample=True, num_return_sequences=8` variant of `generate_texts`.
   - Score each parsed state by its log-likelihood under the probe:
     ```python
     lp = (y*np.log(p) + (1-y)*np.log(1-p)).sum()   # y = φ(candidate state)
     pick = int(np.argmax(lp)); abstain = lp[pick] < tau_dev
     ```
   - For P, also run "PVR-decode": when no sample clears τ, output the probe's argmax state. If the probe encodes applicability (H4), this fixes the no-op steps directly.
3. **Baselines at matched N.**
   - Greedy chain
   - Per-step self-consistency (majority vote)
   - Mean token-logprob reranking
   - An S-verifier: the symbolic MLP trained on the same labels
   - Supa-probe reranking of the *same* Titan samples
   - Oracle pick (the per-step pass@8 ceiling)

**Cost.**
- About 54 parents × 8 candidates × 4 steps × 9 decodes ≈ 15.5k short generations, plus about 1.7k prompt forward passes. That is roughly 15–25 GPU-minutes, which is my estimate, not a measurement.
- Probe training is CPU-only.
- Run a 30-chain pilot first. If per-step pass@8 is not well above greedy, reranking cannot help and only PVR-decode is worth running.

**What would make it compelling.** Each comparison should have a parent-bootstrap 95% CI excluding 0.
- **k=4 exact match, P and R pooled:** PVR beats self-consistency by at least 10 pp.
- **Gap closure:** (PVR − greedy) / (oracle pick − greedy) ≥ 0.5.
- **Label efficiency:** the Titan-probe PVR beats the S-verifier PVR at 256 labels, showing the LM's hidden state makes a label-efficient verifier.
- **Attribution:** the Titan-probe PVR beats the Supa-probe PVR by at least 5 pp. This ties the gain to the sparse delta and connects to H2.
- **Abstention:**
  - Its AURC is lower than token-logprob's.
  - At 50% coverage, k=4 exact match is at least twice the full-coverage rate.

**Why it matters.** A positive result turns "Titan knows more than it shows" into a measurable multi-step gain with no retention cost, because no weights change. That is the RCWM thesis in a form you could deploy. A clean null — oracle pick high but PVR no better than self-consistency — would directly falsify the claim that the latent readout is useful.