# PREREG-1 review: could C1–C3 pass or fail for reasons other than the claim?

**How I checked:** I re-ran the analysis on the pilot TEST split (26 L and 26 R parents) on the CPU only. Nothing touched the confirmation bank. Scripts and JSON are in `C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/docs/prereg_review/lens_confounds/` (a1–a5). So far the dry-run log matches the R7 numbers for L at k=1, 2 and 4.

---

**1. MUST-FIX — C3 measures Supa's output format, not its simulation.**
- **Pilot parse failures:**
  - Supa chain_one at R parsed 0%; none of 208 chains reached step 4.
  - At L k=4, 60% of Supa chains parsed.
- **Why it fails even with the lenient parser:**
  - 92 of 208 Supa R first-step answers still fail to parse.
  - 50 of those hit the 64-token cap while echoing `{"before":…,"event":…,"after":{"X":80,"Y":17`.
  - Most of the rest echo the input and never write an `"after"`.
  - L failures are arithmetic inside the JSON, e.g. `"C": 5 - 3 = 2`.
- **What the parent-mean fill then does:** an unparseable candidate is never selected, and a parent where nothing parses is chosen at random. Supa R k=4 = 1.007 is therefore exactly random.
- **Fill-rule sensitivity:** if unparseable candidates instead get the touched-entity rule from #2, Supa R k=1 goes from 0.997 to 0.29, and L k=4 from 0.836 to 0.748.
- **The §1 pilot table is invalid for C3.** Its Supa numbers (0.835 / 0.994) come from strict-parse chains (D-12), so C3's pilot evidence and power argument do not hold.
- **Expected on the confirmation bank:** with about 56% per-step parse at R, roughly 10% of Supa chains would reach k=4. C3 would pass because of format alone.
- **Fix:**
  - Raise the generation cap to at least 160 tokens for all arms. Titan's longest L/R answer is 34 characters, so its greedy outputs do not change.
  - Register a parser extension used identically for both arms: take the last complete `"after"` object, and the right-hand side of `a op b = c`.
  - Make the primary fill rule "last parsed chain state" for both arms, with parent-mean as a sensitivity analysis.
  - Report parse rates next to C3, and add a C3 restricted to candidates both arms parse.
  - Reword C3 as "usable end-to-end simulator", or pair it with the format-free likelihood ranking (D-09).

**2. MUST-FIX — An answer-free "touched-entity" rule beats both learned comparators.**
- **The rule (F_touched):** φ of the start state, with changed = 1 for every entity named by an applied event (posted events only for ledger; every event for register).
- **Regret at k=4:**

| Method | L | R |
|---|---|---|
| G_chain(Titan) | 0.507 | 0.386 |
| F_touched (no answer, no labels) | 0.599 | 0.683 |
| S | 0.746 | 0.841 |
| B3 | 0.813 | 0.965 |
| Oracle "changed" dims only | 0.56 | 0.59 |

- **Why it matters:**
  - G gets the changed flags right 91% (L) and 99% (R) of the time, mostly because it rewrites the whole state.
  - So C1 and C2 can pass simply because both heads fail an easy sub-problem.
  - G − F_touched at L is −0.092 [−0.213, +0.033]: not significant and short of the −0.10 minimum effect. At R it is −0.298.
- **Fix:**
  - Add F_touched to `decisions.run` and register a contrast C0: G_chain − F_touched, with Holm over four contrasts.
  - Add a validity gate: if B3 or S fails to beat F_touched on the confirmation bank, report C1/C2 as "comparator inadequate", not as support.
  - Pre-declare a secondary analysis with goals that ignore the changed dims. On the pilot G still wins there (k=4 G − B3: L −0.36 [−0.54, −0.18], R −0.49 [−0.59, −0.39]), so this robustness check is cheap and favourable.

**3. MUST-FIX — C2 depends on label budget, so S is not a fair bar.**
- **Labels cost nothing here:** the environment simulator produces them. I trained S on freshly generated parents (`bank.build`, seed 777001, no model involved) and tested on pilot TEST.

| Training parents | L k=4 | R k=4 |
|---|---|---|
| ~133–141 (pilot) | 0.746 | 0.841 |
| 500 | 0.684 | 0.726 |
| 1000 | 0.607 | 0.676 |
| 4000 | **0.203** | 0.588 |

- **Consequence:** at 4,000 parents S beats G at L by 0.30. Pooled G − S is about +0.05, so C2 would fail.
- **Fix:**
  - Reword C2 as "at equal label budget (~1.1k k=4 candidates per family)".
  - Register S@4000 as a secondary with a pre-stated interpretation.
  - Register S-roll as a secondary: a no-LLM one-step learner, trained on the ~4.4k one-step transitions in the pilot TRAIN chains, applied four times. This is the matched "explicit simulator without an LLM".
  - Drop "beats a no-LLM learner" from the headline unless S@abundant also loses.

**4. SHOULD — B3 is also label-limited.**
- **Learning curve** (pilot TRAIN subsets, B3 at the registered layer):
  - L: 0.920, 0.894, 0.863, 0.813 at 16, 35, 70 and 133 parents. It is still falling.
  - R: about 1.0 up to 70 parents, then 0.965 at 141.
- **Risk:** given how far S improved with more labels, B3 at about 4,000 parents is unknown. C1 could reflect the label budget rather than the method.
- **Fix:** before freeze, capture an extra pilot-like TRAIN bank (L and R, k=4 multi-format items only, capture without generation) of at least 1,000 parents per family. Register B3@abundant as a secondary. Otherwise restrict the claim to "~1.1k labels".

**5. SHOULD — Select B3's layer on the endpoint. C1 is not a layer artifact.**
- **Weak selection at R:** the registered R k=4 layer (L8) was chosen by a change-mask ridge whose exact-match score is 0.010, essentially noise. B3 TEST regret there is 0.965.
- **No configuration closes the gap:** I tried every layer (7) with both the answer-slot state and the event-4 boundary state, each with MLP and ridge. The best TEST regret is 0.754 (L) and 0.839 (R), against G at 0.507 and 0.386. Even the best-on-test choice leaves C1 standing.
- **Event-boundary states are no better:** L 0.79–0.91, R 0.90–0.99. They do not need to be added.
- **Fix:**
  - Choose B3's layer by its DEV decision regret (the endpoint) or by its DEV φ score. This costs nothing and removes an objection.
  - Average B3 and S over 5 seeds (seed ranges: B3 L 0.773–0.817, S 0.746–0.778).
  - Report the best-on-test layer as an upper bound for B3.

**6. SHOULD — The real contrast is serial versus single-pass, not simulator versus heads. Reword.**
- **The untrained prompt format does not hurt B3:** at k=1, B3 on Titan's trained one-step frame scores L 0.233 / R 0.356, versus 0.244 / 0.292 on the multi-step prompt.
- **B3 beats G at L k=1:** G − B3 = +0.113 [+0.024, +0.211].
- **What the k=4 gap really compares:** four sequential forward-and-decode passes, each anchored on the model's own written state, against one readout of the k=4 prompt.
- **Proposed claim:** "At horizon 4, Titan's serial self-rollout in its trained one-step frame gives lower regret than single-pass consequence heads on the k=4 prompt state trained on ~1.1k labels."
- **Matched control** (needs only forward passes, no generation, on the confirmation bank; add to §6 step 2):
  - A latent-chained B3: a one-step-frame answer-slot head, trained on pilot one-step transitions, decodes the full next state.
  - That state is re-rendered into the next prompt, four times.
  - It differs from G only in the decoder: probe instead of LM head.

**7. NOTE — The endpoint itself is sound.**
- **No excluded parents:** none in L/R, with goal-summed denominators of 76–312. Every k=4 parent has 8 distinct final states.
- **Raw (unnormalised) regret gives the same order:**
  - L k=4: G 0.36 < F_touched 0.45 < S 0.56 < B3 0.61 < Supa 0.63.
  - R k=4: G 0.23 < F_touched 0.41 < S 0.51 < B3 0.58 < Supa 0.61.
- **Disclose in §4:** the Gaussian goals put about 1/3 of their weight on the "changed" dims (4 of 12 for L, 3 of 9 for R). Those dims are exactly where the touched-entity rule and G's state rewriting score well.

**8. NOTE — Titan is not artificially helped.**
- At most 3 of 832 Titan outputs per family fail to parse, so the lenient parser and the fill rule make no difference for Titan.
- `run_arm` chains only `p["candidates"]`, not the twin. Twin-pair consistency can therefore be computed for e2e only; §5 should say so.