<!-- Model-card retention section for the BWM package, from BWM's independent co-sign of RCWM R25 (D-71). Use at the next approved package rebuild. -->

## Retention (registered guard; RCWM 5c retention suite, sealed draw confirm_1708484032065101819)

Each adapter was loaded alone and unmerged on frozen Rineya-Titan-Sparse (FP32). It was compared with the same model with the adapter turned off, on 375 held-out templated items: 5 categories x 75 (unit conversion, percent/ratio, key-value recall, answerable/unanswerable question twins, format following). Both pre-registered graders, a strict typed grader and a lenient extraction grader, were applied.

Result: the registered retention guard detected no loss for either adapter (RETAINED, GO 5/5 binding). A failure under either grader would have been a NO-GO.

Net change in correct items vs adapter off (out of 375; strict / lenient; pass margin -15 pooled, -11 per category):
- BW_Fdelta_s0: -3 / -5
- BW_Pa_v2_s1: +9 / -10 (the graders disagree in sign; the lenient 95% interval reaches -20)

What this does and does not show. This is a low-power guard on narrow, templated short-answer skills. It is a floor, not proof of general retention. By its design analysis, it catches a loss of 9-12 items in one category only 13-46% of the time, and a loss of 6 items rarely. "No loss detected" does not mean "nothing was lost".

Known weaknesses:
- Unit conversion (base model and adapters).
  - The base model, Titan-Sparse, already does unit conversion worse than its parent Supa 1.7: 49 -> 30 of 75 correct, a detected loss.
  - Both adapters lower it further. Neither drop is significant after multiple-comparison correction.
    - BW_Fdelta_s0: 30 -> 23 of 75 under both graders (8 lost, 1 gained; raw p 0.039, corrected p 0.195). It was one item away from failing the guard.
    - BW_Pa_v2_s1: 30 -> 25 strict / 28 lenient.
  - With either adapter, the model gets 2 of 8 kilogram-to-gram items right, vs 6 of 8 for the base. The errors are wrong arithmetic (for example 4 kg -> "800" g, 15 kg -> "1500" g) or dropped zeros, not formatting.
  - Do not rely on either adapter for unit conversion.
- BW_Fdelta_s0, answerable/unanswerable question twins: 54 -> 48 of 75 under the strict grader, 75 -> 72 under the lenient one. Most of the strict-grader losses add extra words to a correct value (e.g. "freezer 482" instead of "482"). It still answered UNKNOWN on all 75 unanswerable questions.
- BW_Pa_v2_s1, abstention: on 5 of 75 unanswerable questions it gave an invented value instead of UNKNOWN. In one case it was the value from the paired answerable question. Abstention rate 0.933, vs 1.000 for the base model, Supa 1.7, BW_Fdelta_s0 and the RCWM T2 adapter. The guard does not gate this.
- Coverage. One training seed per adapter. Each adapter was tested alone, never both together, and only under the suite's own system prompt, not BWM's prompts. Three candidate categories (arithmetic word problems, counting/sorting, date/time) were excluded before the draw because the reference model scored too low on them. One difficulty level per category; all answers are 11 tokens or fewer.

Provenance: RET_RESULTS.json sha256 fae050afed436849d215eee1a5b8ad929f6e24f9c6c6098af6c8a5b49e4bf326; provenance.json sha256 4b4a925d58eeef78fce7a2d2cef6ecc81988a0ebd29bfd4da0b63036c85d4f73 (= DONE.json provenance_sha256); RET_FREEZE.json sha256 b32ef148...; draw sha256 864481a3adffbb90d620a3caea6bc37c6b1862f8b7f0338535affeb1e463b19b. Adapter files: BW_Fdelta_s0 2ed0bb0064bc...65ae / 3038c779...b3b8c; BW_Pa_v2_s1 ee1c5aec...8110 / f6a2c4ef...94bf. Run by RCWM. BWM recomputed it independently with its own graders and its own implementation of the rule: all 1077 compared fields matched.
