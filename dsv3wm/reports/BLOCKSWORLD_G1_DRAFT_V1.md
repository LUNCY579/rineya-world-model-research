# Blocksworld inherited-interface G1 draft v1

Date: 2026-10-12 America/Chicago. This is a prospective constructed validation
instrument; no native outcome is reported. Root must freeze this draft, source,
generated cases/config/manifest, tokenizer identity and initializer hashes before
execution. It does not authorize training, promotion, old-bank access or a
teacher-unseen generalization claim.

Use private native `Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_Fdelta_s0/adapter`
on the verified frozen Titan-Sparse parent. Weight SHA:
`2ed0bb0064bc6b11ec9c661840a38094e468462cea2ff833cda3bf234ead65ae`;
config SHA `3038c77961f33461e1bef736959765434c0be473d991b0ef92c4813d269b3b8c`;
parent SHA `a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284`.
Load unmerged FP32, TF32 off, four CPU threads, existing exclusive lane/RAM
guards and standard qualified BWM KV decoder. No model parameters change.

`build_block_g1.py` constructs twelve primary one-step cases: six legal actions
that change state and six illegal actions that leave state unchanged, covering
pick-up/put-down/stack/unstack at three through six objects. All twelve initial
parents differ modulo object renaming. Four additional paired diagnostics rename
objects or reverse atom-list order; they reuse primary parents and never enlarge
the independent primary denominator. No goals, solver plans, historical generator,
test signatures, banks or cached model outputs are read.

Primary prompts and labels reuse `blocks.forward_prompt_delta` and
`forward_target_delta` byte for byte. The inherited reply schema is exactly
`{"add":[fact strings],"del":[fact strings],"legal":boolean}`. BWM
`parse_forward_delta` is called unchanged and its raw result is recorded separately.
Termination is absent from this reply/domain; decoder EOS is not a predicted
world termination flag. Name/list-order prompts are explicitly named diagnostics.

Tokenizer-only prospective construction counts every target plus EOS and chooses
generation cap `ceil8(max target+EOS+16)`; context cap is
`ceil64(max prompt+generation cap)` bounded by512. The exact measured token counts
and selected config are written before native execution. No cap changes after
grammar outcomes are seen. Actual numeric magnitudes do not apply to symbolic
blocks; the object-count bound is six. Typed config forbids boolean caps or a GPU
fallback under this CPU G1. The constructor may run tokenization but never a model.

The current pure tokenizer check found max149 prompt tokens and max26 target
tokens including EOS, selecting48 generated tokens and context cap256 by that
fixed rule. These are source-fixture/tokenizer measurements, not native outcomes;
the builder records them again in the actual prospective input manifest.

All sixteen raw generations must finish and be persisted before the exact simulator
scores any action. No simulator, applicability check, state repair or returned
score influences generation. Failures/abstentions remain in the denominator.

Report for every case: raw text/JSON, typed raw legal flag, raw add/delete lists,
naive candidate fact set **before** deletion-support or structural checks,
unsupported deletions, type/format failures, structural validity, and the unchanged
source parser's acceptance/state/flag. Unsupported subtraction can accidentally
produce a correct naive set; that never turns it into an accepted faithful delta.
Avoid lossy `state_json` conversion of inconsistent raw facts that could drop
extra holdings and conceal an error. Strict duplicate/type/vocabulary diagnostics
can refuse outputs the permissive source parser accepts; record that distinction.

Accepted correctness requires complete typed format, supported deletions and
structurally consistent state plus the original parser's answer. These symbolic
checks are explicit and are not learned rule knowledge. Separately report raw
legal accuracy, raw delta exactness, prefilter candidate-state equality and the
legal/illegal strata. Existing P-15 shows illegal raw flags can be wrong while a
support check protects downstream planning; qualification must not hide that.

The proposed operational G1 gate is >=11/12 well-formed primary replies,
>=10/12 accepted `(next facts,legal)` tuples and >=5/6 accepted legal-changing
tuples. Raw legality/support dependence is mandatory descriptive evidence,
never silently substituted by the filtered score. A pass qualifies this tiny
inherited interface only; a failure is preserved and needs a new amendment before
changing cases, gates, schema or budget. No efficacy/retention pass follows.

Independent-parent interpretation is honest about the finite graph pool: valid
unlabelled initial states have p(n)+p(n-1) tower/held configurations. A bounded
new <=6 permutation canonicalizer is isolated in this collector; frozen
domains.py's <=5 canonicalization stays unchanged. These constructions may
overlap inherited training exposure, even when names differ. Neither new names
nor a new file establishes teacher-unseen states. Large128/24 quotas and paired
benchmark significance claims are outside this G1.

Root reproduction sequence: construct a new dataset directory using
`build_block_g1.py --output <new path>`; prospectively bind its config/cases/manifest,
both new scripts, this draft, reused source files and the exact private initializer
in a new freeze; then call `evaluate_block_g1.py --config <config.json> --cases
<cases.jsonl> --manifest <manifest.json> --freeze <new freeze> --out <new result>`.
Every output must be new. Models were not loaded or executed while writing this
draft or its pure tests.
