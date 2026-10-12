# Amendment 2: standard cache feasibility after the uncached A0 result

The original A0 generation screen is preserved: 0/2 correctness at all horizons,
all malformed, 635.75 seconds, 12 transition queries, 768 output tokens. No
native efficacy conclusion follows from four optimizer steps and this new
response format. This result was seen before this amendment; subsequent cache
measurements and a chat-format diagnostic are exploratory development evidence.

`bwmn.lm.sample_crn` already implements standard FP32 KV cached greedy generation
with `logits_to_keep=1`, EOS accounting, causal positions and resource/thermal
guards. Reuse that existing decoder explicitly rather than reproduce full
prefix/head computation at each token. No MLA conversion, quantization, latent
memory training, prefix compression or adapter merging is introduced.

Before comparisons, `cache_probe.py` checks cached versus full-prefix next-token
logits (rtol/atol1e-4), eight successive argmax decisions and greedy text on a
fresh original-format native fixture, and records latency. Its scope is narrow;
it is not a universal cache-parity guarantee. Preserve failure if it does not
pass; do not silently use cached decoding.

HF cached/full greedy raw token IDs must match; BWM greedy text, actual token
count and EOS status must agree. A passed probe's file/source hashes are required
by the cached CLI. A new source/config/data/amendment freeze binds this instrument.
Configuration additionally rejects boolean token horizons; selected integer
pilot configurations and all trained objectives are unchanged. Original v2
reproduction is pinned to its archived commit; latest runs verify the v3 freeze.

Then evaluate A0/A1/A2/A5 on the SAME first two register dev parents at horizons
1/2/4, plain conditioning identical to v1, greedy max64 tokens, standard KV.
All comparative arms use the same cached instrument. Preserve the original
uncached A0 result as a separate engineering reference. Record raw outputs,
query/decoder counts, processed-token positions, and analytic resident cache
bytes separately from measured GPU allocation (none on CPU). Do not equate
this standard-cache improvement with a DeepSeek/MLA effect.

A separate **post hoc chat-format diagnostic** uses the existing non-thinking
Titan chat prefix with the same operative rules/state/actions and no labels.
It changes inference conditioning without retraining and must be clearly named;
it does not enter a matched DeepSeek comparison or confirmation result. If
grammar failures dominate, that identifies inadequate operational training or
representation alignment, and supports a NEW development protocol before
larger training. No confirmation labels may be accessed.

Planning remains the one-parent register feasibility screen with common upper
caps, exact scoring only after selection. Standard-cache processed tokens are
bounded conservatively by prompt+decoded tokens per query, including the custom
decoder's unused last forward when capped without EOS. Costs are explicitly
different from the original no-cache formula. Retention remains a six-fixture
loss-only diagnostic with null retention-pass status.
