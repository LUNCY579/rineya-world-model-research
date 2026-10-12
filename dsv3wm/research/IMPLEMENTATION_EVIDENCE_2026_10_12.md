# Executed implementation and paper transfer — 2026-10-12 UTC

This addendum describes the implementation that actually trained. The earlier
`DEEPSEEK_TRANSFER.md` remains the original design review; its closing pilot
recommendation is not a description of the later native A1 architecture.
This document does not change registered inputs, outcomes or gate thresholds.

## H1: sequential local token supervision

DeepSeek-V3 uses sequential auxiliary modules with normalized previous-depth
representations, the next conditioning token's shared embedding, a projection,
an auxiliary Transformer block, and the shared output head. Its depth losses are
averaged and weighted; the auxiliary modules can be omitted at inference.
Those are language-token mechanisms, not demonstrations of world-state
prediction. [DeepSeek-V3 v2, Section 2.2](https://arxiv.org/html/2412.19437v2#S2.SS2)

The executed Rineya module in `src/dsv3wm/objectives.py` is sequential across
auxiliary depths, but uses a local residual bottleneck at each position:

```text
h_i^(k) = h_i^(k-1)
          + U_k SiLU(D_k [RMSNorm(h_i^(k-1)); RMSNorm(Emb(t_(i+k)))])
target at depth k = t_(i+k+1)
```

There is no auxiliary attention or auxiliary Transformer block. The backbone
produces the causal hidden states; the local modules consume preceding-depth
representations and teacher-forced preceding tokens, never their own target
token. Response-only windows are supervised only when every participating
label position is unmasked. Prompt-boundary and future-token intervention
tests check this dependency. This is a compact adaptation, not a reproduction
of V3's full module, independent parallel output heads, MLA or MoE.

The executed native A1 uses **two total predicted token positions**: ordinary
next-token CE plus one additional future-token depth. Hidden size1024 and
bottleneck rank16 give `3*1024*16 + 2*1024 = 51,200` auxiliary parameters.
The auxiliary weight is0.1; the shared embedding/output weights remain frozen.
Auxiliary gradients reach the unmerged LoRA through the causal hidden states.
The auxiliary module is absent during all state/planning inference.

Source SHA256: `00a7ce1271f0ae29894b0ee236e335581e78d22c70eaf61ac247b872e49f6973`.
Warm training is bound by `prereg/RECOVERY_TRAIN_FREEZE_V2.json`, SHA256
`0d52132ee1f39c3fc5aa4210066790031ab2b363ab0e871f2454175263cb3ab3`.
Actual native counts are in `results/recovery-v1/TRAINING_SUMMARY_01.json`:
A1 has10,143,744 trainable parameters,32 updates,466 ordinary and434 auxiliary
supervised tokens. Inference activates the same606,142,464 parameters as A0.

Four total token positions, shared auxiliary modules, shuffled auxiliary
targets and different weights have implementation tests. They have **no native
recovery ablation outcome**. A1 has no equal-auxiliary-capacity negative-control
measurement. Its extra supervision and training time must remain explicit.

## H2: separate complete response targets at state horizons

A2 uses the same frozen parent and registered unmerged F initialization as A0,
but supervises separate responses after action prefixes of lengths1,2,4.
The loss weights are1,0.5,0.5. Each input contains the initial state, rules and
ordered actions; it contains no intermediate truth states or goal. Training is
teacher-forced token CE on the full legacy state/legality/termination response.
The additional legality-head weight is0, while Boolean response tokens still
receive ordinary CE. This is not a standalone numerical state-vector loss.

Independent rollouts feed each predicted state into the next one-step query.
Direct future-state responses and own-prediction rollouts are scored separately.
They therefore test different mechanisms. Full-versus-delta/mixed targets,
an extra legality head and autoregressive rollout-loss variants have no native
recovery comparison; generic code support is not evidence that they were run.

The32 A2 updates use96 nominal decoder/objective forwards and17,716 input
positions. A5 uses96 one-step updates/forwards and12,768 positions. A2 receives
38.8% more input positions and takes28.9% more measured training time. Equal
nominal forwards do not make this an exactly matched compute or data contrast.
Backward recomputation and vocabulary-head work are not measured FLOPs.

## What the executed results establish

On the fixed24-parent development bank, A0/A1/A2/A5 h4 own-state accuracy is
24/22/24/23; direct h4 accuracy is0/24 for every arm. A1/A2 have complete direct
reply schemas on24/24 cases, while still predicting no accepted correct h4
states. Both lose the retained boundary-legality fixture. Neither passes the
fixed predictive gain gate. The inherited B0 rollout is already at ceiling.

These findings separate grammar improvement from demonstrated dynamics gain.
They do not reject the DeepSeek result, establish equivalence, or identify a
causal explanation for this pilot's failures. State-copy and action-shortcut
counts in `reports/RECOVERY_FAILURE_AUDIT_V1.md` are observational diagnostics;
new controlled interventions would be needed to attribute a mechanism.

## H3 and the conditional mechanisms

The new H3 pipeline uses learned B0 search on a fixed TRAIN-only task pool,
then independent exact verification, with hard negatives preserved outside
positive plan CE. A3/A4 warm A2/A0 and have real extra-one-step C3/C4 controls.
No teacher predictions or student training outcome exists yet: the first
attempt failed shared-lane admission before loading. This verifier pipeline
is our proposed world-planning adaptation, not a direct replication of V3's
reasoning post-training procedure.

H4 specialist routing has not been justified by complementary native specialist
measurements. H5 MLA/latent-history conversion remains unexecuted. Standard
six-token KV parity is a narrow engineering result; it does not establish
compressed memory, broad trajectory efficiency or a reason to alter the dense
backbone. Current integration is unqualified for every learning mechanism.
