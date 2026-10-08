# Executive decision

**Build a consequence-conditioned decision model for the exact Rineya 0.6B-class checkpoint. Start with a small, independently testable predictor attached to frozen Rineya representations; advance to typed consequence and action slots inside Rineya only when the predictive representation improves real decisions.**

The motivating idea is to give actions a learned meaning in terms of what they cause. Instead of training a network only to output a scalar approval score, train it to predict a reusable representation of consequences. A separate goal readout determines which consequences are valuable. Attention can relate a proposed action to distant context, prerequisites, concurrent work and delayed outcomes. This is the user's central proposal, formalized here as **RCWM-001: Rineya Consequence World Model**. Earlier working names such as reward-world-model and reward-token Transformer refer to this same research line, not additional projects. [I03]

The research objective is not to demonstrate that numerical rewards can be embedded. It is to demonstrate that **learning and reusing consequence structure gives a better accuracy–cost–adaptation trade-off than a comparably powerful scalar evaluator**. The most persuasive result would be better action selection on new goals or unseen compositions while holding environment experience, model capacity, candidate plans and evaluation compute constant.

## The first decisive experiment

Use a deterministic text/symbolic environment with exact state snapshots, executable actions and an independent outcome checker. Collect multiple action branches from the same starting state. Freeze the Rineya language backbone. Compare a matched scalar Transformer, a measured-feature predictor and a learned-consequence predictor. Hold out complete problem families and goal combinations. Ask whether the learned predictor chooses better actions using fewer new reward labels, not whether its embedding plot looks attractive.

A successful first milestone is deliberately narrow: **on held-out parents, the consequence representation supports better goal-conditioned decisions at a measured cost and without changing the frozen language model.** A null result is informative. It may show that the task lacks reward diversity, the representation is insufficient, the proposed actions are poor, or scalar prediction is already adequate.

## What this program is, and is not

This is a research plan with an executable synthetic reference scaffold, not a delivered trained Rineya world model. No production checkpoint is included. No new Rineya capability gain, native GPU qualification, remote branch, paid job or external account action was performed for this bundle. The reference code checks selected mechanics in a small CPU environment. It does not implement all later research arms.

World-model learning and reinforcement learning are overlapping components, not competing taxonomies. A model of immediate rewards is a reward model; a model of policy-conditioned cumulative rewards is a value model. A predictor becomes a useful world model for this project when its action-conditioned consequences correspond to independently observable outcomes and improve control. Modeling Rineya's hidden-state changes alone is an internal computation model, which must be evaluated separately from an external environment model. Dreamer, successor features and trajectory modeling provide precedents, but do not establish that this Rineya adaptation will succeed. [S01, S02, S07]

## Five decisions to freeze before expensive work

1. **Subject:** the exact selected Rineya checkpoint, tokenizer, loader and executor versions; no substitution of a stock Qwen model or the older 344M student.
2. **Authority:** real environment feedback or an independent verifier defines correctness; predicted utility never grades its own success.
3. **Comparison:** the scalar evaluator receives the same context, action sequence and compute allowance as the consequence evaluator.
4. **Information:** predictions never receive future observations, actual future rewards or future actions chosen using hidden observations, unless those actions were explicitly fixed as an intervention.
5. **Promotion:** practical-effect, uncertainty, cost and retention gates are written before sealed evaluation.

The first three reading priorities are successor features for the factorization, DiSPO for distributions of achievable futures, and SF-GEN for the direct language-generation precedent. Trajectory Transformer and masked trajectory models establish that token-like trajectory representations and flexible inference masks are already viable design patterns. The Rineya contribution must therefore be a demonstrated transfer, composition, efficiency or preservation improvement—not a claim to have invented reward tokens. [S02, S04, S05, S07, S09]

# Scope, provenance and evidence discipline

## Rineya starting point

The September 30 research handbook identifies a historical experimental subject named `Rineya-Titan-Experimental-1.0.0-devfreeze`, with 596,049,920 base parameters. It also records that a laptop route passed short native mechanics checks but encountered a prefix-versus-bulk likelihood discrepancy during capture qualification. Those are historical report statements, not a fresh audit of the current repository. Stage G0 must re-establish the active checkpoint and current qualification state before collecting hidden representations. [I01]

The target deployment envelope is the user's RTX 4070 Laptop GPU with 8 GB of VRAM and constrained host resources. This report uses that envelope for design estimates; actual free memory, operating-system overhead, thermal behavior and current RAM configuration must be measured locally. Do not convert a short-sequence load success into a claim that long-context training fits.

Preserve existing Rineya acquisition and retention work as independent instruments. A base model can remain byte-identical while an attached planner degrades deployed behavior. Consequently, report both **weight preservation** and **system-level behavioral preservation**. Existing composable-skill reports specifically motivate paired evaluation, sealed splits and full resource accounting. [I02]

## Evidence labels

**Literature finding** means a claim supported by a named primary publication in its own setting. **Historical internal record** means a statement from a previous Rineya report, not independently reproduced here. **Derived example** means transparent mathematics or a constructed example. **Proposal** means a design, threshold or hypothesis requiring execution. **Reference mechanics** means code actually run in the supplied synthetic CPU scaffold. **Native result** is reserved for a measured experiment using the pinned Rineya checkpoint; none is claimed in this delivery.

The annotated register lists 43 primary papers, author project pages and official documentation references. Selected methods were examined for Dreamer, successor features and DiSPO; many surrounding papers were reviewed at the primary abstract or project-summary level. This is a targeted synthesis, not an exhaustive systematic review or a full audit of every cited codebase. Publication venue and version are separated from access date. Recent preprints are explicitly not treated as independent replication.

## Research authorization boundaries

This bundle authorizes no background job by itself. A later implementation handoff should state the repository, parent revision, permitted directories, model identity, data scope, compute budget and stopping rule. The proposed branch is `research/rineya-consequence-world-model-v1`; it has not been created by this deliverable. Synthetic tests may run without a checkpoint. Native extraction, training, model downloads, paid APIs, cloud rental and remote pushes need the user's applicable authorization and the repository's current policies.

The main scientific question should remain stable even if a particular implementation fails. Preserve reports and failing tests instead of deleting an experiment because it did not support the hypothesis. Conversely, architectural elegance is not a reason to continue spending compute after a well-controlled simpler baseline dominates.
