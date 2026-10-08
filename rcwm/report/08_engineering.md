# Resource engineering for an 8 GB laptop

## Memory arithmetic before kernel tuning

For the historical 596,049,920-parameter subject, BF16 weights alone require approximately 1.11 GiB. A rough 16-bytes-per-parameter full-training accounting gives approximately 8.88 GiB before activations, attention workspaces and runtime overhead. These are derived storage estimates, not measured device usage. They explain why full fine-tuning is not the first route in an 8 GB envelope.

A small sidecar can be trained from cached frozen representations with no Rineya weights resident during that training phase. Native adapters reduce optimizer state but still require activations for gradient propagation. Quantization can reduce frozen-weight memory, but does not eliminate context activations, and the exact quantized route must pass capture and behavior parity checks. [S27, S28]

Caching strategy matters. A single BF16 vector of width 1,024 for each of 100,000 events occupies about 195 MiB. Saving all 28 such layer vectors would use about 5.34 GiB before metadata. Saving every token of every context is much larger. These are illustrative dimensions; the actual checkpoint configuration remains authoritative. Cache selected action-boundary summaries first, and use chunked files rather than loading the entire corpus into host memory.

## Three operating envelopes

**CPU mechanics:** no Rineya checkpoint, tiny synthetic states, short horizons and small Transformer. This is the supplied reference route. It validates accounting and interfaces, not language intelligence.

**Frozen capture plus sidecar:** load the qualified Rineya route only to extract bounded representations; flush cache shards; train the small predictor separately. Begin at 512 tokens and one candidate microbatch. Increase context or candidate count only after measuring peak memory and timing.

**Native adapter experiment:** register trainable modules, enable appropriate checkpointing, use a microbatch of one if necessary, accumulate gradients and remeasure the exact precision/cache/mask combination. Increase one dimension at a time. A previous short forward pass does not authorize an 8,192-token backward pass.

Reserve explicit VRAM and host-memory headroom, and stop before the operating system enters uncontrolled paging. The suggested native configuration sets a conservative peak-allocation admission target rather than assuming the whole advertised 8 GB is available. Reserved-memory and process-tree-RAM checks remain necessary. Guard against thermal throttling when comparing runs separated in time.

## Numerical qualification

Before capture, compare single-prefix and bulk causal predictions at the same token positions. Check padding side, position IDs, attention masks, cache use, dropout mode, precision and fused-kernel settings. Compare both selected hidden vectors and raw next-token log probabilities where accessible. A small score discrepancy can change candidate ranking near a decision boundary.

Test save/resume continuity, immutable base hashes, optimizer-step counts and accumulation scaling. Ensure that a skipped or failed batch does not silently count as an update. Store maximum errors and the declared tolerances, not only a pass/fail word. The historical Rineya capture issue makes this gate especially important, but the report does not assume that the historical issue remains unresolved today. [I01]

# Preservation and integration with existing Rineya work

## Keep the base anchor and the scientific question separate

The consequence model is a shared interface that can serve several existing directions. It does not replace their controls. Hidden-state research supplies candidate representations; TAPS/CDE supplies possible compute-allocation actions; CSS and retention work supply adapter and preservation mechanisms; post-training supplies trajectories and optional policy updates. Integrate through contracts and shared logging rather than immediately merging every experimental branch.

Adaptive recurrence is a later action family because it requires a reliably executable recurrence mechanism and a measured benefit. A world model cannot rescue a recurrence operator that does not produce useful state changes. First compare the actual effects of additional recurrence against ordinary extra tokens or a matched tool action. Then ask whether the consequence predictor can choose when to use it.

For skill composition, keep acquisition, old-skill retention and joint execution separate. A world-model adapter may retain old answers but fail when two skills must interact in one plan. Evaluate held-out compositions and real action sequences, not only separate skill test scores. Low cosine similarity between parameter deltas does not certify functional independence. [I02]

## Native mode isolation

A `language` mode should preserve the original prompt path, tokenizer, model weights and output head. A `consequence` mode adds the registered projections/adapters and heads. Unit-test that disabling the new mode restores the original forward behavior within the required tolerance. If a shared adapter changes normal language behavior, report it as a changed model, not an unchanged base because only a small parameter fraction was trained.

Do not merge an adapter or delta into the anchor merely to simplify packaging during research. Persist the anchor identity, adapter identity, target modules, precision and merge recipe separately. Quantized or low-precision deployment can alter small updates; qualify the artifact that will actually run, not only a higher-precision training copy.

## Teacher and data reuse

Existing post-training traces can be useful only when their action boundaries, executor semantics and reward provenance are known. Free-form reasoning text is not automatically an executable trajectory. Tag unsupported retrospective labels as such. When teacher-generated actions or labels are used, count teacher cost and retain the possibility that multiple models share the same systematic error.

A distilled planner should be judged against real outcomes, not merely its ability to reproduce a teacher's preferred plan. Retention replay must come from allowed training data. Old sealed suites remain evaluation-only, and current experiment failures must not be inserted into training while preserving the same claimed test split.

# Engineering, safety and release controls

## Repository organization and interfaces

Use an isolated experiment package with typed contracts, environment adapters, representation models, utility heads, planners, data loaders, evaluators and tests. Connect to the current repository through a narrow `FrozenHistoryEncoder` protocol, rather than importing unverified native internals throughout the experiment. A later integration pull request should list which source files change, why they change and how the ordinary language route remains covered.

The bundle includes standalone schemas, proposed configurations, a machine-readable experiment registry, acceptance templates and an implementation handoff. The reference package is intentionally smaller than this target architecture. Its implementation-status table identifies missing native, self-supervised and production capabilities. Do not interpret a directory name or a passing synthetic test as implementation of the entire research program.

## Data and checkpoint reproducibility

A run manifest records code revision, dirty-tree state, checkpoint and tokenizer hashes, environment version, split manifest, feature schema, action registry, mask mode, seed, hardware/software versions, precision, optimization parameters, effective batch, costs and artifact hashes. Record both requested and actually executed budgets. Save atomic checkpoint files and their dependent normalizers/readouts together.

Deterministic seeds are useful, but bitwise equivalence across different devices or kernel versions should not be promised. Pin the tested environment and quantify tolerances. Re-run an independent reproduction before an important promotion. The supplied validation results describe this container's CPU environment only.

## Safety boundaries for learned utility

Tool permission and irreversible-action checks occur before learned scoring. The model cannot authorize file deletion, payment, account changes or external communication. Use mocked tools or a sandbox environment. Constraint violations are separately reported and cannot be averaged away by a high task reward.

Test malicious instructions embedded in retrieved documents, forged tool outputs, stale evidence, missing prerequisites, repeated no-op loops and candidates designed to exploit the scorer. Do not execute model-generated code without a bounded sandbox. Tool outputs are data; they must not change the source checkpoint, evaluation split or logging policy.

## Release gates and honest claims

A research artifact may be released with a negative result. It may not be labeled a trained Rineya world model without a pinned native checkpoint evaluation. A model card should state training data, intended tasks, failure modes, prohibited actions, calibration limits and whether goals are in-span or extrapolative. A dataset card should describe labels, provenance, privacy, split grouping and license review.

Third-party papers and code are linked through the resource register rather than bundled indiscriminately. Verify the exact repository license and revision before copying code or downloading datasets; a paper's open-access status does not determine its software or dataset license. The ZIP contains original planning material and original reference code, not a redistribution of the cited implementations.
