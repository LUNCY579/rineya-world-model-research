# Validation scope and known limitations

60 final CPU tests passed. Initial construction exposed a reserved-method name collision
(`type` versus `nn.Module.type`), fixed as `type_embedding`; final tests were rerun.

The tiny supervised feature demo lowers average check MSE, but underpredicts the safe prerequisite
chain's object-gained feature (0.063752 versus exact 1.0). This is a known modeling limitation,
not a test harness failure. It is preserved rather than hidden. No per-plan quality promotion is claimed.

No native Rineya checkpoint, GPU, long-context capture, calibrated uncertainty, learned VQ,
self-supervised outcome target encoder, production DAG execution or retention experiment was run.
The toy train/check sets share a very small state-space support. All research promotions remain gated.
