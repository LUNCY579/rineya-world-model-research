# RCWM CPU reference mechanics

This is original research scaffolding, not a native Rineya implementation or a production library.
It models a short action sequence with a tiny causal Transformer, a learned latent bottleneck,
and supervised **measured** consequence features. A separate linear readout changes utility
without changing the predicted physical consequences. The codebook utility is train-only k-means,
not a trained VQ model. The event mask utilities are not a scheduler.

## Run

Use a virtual environment. Install a suitable PyTorch build for your platform explicitly; the
included run was CPU-only. From this directory:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python -m rcwm.demo --out ../validation/reproduced_demo --steps 250 --seed 17
```

The broad dependency ranges describe the intended interface, not universal compatibility.
See `../validation/environment_observed.json` for versions actually tested. No weights are
fetched and no network or external account actions are performed by the package itself.
Installing dependencies can access package repositories. The supplied checkpoint is a tiny
synthetic demo artifact only; no Rineya weights are supplied. Load checkpoints only from a
trusted local source using `torch.load(..., weights_only=True)`.

## Implemented

Causal action-token prediction; finite observed-return accounting; explicit utility readout;
parent-stable splitting; strict precommitted-plan checks; basic error/utility metrics;
train-only clustering; information-availability helpers; test coverage; CPU demo.

## Not implemented

Native Rineya loading/capture, self-supervised outcome target encoder, VQ training, learned
natural-language goal mapping, uncertainty calibration, joint stochastic futures, production
DAG scheduling, imagined RL, policy optimization, long-context qualification, visual world
modeling and actual retention evaluation. These are specified in the plan, not disguised as
finished work behind placeholder functions. `FrozenHistoryEncoder` is an integration protocol.

## Scientific limitations

The synthetic train/check samples share a very small state-space support. The demo checks that
training runs and useful feature prediction is possible; it is not evidence of semantic discovery,
out-of-distribution transfer or Rineya capability. Reported timing is container CPU timing only.
Boolean masks in this package follow `nn.Transformer`: True means blocked. Do not pass them
unchanged to an API using opposite semantics.
