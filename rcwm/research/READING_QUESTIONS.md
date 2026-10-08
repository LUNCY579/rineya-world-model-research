# Questions to answer while reading

**S02 / successor features:** What exactly is linear in the reward? Which dynamics are shared? Where
does the continuation policy enter? What error assumptions are needed for policy-improvement claims?

**S04 / DiSPO:** What distribution is learned? Which outcomes are reachable under dataset support?
How is a chosen feature target turned into an action? What is lost by replacing the distribution with
its mean? Which pieces could fit a compact-language setting without a diffusion model?

**S05 / SF-GEN:** What is already established for language generation? Which part of the base model
is unchanged? Does the target capture executable environment effects or properties of generated text?
Which baseline would distinguish Rineya's proposed contribution from this precedent?

**S07–S09 / trajectory models:** Which variables are tokenized? Which are observed at inference?
How is planning performed? Can a future action selected by a behavior policy leak an observation?
Does the masking scheme represent joint outcomes or only marginal predictions?

**S14–S16 / representation discovery:** What prevents collapse? Which augmentations preserve
consequences? When are negative pairs actually equivalent? Does quantization preserve rare hazards?

**S22–S26 / model risk:** How does error grow with horizon/search width? How is support tested?
Does uncertainty predict failure? What independent measure distinguishes actual gain from proxy gain?

**S29–S34 / evaluation:** What is the independent unit? Which context skill is tested? Are goals and
branches grouped? What costs are excluded? What result would justify the smallest useful claim?
