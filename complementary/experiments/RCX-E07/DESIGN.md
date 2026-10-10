# RCX-E07/E08: relational value learning for Blocksworld planning (RGM-001 × RVM-001; CPU mechanism tier)

**Status:** DRAFT. This work starts after the first validated E01 milestone.

**Bundle mapping:**
- E07 "relational representation comparison";
- E08 "learned heuristic with exact dynamics";
- it also seeds RVM E06, a value/dead-end critic for beam ranking.

## Why this is complementary to BWM

BWM plans *in imagination*: an LM proposer plus an LM forward model F, beam-ranked by a hand-coded heuristic (unmet goal
facts, then prefix length). Its 37 unsolved hard-110 instances were all unverified: 23 `dead_end` and 14 `budget`.

BWM has no learned value, critic or successor representation (survey, `docs/understanding/bwm_program.md`). A value
estimate that generalizes *structurally* to more blocks than seen in training is exactly the missing component. The
relational question is which representation gives that structural generalization:
- graph (GNN over objects and relations);
- flat (MLP on a fixed-size encoding);
- sequence (the character transformer from E01 reading BWM's `state_json` text).

## Design

- **Simulator.** BWM `bwmn/blocks.py`, imported read-only from this branch's `bwm/code` copy. Its sha `b1e28ed5…` is
  identical to BWM's live copy. RCX writes nothing into BWM trees.
- **Instances.** Fresh, from `blocks.generate(rng, n)` with RCX seeds `202612xx`. Instances whose `signature()` hits
  BWM's `test_signatures()` or `hard_signatures()` are excluded, and so are BWM's large-n invariants.
- **Labels.** Exact optimal cost-to-go h*(s, g) from BFS, plus dead-end and unsolvable flags where relevant.
- **Representations at matched labels.**
  - **Graph:** a typed relational GNN with nodes = blocks plus a table/hand node and edges = on, clear, holding, plus
    *goal* edges (goal-on, goal-ontable).
  - **Flat:** an MLP on a padded n_max × n_max adjacency encoding, with a size cap.
  - **Sequence:** the character transformer on the BWM text rendering.
  - **Canonical sequence:** a renaming-canonicalized text rendering.
- **Training range.** 3–6 blocks. **Test range:** 7–12 blocks (structural OOD), plus held-out names and shuffled fact
  orders (surface).
- **E07 endpoints.**
  - Exact h* accuracy and mean absolute error by block count.
  - Ranking quality: Kendall τ between predicted and true h* across successors.
- **E08 endpoints.** Solved within a fixed expansion budget on fresh 8–15 block instances, using:
  - greedy best-first search or beam search with exact dynamics;
  - learned h compared with (a) unmet-goal count (BWM's ranking), (b) h = 0, (c) a classical h_add-style heuristic, and
    (d) the oracle h* ceiling.

  Plan-length optimality gap is secondary.
- **Bridge to BWM (later, read-only).** `plan_beam(..., unmet=learned_h)` with exact F (tier B, specialist) and later
  with BWM's frozen LM F.

## Claims this cannot make

It cannot make any claim about Rineya-core reasoning. A learned heuristic is an external specialist (tier B) unless it
is distilled into the dense student (RSM/E31).
