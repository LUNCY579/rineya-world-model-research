"""RCX-E07/E08 Blocksworld infrastructure (RGM-001 x RVM-001, CPU mechanism tier).

Modules:
  bwm_import  pinned, read-only, bytecode-free import of BWM's Blocksworld simulator (fail closed on sha mismatch)
  excl        BWM's exclusion recipe (scripts/bw_b3_prep.py::fresh + test_signatures) -> EXCLUSIONS.json, is_excluded
  space       exact state spaces (compact codes), exact h*(s) for all states by backward BFS, optimal-action sets
  data        fresh instances (RCX seeds 20261270-20261289), exclusion filter, labelled datasets split by instance
  encode      (state, goal) encodings: relational graph (GNN), flat padded vector (MLP), text (BWM rendering)
  models      RelGNN (R-GCN style, pure torch) and FlatMLP value models, seeded CPU trainer
  heuristics  unmet goal facts (BWM ranking), misplaced blocks, h_add, h0, oracle h*
  search      GBFS and beam search with exact dynamics (B.step), batched heuristics, B.execute validation

Nothing here writes inside the BWM bundle. torch is imported only by `models` (and by callers of it).
"""
