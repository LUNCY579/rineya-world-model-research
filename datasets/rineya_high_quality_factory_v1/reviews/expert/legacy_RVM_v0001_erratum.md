# Independent RVM-001 v0001 metadata erratum

This new review finds six released records with a false `equal_length_routes` competency tag. I missed this factual tag in the earlier expert review. The student prompts and critic targets are correct. The original release and earlier receipts are preserved; no release bytes were edited.

Reviewed file: `data/RVM-001/releases/v0001/canonical.jsonl`, SHA256 `262d9bc470353c3aacb105b1bde2bb5b87f1e9771472966ab7f1c53ac672bac3`. Actual reviewer execution `/root/expert_reviewer` independently checked all60 released record hashes, visible tasks and full critic targets. No target mismatch occurred. This new review uses only public contracts/canonical rows and the expert's own verifier/path counter, with no producer/reference/oracle or other reviewer verdicts.

Affected families are `RVM-001-pilot-v2-p0000`, `p0012` and `p0024`, both `base-forward` and `base-reverse` renderings. Every exact released record hash and original reviewed candidate hash is recorded in `legacy_RVM_v0001_erratum.jsonl` and `legacy_RVM_v0001_metadata_audit.json`. These six new receipts are **REVISE for factual metadata**, with target_correctness5 and moderate `false_competency_tag` findings.

Each world has three nodes: start→middle→goal and middle→start. The goal has no outgoing edge. A goal-reaching sequence must take the sole two-step start/middle back cycle some number k times and then follow start→middle→goal. Hence move count is2+2k, and there is exactly one path of each possible move count. Repeated cycles create different-length routes, not equal-length alternatives.

- p0000: Fjord→Beryl costs2, Beryl→Dune3, Beryl→Fjord1, fuel12. Feasible goal paths have2,4,6 moves with costs5,8,11, one sequence each.
- p0012: Cedar→Fjord costs4, Fjord→Dune3, Fjord→Cedar1, fuel14. Feasible goal paths have2 or4 moves with costs7 or12, one each.
- p0024: Cedar→Aster costs3, Aster→Fjord2, Aster→Cedar1, fuel12. Feasible goal paths have2 or4 moves with costs5 or9, one each.

The independent memoized path counter exhaustively counts every feasible terminal-goal sequence for every length up to initial fuel/minimum positive cost, confirming the analytic proof. These are not equal-route worlds, even though duplicate submitted candidates legitimately form ranking ties. The distinction matters for competency stratification and route-tie evaluation; it does not change critic answers.

Recommended handling is an exact-identity exclusion or metadata erratum overlay plus a newly reviewed superseding metadata-corrected shard. Preserve original released bytes/history. This reviewer has produced evidence only and has not edited release/export files or authorized release.
