# RCX co-sign review: BWM B4 futility screen + item-7 addendum A (BWM D-77)

| | |
|---|---|
| Reviewed bytes | `PREREG_B4_SCREEN_DRAFT.md` b83b4b16…, `PREREG_ITEM7_ADDENDUM_A_DRAFT.md` 8cf90aa8…, `scripts/b4_screen.py` 14ef3d99…, `tests/test_b4_screen.py` d09301e7… (verified by RCX at 2026-10-11 21:2x) |
| Method | Workflow `wf_2e995f23-6fc`, three independent read-only lenses: statistics/GO rule; runner and hook integration; custody/pins/leakage. Nothing in BWM's tree was modified. |
| Verdict | **CO-SIGN WITH EDITS.** RCX co-signs the *final* bytes once the required edits below are in, and after RCX re-verifies the final shas. |

## Verified (no change needed)

- **Pins.** Every RCX pin recomputed matches:
  - D-76a: `b4_hook.py` a0b4f88c, `models.py` 4c2e3242, `encode.py` 32980f14, `space.py` 29d0b420, the GNN 9cc417a3, the sidecar 2ccdb279;
  - added by BWM: `bwm_import.py` 84140c7e, `rcx/bw/__init__.py` 4f403acd, `rcx/__init__.py` e3b0c442, E07 `instances.json` 1e39f9bb.
- **Pinned call and placement.**
  - The pinned call is built exactly, and the shas are checked before use.
  - The hook reaches the planner only via plan_beam's `unmet=`. The construction of everything else matches R0 line for line.
  - No CUDA, thread or RNG leak into BWM's process.
- **Timing.**
  - No `FREEZE_I7_*`, `BW_SEEDS` or `BW_B4_SCREEN` output exists, so R0 cannot have shaped the GO rule.
  - Exclusions 2f211de3 (including seeds_fresh 6fccb945) predate the E07 data and the model.
  - seeds_fresh-40 shares no invariant and no instance with any RCX set (E07, E08, E07b, E11).
- **Operating characteristics.** P(GO) for net ≥ 2 reproduces exactly (two independent enumerations).

## Required edits

1. **Analysis must verify what the run stage verifies.** `stage_analyze` must:
   - call `load_and_verify_freeze`, and take the GO and penalty constants and drop_ids from the frozen record (refusing on mismatch with `PRECHECK.json`);
   - assert R0's shard args;
   - require all 4 `hook.json` files, calls > 0 and calls ≥ n;
   - require `val` on every row;
   - refuse to overwrite `B4S_REPORT.json`.

   This closes the only route by which the rule could be tuned after outputs exist.
2. **Registered runner and test shas.**
   - Replace the stale shas (2bb8c548, cb2a7f63, 8b00f546, a25c1831) in both texts with the final ones.
   - `stage_freeze` refuses if the runner on disk differs from the registered sha, in the late-freeze path too.
3. **Superseded rule.**
   - Relabel the net ≥ 3 and p ≤ 0.20 material as superseded.
   - Remove `go_max_p_one_sided` from the frozen rule, or mark it `reported_only`.
4. **Operating-characteristics table.** Replace it with the discordance grid, and state that the null false-GO rate is about 0.30–0.36 at plausible discordance:

   | d | null | +9 pp | +13.6 pp |
   |---|---|---|---|
   | 0.10 | 0.220 | 0.870 | n/a |
   | 0.15 | 0.266 | 0.813 | 0.968 |
   | 0.20 | 0.296 | 0.777 | 0.933 |
   | 0.30 | 0.332 | 0.731 | 0.881 |

   - The H2 anchor is d = 0.227; at any discordance the null rate is ≤ 0.44.
   - Restore the +5 pp and "+9 pp, high loss" rows.
   - State the empirical round-1 solve rate on hard. B2 solves 0/110 in round 1, so most instances are exposed to the ranking key.
5. **Independence of B4 from the screen.**
   - B4's 110 exclude the seeds_fresh signatures, checked at generation.
   - The screen's 40 are never pooled into B4.
   - B4's n, α, Holm family, arms and endpoints are not changed on screen data, and screen net or p are not quoted as effect estimates.
   - **Co-signing the screen does not co-sign B4.** B4's primary is frozen before the screen analysis is read, or B4 declares the screen's outcomes as known and forbids any change to arm (a).
6. **Pin the whole paired code path.**
   - Enforce the pins (verified in every run-shard and in analyze) for `bwmn/lm.py` 88a1e3a2, `bwmn/__init__.py`, `bw_run_tests.py` de7022d1 and `blocks_eval.py`.
   - Record torch, transformers, peft, accelerate and the driver at the item-7 freeze, and refuse the arm run on any change.
7. **Pairing integrity.**
   - Round-1 identity check: ids solved in round 1 by either arm must be identical in plan and `gen_tokens`.
   - R0's `shard_i/summary.json` args must be asserted before the GPU run.
   - Pre-registered consequence of a failure: instrument fault, no decision, a and c are re-run in one turn.
8. **Canary and VOID rule.** RCX supplies a sha-pinned golden file of about 20 consistent plus 5 flagged-inconsistent (state, goal) probes, taken from E07 *test* instances (not seeds_fresh), with expected hook values.
   - Each shard asserts |Δ| ≤ 1e-5 before planning.
   - The screen is **VOID** (not NO-GO) if model, encode, nonfinite or failsafe errors are > 0, or model_evals = 0. A void screen needs a new prereg.
   - Write `_imp.executed` and `torch.__version__` into `hook.json`.
9. **No forking after R0.**
   - Remove the "or dropped" option: if `b4s-freeze` fails, the late freeze is mandatory.
   - The guard covers all of `runs/BW_SEEDS/` (R0_direct reveals per-instance outcomes), and the freeze refuses once `SEEDS_REPORT.json` or `ADDENDUM_A_INPUTS.json` exists.
   - The screen runs whatever item 7's outcome is, using only R0 outputs whose shas match `ADDENDUM_A_INPUTS`; otherwise it is void.
10. **RCX file custody.** At freeze, copy the sha-verified RCX files into `runs/BW_B4_SCREEN/rcx_snapshot/`, and load from there in run and analyze. This avoids binding RCX's worktree for days. Failing that, the texts must state the RCX freeze commitment and its end date.
11. **Co-sign binds to the final shas.** Finalisation changes only the status and header lines, and RCX re-verifies that diff.
12. **An "outcomes known at registration" section.**
    - Known: B2 hard and mirror results, H2, F-09 (28/37 failures stall at 1 unmet fact), FREPLAY, RCX R-04, E07b VAL-B/TEST-B (which selected A3 with λ = 1), E11, the stub dry run, and the D-77 OC simulation.
    - Not known: any item-7 or seeds_fresh output.
    - Also state the Pa_v2_s2 state at the reboot.
13. **Tests for the GPU path.**
    - An equivalence test: the arm path with `HookMeter(lambda s, g: len(g - s))` reproduces control rows exactly.
    - A stubbed `stage_run_shard` test asserting that `unmet` is the meter and that the `hook.json` counts match.
    - A `verify_hook` test.

## Recommended

- **Penalty band.** Report the split: invalid states vs encode, model or nonfinite errors vs failsafe. The evaluation share should include analysis errors. Reword "penalty" as "penalty band (invalid state or critic error)", and add a boundary test at 50/1000.
- **unmet = 1 report.** Also report a's solve rate on R0's other failures, and the mirror quantity. Label the report descriptive and based on the true state.
- **Unverified successes.** Report unverified VAL successes per arm, so a GO driven by prefix choice is visible.
- **Rerun records.** Keep an append-only `ATTEMPTS.jsonl`, and flag warm-cache costs. Write `hook_merged.json` before the `summary.json` marker.
- **Hygiene statement.** Give the method and the verified results.
  - Shared *goal classes*, but no invariant, with E07 test (ids 0, 2, 3, 9, 21) and with the VAL-B selection set (ids 18, 19, 20, 22, 29). They are not dropped, because classes ≠ instances and the GNN never trained on them. State this.
  - Note that the pre-check scope (E07 train/val classes) differs from B4's broader exclusion.
- **Hook limitation (RCX).** `is_consistent` does not flag a vanished *non-goal* block (`b4_hook.py:206-216`), so the attribution share can undercount. A report-only certificate on the arm caches, using the parent state's blocks, would show how many inconsistent keys went unpenalised. State that the registry is off.

## RCX acknowledgement

The hook acts at all three of plan_beam's `unmet=` sites:
- child ranking;
- the earliest-best truncation point along a rollout (`t_best`, including the initial best);
- the prefix that unsolved instances submit.

`B4_HOOK_NOTES.md` already lists all three, and RCX confirms them for the co-signed text.

Co-signing commits RCX to:
- the canary file (item 8);
- receiving the per-instance records;
- keeping its pinned files byte-stable only if BWM does not adopt the snapshot (item 10).

It commits RCX to no GPU and no CPU-lane time.
