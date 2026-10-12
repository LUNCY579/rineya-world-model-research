# FREPLAY independent second implementation (D-61/D-71 practice; PREREG_FREPLAY §8)

Run 2026-10-11 20:27:21 CDT. Script: `runs/FREPLAY/independent/freplay_indep.py` (sha256
`274a293806bb610688330949680abff36635243e2a7350669d203fd5ff6a7cc9`). It took 6.6 s on the CPU. No model was loaded:
torch False, bwmn.lm False.

## Result

| Comparison | Result |
|---|---|
| Registered `core` sha256 (`runs/FREPLAY/FREPLAY.json`) | `9413d13c18f40cac56049926b47839bbe2d468a92fc8fc1c4158779628c305f8` |
| Independent core, own label strings | `466234178d2989db42914e889153e661f268fa115755fcddbbcb67a4cae3bfb5`: differs in 3 label fields only |
| Independent core, with the 5 registered R2 `claim` label strings substituted | `9413d13c18f40cac56049926b47839bbe2d468a92fc8fc1c4158779628c305f8`: **byte-identical** |
| Computed fields that differ (out of 43,676 leaf fields in the registered core) | **0** |
| `key_id` convention (`sha256(canonical({"a","s"}))[:16]`) | reproduced on all 8 registered harmful keys |
| Inputs verified before reading (prereg §1 sha256 table, outcome files, PDDL dir digests, blocks.py) | 29 fcache files plus all others: OK |

The 3 label differences are quote style only. The independent script took the claim strings from the prereg §7 table,
using double quotes, while the registered run uses single quotes (`/sets/{b2_hard,h2_hard,distill_r1}/R2/claim`).
These strings are labels, not computed values.

Every count, rate, Clopper-Pearson bound, stratum (set × blocks × action), per-instance forest record, R1 bound, R2
reading and coverage check agrees exactly. The independent script computed the Clopper-Pearson bounds with
`scipy.stats.beta.ppf` directly, while the registered script uses its own implementation.

## Independence and limits

- **How it was written.** The script was written in a fresh agent context. Its only sources were:
  - the prereg text (sha 994d74c0…);
  - `bwmn/blocks.py`;
  - the pinned `bwmn/blocks_planner.py`, read for LMForwardBW's cache loading only and not imported;
  - the input file formats.
  `scripts/freplay.py` was neither read nor imported.
- **Same model family.** It is the same model family as the first implementation, not a different team.
- **What it took from the registered output.** It read only the schema (field names, list order) and the label
  strings, so that a byte compare was possible. It also matched the `key_id` convention against the registered
  harmful keys before using it; that is an identifier, not a computed result.
- **Branches the data exercised.** Agreement is shown only on these. The keys fall into 4 subtypes: ok_legal_exact,
  ok_illegal_flag, fi_flag and false_legal_rejected_by_support_check. The other 6 subtypes have 0 keys in both
  implementations.
- **Conflict rule never tested.** There were no cross-file text conflicts, so the §2 conflict rule was never exercised.
  There were also no torn lines.

Outputs:

- `INDEP_CHECK.json` (comparison record);
- `INDEP_CORE.json` (the independent core with its own labels; sha256
  `466234178d2989db42914e889153e661f268fa115755fcddbbcb67a4cae3bfb5`).

The registered `FREPLAY.json` and `FREPLAY.md` were not modified. FREPLAY.md's line "Independent second
implementation … not yet done" was true when it was written, and this file supersedes it.
