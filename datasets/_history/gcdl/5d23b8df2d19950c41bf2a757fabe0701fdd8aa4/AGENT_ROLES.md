# GCDL-001 roles and review custody

11 October 2026. The environment supports four concurrent agent slots, including the coordinator. Eight scientific responsibilities are implemented across parallel work and explicitly separated sequential reviews; there are not eight simultaneously running independent agents. Shared filesystem, model lineage and conversation context limit independence. Separate executable implementations provide stronger ground-truth evidence than repeated model agreement.

## Responsibility map

| Role | Responsibility | Initial assignment / custody |
|---|---|---|
| A — Research Architect | Current RCWM/BWM/RCX evidence, hypotheses, split/evaluation design, claim limits | Architecture agent; `RESEARCH_PLAN.md`, this file, architecture report and source snapshot |
| B — Environment and Oracle Engineer | Executable transitions, action legality, goal/cost formulas and primary evidence | Oracle agent; standalone environments and primary oracle; never native training jobs |
| C — Curriculum and Generator | Fresh states/rules/goals/plans, curriculum quotas and semantic variants | Generator agent; no authority to bypass verifier gates |
| D — Decision Supervision | Ranking/ties/preferences, probability semantics, search proof status | Oracle/generator contract stage; labels validated separately by F |
| E — Adversarial Auditor | Shortcuts, rule reversal, option order, semantic duplication and perturbation consistency | A fresh review context after generation; cannot approve its own examples |
| F — Quality and Integrity Reviewer | Separate checker, schema/split/hash/reproducibility review; PASS/FAIL or quarantine | Independent checker and a review context that did not author generator/oracle implementation |
| G — Operations Engineer | Bounded CPU worker, queue, storage, checkpoints, runner, Git backup and recovery | Lead coordinator; cannot change failed scientific gates to keep production running |
| H — Improvement Coordinator | DEV failures → frozen hypothesis → paired comparison → promotion/rejection | Architect/lead coordination; E/F challenge proposed gains before promotion |

## Initial four-slot schedule

During construction the lead/G, A/H, B and C/D work concurrently on disjoint owned files. The lead integrates after interfaces agree. When a construction slot completes, reuse that slot with a fresh, narrower context for E and F review. Supply the specification and candidate artifacts, not an instruction to approve or the author's desired verdict. After corrections, rerun executable gates and obtain a distinct final review. A reviewer who authored the relevant implementation must disclose that conflict; their review is not described as independent.

The host's asynchronous unattended worker is a deterministic CPU program. It executes generation, two implementations of verification, integrity checks and audited configuration changes. It is not eight LLM agents thinking forever. Service/session availability determines whether fresh E/H agent reviews run. When no fresh reviewer is available, retain the exact-check evidence and disclose that the next independent agent review is pending; do not invent a review signature or claim continuous AI self-teaching.

## Handoffs and immutable evidence

1. A pins the specification, source context and evaluation boundary. D names each target, tie rule and probability/proof scope.
2. C submits a task with generator/config hash and seed. B executes the original inputs and candidate plans; neither rewrites a candidate to repair a failed label.
3. F's independent checker reconstructs transitions/scoring from the public specification. Shared data classes or serialization can be used, but shared transition/scoring helpers must be disclosed and do not count as independent mathematical validation.
4. E checks semantic twins, reverse rules, names, field/list/option order and shortcut floors. A representation change preserves labels only when its inverse semantic mapping verifies that condition.
5. F checks every record mechanically, then audits accepted parent groups with a recorded sampling seed. Verdict cites artifact/code hashes, checks, counts and observed failures. Any oracle disagreement is quarantine until investigated.
6. G promotes an immutable shard only after gates pass, checkpoints and writes its manifest. H consumes TRAIN/DEV summaries, proposes one bounded change and compares it against the frozen baseline. F/E validate promotion evidence.

Each review receipt identifies actor/context, role, code and artifact hashes, tested scope, findings, unresolved limits and verdict. Agent consensus never overrides executable disagreement. Contributing authors may fix their own work, but they do not sign the independent final PASS for that work.

## Sealed evaluation boundary

The generator receives recipes and protected split assignments; H sees only TRAIN/DEV quality and coverage. SEALED labels are read only by a dedicated mechanical audit/evaluation path. The lead can technically access all local files, so this is procedural custody. No secrecy/security claim is made. An audit may report schema, agreement, group-overlap and hash counts without exposing per-item gold. Once a future model's sealed per-item results are inspected and used to adapt the recipe, the instrument is spent; reserve a new independently minted confirmation.

Original RCWM/BWM/RCX reserved banks, private wording pools, retention confirmations and model artifacts remain owned by their programmes. GCDL uses fresh examples and does not operate their queues, leases or CPU-lane files. Exclusions should use allowed signatures/metadata, never open protected answers to mine curriculum examples.

## Authority and escalation

F can quarantine and block promotion. G pauses production at the configured free-space/storage/time/failure limits and persists resumable state. H cannot relax critical correctness/leakage gates to improve throughput. Source/config changes create a new hash and require validation; the worker cannot silently switch versions mid-batch. Native training, paid services and model-weight downloads are outside this programme's current authorization. Long-term persistence and Git retry policy are documented by G in the startup/status materials.
