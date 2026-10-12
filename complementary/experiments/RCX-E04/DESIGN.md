# RCX-E04: an independent legality critic tested on held-out error producers (RVM-001; CPU mechanism tier; tier B specialist)

**Status:** DRAFT design, written 2026-10-12. Nothing has run: no code, no smoke run, no outcome. No prereg is frozen.
The only data access so far was read-only counting of BWM files that are already released to RCX, using
`python -B` for under 5 s on 1 thread (§5). No model or tokenizer was loaded, nothing was trained, and nothing was
written outside this file.

- **Track:** RVM-001, bundle E04 ("Independent validity critic: held-out error producers and near-miss labels"),
  roadmap P0-c `RCX-E04`.
- **Depends on:**
  - CX Q-04 to BWM (§6.5), for the primary test pool;
  - E07 code: `RelGNN`, the encoders, `bwm_import` and `excl`.
- **Lane:** one CPU job of at most 60 minutes. It runs between or after E01R blocks, and E01R has priority (§7).
- **Planned files (none exist yet):**
  - `PREREG_E04.md`
  - `run_e04.py`
  - `rcx/bw/critic.py`
  - `tests/test_e04.py`

## 0. Summary

E04 asks whether a small critic C(s, a) can flag invalid Blocksworld steps from a producer it has never seen. The
critic is CPU-trained only on RCX procedural steps, labelled exactly by `traces.bw_parts`. It is *claim-blind*: it
sees the state and the action, never the producer's output.

- **Primary held-out producer:** BWM's LM proposer `BW_Pa_v2_s1`, as it was queried during the B3 distillation search
  (`distill_r1`): 15,030 unique keys, 1,670 of them illegal.
- **Fallback held-out producers:** F0 and FK on the 240 D-54 key-order keys (2,400 claims, 586 invalid).
- **Primary comparison:** the incumbent legality signal on the same keys, which is the forward model F0's own `legal`
  flag (recall 0.780).
- **Strong baselines, all reported:**
  - a non-learned, domain-agnostic delta-support check;
  - a symbolic precondition inducer that uses no error labels;
  - the exact-validator ceiling.

**Expected outcome, stated up front.** Blocksworld legality is decided by one to three facts. The likely result is
therefore that relational, lifted and symbolic critics are all near ceiling. The decision then turns on:
- whether the operating point transfers to the new producer ("calibrated");
- whether the critic beats F0's flag;
- which qualifying module is simplest.

The main deliverables are a qualified held-out-producer battery and a reference critic for joint item 6, plus a
replay estimate of how many F calls a pre-filter would save. A planner-level gain is a separate, co-signed follow-up.

## 1. Hypothesis, and why it complements Rineya

### 1.1 Definitions

- **Step.** A pair (s, a): an exact Blocksworld state s and an action a emitted by a producer.
- **Invalid step.** a is not executable in s: `traces.bw_parts(s, a)[0]["ok"] is False`. This is identical to
  `bwmn.blocks.step(s, a)[1]` and is checked for every row (§4.4).
- **Claim.** A triple (s, a, y), where y is a forward model's answer text `{"add", "del", "legal"}`.
  - The claim is invalid exactly as in BWM's registered FREPLAY taxonomy (§3 of `PREREG_FREPLAY.md`):
    false-illegal, false-legal (accepted, or rejected by the support check), wrong state, or malformed.
  - Using the same taxonomy means the labels share one definition with BWM, as BWM required in CX Q-02.
- **Claim-blind critic.** A critic whose input is only (s, a): canonical parsed facts plus the action. It never sees
  y, the rule text, the goal or the producer's identity, so it cannot learn producer identity by construction. The
  bundle's concern was that a critic could learn "producer identity" instead of validity.

### 1.2 Primary hypothesis H-E04 (exact and falsifiable)

**The critic.** C\* is a claim-blind relational critic: RelGNN, about 124k parameters, three training seeds. It is
trained only on RCX procedural steps:
- uniform-illegal and edit-distance-1 near-miss producers (§2.2);
- on fresh RCX instances with n = 3–10;
- labelled by `traces.bw_parts`;
- with no LM output of any kind in training.

**The held-out producer.** The illegal actions that BWM's LM proposer `BW_Pa_v2_s1` emitted during the B3
distillation search:
- `distill_r1`: 15,030 unique (s, a) keys, of which 1,670 are illegal;
- 240 problems, n = 6–10;
- queried through the frozen B2 planner with F = `BW_Fdelta_s0`.

**The prediction.** C\* flags these illegal actions such that, at the threshold τ95 frozen on procedural DEV for
invalid-step recall 0.95:

1. precision is ≥ 0.95 in each of the 3 seeds, and the seed-pooled, problem-clustered bootstrap 95% lower bound is
   ≥ 0.90;
2. the realized recall on the held-out producer is ≥ 0.90, so the DEV operating point transfers. This is what
   "calibrated" means here;
3. its recall exceeds the recall of the forward model's own legality bit on the same keys by ≥ 0.10 (F0's
   `legal: false` catches 1,302/1,670 = 0.780), with a clustered-bootstrap lower bound > 0.

**H-E04 is falsified** if any of these holds:
- any seed has precision < 0.95, or the pooled lower bound is < 0.90;
- the realized recall is < 0.90;
- the gain over F0's flag is < 0.10, or its lower bound is ≤ 0;
- a mandatory control misbehaves (§4.5). That case is reported as VOID or REVISE, never as a pass.

**Fallback primary H-E04-fb.** It is used only if BWM has not released `BW_DISTILL_r1/fcache.jsonl` when the prereg
is frozen. The choice depends on availability alone and is made before any score exists.
- **The critic.** The composite audit A\*: a schema gate, then the delta-support check, then disagreement between C\*
  and the claimed `legal` flag.
- **The test set.** The 2,400 claims of F0 (`BW_Fdelta_s0`) and FK (`BW_Fdelta_ko_s0`) on the 240 D-54 key-order
  keys: 586 invalid claims, clustered by key.
- **The bars.** The same precision and recall bars as H-E04.
- **The comparison.** The non-learned schema + support check, requiring Δrecall ≥ +0.30 with lower bound > 0.

### 1.3 Why this is complementary (and where it is not)

- **BWM has no learned critic or value anywhere** (`docs/understanding/bwm_program.md` §6).
  - In the frozen planner, legality comes from two sources:
    - F's own `legal` flag;
    - the domain-agnostic **delta-support check** inside `parse_forward_delta` ("a predicted deletion must be a fact
      of the state").
  - BWM P-15 (FREPLAY, registered) found 0 plan-harmful F errors on all five logged sets. But the support check alone
    rejected 13–28% of the illegal actions that were queried:
    - `distill_r1`: 368/1,670 = 22.0%;
    - B2-hard: 19.6%;
    - H2-hard: 27.8%.
- **Structural fact: why the support check is complete in Blocksworld, and only there.**
  - In 4-operator STRIPS Blocksworld every precondition is also a deleted effect:

    | Action | Preconditions = deletions |
    |---|---|
    | pick-up x | clear, ontable, handempty |
    | put-down x | holding |
    | stack x y | holding, clear y |
    | unstack x y | on, clear, handempty |

  - So an answer that applies an illegal action as if it were legal always deletes a fact that is absent from s, and
    the check rejects it.
  - Consequence: in Blocksworld the support check is a **strong, complete, non-learned baseline for false-legal
    claims**. It is **blind to two error classes:**
    - false-illegal: options are lost. Examples are hard/instance-102 `unstack c j`, and F0's degenerate "illegal"
      mode under unfamiliar renderings;
    - wrong deltas that the check still supports.
  - It is **not** complete in Logistics, where LOAD/UNLOAD keep the vehicle's `at` and DRIVE keeps `in-city` (BWM
    F-09). It is also not complete in ledger or prerequisite worlds. A claim-blind legality predicate is the
    component that would carry over.
- **A cross-programme failure class.** Computed predicates erode under surface diversification:
  - BWM D-54: FK makes 75 false-legal claims;
  - RCWM 5a/5b;
  - RCX R-06.

  A claim-blind, structured critic shares neither F's input channel nor F's rendering shortcut. That is the RVM-001
  question: "an evaluator ... without sharing the forward model's shortcuts".
- **Ownership.**
  - Joint item 6 (a legality-checker LoRA on minimal pairs) is BWM + RCWM, and nobody has started it. BWM D-64 and CX
    Q-02 set RCX's part: "RCX contributes a CPU critic baseline built from `traces.bw_parts`". E04 is that baseline,
    plus the held-out-producer battery that item 6 does not yet have.
  - E04 trains no LoRA.
  - E04 does not redo FREPLAY: it reuses FREPLAY's classification as the reference and must reproduce it (§4.4).
  - E04 touches no RCWM data. RCWM owns #8 (no-LLM learners on T2's ledger rows) and item 6's ledger arm.

**What E04 is *not* expected to change.** BWM P-15 found F is not the bottleneck on hard: the exact-F ceiling is no
higher than the observed 73 and 88. So at the step level a Blocksworld legality critic cannot raise planner
*success*. Its planner-level value is bounded by **F calls saved**: illegal proposals make up 11.1% of distill keys,
19.8% of B2-hard keys and 10.3% of H2-hard keys (FREPLAY). The bundle's "25% fewer expansions" target is therefore not
reachable through legality alone in Blocksworld, and the follow-up target is set accordingly (§6.1).

### 1.4 Tiers: what a positive result would and would not establish

| Rung | What E04 does | Claim allowed |
|---|---|---|
| Instrument | Stage 0. It rebuilds and labels the held-out-producer battery and reproduces BWM's registered counts exactly (§4.4). | "The battery is qualified": exact labels, pinned provenance, leakage audit. |
| Mechanism (CPU, not native) | Stage 1. Critics trained from scratch on CPU. | Tier B external-specialist evidence on Blocksworld with exact labels. |
| Native-A (frozen core, inference only) | Not run in E04. It is a follow-up (§6.2), and it overlaps BWM's adapter-off verifier. | — |
| Native-C (LoRA on Titan-Sparse) | Not run in E04. It belongs to joint item 6 (§6.3). | — |

**A positive result would establish**, at the mechanism tier with exact Blocksworld dynamics:
- A claim-blind critic trained only on procedural `bw_parts` near-misses detects a *held-out LM producer's* invalid
  steps with a DEV-frozen operating point that transfers across a prevalence shift (50% → 11%).
- It is more sensitive than the forward model's own legality bit.
- Which representations transfer: relational, lifted, symbolic or flat.
- Whether a critic that reads the claim learns producer identity (S2).
- A replay estimate of the F-call savings.

**It would not establish:**
- any Rineya-core or native capability, or any "intelligence gain";
- any planner-level success or real latency saving (a replay projection is not a planner run, per COMMON_PROTOCOL);
- transfer to Logistics, ledger or prerequisite domains, where the support check is incomplete;
- that a text-reading critic, such as a LoRA, would transfer: E04's encoders parse facts, so rendering and name shifts
  are invisible to them by design;
- that error-trained learned critics are needed at all, if the symbolic inducer ties them (§4.5 simplicity rule).

**External specialist vs internalized.** E04's critic is an external specialist: tier B when it is used inside a
planner, and assisted. Internalization is a separate step that needs peers:
- a native-C legality LoRA (item 6);
- or F learning the computed legality predicate through the `cmp → calc → after` trace (BWM's FK + `bw_parts` trace
  candidate).

### 1.5 Predictions stated before any score (copied verbatim into the prereg)

1. C\*, L1 (lifted logistic) and S (induced preconditions) reach precision ≥ 0.99 and recall ≥ 0.97 on the held-out
   producer Pa_v2.
2. F0's flag stays at recall 0.780 on Pa_v2's illegal actions. Its misses are almost all pick-ups (365 of the 368),
   where its recall is 0.655.
3. L2 (flat MLP) is in range up to n = 10 but degrades on STRESS n = 11–12, and cannot represent n > 12.
4. A claim-reading critic trained on one producer's errors (F0: mostly false-illegal) misses the other producer's
   opposite-polarity errors (FK: mostly false-legal), with recall < 0.5. The claim-blind composite stays ≥ 0.9.
5. **Most likely decision:** ADVANCE with the simplest qualifying module (S or L1). In Blocksworld, an error-trained
   neural critic is not needed. The handover value is the battery, the recipe and the F-call projection.

## 2. Data

### 2.1 Held-out test producers (external, read-only, never trained on)

| Pool | Path (BWM `native_v1/`) | sha256 (pinned) | Producer of the error | Units | Invalid | Clusters | Status |
|---|---|---|---|---|---|---|---|
| **T-PA (primary)** | `runs/BW_DISTILL_r1/fcache.jsonl`, plus the initial states from `solutions.jsonl` | `055d3b66…` (from BWM's FREPLAY prereg), `fdf80151…` | Proposer `BW_Pa_v2_s1` (the action); F0 = `BW_Fdelta_s0` (the claim) | 15,030 unique (s, a) keys | 1,670 illegal actions; 368 F0 false-legal claims | 240 problems, n = 6–10 | **Release requested (CX Q-04).** Content not opened. |
| **T-KO (fallback primary; S1)** | `runs/BW_KEYORDER/items.jsonl`, `eval.json`; keys rebuilt from pinned `scripts/bw_keyorder.py` | `b3126ea6…`, `a87d02bb…`, script `94561044…` | F0 and FK claims under 5 renderings | 2,400 claims, 240 keys | F0 493, FK 93 | 240 keys (n = 5, 10) | Readable (released 2026-10-11). Regeneration needs BWM's OK on seed 20261013 (§6.5). |

**How the T-PA keys and labels are formed.**
- **Key format** (from `PREREG_FREPLAY.md` §2): `{"s": facts, "a": action, "text": F answer}`.
- **Unit:** a unique (state, action) key. Within the file the last line for a key wins, which is the planner's memo
  semantics.
- **Where the actions come from:** every key's action comes from a Pa_v2 candidate fragment in the frozen B2 beam
  (K 4, W 4, E 24, R 12, M 80, P 1). This is verified at Stage 0 against the planner semantics, read only.
- **Labels:**
  - the action label is `bw_parts(s, a)` ok;
  - the claim label follows the FREPLAY §3 classes.
- **Clusters:** problems. The problem of each key comes from its shard and plan file, as in FREPLAY §5.

**How the T-KO keys are rebuilt.**
- The items file has no (s, a). E04 rebuilds the 240 keys and all 2,400 prompts bit-identically from BWM's own code:
  - `keys(random.Random(20261013))`;
  - then the `eval_` rendering loop in its original rng order, including the `heldout_both` renaming.
- Verification: every item's `prompt_sha` (16 hex), 2,400 of 2,400.
- RCX draws nothing new from 20261013; it only reproduces BWM's published draw. Because BWM D-64 says "RCX avoids
  seeds 20261012/13", this needs BWM's explicit OK (CX Q-04).

### 2.2 Training and selection producers (RCX, procedural, fresh)

All RCX data comes from fresh instances, using `blocks.random_towers`-style starts plus exact walks. RCX seeds (to be
recorded in DECISIONS at the freeze; currently unused per the seed audit of 2026-10-12):

| Seed | Use |
|---|---|
| 20261242 | TRAIN parents |
| 20261243 | DEV parents |
| 20261244 | STRESS parents |
| 20261245 | producer and corruption rng |
| 20261246, 20261247, 20261248 | model seeds s0–s2 |
| 20261249 | bootstrap and T-KO half split |

**States.** One parent = one fresh instance (objects plus random towers). Its states are:
- 10 states from a random walk of exact legal moves;
- 10 states along an exact GBFS (misplaced-blocks) plan towards a random full or partial goal, using
  `full_goal_frac` 0.5 as in E07. These mimic the states a search visits.

**Action producers.** These produce an invalid *step*. Labels come from `bw_parts`.
- **PR-L (valid):** a uniform legal action.
- **PR-U (uniform illegal):** a uniform illegal ground action. This is the same generator family as BWM's D-54 key
  sampler.
- **PR-N (near miss).** It is defined *a priori* as the edit-distance-1 neighbourhood of the legal actions in s,
  restricted to illegal results. Four edit types are sampled with equal weight:
  1. operator substitution, with arity repaired: pick-up x ↔ unstack x y, put-down x ↔ stack x y;
  2. substitution of one argument by another block;
  3. argument swap (stack/unstack x y → y x);
  4. a temporal copy: an action legal in s(t−1) or s(t+1) but illegal in s(t). This is the "incorrect copy" or
     stale-plan error.

  **Disclosure.** FREPLAY's per-operator aggregate (pick-up dominates Pa's illegal actions) is public and known to
  the designer. The PR-N weights are therefore fixed by *edit type*, never by operator, and are not tuned.

**Claim producers** (used only by the S1/S2 claim-reading arms). These corrupt the true answer of `bw_parts`:
- **TC-FL:** an illegal action answered "legal", with its as-if STRIPS effects;
- **TC-FI:** a legal action answered "illegal" with an empty delta;
- **TC-COPY:** a legal action with one add or delete fact dropped, one spurious fact added, or one argument swapped;
- **TC-FMT:** malformed JSON or a wrong key set (the "format-corrupted answers" family). It is caught by the schema
  gate and excluded from critic metrics, but counted end to end.

**Sizes** (planned; label calls ≈ rows):

| Split | Parents | n | States | Rows | Illegal share | Role |
|---|---|---|---|---|---|---|
| TRAIN | 1,600 | 3–10 (200 per n) | ~32k | ~64k action rows (50% PR-L, 25% PR-U, 25% PR-N) + ~64k claim rows (S2 only) | 0.50 | fit |
| DEV | 400 | 3–10 (50 per n) | ~8k | ~16k action rows + claim rows | 0.50 | select the epoch; freeze τ; floors |
| STRESS | 250 | 11–15 (50 per n) | ~5k | ~10k action rows | 0.50 | structural OOD (S3) |

The training sizes cover the test sizes (5–10), so the primary isolates *producer* shift from *size* shift. Size
shift is tested separately on STRESS.

### 2.3 Labels (oracle only, never an LM judgement)

- **Actions:** `traces.bw_parts(s, a)` → `cmp` (each precondition fact, with its truth value and `ok`), `calc` and
  `after`. They are imported read-only by exec of sha-verified bytes (`rcx/bw/bwm_import.py` pattern):
  - `bwmn/traces.py` `5da9ab09…`;
  - `bwmn/blocks.py` `b1e28ed5…`;
  - `bwmn/rw2.py` `1195ea10…`, which `traces.py` imports and is stdlib-only;
  - `scripts/bw_keyorder.py` `94561044…`, whose `keys`/`render`/`rename` and constants are extracted by AST, without
    executing its scipy or peft paths.
- **Claims:** `B.parse_forward_delta(text, s)` plus FREPLAY's schema parse `q` and its §3 class table.
- **Independent cross-check:** an RCX precondition table, written from the PDDL. It must agree with `bw_parts` on 100%
  of rows (G4).
- **Precondition attribution:** `cmp` gives which preconditions fail. This is the step-level analogue of the bundle's
  "first-error identification" (S5).

### 2.4 Splits, the OOD matrix, and leakage controls

**Parent-level partition.** A parent is an instance with all of its states and rows. Whole parents are allocated to
TRAIN, DEV or STRESS before any row is drawn. The test pools are external, so their clusters are fixed (240 problems;
240 keys).

**OOD matrix** (per COMMON_PROTOCOL; every component is reported separately):

| Axis | Contrast |
|---|---|
| **Producer (primary)** | procedural PR-U/PR-N → LM proposer Pa_v2 (T-PA), and → LM forward models F0/FK (T-KO) |
| Prevalence | DEV 0.50 → T-PA 0.111, T-KO F0 0.41 and FK 0.078 |
| State distribution | random walks and GBFS paths → beam-search states in tall-tower problems |
| Structural | STRESS n = 11–15 |
| Surface | T-KO's five renderings and unseen names. Every E04 encoder parses facts and is name-invariant, so this axis matters only to the claim-reading arms (S2). |

No arm is tuned on any test pool.

**Exclusions (binding).**
1. **Reserved and selection sets are never read or evaluated:**
   - hard-110, mirror, PB-600, b3fresh;
   - seeds_fresh (item 7) and the B4 screen;
   - val_large, val_v2;
   - the B2/H2 hard/mirror/PB F caches and `BW_SELECT_v1/fcache_delta.jsonl`.

   RCX instances are filtered with `rcx/bw/excl.py` (`row_exclusion_reason`, with selection sets) exactly as in E07.
2. **State-level de-duplication against the test pools.** Every TRAIN, DEV or STRESS (s, a) row that is
   renaming-equivalent to any T-PA or T-KO (s, a) is dropped. Equivalence is a canonical hash: `encode.canonical_order`
   with the action arguments as distinguished colours. The number dropped is reported by n. At n ≤ 5 the state space
   is tiny, so the count will be large there.
3. **RCX's own spent evaluation instances are excluded from the parents:** the E07 test set, E07b TEST-B, E11a and
   E11b.
4. **No BWM training rows are used:** `distill_rows`, `mix_*`, `BW_data*`.
5. **`solutions.jsonl` trains nothing that is scored on T-PA**, because those are the same 240 problems. Its only
   uses:
   - initial states, for the cluster map;
   - the S7 plan-breakage projection;
   - an S′ variant: preconditions induced from F-verified plans, simulator-free, scored on T-KO only.
6. **Spent and sealed material is never touched:** no `*.secret` file, no SEALED or spent RCX battery, no RCWM bank.
7. **Disclosed prior knowledge.** The label-level aggregates of T-PA and T-KO (prevalence, classes, per-operator and
   per-n counts) are already public through BWM's FREPLAY and D-54 reports, and Stage 0 recomputes them. No critic or
   baseline is *scored* on a test pool before the prereg sha is recorded.

### 2.5 Sources considered and not used

| Source | Why not |
|---|---|
| B2/H2 hard, mirror, PB-602 F caches (67k keys; FREPLAY) | BWM reserved test sets. Only BWM's published aggregates are quoted (illegal shares). |
| `BW_B3_EVAL/plans_*_{b3fresh,mirror}` | RCX never evaluates on those instances. |
| `BW_SEEDS` (item 7), `BW_B4_SCREEN` | Off-limits. Not read. |
| `BW_SELECT_v1/fcache_delta.jsonl` | val_large/val_v2 are BWM selection sets. |
| `solutions.jsonl` as an error source | No invalid steps: 4,888/4,888 steps are legal and F-states exact (§5). Used only as noted in §2.4. |
| BWM RW2 artefacts (`E2/zs_*`, `A1_*`, `A2_vlab*`, `F_titan_*`, `EVAL_pilot_s1`) | A different domain (register world), and BWM's own verifier line (F-03/04/06). |
| RCX E07/E07b/E11 search traces | GBFS/beam with exact dynamics, so 0 invalid steps (E11: 1,200 runs, 0 invalid states). |
| RCX E01N/E01R ledger outputs | Deferred to E04-L (§9). That work is entangled with E01R (priority), DEV-only, and needs rule reading. SEALED outputs are never mined (COMMON_PROTOCOL). |
| RCWM ledger and prerequisite traces | RCWM's #8 and item 6's ledger arm. RCWM banks are off-limits. |
| Factory RVM-001 (v0002–v0004; e.g. v0004 train process_reward 332 rows, dev 66) | Procedural only, with no model-generated errors. It is a graph/fuel claim domain with an exposed dev split and a residual plan-length shortcut (0.812 vs 0.743). Kept as an optional later domain-shift check. |

## 3. Arms

Every learned arm uses **identical TRAIN rows, epochs and DEV selection**. Arms differ only in representation. None
gets the rule, the goal or the producer's claim, except the explicitly claim-reading S2 arms.

| ID | Input | Trained on | Parameters | Role |
|---|---|---|---|---|
| **C\*** (L3) | RelGNN over a critic graph: E07's node/edge vocabulary (blocks + table + hand; on/ontable/holding edges; clear and handempty features), with goal channels replaced by action-argument marks (arg1, arg2) and an operator one-hot on the hand node | TRAIN action rows, binary cross-entropy (BCE) | ~124k (`rcx/bw/models.RelGNN`) | proposed critic, 3 seeds |
| L1 | Lifted logistic: per-operator weights over indicators of lifted facts relative to the action arguments (clear/ontable/holding of x and y, holding(other), handempty, on(x,y), on(y,x), on(x,·), on(·,x), on(y,·), on(·,y)) | same | ~70 | "logistic first" (roadmap); strong learned baseline |
| L2 | FlatMLP on E07's canonical-order slot encoding (N_MAX = 12) plus action slot one-hots | same | ~159k | matched-capacity flat control (E07 analogue); 3 seeds |
| S | Symbolic precondition induction: per operator, the intersection of the lifted facts true in **legal** TRAIN rows; legal iff all induced facts hold | TRAIN legal rows only (**no error labels**) | 0 (deterministic) | strong simple competitor; simplicity rule |
| S′ | as S | the 4,888 F-verified B3 plan steps (simulator-free) | 0 | descriptive, scored on T-KO only |
| F0flag | F0's `legal` field, parsed from the logged answer | — | — | **incumbent / frozen-confidence control; primary comparator** |
| INC | F0 flag, then the support check (the frozen planner's composite) | — | — | reference: perfect on T-PA per FREPLAY, but costs one F call per key |
| SUP | schema gate plus the delta-support check | — | 0 | **strong non-learned baseline** for claims (S1, fallback) |
| A\* | SUP, then C\* disagreement with the claimed flag | — | — | claim audit (S1, fallback primary) |
| CR-PROC, CR-F0, CR-FK | RelGNN and L1 *with the claim* (claimed add/del as typed edges, flag as a global feature) | TC-\* rows / F0's T-KO half-A claims plus PR-L valid rows / FK's mirror | ~130k / ~90 | S2: producer-identity probe, 1 seed each |
| Floors | constant (all legal / all illegal), random, op-only logistic, op × n logistic | TRAIN | — | shortcut floors |
| SHUF | C\* trained with TRAIN labels permuted within (op, n) strata | TRAIN | ~124k | shuffled-label control |
| ORACLE | `bw_parts` ok | — | — | deterministic-validator ceiling (= label source) |

**Training.**
- Adam, lr 2e-3, batch 256, 10 epochs.
- The epoch is selected on DEV loss only. No early stopping on any test pool.
- `torch.set_num_threads(4)`, with `CUDA_VISIBLE_DEVICES=""` plus the `torch.cuda` patch before torch is imported.

**Bundle controls mapped.**

| Bundle control | E04 counterpart |
|---|---|
| random and constant scorers | Floors |
| deterministic validator ceiling | ORACLE |
| frozen confidence | F0flag. Token margins are not on disk, so only the binary flag; BWM F-02 reports a margin AUROC of 0.923 on RW2 only. |
| scalar head with matched capacity | L2 vs C\* |
| identical candidate banks | every arm is scored on identical keys |
| shuffled-label critic | SHUF |
| "do not give only the learned critic the rule or goal" | no arm gets either; S induces |
| "consequence features at equal labels" | **not informative in Blocksworld**: the as-if delete list equals the precondition list, so a consequence critic collapses to the precondition test. Deferred to a domain with state-dependent effects (§9). |

**Label and compute accounting.** For every arm the run log records:
- TRAIN rows and oracle calls (S uses the legal subset only);
- parameters, optimizer steps and wall time.

No LM tokens are involved.

## 4. Endpoints, statistics and decision rule

### 4.1 Primary endpoint and primary comparison (H-E04)

- **Primary endpoint.** Precision of C\* on T-PA, with illegal as the positive class, at τ95. τ95 is frozen per seed
  on DEV as the largest threshold whose DEV invalid recall is ≥ 0.95. The realized T-PA recall is reported with it as
  the calibration guard.
- **Primary comparison (one).** C\* recall at τ95 minus F0flag recall (0.780), paired on the 1,670 illegal keys and
  clustered by problem. The false-reject rate of each arm is reported: F0flag's is 0/13,360 on T-PA.
- **Why this comparator.** F0's flag is the legality signal the planner already has for exactly these proposer steps.
  A critic that does not beat it has no reason to exist. INC, the flag plus the support check, is perfect but costs
  an LM call per key, so it is reported as the cost/accuracy reference (S8), not as the test.

### 4.2 Secondary endpoints (descriptive; no claims)

- **The roadmap's endpoints, threshold-free and oracle-threshold.** These separate discrimination from calibration:
  - AUROC;
  - precision at T-PA-fixed recall 0.95;
  - the **transfer gap**: precision at τ95 on T-PA minus DEV precision reweighted to T-PA's prevalence.
- **S1: claim audit on T-KO**, all arms. A\* vs SUP vs no audit, reported:
  - per producer (F0, FK);
  - per mode;
  - per class (FI, FL, WD, malformed).

  Wrong deltas (17 claims) are outside a legality predicate's scope. A\*'s reachable recall is therefore about 0.97.
- **S2: producer identity.** T-KO keys are split into halves A and B (120 each, stratified by n × legality; seed
  20261249). CR-F0 is trained on half A and tested on FK's half-B claims, and CR-FK the reverse. Both are compared with
  CR-PROC and with the claim-blind A\* on the same half-B claims. Reading: identity learning if a CR arm's recall on the
  opposite-polarity class is < 0.5 while A\*'s is ≥ 0.9. Small n (FK has 46 unique invalid keys), so this is
  descriptive only.
- **S3: representation × structure.** C\*, L1, L2 and S on T-PA, DEV and STRESS (n = 11–15; L2 up to 12 only).
- **S4: calibration.** Expected calibration error (ECE, 15 bins) on T-PA, in three versions:
  - raw;
  - after unsupervised EM prior correction (Saerens et al. 2002, using T-PA scores only);
  - after oracle-prior correction (reference).
- **S5: attribution.** For S and L1, how accurately they say *which* precondition fails, against `bw_parts` `cmp`,
  on T-PA's illegal keys.
- **S6: strata.** Everything by operator (pick-up, put-down, stack, unstack) and by n. Pick-up, where F0's flag recall
  is 0.655, is the stratum of interest.
- **S7: F-call projection, replay only.** At C\*'s operating point on T-PA:
  - the illegal keys pre-rejected (F calls saved, out of 15,030);
  - the legal keys pre-rejected (options lost);
  - the number of the 211 verified B3 plans that contain at least one pre-rejected step, from `solutions.jsonl` plan
    steps (all legal).

  Labelled "cached-replay projection, not a planner run" (COMMON_PROTOCOL).
- **S8.** The share of INC's rejections that C\* reproduces with zero LM calls, and C\*'s extra false rejections.

### 4.3 Statistics

- **Bootstrap.** Clustered over problems (T-PA: 240) or keys (T-KO: 240); 10,000 resamples; seed 20261249; percentile
  95% intervals.
- **Seeds.** Endpoints are computed per seed. The pooled statistic is the seed-mean within each resample. Per-seed
  values are always shown.
- **Paired difference.** Δrecall is computed from per-key discordance inside each resample, so it stays paired. As a
  check, a clustered McNemar (Obuchowski 1998) is also reported.
- **Multiplicity.** One primary comparison, so no Holm correction. Every secondary is descriptive. The fallback
  primary *replaces* the primary; it is not an extra test.
- **Fixed in the prereg:** τ rules, epochs, arms, strata and the priority order under the time cap (§7). There is no
  re-run after the test pools are scored, except a VOID instrument fault, which is pre-declared, logged in
  ATTEMPTS.jsonl, and re-run with identical seeds.

### 4.4 Stage 0 instrument gates (no outcome is opened; a failure means VOID, fix, repeat)

| Gate | Check |
|---|---|
| G1 pins | Every peer file sha-matches (§2.1, §2.3). `bwm_import.check_no_bytecode()` is clean before and after. |
| G2 T-KO | 2,400 of 2,400 `prompt_sha` values match. Re-scoring the items with the pinned `parse_forward_delta` + `B.step` reproduces `eval.json`'s 10 per-mode F0/FK accuracies and every item's `ok` (2,400/2,400). |
| G3 T-PA (if released) | The fcache sha matches. The FREPLAY `distill_r1` classes reproduce exactly: 15,030 keys; 13,360 legal / 1,670 illegal; ok 14,662; FL-rejected-by-support 368; FI 0; wrong state 0; malformed 0; and the by-operator and by-n strata. |
| G4 labels | `bw_parts` ok == `B.step` legal == RCX's independent precondition table on 100% of rows, in all pools. |
| G5 leakage | Exclusion counts reported (instance-level and state-level). 0 remaining renaming-equivalent overlaps between TRAIN/DEV/STRESS and the test pools. |
| G6 floors | On DEV only: SHUF AUROC in [0.45, 0.55]; ORACLE = 1.000; floors as expected. |

`PREREG_E04.md` is frozen, and its sha recorded in DECISIONS, after G1–G5 pass and before any arm is scored on T-PA
or T-KO. G6 runs inside the main job, before the test scoring.

### 4.5 Decision rule (ADVANCE / REVISE / STOP / INCONCLUSIVE)

**ADVANCE**, which qualifies the critic for a co-signed planner screen and for handover to item 6, requires all of
the following:
- (a) each seed has precision at τ95 ≥ 0.95 and realized recall ≥ 0.90;
- (b) the pooled precision lower bound is ≥ 0.90;
- (c) Δrecall vs F0flag is ≥ +0.10, with its lower bound > 0;
- (d) the controls hold: SHUF AUROC on T-PA is in [0.45, 0.55], and C\*'s AUROC beats the op × n floor by ≥ 0.10.

**Simplicity rule (which module advances).**
- S and L1 are judged by the same (a)–(c). S is deterministic, so "each seed" means its single run.
- Among the arms that qualify, the **simplest** advances, in the order S, then L1, then C\*.
- The exception: a more complex arm advances only if it beats the simpler one by ≥ 0.01 precision at τ95, with a
  paired lower bound > 0.
- The bundle's principle: "a simple module that wins is an asset".

**REVISE** if (c) holds but (a) or (b) fails. That is a producer-shift or calibration failure.
- Diagnose it with the transfer gap, S4 and S6.
- The next step is new near-miss families or a prior-corrected τ, re-qualified on a **new** held-out producer. T-PA
  is spent as E04's primary.

**STOP** (learned legality critics in Blocksworld) if either holds:
- the pooled Δrecall vs F0flag is < +0.05;
- C\*'s pooled precision at τ95 is < 0.80.

Reading: there is no evidence that a CPU critic adds to F's own legality bit on a held-out proposer. Keep F's flag
plus the support check, and have E06 use value critics only.

**INCONCLUSIVE** otherwise, i.e. when the intervals straddle the bars.

**Gate for proposing the planner screen.** This is descriptive and does not change the outcome. On ADVANCE, the
planner screen (§6.1) is proposed to BWM only if both hold:
- S7 projects ≥ 5% of unique F queries saved;
- at most 4 of the 211 verified plans contain a pre-rejected step.

**Fallback (H-E04-fb).** The same structure, with A\* in place of C\* and SUP in place of F0flag:
- Δrecall ≥ +0.30 with lower bound > 0;
- clusters = keys.

### 4.6 Power

**T-PA (primary).**
- Units: 1,670 illegal and 13,360 legal keys in 240 problems. On average a problem has 62.6 keys, of which 7.0 are
  illegal.
- Design effects assumed: DEFF = 1.6 (ICC 0.1) to 2.8 (ICC 0.3), i.e. n_eff ≈ 600–1,040 illegal keys.
- **Precision bar.** With true recall 0.95 and false-reject rate (FRR) 0.3%, precision is 0.975 with SE ≈ 0.0065 at
  DEFF 2.8, giving a lower bound of about 0.96. Power to clear LB ≥ 0.90 when the true precision is ≥ 0.95 is > 0.99.
  Note what the bar implies: precision ≥ 0.95 at recall 0.95 and prevalence 0.111 means an FRR ≤ 0.6% on T-PA.
- **Comparison.**
  - With true C\* recall 0.95 (Δ = +0.17): about 350 C\*-only and 65 F0-only catches, SE(Δ) ≈ 0.016 at DEFF 2. The
    probability that Δ̂ ≥ 0.10 with LB > 0 is ≈ 1.
  - Minimum detectable Δ > 0 (80% power, α 0.05): ≈ 0.045.
  - With true Δ = 0.13, P(Δ̂ ≥ 0.10) ≈ 0.97.

**T-KO (fallback).**
- Units: 586 invalid claims, but only 181 unique (key, class) pairs in 162 keys.
- Renderings repeat the same key's error, so DEFF ≈ 3.6 and n_eff ≈ 160.
- A precision of 0.97 gives SE ≈ 0.014. The expected Δ vs SUP is about +0.8 (SUP catches only FL + malformed ≈ 0.15),
  so power ≈ 1.
- S2 (FK half B ≈ 23 unique invalid) has too little power for anything but description.

**Simulation before the freeze.** Using only the Stage 0 label-level cluster structure: 2,000 simulations per cell
over recall 0.85–0.99, FRR 0.1–2% and ICC 0–0.5. Reported in the prereg. No critic score is used.

## 5. Feasibility check of the data on disk (2026-10-12)

All counts below come from read-only inspection: `python -B` under 5 s on 1 thread, with an independent 4-operator
STRIPS step written for the check. No model was loaded.

| Item | Found | Count |
|---|---|---|
| `BW_KEYORDER/items.jsonl` (released) | sha `b3126ea6…`; fields arm/mode/i/n_blocks/legal/prompt_sha/text/ok; **no (s, a)** (rebuild needed) | 2,400 rows = 2 producers × 5 modes × 240 keys. Keys: n = 5: 120 (64 legal / 56 illegal); n = 10: 120 (67 / 53); total 131 legal / 109 illegal |
| F0 claims on T-KO | invalid **493/1,200** (41.1%): false-illegal 472, false-legal 8, wrong delta 10, malformed 3. By mode: canon 2, train_order 108, heldout_rev 131, heldout_other 121, heldout_both 131 | F0 answers "illegal" on **all 131 legal keys** in heldout_rev and heldout_both, a degenerate mode. Unique (key, class): 151; keys with an error: 138 |
| FK claims on T-KO | invalid **93/1,200** (7.75%): false-legal 75, false-illegal 11, wrong delta 7. By mode: canon 9, train_order 18, rev 20, other 12, both 34 | Unique (key, class): 46; keys with an error: 46. Pooled over F0 and FK: 181 unique (key, class); 162 keys |
| `BW_KEYORDER/eval.json` | sha `a87d02bb…`; per-mode F0/FK accuracies, reading "partial" | reproduction target for G2 |
| `scripts/bw_keyorder.py` | sha `94561044…` (= `eval.json` script_sha256); `keys()` is deterministic from seed 20261013 | rebuild is feasible on CPU in seconds |
| `BW_DISTILL_r1/fcache.jsonl` | **not opened** (release pending); line count only | **15,030** lines (= FREPLAY's unique keys; sha `055d3b66…` from BWM's prereg) |
| FREPLAY `distill_r1` (BWM registered report, read) | 13,360 legal / **1,670 illegal** (11.1%); F0 flag "legal" on 368 illegal (22.0% [20.1, 24.1]), all rejected by the support check; false-illegal 0; wrong state 0; harmful 0 | By operator (keys / illegal / F-legal-on-illegal): pick-up 4,545 / 1,057 / 365; put-down 2,760 / 10 / 0; stack 4,018 / 120 / 0; unstack 3,707 / 483 / 3. By n (keys / illegal): 6: 486 / 22; 7: 951 / 60; 8: 3,094 / 285; 9: 3,907 / 424; 10: 6,592 / 879 |
| `BW_DISTILL_r1/solutions.jsonl` (released) | sha `fdf80151…`; 240 rows (211 verified, 29 not); fields include init, goal, plan, f_states | 4,254 steps in verified plans + 634 in unverified rows = 4,888 steps. **All legal**, and the F-predicted states equal the exact next states 4,888/4,888. So it has **0 invalid steps**, and is usable only as a positive or valid source |
| RCX E07/E11 runs | exact-dynamics GBFS/beam traces | 0 invalid steps, as expected |
| Factory RVM-001 | v0004: train process_reward 332, dev 66 (v0002: 2,494 / 638); accepted 444 rows | procedural only; not used |
| CPU lane | `%TEMP%/rineya-cpu-lane.json` absent at inspection (lane free) | — |

**Verdict: FEASIBLE on CPU (about 25–45 min of lane time; cap 60 min; < 1.5 GB RSS).**

- The primary needs **one BWM release** (`fcache.jsonl`, already pinned in BWM's own FREPLAY prereg).
- The fallback primary is feasible **today** with released data, but the T-KO key rebuild needs BWM's OK on seed
  20261013.
- **Power.** It is ample for the primary: 1,670 illegal keys in 240 clusters. It is adequate for the fallback (181
  unique invalid). S2 is descriptive only.
- **Main scientific risk: ceiling effects**, not data. Blocksworld legality is local, so the informative quantities
  are:
  - calibration transfer;
  - the gain over F0's flag (concentrated in pick-ups);
  - the simplicity rule;
  - the S2 identity probe;
  - the F-call projection.

## 6. Native follow-up path, and what peers must agree to

### 6.1 Planner use: tier B external specialist (BWM's call; after B4/B5 in BWM's queue)

**The arm.** An **F pre-filter wrapper**:
- for a key the qualified critic rejects at a stricter frozen operating point (DEV FRR ≤ 0.1%), the wrapper returns
  F's own "illegal" answer format without calling F;
- `blocks_planner.py` (`18a5f22b…`) is not edited, so the planner semantics are unchanged;
- as in the B4 screen, the wrapper's counters (calls, rejections) are reported.

**The test.**
- Endpoint: unique F queries per instance at non-inferior success, paired on a fresh hard-style set (BWM seeds
  202614xx, RCX exclusions).
- Also reported: success (McNemar), options lost to false rejections, and wall time.

**Pre-declared target.** ≥ 10% fewer F queries at non-inferior success, with a margin to be co-signed. Not 25%, which
§1.3 shows is unreachable through legality alone. Per the bundle's critic rule, the critic is applied as a hard filter
*before* any soft ranking.

**Needs:**
- a co-signed prereg;
- a BWM GPU turn;
- operator approval.

**A domain where it may matter more: Logistics (BWM item 9).** There the support check is incomplete, and BWM's draft
needs a legality gate before its freeze. E04's recipe is offered to BWM: procedural minimal pairs from the exact
simulator, then a lifted, symbolic or relational critic, then held-out-producer qualification. RCX does not touch
Logistics, and adopting the recipe is BWM's decision.

### 6.2 Native-A probe (inference only; overlaps BWM's verifier, so joint or BWM-run)

- **Question:** does frozen Titan-Sparse already have the legality predicate? Ask it "is a executable in s?", with the
  adapter off, on the frozen E04 battery (T-KO plus STRESS, and T-PA if it is released).
- **Overlap:** this is BWM's adapter-off verifier line (F-03/F-04/F-06 on RW2; F-02's legality-token margin flagged
  F's RW2 errors at AUROC 0.923). So it is proposed to BWM to run, or to run jointly.
- **Resources:** a GPU or 12 GB of free RAM.

### 6.3 Native-C: joint item 6 (owners BWM + RCWM)

The owners would train either:
- a legality LoRA on frozen FP32 Titan-Sparse (unmerged);
- or BWM's FK + `bw_parts` `cmp → calc → after` trace candidate, which restores the computed predicate through F
  itself.

E04 supplies:
- the frozen battery, with exact labels and the leakage audit;
- the recipe: which near-miss families matter, whether PROC-only suffices, and the size coverage;
- the CPU reference numbers that a LoRA must beat;
- the S2 warning that critics which read claims can learn producer identity.

**Requirements:**
- the item-6 owners' agreement;
- any peer-trained critic that is to be scored on the E04 battery must not train on T-KO or T-PA keys;
- a GPU lease and operator approval;
- E33/5c retention by RCWM for any Titan-Sparse adapter;
- A/B/C/D reporting with matched SFT.

RCX trains no LoRA unless item 6 is handed over.

### 6.4 Dense transfer (far)

Critic-labelled step data could serve as supervision for the computed legality predicate in a fresh 0.6B student
(bundle INTEGRATION_AND_TRANSFER). Every label would first be audited by the exact validator, as RVM-001's dense path
requires.

### 6.5 CX Q-04 (to BWM; FYI to RCWM), to be sent by the coordinator

1. **Release** `runs/BW_DISTILL_r1/fcache.jsonl` (sha `055d3b66…`, 15,030 rows) read-only to RCX, as E04's held-out
   TEST producer.
   - It is never trained on and never redistributed.
   - RCX writes nothing in BWM's tree.
   - If it is not released by the freeze, E04 uses the T-KO fallback.
2. **OK to reproduce the D-54 key-order keys** bit-identically from the pinned `bw_keyorder.py`, using EVAL_SEED
   20261013.
   - This is BWM's own draw; RCX draws nothing new from it.
   - It is verified by the 2,400 prompt shas.
   - BWM D-64 asks RCX to avoid 20261012/13, hence this explicit request.
3. **Confirm the item 6 boundary:** E04 is a CPU claim-blind critic plus a battery, with no LoRA. Peers may score
   item-6 critics on the E04 battery.
4. **Notice:** T-KO and T-PA become E04 test pools. A peer critic trained on them could not be scored cleanly on E04.
   BWM's own FK + trace evaluation is unaffected.
5. **Later, only on ADVANCE and the S7 gate:** a co-signed planner pre-filter screen (§6.1).

RCWM needs to agree to nothing for E04, since no RCWM data or ledger work is involved. It is informed for item 6 and
#8.

## 7. Compute and schedule (single CPU lane; E01R has priority)

**Budget.**

| Stage | Time | Threads / RAM | Notes |
|---|---|---|---|
| Stage 0 (G1–G5) | ≤ 2 min | 1 thread, < 300 MB | run under the lane lock as a 5-minute slot |
| Main job | 25–45 min estimated; **hard cap 60 min** | 4 torch threads, < 1.5 GB RSS | data generation and hashing ~5 min; L1/S/floors < 1 min; L2 3 × ~1.5 min; C\* 3 × ~2–4 min; SHUF ~3 min; CR arms ~6 min; scoring and bootstrap ~3 min |
| Analysis | < 5 min | — | — |

The main-job estimate is calibrated on R-04: 6 fits on 110k rows × 12 epochs took 626 s.

**Priority under the cap**, pre-declared in this order:
1. primary arms: C\* s0–s2, F0flag, S, L1, floors, SHUF;
2. L2;
3. S1 and S2;
4. STRESS.

Arms not reached are reported as not run.

**Guards.**
- `rcx/guard.py`: pause at tz01 ≥ 97 °C until ≤ 93 °C, and while `.thermal_pause` exists.
- `rcx/lane.py`: `acquire --job "E04" --expected-min 60 --need-ram 4`. The job yields between fits if a peer's
  waiting marker appears.
- No GPU, no model loads, no peer writes, `sys.dont_write_bytecode`.
- Before the start, check `Win32_Process` for stray python.

**Placement relative to E01R.** E01R trains 6 CPU LoRA adapters of about 2.6–3.2 h each, needing ≥ 12.5 GB of free
RAM, in the GPU-idle night windows 22:50–07:00 on 10-12, 10-13 and 10-14 (RCWM D-51). Its SEALED evaluations follow,
and **always come first**. E04 needs only about 1.5 GB, so it fits the daytime gaps that E01R *cannot* use (when peer
GPU turns hold RAM) whenever the lane is free.

| When | Step |
|---|---|
| 10-12 (now) | This design. Coordinator sends CX Q-04. No compute. |
| 10-12/13, daytime | Write the code and fast tests (< 5 s, no torch). Stage 0 in a 5-minute lane slot. Draft `PREREG_E04.md`. |
| By 10-13 18:00, or on BWM's reply | Freeze the prereg. If `fcache.jsonl` has not been released by then, freeze with the fallback primary. |
| First free daytime lane gap after the freeze (10-13 or 10-14), finishing by 22:20 | Main job. Otherwise after E01R's training and sealed evaluations (about 10-15/16), as the coordinator sequences it. E04 never takes the lane while E01R's chain is ready or its waiting marker exists. |
| Same day | Analysis, R-11 in EXPERIMENT_RESULTS, FYI to BWM and RCWM. On ADVANCE with the S7 gate passed, propose §6.1 to BWM. |

## 8. Hygiene and stop-the-line rules

- **Read only.** RCX reads only the peer files named in §2.1 and §2.3, by sha. It never reads anything in §2.5 that
  is marked reserved or off-limits. It never opens `*.secret`, SEALED, spent, or RCWM bank files.
- **Writes.** All outputs go to `complementary/runs/RCX-E04/`, with code shas, peer pins, seeds, row counts and
  timings.
- **Stop-the-line.** Any of the following halts the run, which is recorded as VOID:
  - a pin mismatch;
  - a G2/G3 count that is not reproduced exactly;
  - a label disagreement in G4;
  - a non-zero G5 overlap after exclusion;
  - a bytecode write into BWM's tree.

  No test-pool score is computed in these cases.
- **Fixed after the freeze.** Arms, τ rules, epochs, seeds and strata are fixed by the prereg. Any change needs an
  amendment written before the test pools are scored.

## 9. Relation to E05/E06, and deferred extensions

- **E05 (successor features vs scalar value at equal labels).** The registry gates it on "E04 passes or documented
  independent rationale". E05 has an independent rationale: goal-weight re-ranking under fixed dynamics. So E04's
  outcome informs E05 but does not block it.
- **E06 (value-guided bounded search).**
  - On ADVANCE, E04's critic is the hard-legality stage that precedes any soft value ranking.
  - On STOP, E06 proceeds with value critics only.
- **E04-L (deferred): a ledger-domain critic.** RCX's own rule-conditioned ledger (RCL) outputs exist only for DEV.
  They come from Titan OFF, T2 s0, N-VR and N-CFSD, with exact labels from the R-01 interpreter. In that domain the
  support check does not apply, so a critic would have real headroom.
  - It is deferred because a critic there must *read the supplied rule*, which is the E01R question itself.
  - It would be considered only after E01R is analysed, on DEV outputs only, and never on SEALED or spent material.
- **The consequence-feature control and trajectory-level first-error identification** need a domain with
  state-dependent effects. They also need logged proposer fragments, which `fcache.jsonl` does not keep (it stores
  unique keys only). Both are deferred to the planner follow-up or E04-L.
