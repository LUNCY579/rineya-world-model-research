# SPEC_RCL v1: the rule-conditioned ledger world (RCX-E01 instrument)

**Status:** design spec, written 2026-10-09 by the RCX session.

This spec is the single source of truth for:
- the generator (`rcx/rcl/world.py`);
- the exact interpreter;
- the **independently written** oracle (`rcx/rcl/oracle_indep.py`), which must be implemented from this text alone,
  without reading `world.py`;
- the rule-text parser.

**Relation to RCWM.** It is a *rule-parameterized* generalization of RCWM's `ledger_replay` family (`rcwmt/env.py`). In
RCWM, "applied only if `status == "posted"`" is hard-coded. Here the applicability rule is *supplied in the input* and
varies.

## 1. State, event and dynamics

- **State.** A mapping from account name to integer balance. It has 4 accounts in TRAIN-surface renderings and 5–6 in the
  `A_many` surface.
- **Event.** A JSON object with exactly these 5 keys:
  - `account`: the target account name (a key of the state);
  - `amount`: an integer;
  - `kind`: `"add"` or `"transfer"` in the arith world, and `"set"` or `"swap"` in the copy world (§1.1). `"scale"` is
    reserved for the native kit and is not used in the CPU study;
  - `other`: a state key different from `account`, present for every kind but used only by `transfer`;
  - `status`: a non-empty lowercase-or-mixed string of letters, digits, `_` or `-`, never containing spaces.
- **Dynamics (`calc`).** These apply only when the rule's applicability predicate is true:
  - `add`: `after[account] = before[account] + amount`.
  - `transfer`: `after[account] = before[account] - amount` and `after[other] = before[other] + amount`.
  - `scale` (native kit only): `after[account] = before[account] * amount`.
  - All other accounts are unchanged.
- **No-op.** If the predicate is false, `after == before` exactly.
- **RCWM equivalence.** With the canonical *posted rule* (`S+{posted}`) and RCWM's kinds and fields, the transition must
  equal `rcwmt.env.step("ledger_replay", before, ev)` once `tick` is added. This is checked in teacher qualification.

### 1.1 Copy-world variant (CPU mechanism tier; added 2026-10-10, CX D-09)

This variant is arithmetic-free. It factors out the calculation A, so that small learners trained from scratch can be
studied on applicability C.

| Kind | Effect | Touched accounts |
|---|---|---|
| `set` | `after[account] = amount` | `account` |
| `swap` | `after[account] = before[other]` and `after[other] = before[account]` | `account`, then `other` |

- `amount` ranges: 0–98 for `set` (100–999 in the `M2_amt` stratum), and 1–12 for `swap`, where it is ignored.
- `K+` clauses may name `set` or `swap`.
- In this variant, `AGE`/`ALT` thresholds are drawn from 10–90.
- Everything else (rules, rendering, targets, strata) is unchanged.
- The arith world (`add`, `transfer`) remains the instrument for native, pretrained-core runs.

## 2. Rule language (applicability predicate)

A rule is an AST node. Atoms:

| Tag | Meaning | Fields |
|---|---|---|
| `S+` | `event.status ∈ set` | `set`: a list of 1–3 distinct status strings |
| `S-` | `event.status ∉ set` | `set` |
| `K+` | `event.kind ∈ kinds` | `kinds`: a non-empty list of distinct kinds from {add, transfer, set, swap}. Generators emit one kind from the active world; parsers accept any such list. |
| `AGE` | `event.amount >= t` | `t`: an integer |
| `ALT` | `event.amount < t` | `t` |
| `BGE` | `before[event.account] >= t` | `t` |

Compositions: `AND(x, y)` and `OR(x, y)`, where x and y are atoms. There is no deeper nesting in v1.

**Status matching** is exact and case-sensitive string equality. `"Void"` is not `"void"`. Every gold label follows this
literal-equality semantics, and every rendered rule states the statuses verbatim.

**Canonical JSON form** of a rule:
- atoms: `{"op":"S+","set":["a","b"]}`, `{"op":"AGE","t":5}`;
- compositions: `{"op":"AND","args":[x,y]}`.

## 3. Rendering

An input text has three lines, separated by `\n`, always in this order:

```
RULE: <rule sentence>
STATE: <state JSON>
EVENT: <event JSON>
```

- **JSON.** JSON objects are compact: `,` and `:` separators, no spaces, integers as plain decimals. The *key order* is a
  rendering factor (§3.2). The `other` and `account` values are account names.
- **Status lists in rule sentences** use a list renderer:
  - 1 item: `x`;
  - 2 items: `x or y`;
  - 3 items: `x, y or z`.

  The status tokens appear verbatim, without quotes.
- **Template wording** comes from the closed template pools in `rcx/rcl/templates.py`. The parser inverts these exactly.
  An independent oracle may instead consume the canonical AST, which is stored in every record as `rule_ast`. It must
  then also verify that the rendered sentence round-trips through its own regex parser for at least the TRAIN template
  pool.

### 3.1 Template pools

- Each atom/polarity family has **train** templates (ids `t0…`) and **held-out** templates (ids `h0…`).
- `t0` is the *canonical* template.
- The held-out templates never appear in TRAIN or DEV-in-surface renderings.

### 3.2 Key orders

- **State.** Keys in canonical (sorted) order or in a permutation.
- **Event.** One of the 120 permutations of the 5 keys.
- **Split of permutations.**
  - Held-out event orders (fixed): the reversed canonical order `status,other,kind,amount,account` plus 2 fixed
    permutations, `kind,status,account,other,amount` and `other,amount,status,account,kind`.
  - Train orders: all other permutations.
- The held-out state order is the reversed canonical order.

### 3.3 Account names

- `letters` (train): A, B, C, D.
- `letters2` (held-out): P, Q, R, S, T, U.
- `words` (held-out): north, south, east, west, pine, oak. These are used for 4 to 6 accounts.

## 4. Targets (output formats)

All target JSON is compact, and the literal key order shown here is mandatory.

| Format | Shape |
|---|---|
| `ans` | `{"after":{…}}`. The `after` object uses the **same key order as the input STATE**. |
| `trace` | `{"cmp":{"status":S,"listed":L,"ok":OK},"calc":{…},"after":{…}}` |
| `fill` | `{"fill":"<F>","after":{…}}` |

Details for `trace` and `fill`:
- `listed` is whether `S` is in the set of the rule's *status atom*. For rules with no status atom (pure `K+`/`AGE`/…),
  `listed` is `null`. For compositions it refers to the first status atom found left to right.
- `ok` is the full predicate value.
- `calc` holds the values of the touched accounts (`account`, plus `other` for transfer) after the chosen branch. When
  `ok` is false, `calc` holds their unchanged values. Keys are in the order `account` then `other`.
- `<F>` is `" ."` repeated n times. n is a constant per event kind, chosen from TRAIN only, at least the longest
  `trace` target's length in characters for that kind (D-36 amendment 4b analog).

**Parsing.** Parsers are strict. They reject:
- other key orders;
- extra keys;
- non-integer balances;
- trailing text other than whitespace.

A malformed output scores as wrong. It is never repaired.

## 5. Vocabulary pools (status words)

- `posted` and `void` are the fixed-rule anchor words (as in RCWM T2).
- **Real-word pools** are disjoint across TRAIN, DEV and SEALED. They are listed in `rcx/rcl/vocab.py`, frozen at
  creation, and checked for overlap.
- **Nonce words** are generated from a seeded syllable grammar (CV or CVC syllables, 2–3 syllables). Each nonce word is
  assigned to a split by `sha256(word)` bucket, and DEV and SEALED buckets are disjoint from TRAIN.
- **Exclusions.**
  - Any nonce word within edit distance ≤ 1 of a real-pool word, `posted`, `void` or another split's word is rejected.
  - **PREREG-3's vocabulary** (pending, draft, cancelled, reversed, held, approved, active, ok, dax, wug, Void, VOID,
    voided, unposted, not_posted, "not posted") and the factory's exposed words (xelth, quom) are **excluded from every
    RCX pool**. This keeps PREREG-3's bank usable as an external descriptive check of an RCX-trained native adapter.
- **Near-miss probes** (diagnostic stratum only): case and morphology variants of TRAIN's no-op words, for example
  `Void`, `VOID`, `voided` for `void`. Under literal semantics these are *not* `void`.

## 6. Parents, variants and splits

- **Parent.** A parent is one semantic instance: (state, event skeleton, rule skeleton). Its variants are produced by
  counterfactual edits:

  | Variant | Edit |
  |---|---|
  | `base` | — |
  | `rule_flip` | the status atom's polarity is flipped (S+ ↔ S-); the gold decision flips |
  | `status_swap` | the event status is replaced so that the gold decision flips |
  | `set_swap` | the rule's set is replaced by a disjoint set, so `listed` flips |

  All variants of a parent stay in the parent's split.
- **Split assignment.**
  - Parents are generated per split from split-specific seeds and split-specific vocabulary pools.
  - Each parent carries `semantic_hash = sha256(canonical JSON of {rule_ast, state (sorted), event (sorted)})`.
  - A cross-split audit asserts zero semantic-hash overlap. It also asserts zero overlap of
    (state, event-without-status) skeletons between TRAIN and test parents.
- **Rendering.** A rendering is chosen *after* the parent exists. It is drawn from a split-appropriate surface: `in`
  (train pools) or `ho` (held-out pools: templates `h*`, held-out key orders, `letters2` or `words` names, 5–6 accounts).

## 7. Test battery strata

Every stratum is built from fresh DEV (or SEALED) parents.

**Fixed-rule strata.** Rule `S+{posted}`, canonical wording, except that `ho` renderings use held-out paraphrases.

| Stratum | Content | Gold |
|---|---|---|
| `F0_posted` | status `posted` | apply |
| `F1_void` | status `void` | no-op |
| `F2_unseen_real` | DEV/SEALED real words | no-op |
| `F3_unseen_nonce` | DEV/SEALED nonce words | no-op |
| `F4_nearmiss` | `Void`, `VOID`, `voided`, `void_` (diagnostic) | no-op |
| `F5_postedlike` | `nonposted`, `posted_not`, `reposted`, `postedx`, `un-posted` | no-op |

**Varied-rule strata.** Status sets come from DEV/SEALED pools, mixing real and nonce words.

| Stratum | Content | Gold |
|---|---|---|
| `V1_pos` | `S+{set}`, status in the set or not, balanced | per rule |
| `V2_neg` | `S-{set}`, status in the set or not, balanced | per rule |
| `V3_inverted_twins` | parent pairs `(S+{x}, status x)` and `(S-{x}, status x)` | apply / no-op; scored at parent level (both twins exact) |
| `V5_posted_override` | rules that override the prior on `posted`: `S+{x}` with status `posted`, and `S-{posted}` with status `posted` | no-op for both |
| `V6_compose` | `AND`/`OR` of a status atom with `K+`, `AGE` or `ALT` atoms (E02-flavored, secondary) | per rule |
| `V7_bigset` | `S+` or `S-` with set size 3 | per rule |

**Magnitude strata** (report only):
- `M1_bal`: balances 100–999.
- `M2_amt`: |amount| 13–99.

**Surface axes.** Each stratum is rendered in both `in` and `ho` surfaces, paired by parent. The `ho` surface may also be
split into its components: template only, key order only, names only, and many accounts.

## 8. Shortcut floors

These are deterministic heuristics, scored on every stratum.

| Floor | Rule |
|---|---|
| `COPY` | after = before |
| `ALWAYS` | always apply |
| `TRIGGER` | no-op iff the status is a TRAIN-observed no-op word; otherwise apply |
| `CONTAINS_POSTED` | apply iff `"posted"` is a substring of the status |
| `STATUS_LOOKUP` | per-status majority decision from TRAIN; unseen statuses get the TRAIN majority decision |
| `RULE_WORD_MATCH` | apply iff the event status appears verbatim in the RULE line |
| `MAJORITY` | the TRAIN majority decision |
| `ORACLE` | the exact interpreter on the parsed rule (ceiling; tier B) |

## 9. Metrics

- **Exact transition.** The parsed `after` equals the gold `after` exactly, keys and values included. Malformed output is
  wrong.
- **Decision accuracy.** The predicted action is classified as `noop` (after == before), `apply` (after == gold applied
  state) or `other`. Decision accuracy is the agreement of noop/apply-intent with gold. Intent is `noop` if after ==
  before, otherwise `apply`.
- **Error direction.**
  - `applied_as_posted`: gold no-op, predicted ≠ before.
  - `false_copy`: gold apply, predicted == before.
  - `arith`: decision right, value wrong.
  - `format`: unparseable.
- **Aggregation.** Variants are averaged within a parent first (parent-macro), then averaged over parents. For twin
  strata (`V3`), the parent scores 1 only if both twins are exact.
