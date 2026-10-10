# D-54 key-order augmentation for the Blocksworld world model (joint roadmap 5b surface diversification)

Written 2026-10-10 06:00 CDT, before any augmented row or adapter exists.

**Why (F-08).** The delta world model `BW_Fdelta_s0` scores 1.000 exact on its training format, 0.992 with unseen
block names and 1.000 with shuffled fact lists, but only 0.458 when the JSON keys come in reversed order. It learned the
transition rule over symbols, but tied to one serialization. This is the same class of failure as RCWM PREREG-3
(trigger-level). D-54 tests whether augmenting the INPUT key order repairs it **as a rule**: generalising to key
orders never seen in training, not just widening the set of accepted triggers.

## Arms (both trained with the same recipe, D-03)

- **F0** = `BW_Fdelta_s0` (existing): `runs/BW_data/forward_delta_train.jsonl`, 600 steps, batch 16, lr 1e-4, seed 0.
- **FK** = `BW_Fdelta_ko_s0`, built from the SAME 12,000 transitions with the same targets and the same steps, seed,
  lr and max-len.
  - Each prompt is re-rendered: with probability 0.5 canonical; otherwise a random TRAINING key order (top-level
    action/state order and state-key order), and, with probability 0.5 within those rows, shuffled fact lists.
  - Seed 20261012 (`scripts/bw_keyorder.py data`).
  - Targets keep their literal canonical form (D-54: input only).

## Key orders

The state object has 4 keys (clear, hand, on, table), which allow 24 orders.

- **Held out (never in FK's training data):**
  - H1 = (table, on, hand, clear), the full reversal, which is F-08's key_order mode;
  - H2 = (hand, table, clear, on);
  - H3 = (on, clear, table, hand).
- **Training:** the other 21 state orders, with both top-level orders.

## Evaluation (`scripts/bw_keyorder.py eval`; GPU or CPU, shared-prefix greedy, 40 new tokens)

- **Keys:** 240 fresh (state, action) keys, 120 at 5 blocks and 120 at 10 blocks, about 50% illegal. They are built as in
  `bw_surface_audit.py` with seed 20261013 (never used for training data).
- **Scoring:** exact match of the parsed delta against the simulator.
- **Modes** (both adapters, same keys, paired):
  - `canon`: the training format;
  - `train_order`: a random non-canonical training order, with shuffled lists;
  - `heldout_rev`: H1 with top-level (state, action);
  - `heldout_other`: H2 or H3 with a random top-level order;
  - `heldout_both`: H1 + unseen block names (p-z and two-letter) + shuffled lists.

## Registered reading (FK only; F0 is reported alongside)

- **"Augmentation generalises to unseen key orders (rule-level within key permutations)"**: FK ≥ 0.95 on
  `heldout_rev` AND `heldout_other`, and FK on `canon` ≥ F0 on `canon` − 0.01.
- **"Trigger widened, not a rule"**: FK ≥ 0.95 on `train_order` but < 0.90 on either held-out mode.
- **Otherwise "partial"**, with the numbers.
- **Statistics:** FK vs F0 per mode, paired exact McNemar, with Clopper-Pearson CIs.

## Not claimed

- **Name and list robustness beyond F-08:** already at 0.99-1.00 there.
- **Planning results:** the frozen B2 planner always renders the canonical format, so FK does not touch any frozen
  result.
- **Other serializations:** robustness to formats other than JSON key and list order (natural language, other
  separators) is not tested.

## Amendment 1 (2026-10-10 ~11:00 CDT; before any FK adapter or evaluation output existed)

The original text above is unchanged; its sha256 was 6b704388bf7df07a3cb84ecb2330d42051d202eb8421b3bcba315c8e6fcd7965.
Agreed with RCWM so that D-54 and RCWM's T2b-ledger share one wording (RCWM D-45).

- **Reading, replacing the registered reading above:**
  - "generalises to unseen key orders" iff all three hold:
    - FK ≥ 0.95 on `heldout_rev` and on `heldout_other` (the absolute bar);
    - FK's `canon` accuracy ≥ F0's `canon` − 0.01 (non-inferiority);
    - on each held-out mode FK beats F0 with more FK-only than F0-only discordant keys, exact McNemar p < 0.05
      (significantly above the unaugmented arm).
  - "trigger widened, not a rule" iff FK ≥ 0.95 on `train_order` while FK stays near F0 on both held-out modes,
    meaning FK − F0 < 0.10 or McNemar p ≥ 0.05.
  - Otherwise "partial", with the per-mode table.
- **Instrumentation added after the post-reboot audit (no registered quantity changes):**
  - per-item outputs in `runs/BW_KEYORDER/items.jsonl`: arm, mode, key, blocks, legality, prompt hash, generated
    text, tokens, EOS, correct;
  - Clopper-Pearson 95% CIs per arm and mode (registered in the original text but not computed by the first script
    version);
  - the sha256 of both adapters, with an assert that F0 ≠ FK, and of the rows file, script, prereg and items;
  - device and library versions.
- **Data:** the training data is unchanged (sha 82ec3418…, regenerates byte-identically). Note that 76 of the
  6,099 "augmented" rows drew the canonical order with unshuffled lists, so their text equals the canonical text:
  6,023 rows are effectively non-canonical.
