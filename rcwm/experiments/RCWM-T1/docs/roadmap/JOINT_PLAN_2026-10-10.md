# RCWM + BWM joint plan after the 2026-10-10 reboot (agreed by message; RCWM D-45, BWM side by message)

## Where the programme stands (confirmed, preregistered)
- PREREG-1: Titan-Sparse chains its own one-step predictions into useful 4-step consequences (regret 0.448).
- PREREG-2: the T2 rule-coverage LoRA makes ledger rollout decisions near-oracle in distribution (regret 0.544 -> 0.056).
- PREREG-3: T2's void no-op is a token cue (TRIGGER); out of range it fails on magnitudes, key order and word names.
- PREREG-LB (joint): T2 as world model inside BWM's beam planner enables planning on its training surface
  (0.706 vs OFF 0.078; T2/ORACLE 0.794); off the training surface it solves 5/180.
- BWM B2: world model + beam search beats the proposer alone on Blocksworld hard (73 vs 5 /110), mirror (61 vs 6)
  and PlanBench-602 (602 vs 584); BW world model is name-invariant but key-order-locked (F-08).

## GPU turn order (strict alternation, 4.5 h cap per turn)
1. RCWM: 5a prerequisite representation screen (D-38; ~3.8 h) - running since 09:27.
2. BWM: B3 search distillation resume (~5.7 GPU-h over 2 turns).
3. RCWM: 5b register trace screen (D-44; ~4.1 h).
4. BWM: B3 remainder, then D-54 BW world-model key-order retrain (bundleG, ~1.9 h).
5. RCWM: T2b-ledger seeds (new, below), then its sealed confirmation.
6. BWM: item 7 seeds (release gate), item 9 transfer.
7. Joint: PREREG-4.

## New RCWM item: T2b-ledger (follow-up to PREREG-3 and PREREG-LB)
A surface-diversified ledger rule adapter on the T2 recipe: status vocabulary with held-out words, wider magnitudes,
input key-order augmentation with held-out orders, word account names and more accounts; targets unchanged.
Confirmation (sealed, preregistered): PREREG-3's out-of-range map on fresh held-out words/orders/names/magnitudes,
in-distribution non-inferiority against T2 s0, the 5c retention guard, and PREREG-LB-style planning on the held-out
surface (today's 5/180 is the number to beat). Shared wording with BWM D-54: "generalises to unseen X" vs "trigger
widened" vs "partial".

## PREREG-4 (joint, end of the cycle)
One T3 adapter (ledger + the winning P and R formats, surface-diversified), 3-seed gate, 5c retention guard on a fresh
sealed draw, an LB-style planning check on held-out surfaces; BWM mirrors it with FK in the frozen B2 planner.

## Lower priority (RCWM)
Item 5 (dissect T2: 8-15-only LoRA, alpha sweep, SD/NAT controls); item 6 (k = 8/16 horizons, seed-consensus
certificates; no training).

## Standing rules
Nothing goes to the vault or Hugging Face without the user's explicit go-ahead; adapters stay unmerged (D-28); every
claim gets an out-of-range and a surface-form audit (D-35); one RCWM waiter at a time; stale locks are cleared by
their owner or with the user's explicit authorisation.
