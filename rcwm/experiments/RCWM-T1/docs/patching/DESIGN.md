# Where does Titan's transition computation live? Activation patching and weight-spliced hybrids

**Status:** exploratory mechanism test, follow-up to RESULTS_PILOT R9. Not part of any preregistration.

**Code:**
- `scripts/run_patch.py`: the run. Its module docstring holds the same tables as this note.
- `scripts/patch_test_tiny.py`: an end-to-end CPU test using two tiny random Qwen3 models.

**Launched by `scripts/master_chain.sh`** after the confirmation analysis, under the shared GPU lease. The script line is:

```
[ -f runs/bank_pilot/patch/patch_results.json ] || step scripts/run_patch.py --device cuda --out-name patch
```

So the command that actually runs is:

```
python -u scripts/run_patch.py --device cuda --out-name patch
```

This uses the defaults `--layers 8,16,24 --dmis-layers 24 --hybrid-layers 16,24`. The completion marker is `runs/bank_pilot/patch/patch_results.json`. A crash or a sanity abort (exit 3) leaves no marker, so the chain retries on its next launch, and the retry resumes from `patch_progress.json`.

## 1. Question and background

**What is known so far:**
- Titan-Sparse is Supa 1.7 plus a sparse delta on blocks 8–27. Embeddings, blocks 0–7 and the final norm are bit-identical between the two models.
- R5: 93–99.6% of the answer-slot shift at L12–20 (70–84% at L24) is a constant per-family offset.
- R9: steering with that offset neither installs the skill in Supa nor removes it from Titan. It changes output format and mode.

**The question:** where in depth, and at which positions, does the item-specific transition computation happen?

**Notation:**
- L means `hidden_states[L]`, the output of block L-1.
- "lower delta" means delta blocks 8..L-1; "upper delta" means delta blocks L..27.
- A_T = accuracy of `base:sparse` (Titan); A_S = accuracy of `base:supa`.
- FR = (acc − A_S) / (A_T − A_S), computed on items whose state changes.

## 2. Conditions

Every condition is one full greedy generation over all 640 pilot TEST one-frame items, in bank order, with 16-row left-padded batches.

**Scoring:**
- `run_steer.score` and per-item lenient correctness, as in `run_steer_cpu`.
- Output format classes, from the existing parsers:
  - `bare`: the strict parse succeeds;
  - `wrapped`: only the lenient parse succeeds;
  - `unparsed`: neither succeeds.
- Echo rate: how often the output equals the item's own before-state.

| Condition | What is done | Receiver / host | Default L |
|---|---|---|---|
| `base:sparse`, `base:supa` | nothing | – | – |
| `T2S_last@L` | Titan's block-L output written into Supa at the last prompt position, prefill only | Supa | 8, 16, 24 |
| `T2S_all@L` | Titan's block-L outputs at all real prompt positions, prefill only | Supa | 8, 16, 24 |
| `S2T_all@L` | Supa's block-L outputs at all real prompt positions, prefill only | Titan | 8, 16, 24 |
| `T2S_mis@L` | Titan's all-position states of a different item j (same family, different parent, gold and before-state; same token count if possible), right-aligned | Supa | 8, 16, 24 |
| **`T2S_dmis@L`** (added) | additive at every real prompt position of item i: h ← h_own + (h_T(j) − h_S(j)), using the T2S_mis donor j, right-aligned | Supa | 24 |
| **`HlowT@L`** (added) | Supa with Titan's blocks 0..L-1 (in effect 8..L-1) at **every** position, prompt and decode | Supa | 16, 24 |
| **`HupT@L`** (added) | Titan with Supa's blocks 0..L-1, i.e. Supa + Titan's blocks L..27, at every position | Titan | 16, 24 |

**How the hybrids are built:**
- In place on the model already loaded, so only one model is in memory.
- The other core's tensors are read one at a time through safetensors, as `runtime._apply_delta` does:
  - Titan's file is FP32 and is copied as is.
  - Supa's file is BF16 and is upcast, which is exactly what the FP32 load does.
- Every spliced block must hash-equal the source core's block, or the run aborts.
- The hybrids need no staging, no disk and no extra VRAM.

**KV and decode semantics of the prefill patches.** These semantics decide what each patch can show.
- Blocks below L cache their prompt keys and values from the receiver's own stream.
- Only blocks ≥ L see the patched prompt stream.
- Every generated token is processed by the receiver's own blocks, all of them.

L and R answers start with `{"` in both arms. On Titan's correct changed items, the first token that differs from echoing the before-state comes at:
- index 3–19 for L;
- index 3–13 for R;
- index 0–2 for P (checked with the tokenizer on `runs/bank_pilot/steer_cpu`).

So all of the L/R answer content is produced at decode steps, which the prefill patches never reach.

## 3. What each condition can and cannot show

**`T2S_all@L`**
- What it sets up: Supa's upper blocks see Titan's lower-delta encoding of the prompt. Supa's lower blocks still build the lower prompt KV and run every generated token.
- HIGH means that encoding is sufficient for Supa's upper blocks. A generic mode or rule unlock is not excluded; `T2S_dmis` tests that.
- LOW does not show that the lower delta is unneeded. The delta may act at decode positions, or it may need Titan's upper blocks.

**`S2T_all@L`**
- What it removes: only the lower delta's effect on Titan's block-L prompt stream. Titan's delta blocks 8..L-1 still run every decode token over Titan's own lower prompt KV.
- HIGH is predicted both by computation above L and by lower-delta work at decode (the "leak"), so it is uninformative on its own.
- LOW means Titan's upper blocks need the lower delta's prompt-stream output. If parse collapses instead, the two streams are incompatible.
- `S2T_all − HupT` is the leak share.

**`T2S_last@L`**
- Only one position is patched.
- For L/R it is a lower bound: decode steps copy payload digits from Supa's own stream, so `T2S_last << T2S_all` does not localise the information to earlier positions.
- It is least confounded on P, where Titan's whole answer is a few digit tokens starting at index 0.

**`T2S_mis@L`**
- This is a leak calibration, not a control for mode versus computation: both hypotheses predict that the output follows the donor.
- Read `donor_match_changed`, which counts only donors whose state changes. On P, 146 of 224 donors do not change, and echoing their before-state would also count as a match.
- Compare it with the paired `T2S_all@L` accuracy on the same donors (`_derived.mis_reference`).
  - If they are about equal, the patched upper stream overrides the receiver's own lower KV and decode path, and the T2S readings can be trusted.
  - A high receiver echo rate (`echo_before_changed`) means a large leak.
- Receiver accuracy is capped at A_S under every hypothesis.

**`T2S_dmis@L`**
- What it changes: it keeps item i's own Supa stream and the family-level part of the shift, and swaps the item-specific remainder of the shift for item j's.
- Read it only where `T2S_all@L` is HIGH.

**`HlowT@L`** applies the lower delta at every position, including decode and the lower KV. There is no leak.

**`HupT@L`** applies the upper delta at every position, on top of Supa's lower stack.

**Built-in identity checks:**
- `HlowT@L` must give the same first token as `T2S_all@L` on every item. The prompt computation above L is identical by construction, and pad positions are masked as keys.
- `HupT@L` must likewise match `S2T_all@L`.
- `_derived.first_token_identity` should be 1.0. A lower value can only come from cross-load nondeterminism.

**`ft_donor`** (first-token agreement):
- uninformative for L/R;
- for P, it mixes the mode switch with the first digit. The constant offset alone switches P's first token from `{"` to a digit (R5).

## 4. Decision table

The rule, fixed before the run:
- Computed per family, on changed items. For `T2S_mis` and `T2S_dmis`, only exact-length pairs are used.
- FR from the per-item results; HI if FR ≥ 0.7, LO if FR ≤ 0.2, otherwise MID.
- FR is undefined if A_T − A_S < 0.10. In the CPU preview the gaps were L 0.40, P 0.46 and R 0.57, so FR is defined for all three families.
- Each FR comes with a 95% CI from a paired parent-cluster bootstrap: 2000 draws, seed 17, parents resampled.
- `run_patch.py` computes all of this into `patch_results.json["_derived"]` and logs one decision line per condition.

| T2S_all@L | HlowT@L | HupT@L | Reading at cut L |
|---|---|---|---|
| HI | HI | LO | Delta blocks 8..L-1 compute the transition into the **prompt** stream, and Supa's upper blocks read it out. `T2S_dmis@L` then decides between the two readings in the next table. |
| LO | HI | LO | The lower delta is necessary and sufficient, but it works on decode tokens or through the lower-layer prompt KV, not through the block-L prompt stream. |
| LO | LO | HI | The computation is in delta blocks ≥ L. `S2T_all − HupT` is the leak share. |
| LO | LO | LO | Both halves are needed (serial or distributed computation), or the stitching is incompatible. Check parse and format: a collapse means incompatibility. |
| HI | HI | HI | The two halves are redundant. |
| HI | LO | any | Not a clean cut. Titan's lower blocks at decode positions conflict with Supa's upper blocks (incompatibility). |
| any MID | | | Partial: the computation straddles L. |

**How to read `T2S_dmis@L`** (only when `T2S_all@L` is HI):

| T2S_dmis@L outcome | Reading |
|---|---|
| receiver accuracy HI (close to `T2S_all@L`) | A generic family-level mode/rule signal unlocks Supa's own computation. Check per operation; R9 saw this kind of effect on R swap. |
| receiver accuracy LO, with format classes like `T2S_all@L` | Item-specific computation below L. |
| `donor_match_changed` > 0 | The shift itself carries item j's answer. |
| parse or format collapses | Off-manifold; uninterpretable. |

**Other readings:**

| Pattern | Reading |
|---|---|
| `S2T_all@L` LO | Titan's upper blocks need the lower delta's block-L prompt stream, even though Titan's lower decode path was intact. |
| Parse or format moves but accuracy does not | A format/mode effect only (R9). |

**Across L:**
- The cut at which `HlowT(L)` rises from A_S (L ≤ 8) to A_T (L = 28) is where the computation completes.
- The cut at which `HupT(L)` falls shows which blocks are necessary.

**At L = 8:**
- `T2S_last`, `T2S_all` and `S2T_all` are exact no-ops (`noop_exact` must be true): a full-scale check of the cross-model staging on the real GPU.
- `T2S_mis@8` is the leak calibration with only blocks 0–7 of the receiver's own KV.

## 5. Sanity gates

Any failure aborts with exit 3 before that model's results are written.

**After every model load (A Titan, B Supa, C Titan)**, on the first 4 items of each family (12 items, in their own batches):
- Identity patches at every L, both `all` and `last`, using the model's own states staged on disk, must reproduce the unpatched texts exactly, with patch max|d| = 0.
- At each `--dmis-layers` L, the additive path with a zero difference (own − own) must also reproduce them exactly.
- A zero vector at the last position must change at least one text, which shows the hook is live.
- Each hook must fire exactly once per `generate()`, during prefill, and never on decode steps (counts are checked).
- Donor and receiver token ids must be equal per row.
- Every sanity record now also stores its reference texts.

**`base:sparse` is only committed after both models' sanity checks have passed.**

**Splice sanity, before the first hybrid on each host:**
- Every block that differs is spliced in from the other core and must hash-equal that core's block. This is a hard gate.
- The spliced model must then reproduce the other core's sanity texts. If it doesn't, a WARNING is logged and the result recorded. With equal hashes, a mismatch can only be cross-load nondeterminism.
- No hybrid result can be committed unless this process passed the splice sanity for that host.

## 6. Runtime, memory and disk projection

Figures come from `python scripts/run_patch.py --device cpu --plan`, which only tokenizes and loads no model.

| | Spec conditions only | With the additions (default) |
|---|---|---|
| Conditions | 14 | 19 (+ `T2S_dmis@24`, `HlowT@16/24`, `HupT@16/24`) |
| Sanity generations | 3 loads × 120 | 3 loads × 132 + 2 × 12 splice sanity |
| Greedy generations | 9,320 | **12,580** (+3,260, **+35.0%**) |
| `generate()` calls (16-row batches) | – | 865 |
| Hours at 2.7 generations/s | 0.96 | **1.29** |
| Hours at 4.4 generations/s (post-reboot rate) | 0.59 | 0.79 |

**Other time costs:**
- Loading and hashing three models: about 5 min.
- Hybrid splices (reading 8–20 blocks per step, then hashing them): about 1 min in total.
- Thermal pauses, if any.

**Upper bound:** about 2 h. Supa-like outputs, and patched outputs that drift, can run to the 160-token cap. Steering runs did this on P and R.

**GPU memory:**
- One FP32 model (2.4 GB) plus KV cache at batch 16 (about 1.1 GiB), inside the 5.6 GB per-process cap (0.70 × 8 GB).
- A splice adds at most about 25 MB, transiently: one tensor plus its FP32 upcast.

**Host RAM:**
- One model at a time, loaded with `device_map` straight onto the GPU.
- Splice reads are mmap'd one tensor at a time (≤ 12.6 MB each).
- Hashing copies at most 64 MiB to the host at a time.

**Disk:**
- Staging: 80,848 prompt tokens (111–143 per item) × 1024 × 4 B × 3 layers = 0.93 GiB per model, 1.85 GiB at peak, plus about 20 MB of sanity staging.
- The script refuses to start unless free space covers the need plus 2 GiB. 38 GiB is free now.
- Staged arrays are deleted at the end unless `--keep-states` is given.

**Mismatch donors:**
- 634 of 640 pairs have equal length. The other 6 are ledger items whose donor is one token longer; mis/dmis aggregates are reported without them.
- Donors are reused: 122, 148 and 133 distinct donors per family, each used at most 4–5 times.

## 7. Outputs

All under `runs/bank_pilot/patch/`:
- `patch_results.json`: per condition, `score`, `correct`, `parse`, `format`, `echo_before_changed`, `first_token`, the patch counters, `ft_*` and `noop_*`.
  - Mis/dmis conditions add `donor_match`, `donor_match_changed`, `echo_donor_before_changed` and `exact_len_subset`.
  - Hybrid conditions add `differing_blocks_from_src` and `ft_equals_counterpart`.
  - The file also contains `_meta`, `_sanity` and `_derived`.
- `patch_sanity.json`
- `gen_<cond>.json`: the generated texts.
- `patch_progress.json`: present only while the run is in progress.

## 8. Review disposition (builder → hooks and design verifiers)

**Applied:**

- **MUST-FIX, both verifiers: the interpretation table overread S2T/T2S.**
  - Rewrote the table (docstring and section 4).
  - Added a "what each condition can and cannot show" section.
  - Marked `ft_donor` as uninformative for L/R.
  - Added the weight-spliced hybrids, which remove the decode-time leak.
- **MUST-FIX, design: T2S_mis is not a mode-vs-computation control, and donor_match is inflated.**
  - Deleted those rows; `T2S_mis` is now described as a leak calibration.
  - Added `donor_match_changed`, the echo rates (receiver and donor before-state), `exact_len_subset`, and the paired references in `_derived` (`T2S_all@L` and `base:sparse` accuracy on the same donors).
  - The mode-vs-item question moved to `T2S_dmis`.
- **SHOULD, hooks: the 6 length-mismatched pairs shift the prefix and the attention sink.**
  - Right alignment is kept, because the spec prescribed it.
  - Mis/dmis aggregates are reported on exact-length pairs (`exact_len_subset`; `_derived` uses only those), and the 6 items are listed.
- **SHOULD, design: T2S_last is confounded for L/R.** Rewrote its reading as a lower bound for L/R, least confounded on P. The printed `ft_donor` is now P-only.
- **SHOULD, design: addition 1 (stitched hybrids).** Adopted at `--hybrid-layers 16,24`, with the splice hash gate, the endpoint check and the first-token identity checks.
- **SHOULD, design: addition 2 (difference swap).** Adopted at `--dmis-layers 24` only.
  - Running it at 16 and 24 as well would bring the additions to +41.5%, over the 40% budget.
  - L = 24 is the more useful layer for two reasons:
    - `T2S_dmis` can only be read where `T2S_all` is HI, which is more likely at 24.
    - The item-specific remainder is larger at 24 (constant share 70–84%, versus 93–99.6% at L12–20).
- **NOTE, design: make the readings categorical.** Implemented: fixed FR thresholds, paired parent-cluster bootstrap CIs and format classes in `_derived` and in each record.

**Rejected:**

- **SHOULD, design: drop L = 8 from the default.** Kept `--layers 8,16,24`, for three reasons:
  - The spec fixed it, and `master_chain.sh` is frozen and passes no `--layers`.
  - L = 8 is the only full-scale (640 items, both directions) exact-no-op check of the cross-model staging on the real GPU.
  - `T2S_mis@8` calibrates the leak through blocks 0–7 of the receiver's own KV.
  - Cost: 2,560 generations, about 16 min at 2.7/s.
  - To drop it, change the launch line to `--layers 16,24`. That edits `master_chain.sh` and needs the owner's approval.
- **Segment-wise alignment for non-exact pairs** (hooks SHOULD option b): not adopted. It deviates from the spec's right alignment, and filtering the 6 pairs is enough.

**No change needed:**
- Hooks NOTE: the mechanics were independently confirmed.
- Hooks NOTE: one-model-at-a-time lease wiring confirmed.
- Design NOTE: decode-time activation patching (needs two models in lockstep, which breaks the one-model / 5.6 GB rule; the hybrids cover the question).
- Design NOTE: a prefix-only patch (third priority).

## 9. Caveats and post-hoc analysis

**Caveats:**
- Exploratory: these are pilot TEST items, and TEST was seen during development.
- One greedy decode per condition.
- The patches are prefill-only.
- Items cluster by parent (26–28 parents per family), so read the CIs.
- P has only 87 changed items, so P's FR is coarse.
- Bit-exact identities on the real GPU rest on deterministic single-stream cuBLAS/SDPA. Within a load this is gated; across loads it is recorded.

**Post hoc, no GPU, from `gen_*.json`:**
- Break results down by operation and status, following R8:
  - L: void vs posted;
  - P: applicable vs inapplicable;
  - R: swap / add / cond / mix.
- Compare format classes against accuracy per condition.

## 10. Test (`python scripts/patch_test_tiny.py`)

**Setup:**
- CPU only, with `CUDA_VISIBLE_DEVICES=""` and `is_available` patched. No real weights are touched.
- Two tiny random Qwen3 models: 4 blocks, hidden 64, real vocab.
  - "supa" is BF16-exact and saved as a BF16 safetensors file.
  - "sparse" shares the embeddings, final norm and block 0 with "supa", and is saved as FP32.
- 18 real pilot prompts.
- Flags: `--layers 1,2,4 --dmis-layers 1,4 --hybrid-layers 2,4`, giving 20 conditions.

**Coverage:**
- item selection
- alignment, including the additive dmis patch
- hook guards
- crash and resume
- fresh run equals resumed run
- the sanity checks, including the additive zero-diff check
- exact no-ops at L = 1, including `T2S_dmis@1`
- the donor's first token at L = 4
- `T2S_dmis@4` first token equals the independently computed head(norm(S_i + T_j − S_j))
- splice sanity on both hosts
- `HlowT@4` equals `base:sparse` and `HupT@4` equals `base:supa`, exactly
- `HlowT@2` / `HupT@2` first tokens equal `T2S_all@2` / `S2T_all@2`
- a rerun of a finished directory, and `--plan`
- sanity abort
- the `--device cuda` wiring with a stubbed runtime
- the wrong-splice-source hash gate
- `derived()` on synthetic data, including a check that the bootstrap resamples parents, not items

**Last result (2026-10-07):** `ALL 56/56 CHECKS PASSED in 139s; peak RSS 887 MiB`, exit 0.
