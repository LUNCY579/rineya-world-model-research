"""Inference FLOPs for decoder-only LMs (Qwen3 / Titan): one formula for the PlanBench comparators
(scripts/planbench_qwen.py) and the BWM planner (bwmn/blocks_planner.py callers), so iso-FLOP comparisons count the
same things on both sides.

Per token processed (prompt, forced or generated), with the model run causally over one growing sequence:
  dense      2 * N_nonemb                       (every non-embedding weight: one multiply-add)
  attention  4 * L * H * d_head * n_keys        (QK^T scores + softmax-weighted V, multiply-add each; n_keys = number
                                                 of unmasked keys = position + 1, i.e. causal pairs only; softmax,
                                                 RoPE and norms are not counted)
  lm_head    2 * d_model * vocab                (only for tokens whose logits are used: one per generated token)
So a segment of n_new tokens appended to n_ctx cached tokens costs
  2*N*n_new + 4*L*H*d_head*(n_new*n_ctx + n_new*(n_new+1)/2) + 2*d*V*n_logits.
Embedding lookups are free. Re-processing tokens (a KV cache that could not be reused, prefix re-prefill) is counted
again in full. LoRA adapters add their own parameters to N (extra_nonemb), unmerged or not.
Generated tokens include the stop token (as lm.py's n_tokens and planbench_qwen's gen_tokens count them); the formula
processes every token of the sequence once, so the never-fed-back last token is counted too, on both sides.
Iso-FLOP budgets: max_total_tokens inverts flops_breakdown for a given prompt length. BWM planner compute (proposer
samples sharing a prefilled prompt; F queries) uses shared_prefix_breakdown / flops_breakdown with the same constants,
so the comparator and the planner are counted by one formula (see scripts/planbench_qwen.py --iso-ledger).
Pure functions over config.json / safetensors headers: importing this module has no side effects and no torch import.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path


def flop_params(cfg_path: Path, extra_nonemb: int = 0) -> dict:
    """FLOP-relevant shape of a Qwen3-style decoder from its config.json. extra_nonemb: active adapter parameters."""
    c = json.loads(Path(cfg_path).read_text())
    d, L, h, kv, ff, V = (c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"],
                          c["intermediate_size"], c["vocab_size"])
    if c.get("use_sliding_window"):
        raise ValueError(f"{cfg_path}: sliding-window attention is not covered by the causal-pairs formula")
    hd = c.get("head_dim") or d // h
    attn = d * h * hd + 2 * d * kv * hd + h * hd * d
    if c.get("attention_bias"):
        attn += h * hd + 2 * kv * hd
    per_layer = attn + 2 * hd + 3 * d * ff + 2 * d  # q/k/v/o, q_norm+k_norm (Qwen3), gate/up/down, 2 RMSNorms
    return {"n_nonemb": L * per_layer + d + int(extra_nonemb), "d_model": d, "vocab": V, "n_layers": L, "n_heads": h,
            "head_dim": hd}


def safetensors_nonemb(model_dir: Path) -> int:
    """Non-embedding parameter count from safetensors headers only (no weights read)."""
    files = sorted(Path(model_dir).glob("*.safetensors"))
    n = 0
    for f in files:
        with open(f, "rb") as fh:
            hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in hdr.items():
            if k == "__metadata__" or "embed_tokens" in k or "lm_head" in k:
                continue
            p = 1
            for s in v["shape"]:
                p *= s
            n += p
    return n


def attn_pairs(n_ctx: int, n_new: int) -> int:
    """Causal (query, key) pairs when n_new tokens follow n_ctx cached ones: sum over t in [n_ctx, n_ctx+n_new) of
    (t + 1), each query attending to itself and every earlier position."""
    return n_new * n_ctx + n_new * (n_new + 1) // 2


def forward_flops(fp: dict, n_new: int, n_ctx: int = 0, n_logits: int = 0) -> int:
    """FLOPs to run n_new tokens through the model after n_ctx cached tokens, computing n_logits LM-head rows."""
    return (2 * fp["n_nonemb"] * n_new + 4 * fp["n_layers"] * fp["n_heads"] * fp["head_dim"] * attn_pairs(n_ctx, n_new)
            + 2 * fp["d_model"] * fp["vocab"] * n_logits)


def flops_breakdown(fp: dict, prompt: int, gen: int, forced: int = 0, reprefill: int = 0) -> dict:
    """One completion: the prompt, budget-forcing tokens and generated tokens form one causal sequence of length
    T = prompt + forced + gen (prefill + decode); `reprefill` tokens (a prefix processed a second time because a KV
    cache could not be reused) are added in full. Returns the terms and their total."""
    T = prompt + forced + gen
    dense = 2 * fp["n_nonemb"] * (T + reprefill)
    attn = 4 * fp["n_layers"] * fp["n_heads"] * fp["head_dim"] * (attn_pairs(0, T) + attn_pairs(0, reprefill))
    head = 2 * fp["d_model"] * fp["vocab"] * gen
    return {"dense": dense, "attention": attn, "lm_head": head, "total": dense + attn + head}


def generation_flops(fp: dict, prompt: int, gen: int, forced: int = 0, reprefill: int = 0) -> int:
    """Total of flops_breakdown. For K samples sharing one cached prompt (BWM proposer, shared-prefix sampler) use
    shared_prefix_breakdown (= forward_flops(fp, P) + sum_k forward_flops(fp, G_k, n_ctx=P, n_logits=G_k))."""
    return flops_breakdown(fp, prompt, gen, forced, reprefill)["total"]


def shared_prefix_breakdown(fp: dict, prompt: int, gens: list[int], sample_pairs: int | None = None) -> dict:
    """K samples continuing one prompt that is prefilled once (the BWM proposer's shared-prefix sampler): the prompt
    is processed once, then sample k processes its G_k tokens after the `prompt` cached ones. With one sample this
    equals flops_breakdown(fp, prompt, G). sample_pairs: when only sum(G_k) is logged (gens=[sum]), the caller passes
    its bound on sum_k G_k(G_k+1)/2, the within-sample causal pairs (see beam_proposer_pairs_bound)."""
    g = sum(gens)
    own = sum(x * (x + 1) // 2 for x in gens) if sample_pairs is None else int(sample_pairs)
    a = 4 * fp["n_layers"] * fp["n_heads"] * fp["head_dim"]
    dense = 2 * fp["n_nonemb"] * (prompt + g)
    attn = a * (attn_pairs(0, prompt) + prompt * g + own)
    head = 2 * fp["d_model"] * fp["vocab"] * g
    return {"dense": dense, "attention": attn, "lm_head": head, "total": dense + attn + head}


def beam_proposer_pairs_bound(gen_total: int, cap: int) -> int:
    """Upper bound on sum_k G_k(G_k+1)/2 when the per-sample lengths are unlogged but every G_k <= cap and
    sum_k G_k = gen_total (convexity: as many samples as possible at the cap)."""
    full, rest = divmod(int(gen_total), int(cap))
    return full * cap * (cap + 1) // 2 + rest * (rest + 1) // 2


def max_total_tokens(fp: dict, prompt: int, flop_cap: float, forced: int = 0) -> int | None:
    """Largest causal sequence length T (prompt + forced + generated) whose flops_breakdown total is <= flop_cap, with
    generated = T - prompt - forced. None if even T = prompt + forced exceeds the cap. Used for iso-FLOP budgets."""
    def cost(T):
        return flops_breakdown(fp, prompt, T - prompt - forced, forced)["total"]

    lo = prompt + forced
    if cost(lo) > flop_cap:
        return None
    hi = lo + 1
    while cost(hi) <= flop_cap:
        hi = lo + 2 * (hi - lo)
    while hi - lo > 1:  # invariant: cost(lo) <= cap < cost(hi)
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if cost(mid) <= flop_cap else (lo, mid)
    return lo


def safetensors_params(path: Path) -> int:
    """Parameter count of one safetensors file (e.g. a LoRA adapter_model.safetensors) from its header only."""
    with open(path, "rb") as fh:
        hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
    n = 0
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        p = 1
        for s in v["shape"]:
            p *= s
        n += p
    return n
