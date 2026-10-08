"""Generative baselines: what the frozen model can SAY about a plan's consequences.

  e2e        one greedy answer to the k-event prompt (multi format)
  chain_one  k chained greedy calls in Titan's trained one-step frame; each call receives the
             model's own previous answer as `before` (a self-rollout, no oracle re-anchoring)
  chain_m1   the same chain, but each call uses the multi-step rule with a single event, which
             states the inapplicable-is-no-op rule explicitly

Decoding follows the programme scorers: greedy (generation_config samples by default, so
do_sample=False is forced), EOS list [151645, 151643], thinking disabled, FP32.
"""
from __future__ import annotations

import torch

from . import env
from .capture import render
from .runtime import breathe, throttle_list

EOS = [151645, 151643]
PAD = 151643


@torch.no_grad()
def generate_texts(model, tok, message_lists: list[list[dict]], max_new_tokens: int,
                   max_batch: int = 32, token_budget: int = 8000) -> list[str]:
    """Greedy batches sized so batch x (prompt + new tokens) <= token_budget: the FP32 KV cache is
    224 KiB/token, and overflowing 8 GB VRAM pages into host memory (observed 2026-10-06).
    PREREG-1 Amendment 1 (operational): 2000/16 -> 8000/32 so 160-token caps do not shrink
    batches to ~5 sequences; worst-case KV ~1.8 GB, within the 0.70 VRAM cap."""
    prompts = [render(tok, m) for m in message_lists]
    enc_all = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(enc_all[i]))
    out = [None] * len(prompts)
    batches, cur = [], []
    for i in order:
        width = len(enc_all[i]) + max_new_tokens
        if cur and (len(cur) + 1 > max_batch or (len(cur) + 1) * width > token_budget):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    for idx in batches:
        L = max(len(enc_all[i]) for i in idx)
        ids = torch.full((len(idx), L), PAD, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):                       # left padding for generation
            x = enc_all[i]
            ids[r, L - len(x):] = torch.tensor(x)
            att[r, L - len(x):] = 1
        g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), do_sample=False,
                           max_new_tokens=max_new_tokens, eos_token_id=EOS, pad_token_id=PAD,
                           temperature=None, top_p=None, top_k=None, logits_processor=throttle_list())
        for r, i in enumerate(idx):
            new = g[r, L:].tolist()
            while new and new[-1] in EOS:
                new = new[:-1]
            if any(t in EOS for t in new):
                new = new[: next(j for j, t in enumerate(new) if t in EOS)]
            out[i] = tok.decode(new, skip_special_tokens=True).strip()
        breathe()
    return out


@torch.no_grad()
def sample_with_state(model, tok, messages: list[dict], n: int, temperature: float, max_new_tokens: int,
                      layer: int, seed: int) -> dict:
    """For ONE prompt: the answer-slot hidden state at `layer`, the greedy answer, and n samples,
    each with its mean token log-probability under the model (for logprob-rerank baselines)."""
    text = render(tok, messages)
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    h = out.hidden_states[layer][0, -1].float().cpu().numpy()
    res = {"h": h, "cands": []}
    for mode in ("greedy", "sample"):
        kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None, num_return_sequences=1) if mode == "greedy" \
            else dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0, num_return_sequences=n)
        if mode == "sample":
            torch.manual_seed(seed)
        g = model.generate(input_ids=ids, max_new_tokens=max_new_tokens, eos_token_id=EOS, pad_token_id=PAD,
                           output_scores=True, return_dict_in_generate=True, **kw)
        trans = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
        L = ids.shape[1]
        for r in range(g.sequences.shape[0]):
            new = g.sequences[r, L:].tolist()
            lp = trans[r].float().cpu().numpy()
            cut = len(new)
            for j, t in enumerate(new):
                if t in EOS:
                    cut = j
                    break
            mean_lp = float(lp[: cut + 1].mean()) if cut + 1 <= len(lp) else float(lp.mean())
            res["cands"].append({"text": tok.decode(new[:cut], skip_special_tokens=True).strip(),
                                 "mean_logprob": mean_lp, "greedy": mode == "greedy"})
    return res


def _decode_rows(tok, seqs, scores_trans, L):
    out = []
    for r in range(seqs.shape[0]):
        new = seqs[r, L:].tolist()
        lp = scores_trans[r].float().cpu().numpy()
        cut = len(new)
        for j, t in enumerate(new):
            if t in EOS:
                cut = j
                break
        n_lp = min(cut + 1, len(lp))
        out.append((tok.decode(new[:cut], skip_special_tokens=True).strip(),
                    float(lp[:n_lp].mean()) if n_lp else 0.0))
    return out


@torch.no_grad()
def sample_batch(model, tok, message_lists: list[list[dict]], n: int, temperature: float,
                 max_new_tokens: int, layer: int, seed: int) -> list[dict]:
    """Batched version of sample_with_state over prompts of ONE family (similar lengths).
    Left padding; RoPE is relative, so answer-slot states match unpadded ones up to float error."""
    prompts = [render(tok, m) for m in message_lists]
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    L = max(len(x) for x in enc)
    ids = torch.full((len(enc), L), PAD, dtype=torch.long)
    att = torch.zeros((len(enc), L), dtype=torch.long)
    for r, x in enumerate(enc):
        ids[r, L - len(x):] = torch.tensor(x)
        att[r, L - len(x):] = 1
    ids, att = ids.cuda(), att.cuda()
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=att, position_ids=pos, output_hidden_states=True,
                use_cache=False, logits_to_keep=1)
    H = out.hidden_states[layer][:, -1].float().cpu().numpy()
    del out
    g = model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=max_new_tokens,
                       eos_token_id=EOS, pad_token_id=PAD, temperature=None, top_p=None, top_k=None,
                       output_scores=True, return_dict_in_generate=True, logits_processor=throttle_list())
    greedy = _decode_rows(tok, g.sequences, model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True), L)
    torch.manual_seed(seed)
    s = model.generate(input_ids=ids, attention_mask=att, do_sample=True, temperature=temperature, top_p=1.0,
                       top_k=0, num_return_sequences=n, max_new_tokens=max_new_tokens, eos_token_id=EOS,
                       pad_token_id=PAD, output_scores=True, return_dict_in_generate=True, logits_processor=throttle_list())
    samp = _decode_rows(tok, s.sequences, model.compute_transition_scores(s.sequences, s.scores, normalize_logits=True), L)
    breathe()
    res = []
    for r in range(len(enc)):
        cands = [{"text": greedy[r][0], "mean_logprob": greedy[r][1], "greedy": True}]
        cands += [{"text": t, "mean_logprob": lp, "greedy": False} for t, lp in samp[r * n:(r + 1) * n]]
        res.append({"h": H[r], "cands": cands})
    return res


def max_tokens(fam: str) -> int:
    # 160 for every family and arm (PREREG-1 M1). History: P 12 -> 40 and L/R 56 -> 64 at 13:45
    # (verification B3); 64 still truncated base Supa's wrapped answers. Titan's L/R answers are
    # <= 34 characters, so its greedy outputs are unaffected.
    return 160


def e2e(model, tok, items: list[dict]) -> list[str]:
    res = [None] * len(items)
    for fam in env.FAMILIES:
        ix = [i for i, it in enumerate(items) if it["family"] == fam]
        if not ix:
            continue
        txt = generate_texts(model, tok, [items[i]["messages"] for i in ix], max_tokens(fam))
        for i, t in zip(ix, txt):
            res[i] = t
    return res


def chain(model, tok, chains: list[dict], mode: str) -> list[dict]:
    """chains: [{family, s0, events}] -> per chain: list of predicted states (None once broken)."""
    states = [[c["s0"]] for c in chains]
    texts = [[] for _ in chains]
    H = max(len(c["events"]) for c in chains)
    for t in range(H):
        live = [i for i, c in enumerate(chains) if t < len(c["events"]) and states[i][-1] is not None]
        for fam in env.FAMILIES:
            ix = [i for i in live if chains[i]["family"] == fam]
            if not ix:
                continue
            if mode == "chain_one":
                msgs = [env.one_step_messages(fam, states[i][-1], chains[i]["events"][t]) for i in ix]
            elif mode == "chain_m1":
                msgs = [env.multi_step_messages(fam, states[i][-1], [chains[i]["events"][t]]) for i in ix]
            else:
                raise ValueError(mode)
            txt = generate_texts(model, tok, msgs, max_tokens(fam))
            for i, s in zip(ix, txt):
                texts[i].append(s)
                states[i].append(env.parse_state(fam, s, lenient=True))
        for i, c in enumerate(chains):
            if t < len(c["events"]) and states[i][-1] is None and len(states[i]) == t + 1:
                states[i].append(None)
    return [{"states": s, "texts": x} for s, x in zip(states, texts)]
