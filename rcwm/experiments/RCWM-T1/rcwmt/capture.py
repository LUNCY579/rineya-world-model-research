"""Hidden-state capture at action boundaries.

For every item we record the residual stream at:
  * the answer slot: the last prompt token (after `<think>\\n\\n</think>\\n\\n`), where the model
    must already hold whatever it will use to emit the consequence;
  * each event boundary: the token containing the closing brace of event j in the user JSON.
    Causal attention means this position has seen the start state and events 1..j only, so it is
    a prospective encoding of the prefix plan, never of later events.

Batches are right-padded; captured positions are always before padding, and G0 checks batched
values against unpadded single-sequence forwards and against a truncated-prefix forward.
"""
from __future__ import annotations

import numpy as np
import torch

from . import env

LAYERS = (0, 4, 8, 12, 16, 20, 24, 28)   # hidden_states index: 0 = embeddings, i = after block i (28 = final norm)


def render(tok, messages) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def encode_item(tok, item: dict) -> dict:
    """Token ids plus capture positions. Event-boundary positions only for multi-format items."""
    text = render(tok, item["messages"])
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    ev_pos = []
    if item["format"] == "multi":
        user = item["messages"][1]["content"]
        base = text.index(user)
        assert text.count(user) == 1
        for c in env.event_end_char_offsets(user, item["k"]):
            ch = base + c
            tix = next(i for i, (a, b) in enumerate(offs) if a <= ch < b)
            ev_pos.append(tix)
    return {"ids": ids, "last": len(ids) - 1, "events": ev_pos}


@torch.no_grad()
def forward_hidden(model, batch_ids: list[list[int]], layers=LAYERS):
    """Right-padded batch forward. Returns hidden_states tuple (CPU not yet)."""
    L = max(len(x) for x in batch_ids)
    pad = 151643
    ids = torch.full((len(batch_ids), L), pad, dtype=torch.long)
    att = torch.zeros((len(batch_ids), L), dtype=torch.long)
    for i, x in enumerate(batch_ids):
        ids[i, : len(x)] = torch.tensor(x)
        att[i, : len(x)] = 1
    out = model(input_ids=ids.cuda(), attention_mask=att.cuda(), output_hidden_states=True,
                use_cache=False, logits_to_keep=1)
    return out.hidden_states


@torch.no_grad()
def capture(model, encoded: list[dict], max_batch_tokens: int = 2000, layers=LAYERS,
            log_every: int = 50, last_out=None, events_out=None, ev_row=None):
    """Write answer-slot states to last_out [N, nL, D] and event-boundary states of items listed
    in ev_row {item_index: row} to events_out [M, kmax, nL, D] (both fp16, may be memmaps).
    With no outputs supplied, allocates in memory and returns (last, {i: [k, nL, D]})."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["ids"]))
    N, D = len(encoded), model.config.hidden_size
    last = last_out if last_out is not None else np.zeros((N, len(layers), D), dtype=np.float16)
    events: dict[int, np.ndarray] = {}
    nonfinite = 0
    batches, cur, cur_max = [], [], 0
    for i in order:
        n = len(encoded[i]["ids"])
        if cur and max(cur_max, n) * (len(cur) + 1) > max_batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, n)
    if cur:
        batches.append(cur)
    for bi, b in enumerate(batches):
        hs = forward_hidden(model, [encoded[i]["ids"] for i in b], layers)
        stack = torch.stack([hs[l] for l in layers], dim=0)          # [nL, B, T, D]
        for j, i in enumerate(b):
            e = encoded[i]
            v = stack[:, j, e["last"], :].float().cpu().numpy().astype(np.float16)
            nonfinite += int((~np.isfinite(v)).sum())
            last[i] = v
            if e["events"] and (ev_row is None or i in ev_row):
                w = stack[:, j, e["events"], :].permute(1, 0, 2).float().cpu().numpy().astype(np.float16)
                nonfinite += int((~np.isfinite(w)).sum())
                if events_out is not None:
                    events_out[ev_row[i], : w.shape[0]] = w
                else:
                    events[i] = w
        del hs, stack
        from .runtime import breathe
        breathe()
        if log_every and bi % log_every == 0:
            print(f"    batch {bi + 1}/{len(batches)} (B={len(b)}, T={len(encoded[b[-1]]['ids'])})", flush=True)
    if nonfinite:
        raise FloatingPointError(f"{nonfinite} non-finite values after fp16 conversion")
    return last, events


@torch.no_grad()
def parity_check(model, encoded: list[dict], n: int = 32, layers=LAYERS) -> dict:
    """G0: batched-padded vs single unpadded vs truncated-prefix forward at captured positions."""
    idx = list(range(0, len(encoded), max(1, len(encoded) // n)))[:n]
    sub = [encoded[i] for i in idx]
    last_b, ev_b = capture(model, sub, layers=layers, log_every=0)
    worst = {"batch_vs_single_last": 0.0, "batch_vs_single_event": 0.0, "prefix_vs_bulk_event": 0.0}
    rel = {k: 0.0 for k in worst}
    for j, e in enumerate(sub):
        hs = forward_hidden(model, [e["ids"]], layers)
        single = torch.stack([hs[l][0] for l in layers], 0).float().cpu().numpy()   # [nL, T, D]
        a = single[:, e["last"], :]
        d = np.abs(a - last_b[j].astype(np.float32)).max()
        worst["batch_vs_single_last"] = max(worst["batch_vs_single_last"], float(d))
        rel["batch_vs_single_last"] = max(rel["batch_vs_single_last"], float(d / (np.abs(a).max() + 1e-9)))
        for q, p in enumerate(e["events"]):
            s = single[:, p, :]
            d = np.abs(s - ev_b[j][q].astype(np.float32)).max()
            worst["batch_vs_single_event"] = max(worst["batch_vs_single_event"], float(d))
            rel["batch_vs_single_event"] = max(rel["batch_vs_single_event"], float(d / (np.abs(s).max() + 1e-9)))
            hp = forward_hidden(model, [e["ids"][: p + 1]], layers)
            pre = torch.stack([hp[l][0, p] for l in layers], 0).float().cpu().numpy()
            d = np.abs(pre - s).max()
            worst["prefix_vs_bulk_event"] = max(worst["prefix_vs_bulk_event"], float(d))
            rel["prefix_vs_bulk_event"] = max(rel["prefix_vs_bulk_event"], float(d / (np.abs(s).max() + 1e-9)))
    return {"n_items": len(sub), "max_abs": worst, "max_rel_to_rowmax": rel,
            "note": "fp16 storage rounding (~5e-4 relative) bounds batch-vs-single comparisons from below"}
