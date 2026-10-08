"""Off-task representation and behaviour shift (mechanism experiment M).

On a fixed neutral prompt set drawn from the family's preservation slice
(data/offtask_prompts.jsonl, 248 prompts, 62 reply classes), record per arm:
  * answer-slot residual states at capture.LAYERS
  * the full first-answer-token log-probability vector
Comparing an arm with Supa on these prompts measures how much a weight delta perturbs behaviour
it was NOT trained on. That is the representational side of forgetting, measured without any
retention instrument.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .capture import LAYERS, render

PROMPTS = Path(__file__).resolve().parents[1] / "data" / "offtask_prompts.jsonl"
PAD = 151643


def load_prompts():
    return [json.loads(l) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()]


@torch.no_grad()
def capture_offtask(model, tok, out_dir: Path, batch: int = 8) -> dict:
    rows = load_prompts()
    enc = [tok(render(tok, r["messages"]), add_special_tokens=False)["input_ids"] for r in rows]
    N = len(rows)
    H = np.lib.format.open_memmap(out_dir / "offtask_last.npy", mode="w+", dtype=np.float16, shape=(N, len(LAYERS), model.config.hidden_size))
    LP = np.lib.format.open_memmap(out_dir / "offtask_logprobs.npy", mode="w+", dtype=np.float16, shape=(N, model.config.vocab_size))
    order = sorted(range(N), key=lambda i: len(enc[i]))
    for s in range(0, N, batch):
        idx = order[s: s + batch]
        L = max(len(enc[i]) for i in idx)
        ids = torch.full((len(idx), L), PAD, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, L - len(enc[i]):] = torch.tensor(enc[i])
            att[r, L - len(enc[i]):] = 1
        ids, att = ids.cuda(), att.cuda()
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        out = model(input_ids=ids, attention_mask=att, position_ids=pos, output_hidden_states=True,
                    use_cache=False, logits_to_keep=1)
        hs = torch.stack([out.hidden_states[l][:, -1] for l in LAYERS], 1).float().cpu().numpy()
        lp = torch.log_softmax(out.logits[:, -1].float(), -1).cpu().numpy()
        for r, i in enumerate(idx):
            H[i] = hs[r].astype(np.float16)
            LP[i] = np.maximum(lp[r], -60.0).astype(np.float16)
        del out
        from .runtime import breathe
        breathe()
    H.flush(); LP.flush()
    meta = {"n": N, "layers": list(LAYERS), "prompts_sha256": __import__("hashlib").sha256(PROMPTS.read_bytes()).hexdigest(),
            "ids": [r["id"] for r in rows], "reply_class": [r["reply_class"] for r in rows]}
    (out_dir / "offtask_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return meta
