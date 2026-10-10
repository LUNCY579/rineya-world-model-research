"""CPU LoRA SFT on a frozen Rineya core with the exact BWM D-03 / RCWM T2 recipe.

Recipe parity (copied from RCWM-T1/vendor_bwm/bwmn/lm.py, sha 3d3f8d00…, which vendors BWM native_v1/bwmn/lm.py;
functions encode_pair, collate, chunked_ce, wsd_lr and the train_sft loop are reproduced verbatim in logic):
LoRA r16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down; AdamW(0.9, 0.95, eps 1e-8, wd 0); WSD schedule
(10% warmup, 70% flat, 20% linear decay to 10%); token-mean CE on target + EOS only; batch 16 as 4 micro-batches
weighted by supervised tokens; grad clip 1.0; gradient checkpointing; explicit row order for matched arms.
Differences (CPU only, documented): no CUDA calls; the vendored gpu_yield()/thermal_guard() (which spawns nvidia-smi
every 10 s, I-27) is replaced by rcx.guard.ThermalGuard.check() between micro-batches; FP32 throughout.
The registered T2 adapter is never touched here. Adapters are saved unmerged (D-28).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

CORES = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
PATHS = {"sparse": CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze",
         "supa": CORES / "Rineya-Supa-1.7.0-devfreeze"}
EOS_IDS = [151645, 151643]
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_core(name="sparse"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(PATHS[name]), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(PATHS[name]), dtype=torch.float32, low_cpu_mem_usage=True,
                                                 local_files_only=True, attn_implementation="sdpa").eval()
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    assert tok.eos_token_id == 151645 and tok.pad_token_id == 151643
    assert all(p.dtype == torch.float32 for p in model.parameters())
    return model, tok


def attach_lora(model, seed=0, rank=16, alpha=32, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(TARGETS), bias="none",
                     task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


@dataclass
class Encoded:
    input_ids: list
    labels: list
    n_target: int


def encode_pair(tok, prompt, target, max_len, eos=True):
    p = tok(prompt, add_special_tokens=False).input_ids
    t = tok(target, add_special_tokens=False).input_ids
    if eos:
        t = t + [tok.eos_token_id]
    ids = p + t
    if len(ids) > max_len:
        return None
    return Encoded(ids, [-100] * len(p) + t, len(t))


def collate(batch, pad_id):
    L = max(len(e.input_ids) for e in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        lab[i, :n] = torch.tensor(e.labels)
        att[i, :n] = 1
    return ids, lab, att


def _decoder_and_head(model):
    m = model
    if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
        m = m.base_model.model
    return m.model, m.lm_head


def _ce_sum_chunk(head, hs, tgt):
    return F.cross_entropy(head(hs).float(), tgt, reduction="sum")


def chunked_ce(model, ids, att, lab, chunk=256):
    from torch.utils.checkpoint import checkpoint
    dec, head = _decoder_and_head(model)
    h = dec(input_ids=ids, attention_mask=att).last_hidden_state
    tgt = lab[:, 1:]
    mask = tgt != -100
    hs = h[:, :-1][mask]
    t = tgt[mask]
    total = hs.new_zeros((), dtype=torch.float32)
    for i in range(0, hs.shape[0], chunk):
        total = total + checkpoint(_ce_sum_chunk, head, hs[i:i + chunk], t[i:i + chunk], use_reentrant=False)
    return total / t.numel(), t.numel()


def wsd_lr(step, total, peak, warm=0.1, decay=0.2, floor=0.1):
    w = max(1, int(total * warm))
    d0 = int(total * (1 - decay))
    if step < w:
        return peak * (step + 1) / w
    if step < d0:
        return peak
    prog = (step - d0) / max(1, total - d0)
    return peak * (1 - (1 - floor) * prog)


def row_order_for(n_rows, steps, batch, seed):
    """Same construction as the vendored train_generic.py (rng seeded 10007*seed+17)."""
    rng = random.Random(10007 * seed + 17)
    order = []
    while len(order) < steps * batch:
        idx = list(range(n_rows))
        rng.shuffle(idx)
        order.extend(idx)
    return order


def train_sft(model, tok, rows, *, steps, batch_size=16, lr=1e-4, max_len=512, seed=0, log_every=10,
              logger=print, grad_clip=1.0, row_order=None, micro=4, guard=None, ckpt_dir=None, ckpt_every=0):
    """Masked next-token SFT (D-03). Supports resume from ckpt_dir (adapter weights + optimizer + step), only into the
    identical run: a checkpoint written with other steps/batch/lr/seed/micro/clip, encoded rows or row order is
    refused (resuming it would silently splice two different trainings under the new run's provenance)."""
    random.seed(seed)
    torch.manual_seed(seed)
    enc = []
    for r in rows:
        e = encode_pair(tok, r["prompt"], r["target"], max_len)
        if e is None:
            raise ValueError("overlength row; refuse to train with silent drops")
        enc.append(e)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    if row_order is None:
        row_order = row_order_for(len(enc), steps, batch_size, seed)
    run_id = {"steps": steps, "batch_size": batch_size, "lr": lr, "seed": seed, "micro": micro, "grad_clip": grad_clip,
              "rows_sha256": hashlib.sha256(json.dumps([[e.input_ids, e.labels] for e in enc]).encode()).hexdigest(),
              "order_sha256": hashlib.sha256(json.dumps([int(i) for i in row_order[:steps * batch_size]])
                                             .encode()).hexdigest()}
    start, hist, gns = 0, [], []
    tokens_seen = sup_tokens = 0
    t_prev = 0.0
    if ckpt_dir and (Path(ckpt_dir) / "state.pt").exists():
        st = torch.load(Path(ckpt_dir) / "state.pt", map_location="cpu", weights_only=False)
        if st.get("run_id") != run_id:
            raise ValueError(f"{Path(ckpt_dir) / 'state.pt'} was written by a different run (args, rows or row order "
                             f"differ: {st.get('run_id')} vs {run_id}); refusing to resume")
        named = {n: p for n, p in model.named_parameters() if p.requires_grad}
        for n, v in st["lora"].items():
            named[n].data.copy_(v)
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["torch_rng"])
        random.setstate(st["py_rng"])
        start, hist, gns = st["step"], st["hist"], st["gns"]
        tokens_seen, sup_tokens, t_prev = st["tokens_seen"], st["sup_tokens"], st["wall_s"]
        logger(f"resumed from step {start}")
    model.train()
    t0 = time.time()
    guard_wait = 0.0
    for s in range(start, steps):
        b = [enc[row_order[s * batch_size + j]] for j in range(batch_size)]
        cur = wsd_lr(s, steps, lr)
        for g in opt.param_groups:
            g["lr"] = cur
        n_total = sum(e.n_target for e in b)
        opt.zero_grad(set_to_none=True)
        loss_val, n_sup = 0.0, 0
        for m0 in range(0, batch_size, micro):
            if guard is not None:
                tg = time.time()
                guard.check()
                guard_wait += time.time() - tg
            ids, lab, att = collate(b[m0:m0 + micro], tok.pad_token_id)
            mloss, mn = chunked_ce(model, ids, att, lab)
            if not torch.isfinite(mloss):
                raise FloatingPointError(f"non-finite loss at step {s}")
            (mloss * (mn / n_total)).backward()
            loss_val += float(mloss) * mn / n_total
            n_sup += mn
            tokens_seen += int(att.sum())
        gn = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if not torch.isfinite(gn):
            raise FloatingPointError(f"non-finite grad at step {s}")
        opt.step()
        sup_tokens += int(n_sup)
        hist.append(loss_val)
        gns.append(float(gn))
        if (s + 1) % log_every == 0 or s == start:
            k = min(log_every, len(hist))
            logger(f"step {s + 1}/{steps} loss {sum(hist[-k:]) / k:.4f} lr {cur:.2e} gn {float(gn):.2f} "
                   f"{t_prev + time.time() - t0:.0f}s guard_wait {guard_wait:.0f}s")
        if ckpt_dir and ckpt_every and (s + 1) % ckpt_every == 0 and s + 1 < steps:
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            lora = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            tmp = Path(ckpt_dir) / "state.pt.tmp"
            torch.save({"run_id": run_id, "lora": lora, "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                        "py_rng": random.getstate(), "step": s + 1, "hist": hist, "gns": gns,
                        "tokens_seen": tokens_seen, "sup_tokens": sup_tokens,
                        "wall_s": t_prev + time.time() - t0}, tmp)
            os.replace(tmp, Path(ckpt_dir) / "state.pt")
    model.eval()
    model.gradient_checkpointing_disable()
    return {"steps": steps, "batch_size": batch_size, "rows_available": len(enc), "lr_peak": lr,
            "final_loss_avg50": sum(hist[-50:]) / len(hist[-50:]),
            "loss_curve_every50": [round(sum(hist[i:i + 50]) / len(hist[i:i + 50]), 5) for i in range(0, len(hist), 50)],
            "max_grad_norm": max(gns), "tokens_processed": tokens_seen, "supervised_tokens": sup_tokens,
            "wall_s": round(t_prev + time.time() - t0, 1), "guard_wait_s": round(guard_wait, 1)}


def save_adapter(model, out_dir, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    (out_dir / "RUN_META.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


@torch.no_grad()
def generate(model, tok, prompts, max_new=64, max_batch_tokens=3000, guard=None, log=None):
    """Greedy, left-padded, token-bounded batches; returns strings in prompt order."""
    pad = tok.pad_token_id
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    outs = [None] * len(enc)
    i, nb, t0 = 0, 0, time.time()
    while i < len(order):
        L = len(enc[order[i]]) + max_new
        bs = max(1, min(32, max_batch_tokens // L))
        idx = order[i:i + bs]
        if guard is not None:
            guard.check()
        maxlen = max(len(enc[j]) for j in idx)
        ids = torch.full((len(idx), maxlen), pad, dtype=torch.long)
        att = torch.zeros((len(idx), maxlen), dtype=torch.long)
        for r, j in enumerate(idx):
            ids[r, maxlen - len(enc[j]):] = torch.tensor(enc[j])
            att[r, maxlen - len(enc[j]):] = 1
        out = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=EOS_IDS, pad_token_id=pad)
        for r, j in enumerate(idx):
            outs[j] = tok.decode(out[r, maxlen:], skip_special_tokens=True)
        i += bs
        nb += 1
        if log and nb % 10 == 0:
            log(f"  gen {i}/{len(order)} {time.time() - t0:.0f}s")
    return outs
