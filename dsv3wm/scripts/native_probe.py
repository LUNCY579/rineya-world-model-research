"""G1/G2 native parity and one-gradient smoke on permitted constructed fixtures."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from dsv3wm.config import Config
    from dsv3wm.native import runtime, load, sha256
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    cfg = Config(device=args.device, steps=1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('probe record already exists')
    result = {'status':'started', 'config':cfg.dict(), 'command':sys.argv}
    started = time.perf_counter()
    try:
        with runtime(cfg, 'native-probe'):
            import torch
            from dsv3wm.objectives import token_objective, SequentialMTP
            model, tok, lm = load(cfg)
            from bwmn import rw2
            prompt = rw2.forward_prompt(4, 'double', 16)
            label = rw2.forward_target(8, True, False)
            enc = lm.encode_pair(tok, prompt, label, 384)
            ids, labels, att = (x.to(next(model.parameters()).device) for x in lm.collate([enc], tok.pad_token_id))
            decoder, head = lm._decoder_and_head(model)
            with torch.no_grad():
                baseline, _ = lm.chunked_ce(model, ids, att, labels, chunk=8)
                experimental, _, h = token_objective(decoder, head, model.get_input_embeddings(), ids, att, labels, chunk=8)
            torch.testing.assert_close(baseline, experimental, rtol=1e-6, atol=1e-6)
            pm = lm.attach_lora(model, rank=4, alpha=8, dropout=0, targets=('q_proj','v_proj'), seed=0)
            pm.eval()
            pdec, phead = lm._decoder_and_head(pm)
            with torch.no_grad():
                initialized, _, ph = token_objective(pdec, phead, pm.get_input_embeddings(), ids, att, labels, chunk=8)
                torch.testing.assert_close(h, ph, rtol=0, atol=0)
                torch.testing.assert_close(baseline, initialized, rtol=1e-6, atol=1e-6)
                cutoff = len(tok(prompt, add_special_tokens=False).input_ids)
                changed = ids.clone()
                changed[:, cutoff:] = tok.pad_token_id
                alternate = pdec(input_ids=changed, attention_mask=att, use_cache=False).last_hidden_state
                torch.testing.assert_close(ph[:, :cutoff], alternate[:, :cutoff], rtol=0, atol=0)
            pm.train()
            pm.config.use_cache = False
            pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            pm.enable_input_require_grads()
            mtp = SequentialMTP(model.config.hidden_size, 16, 2).to(ids.device)
            loss, metrics, _ = token_objective(pdec, phead, pm.get_input_embeddings(), ids, att, labels, mtp, chunk=8)
            loss.backward()
            trainable = [p for p in pm.parameters() if p.requires_grad] + list(mtp.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
            result.update(status='passed', native_parent_sha256=sha256(lm.MODELS[cfg.core]/'model.safetensors'),
                          baseline_ce=float(baseline), experimental_ce=float(experimental),
                          initialized_lora_ce=float(initialized), auxiliary_loss=float(loss.detach()),
                          causal_prefix_max_error=0.0, disabled_lora_max_error=0.0,
                          input_tokens=ids.numel(), supervised_tokens=metrics['base_tokens'], auxiliary_tokens=metrics['aux_tokens'],
                          finite_gradient_parameters=sum(p.numel() for p in trainable),
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if cfg.device=='cuda' else 0,
                          torch_version=torch.__version__, transformers_version=__import__('transformers').__version__)
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['duration_s'] = time.perf_counter() - started
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
