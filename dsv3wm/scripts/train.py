"""Native FP32 LoRA pilot/train CLI. No confirmation inputs or labels accepted."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from dsv3wm.config import Config
    from dsv3wm.native import runtime, load, response_prompt, sha256
    from dsv3wm.data import conditioning, target
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--distill', help='independently verified TRAIN-only trajectories for A3/A4')
    args = parser.parse_args()
    cfg = Config.read(args.config)
    rows = [json.loads(line) for line in Path(args.train).read_text(encoding='utf-8').splitlines()]
    if not rows or any(r.get('split') != 'train' for r in rows):
        raise ValueError('training accepts only labelled training rows')
    distill = []
    if cfg.arm in {'A3', 'A4'}:
        if not args.distill:
            raise ValueError('A3/A4 require independently verified distillation file')
        distill = [json.loads(line) for line in Path(args.distill).read_text(encoding='utf-8').splitlines()]
        if not distill or any(r.get('split') != 'train' or r.get('verified') is not True for r in distill):
            raise ValueError('unverified/nontraining trajectory forbidden')
        from dsv3wm.planning import verify_distillation_row
        if not all(verify_distillation_row(r) for r in distill):
            raise ValueError('distillation exact re-verification failed')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    provenance = {'config': cfg.dict(), 'config_sha256': sha256(args.config), 'data_sha256': sha256(args.train),
                  'status': 'started', 'command': sys.argv, 'started_unix': time.time()}
    if args.distill:
        provenance['distill_sha256'] = sha256(args.distill)
    (out / 'run.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    try:
        with runtime(cfg, 'train-' + cfg.arm):
            import torch
            from torch import nn
            from dsv3wm.objectives import SequentialMTP, token_objective, multi_horizon_loss
            model, tok, lm = load(cfg)
            pm = lm.attach_lora(model, rank=cfg.rank, alpha=cfg.rank * 2, dropout=0.0,
                                targets=('q_proj', 'v_proj'), seed=cfg.seed)
            pm.config.use_cache = False
            pm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            pm.enable_input_require_grads()
            decoder, head = lm._decoder_and_head(pm)
            embedding = pm.get_input_embeddings()
            device = next(pm.parameters()).device
            mtp = SequentialMTP(model.config.hidden_size, cfg.mtp_rank, cfg.mtp_tokens, cfg.share_aux).to(device) if cfg.arm == 'A1' else None
            legality = nn.Linear(model.config.hidden_size, 1).to(device) if cfg.legality_weight else None
            parameters = [p for p in pm.parameters() if p.requires_grad]
            if mtp:
                parameters.extend(mtp.parameters())
            if legality:
                parameters.extend(legality.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=cfg.lr)
            # Seeded shared row order and same LoRA initialization across all arms.
            import random
            rng = random.Random(177 + cfg.seed)
            order = [rng.randrange(len(rows)) for _ in range(cfg.steps * cfg.batch_size)]
            counts = {'input_tokens': 0, 'supervised_tokens': 0, 'aux_tokens': 0, 'supervised_transitions': 0,
                      'decoder_forwards': 0, 'planning_examples': 0}
            log = []
            started = time.perf_counter()
            pm.train()
            for step in range(cfg.steps):
                lm.thermal_guard()
                optimizer.zero_grad(set_to_none=True)
                for index in order[step * cfg.batch_size:(step + 1) * cfg.batch_size]:
                    row = rows[index]
                    losses = {}
                    legal_losses = []
                    for horizon in cfg.horizons:
                        modes = ('full', 'delta') if cfg.target_mode == 'mixed' else (cfg.target_mode,)
                        mode_losses = []
                        for mode in modes:
                            prompt = response_prompt(conditioning(row, horizon), mode)
                            label = json.dumps(target(row, horizon, mode), sort_keys=True, separators=(',', ':'))
                            encoded = lm.encode_pair(tok, prompt, label, cfg.max_length)
                            if encoded is None:
                                raise ValueError('overlength row; refusing silent truncation/drop')
                            ids, labels, att = (x.to(device) for x in lm.collate([encoded], tok.pad_token_id))
                            loss, stats, hidden = token_objective(decoder, head, embedding, ids, att, labels, mtp,
                                                                 cfg.mtp_weight, cfg.ce_chunk, cfg.shuffled_aux)
                            if legality:
                                prompt_end = len(tok(prompt, add_special_tokens=False).input_ids) - 1
                                logits = legality(hidden[:, prompt_end]).flatten()
                                gold = logits.new_tensor([float(row['labels']['legal'][horizon - 1])])
                                legal_losses.append(nn.functional.binary_cross_entropy_with_logits(logits, gold))
                            mode_losses.append(loss)
                            counts['input_tokens'] += int(ids.numel())
                            counts['supervised_tokens'] += stats['base_tokens']
                            counts['aux_tokens'] += stats['aux_tokens']
                            counts['supervised_transitions'] += 1
                            counts['decoder_forwards'] += 1
                        losses[horizon] = torch.stack(mode_losses).mean()
                    legal_loss = torch.stack(legal_losses).mean() if legal_losses else None
                    loss = multi_horizon_loss(losses, cfg.lambda2, cfg.lambda4, legal_loss, cfg.legality_weight)
                    if distill:
                        dr = distill[(step * cfg.batch_size + index) % len(distill)]
                        enc = lm.encode_pair(tok, dr['prompt'], dr['target'], cfg.max_length)
                        if enc is None:
                            raise ValueError('overlength verified plan')
                        ids, lab, att = (x.to(device) for x in lm.collate([enc], tok.pad_token_id))
                        ploss, stats, _ = token_objective(decoder, head, embedding, ids, att, lab, chunk=cfg.ce_chunk)
                        loss = loss + ploss
                        counts['input_tokens'] += int(ids.numel())
                        counts['supervised_tokens'] += stats['base_tokens']
                        counts['decoder_forwards'] += 1
                        counts['planning_examples'] += 1
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    (loss / cfg.batch_size).backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                entry = {'step': step + 1, 'loss': float(loss.detach()), 'elapsed_s': time.perf_counter() - started}
                log.append(entry)
                print(json.dumps(entry), flush=True)
                if cfg.device == 'cuda':
                    lm.gpu_yield()
            pm.save_pretrained(out / 'adapter')
            auxiliaries = {'mtp': mtp.state_dict() if mtp else None, 'legality': legality.state_dict() if legality else None}
            torch.save(auxiliaries, out / 'auxiliary.pt')
            provenance.update(status='pilot-trained', **counts, optimizer_steps=cfg.steps,
                trainable_parameters=sum(p.numel() for p in parameters),
                lora_parameters=sum(p.numel() for p in pm.parameters() if p.requires_grad),
                parent_parameters=sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in pm.parameters() if p.requires_grad),
                training_duration_s=time.perf_counter() - started, loss_curve=log,
                decoder_flops_proxy=6 * 596049920 * counts['input_tokens'],
                flops_proxy_scope='dense decoder only; excludes auxiliary/head/vocabulary operations and recomputation',
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if cfg.device == 'cuda' else 0,
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if cfg.device == 'cuda' else 0,
                versions={'python':sys.version, 'torch':torch.__version__, 'transformers':__import__('transformers').__version__,
                          'peft':__import__('peft').__version__}, core_sha256=sha256(lm.MODELS[cfg.core] / 'model.safetensors'),
                adapter_sha256=sha256(out / 'adapter' / 'adapter_model.safetensors'))
    except BaseException as exc:
        provenance.update(status='failed', error=repr(exc))
        raise
    finally:
        (out / 'run.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
