"""Development CLI; confirmation deliberately requires a future complete G5 freeze."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from dsv3wm.config import Config
    from dsv3wm.native import runtime, load, Predictor, sha256, response_prompt, verify_freeze
    from dsv3wm.data import conditioning, target
    from dsv3wm.evaluation import score_dynamics
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--out', required=True)
    parser.add_argument('--domain', choices=['register','ledger','blocks'])
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--horizons', default='1,2,4')
    parser.add_argument('--freeze', default=str(Path(__file__).resolve().parents[1]/'prereg/EVALUATION_FREEZE_V2.json'))
    args = parser.parse_args()
    cfg = Config.read(args.config)
    freeze_sha = verify_freeze(args.freeze)
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines()]
    if any(r.get('split') != 'dev' for r in rows):
        raise ValueError('only labelled dev allowed; confirmation registration not yet complete')
    if args.domain:
        rows = [r for r in rows if r['domain'] == args.domain]
    if args.limit <= 0:
        raise ValueError('positive limit required')
    rows = rows[:args.limit]
    if not rows:
        raise ValueError('empty development cohort')
    horizons = tuple(int(h) for h in args.horizons.split(','))
    if set(horizons) - {1,2,4}:
        raise ValueError('invalid horizons')
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('evaluation record exists')
    result = {'status':'started', 'config':cfg.dict(), 'data_sha256':sha256(args.data), 'command':sys.argv,
              'freeze_sha256':freeze_sha, 'evaluated_unique_parents':len({r['parent_world_id'] for r in rows}),
              'instrument':'new development / no simulator feedback / greedy / use_cache=False'}
    started = time.perf_counter()
    try:
        with runtime(cfg, 'evaluate-' + cfg.arm):
            model, tok, lm = load(cfg, args.adapter)
            predictor = Predictor(model, tok, cfg)
            # Teacher-forced CE is separated from generation outcomes.
            import torch
            ces = []
            model.eval()
            for row in rows:
                prompt = response_prompt(conditioning(row, 1), 'full' if cfg.target_mode=='mixed' else cfg.target_mode)
                label = json.dumps(target(row, 1, 'full' if cfg.target_mode=='mixed' else cfg.target_mode), sort_keys=True, separators=(',',':'))
                encoded = lm.encode_pair(tok, prompt, label, cfg.max_length)
                if encoded is None:
                    raise ValueError('overlength dev case')
                ids, lab, att = (x.to(next(model.parameters()).device) for x in lm.collate([encoded], tok.pad_token_id))
                with torch.no_grad():
                    ce, _ = lm.chunked_ce(model, ids, att, lab, cfg.ce_chunk)
                ces.append(float(ce))
            result.update(score_dynamics(predictor, rows, horizons))
            result.update(status='pilot-evaluated', teacher_forced_one_step_ce=sum(ces)/len(ces),
                          forward_model_calls=predictor.calls, input_tokens=predictor.input_tokens, output_tokens=predictor.output_tokens,
                          autoregressive_decoder_forwards=predictor.output_tokens, decoder_token_positions=predictor.decoder_token_positions,
                          peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if cfg.device=='cuda' else 0)
            if args.adapter:
                result['adapter_sha256'] = sha256(Path(args.adapter)/'adapter_model.safetensors')
    except BaseException as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        result['duration_s'] = time.perf_counter() - started
        out.write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
