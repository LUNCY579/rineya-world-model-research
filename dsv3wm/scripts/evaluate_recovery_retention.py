"""Greedy behavior on six fixed historical-format fixtures, with no output repair."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def strict_json(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate key')
            value[key] = item
        return value
    try:
        value = json.loads(text.strip(), object_pairs_hook=unique)
        return value if isinstance(value, dict) else None
    except (ValueError, AttributeError):
        return None


def score_reply(fixture, generated):
    expected = json.loads(fixture['target'])
    parsed = strict_json(generated['text'])
    valid = (parsed is not None and set(parsed) == set(expected)
             and all(type(parsed[key]) is type(value) for key, value in expected.items()))
    return {'id': fixture['id'], 'domain': fixture['domain'], 'generated': generated,
            'parsed': parsed, 'schema_valid': valid, 'correct': valid and parsed == expected}


def main():
    from dsv3wm import REPO
    from dsv3wm.native import EXPECTED_CORE_SHA, boundary, load, runtime, sha256, verify_freeze
    from dsv3wm.provenance import verify_bound_inputs
    from dsv3wm.recovery import RecoveryConfig, load_warm
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'adapter', 'fixtures', 'freeze', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    cfg = RecoveryConfig.read(args.config)
    if cfg.device != 'cpu' or cfg.generation_tokens != 32:
        raise ValueError('retention pilot binds CPU/greedy32 tokens')
    if args.out.exists():
        raise FileExistsError('retention result exists')
    freeze_sha = verify_freeze(args.freeze)
    paths = [args.config, args.adapter / 'adapter_model.safetensors', args.adapter / 'adapter_config.json',
             args.fixtures, Path(__file__), ROOT / 'src/dsv3wm/recovery.py', ROOT / 'src/dsv3wm/native.py',
             ROOT / 'src/dsv3wm/provenance.py', REPO / 'bwm/code/bwmn/lm.py',
             REPO / 'bwm/code/bwmn/rw2.py', REPO / 'bwm/code/scripts/planbench_qwen.py',
             REPO / 'bwm/code/bwmn/gpu_lease.py']
    verify_bound_inputs(args.freeze, paths)
    document = json.loads(args.fixtures.read_text(encoding='utf-8'))
    fixtures = document['fixtures']
    if (document.get('schema') != 'dsv3wm-retention-fixtures-v1' or len(fixtures) != 6
            or len({row['id'] for row in fixtures}) != 6):
        raise ValueError('fixed six-fixture historical-format collection required')
    # These unchanged fixtures were frozen for the earlier CE audit. Behavior is
    # newly measured; previous losses never determine which replies are retained.
    result = {'status': 'started', 'command': sys.argv, 'config': cfg.dict(),
              'freeze_sha256': freeze_sha, 'fixture_sha256': sha256(args.fixtures),
              'adapter_sha256': sha256(args.adapter / 'adapter_model.safetensors'),
              'parent_sha256': EXPECTED_CORE_SHA, 'oracle_feedback': False,
              'scope': 'six fixed constructed cases; narrow behavioral retention, no broad skill pass',
              'retention_pass': None, 'rows': []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    started = time.perf_counter()
    try:
        with runtime(cfg, 'recovery-retention-' + cfg.arm):
            model, tokenizer, lm = load(cfg)
            model = load_warm(model, args.adapter)
            model.eval()
            lm.DUTY_RATIO = 0.0
            prompts = [row['prompt'] for row in fixtures]
            lengths = [len(tokenizer(prompt, add_special_tokens=False).input_ids) for prompt in prompts]
            if any(length > cfg.max_length for length in lengths):
                raise ValueError('overlength retention fixture; no truncation')
            for row, prompt in zip(fixtures, prompts):
                rendered = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                                                        add_generation_prompt=True, enable_thinking=False)
                if rendered != prompt:
                    raise ValueError('historical prompt/template drift')
            boundary(cfg)
            replies = lm.sample_crn(model, tokenizer, prompts, [None] * len(prompts), max_new_tokens=32)
            if len(replies) != len(fixtures):
                raise RuntimeError('retention reply cardinality mismatch')
            result['rows'] = [score_reply(row, reply) for row, reply in zip(fixtures, replies)]
            result.update(status='development-evaluated', model_queries=len(fixtures),
                          input_tokens=sum(lengths), output_tokens=sum(row['n_tokens'] for row in replies),
                          processed_positions_upper_bound=len(fixtures) * (max(lengths) + 32),
                          correct=sum(row['correct'] for row in result['rows']),
                          schema_valid=sum(row['schema_valid'] for row in result['rows']))
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        result['duration_s'] = time.perf_counter() - started
        args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'out': str(args.out), 'correct': result['correct'], 'n': 6}), flush=True)


if __name__ == '__main__':
    main()
