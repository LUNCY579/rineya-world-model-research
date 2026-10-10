#!/usr/bin/env python3
"""Owned bounded, opaque-byte scanner/copy review. Never runs Git or backup main."""
import copy
import datetime
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREEXEC = HERE / 'replay_r1.preexec.json'
RESULT = HERE / 'replay_r1.execution.json'
ACTOR = '/root/adversarial_auditor'


def binding(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'size': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShortReads:
    def __init__(self, data, cuts):
        self.data = data
        self.position = 0
        self.cuts = list(cuts)

    def read(self, count):
        if self.position == len(self.data):
            return b''
        next_cut = self.cuts.pop(0) if self.cuts else len(self.data)
        end = min(self.position + count, next_cut, len(self.data))
        if end <= self.position:
            raise ValueError('Non-increasing authored short-read fixture')
        value = self.data[self.position:end]
        self.position = end
        return value


def main():
    frozen = json.loads(PREEXEC.read_bytes())
    for row in frozen['bindings']:
        if binding(ROOT / row['path']) != row:
            raise ValueError('Preexecution binding differs; no target import occurred')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    scanner = load(ROOT / 'backup_logs/exact_four_public_metadata_secret_scan.py', 'red_exact_scanner_r1')
    runner = load(ROOT / 'backup_logs/run_provisional_archival_checkpoint_v4.py', 'red_backup_runner_r1')
    original = load(ROOT / 'backup_logs/run_provisional_archival_checkpoint_v3.py', 'red_original_backup_r1')
    patterns = original.SECRET_PATTERNS
    results = []

    def check(name, expected, fn, detail=None):
        error_type = None
        try:
            fn()
            outcome = 'RETURNED'
        except Exception as error:
            outcome = 'RAISED'
            error_type = type(error).__name__
        results.append({'name': name, 'expected': expected, 'outcome': outcome,
                        'passed': outcome == expected, 'error_type': error_type,
                        'detail': detail or {}})

    def equal(actual, expected):
        if actual != expected:
            raise AssertionError('Independent exact comparison differs')

    check('all_original_regex_bytes_flags_equal', 'RETURNED', lambda: equal(
        [(p.pattern, p.flags) for p in runner.SECRET_PATTERNS],
        [(p.pattern, p.flags) for p in patterns]))
    check('fixed_config_matches_root_readback', 'RETURNED', lambda: equal(
        scanner.validate_config()['files'], json.loads((ROOT / frozen['diagnosis_path']).read_bytes())['files']))
    chunks = (1, 2, 3, 7, 31, 33, 34, 35, 64, 257, 4095, 4096, 4097, 65536)
    for i, row in enumerate(scanner.EXPECTED_FILES):
        path = ROOT / row['path']
        data = path.read_bytes()  # Opaque bytes only; never parse/import these four payloads.
        expected = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                    'git_blob': hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()}
        match = row['matches'][0]
        check(f'opaque_root_binding_{i}', 'RETURNED', lambda row=row, data=data: equal(
            (len(data), hashlib.sha256(data).hexdigest(), hashlib.sha256(data[match['start']:match['end']]).hexdigest(),
             data[match['start']-2:match['start']] == b'ta'),
            (row['size'], row['sha256'], match['match_sha256'], True)))
        for chunk in chunks:
            check(f'positive_file_{i}_chunk_{chunk}', 'RETURNED',
                  lambda path=path, chunk=chunk: equal(scanner.proof(path, patterns, chunk), expected),
                  {'file_index': i, 'chunk': chunk, 'source_sha256': row['sha256']})
        for cut in range(match['start'] - 2, match['end'] + 3):
            check(f'match_boundary_file_{i}_{cut}', 'RETURNED',
                  lambda cut=cut: equal(scanner.scan_stream(ShortReads(data, [cut, cut+1]), len(data), patterns, row),
                                        (expected, row['matches'])),
                  {'file_index': i, 'first_cut': cut, 'second_cut': cut+1})
        if len(data) > match['start'] + 4097:
            for cut in (match['start']+4095, match['start']+4096, match['start']+4097):
                check(f'overlap_context_file_{i}_{cut}', 'RETURNED',
                      lambda cut=cut: equal(scanner.scan_stream(ShortReads(data, [cut, cut+1]), len(data), patterns, row),
                                            (expected, row['matches'])),
                      {'file_index': i, 'first_cut': cut, 'second_cut': cut+1})
        variants = {
            'changed_nonmatch_same_size': b'X' + data[1:],
            'appended': data+b'!',
            'truncated': data[:-1],
            'extension_same_size': data[:match['end']] + b'A' + data[match['end']+1:],
            'changed_digest_same_span': data[:match['start']+3]+b'Z'+data[match['start']+4:],
            'changed_preceding_context': data[:match['start']-2]+b'za'+data[match['start']:],
            'duplicate_exact_match_same_size': data[:-35]+b' '+data[match['start']:match['end']],
            'other_pattern_same_size': data[:-45]+b'\npassword = '+b'Q'*33,
        }
        for name, payload in variants.items():
            for chunk in (1, 31, 65536):
                check(f'reject_{name}_{i}_chunk_{chunk}', 'RAISED',
                      lambda payload=payload, chunk=chunk: scanner.scan_stream(io.BytesIO(payload), len(payload), patterns, row, chunk),
                      {'file_index': i, 'payload_sha256': hashlib.sha256(payload).hexdigest(), 'chunk': chunk})
        check(f'unbound_exact_bytes_{i}', 'RAISED', lambda: scanner.scan_stream(io.BytesIO(data), len(data), patterns))
        for key, value in (('path', row['path'].upper()), ('size', row['size']+1), ('sha256', '0'*64)):
            changed = copy.deepcopy(row);changed[key] = value
            check(f'reject_binding_{key}_{i}', 'RAISED', lambda changed=changed: scanner.scan_stream(io.BytesIO(data), len(data), patterns, changed))
        for key, value in (('start', match['start']+1), ('end', match['end']+1), ('pattern_index', 0),
                           ('pattern_index', True), ('match_sha256', '0'*64), ('preceding_literal_ta', 1)):
            changed = copy.deepcopy(row);changed['matches'][0][key] = value
            check(f'reject_match_binding_{key}_{type(value).__name__}_{i}', 'RAISED',
                  lambda changed=changed: scanner.scan_stream(io.BytesIO(data), len(data), patterns, changed))
    keys = [b'gh'+b'p_'+b'Q'*35, b'sk'+b'-'+b'Q'*35, b'sk'+b'-proj-'+b'Q'*35,
            b'AK'+b'IA'+b'Q'*16, b'-----BEGIN '+b'PRIVATE KEY-----', b'password = '+b'Q'*35]
    for i, key in enumerate(keys):
        for chunk in (1, 2, 7, 31, 65536):
            payload = b'ordinary prefix\n'+key+b'\n'
            check(f'other_original_pattern_{i}_chunk_{chunk}', 'RAISED',
                  lambda payload=payload, chunk=chunk: scanner.scan_stream(io.BytesIO(payload), len(payload), patterns, None, chunk))
    for chunk in (1, 7, 65536):
        payload = b'ordinary prefix password = '+b'Q'*35+b'\n'
        check(f'inline_assignment_negative_control_{chunk}', 'RETURNED',
              lambda payload=payload, chunk=chunk: scanner.scan_stream(io.BytesIO(payload), len(payload), patterns, None, chunk))
    row = scanner.EXPECTED_FILES[0]
    data = (ROOT / row['path']).read_bytes()
    for name, wrong in [('empty', []), ('removed', patterns[:-1]), ('reordered', list(reversed(patterns))),
                        ('flags', [re.compile(patterns[0].pattern, re.I)]+patterns[1:])]:
        check(f'reject_pattern_change_{name}', 'RAISED', lambda wrong=wrong: scanner.scan_stream(io.BytesIO(data), len(data), wrong, row))
    for size, chunk in ((True, 1), (len(data), True), (len(data), 0), (len(data), 65537)):
        check(f'reject_stream_argument_{type(size).__name__}_{chunk}', 'RAISED',
              lambda size=size, chunk=chunk: scanner.scan_stream(io.BytesIO(data), size, patterns, row, chunk))
    with tempfile.TemporaryDirectory(prefix='owned-fixtures-', dir=HERE) as directory:
        temporary = Path(directory)
        copied = temporary / 'copy.json';copied.write_bytes(data)
        link = temporary / 'alias.json';link.symlink_to(ROOT / row['path'])
        for name, path in [('copy', copied), ('symlink', link), ('dot', str(ROOT)+'/./'+row['path']),
                           ('double_slash', str(ROOT)+'//'+row['path']), ('parent', str(ROOT)+'/x/../'+row['path']),
                           ('relative', Path(row['path'])), ('outside', Path('/tmp/opaque-context'))]:
            check(f'reject_path_{name}', 'RAISED', lambda path=path: scanner.proof(path, patterns))
        factory = temporary / 'factory';factory.mkdir()
        worktree = temporary / 'copy-target';worktree.mkdir()
        source = factory / 'ordinary.txt';source.write_bytes(b'normal fixture\n')
        expected = runner.proof(source)
        target = worktree / runner.PREFIX / source.name
        check('actual_copy_normal_hash_chain', 'RETURNED',
              lambda: equal(runner.copy_one(factory, worktree, source.name, expected), 'COPIED_AND_REVERIFIED'))
        check('actual_copy_unchanged_reverification', 'RETURNED',
              lambda: equal(runner.copy_one(factory, worktree, source.name, expected), 'UNCHANGED_TARGET_REVERIFIED'))
        source.write_bytes(b'altered fixture\n')
        target.write_bytes(b'prior untouched\n')
        check('actual_copy_changed_source_rejects', 'RAISED', lambda: runner.copy_one(factory, worktree, source.name, expected))
        check('actual_copy_failure_keeps_old_target', 'RETURNED', lambda: equal(target.read_bytes(), b'prior untouched\n'))
        check('actual_copy_failure_no_leftover_temp', 'RETURNED', lambda: equal(list(target.parent.glob('.archival-copy-*')), []))
        source.write_bytes(data)
        spec = {'approval': 'provisional_checkpoint', 'approved_by': '/root', 'validation_status': 'passed',
                'files': [{'path': source.name, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]}
        check('alternate_factory_path_inventory_rejects', 'RAISED', lambda: runner.inventory(factory, spec))
        for path in ('./ordinary.txt', 'x/../ordinary.txt', 'ordinary.txt//', '/ordinary.txt', '.git/config'):
            check('strict_safe_path_'+hashlib.sha256(path.encode()).hexdigest()[:8], 'RAISED', lambda path=path: runner.safe(factory, path))
    for row in frozen['bindings']:
        if binding(ROOT / row['path']) != row:
            raise ValueError('Reviewed source or opaque input changed during replay')
    output = {'schema': 'backup-exact-public-id-adversarial-execution-v1', 'actor': ACTOR,
              'started_at_utc': started, 'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'preexecution': binding(PREEXEC), 'reviewed_bindings_before_after_identical': True,
              'checks': results, 'count': len(results), 'passed': sum(x['passed'] for x in results),
              'failed': sum(not x['passed'] for x in results),
              'scope': 'Direct actual scanner/copy functions only; opaque four metadata bytes; no Git/main/remote or RSM code execution',
              'model_disclosure': 'Same-model, root-guided false-positive context; independent authored fixtures and actual execution; no dataset judgment'}
    with RESULT.open('x') as handle:
        json.dump(output, handle, indent=2);handle.write('\n')
    print(json.dumps({'count': output['count'], 'passed': output['passed'], 'failed': output['failed'],
                      'result': binding(RESULT)}))


if __name__ == '__main__':
    main()
