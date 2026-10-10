#!/usr/bin/env python3
"""Map exact joined six-baseline custody to the frozen runner's supported schema."""
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
FACTORY = Path(__file__).resolve().parent.parent
LOGS = FACTORY / 'backup_logs'
AUTHOR = '/root/backup_manager'
PARENT = '4d0b5eff1af3184fedf7fa25ada729a6cc909268'
JOINED = 'backup_logs/BELIEF_V4_R2_JOINED_CUSTODY.json'
JOINED_SHA = '32c1fbeb9cef515b532e4097a750cd325470010768f594c413439a5720db85c9'
RUNNER = 'backup_logs/run_provisional_archival_checkpoint_v2.py'
RUNNER_SHA = 'df03bdebc7d9cb6b24c65f82f1e17b42e8f857a106fafc3bfd2ac3ac341a0897'
FREEZE = 'backup_logs/PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER_SOURCE_FREEZE.json'
OUTPUT = 'backup_logs/PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER.json'


def proof(path):
    data = path.read_bytes()
    return {'path': path.relative_to(FACTORY).as_posix(), 'size': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def require(value, message):
    if not value:
        raise ValueError(message)


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def normalized_joined(joined, original, runner):
    require(joined['status'] == 'BACKED_UP_FROZEN_INVENTORY_AND_REMOTE_LOSSLESS_PROOF_JOINED'
            and joined['commit_sha'] == PARENT, 'Exact completed parent join required')
    # This is only a schema mapping. The immutable runner validates all six
    # fixed labels, exact paths, boolean existence types and known SHA256 bytes.
    return runner.normalize_baselines(original, joined['six_protected_git_baselines'])


def main():
    source = proof(Path(__file__))
    freeze = json.loads((FACTORY / FREEZE).read_bytes())
    require(freeze['source'] == source and freeze['source_frozen_before_execution'] is True,
            'Adapter source must be frozen before execution')
    for binding in freeze["frozen_inputs"]:
        require(proof(FACTORY / binding["path"]) == binding, "Frozen adapter preparation dependency changed")
    require(proof(FACTORY / JOINED)['sha256'] == JOINED_SHA, 'Exact authoritative join SHA differs')
    require(proof(FACTORY / RUNNER)['sha256'] == RUNNER_SHA, 'Frozen runner differs')
    require(not (FACTORY / OUTPUT).exists(), 'Fresh adapter output required')
    sp = importlib.util.spec_from_file_location('frozen_runner_for_v4_adapter', FACTORY / RUNNER)
    runner = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(runner)
    require(proof(FACTORY / RUNNER)["sha256"] == RUNNER_SHA, "Frozen normalizer changed during import")
    original = FACTORY.parent / 'rineya-world-model-research'
    worktree = FACTORY.parent / '.rineya-dataset-git-worktree'
    joined = json.loads((FACTORY / JOINED).read_bytes())
    rows = normalized_joined(joined, original, runner)
    adapter = {'protected_git_byte_baselines': rows}
    runner.original_custody(original, adapter)
    private = runner.dedicated_custody(worktree, PARENT)
    observations = []
    for row in rows:
        path = Path(row['path'])
        actual_exists = path.exists()
        require(actual_exists is row['expected_exists'], 'Actual protected existence differs')
        item = dict(row, actual_exists=actual_exists, matches_frozen_baseline=True)
        if actual_exists:
            item['actual_sha256'] = runner.proof(path, False)['sha256']
            require(item['actual_sha256'] == row['sha256'], 'Actual protected Git bytes differ')
        observations.append(item)
    adapter.update(status='EXACT_JOINED_SIX_BASELINE_SCHEMA_ADAPTER_ACTUAL_CUSTODY_PASS',
                   author_execution=AUTHOR, actual_runner_execution=AUTHOR,
                   at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   source=source, source_freeze=proof(FACTORY / FREEZE),
                   joined_custody=proof(FACTORY / JOINED), frozen_runner=proof(FACTORY / RUNNER),
                   expected_parent_sha=PARENT, actual_private_admin=private,
                   actual_six_protected_baseline_readbacks=observations,
                   original_main_clean=True, dedicated_clean=True,
                   no_git_mutation=True, grants_data_approval=False,
                   schema_mapping_only_no_guard_relaxation=True,
                   local_postinventory=True)
    require(runner.normalize_baselines(original, adapter) == rows, 'Mapped six baseline rows differ')
    runner.original_custody(original, adapter)
    require(proof(FACTORY / JOINED)['sha256'] == JOINED_SHA
            and proof(FACTORY / RUNNER)['sha256'] == RUNNER_SHA
            and proof(Path(__file__)) == source, 'Frozen inputs changed at finish')
    for binding in freeze["frozen_inputs"]:
        require(proof(FACTORY / binding["path"]) == binding, "Frozen adapter preparation dependency changed at finish")
    exclusive(FACTORY / OUTPUT, adapter)
    print(json.dumps({'status': adapter['status'], 'adapter': proof(FACTORY / OUTPUT)}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        path = LOGS / ('PROSPECTIVE_ARCHIVAL_V3_CUSTODY_ADAPTER_FAILURE_' +
                       datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
        exclusive(path, {'status': 'FAILED_NEW_SCHEMA_ADAPTER_PRESERVED',
                         'author_execution': AUTHOR, 'actual_runner_execution': AUTHOR,
                         'source': proof(Path(__file__)), 'error_type': type(error).__name__,
                         'error': str(error), 'no_git_mutation': True})
        raise
