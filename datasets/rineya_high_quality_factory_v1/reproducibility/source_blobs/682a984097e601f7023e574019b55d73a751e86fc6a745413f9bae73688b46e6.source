"""Freeze reviewer inputs without stored targets or other reviewer judgments.

Actual author/runner: root coordinator. Original candidate hashes are retained
as join keys; they are not hashes of the deliberately incomplete view records.
This command creates evidence, never predictions or approvals.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from factory.common import ROOT, candidate_paths, record_hash, write_json, write_jsonl


def create_view(batch):
    paths = candidate_paths(batch)
    if not paths:
        raise ValueError('No frozen candidate files for batch')
    sources = {}
    rows = []
    removed = set()
    allowed = {'sample_id', 'record_hash', 'semantic_parent_id', 'semantic_family_id',
               'track_id', 'split', 'semantic_task', 'rendered_input'}
    for path in paths:
        content = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = {
            'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        for line in content.splitlines():
            if not line:
                continue
            row = json.loads(line)
            if record_hash(row) != row['record_hash']:
                raise ValueError('Candidate hash mismatch')
            if row['review_status'] != 'PENDING' or row['review_receipt_ids']:
                raise ValueError('Expected unreviewed candidate records')
            removed.update(set(row) - allowed)
            view = {key: value for key, value in row.items() if key in allowed}
            rows.append(view)
        if path.read_bytes() != content:
            raise RuntimeError('Candidate changed while creating view')
    if len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    target = ROOT / 'reviews/input_only' / f'{batch}.jsonl'
    manifest = target.with_suffix('.manifest.json')
    if target.exists() or manifest.exists():
        raise FileExistsError('Input-only evidence is immutable; use a fresh batch')
    write_jsonl(target, rows)
    payload = target.read_bytes()
    report = {
        'view_author_execution': '/root', 'view_runner_execution': '/root',
        'view_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'batch': batch,
        'sources': sources, 'view': str(target.relative_to(ROOT)),
        'view_bytes': len(payload), 'view_sha256': hashlib.sha256(payload).hexdigest(),
        'records': len(rows), 'source_families': len({r['semantic_family_id'] for r in rows}),
        'removed_fields': sorted(removed),
        'join_key_note': 'record_hash belongs to the complete original candidate, not the view.',
        'scope': 'Student-visible inputs plus record/family/split join keys only. Stored targets, producer property/role/locator claims and review evidence are absent. Derive and freeze expected outputs first, then compare original metadata/targets in a separate stage. No prediction, judgment or approval supplied.'}
    write_json(manifest, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    args = parser.parse_args()
    print(json.dumps(create_view(args.batch), indent=2))
