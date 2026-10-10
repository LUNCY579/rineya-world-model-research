"""Freeze a reviewer-only six-field view without targets or producer controls."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from factory.catalog import catalog
from factory.common import ROOT, canonical
from factory.lineage import source_key
from factory.pipeline import schema_validate

CANDIDATE_SHA = '57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23'


def main():
    source = ROOT / 'data/RGM-001/candidate_renderings/planning-gap-v2.jsonl'
    view = ROOT / 'reviews/input_only/planning-gap-v2.jsonl'
    freeze = ROOT / 'reviews/input_only/planning-gap-v2.freeze.json'
    assert not view.exists() and not freeze.exists()
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CANDIDATE_SHA
    rows = [json.loads(line) for line in raw.splitlines() if line]
    assert len(rows) == 408 and len({row['sample_id'] for row in rows}) == 408
    public = []
    for row in rows:
        schema_validate(row)
        public.append({'sample_id': row['sample_id'], 'record_hash': row['record_hash'],
                       'source_family_key': source_key(row), 'split': row['split'],
                       'semantic_task': row['semantic_task'], 'rendered_input': row['rendered_input']})
    fields = {'sample_id', 'record_hash', 'source_family_key', 'split', 'semantic_task', 'rendered_input'}
    assert all(set(row) == fields for row in public)
    families = {row['source_family_key'] for row in public}
    assert len(families) == 48
    parent = [json.loads(line) for line in (ROOT / 'reviews/input_only/planning-gap-v1.jsonl').read_bytes().splitlines() if line]
    assert families == {row['source_family_key'] for row in parent}
    active_rows, active = catalog()
    assert not families.intersection(source_key(row) for row in active_rows)
    assert Counter(row['split'] for row in public) == Counter(train=340, dev_challenge=68)
    raw_typed_variants = len({canonical(row['semantic_task']) for row in public})
    declared_context_units = len({(source_key(row), row.get('controlled_context')) for row in rows})
    assert declared_context_units == 204
    data = ''.join(canonical(row) + '\n' for row in public).encode()
    with view.open('xb') as handle:
        handle.write(data)
    assert source.read_bytes() == raw
    result = {'root_execution': '/root', 'at_utc': datetime.now(timezone.utc).isoformat(),
              'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'candidate_sha256': CANDIDATE_SHA, 'input_view_sha256': hashlib.sha256(data).hexdigest(),
              'input_view_bytes': len(data), 'input_fields': sorted(fields), 'rows': 408,
              'declared_family_context_units': declared_context_units, 'raw_typed_task_variants': raw_typed_variants,
              'planned_normalized_semantic_inputs': 204, 'independent_normalized_input_count': 'PENDING separate finite batch audit',
              'revised_source_families': 48, 'additional_families_relative_v1': 0,
              'active_source_family_collisions': 0, 'active_approved_records': active['active_records'],
              'active_source_families': active['active_source_families'],
              'canonical_schema_and_prompt_boundary_pass': True, 'candidate_bytes_unchanged': True,
              'review_requirement': 'Each actual expert/red must freeze ALL own input-derived predictions/firstjudgments before candidate gold/control manifests/producer private source/formal private expectations/otherverdicts',
              'scope': 'Reviewer-only problem view with custody IDs/hashes/sourcekeys/splits; no gold, producer version/tags/property witnesses/control metadata. No student export or data approval.'}
    with freeze.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
