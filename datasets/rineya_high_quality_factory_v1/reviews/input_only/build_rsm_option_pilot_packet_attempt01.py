"""Root input-only review packet; preserves original hashes, excludes gold.

No factory/oracle imports, generation, semantic judgments or approval.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'data/RSM-001/candidate_renderings/option-resumption-pilot-v1.jsonl'
SOURCE_SHA = 'c20eebd1eba211bd069ae9424d2d5a8094f657432fbe25ffffdcc5d307756bef'
FIELDS = ('sample_id', 'semantic_parent_id', 'semantic_family_id', 'track_id',
          'split', 'semantic_task', 'rendered_input', 'record_hash')
OUTPUT = 'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl'
MANIFEST = 'reviews/input_only/RSM-option-resumption-pilot-v1.INPUT_CUSTODY.json'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def main():
    raw = (ROOT / SOURCE).read_bytes()
    assert sha256(raw).hexdigest() == SOURCE_SHA
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 216 and len({row['sample_id'] for row in rows}) == 216
    assert all(row['review_status'] == 'PENDING' and row['review_receipt_ids'] == [] for row in rows)
    for row in rows:
        assert row['record_hash'] == sha256(canonical({k: v for k, v in row.items() if k != 'record_hash'}).encode()).hexdigest()
    views = [{key: row[key] for key in FIELDS} for row in rows]
    packet = ''.join(canonical(row) + '\n' for row in views).encode()
    manifest = {'at_utc': datetime.now(timezone.utc).isoformat(),
                'actual_author_execution': '/root', 'input_only_packet': OUTPUT,
                'input_only_sha256': sha256(packet).hexdigest(), 'input_only_bytes': len(packet),
                'original_candidate_sha256': SOURCE_SHA, 'records': len(views),
                'fields': list(FIELDS), 'record_hash_scope': 'original full PENDING candidate; not a hash of this reduced view',
                'verified_target_or_private_witnesses_in_packet': False, 'data_approval': False,
                'review_policy': 'Freeze own predictions and initial input judgments before gold or other initial verdicts.'}
    assert (ROOT / SOURCE).read_bytes() == raw
    for name, content in [(OUTPUT, packet), (MANIFEST, (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode())]:
        with (ROOT / name).open('xb') as stream:
            stream.write(content)
    print(json.dumps({'packet_sha256': sha256(packet).hexdigest(), 'bytes': len(packet), 'records': len(views), 'gold_excluded': True}))


if __name__ == '__main__':
    main()
