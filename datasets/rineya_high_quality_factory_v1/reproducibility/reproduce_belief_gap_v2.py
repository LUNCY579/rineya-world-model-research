"""Re-run frozen producer code in a private scratch tree, preserving failures.

Actual author/runner: /root. This proves deterministic byte reproduction,
not independent semantics or reviewer approval. Original data stays read-only.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import traceback

from factory.common import ROOT, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    from generators.other_tracks import belief_gap_v2 as producer
    batch = 'belief-gap-v2'
    manifest_path = ROOT/f'manifests/{batch}_GENERATION.json'
    manifest = json.loads(manifest_path.read_text())
    source = ROOT/'generators/other_tracks/belief_gap_v2.py'
    assert sha(source) == manifest['source_sha256'], 'Frozen producer source changed'
    frozen = manifest['artifact_sha256']
    assert all(sha(ROOT/p) == digest for p, digest in frozen.items()), 'Frozen artifact changed'
    dependencies = sorted(set(manifest['public_input_sha256']) | {
        'research/curricula/RBM-EVIDENCE-GAP-ROOT_FREEZE.json',
        'research/curricula/RBM-EVIDENCE-GAP-REVISION-V2-ROOT_FREEZE.json',
        'manifests/belief-gap-v2_MARKER_RULE.json',
        'data/RBM-001/candidate_renderings/belief-gap-v1.jsonl',
        'generators/other_tracks/belief_gap_v2.py',
        'schemas/EXTENDED_SEMANTICS.md', 'factory/common.py',
        'generators/other_tracks/reference.py', 'generators/other_tracks/generate.py',
        'generators/other_tracks/belief_gap.py',
        'data/RBM-001/semantic_parents/extended-pilot-v1.jsonl',
        'data/RBM-001/semantic_parents/extended-pilot-v2.jsonl'})
    dependency_hashes = {p: sha(ROOT/p) for p in dependencies}
    expected = {**frozen, str(manifest_path.relative_to(ROOT)): sha(manifest_path)}
    result = {'runner_execution': '/root', 'script_author_execution': '/root',
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'batch': batch,
              'script_sha256': sha(Path(__file__)),
              'producer_source_sha256': sha(source), 'dependencies_sha256': dependency_hashes,
              'original_artifacts_sha256': expected,
              'scope': 'Same frozen producer algorithm, independent coordinator execution. Exact core artifact reproduction only; elapsed-time cost manifest is volatile. No additional corpus/family records or data approval.'}
    original_root = producer.ROOT
    with tempfile.TemporaryDirectory(prefix='rineya-belief-reproduction-') as directory:
        scratch = Path(directory)
        try:
            for p in dependencies:
                destination = scratch/p
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/p, destination)
            producer.ROOT = scratch
            rows = producer.generate(batch, manifest['seed'])
            comparisons = {}
            for p, digest in frozen.items():
                path = scratch/p
                comparisons[p] = {'bytes': path.stat().st_size, 'expected_sha256': digest,
                                  'actual_sha256': sha(path), 'exact_bytes_equal': path.read_bytes() == (ROOT/p).read_bytes()}
            result['core_artifact_comparisons'] = comparisons
            result['reproduced_correlated_records'] = len(rows)
            assert all(item['exact_bytes_equal'] for item in comparisons.values()), 'Reproduction mismatch'
            result['pass'] = True
        except Exception:
            result['pass'] = False
            result['failure_traceback'] = traceback.format_exc()
            failure_root = ROOT/'reproducibility/failed_belief_gap_v2'
            assert not failure_root.exists(), 'Preserve existing first failure'
            for p in frozen:
                path = scratch/p
                if path.exists():
                    destination = failure_root/p
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, destination)
            result['failed_outputs_preserved_under'] = str(failure_root.relative_to(ROOT))
        finally:
            producer.ROOT = original_root
    result['original_sources_and_artifacts_unchanged'] = all(
        sha(ROOT/p) == digest for p, digest in {**dependency_hashes, **expected}.items())
    result['pass'] = result['pass'] and result['original_sources_and_artifacts_unchanged']
    result['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    target = ROOT/'quality_reports/BELIEF_GAP_V2_REPRODUCTION.json'
    if target.exists():
        raise FileExistsError('Reproduction execution evidence is immutable')
    write_json(target, result)
    print(json.dumps({'pass': result['pass'], 'artifacts': len(result.get('core_artifact_comparisons', {})),
                      'records': result.get('reproduced_correlated_records'),
                      'original_unchanged': result['original_sources_and_artifacts_unchanged']}, indent=2))
    if not result['pass']:
        raise RuntimeError('Reproduction failed; exact evidence retained')


if __name__ == '__main__':
    run()
