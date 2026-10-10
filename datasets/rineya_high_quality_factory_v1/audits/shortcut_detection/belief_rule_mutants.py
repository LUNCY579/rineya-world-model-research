"""Measure fixed incomplete belief rules; no classifier fitting or approval.

Actual author: /root. These rules were specified by the frozen old-release
diagnosis, before new candidate generation. Counts distinguish correlated
renderings, base contexts and independent source families.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from factory.common import ROOT, canonical, candidate_paths, record_hash, write_json
from oracles.independent import solve


def measure(rows):
    totals = Counter()
    by_context = defaultdict(Counter)
    failures = []
    for row in rows:
        if record_hash(row) != row['record_hash']:
            raise AssertionError('Record hash mismatch: ' + row['sample_id'])
        task = row['semantic_task']
        expected = solve(task)
        if canonical(expected) != canonical(row['verified_target']):
            raise AssertionError('Exact typed target mismatch: ' + row['sample_id'])
        support = len(expected['surviving_worlds'])
        observed = {o['fact'] for o in task['observations']}
        context = row.get('controlled_context', 'legacy')
        scores = Counter(records=1, queries=len(task['queries']))
        wrong = []
        for query, answer in zip(task['queries'], expected['queries']):
            usefulness = support > 1 and query['fact'] not in observed
            entailment = support > 0 and (support == 1 or query['fact'] in observed)
            useful_wrong = usefulness != answer['can_reduce_uncertainty']
            entailed_wrong = entailment != answer['entailed']
            scores['usefulness_errors'] += useful_wrong
            scores['entailment_errors'] += entailed_wrong
            scores['joint_flag_query_errors'] += useful_wrong or entailed_wrong
            scores['unobserved_multiworld_invariant_queries'] += (
                support >= 2 and query['fact'] not in observed and answer['entailed'])
            scores['useful_after_multiple_observations_queries'] += (
                len(task['observations']) >= 2 and answer['can_reduce_uncertainty'])
            if useful_wrong or entailed_wrong:
                wrong.append({'query_id': query['query_id'], 'fact': query['fact'],
                              'support': support, 'observed': query['fact'] in observed,
                              'heuristic_useful': usefulness, 'heuristic_entailed': entailment,
                              'exact_query': answer})
        scores['records_with_any_flag_error'] += bool(wrong)
        totals.update(scores)
        by_context[context].update(scores)
        if wrong:
            failures.append({'sample_id': row['sample_id'], 'record_hash': row['record_hash'],
                             'source_family': row['semantic_family_id'], 'context': context,
                             'errors': wrong})
    return {'counts': dict(totals), 'context_counts': {k: dict(v) for k, v in by_context.items()},
            'exact_typed_target_errors': 0, 'source_families': len({r['semantic_family_id'] for r in rows}),
            'counterexamples': failures}


def run(batch):
    paths = candidate_paths(batch)
    if not paths:
        raise ValueError('No candidate batch')
    paths = [p for p in paths if p.parent.parent.name == 'RBM-001']
    if not paths:
        raise ValueError('No belief rows')
    legacy_path = ROOT / 'data/RBM-001/releases/v0001/canonical.jsonl'
    snapshots = {p: p.read_bytes() for p in paths + [legacy_path]}
    rows = [json.loads(line) for p in paths for line in snapshots[p].splitlines() if line]
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(content).hexdigest()
               for p, content in snapshots.items()}
    for p in [Path(__file__).resolve(), ROOT/'oracles/independent.py', ROOT/'oracles/extended_independent.py']:
        sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    base = [r for r in rows if r.get('schedule_index') == 0 and r['renderer'] == 'forward']
    if len(base) != len({(r['semantic_family_id'], r.get('controlled_context')) for r in rows}):
        raise AssertionError('Missing or duplicate decoded base context')
    report = {
        'actual_author_execution': '/root', 'actual_runner_execution': '/root',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'batch': batch,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'independent_answer_oracle_author': '/root/formal_oracle',
        'independent_answer_oracle_sha256': hashlib.sha256((ROOT/'oracles/independent.py').read_bytes()).hexdigest(),
        'sources_sha256': sources,
        'fixed_mutants': {
            'usefulness': 'final support > 1 AND query fact absent from observations',
            'entailment': 'nonempty final support AND (support = 1 OR queried fact observed)'},
        'legacy_release': measure([json.loads(line) for line in snapshots[legacy_path].splitlines() if line]),
        'new_rendered_rows': measure(rows), 'new_base_contexts': measure(base),
        'scope': 'Deterministic incomplete-rule diagnostics on exposed synthetic data. Flag projections are partial target predictions; exact full typed targets are independently checked separately. No fitting, language-model training, held-out efficacy or independent release approval. Rotations, contexts and field orders remain correlated within source families.'}
    for relative, digest in sources.items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Candidate changed during probe')
    target = ROOT / 'audits/shortcut_detection' / f'{batch}-fixed-rule-mutants.json'
    if target.exists():
        raise FileExistsError('Preserve first diagnostic; use a new batch')
    write_json(target, report)
    return {key: report[key]['counts'] for key in ['legacy_release', 'new_rendered_rows', 'new_base_contexts']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.batch), indent=2))
