"""Actual adjudicator input-derived audit; no producer/reviewer implementation imports."""
from pathlib import Path
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / 'reviews/adjudication'


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar(x):
    assert x is None or type(x) in (bool, int, str)
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def rows(path):
    return [json.loads(line) for line in (BASE / path).read_text().splitlines()]


def partition(worlds, fact):
    bins = {}
    for world in worlds:
        value = world['facts'][fact]
        key = scalar(value)
        bins.setdefault(key, [value, 0])[1] += 1
    return [{'value': bins[k][0], 'count': bins[k][1]} for k in sorted(bins)]


def solve(task):
    assert set(task) == {'kind', 'worlds', 'observations', 'queries'} and task['kind'] == 'belief'
    worlds = task['worlds']
    assert worlds and len({w['world_id'] for w in worlds}) == len(worlds)
    facts = set(worlds[0]['facts'])
    for world in worlds:
        assert set(world) == {'world_id', 'facts'} and type(world['world_id']) is str
        assert set(world['facts']) == facts
        for value in world['facts'].values():
            scalar(value)
    assert len({q['query_id'] for q in task['queries']}) == len(task['queries'])
    prefixes = [worlds]
    for observation in task['observations']:
        assert set(observation) == {'fact', 'equals'} and observation['fact'] in facts
        prefixes.append([w for w in prefixes[-1] if scalar(w['facts'][observation['fact']]) == scalar(observation['equals'])])
    final = prefixes[-1]
    answers = []
    for query in task['queries']:
        assert set(query) == {'query_id', 'fact'} and query['fact'] in facts
        bins = partition(final, query['fact'])
        entailed = bool(final) and len(bins) == 1
        answers.append({'query_id': query['query_id'], 'counts': bins,
                        'distinct_values': len(bins), 'entailed': entailed,
                        'guaranteed': bins[0]['value'] if entailed else None,
                        'can_reduce_uncertainty': len(bins) > 1})
    target = {'surviving_worlds': sorted(w['world_id'] for w in final),
              'belief_sizes': [len(p) for p in prefixes[1:]], 'queries': answers}
    # Independently form each prefix by conjunction over the original full table.
    for n, actual in enumerate(prefixes):
        direct = [w for w in worlds if all(type(w['facts'][o['fact']]) is type(o['equals']) and
                 w['facts'][o['fact']] == o['equals'] for o in task['observations'][:n])]
        assert sorted(w['world_id'] for w in actual) == sorted(w['world_id'] for w in direct)
    return target, prefixes


def ids(worlds):
    return sorted(w['world_id'] for w in worlds)


def alias(value):
    if type(value) is bool:
        return int(value)
    if type(value) is int and value in (0, 1):
        return bool(value)
    return object()


def property_audit(record, task, target, prefixes):
    observations = task['observations']
    final = prefixes[-1]
    qmap = {q['query_id']: q for q in task['queries']}
    amap = {q['query_id']: q for q in target['queries']}
    payloads = {canonical(w['facts']) for w in final}
    observed = {o['fact'] for o in observations}
    named = {q['fact'] for q in task['queries']}
    strict = [i for i in range(len(observations)) if 0 < len(prefixes[i + 1]) < len(prefixes[i])]
    generic = {'finite_synthetic_proxy', 'finite_noiseless_filter', 'possible_value_partition'}
    tags = set(record['skill_tags']) - generic
    assert generic <= set(record['skill_tags']) and tags == set(record['property_witnesses'])
    for name, loc in record['property_witnesses'].items():
        assert {'query_ids', 'observation_indices', 'prefix_indices'} <= set(loc)
        assert set(loc) <= {'query_ids', 'observation_indices', 'prefix_indices', 'supporting_world_ids', 'note'}
        qi, oi, pi = loc['query_ids'], loc['observation_indices'], loc['prefix_indices']
        assert len(qi) == len(set(qi)) and set(qi) <= set(qmap)
        assert len(oi) == len(set(oi)) and all(type(i) is int and 0 <= i < len(observations) for i in oi)
        assert len(pi) == len(set(pi)) and all(type(i) is int and 0 <= i <= len(observations) for i in pi)
        selected = [amap[q] for q in qi]
        def invariant(a):
            return len(payloads) >= 2 and a['distinct_values'] == 1 and qmap[a['query_id']]['fact'] not in observed
        if name == 'multiworld_unobserved_invariant':
            valid = any(invariant(a) for a in selected)
        elif name == 'conjunction_required_invariant':
            valid = False
            for a in selected:
                if not invariant(a):
                    continue
                fact = qmap[a['query_id']]['fact']
                for i, j in itertools.combinations(oi, 2):
                    if observations[i]['fact'] == observations[j]['fact']:
                        continue
                    supports = []
                    for omit in (i, j):
                        s = [w for w in task['worlds'] if all(scalar(w['facts'][o['fact']]) == scalar(o['equals'])
                             for k, o in enumerate(observations) if k != omit)]
                        supports.append(s)
                    valid |= all(len(partition(s, fact)) >= 2 for s in supports)
        elif name == 'useful_after_conjunctive_evidence':
            valid = bool(final) and len(observed) >= 2 and len(strict) >= 2 and any(a['distinct_values'] > 1 for a in selected)
        elif name == 'multiworld_all_queries_redundant':
            valid = len(payloads) >= 2 and all(a['distinct_values'] == 1 for a in target['queries']) and any(
                len(partition(final, f)) > 1 for f in set(task['worlds'][0]['facts']) - named)
        elif name == 'multiworld_entailed_null':
            valid = len(payloads) >= 2 and any(a['entailed'] and a['guaranteed'] is None for a in selected)
        elif name == 'ambiguous_null_partition':
            valid = any(a['distinct_values'] > 1 and any(c['value'] is None for c in a['counts']) for a in selected)
        elif name == 'cross_fact_inconsistent_conjunction':
            valid = False
            for i, j in itertools.combinations(oi, 2):
                a, b = observations[i], observations[j]
                sa = [w for w in task['worlds'] if scalar(w['facts'][a['fact']]) == scalar(a['equals'])]
                sb = [w for w in task['worlds'] if scalar(w['facts'][b['fact']]) == scalar(b['equals'])]
                valid |= a['fact'] != b['fact'] and bool(sa) and bool(sb) and not (set(ids(sa)) & set(ids(sb))) and not final
        elif name == 'post_empty_absorption':
            valid = False
            for j in oi:
                o = observations[j]
                earlier_empty = [n for n in pi if n <= j and n > 0 and not prefixes[n]]
                individual = [w for w in task['worlds'] if scalar(w['facts'][o['fact']]) == scalar(o['equals'])]
                valid |= bool(earlier_empty) and bool(individual) and not final and all(not p for p in prefixes[j + 1:])
        elif name == 'nonempty_repeated_observation':
            valid = any(prefixes[j] and ids(prefixes[j]) == ids(prefixes[j + 1]) and
                        any(canonical(observations[j]) == canonical(observations[i]) for i in oi if i < j)
                        for j in oi)
        elif name == 'typed_observation_separator':
            valid = False
            for j in oi:
                o = observations[j]
                av = alias(o['equals'])
                if type(av) is object:
                    continue
                pre = prefixes[j]
                exact = any(scalar(w['facts'][o['fact']]) == scalar(o['equals']) for w in pre)
                other = any(scalar(w['facts'][o['fact']]) == scalar(av) for w in pre)
                valid |= exact and other and bool(prefixes[j + 1])
        elif name == 'typed_query_partition_separator':
            valid = any(any(scalar(c['value']) == scalar(alias(d['value'])) for c in a['counts'] for d in a['counts']
                            if type(alias(d['value'])) is not object) for a in selected)
        elif name == 'skewed_possible_value_partition':
            valid = any(a['distinct_values'] > 1 and len({c['count'] for c in a['counts']}) > 1 for a in selected)
        else:
            raise AssertionError(('unpublished property', name))
        assert valid, (record['sample_id'], name)
        if 'supporting_world_ids' in loc:
            sw = loc['supporting_world_ids']
            assert sw == sorted(set(sw)) and set(sw) <= set(ids(task['worlds']))
    return tags


def main():
    paths = {
        'candidate': 'data/RBM-001/candidate_renderings/belief-gap-v1.jsonl',
        'expert': 'reviews/expert/belief-gap-v1.jsonl',
        'red': 'reviews/adversarial/belief-gap-v1.jsonl',
        'oracle': 'reviews/oracle/belief-gap-v1.jsonl',
        'expert_first': 'reviews/expert/belief-gap-v1.first_judgments.jsonl',
        'red_first': 'reviews/adversarial/belief-gap-v1.first_judgments.jsonl',
        'expert_execution': 'reviews/expert/belief-gap-v1.completed_execution.json',
        'expert_final_binding': 'reviews/expert/belief-gap-v1.final_artifact_receipt.json',
        'red_execution': 'audits/execution/belief-gap-v1-adversarial.json',
        'red_shortcut': 'audits/shortcut_detection/belief-gap-v1-adversarial.json',
        'root_belief_audit': 'audits/diversity/belief-gap-v1-belief-controls.json',
        'root_leakage': 'audits/leakage/belief-gap-v1.json',
        'public_properties': 'schemas/BELIEF_CONTROL_PROPERTIES.md',
        'public_curriculum': 'research/curricula/RBM-EVIDENCE-GAP.md',
        'public_semantics': 'schemas/EXTENDED_SEMANTICS.md',
    }
    hashes = {k: {'path': p, 'sha256': sha(BASE / p)} for k, p in paths.items()}
    assert hashes['candidate']['sha256'] == '423e3dc3b2ba79a656ce01a4e0ed25a40be0f1e74fa5362770f914e5555b027e'
    assert hashes['red']['sha256'] == 'eec54648acbcb9800a2f6959aeaf16daf89ff906caf465ec6fd5623b38be172e'
    assert hashes['red_execution']['sha256'] == '30f17ee0820527d870722b16920d705e944736ed4d3a1df4621f53adf1afe88c'
    rex = json.loads((BASE / paths['red_execution']).read_text())
    assert rex['status'] == 'completed' and rex['candidate_sha256'] == hashes['candidate']['sha256']
    assert rex['final_receipts_sha256'] == hashes['red']['sha256']
    assert sha(BASE / paths['expert']) == sha(BASE / paths['expert_first'])
    candidates = rows(paths['candidate'])
    reviewers = {k: {x['sample_id']: x for x in rows(paths[k])} for k in ('expert', 'red', 'oracle')}
    assert len(candidates) == 1296 and len({r['sample_id'] for r in candidates}) == 1296
    assert all(len(v) == 1296 and set(v) == {r['sample_id'] for r in candidates} for v in reviewers.values())
    details, properties = [], Counter()
    groups, split_stats, family_stats, scalar_stats, context_stats = defaultdict(list), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    scalar_renames_pass = 0
    marker_breaks = 0
    expected_by_id = {}
    for r in candidates:
        sid = r['sample_id']
        assert hashlib.sha256(canonical({k: v for k, v in r.items() if k != 'record_hash'}).encode()).hexdigest() == r['record_hash']
        prompt = r['rendered_input']
        assert prompt.count('\nTask JSON:\n') == 1
        task = json.loads(prompt.split('\nTask JSON:\n', 1)[1])
        assert canonical(task) == canonical(r['semantic_task'])
        assert 'controlled_context' not in prompt and 'property_witnesses' not in prompt
        target, prefixes = solve(task)
        expected_by_id[sid] = target
        assert canonical(target) == canonical(r['verified_target'])
        for name, data in reviewers.items():
            receipt = data[sid]
            assert receipt['record_hash'] == r['record_hash']
            if name == 'oracle':
                assert receipt['pass'] is True and not receipt['errors'] and not receipt['metadata_errors']
        tags = property_audit(r, task, target, prefixes)
        properties.update(tags)
        empty = not prefixes[-1]
        marker = any(type(o['equals']) is str and o['equals'] in {'stone', 'unknown'} for o in task['observations'])
        assert marker == empty
        for stats in (split_stats[r['split']], family_stats[str(r['source_family_index'])], context_stats[r['controlled_context']]):
            stats.update(records=1, empty_records=int(empty), marker_flag_correct=int(marker == empty))
        for value in {scalar(o['equals']) for o in task['observations']}:
            scalar_stats[value]['empty' if empty else 'nonempty'] += 1
        strings = sorted({v for w in task['worlds'] for v in w['facts'].values() if type(v) is str} |
                         {o['equals'] for o in task['observations'] if type(o['equals']) is str})
        mapping = {v: f'adjudicator_opaque_scalar_{i:03}' for i, v in enumerate(strings)}
        assert len(set(mapping.values())) == len(mapping) and not {'stone', 'unknown'} & set(mapping.values())
        renamed = copy.deepcopy(task)
        for w in renamed['worlds']:
            w['facts'] = {f: mapping.get(v, v) if type(v) is str else v for f, v in w['facts'].items()}
        for o in renamed['observations']:
            if type(o['equals']) is str:
                o['equals'] = mapping[o['equals']]
        rt, rp = solve(renamed)
        mapped = copy.deepcopy(target)
        for a in mapped['queries']:
            for c in a['counts']:
                if type(c['value']) is str:
                    c['value'] = mapping[c['value']]
            a['counts'].sort(key=lambda c: scalar(c['value']))
            if type(a['guaranteed']) is str:
                a['guaranteed'] = mapping[a['guaranteed']]
        assert canonical(mapped) == canonical(rt) and [ids(p) for p in rp] == [ids(p) for p in prefixes]
        scalar_renames_pass += 1
        marker_breaks += int(empty)
        assert r['source_family_index'] in (12, 13, 18, 19)
        assert r['split'] == ('train' if r['source_family_index'] % 10 < 8 else 'dev_challenge')
        roles = r['query_role_assignments']
        assert set(r['fact_role_assignments'].values()) == set(task['worlds'][0]['facts'])
        assert r['schedule_index'] == 9 * r['id_shift'] + 3 * r['position_shift'] + r['fact_name_shift']
        for role_idx, role in enumerate(('invariant_query', 'variation_query', 'observed_control')):
            loc = next(a for a in roles if a['role'] == role)
            assert loc['position'] == (role_idx + r['position_shift']) % 3
            assert loc['query_id'] == r['query_id_pool'][(role_idx + r['id_shift']) % 3]
            q = task['queries'][loc['position']]
            assert q['query_id'] == loc['query_id'] and q['fact'] == r['queried_fact_name_pool'][(role_idx + r['fact_name_shift']) % 3]
        groups[(r['source_family_index'], r['controlled_context'], r['renderer'])].append(r)
        details.append({'sample_id': sid, 'record_hash': r['record_hash'], 'context': r['controlled_context'],
                        'source_family_index': r['source_family_index'], 'split': r['split'],
                        'independent_expected_sha256': hashlib.sha256(canonical(target).encode()).hexdigest(),
                        'prefix_survivors': [ids(p) for p in prefixes], 'empty': empty, 'marker_present': marker,
                        'property_tags_independently_witnessed': sorted(tags),
                        'expert_decision': reviewers['expert'][sid]['decision'], 'red_decision': reviewers['red'][sid]['decision'],
                        'scalar_bijection_probes': 1, 'marker_wrong_after_opaque_rename': empty})
    assert len(groups) == 48
    for key, group in groups.items():
        assert len(group) == 27 and {r['schedule_index'] for r in group} == set(range(27))
        for role in ('invariant_query', 'variation_query', 'observed_control'):
            triples = Counter()
            actual_hist = defaultdict(list)
            for r in group:
                loc = next(a for a in r['query_role_assignments'] if a['role'] == role)
                q = r['semantic_task']['queries'][loc['position']]
                triple = (loc['query_id'], loc['position'], q['fact'])
                triples[triple] += 1
            assert len(triples) == 27 and set(triples.values()) == {1}
        # At each actual nuisance triple, all three roles occur and the full typed target/provenance histogram agrees.
        hist = defaultdict(list)
        for r in group:
            for pos, q in enumerate(r['semantic_task']['queries']):
                a = copy.deepcopy(expected_by_id[r['sample_id']]['queries'][pos])
                del a['query_id']
                sig = canonical({'answer': a, 'sizes': expected_by_id[r['sample_id']]['belief_sizes'],
                                 'survivors': expected_by_id[r['sample_id']]['surviving_worlds']})
                hist[(q['query_id'], pos, q['fact'])].append(sig)
        assert len(hist) == 27 and len({canonical(sorted(v)) for v in hist.values()}) == 1
    indexed = {(r['source_family_index'], r['controlled_context'], r['schedule_index'], r['renderer']): r for r in candidates}
    for r in candidates:
        if r['renderer'] == 'forward':
            other = indexed[(r['source_family_index'], r['controlled_context'], r['schedule_index'], 'reverse')]
            assert canonical(r['semantic_task']) == canonical(other['semantic_task'])
            assert canonical(r['verified_target']) == canonical(other['verified_target'])
            assert r['rendered_input'] != other['rendered_input']
    prior_path = 'data/RBM-001/releases/v0001/canonical.jsonl'
    prior = rows(prior_path)
    new_families = {r['source_family_key'] for r in candidates}
    prior_families = {r['semantic_family_id'] for r in prior}
    # The older release predates source_family_key; use its actual pNNNN family index.
    prior_indices = {int(f.rsplit('-p', 1)[1]) for f in prior_families}
    assert not {r['source_family_index'] for r in candidates} & prior_indices
    assert not {canonical(r['semantic_task']) for r in candidates} & {canonical(r['semantic_task']) for r in prior}
    assert not {len(r['semantic_task']['worlds']) for r in candidates} & {len(r['semantic_task']['worlds']) for r in prior}
    # A/D are retained only as dependent constituents, not partial family release approval.
    remaining = [r for r in candidates if reviewers['red'][r['sample_id']]['decision'] == 'ACCEPT']
    missing = {str(f): sorted(set('ABCDEF') - {r['controlled_context'] for r in remaining if r['source_family_index'] == f})
               for f in (12, 13, 18, 19)}
    assert all(v == ['E', 'F'] for v in missing.values())
    oracle_counts = Counter(x['pass'] for x in reviewers['oracle'].values())
    report = {'actual_execution_id': '/root/research_architect', 'status': 'COMPLETED_INDEPENDENT_INPUT_RECOMPUTATION',
              'input_evidence_hashes': hashes, 'prior_release_binding': {'path': prior_path, 'sha256': sha(BASE / prior_path)},
              'records': len(candidates), 'target_hash_prompt_schema_typed_two_solver_pass': len(candidates),
              'property_counts': dict(sorted(properties.items())), 'full_nuisance_triple_groups_pass': len(groups),
              'field_order_pairs': 648, 'source_families': 4, 'base_contexts': 24,
              'actual_split_stats': {k: dict(v) for k, v in split_stats.items()},
              'actual_family_stats': {k: dict(v) for k, v in family_stats.items()},
              'actual_context_stats': {k: dict(v) for k, v in context_stats.items()},
              'typed_observation_scalar_anywhere_outcome_histogram': {k: dict(v) for k, v in scalar_stats.items()},
              'marker_rule': 'empty iff any observation.equals is a string in {stone,unknown}',
              'marker_empty_flag_correct': 1296, 'marker_flag_query_observations_correct': 3888,
              'opaque_scalar_bijection_equivariance_pass': scalar_renames_pass,
              'marker_rule_wrong_after_bijection': marker_breaks,
              'expert_decisions': dict(Counter(r['decision'] for r in reviewers['expert'].values())),
              'red_decisions': dict(Counter(r['decision'] for r in reviewers['red'].values())),
              'oracle_pass_rows': oracle_counts[True], 'reviewer_disagreement_rows': 432,
              'red_accept_partial_release_missing_contexts': missing,
              'source_family_index_overlap_with_prior': 0, 'exact_task_overlap_with_prior': 0,
              'prior_lineage_field': 'semantic_family_id pNNNN indices; older release has no source_family_key',
              'different_world_count_sizes_than_prior': True,
              'independence': 'Actual original public-design author/later adjudicator; no producer or initial reviewer authorship. Reviewer findings read as authorized adjudication evidence, not a blind first review. Same-model/common-design risks.',
              'probe_scope': 'Audit only; scalar bijections create zero dataset records/families.',
              'per_record': details}
    # Preserve current machine reports as content-addressed bytes before possible corrected reruns.
    snapshots = []
    for k in ('root_belief_audit', 'root_leakage'):
        h = hashes[k]['sha256']
        rel = f'reviews/adjudication/belief-gap-v1-source-snapshots/{h}.json'
        dst = BASE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            with dst.open('xb') as f:
                f.write((BASE / paths[k]).read_bytes())
        assert sha(dst) == h
        snapshots.append({'original_path': paths[k], 'sha256': h, 'preserved_path': rel})
    report['preserved_machine_source_snapshots'] = snapshots
    old_sources = [
        'audits/leakage/history/belief-gap-v1-70a9aae7018bff7489e8967e0c97eed74e3e693353563443650c0c19ea0e1daf.json',
        'audits/diversity/history/belief-gap-v1-belief-controls-d6ecd515ca09f88b2f293c7050a4d79df349f6e221da425f70ca1fd1800cf11d.json',
        'reviews/oracle/history/belief-gap-v1-38c76a2aa17296728dff99c1823af384326fb49ad78c6ca3d2c9154dbd1239c7.jsonl',
    ]
    report['preserved_first_machine_reports'] = [{'path': p, 'sha256': sha(BASE / p)} for p in old_sources]
    current_leakage = json.loads((BASE / paths['root_leakage']).read_text())
    report['current_machine_counts'] = {k: current_leakage[k] for k in
        ('records', 'oracle_failures', 'metadata_failures', 'validation_failures', 'belief_control_error_count')}
    assert all(report['current_machine_counts'][k] == 0 for k in report['current_machine_counts'] if k != 'records')
    dest = OUT / 'belief-gap-v1.independent_adjudicator_audit.json'
    with dest.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(canonical({k: v for k, v in report.items() if k not in ('per_record', 'input_evidence_hashes')}))


if __name__ == '__main__':
    main()
