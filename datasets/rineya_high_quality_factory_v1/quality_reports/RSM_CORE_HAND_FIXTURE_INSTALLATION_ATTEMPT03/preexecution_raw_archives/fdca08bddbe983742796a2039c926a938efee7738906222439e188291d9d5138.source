"""Validate derived formats against independently authored oracle semantics."""
import json
from collections import Counter
from jsonschema import Draft202012Validator
from factory.common import ROOT, canonical, digest, load_jsonl
from oracles.independent import solve
from factory.rsm_controls import validate_rsm_specialized


def validate_specialized(path, rows, batch=None):
    validate_rsm_specialized(path,rows,batch=batch)
    counts = {}
    by_input = {r['rendered_input']: r for r in rows}
    for file in sorted(path.glob('*.jsonl')):
        name = file.name
        records = load_jsonl(file)
        if name.endswith('-preference.jsonl'):
            schema = json.loads((ROOT/'schemas/preference/v1.schema.json').read_text())
            paired = {}
            for item in records:
                Draft202012Validator(schema).validate(item)
                data = json.loads(item['prompt'].split('World and response options JSON:\n', 1)[1])
                chosen, rejected = json.loads(item['chosen']), json.loads(item['rejected'])
                assert len(data['options']) == 2 and canonical(chosen) != canonical(rejected)
                assert sorted(map(canonical, data['options'])) == sorted(map(canonical, [chosen, rejected]))
                split = 'dev_challenge' if name.startswith('dev_challenge-') else 'train'
                assert any(r['split']==split and canonical(r['semantic_task']['world'])==canonical(data['world'])
                           and canonical(chosen) in map(canonical,r['semantic_task']['candidates']) and canonical(rejected) in map(canonical,r['semantic_task']['candidates'])
                           for r in rows), 'preference outside canonical split'
                verdict = solve({'kind':'critic','world':data['world'],'candidates':data['options']})
                ranking = {cid:i for i,group in enumerate(verdict['ranking']) for cid in group}
                assert ranking[chosen['candidate_id']] < ranking[rejected['candidate_id']], 'preference must be strict'
                key = digest({'world':data['world'],'chosen':chosen,'rejected':rejected})
                paired.setdefault(key, []).append(list(map(canonical,data['options'])).index(canonical(chosen)))
            assert all(sorted(v) == [0,1] for v in paired.values()), 'preference positions must be crossed'
        elif name.endswith('-process_reward.jsonl'):
            schema = json.loads((ROOT/'schemas/process_reward/v1.schema.json').read_text())
            for item in records:
                Draft202012Validator(schema).validate(item)
                parent = by_input[item['input']]
                assert canonical(item['candidate']) in map(canonical,parent['semantic_task']['candidates'])
                task = {'kind':'critic','world':parent['semantic_task']['world'],'candidates':[item['candidate']]}
                assert canonical(solve(task)['evaluations'][0]) == canonical(item['labels']), 'candidate process labels mismatch'
        elif name.endswith('-trajectories.jsonl'):
            for item in records:
                parent = by_input[item['input']]
                verdict = solve(parent['semantic_task'])
                assert verdict['valid'] and verdict['reaches_goal'], 'trajectory must reach goal legally'
                assert canonical(item) == canonical({'input':parent['rendered_input'],'actions':parent['semantic_task']['plan'],
                                'steps':verdict['steps'],'after':verdict['after']})
        elif any(name.endswith('-'+tag+'.jsonl') for tag in ['skill_demonstrations','memory_updates','belief_updates']):
            for item in records:
                parent = by_input[item['input']]
                assert canonical(item) == canonical({'input':parent['rendered_input'],'target':solve(parent['semantic_task'])})
        else:
            continue
        # Membership in a split is checked from the original input, except the
        # transformed preference prompts whose full world and candidates are above.
        split = 'dev_challenge' if name.startswith('dev_challenge-') else 'train'
        for item in records:
            if 'input' in item: assert by_input[item['input']]['split'] == split
        selected = [r for r in rows if r['split'] == split]
        if any(name.endswith('-'+tag+'.jsonl') for tag in ['skill_demonstrations','memory_updates','belief_updates']):
            assert Counter(item['input'] for item in records) == Counter(r['rendered_input'] for r in selected), 'specialized one-per-row coverage mismatch'
        elif name.endswith('-process_reward.jsonl'):
            observed = Counter(canonical({'input':item['input'],'candidate':item['candidate']}) for item in records)
            expected = Counter(canonical({'input':r['rendered_input'],'candidate':candidate})
                               for r in selected for candidate in r['semantic_task']['candidates'])
            assert observed == expected, 'process candidate coverage mismatch'
        elif name.endswith('-trajectories.jsonl'):
            expected = []
            for r in selected:
                verdict = solve(r['semantic_task'])
                if verdict['valid'] and verdict['reaches_goal']:
                    expected.append(r['rendered_input'])
            assert Counter(item['input'] for item in records) == Counter(expected), 'selected trajectory coverage mismatch'
        elif name.endswith('-preference.jsonl'):
            # The export samples one strict pair per submitted-plan length
            # difference bucket in each canonical row; field-order twins cross
            # the option positions. This checks coverage without choosing the
            # exporter's particular representative pair.
            expected = Counter()
            for r in selected:
                task = r['semantic_task']
                verdict = solve(task)
                rank = {cid:i for i,group in enumerate(verdict['ranking']) for cid in group}
                buckets = set()
                for chosen in task['candidates']:
                    for rejected in task['candidates']:
                        if rank[chosen['candidate_id']] < rank[rejected['candidate_id']]:
                            delta = len(chosen['plan']) - len(rejected['plan'])
                            buckets.add((delta > 0) - (delta < 0))
                for bucket in buckets:
                    expected[(canonical(task['world']), bucket)] += 1
            observed = Counter()
            for item in records:
                data = json.loads(item['prompt'].split('World and response options JSON:\n', 1)[1])
                chosen, rejected = json.loads(item['chosen']), json.loads(item['rejected'])
                delta = len(chosen['plan']) - len(rejected['plan'])
                observed[(canonical(data['world']), (delta > 0) - (delta < 0))] += 1
            assert observed == expected, 'preference length-bucket coverage mismatch'
        counts[name] = len(records)
    return counts
