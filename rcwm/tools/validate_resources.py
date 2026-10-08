#!/usr/bin/env python3
"""Validate bundled research resources. Does not qualify a native experiment."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import jsonschema
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'reference'))
from rcwm.data import validate_open_loop


def validate_record(record: dict) -> None:
    schema=json.loads((ROOT/'schemas/trajectory.schema.json').read_text())
    jsonschema.Draft202012Validator(schema).validate(record)
    validate_open_loop(record)
    actions=record['actions']
    ids=[a['action_id'] for a in actions]
    if len(set(ids)) != len(ids):raise ValueError('duplicate action IDs')
    for i,a in enumerate(actions):
        if any(p not in ids[:i] for p in a['parents']):
            raise ValueError('dependencies must refer to earlier action IDs')
    seen=set(); previous_end=record['decision_time']
    for o in record['actual_outcomes']:
        if o['action_id'] not in ids or o['action_id'] in seen:
            raise ValueError('unknown or duplicate actual outcome')
        seen.add(o['action_id'])
        if o['end'] < o['start'] or o['start'] < previous_end:
            raise ValueError('initial serial schema requires nonoverlapping increasing times')
        previous_end=o['end']
        if len(o['features']) != len(o['feature_valid']):
            raise ValueError('feature/value mask mismatch')
        if 'goal_weights' in record and len(record['goal_weights']) != len(o['features']):
            raise ValueError('goal/feature dimension mismatch')


def main():
    for p in (ROOT/'schemas').glob('*.json'):
        jsonschema.Draft202012Validator.check_schema(json.loads(p.read_text()))
    records=[json.loads(line) for line in (ROOT/'examples/trajectories.jsonl').read_text().splitlines()]
    parent_splits={}
    for r in records:
        validate_record(r)
        if r['parent_id'] in parent_splits and parent_splits[r['parent_id']]!=r['split']:
            raise ValueError('parent appears in multiple splits')
        parent_splits[r['parent_id']]=r['split']
    validate_record(json.loads((ROOT/'examples/trajectory.json').read_text()))
    for ex,sc in [('prediction.json','prediction.schema.json'),('run_manifest.json','run_manifest.schema.json')]:
        jsonschema.validate(json.loads((ROOT/'examples'/ex).read_text()),
                            json.loads((ROOT/'schemas'/sc).read_text()))
    for p in (ROOT/'configs').glob('*.yaml'):
        data=yaml.safe_load(p.read_text())
        if not isinstance(data,dict):raise ValueError(f'nonmapping config {p.name}')
    sources=json.loads((ROOT/'research/sources.json').read_text())
    if len({r['id'] for r in sources})!=len(sources):raise ValueError('duplicate source IDs')
    if any(not r['url'].startswith('https://') for r in sources):raise ValueError('invalid source URL')
    import re
    for p in (ROOT/'report').glob('*.md'):
        for sid in re.findall(r'\bS\d{2}\b',p.read_text()):
            if sid not in {s['id'] for s in sources}:raise ValueError(f'unknown source {sid}')
    result={'status':'PASS','schema_files':len(list((ROOT/'schemas').glob('*.json'))),
            'trajectory_records':len(records),'independent_example_parent_ids':len(parent_splits),
            'config_files':len(list((ROOT/'configs').glob('*.yaml'))),'source_entries':len(sources),
            'scope':'Bundled example/schema/config validation only; no native qualification.'}
    (ROOT/'validation/resource_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
