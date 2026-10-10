"""Cross-release family and exact-semantic reuse checks."""
import collections
import re
import copy
from factory.common import digest


def source_key(row):
    if row.get('source_family_key'):return row['source_family_key']
    match=re.search(r'-p(\d+)$',row['semantic_parent_id'])
    assert match, 'missing source family index'
    return f"{row['track_id']}-seed{row['generation_seed']}-p{int(match.group(1)):04d}"


def semantic_fingerprint(task):
    task=copy.deepcopy(task);task.pop('annotation',None)
    if task['kind']=='critic':
        for candidate in task['candidates']:candidate.pop('candidate_id',None)
        task['candidates']=sorted(task['candidates'],key=digest)
        world=task['world'];world.pop('plan',None)
    else:world=task
    if world['kind']=='graph':
        world['nodes']=sorted(world['nodes']);world['edges']=sorted(world['edges'])
    return digest(task)


def cross_release(rows, released):
    old_keys=collections.defaultdict(set)
    old_semantics=collections.defaultdict(set)
    old_ids=set()
    for row in released:
        old_keys[source_key(row)].add(row['split'])
        old_semantics[semantic_fingerprint(row['semantic_task'])].add(row['semantic_parent_id'])
        old_ids.add(row['sample_id'])
    result={'family_overlap':[],'split_leakage':[],'cross_parent_semantic_duplicates':[],'already_released_samples':[]}
    seen=set()
    for row in rows:
        key=source_key(row)
        if key in old_keys and key not in seen:
            result['family_overlap'].append(key)
            if old_keys[key]!={row['split']}:result['split_leakage'].append(key)
            seen.add(key)
        other=old_semantics[semantic_fingerprint(row['semantic_task'])]-{row['semantic_parent_id']}
        if other:result['cross_parent_semantic_duplicates'].append({'sample_id':row['sample_id'],'prior_parents':sorted(other)})
        if row['sample_id'] in old_ids:result['already_released_samples'].append(row['sample_id'])
    return result
