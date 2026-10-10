"""Read-only exporter verification, using expert-owned prompt-derived semantics."""
import collections
import copy
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from factory.exports import specialized
from factory.common import canonical
from independent_review import critic, same
from jsonschema import Draft202012Validator

def without_id(c):
    return {k:v for k,v in c.items() if k != 'candidate_id'}

def run(batch):
    source = BASE / f'data/RVM-001/candidate_renderings/{batch}.jsonl'
    rows = [json.loads(l) for l in source.read_text().splitlines()]
    schema = json.loads((BASE/'schemas/process_reward/v1.schema.json').read_text())
    validator = Draft202012Validator(schema)
    counts = collections.Counter()
    crossings = collections.defaultdict(list)
    examples = []
    for r in rows:
        task=json.loads(r['rendered_input'].split('Task JSON:',1)[1])
        computed=critic(task)
        assert same(computed,r['verified_target']), r['sample_id']
        rank={cid:i for i,g in enumerate(computed['ranking']) for cid in g}
        labels={x['candidate_id']:x for x in computed['evaluations']}
        emitted=specialized([r],r['split'])
        pairs=emitted[r['split']+'-preference.jsonl']
        for p in emitted[r['split']+'-process_reward.jsonl']:
            validator.validate(p)
            assert same(p['labels'],labels[p['candidate']['candidate_id']])
            counts['process_labels']+=1
        normalized=[]
        for p in pairs:
            good=json.loads(p['chosen']);bad=json.loads(p['rejected'])
            data=json.loads(p['prompt'].split('World and response options JSON:\n',1)[1])
            assert same(data['world'],task['world'])
            assert len(data['options'])==2 and good in data['options'] and bad in data['options']
            assert rank[good['candidate_id']] < rank[bad['candidate_id']]
            bucket=(len(good['plan'])>len(bad['plan']))-(len(good['plan'])<len(bad['plan']))
            cdelta=len(p['chosen'])-len(p['rejected'])
            pos=data['options'].index(good)
            counts['pairs']+=1;counts[f'plan_diff_{bucket}']+=1
            counts[f'char_diff_{(cdelta>0)-(cdelta<0)}']+=1
            counts[f'position_{pos}']+=1
            counts['lower_id_chosen']+=int(good['candidate_id']<bad['candidate_id'])
            counts['chosen_invalid']+=int(not labels[good['candidate_id']]['valid'])
            counts['chosen_wrong_claim']+=int(not labels[good['candidate_id']]['claim_correct'])
            crossings[(r['semantic_parent_id'],bucket,canonical(without_id(good)),canonical(without_id(bad)))].append(pos)
            normalized.append((bucket,without_id(good),without_id(bad)))
            if len(examples)<6:examples.append({'source':r['sample_id'],'chosen':good,'rejected':bad,'ranking_groups':[rank[good['candidate_id']],rank[bad['candidate_id']]],'chosen_labels':labels[good['candidate_id']],'rejected_labels':labels[bad['candidate_id']]})
        # Deliberately reverse ID ordering, then reorder candidates/evaluation
        # arrays and ranking tie groups. Selection must retain semantic pairs.
        renamed=copy.deepcopy(r)
        mapping={c['candidate_id']:f'renamed-{len(task["candidates"])-i:03d}' for i,c in enumerate(task['candidates'])}
        for c in renamed['semantic_task']['candidates']:c['candidate_id']=mapping[c['candidate_id']]
        renamed['semantic_task']['candidates'].reverse()
        for e in renamed['verified_target']['evaluations']:e['candidate_id']=mapping[e['candidate_id']]
        renamed['verified_target']['evaluations'].reverse()
        renamed['verified_target']['ranking']=[[mapping[c] for c in g] for g in renamed['verified_target']['ranking']]
        renamed_pairs=specialized([renamed],r['split'])[r['split']+'-preference.jsonl']
        normal2=[]
        for p in renamed_pairs:
            good=json.loads(p['chosen']);bad=json.loads(p['rejected'])
            bucket=(len(good['plan'])>len(bad['plan']))-(len(good['plan'])<len(bad['plan']))
            normal2.append((bucket,without_id(good),without_id(bad)))
        assert same(normalized,normal2), r['sample_id']+' semantic selection changed under renaming/reordering'
        counts['rename_and_order_invariance']+=1
    malformed={'input':'x','candidate':{},'labels':{'valid':'yes','claim_correct':None,'progress':[],'usefulness':{}}}
    assert list(validator.iter_errors(malformed)), 'malformed labels accepted'
    # Mutation tests of complete otherwise-valid row to test specific domains.
    valid=specialized([rows[0]],rows[0]['split'])[rows[0]['split']+'-process_reward.jsonl'][0]
    mutations=[('valid','yes'),('claim_correct',None),('reaches_goal',1),('remaining_moves',-1),('remaining_moves',False),('progress',[]),('usefulness',3),('first_error',-1),('first_error','other'),('repair_plan',[5])]
    for field,value in mutations:
        altered=copy.deepcopy(valid);altered['labels'][field]=value
        assert list(validator.iter_errors(altered)), (field,value)
    return {'batch':batch,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'rows':len(rows),'counts':dict(counts),'position_crossings':dict(collections.Counter(str(sorted(v)) for v in crossings.values())),'examples':examples,'typed_schema_mutations_rejected':len(mutations)+1}

if __name__=='__main__':
    reports=[run(b) for b in ('pilot-v2','expansion-v1')]
    (BASE/'reviews/expert/exporter_revised_recomputation.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='examples'} for r in reports],indent=2))
