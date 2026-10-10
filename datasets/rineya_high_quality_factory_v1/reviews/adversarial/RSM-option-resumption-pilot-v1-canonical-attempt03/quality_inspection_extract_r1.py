"""Owned manual-inspection extractor, not a scorer and no semantic helper imports.
Read actual canonical tasks and frozen own factual FIRST/predictions only. It emits
full family programs/context deltas and all216 row summaries for actual agent use.
Source/current inputs must match the separately anchored inspection preexec first.
"""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
ACTOR='/root/adversarial_auditor'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def same(a,b):return type(a) is type(b) and (set(a)==set(b) and all(same(a[k],b[k]) for k in a) if type(a) is dict else len(a)==len(b) and all(same(x,y) for x,y in zip(a,b)) if type(a) is list else a==b)
def bind(path):
    raw=path.read_bytes();return {'path':str(path.relative_to(ROOT)),'size_bytes':len(raw),'sha256':sha(raw)}
def projection(output):return {'status':output['status'],'state':output['checkpoint']['state'] if output['checkpoint'] else None,'events':[e for e in output['new_events'] if e['event']!='enter']}
def write(name,obj):
    with (OUT/name).open('x') as handle:json.dump(obj,handle,indent=2);handle.write('\n')
    return bind(OUT/name)
def main():
    expected=sys.argv[1];preexec=OUT/'QUALITY_INSPECTION_PREEXEC.json'
    if sha(preexec.read_bytes())!=expected:raise ValueError('external inspection preexec SHA')
    freeze=json.loads(preexec.read_text())
    for row in freeze['files']:
        p=ROOT/row['path'];a=p.read_bytes();b=(ROOT/row['archive']).read_bytes()
        if a!=b or len(a)!=row['size_bytes'] or sha(a)!=row['sha256']:raise ValueError('inspection exactpair')
    sourcefreeze=json.loads((OUT/'SOURCE_FREEZE.json').read_text())
    for row in sourcefreeze['operative_bindings']:
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p;a=p.read_bytes();b=(ROOT/row['archive']).read_bytes()
        if a!=b or len(a)!=row['size_bytes'] or sha(a)!=row['sha256']:raise ValueError('original closure exactpair')
    rows=read(ROOT/'data/RSM-001/candidate_renderings/option-resumption-pilot-v1.jsonl')
    facts={r['sample_id']:r for r in read(OUT/'FACTUAL_FIRST.jsonl')}
    first={r['sample_id']:r['expected_output'] for r in read(ROOT/'reviews/adversarial/RSM-option-resumption-pilot-v1-input-first-attempt01/frame_predictions.jsonl')}
    families={}
    for row in rows:families.setdefault(row['source_family_index'],{})[(row['controlled_context'],row['renderer'])]=row
    summaries=[];programs=[];operational=[]
    for index,group in sorted(families.items()):
        frow=group[('F','forward')];rrow=group[('R','forward')];ft=frow['semantic_task'];rt=rrow['semantic_task']
        f=first[frow['sample_id']];a=first[group[('A','forward')]['sample_id']];r=first[rrow['sample_id']]
        witness=facts[rrow['sample_id']]['actual_witness_facts']
        # Excluding ALL enter events, not just one binding key, checks genuine
        # effect/branch/return behavior. Final-state mutation is stronger evidence
        # than a copied enter-event binding change for this particular locator.
        op={'family_index':index,'FA_operational_projection_differs':not same(projection(f),projection(a)),
            'FA_final_state_differs':not same(f['checkpoint']['state'],a['checkpoint']['state']),
            'saved_locator':witness['saved_binding_probe']['actual_path'],
            'saved_locator_diagnostic_final':witness['saved_binding_probe']['diagnostic_final_state'],
            'unmodified_R_final':r['checkpoint']['state'],
            'saved_locator_changes_final_state':not same(witness['saved_binding_probe']['diagnostic_final_state'],r['checkpoint']['state'])}
        operational.append(op)
        deltas=[]
        for context in 'FRTPUAISX':
            row=group[(context,'forward')];t=row['semantic_task'];o=first[row['sample_id']]
            deltas.append({'context':context,'entry':t['entry'],'mode':t['mode'],'budget':t['event_budget'],
                'history_events':len(t['history']),'checkpoint':t['checkpoint'],
                'root_initiation':t['options'][t['entry']['option']]['initiation'],
                'root_success':t['options'][t['entry']['option']]['success'],
                'status':o['status'],'final_state':o['checkpoint']['state'] if o['checkpoint'] else None})
        programs.append({'family_index':index,'family':frow['semantic_family_id'],'source_stratum':frow['source_provenance']['source_stratum'],
            'initial_state':ft['initial_state'],'entry':ft['entry'],'options':ft['options'],'context_inputs':deltas,'operational_facts':op})
    for row in rows:
        sid=row['sample_id'];o=first[sid];t=row['semantic_task'];fact=facts[sid]
        summaries.append({'sample_id':sid,'record_hash':row['record_hash'],'family_index':row['source_family_index'],'context':row['controlled_context'],
            'renderer':row['renderer'],'split':row['split'],'difficulty':row['difficulty'],'explanation':row['source_provenance']['curriculum_difficulty_explanation'],
            'mode':t['mode'],'history_events':len(t['history']),'budget':t['event_budget'],'status':o['status'],'new_events':len(o['new_events']),
            'new_effects':o['new_effects'],'final_state':o['checkpoint']['state'] if o['checkpoint'] else None,
            'failed_factual_checks':[c for c in fact['checks'] if not c['passed']],'target_facts':fact['target_facts'],
            'retained_frames':t['checkpoint']['frames'] if t['checkpoint'] else None,'tags':row['skill_tags']})
    files=[write('MANUAL_ROW_INSPECTION.json',summaries),write('MANUAL_FAMILY_PROGRAMS.json',programs),write('OPERATIONAL_ARGUMENT_INSPECTION.json',operational)]
    # Source/current closure is rechecked after extraction; no scores/decisions.
    for row in freeze['files']+sourcefreeze['operative_bindings']:
        p=Path(row['path']);p=p if p.is_absolute() else ROOT/p;a=p.read_bytes();b=(ROOT/row['archive']).read_bytes()
        if a!=b or len(a)!=row['size_bytes'] or sha(a)!=row['sha256']:raise ValueError('postinspection exactpair')
    execution=write('MANUAL_EXTRACTION_EXECUTION.json',{'actor':ACTOR,'stage':'CANONICAL_AGENT_INSPECTION_MATERIAL_ONLY','rows':len(rows),'families':len(families),
        'files':files,'preexec':bind(preexec),'no_quality_scores_or_decisions':True,'new_semantic_helper_imports':0,'all_original_and_new_current_raw_before_after':True})
    print(json.dumps({'execution':execution,'files':files,'operational':operational,'rows':len(rows)}))
if __name__=='__main__':main()
