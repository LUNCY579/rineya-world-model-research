"""Fresh input-only stronger numeric/coarse attacks after own first predictions.
No canonical targets/control manifest read; first files remain immutable.
"""
import copy,hashlib,importlib.util,json,sys
from collections import Counter,defaultdict
from pathlib import Path
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent
PREP=R/'reviews/adversarial/RSM-option-resumption-public-preparation-20261010T1112Z'

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'))
def integers(v):
    if type(v) is int:yield v
    elif type(v) is dict:
        for a in v.values():yield from integers(a)
    elif type(v) is list:
        for a in v:yield from integers(a)
def status_hist(rows,pred):return dict(Counter(pred[r['sample_id']]['status'] for r in rows))
def main():
    freeze=json.loads((O/'SHORTCUT_PREEXEC.json').read_text())
    for b in freeze['bindings']:
        p=R/b['path'] if not Path(b['path']).is_absolute() else Path(b['path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest()==b['sha256']
    sys.path.insert(0,str(PREP));import frame_solver_r1 as frame
    rows=[frame.strict_json(x) for x in (R/'reviews/input_only/RSM-option-resumption-pilot-v1.jsonl').read_text().splitlines()]
    pred={r['sample_id']:r['expected_output'] for r in map(json.loads,(O/'frame_predictions.jsonl').read_text().splitlines())}
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']!='train'];major=Counter(pred[r['sample_id']]['status'] for r in train).most_common(1)[0][0]
    def shallow(task):
        cp=task['checkpoint'];history=task['history']
        if cp is None:return [False,False,False]
        effects=[e for e in history if e['event']=='effect'];state_bad=bool(effects and not frame.typed_equal(cp['state'],effects[-1]['state']))
        frames=cp['frames'];binding_bad=any(a['instance_id']==b['instance_id'] and not frame.typed_equal(a['bindings'],b['bindings']) for a in frames for b in frames)
        cursor_bad=False
        if history and history[-1]['event']=='effect' and frames:
            last=history[-1];top=frames[-1]
            if top['instance_id']==last['instance_id'] and top['body_path']==last['body_path']:cursor_bad=top['next_index']!=last['index']+1
        return [state_bad,binding_bad,cursor_bad]
    def coarse(row):
        t=row['semantic_task'];root=t['options'][t['entry']['option']]
        nums=list(integers(root['success']))
        return [t['mode'],len(t['history']),t['event_budget'],sorted(canonical(v) for v in t['entry']['args'].values()),max(nums,default=0)]
    features={
      'student_root_success_max_literal':lambda r:max(integers(r['semantic_task']['options'][r['semantic_task']['entry']['option']]['success']),default=0),
      'student_program_max_abs_integer':lambda r:max(abs(n) for n in integers(r['semantic_task']['options'])),
      'student_numeric_sign_count':lambda r:sum(n<0 for n in integers(r['semantic_task'])),
      'student_mode_budget_rootargs_successmax':coarse,
      'student_coarse_plus_three_shallow_checkpoint_flags':lambda r:coarse(r)+shallow(r['semantic_task']),
      'student_shallow_checkpoint_flags':lambda r:shallow(r['semantic_task']),
      'student_rendered_top_first_key':lambda r:next(iter(frame.strict_json(r['rendered_input'].rsplit('Task JSON:\n',1)[1]))),
      'student_rendered_top_key_order':lambda r:list(frame.strict_json(r['rendered_input'].rsplit('Task JSON:\n',1)[1])),
      'student_program_declared_opcode_counts':lambda r:dict(Counter(findops(r['semantic_task']['options']))),
      'external_join_row_context':lambda r:r['sample_id'].split('-')[-2],
    }
    def baseline(fn):
        table=defaultdict(Counter)
        for r in train:table[canonical(fn(r))][pred[r['sample_id']]['status']]+=1
        def predict(row):
            c=table.get(canonical(fn(row)))
            return sorted(c,key=lambda k:(-c[k],k))[0] if c else major
        errors=[{'sample_id':r['sample_id'],'feature':fn(r),'prediction':predict(r),'actual_own_status':pred[r['sample_id']]['status']} for r in dev if predict(r)!=pred[r['sample_id']]['status']]
        counts=defaultdict(lambda:[0,0])
        for r in dev:
            a=pred[r['sample_id']]['status'];counts[a][1]+=1;counts[a][0]+=a==predict(r)
        collision=None;allgroups=defaultdict(list)
        for r in rows:allgroups[canonical(fn(r))].append(r)
        for g in allgroups.values():
            for a in g:
                for b in g:
                    if pred[a['sample_id']]['status']!=pred[b['sample_id']]['status']:
                        collision={'sample_ids':[a['sample_id'],b['sample_id']],'feature':fn(a),'statuses':[pred[a['sample_id']]['status'],pred[b['sample_id']]['status']]};break
                if collision:break
            if collision:break
        return {'train_keys':len(table),'dev_correct':len(dev)-len(errors),'dev_rows':len(dev),'dev_accuracy':1-len(errors)/len(dev),
                'dev_macro_accuracy':sum(a/b for a,b in counts.values())/len(counts),'dev_key_seen_rows':sum(canonical(fn(r)) in table for r in dev),
                'first_errors':errors[:4],'feature_collision':collision},predict
    results={};predictors={}
    for name,fn in features.items():results[name],predictors[name]=baseline(fn)
    counterexamples=[];checks=[]
    for row in rows:
        if not row['sample_id'].endswith('-F-forward'):continue
        family=row['semantic_family_id'];srow=next(r for r in rows if r['semantic_family_id']==family and r['sample_id'].endswith('-S-forward'))
        new=copy.deepcopy(row);task=new['semantic_task'];root=task['options'][task['entry']['option']]
        n=features['student_root_success_max_literal'](srow)
        root['success']['all'].append({'left':{'const':n},'cmp':'eq','right':{'const':n}})
        actual=frame.solve(task);equal=frame.typed_equal(actual,pred[row['sample_id']]);checks.append({'name':'truthful_large_success_tautology_'+row['sample_id'],'passed':equal})
        for name in ['student_root_success_max_literal','student_program_max_abs_integer','student_mode_budget_rootargs_successmax','student_coarse_plus_three_shallow_checkpoint_flags']:
            oldstatus=predictors[name](row);newstatus=predictors[name](new)
            if newstatus!=actual['status']:
                counterexamples.append({'parent_sample_id':row['sample_id'],'parent_record_hash':row['record_hash'],'matched_success_failure_sample':srow['sample_id'],
                    'baseline':name,'old_prediction':oldstatus,'new_wrong_prediction':newstatus,'own_status':actual['status'],
                    'changed_path':['options',task['entry']['option'],'success','all','append'],'added_atom':root['success']['all'][-1],
                    'full_typed_output_unchanged':equal,'mutation':'one always-true success atom; no events/body effects changed','descendant_is_new_source_family':False})
    report={'actor':'/root/adversarial_auditor','stage':'input-only numeric/coarse supplement after initial predictions freeze',
        'gold_access':False,'train_only':True,'model_training':False,'baselines':results,'actual_transform_checks':checks,
        'minimal_counterexamples':counterexamples,'input_status_distribution':status_hist(rows,pred),
        'inferences':['Shallow checkpoint flags check visible local consistency without executing full program. Valid semantic preflight may be useful, but whole-cohort status transfer can bypass procedure execution.',
                      'Coarse status lookup success does not solve typed event traces/state. Tautology descendants are audit interventions, not new independent program families.',
                      'Only own predicted outputs supply labels; canonical correctness still UNKNOWN. No corpus control claims or release approval.'],
        'before_freeze':freeze,'all_bound_bytes_after_equal':True}
    for b in freeze['bindings']:
        p=R/b['path'] if not Path(b['path']).is_absolute() else Path(b['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==b['sha256']
    p=O/'NUMERIC_SHORTCUT_EXECUTION.json'
    with p.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'execution_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'checks':len(checks),'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),'counterexamples':len(counterexamples),'status_baselines':{k:v['dev_correct'] for k,v in results.items()}}))

def findops(value):
    if type(value) is dict:
        if 'op' in value:yield value['op']
        for v in value.values():yield from findops(v)
    elif type(value) is list:
        for v in value:yield from findops(v)

if __name__=='__main__':main()
