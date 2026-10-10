#!/usr/bin/env python3
"""Owned same-task renderer surface supplement; no current gold/manifests read."""
import collections,copy,hashlib,importlib.util,json,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1];EXEC='/root/adversarial_auditor'
BINDINGS={'reviews/input_only/planning-gap-v2-renderer-r1.jsonl':'5f0b3490720933caa7895f3223244e5e72ed8ca5fb89a97f2eeb516c151f58e3','reviews/adversarial/planning-gap-v2-renderer-r1.input_predictions.jsonl':'6738599d9285a779772ce36f61b6f243cf4f85e3e26744d0042cf2aa853d2d3e','reviews/adversarial/planning-gap-v2-renderer-r1.input_first_judgments.jsonl':'ecad49409af2f3715a91f4f2d3b2041cb5a09779718633739adacd59e5d8f0dc','audits/rgm_planning_gap_v2_renderer_r1_adversarial_input.py':'22a26b7e88736badf99eaaa9a2372247fed1f4eb24a607b4e7aac62b25659b47'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def jb(v):return (json.dumps(v,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
def freeze(p,data):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
def main():
 for rel,h in BINDINGS.items():assert sha(ROOT/rel)==h
 source=pathlib.Path(__file__);sh=sha(source);freeze(ROOT/f'audits/history/rgm_planning_gap_v2_renderer_r1_adversarial_surface-{sh}.py',source.read_bytes())
 pre=ROOT/'audits/execution/planning-gap-v2-renderer-r1-adversarial-surface.preexec.json'
 freeze(pre,jb({'actual_execution_id':EXEC,'stage':'SOURCE_FROZEN_BEFORE_NEW_SURFACE_DIAGNOSTICS','source_sha256':sh,'bindings':BINDINGS,'read_scope':'Fresh stripped input and own frozen predictions/first judgments/solver only','review_mode':'Prior original canonical exposure; same tasks and model, public guided same-task presentation correction. First fresh predictions frozen before diagnostics; no fresh gold/manifest or other reviewer read.','dataset_approval':False}))
 print(canon({'stage':'surface_source_frozen','source_sha256':sh,'preexec_sha256':sha(pre)}),flush=True)
 spec=importlib.util.spec_from_file_location('own_frozen_renderer_input',ROOT/'audits/rgm_planning_gap_v2_renderer_r1_adversarial_input.py');I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I)
 start=time.perf_counter();rows=[json.loads(l) for l in (ROOT/'reviews/input_only/planning-gap-v2-renderer-r1.jsonl').open()];preds={p['sample_id']:p for p in (json.loads(l) for l in (ROOT/'reviews/adversarial/planning-gap-v2-renderer-r1.input_predictions.jsonl').open())}
 features={};mutation_counts=collections.Counter();mutation_examples={};bytasks=collections.defaultdict(list)
 for r in rows:
  t=r['semantic_task'];lead,payload=I.strict_payload(r['rendered_input']);style=r['sample_id'].rsplit('-',1)[1];bytasks[canon(I.normalized_task(t))].append(r)
  f={'kind':t['kind'],'size':len(t.get('nodes',t.get('on',{}))),'plan_length':len(t['plan']),'prompt_length':len(r['rendered_input']),'payload_length':len(r['rendered_input'].split('Task JSON:\n')[1]),'renderer':style,'top_order':tuple(payload),'prompt_tokens':len(r['rendered_input'].split()),'prompt_char_multiset':tuple(sorted(collections.Counter(r['rendered_input']).items()))}
  if t['kind']=='blocks':
   names=sorted(t['on']);goalkeys=list(payload['goal_on']);rank=lambda n:names.index(n) if n in names else -1
   f.update(goal_count=len(goalkeys),goal_key_rank_sequence=tuple(map(rank,goalkeys)),goal_key_names=tuple(goalkeys),goal_first_rank=rank(goalkeys[0]) if goalkeys else -1,goal_last_rank=rank(goalkeys[-1]) if goalkeys else -1,goal_key_name_lengths=tuple(map(len,goalkeys)),goal_value_rank_sequence=tuple(rank(v) for v in payload['goal_on'].values()),goal_TABLE_value_positions=tuple(i for i,v in enumerate(payload['goal_on'].values()) if v=='TABLE'),goal_order_equals_on_relative_order=goalkeys==[b for b in payload['on'] if b in goalkeys],goal_order_strictly_sorted=goalkeys==sorted(goalkeys),goal_key_initials=tuple(k[0] for k in goalkeys))
   # Duplicate nested public block keys must be rejected, even when last value agrees.
   b=goalkeys[0];dup=json.dumps(payload,separators=(',',':'));needle='"goal_on":{';ix=dup.index(needle)+len(needle);dup=dup[:ix]+json.dumps(b)+':'+json.dumps(payload['goal_on'][b])+','+dup[ix:]
   try:I.strict_payload(lead+'Task JSON:\n'+dup)
   except ValueError:mutation_counts['duplicate_nested_goal_key_rejected']+=1
   else:raise AssertionError('duplicate nested goal key accepted')
   changed=copy.deepcopy(payload);changed['goal_on'][b]=True
   try:I.normalized_task(changed)
   except (ValueError,TypeError):mutation_counts['boolean_goal_support_type_rejected']+=1
   else:raise AssertionError('boolean support type accepted')
  features[r['sample_id']]=f
 assert len(bytasks)==204 and all(len(v)==2 for v in bytasks.values())
 # Every renderer pair is visibly distinct and has identical target and character frequency.
 for rs in bytasks.values():
  a,b=rs;assert a['rendered_input']!=b['rendered_input'];assert canon(preds[a['sample_id']]['expected_target'])==canon(preds[b['sample_id']]['expected_target']);assert collections.Counter(a['rendered_input'])==collections.Counter(b['rendered_input'])
  mutation_counts['paired_target_and_character_multiset_invariant']+=1
 # Independently infer controlled semantic pairs; check shared goal order preservation per style.
 units=[next(r for r in rs if r['sample_id'].endswith('-forward')) for rs in bytasks.values()]
 controls,bases,errors=I.derived_family_controls(units,preds);assert not errors
 shared_order=[]
 for c in controls:
  if c['name']!='partial_goal_extension_counterfactual':continue
  weak,strong=[next(r for r in rows if r['sample_id']==sid) for sid in c['sample_ids']]
  for style in ('forward','reverse'):
   wr=next(r for r in rows if r['source_family_key']==weak['source_family_key'] and canon(I.normalized_task(r['semantic_task']))==canon(I.normalized_task(weak['semantic_task'])) and r['sample_id'].endswith('-'+style))
   sr=next(r for r in rows if r['source_family_key']==strong['source_family_key'] and canon(I.normalized_task(r['semantic_task']))==canon(I.normalized_task(strong['semantic_task'])) and r['sample_id'].endswith('-'+style))
   _,wp=I.strict_payload(wr['rendered_input']);_,sp=I.strict_payload(sr['rendered_input']);common=set(wp['goal_on'])&set(sp['goal_on']);wo=[k for k in wp['goal_on'] if k in common];so=[k for k in sp['goal_on'] if k in common];assert wo==so
   shared_order.append({'weak_sample_id':wr['sample_id'],'strong_sample_id':sr['sample_id'],'shared_goal_key_order':wo,'renderer':style})
 assert len(shared_order)==24
 # Count-only lookup tests separately by renderer, using only student-visible surface features.
 keys=[(k,) for k in sorted(set().union(*features.values())) if k!='prompt_char_multiset']+[('kind','goal_count','goal_first_rank','goal_last_rank'),('kind','goal_key_rank_sequence','plan_length'),('kind','goal_value_rank_sequence','goal_count'),('kind','goal_key_name_lengths','payload_length'),('kind','goal_TABLE_value_positions','goal_count','plan_length'),('kind','prompt_length','plan_length'),('kind','goal_key_initials','goal_count')]
 results=[]
 def label(r,name):
  p=preds[r['sample_id']];a=p['expected_target']
  if name in ('valid','reaches_goal'):return a[name]
  if name=='fresh_feasible':return a['shortest_plan'] is not None
  if name=='minimum_moves':return p['search_evidence']['fresh_minimum_moves']
  if name=='first_failure_reason':
   f=p['search_evidence']['first_failure']
   return None if f is None else f['reasons'][0] if r['semantic_task']['kind']=='graph' else p['input_derived_six_guard_trace']['sole_failure_guard'] or 'MULTIPLE'
 for style in ('forward','reverse'):
  selected=[r for r in rows if r['sample_id'].endswith('-'+style)];train=[r for r in selected if r['split']=='train'];dev=[r for r in selected if r['split']=='dev_challenge'];assert len(train)==170 and len(dev)==34
  for name in ('valid','reaches_goal','fresh_feasible','minimum_moves','first_failure_reason'):
   majority=collections.Counter(canon(label(r,name)) for r in train).most_common(1)[0][0]
   results.append({'renderer':style,'label':name,'features':['TRAIN_MAJORITY'],'dev_correct':sum(canon(label(r,name))==majority for r in dev),'dev_count':34})
   for ks in keys:
    feature=lambda r:canon([features[r['sample_id']].get(k,'NOT_APPLICABLE') for k in ks])
    table=collections.defaultdict(collections.Counter)
    for r in train:table[feature(r)][canon(label(r,name))]+=1
    guesses={k:c.most_common(1)[0][0] for k,c in table.items()}
    results.append({'renderer':style,'label':name,'features':ks,'dev_correct':sum(canon(label(r,name))==guesses.get(feature(r),majority) for r in dev),'dev_count':34,'dev_covered':sum(feature(r) in guesses for r in dev),'mixed_train_keys':sum(len(c)>1 for c in table.values()),'train_keys':len(table),'wrong_sample_ids':[r['sample_id'] for r in dev if canon(label(r,name))!=guesses.get(feature(r),majority)][:3]})
 # Prompt length/order changes cannot change character frequencies or correct outputs.
 for rel,h in BINDINGS.items():assert sha(ROOT/rel)==h
 value={'actual_execution_id':EXEC,'completed':True,'source_sha256':sh,'preexec_sha256':sha(pre),'bindings_before_after':BINDINGS,'duration_seconds':time.perf_counter()-start,'fresh_gold_and_manifest_not_read':True,'prior_original_canonical_exposure':True,'same_model_public_guided_same_task_reuse':True,'actual_counts':dict(mutation_counts),'all24_goal_extension_shared_order_checks':shared_order,'semantic_pair_counts':dict(collections.Counter(c['name'] for c in controls)),'surface_feature_rows':[{'sample_id':r['sample_id'],'record_hash':r['record_hash'],'features':features[r['sample_id']]} for r in rows],'per_renderer_train170_to_exposed_dev34_lookups':results,'limits':['All204 tasks/48 families reused; cannot count new prompt bytes as semantic diversity.','Five labels and finite surface/lookup rules measured, not arbitrary learners or untested interactions.','Counts evaluated on tiny exposed dev; prior canonical exposure and common model/public corrective design correlate this review.','Student outcome-driven trace/plan output length is intended semantics; input surface length associations are assessed separately.','Fresh canonical targets/control IDs/property metadata remain unobserved; no dataset/publication approval.']}
 out=ROOT/'audits/shortcut_detection/planning-gap-v2-renderer-r1-surface.json';freeze(out,jb(value));ex=ROOT/'audits/execution/planning-gap-v2-renderer-r1-adversarial-surface.json';freeze(ex,jb({'actual_execution_id':EXEC,'completed':True,'source_sha256':sh,'preexec_sha256':sha(pre),'report_sha256':sha(out),'bindings_before_after':BINDINGS,'actual_counts':dict(mutation_counts),'dataset_approval':False}))
 best={}
 for style in ('forward','reverse'):
  for name in ('valid','reaches_goal','fresh_feasible','minimum_moves','first_failure_reason'):
   relevant=[r for r in results if r['renderer']==style and r['label']==name];best[style+':'+name]=max(relevant,key=lambda r:r['dev_correct'])
 print(canon({'completed':True,'source_sha256':sh,'execution_sha256':sha(ex),'report_sha256':sha(out),'counts':dict(mutation_counts),'best_actual_lookups':best}),flush=True)
if __name__=='__main__':main()
