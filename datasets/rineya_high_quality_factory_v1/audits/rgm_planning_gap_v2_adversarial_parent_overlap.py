#!/usr/bin/env python3
"""Supplement invariant parent overlap; retain frozen first/final canonical verdicts."""
import collections,hashlib,importlib.util,json,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1];EXEC='/root/adversarial_auditor'
BIND={'reviews/input_only/planning-gap-v1.jsonl':'2a6c9d785f9e2de47233de06bad3371e4a4f072e9be1133340913ceec30a6654','data/RGM-001/candidate_renderings/planning-gap-v2.jsonl':'57dfa90e3946fbb657d03c33388e2d67fb3fa9107e4b4de4ffddef0b16289c23','data/RGM-001/manifests/planning-gap-v2.family_controls.json':'9c4497cf61e1d77cea24d808eca965c00e10e975cf235bd7d1aa2667e259da17','reviews/adversarial/planning-gap-v2.first.jsonl':'a1e3867be69eb816d3a82c0c1bacc5903ece38513072d2a18e82755bfa15e882','reviews/adversarial/planning-gap-v2.jsonl':'a1e3867be69eb816d3a82c0c1bacc5903ece38513072d2a18e82755bfa15e882','audits/rgm_planning_gap_v2_adversarial_input.py':'c1f27c9f2aeddfa0b30b77e3e55afa4e96a23d4137df228fccb9b44b7d0778e0'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def freeze(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(x,sort_keys=True,indent=2)+'\n').encode()
 if p.exists():assert p.read_bytes()==b
 else:p.write_bytes(b)
def main():
 for p,h in BIND.items():assert sha(ROOT/p)==h
 source=pathlib.Path(__file__);sh=sha(source);snapshot=ROOT/f'audits/history/rgm_planning_gap_v2_adversarial_parent_overlap-{sh}.py'
 if snapshot.exists():assert snapshot.read_bytes()==source.read_bytes()
 else:snapshot.write_bytes(source.read_bytes())
 pre=ROOT/'audits/execution/planning-gap-v2-adversarial-parent-overlap.preexec.json';freeze(pre,{'actual_execution_id':EXEC,'source_sha256':sh,'bindings':BIND,'stage':'SOURCE_FREEZE_BEFORE_PARENT_INVARIANT_EXECUTION','read_scope':'Authorized parent public inputs/current candidate-manifest and own frozen sources only; no other verdict/private code read. Existing first/final judgments preserved.'})
 spec=importlib.util.spec_from_file_location('own_input',ROOT/'audits/rgm_planning_gap_v2_adversarial_input.py');I=importlib.util.module_from_spec(spec);spec.loader.exec_module(I);P=I.P;started=time.perf_counter()
 old=[json.loads(l) for l in (ROOT/'reviews/input_only/planning-gap-v1.jsonl').read_text().splitlines()];current={r['sample_id']:r for r in [json.loads(l) for l in (ROOT/'data/RGM-001/candidate_renderings/planning-gap-v2.jsonl').read_text().splitlines()]};manifest=json.loads((ROOT/'data/RGM-001/manifests/planning-gap-v2.family_controls.json').read_text())
 old_units={canon(I.normalized_task(r['semantic_task'])):r for r in old};family=collections.defaultdict(list);solved={}
 for key,r in old_units.items():
  answer,info=P.solve(r['semantic_task']);solved[key]=(answer,info)
  if answer['reaches_goal']:family[r['source_family_key']].append(r)
 oldbases={fk:min(rs,key=lambda r:(len(r['semantic_task']['plan']),r['sample_id'])) for fk,rs in family.items()};assert len(oldbases)==48
 comparisons=[];sets=collections.defaultdict(lambda:collections.defaultdict(set))
 for f in manifest['families']:
  fk=f['source_family_key'];new=current[f['base_sample_id']];prior=oldbases[fk];a,b=new['semantic_task'],prior['semantic_task'];assert a['kind']==b['kind'];sigs={}
  if a['kind']=='graph':
   sigs={'weighted_start_goal':(I.graph_signature(a),I.graph_signature(b)),'unweighted_topology':(I.graph_signature(a,False,False),I.graph_signature(b,False,False))}
  else:sigs={'joint_initial_goal':(I.blocks_signature(a),I.blocks_signature(b)),'initial_shape':(I.blocks_signature(a,False),I.blocks_signature(b,False))}
  for name,(ns,os) in sigs.items():sets[name]['current'].add(ns);sets[name]['parent'].add(os)
  comparisons.append({'source_family_key':fk,'current_base_sample_id':new['sample_id'],'parent_base_sample_id':prior['sample_id'],'same_lineage_key':True,'literal_normalized_task_equal':canon(I.normalized_task(a))==canon(I.normalized_task(b)),'invariant_class_comparison':{n:{'current_signature':ns,'parent_signature':os,'same_within_lineage':ns==os} for n,(ns,os) in sigs.items()}})
 for p,h in BIND.items():assert sha(ROOT/p)==h
 report={'actual_execution_id':EXEC,'completed':True,'source_sha256':sh,'preexec_sha256':sha(pre),'bound_inputs_and_first_final_before_after':BIND,'parent_normalized_tasks_independently_solved':len(solved),'same_lineage_family_count':48,'additional_lineage_families':0,'parent_base_inference':'Shortest successful submitted plan per family, independently solved from public input only; no parent targets or verdict read.','invariant_set_overlap':{n:{'current_classes':len(v['current']),'parent_classes':len(v['parent']),'intersection_classes':len(v['current']&v['parent']),'current_absent_from_parent':len(v['current']-v['parent'])} for n,v in sets.items()},'per_lineage_base_comparisons':comparisons,'interpretation':'Earlier canonical audit world-set/semantic-task equality counts are literal normalized payload equality, retaining entity names. Zero literal matches is not evidence that every structural class or mechanism changed. This supplement records invariant overlap honestly; parentV1 is pending revised lineage, while prospective novelty gates compare against prior ACTIVE inputs. No additional-family or primitive grammar claim follows.','common_model_public_guided_own_V1_V2_reuse_disclosed':True,'original_first_final_unchanged':True,'no_release_authority':True,'actual_duration_seconds':time.perf_counter()-started}
 out=ROOT/'audits/diversity/planning-gap-v2-parent-invariant-overlap.json';freeze(out,report);print(canon({'source_sha256':sh,'preexec_sha256':sha(pre),'report_sha256':sha(out),'parent_tasks_solved':len(solved),'invariant_overlap':report['invariant_set_overlap']}))
if __name__=='__main__':main()
