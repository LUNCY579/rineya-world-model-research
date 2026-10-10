"""Public hash/count and abstract hash-dependency checks only; no RSM interpreter."""
from pathlib import Path
import datetime,hashlib,json
D=Path(__file__).resolve().parent;BASE=D.parents[2]
def bind(p):
 b=(BASE/p).read_bytes();return {'path':p,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
def cyclic(edges):
 seen=set();active=set()
 def visit(v):
  if v in active:return True
  if v in seen:return False
  active.add(v)
  if any(visit(w) for w in edges.get(v,[])):return True
  active.remove(v);seen.add(v);return False
 return any(visit(v) for v in edges)
def main():
 pre=json.loads((D/'preexecution.json').read_text())
 for b in pre['bindings']:assert bind(b['path'])==b
 rf=json.loads((BASE/'research/curricula/RSM_OPTION_METADATA_V1_AMENDMENT_R1_ROOT_FREEZE.json').read_text())
 for b in rf['bindings']:assert bind(b['path'])['sha256']==b['sha256'] and bind(b['path'])['bytes']==b['bytes']
 fixture=json.loads((D/'hand-hash-dependencies.json').read_text());results=[]
 for c in fixture['cases']:
  actual=cyclic(c['edges']);assert actual==c['expected_cycle'];results.append({'case_id':c['case_id'],'cycle':actual,'pass':True})
 cfg=json.loads((BASE/'research/curricula/RSM-OPTION-RESUMPTION-CONFIG-20261010.json').read_text());n=sum(map(len,cfg['source_family_indices_by_stratum'].values()))
 assert n==12 and len(cfg['required_contexts'])==9 and len(cfg['renderers'])==2 and n*(8*2+9)==300 and n*9*2*9==1944
 result={'status':'COMPLETED_PUBLIC_AMENDMENT_REVIEW_ONLY','actual_author_execution_id':'/root/research_architect','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'conclusion':'REVISE_OPTIONAL_CANONICAL_MEMBER_HASH_INTERFACE_BEFORE_PRODUCTION','mandatory_semantics_counts_splits_inconsistent':False,'earlier_critique_substantially_resolved':True,'remaining_issue':'Optional canonical pair_record_hashes can contain owner full hash and create content-addressing cycles; prefer external manifest member hashes.','hand_abstract_dependency_checks':results,'hand_scope':'abstract metadata hash dependency graph only; no candidate, full record hash fixture, task interpreter or solver','source_counts_are_prospective':True,'data_approval':False,'private_code_or_future_reviews_read':False,'bindings_before':pre['bindings'],'bindings_after':[bind(b['path']) for b in pre['bindings']],'preexecution':bind(str((D/'preexecution.json').relative_to(BASE))),'design_adjudicator_overlap':True,'same_model_common_guidance_risk':True}
 assert result['bindings_before']==result['bindings_after']
 with (D/'execution.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps({'execution':bind(str((D/'execution.json').relative_to(BASE))),'hand_checks':len(results),'data_approval':False,'conclusion':result['conclusion']}))
if __name__=='__main__':main()
