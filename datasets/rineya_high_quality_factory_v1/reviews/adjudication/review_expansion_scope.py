"""Read-only scope/lineage audit of expansion candidates and prior accepted releases.
Produces audit artifacts only, never sample decisions or release bytes.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,collections,re,sys,copy
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reviews/adjudication'
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def dig(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l]
sys.path.insert(0,str(ROOT))
from oracles.independent import solve
tracks=['RCM-001','RGM-001','RVM-001'];allrows=[];details={};sourcekeys={};prior_sourcekeys={};all_sources=[]
for tid in tracks:
 p=ROOT/f'data/{tid}/candidate_renderings/expansion-v1.jsonl';oldp=ROOT/f'data/{tid}/releases/v0001/canonical.jsonl'
 new=read(p);old=read(oldp);all_sources+=[p,oldp];allrows+=new
 families=collections.defaultdict(list)
 for r in new:families[r['semantic_family_id']].append(r)
 assert len(families)==150 and all(len({r['split'] for r in g})==1 and len({r['semantic_parent_id'] for r in g})==1 for g in families.values())
 assert all(dig({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'] for r in new)
 nkeys={r['source_family_key'] for r in new};okeys={r.get('source_family_key') for r in old if r.get('source_family_key')}
 # Old RGM/RVM pilots without source keys have exactly prior indices0..29.
 old_indices={int(re.search(r'-p(\d+)-',r['sample_id']).group(1)) for r in old}
 new_indices={int(re.search(r'-p(\d+)-',r['sample_id']).group(1)) for r in new}
 assert not nkeys&okeys and not old_indices&new_indices
 assert new_indices==set(range(30,180))
 newtasks={dig(r['semantic_task']) for r in new};oldtasks={dig(r['semantic_task']) for r in old}
 assert not newtasks&oldtasks
 def stripped(task):return {k:v for k,v in task.items() if k!='annotation'}
 assert not {dig(stripped(r['semantic_task'])) for r in new}&{dig(stripped(r['semantic_task'])) for r in old}
 representative=[g[0] for g in families.values()]
 d=dict(records=len(new),independent_source_families=150,prior_release_records=len(old),new_source_indices=[30,179],source_key_overlap=0,prior_source_index_overlap=0,exact_task_overlap=0,annotation_stripped_exact_overlap=0,split_counts=dict(collections.Counter(r['split'] for r in new)),structural_difficulty=dict(collections.Counter(str(r['structural_difficulty']) for r in new)),skill_tag_family_counts=dict(collections.Counter(tag for r in representative for tag in r['skill_tags'])),challenge_axis_family_counts=dict(collections.Counter(tag for r in representative for tag in r['challenge_axes'])),all_family_variants_same_parent_and_split=True,challenge_claim='Exposed development axes; no sealed evaluation or broad structural-template holdout.',source_key_caveat='Family IDs certify generator lineage;150 source worlds do not imply150 distinct mechanism classes.')
 if tid=='RCM-001':
  dev=[r for r in new if r['split']=='dev_challenge' and r['target_format']=='explicit_trace' and r['renderer']=='forward']
  counts=collections.Counter();sets=collections.defaultdict(set)
  for r in dev:
   t=r['semantic_task'];root=t['rule'];wrapper=root if root['op']=='all' else root['args'][0]
   assert wrapper['op']=='all' and len(wrapper['args'])==2
   added=wrapper['args'][1];assert added['op']=='any' and added['args'][0]['op']=='gt'
   comparison=added['args'][0];assert comparison['args'][0]=={'var':'event.amount'}
   boundary=comparison['args'][1]['const']
   key='all_amount_true' if all(e['amount']>boundary for e in t['events']) else 'amount_varies'
   counts[key]+=1;sets[key].add(r['semantic_family_id'])
   stripped_task=copy.deepcopy(t)
   stripped_task['rule']=wrapper['args'][0] if root['op']=='all' else {'op':'not','args':[wrapper['args'][0]]}
   actual=solve(t);without=solve(stripped_task)
   if canon(actual['after'])!=canon(without['after']):counts['final_changed']+=1;sets['final_changed'].add(r['semantic_family_id'])
   if [x['cmp']['rule'] for x in actual['steps']]!=[x['cmp']['rule'] for x in without['steps']]:counts['rule_trace_changed']+=1;sets['rule_trace_changed'].add(r['semantic_family_id'])
  d['nested_dev_wrapper']=dict(dev_twin_tasks=len(dev),task_counts=dict(counts),affected_family_counts={k:len(v) for k,v in sets.items()},conclusion='Nested wrapper is meaningful in a subset:14 families have varying added amount gates;16 have always-true added gates and serve as redundant-shape controls. 12 families change final state under wrapper removal. This is a bounded nested-control challenge, not broad predicate discovery.')
 if tid=='RGM-001':
  graphs=[r for r in representative if r['semantic_task']['kind']=='graph'];blocks=[r for r in representative if r['semantic_task']['kind']=='blocks']
  d['graph_family_scope']=dict(graph_families=len(graphs),actual_outdegree_branch_families=sum(any(sum(e[0]==n for e in r['semantic_task']['edges'])>1 for n in r['semantic_task']['nodes']) for r in graphs),node_count_distribution=dict(collections.Counter(str(len(r['semantic_task']['nodes'])) for r in graphs)),limitation='Branched_resource_graph labels name the generator family; not every generated world has an actual branch.')
  d['block_family_scope']=dict(block_families=len(blocks),initial_root_counts=dict(collections.Counter(str(sum(v=='TABLE' for v in r['semantic_task']['on'].values())) for r in blocks)),goal_types=dict(collections.Counter('full' if len(r['semantic_task']['goal_on'])==len(r['semantic_task']['on']) else 'partial' for r in blocks)),limitation='Multi-stack task family includes initially single-stack and trivial/satisfied goal controls; these are not all equally compositional.')
  dg=[r for r in new if r['split']=='dev_challenge']
  d['development_goal_labels']=dict(collections.Counter(str(r['verified_target']['reaches_goal']) for r in dg))
  assert set(d['development_goal_labels'])=={'True','False'}
 if tid=='RVM-001':
  d['critic_goal_scope']='Same explicit validity/progress/usefulness ranking objective and recurring corruption families. Branched worlds vary candidate behavior, but no held-out goal functions or error-producer family claim is supported.'
 details[tid]=d
scope=dict(actor='/root/research_architect',role='Quality Adjudicator: preliminary scope/lineage inspection only',created_at_utc=datetime.now(timezone.utc).isoformat(),batch='expansion-v1',records=len(allrows),independent_source_families=450,prior_release_lineage='v0001 current accepted records, not superseded candidate revisions',tracks=details,inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in all_sources},approval_status='PENDING until both independent final reviewer receipts are complete; this scope inspection issues no sample approvals.',producer_or_release_writes=False,protected_evaluations_opened=False)
(OUT/'expansion-v1_scope_review.json').write_text(json.dumps(scope,indent=2)+'\n')
report='''# Preliminary expansion-v1 scope and lineage review

Execution `/root/research_architect`; this document is a scope audit and issues no acceptance verdicts. Expert and adversarial final receipts are required before adjudication. All producer/release bytes remain unchanged.

Expansion contains1,860 renderings across450 fresh source families: RCM1,200/150, RGM360/150 and RVM300/150. Source indices30–179 do not overlap accepted pilots0–29. Exact semantic-task hashes, including hashes with irrelevant causal annotation removed, do not match current accepted v0001 tasks. Every new family retains one parent ID and split. Repeated twins/format/rendering or matched-plan variants are not counted as independent families.

The150 source families per track represent distinct generated world instances, not150 independent mechanisms. Causal data retains six base predicate classes; relational and critic worlds share known executable primitives and goal/scoring rules. Numerical/name/shape challenges remain exposed development data. This factory creates no sealed generalization instrument; a generic structural-template holdout is explicitly false in the current leakage audit.

Causal development has30 held-out nested AST-wrapper families. In16 families, the added amount gate is always true on supplied events, leaving the base predicate unchanged; these are redundant-shape controls. In14 families, the extra gate varies. Removing the wrapper for a diagnostic oracle comparison changes rule traces in25 of60 twin tasks (14 families) and final states in21 of60 (12 families). Meaningful nested control is therefore present in a subset and must be counted separately from redundant controls. No Level5–7 label or broad causal discovery is claimed. Later status words may overlap training; only first-status vocabulary is held out.

Relational expansion has90 graph families;77 contain an actual outdegree branch and13 preserve simpler controls. Its60 block families have40 full and20 partial goal maps. Initial stack roots distribute as26 two-root,16 one-root,14 three-root,3 four-root and1 five-root worlds. Generator-family skill tags do not certify equal compositional complexity in every sample. Development goal labels include both outcomes (60true/60false renderings), addressing the former collapsed development label. Fresh shortest-plan horizons and fuel/partial-goal cases are bounded exact tasks.

Critic expansion changes graphs and candidate outcomes while preserving the explicit ranking objective and recurring positive claim offsets/self-loop/repeated-node error scaffold. This is larger/name-shifted candidate evaluation, not held-out reward functions, unseen corruption producers or successor-feature training.

Prior exact-record review receipts cannot approve any new expansion hash. Machine oracle/schema/duplicate passes are prerequisites, not substitutes for independent expert/adversarial finals. The final adjudicator will preserve disagreements and unresolved serious issues per record.
'''
(OUT/'EXPANSION_V1_SCOPE_REVIEW.md').write_text(report)
print(json.dumps({'records':len(allrows),'families':450,'prior_source_index_overlap':0,'prior_exact_task_overlap':0,'all_family_splits_valid':True,'nested_wrapper_scope_limit':'14 meaningful nested-control families;16 redundant-shape families;12 families change final state under wrapper removal'},indent=2))
