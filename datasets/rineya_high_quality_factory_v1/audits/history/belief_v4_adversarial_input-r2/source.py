#!/usr/bin/env python3
"""Actual independent V4 input-only reexecution; /root/adversarial_auditor.
Own prospective typed set/bitmask solver and hand toys are reused explicitly.
No actual V4 candidate/gold/control manifest or other reviewer implementation access.
"""
import collections,copy,hashlib,itertools,json,pathlib,re,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
ACTOR='/root/adversarial_auditor'
def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
def digest(x):return hashlib.sha256(canon(x).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(x,m):
 if not x:raise ValueError(m)
def freeze(p,raw):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():require(p.read_bytes()==raw,'immutable output differs '+str(p))
 else:p.write_bytes(raw)
def jsonbytes(x):return (json.dumps(x,sort_keys=True,indent=2,ensure_ascii=False)+'\n').encode()
def jlbytes(xs):return ''.join(canon(x)+'\n' for x in xs).encode()
def scalar(v): return v is None or type(v) in (str,int,bool)
def typed(v): require(scalar(v),'Invalid JSON scalar'); return canon(v)
def payload_count(t,ids): return len({canon(w['facts']) for w in t['worlds'] if w['world_id'] in ids})
def partition(t,ids,f):
    counts=collections.Counter(typed(w['facts'][f]) for w in t['worlds'] if w['world_id'] in ids)
    return [{'value':json.loads(v),'count':counts[v]} for v in sorted(counts)]
def answer(q,bins,n):
    entailed=bool(n) and len(bins)==1
    return {'query_id':q['query_id'],'counts':bins,'distinct_values':len(bins),'entailed':entailed,'guaranteed':bins[0]['value'] if entailed else None,'can_reduce_uncertainty':len(bins)>1}

def own_prior_bitmask_solve(t):
    # Own third implementation uses typed equality keys and a prefix bitmask over world positions.
    require(type(t) is dict and set(t)=={'kind','worlds','observations','queries'} and t['kind']=='belief','Belief shape')
    worlds,obs,queries=t['worlds'],t['observations'],t['queries']
    require(all(type(x) is list for x in (worlds,obs,queries)),'Lists')
    require(all(type(w) is dict and set(w)=={'world_id','facts'} and type(w['world_id']) is str and type(w['facts']) is dict for w in worlds),'World shape')
    fields=set(worlds[0]['facts']) if worlds else set();require(len({w['world_id'] for w in worlds})==len(worlds),'Unique worlds')
    require(all(set(w['facts'])==fields and all(type(f) is str and scalar(v) for f,v in w['facts'].items()) for w in worlds),'Typed consistent facts')
    require(all(type(o) is dict and set(o)=={'fact','equals'} and type(o['fact']) is str and o['fact'] in fields and scalar(o['equals']) for o in obs),'Observations')
    require(all(type(q) is dict and set(q)=={'query_id','fact'} and type(q['query_id']) is str and q['fact'] in fields for q in queries),'Queries');require(len({q['query_id'] for q in queries})==len(queries),'Unique queries')
    def equal(v,u):return (type(v),v)==(type(u),u)
    mask=(1<<len(worlds))-1
    def ids(mask):return sorted(w['world_id'] for j,w in enumerate(worlds) if mask&(1<<j))
    prefixes=[ids(mask)];individual=[]
    for o in obs:
        support=sum(1<<j for j,w in enumerate(worlds) if equal(w['facts'][o['fact']],o['equals']))
        individual.append(ids(support));mask&=support;prefixes.append(ids(mask))
    def count_at(mask,f):
        bins=collections.Counter(canon(w['facts'][f]) for j,w in enumerate(worlds) if mask&(1<<j))
        return [{'value':json.loads(k),'count':bins[k]} for k in sorted(bins)]
    final=ids(mask);outputs=[]
    for q in queries:
        bins=count_at(mask,q['fact']);ent=bool(mask) and len(bins)==1
        outputs.append({'query_id':q['query_id'],'counts':bins,'distinct_values':len(bins),'entailed':ent,'guaranteed':bins[0]['value'] if ent else None,'can_reduce_uncertainty':len(bins)>1})
    result={'surviving_worlds':final,'belief_sizes':[len(x) for x in prefixes[1:]],'queries':outputs}
    detail={'prefix_survivors':prefixes,'individual_observation_supports':individual,'prefix_query_partitions':[[{'query_id':q['query_id'],'fact':q['fact'],'bins':partition(t,p,q['fact'])} for q in queries] for p in prefixes],'world_count':len(worlds),'fact_count':len(fields),'observation_count':len(obs),'query_count':len(queries),'distinct_complete_world_payloads':payload_count(t,prefixes[0])}
    return result,detail

def toys():
    cases=[]
    def probe(name,worlds,obs,qs,expected):
        t={'kind':'belief','worlds':[{'world_id':i,'facts':f} for i,f in worlds],'observations':obs,'queries':qs}
        actual,detail=solve(t);require(canon(actual)==canon(expected),'Toy '+name);cases.append({'name':name,'task':t,'expected_target':expected,'actual_target':actual,'prefix_survivors':detail['prefix_survivors'],'passed':True})
    q=[{'query_id':'q','fact':'v'}]
    probe('boolean_true_is_not_integer_one',[('a',{'x':True,'v':'yes'}),('b',{'x':1,'v':'no'})],[{'fact':'x','equals':True}],q,{'surviving_worlds':['a'],'belief_sizes':[1],'queries':[{'query_id':'q','counts':[{'value':'yes','count':1}],'distinct_values':1,'entailed':True,'guaranteed':'yes','can_reduce_uncertainty':False}]})
    probe('boolean_false_is_not_integer_zero',[('a',{'x':False,'v':False}),('b',{'x':0,'v':0})],[{'fact':'x','equals':0}],q,{'surviving_worlds':['b'],'belief_sizes':[1],'queries':[{'query_id':'q','counts':[{'value':0,'count':1}],'distinct_values':1,'entailed':True,'guaranteed':0,'can_reduce_uncertainty':False}]})
    probe('typed_counts_multiplicity',[('a',{'v':True}),('b',{'v':1}),('c',{'v':1}),('d',{'v':None})],[],q,{'surviving_worlds':['a','b','c','d'],'belief_sizes':[],'queries':[{'query_id':'q','counts':[{'value':1,'count':2},{'value':None,'count':1},{'value':True,'count':1}],'distinct_values':3,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':True}]})
    probe('entailed_null_multiworld',[('a',{'x':0,'v':None}),('b',{'x':1,'v':None})],[],q,{'surviving_worlds':['a','b'],'belief_sizes':[],'queries':[{'query_id':'q','counts':[{'value':None,'count':2}],'distinct_values':1,'entailed':True,'guaranteed':None,'can_reduce_uncertainty':False}]})
    probe('empty_absorbs_individually_supported_repeat',[('a',{'x':'stone','y':'silk','v':1}),('b',{'x':'reed','y':'linen','v':2})],[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'},{'fact':'x','equals':'stone'}],q,{'surviving_worlds':[],'belief_sizes':[1,0,0],'queries':[{'query_id':'q','counts':[],'distinct_values':0,'entailed':False,'guaranteed':None,'can_reduce_uncertainty':False}]})
    probe('stone_and_linen_supported_not_empty',[('a',{'x':'stone','y':'linen','v':'unknown'}),('b',{'x':'reed','y':'silk','v':None})],[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'}],q,{'surviving_worlds':['a'],'belief_sizes':[1,1],'queries':[{'query_id':'q','counts':[{'value':'unknown','count':1}],'distinct_values':1,'entailed':True,'guaranteed':'unknown','can_reduce_uncertainty':False}]})
    probe('reorder_changes_trace_preserves_final',[('a',{'x':'r','y':'s','v':0}),('b',{'x':'r','y':'t','v':1}),('c',{'x':'q','y':'t','v':2})],[{'fact':'y','equals':'t'},{'fact':'x','equals':'q'}],q,{'surviving_worlds':['c'],'belief_sizes':[2,1],'queries':[{'query_id':'q','counts':[{'value':2,'count':1}],'distinct_values':1,'entailed':True,'guaranteed':2,'can_reduce_uncertainty':False}]})
    bads=[]
    for name,t in [('float_scalar',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1.0}}],'observations':[],'queries':[]}),('duplicate_world',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1}},{'world_id':'a','facts':{'x':2}}],'observations':[],'queries':[]}),('missing_fact',{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1}}],'observations':[{'fact':'y','equals':1}],'queries':[]})]:
        try:solve(t)
        except Exception as e:bads.append({'name':name,'task':t,'rejected':True,'error':str(e)})
        else:raise ValueError('Malformed probe accepted '+name)
    return {'manually_authored_exact_toys':cases,'malformed_probes':bads}

def solve(t):
 # New direct set filtering independently agrees with our prior positional bitmask implementation.
 require(type(t) is dict and set(t)=={'kind','worlds','observations','queries'} and t['kind']=='belief','typed belief shape')
 worlds,obs,qs=t['worlds'],t['observations'],t['queries'];require(all(type(x) is list for x in (worlds,obs,qs)),'typed lists')
 require(all(type(w) is dict and set(w)=={'world_id','facts'} and type(w['world_id']) is str and type(w['facts']) is dict for w in worlds),'world records')
 facts=set(worlds[0]['facts']) if worlds else set();require(len({w['world_id'] for w in worlds})==len(worlds),'distinct world IDs')
 require(all(set(w['facts'])==facts and all(type(k) is str and scalar(v) for k,v in w['facts'].items()) for w in worlds),'typed fact maps')
 require(all(type(o) is dict and set(o)=={'fact','equals'} and type(o['fact']) is str and o['fact'] in facts and scalar(o['equals']) for o in obs),'typed observations')
 require(all(type(q) is dict and set(q)=={'query_id','fact'} and type(q['query_id']) is str and type(q['fact']) is str and q['fact'] in facts for q in qs),'typed queries');require(len({q['query_id'] for q in qs})==len(qs),'distinct query IDs')
 current={w['world_id'] for w in worlds};prefixes=[sorted(current)];individual=[]
 for o in obs:
  support={w['world_id'] for w in worlds if typed(w['facts'][o['fact']])==typed(o['equals'])};individual.append(sorted(support));current&=support;prefixes.append(sorted(current))
 result={'surviving_worlds':sorted(current),'belief_sizes':[len(p) for p in prefixes[1:]],'queries':[answer(q,partition(t,current,q['fact']),len(current)) for q in qs]}
 return result,{'prefix_survivors':prefixes,'individual_observation_supports':individual}

DISCLOSURE='Fresh input-only V4 execution with original V1/V2/V3 canonical exposure, own original solver/prediction/toy reuse, same model/common public design and root-guided metadata correction. The allowed public clarification contains an historical verdict summary. This execution does not inspect fresh gold, actual controls, producer/private/formal implementations or another reviewer. Reused semantics are not unseen/blind evaluation.'
FIELDS={'record_hash','rendered_input','sample_id','semantic_family_id','semantic_parent_id','semantic_task','split','track_id'}
VIEW='reviews/input_only/belief-gap-v4.jsonl'
OLD='reviews/input_only/belief-gap-v3.jsonl'
OLDP='reviews/adversarial/belief-gap-v3.input_predictions.jsonl'
PREEXEC='audits/execution/belief-gap-v4-input-only-adversarial-r2.preexec.json'
PREFIX='reviews/adversarial/belief-gap-v4.input_'

def duplicate_free(pairs):
 d={}
 for k,v in pairs:
  require(k not in d,'duplicate JSON object key '+k);d[k]=v
 return d
def loadjson(raw):return json.loads(raw,object_pairs_hook=duplicate_free)
def rows(path):return [loadjson(line) for line in path.read_bytes().splitlines() if line]
def reverse_fields(x):
 if type(x) is dict:return {k:reverse_fields(v) for k,v in reversed(list(x.items()))}
 if type(x) is list:return [reverse_fields(v) for v in x]
 return x
def input_key(r):return digest({k:r[k] for k in ('semantic_task','rendered_input','split','track_id')})
def parse_prompt(r):
 marker='Task JSON:\n';p=r['rendered_input'];require(type(p) is str and p.count(marker)==1,'one visible task JSON marker')
 lead,raw=p.split(marker);t=loadjson(raw);require(canon(t)==canon(r['semantic_task']),'typed prompt/payload binding')
 require(all(s in lead for s in ('type-sensitive','true differs from integer 1','Empty beliefs remain empty','Query rows follow input order','distinct_values>1')),'visible semantic contract sufficiency')
 return t,lead,tuple(t)
def obs_reorder_expected(t):
 m=copy.deepcopy(t);m['observations'].reverse();return solve(m)[0]

def metamorphics(t,a,d):
 checks=[]
 def equal(name,m,expected=a):
  got,detail=solve(m);require(canon(got)==canon(expected),name);checks.append(name);return detail
 equal('object_order',reverse_fields(t))
 m=copy.deepcopy(t);m['worlds'].reverse();equal('world_list_order',m)
 m=copy.deepcopy(t);mp={w['world_id']:'renamed_'+str(i) for i,w in enumerate(t['worlds'])}
 for w in m['worlds']:w['world_id']=mp[w['world_id']]
 e=copy.deepcopy(a);e['surviving_worlds']=sorted(mp[i] for i in a['surviving_worlds']);equal('world_ID_equivariance',m,e)
 m=copy.deepcopy(t);m['queries'].reverse();e=copy.deepcopy(a);e['queries'].reverse();equal('query_order',m,e)
 m=copy.deepcopy(t);e=copy.deepcopy(a)
 for i,(q,o) in enumerate(zip(m['queries'],e['queries'])):q['query_id']='query_'+str(i);o['query_id']=q['query_id']
 equal('query_ID_equivariance',m,e)
 mp={f:'fresh_fact_'+str(i) for i,f in enumerate(t['worlds'][0]['facts'])} if t['worlds'] else {}
 m=copy.deepcopy(t)
 for w in m['worlds']:w['facts']={mp[k]:v for k,v in w['facts'].items()}
 for o in m['observations']:o['fact']=mp[o['fact']]
 for q in m['queries']:q['fact']=mp[q['fact']]
 equal('consistent_fact_rename',m)
 m=copy.deepcopy(t);m['observations'].reverse();aa,dd=solve(m);bb,bd=own_prior_bitmask_solve(m)
 require(canon(aa)==canon(bb),'reordered trace independent recomputation');require(aa['surviving_worlds']==a['surviving_worlds'] and canon(aa['queries'])==canon(a['queries']),'observation commutation final support')
 checks.append('observation_reverse_final_and_recomputed_prefix')
 m=copy.deepcopy(t)
 if m['observations']:
  m['observations'].append(copy.deepcopy(m['observations'][-1]));e=copy.deepcopy(a);e['belief_sizes'].append(len(a['surviving_worlds']));equal('repeat_idempotence_and_empty_absorption',m,e)
 return checks

def scalar_features(t,d,prompt):
 obs=t['observations'];values=[o['equals'] for o in obs];scalar_types=[type(v).__name__ for v in values]
 queryfacts={q['fact'] for q in t['queries']};ys={o['fact'] for o in obs if o['fact'] not in queryfacts}
 y=next(iter(ys)) if len(ys)==1 else None
 yfreq=collections.Counter(typed(w['facts'][y]) for w in t['worlds']) if y else collections.Counter()
 observedfreq=[sum(typed(w['facts'][o['fact']])==typed(o['equals']) for w in t['worlds']) for o in obs]
 tablefreq=sorted(sorted(collections.Counter(typed(w['facts'][f]) for w in t['worlds']).values()) for f in t['worlds'][0]['facts'])
 return {'world_count':len(t['worlds']),'fact_count':len(t['worlds'][0]['facts']),'observation_count':len(obs),'query_count':len(t['queries']),'prompt_length':len(prompt),'prompt_length_10char_bucket':len(prompt)//10,
 'observation_scalar_sequence':canon(values),'observation_scalar_multiset':canon(sorted(typed(v) for v in values)),'observation_scalar_types':canon(scalar_types),'observation_scalar_lengths':canon([len(typed(v)) for v in values]),
 'first_scalar':typed(values[0]) if values else None,'second_scalar':typed(values[1]) if len(values)>1 else None,'first_prefix_size':len(d['prefix_survivors'][1]) if obs else len(t['worlds']),
 'individual_support_sizes':canon(observedfreq),'table_column_count_multiset':canon(tablefreq),'auxiliary_table_frequency':canon(sorted(yfreq.items())),
 'observation_name_sequence':canon([o['fact'] for o in obs]),'obs_auxiliary_frequency_joint':canon([values,sorted(yfreq.items())]),'coarse_first_prefix_joint':canon([len(t['worlds']),len(t['worlds'][0]['facts']),len(obs),len(t['queries']),len(d['prefix_survivors'][1]) if obs else len(t['worlds'])])}
def lookup_measure(examples,feature,label):
 hist=collections.defaultdict(collections.Counter)
 for x in examples:hist[canon(x['features'][feature])][canon(x[label])]+=1
 fitted=sum(max(c.values()) for c in hist.values());ambiguous=sum(len(c)>1 for c in hist.values())
 train=collections.defaultdict(collections.Counter);tm=collections.Counter()
 for x in examples:
  if x['split']=='train':train[canon(x['features'][feature])][canon(x[label])]+=1;tm[canon(x[label])]+=1
 majority=sorted(tm,key=lambda k:(-tm[k],k))[0];correct=unseen=devn=0
 for x in examples:
  if x['split']!='dev_challenge':continue
  devn+=1;c=train.get(canon(x['features'][feature]));unseen+=not bool(c);pred=sorted(c,key=lambda k:(-c[k],k))[0] if c else majority;correct+=pred==canon(x[label])
 return {'feature':feature,'label':label,'records':len(examples),'fitted_correct':fitted,'fitted_accuracy':fitted/len(examples),'feature_buckets':len(hist),'mixed_label_buckets':ambiguous,'train_lookup_exposed_dev_correct':correct,'exposed_dev_records':devn,'dev_unseen_feature_fallbacks':unseen,'interpretation':'CPU finite fitted/train-table lookup; no learned model, heldout generalization or full-output claim'}
def minimal_counterexamples():
 # Fresh hand-derived attacks on scalar-only, length/count, Python equality, and null entailment rules.
 cases=[]
 def add(name,t,shortcut,expected):
  a,d=solve(t);require(canon(a)==canon(expected),'fresh hand expected '+name);cases.append({'name':name,'typed_input':t,'own_exact_target':a,'incorrect_shortcut_prediction':shortcut,'prefix_supports':d['prefix_survivors']})
 qs=[{'query_id':'q','fact':'v'}]
 t={'kind':'belief','worlds':[{'world_id':'a','facts':{'x':'stone','y':'linen','v':None}},{'world_id':'b','facts':{'x':'reed','y':'silk','v':True}}],'observations':[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'}],'queries':qs}
 add('stone_AND_linen_does_not_imply_empty',t,{'empty':True},{'surviving_worlds':['a'],'belief_sizes':[1,1],'queries':[{'query_id':'q','counts':[{'value':None,'count':1}],'distinct_values':1,'entailed':True,'guaranteed':None,'can_reduce_uncertainty':False}]})
 u=copy.deepcopy(t);u['worlds'][0]['facts']['y']='silk';u['worlds'][1]['facts']['y']='linen';a,d=solve(u);cases.append({'name':'same_observations_counts_and_firstprefix_opposite_status','typed_input_pair':[t,u],'own_targets':[solve(t)[0],a],'same_observations':True,'same_first_prefix_support':solve(t)[1]['prefix_survivors'][1]==d['prefix_survivors'][1]})
 t={'kind':'belief','worlds':[{'world_id':'a','facts':{'x':True,'v':None}},{'world_id':'b','facts':{'x':1,'v':'integer'}}],'observations':[{'fact':'x','equals':1}],'queries':qs}
 add('Python_bool_int_equality_is_incorrect',t,{'surviving_worlds':['a','b']},{'surviving_worlds':['b'],'belief_sizes':[1],'queries':[{'query_id':'q','counts':[{'value':'integer','count':1}],'distinct_values':1,'entailed':True,'guaranteed':'integer','can_reduce_uncertainty':False}]})
 # Deliberate prompt mutation must be caught by actual typed parser/binding.
 negatives=[]
 for name,raw,t in [('duplicate_outer_key','{"kind":"belief","kind":"belief","worlds":[],"observations":[],"queries":[]}',{'kind':'belief','worlds':[],'observations':[],'queries':[]}),('typed_bool_int_prompt_mismatch',canon({'kind':'belief','worlds':[{'world_id':'a','facts':{'x':True}}],'observations':[],'queries':[]}),{'kind':'belief','worlds':[{'world_id':'a','facts':{'x':1}}],'observations':[],'queries':[]})]:
  r={'rendered_input':'type-sensitive true differs from integer 1 Empty beliefs remain empty Query rows follow input order distinct_values>1\nTask JSON:\n'+raw,'semantic_task':t}
  try:parse_prompt(r)
  except Exception as e:negatives.append({'name':name,'raw_prompt':r['rendered_input'],'typed_payload':t,'actual_rejection':str(e)})
  else:raise ValueError('prompt corruption passed '+name)
 return {'fresh_minimal_semantic_counterexamples':cases,'actual_prompt_mutations_rejected':negatives}

def cohort_analysis(fresh,targets,details,styles):
 byfamily=collections.defaultdict(list)
 for r in fresh:byfamily[r['semantic_family_id']].append(r)
 normalized={};conditional=collections.defaultdict(list);roles={};family_tables={};crossing_receipts=[]
 for fam,rs in byfamily.items():
  base=rs[0]['semantic_task'];qcols={q['fact'] for q in base['queries']}
  def vector(t,f):return canon(sorted((w['world_id'],typed(w['facts'][f])) for w in t['worlds']))
  basevec={vector(base,f):'queried_column_'+str(i) for i,f in enumerate(sorted(qcols))};require(len(basevec)==3,'unique actual query-column vectors')
  table_set=set()
  for r in rs:
   t=r['semantic_task'];mp={q['fact']:basevec[vector(t,q['fact'])] for q in t['queries']};require(len(mp)==3,'coherent input-derived query-column rename')
   nt=copy.deepcopy(t)
   for w in nt['worlds']:w['facts']={mp.get(f,f):v for f,v in w['facts'].items()}
   for o in nt['observations']:o['fact']=mp.get(o['fact'],o['fact'])
   for q in nt['queries']:q['fact']=mp[q['fact']]
   normalized[r['sample_id']]=nt;table_key=digest(nt['worlds']);table_set.add(table_key)
   case=digest({'worlds':nt['worlds'],'observations':nt['observations']});conditional[(fam,case,styles[r['sample_id']])].append(r)
   roles[r['sample_id']]={q['query_id']:mp[q['fact']] for q in t['queries']}
  family_tables[fam]=len(table_set)
 for (fam,case,style),rs in conditional.items():
  require(len(rs)==27,'actual conditional 27 rows')
  rec={'source_family_alias':fam,'input_derived_case_sha256':case,'observed_object_style':style,'rows':27,'query_column_crossings':[]}
  for role in ('queried_column_0','queried_column_1','queried_column_2'):
   triples=collections.Counter();outcomes=collections.Counter()
   for r in rs:
    for i,q in enumerate(r['semantic_task']['queries']):
     if roles[r['sample_id']][q['query_id']]==role:
      triples[(q['query_id'],i,q['fact'])]+=1;ans=targets[r['sample_id']]['queries'][i];outcomes[canon({k:v for k,v in ans.items() if k!='query_id'})]+=1
   require(len(triples)==27 and set(triples.values())=={1},'input-derived complete actual ID/position/name product');require(len(outcomes)==1,'same semantic outcome across nuisance triple')
   rec['query_column_crossings'].append({'column':role,'unique_triples':27,'triple_multiplicity':1,'typed_outcome_histogram':dict(outcomes)})
  crossing_receipts.append(rec)
 taskgroups=collections.defaultdict(list)
 for r in fresh:taskgroups[digest(r['semantic_task'])].append(r)
 fieldpairs=[]
 for taskhash,rs in taskgroups.items():
  require(len(rs)==2 and len({styles[r['sample_id']] for r in rs})==2,'exact typed-task object-order pair')
  require(canon(targets[rs[0]['sample_id']])==canon(targets[rs[1]['sample_id']]),'field pair exact expected outputs')
  fieldpairs.append({'sample_ids':[r['sample_id'] for r in rs],'typed_task_sha256':taskhash})
 # Reconstruct factual scalar pairs without scalar/context/schedule annotations or ID decoding.
 scalarpairs={};oppositepairs={};tasklookup={(r['semantic_family_id'],styles[r['sample_id']],digest(r['semantic_task'])):r for r in fresh}
 for r in fresh:
  t=r['semantic_task'];qfacts={q['fact'] for q in t['queries']};ys={o['fact'] for o in t['observations'] if o['fact'] not in qfacts};require(len(ys)==1,'one actual unqueried observation column');y=next(iter(ys))
  candidates=[]
  for v in sorted({w['facts'][y] for w in t['worlds'] if type(w['facts'][y]) is str}):
   if v=='linen':continue
   u=copy.deepcopy(t)
   def swap(x):return v if x=='linen' else 'linen' if x==v else x
   for w in u['worlds']:w['facts'][y]=swap(w['facts'][y])
   for o in u['observations']:
    if o['fact']==y:o['equals']=swap(o['equals'])
   partner=tasklookup.get((r['semantic_family_id'],styles[r['sample_id']],digest(u)))
   if partner:candidates.append((v,partner))
  require(len(candidates)==1,'actual nontrivial fact-local scalar partner uniquely resolves');value,p=candidates[0]
  require(canon(targets[r['sample_id']])==canon(targets[p['sample_id']]) and details[r['sample_id']]['prefix_survivors']==details[p['sample_id']]['prefix_survivors'],'scalar exact output and every prefix invariance')
  ids=tuple(sorted((r['sample_id'],p['sample_id'])));scalarpairs[ids]={'sample_ids':list(ids),'unqueried_column':y,'exchanged_strings':['linen',value]}
 obs_groups=collections.defaultdict(list)
 for r in fresh:
  t=r['semantic_task'];obs_groups[(r['semantic_family_id'],styles[r['sample_id']],digest({'observations':t['observations'],'queries':t['queries']}))].append(r)
 for key,rs in obs_groups.items():
  empty=[r for r in rs if not targets[r['sample_id']]['surviving_worlds']];positive=[r for r in rs if targets[r['sample_id']]['surviving_worlds']]
  for l in empty:
   matches=[r for r in positive if details[l['sample_id']]['prefix_survivors'][1]==details[r['sample_id']]['prefix_survivors'][1] and len(l['semantic_task']['worlds'])==len(r['semantic_task']['worlds'])]
   require(len(matches)==1,'unique actual same observations opposite status matched table');p=matches[0]
   y=next(o['fact'] for o in l['semantic_task']['observations'] if o['fact'] not in {q['fact'] for q in l['semantic_task']['queries']})
   lw,rw=l['semantic_task']['worlds'],p['semantic_task']['worlds'];require([w['world_id'] for w in lw]==[w['world_id'] for w in rw],'same world-ID ordering')
   diffs=[(a['world_id'],f,typed(a['facts'][f]),typed(b['facts'][f])) for a,b in zip(lw,rw) for f in a['facts'] if typed(a['facts'][f])!=typed(b['facts'][f])]
   require(bool(diffs) and {x[1] for x in diffs}=={y},'opposite tables differ solely in actual unqueried auxiliary column')
   oppositepairs[(l['sample_id'],p['sample_id'])]={'sample_ids':[l['sample_id'],p['sample_id']],'observation_length':len(l['semantic_task']['observations']),'first_prefix_ids':details[l['sample_id']]['prefix_survivors'][1],'changed_column':y,'changed_world_cells':len(diffs)}
 return {'actual_families':len(byfamily),'family_sizes':{k:len(v) for k,v in byfamily.items()},'decoded_scalar_tables_per_family':family_tables,'normalized_base_cases':len({(r['semantic_family_id'],digest({'worlds':normalized[r['sample_id']]['worlds'],'observations':normalized[r['sample_id']]['observations']})) for r in fresh}),'conditional_full27_groups':len(conditional),'conditional_crossings':crossing_receipts,'field_order_pairs':fieldpairs,'scalar_invariance_pairs':list(scalarpairs.values()),'same_observations_opposite_status_pairs':list(oppositepairs.values()),'interpretation':'All relationships reconstructed from visible inputs. No actual producer context/scalar/role/property/manifest declarations inspected; derived columns are not attribution of producer role names.'},roles

def main():
 started=time.time();pre=loadjson((ROOT/PREEXEC).read_bytes());before={p:sha(ROOT/p) for p in pre['file_bindings']}
 runtime_before={p:sha(pathlib.Path(p)) for p in pre['runtime_dependencies']};require(runtime_before==pre['runtime_dependencies'],'runtime dependencies before');require(before==pre['file_bindings'],'all bound allowed bytes before execution');require(sha(pathlib.Path(__file__))==pre['new_source_sha256'],'new source binding')
 calibration=toys();attacks=minimal_counterexamples()
 fresh,old=rows(ROOT/VIEW),rows(ROOT/OLD);require(len(fresh)==len(old)==3456,'actual cohort counts')
 oldlookup=collections.defaultdict(list)
 for r in old:require(set(r)==FIELDS,'old target-free eight-field boundary');oldlookup[input_key(r)].append(r)
 targets={};details={};styles={};leads=collections.Counter();predictions=[];judgments=[];joins=[];familylineage=collections.defaultdict(set);metacount=collections.Counter()
 for r in fresh:
  require(set(r)==FIELDS,'fresh target-free eight-field boundary');require(r['track_id']=='RBM-001' and r['split'] in ('train','dev_challenge'),'track/split');require(re.fullmatch('[0-9a-f]{64}',r['record_hash']) is not None,'opaque complete record hash join key')
  require(len(oldlookup[input_key(r)])==1,'exact one independently constructed typed task/prompt/split/track join');o=oldlookup[input_key(r)][0]
  familylineage[r['semantic_family_id']].add(o['semantic_family_id']);joins.append({'sample_id':r['sample_id'],'prior_sample_id':o['sample_id'],'exact_input_sha256':input_key(r)})
  t,lead,style=parse_prompt(r);leads[lead]+=1;styles[r['sample_id']]=style
  a,d=solve(t);b,bd=own_prior_bitmask_solve(t);require(canon(a)==canon(b) and d['prefix_survivors']==bd['prefix_survivors'],'fresh set/bitmask recomputation agreement')
  checks=metamorphics(t,a,d);metacount.update(checks);targets[r['sample_id']]=a;details[r['sample_id']]=d
  predictions.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_execution_id':ACTOR,'stage':'FRESH_INPUT_ONLY_BEFORE_V4_CANONICAL_ACCESS','typed_input_sha256':digest(t),'exact_student_input_and_split_sha256':input_key(r),'expected_target':a,'expected_target_sha256':digest(a),'prefix_supports':d,'prediction_method':'full fresh typed set filtering; own prior positional bitmask separately reexecuted on every fresh input','prior_case_match_sample_id':o['sample_id'],'prior_gold_exposure_disclosure':DISCLOSURE})
  judgments.append({'receipt_id':'adv-belief-gap-v4-input-'+digest([r['sample_id'],r['record_hash']])[:24],'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_execution_id':ACTOR,'reviewer_role':'Adversarial Red-Team input-only','additional_audit_role':'diversity and leakage auditor','decision':'INPUT_WELL_FORMED_AND_EXACTLY_SOLVED','dataset_decision':None,'gold_access':False,'confidence':0.99,'critical_defects':[],'issues':[],'scores':{'semantic_correctness':5,'target_correctness':None,'instruction_clarity':5,'reasoning_validity':5,'generalization_value':None,'difficulty_quality':None,'diversity':None,'shortcut_resistance':None},'evidence':{'own_expected_target_sha256':digest(a),'exact_prior_input_join':o['sample_id'],'independent_fresh_set_and_own_bitmask_agree':True,'typed_prompt_binding':True,'metamorphic_checks':checks},'scope_limitation':DISCLOSURE+' Full record hash cannot be recomputed from a stripped view; stored target and all producer controls remain unreviewed.'})
 require(len({r['sample_id'] for r in fresh})==3456 and len({r['record_hash'] for r in fresh})==3456,'unique fresh IDs and full record hash join keys')
 require(len(leads)==1 and all(len(x)==1 for x in familylineage.values()) and len(familylineage)==4,'exact common contract/four independently joined families')
 pp=ROOT/(PREFIX+'predictions.jsonl');jp=ROOT/(PREFIX+'first_judgments.jsonl');freeze(pp,jlbytes(predictions));freeze(jp,jlbytes(judgments))
 first={'stage':'ALL_FRESH_EXPECTATIONS_AND_INITIAL_JUDGMENTS_FROZEN_BEFORE_V4_GOLD','actor':ACTOR,'records':3456,'predictions_sha256':sha(pp),'initial_judgments_sha256':sha(jp),'input_sha256':before[VIEW],'canonical_access':False,'dataset_approval':False,'source_sha256':pre['new_source_sha256'],'own_prior_predictions_crosscheck_performed_after_this_freeze':True}
 freeze(ROOT/(PREFIX+'expectation_freeze.json'),jsonbytes(first));print('FIRST_FREEZE '+canon(first),flush=True)
 # Only now inspect all own earlier predictions, after every fresh target and exact input join.
 oldpred={r['sample_id']:r for r in rows(ROOT/OLDP)};crosschecks=[]
 for join in joins:
  p=oldpred[join['prior_sample_id']];require(canon(p['expected_target'])==canon(targets[join['sample_id']]),'fresh full recomputation vs own old frozen expectation');require(p['prefix_evidence']['prefix_survivors']==details[join['sample_id']]['prefix_survivors'],'fresh exact prefix IDs vs own old expectation');crosschecks.append(join)
 lineage=loadjson((ROOT/'reviews/input_only/belief-gap-v4.input_lineage.json').read_bytes());claimed={j['fresh_sample_id']:j for j in lineage['joins']}
 require(len(claimed)==3456 and all(claimed[j['sample_id']]['prior_sample_id']==j['prior_sample_id'] and claimed[j['sample_id']]['same_typed_student_input_and_split'] is True for j in joins),'root lineage actual ID joins independently reconstructed')
 co,roles=cohort_analysis(fresh,targets,details,styles)
 examples=[];queries=[];rulehist=collections.defaultdict(collections.Counter);status=collections.defaultdict(collections.Counter);qhist=collections.defaultdict(collections.Counter)
 for r in fresh:
  sid=r['sample_id'];t=r['semantic_task'];a=targets[sid];d=details[sid];empty=not bool(a['surviving_worlds']);feat=scalar_features(t,d,r['rendered_input'])
  examples.append({'sample_id':sid,'split':r['split'],'features':feat,'empty':empty});status[r['split']]['empty' if empty else 'nonempty']+=1
  marker=any(type(o['equals']) is str and o['equals'] in ('stone','unknown') for o in t['observations']);linen=any(type(o['equals']) is str and o['equals']=='linen' for o in t['observations'])
  for name,pred in [('original_marker',marker),('marker_AND_linen',marker and linen)]:rulehist[name]['correct' if pred==empty else 'incorrect']+=1;rulehist[name][r['split']+'_correct' if pred==empty else r['split']+'_incorrect']+=1
  for i,(q,z) in enumerate(zip(t['queries'],a['queries'])):
   bins=partition(t,{w['world_id'] for w in t['worlds']},q['fact']);qfeat={'query_ID':q['query_id'],'query_position':i,'fact_name':q['fact'],'ID_position_fact':canon([q['query_id'],i,q['fact']]),'full_table_column_bins':canon(bins),'full_table_column_frequency':canon(sorted(z['count'] for z in bins)),'full_table_column_types':canon(sorted(type(z['value']).__name__ for z in bins)),'world_count_and_query_position':canon([len(t['worlds']),i]),'query_column_role_from_public_table':roles[sid][q['query_id']],'queried_column_bins_observed_scalars_joint':canon([bins,[o['equals'] for o in t['observations']]])}
   queries.append({'sample_id':sid,'split':r['split'],'features':qfeat,'useful':z['can_reduce_uncertainty'],'entailed':z['entailed']});qhist[r['split']]['useful_true' if z['can_reduce_uncertainty'] else 'useful_false']+=1
 # Every baseline label is independently computed; no model training and no private claim labels.
 baseline=[lookup_measure(examples,k,'empty') for k in examples[0]['features']];querybaseline=[lookup_measure(queries,k,'useful') for k in queries[0]['features']]
 actual_ce=[]
 for name,selector in [('original_marker',lambda t:any(type(o['equals']) is str and o['equals'] in ('stone','unknown') for o in t['observations'])),('marker_AND_linen',lambda t:any(type(o['equals']) is str and o['equals'] in ('stone','unknown') for o in t['observations']) and any(o['equals']=='linen' for o in t['observations']))]:
  bad=[r for r in fresh if selector(r['semantic_task'])!=(not bool(targets[r['sample_id']]['surviving_worlds']))];m=min(bad,key=lambda r:(len(r['semantic_task']['worlds']),len(r['rendered_input']),r['sample_id']));actual_ce.append({'rule':name,'sample_id':m['sample_id'],'record_hash':m['record_hash'],'public_task':m['semantic_task'],'shortcut_empty_prediction':selector(m['semantic_task']),'own_target':targets[m['sample_id']]})
 rawgroups=collections.defaultdict(list)
 for r in fresh:rawgroups[digest(r['rendered_input'])].append(r['sample_id'])
 splits_by_task=collections.defaultdict(set)
 for r in fresh:splits_by_task[digest(r['semantic_task'])].add(r['split'])
 priorjoin= {'matched_rows':len(joins),'every_typed_task_rendered_input_split_track_equal':True,'fresh_to_old_family_mapping':{k:sorted(v) for k,v in familylineage.items()},'independently_recomputed_prior_expectation_matches':len(crosschecks),'added_input_tasks':0,'added_source_families':0,'actual_full_record_hash_validation':'not possible from stripped view; opaque hashes bound for later canonical join','root_lineage_join_recomputed':True}
 shortcut={'actual_execution_id':ACTOR,'stage':'INPUT_ONLY','disclosure':DISCLOSURE,'status_by_split':dict(status),'query_usefulness_by_split':dict(qhist),'status_majority_accuracy':max(sum(v[x] for v in status.values()) for x in ('empty','nonempty'))/3456,'query_usefulness_majority_accuracy':max(sum(v[x] for v in qhist.values()) for x in ('useful_true','useful_false'))/len(queries),'frozen_literal_rule_actual_counts':dict(rulehist),'input_only_status_lookup_baselines':baseline,'input_only_query_usefulness_lookup_baselines':querybaseline,'actual_sample_counterexamples':actual_ce,'fresh_handmade_counterexamples':attacks,'strongest_fitted_status_baseline':max(baseline,key=lambda x:x['fitted_correct']),'strongest_exposed_dev_status_baseline':max(baseline,key=lambda x:x['train_lookup_exposed_dev_correct']),'strongest_fitted_query_usefulness_baseline':max(querybaseline,key=lambda x:x['fitted_correct']),'scope_limits':['Finite input-table/count/frequency fits may exploit common source motifs and are not demonstrations of learned shortcuts.','Question flags have strong class imbalance; target-format/output length cannot be used as an input feature here.','No producer role/locator/claim/difficulty or full manifest reviewed.','Four sources and exposed development, all original inputs previously seen; no sealed OOD, novelty, Bayesian/information-action utility, architecture or efficacy result.']}
 diversity={'actual_execution_id':ACTOR,'stage':'INPUT_ONLY','disclosure':DISCLOSURE,'records':3456,'train_dev':dict(collections.Counter(r['split'] for r in fresh)),'prior_lineage':priorjoin,'cohort_reconstruction':co,'exact_prompt_duplicate_groups':[ids for ids in rawgroups.values() if len(ids)>1],'exact_typed_task_groups':len(splits_by_task),'cross_split_exact_typed_task_groups':sum(len(x)>1 for x in splits_by_task.values()),'input_field_boundary_exact':sorted(FIELDS),'visible_contract_variants':len(leads),'visible_object_order_styles':{canon(k):v for k,v in collections.Counter(styles.values()).items()},'metamorphic_actual_counts':dict(metacount),'calibration':calibration,'scope_limits':shortcut['scope_limits']}
 sp='audits/shortcut_detection/belief-gap-v4-input-only-adversarial-r2.json';dp='audits/diversity/belief-gap-v4-input-only-adversarial-r2.json'
 freeze(ROOT/sp,jsonbytes(shortcut));freeze(ROOT/dp,jsonbytes(diversity))
 joinpath=PREFIX+'lineage.json';freeze(ROOT/joinpath,jsonbytes({'actor':ACTOR,'actual_recomputed_joins':joins,'prior_input_only_view_hash':before[OLD],'fresh_input_only_view_hash':before[VIEW]}))
 runtime_after={p:sha(pathlib.Path(p)) for p in pre['runtime_dependencies']};require(runtime_after==runtime_before,'runtime unchanged after');after={p:sha(ROOT/p) for p in pre['file_bindings']};require(after==before,'exact allowed sources/inputs unchanged after all executions')
 outputs={p:sha(ROOT/p) for p in [str(pp.relative_to(ROOT)),str(jp.relative_to(ROOT)),PREFIX+'expectation_freeze.json',sp,dp,joinpath]}
 execution={'actual_execution_id':ACTOR,'completed':True,'stage':'FINAL_INPUT_ONLY_AWAITING_CANONICAL_AUTHORIZATION','all_fresh_predictions_and_initial_judgments':3456,'full_targets_recomputed':3456,'fresh_set_vs_own_bitmask_checks':3456,'prior_expectation_crosschecks_after_recomputation':3456,'hand_exact_toys':7,'malformed_toys':3,'fresh_semantic_counterexamples':len(attacks['fresh_minimal_semantic_counterexamples']),'fresh_prompt_corruption_rejections':len(attacks['actual_prompt_mutations_rejected']),'metamorphic_counts':dict(metacount),'actual_input_controls':{k:len(co[k]) if type(co[k]) is list else co[k] for k in ('actual_families','normalized_base_cases','conditional_full27_groups','field_order_pairs','scalar_invariance_pairs','same_observations_opposite_status_pairs')},'runtime_dependencies_before':runtime_before,'runtime_dependencies_after':runtime_after,'file_bindings_before':before,'file_bindings_after':after,'preexec_sha256':sha(ROOT/PREEXEC),'source_sha256':pre['new_source_sha256'],'outputs':outputs,'wall_seconds':time.time()-started,'canonical_or_other_reviewer_access':False,'dataset_approval':False,'input_semantic_defects_found':0,'disclosure':DISCLOSURE,'residual_limits':shortcut['scope_limits']}
 ep='audits/execution/belief-gap-v4-input-only-adversarial-r2.json';freeze(ROOT/ep,jsonbytes(execution));print('FINAL '+canon({'execution_path':ep,'execution_sha256':sha(ROOT/ep),'outputs':outputs,'counts':execution['actual_input_controls'],'status_majority':shortcut['status_majority_accuracy'],'usefulness_majority':shortcut['query_usefulness_majority_accuracy'],'strongest_dev_status':shortcut['strongest_exposed_dev_status_baseline']}),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as e:
  failure={'actor':ACTOR,'source_sha256':sha(pathlib.Path(__file__)),'actual_failure_type':type(e).__name__,'actual_failure':str(e),'fresh_canonical_access':False,'completed':False}
  p=ROOT/'audits/execution/belief-gap-v4-input-only-adversarial-r2.failure.json';freeze(p,jsonbytes(failure));print('FAILURE '+canon(failure),flush=True);raise
