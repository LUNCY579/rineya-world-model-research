"""NONPRODUCTION public-only pair-join demonstration; /root/formal_oracle.

Does not import or alter any existing checker/producer/factory source. Does not
read any candidate/gold/private manifest/review. Three-world toy only.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import hashlib
import io
import json
import sys
import traceback
import unittest

BASE=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
NAMES=('same_prefix_empty_nonempty_counterfactual','contradiction_equals_mixed_outcomes')
AGGREGATES={'auxiliary_scalar_transposition_invariance','same_observations_auxiliary_table_counterfactual','context_auxiliary_scalar_variant_crossing'}
SOURCE_HASHES={
 'schemas/BELIEF_CONTROL_PROPERTIES.md':'200631973256bbe5dcd7b99f3d52cb95f89c442acd29ea4de1fe82dd8b5f3034',
 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V2.md':'46e3120dfd1ee931eec9d1e5e0e75f0b2aedb1d85764fd755afe9f7d36cf121f',
 'schemas/BELIEF_CONTROL_PROPERTIES_REVISION_V3.md':'3a223952d9664577d7b637ddfb3e18d329863a3d5371a6c47cad233a1f50ff11',
 'schemas/BELIEF_CONTROL_PROPERTIES_METADATA_V4.md':'40b552ecf47c11bfffb2669283640dda991d6dc6e7094ed3051fb9cad2489fce',
 'research/curricula/RBM-EVIDENCE-GAP.md':'004fb0ec0c1a1104ba052fbda4882f48523d7d34dd45b2b88e8c890713b4a120',
 'research/curricula/RBM-EVIDENCE-GAP-REVISION-V2.md':'c2eb841ed546848088db4e0499e8e8f7087aa3d213af333ff39237ae6558bd33',
 'research/curricula/RBM-EVIDENCE-GAP-REVISION-V3.md':'7e9c9f9160425ab7e455f57d72385e54f399835dfa351dcf75f6fc2842b1dc33',
 'research/curricula/BELIEF-METADATA-V4-CLARIFICATION-EXECUTION.json':'2d100ce2ad27ded95275316991f3ec8ace9ef9daf2a40c50c3f06946c777ec0d',
 'reviews/oracle/RBM-v3-public-interface-interpretation-20261010T040233Z.json':'11c54b46e7ccc127f4a8bb58a6595609cd1eaec2d33e98b871bebdddfd62c2f9',
 'feedback/improvement_requests/RBM-INTERFACE-012.json':'240aa6a31ec822e1cf52fd8f4f68597a75c87a48f7c72cbf9a19ba44b4dce962'}
COUNTS=Counter()
NEGATIVE_EVENTS=[]
def enc(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def equal(a,b):return enc(a)==enc(b)
def need(test,message):
    if not test:raise ValueError(message)

def derive(task):
    need(type(task) is dict and set(task)=={'kind','worlds','observations','queries'} and task['kind']=='belief','malformed task')
    worlds=task['worlds'];need(type(worlds) is list and worlds,'toy requires nonempty worlds')
    ids=[];keys=None
    for world in worlds:
        need(type(world) is dict and set(world)=={'world_id','facts'} and type(world['world_id']) is str,'malformed world')
        need(type(world['facts']) is dict,'facts object required')
        keys=keys if keys is not None else set(world['facts'])
        need(set(world['facts'])==keys and all(type(x) is str for x in keys),'fact domain mismatch')
        need(all(v is None or type(v) in (bool,int,str) for v in world['facts'].values()),'scalar domain excludes floats')
        ids.append(world['world_id'])
    need(len(ids)==len(set(ids)),'duplicate world IDs')
    support=list(worlds);prefixes=[sorted(ids)]
    for observation in task['observations']:
        need(type(observation) is dict and set(observation)=={'fact','equals'} and observation['fact'] in keys,'malformed observation')
        value=observation['equals'];need(value is None or type(value) in (bool,int,str),'malformed equality')
        support=[w for w in support if equal(w['facts'][observation['fact']],value)]
        prefixes.append(sorted(w['world_id'] for w in support))
    answers=[];query_ids=[]
    for query in task['queries']:
        need(type(query) is dict and set(query)=={'query_id','fact'} and query['fact'] in keys and type(query['query_id']) is str,'malformed query')
        query_ids.append(query['query_id']);bins=Counter(enc(w['facts'][query['fact']]) for w in support)
        answers.append({'query_id':query['query_id'],'bins':[[json.loads(v),n] for v,n in sorted(bins.items())],
                        'distinct_values':len(bins),'entailed':bool(support) and len(bins)==1,'useful':len(bins)>1})
    need(len(query_ids)==len(set(query_ids)),'duplicate query IDs')
    return {'prefixes':prefixes,'answers':answers,'final':prefixes[-1]}

def claim_name(claim):
    need(type(claim) is dict,'claim object required')
    allowed={'name','property','scope','source_family_key','evidence_manifest','control_id','pair_sample_ids','probe_id'}
    need(set(claim)<=allowed and (('name' in claim)+('property' in claim)==1),'closed claim vocabulary/exclusive name')
    return claim.get('name',claim.get('property'))

def join(rows,claimant_id,claim,entries,candidate_sha,matched_owner=False):
    name=claim_name(claim)
    if name in AGGREGATES:
        need(claim.get('scope')=='corpus_variant','aggregate supplied scope required')
        return {'scope':'DEFERRED_FULL_GROUP_PROOF','note':'Aggregate control not narrowed to inherited inline-pair specialization; no truth approval.'}
    need(name in NAMES and claim.get('scope')=='corpus_variant','unknown inherited name/scope')
    need({'pair_sample_ids','evidence_manifest','control_id'}<=set(claim) and 'probe_id' not in claim,'required inherited fields/corpus-versus-probe')
    for key in ('control_id','evidence_manifest'):need(type(claim[key]) is str and bool(claim[key]),'nonempty descriptive pointer/ID required')
    pointer=Path(claim['evidence_manifest']);need(not pointer.is_absolute() and '..' not in pointer.parts,'repository-relative manifest path required')
    claimant=rows[claimant_id];family=claimant['source_family_key']
    need(claim.get('source_family_key',family)==family,'claimant family mismatch')
    pair=claim['pair_sample_ids'];need(type(pair) is list and len(pair)==2 and len(set(pair))==2 and all(type(x) is str and x in rows for x in pair),'actual distinct pair IDs required')
    a,b=[rows[x] for x in pair]
    need(a['source_family_key']==b['source_family_key']==family,'foreign pair family')
    need((a['controlled_context'],b['controlled_context']) in (('E','G'),('F','H')),'ordered empty/nonempty contexts required')
    for key in ('auxiliary_scalar_variant','schedule_index','renderer'):
        need(equal(a[key],b[key]),'pair coordinate mismatch '+key)
    need(type(a['schedule_index']) is int and 0<=a['schedule_index']<27,'schedule index excludes bool')
    need(a['auxiliary_scalar_variant'] in ('identity','transposed') and a['renderer'] in ('forward','reverse'),'unknown variant/style')
    ta,tb=a['semantic_task'],b['semantic_task'];da,db=derive(ta),derive(tb)
    length=len(ta['observations']);need(length==(2 if a['controlled_context']=='E' else 3) and len(tb['observations'])==length,'length/context mismatch')
    need(equal(ta['worlds'],tb['worlds']) and equal(ta['queries'],tb['queries']),'pair worlds/ordered queries differ')
    need(all(equal(x['fact'],y['fact']) for x,y in zip(ta['observations'],tb['observations'])),'observation fact sequence differs')
    changed=[i for i,(x,y) in enumerate(zip(ta['observations'],tb['observations'])) if not equal(x,y)]
    need(changed==[1] and equal(ta['observations'][0],tb['observations'][0]),'only second equality may change')
    need(da['prefixes'][1]==db['prefixes'][1] and da['prefixes'][1],'exact first-prefix support mismatch')
    need(not da['final'] and db['final'],'opposite final statuses required')
    need(ta['observations'][0]['fact']!=ta['observations'][1]['fact'],'contradiction must be cross-fact')
    for observation in ta['observations'][:2]:
        single=deepcopy(ta);single['observations']=[observation];need(derive(single)['final'],'contradiction equality individually unsupported')
    if length==3:
        need(equal(ta['observations'][0],ta['observations'][2]) and equal(tb['observations'][0],tb['observations'][2]),'F/H repeat first observation')
        need(not da['prefixes'][2] and da['prefixes'][3]==[] and db['prefixes'][2]==db['prefixes'][3],'absorption/nonempty repeat mismatch')
    required={'name','scope','source_family_key','auxiliary_scalar_variant','observation_length','control_id','pair_sample_ids','exact_candidate_sha256'}
    matching=[entry for entry in entries if type(entry) is dict and entry.get('control_id')==claim['control_id'] and entry.get('name')==name and equal(entry.get('pair_sample_ids'),pair)]
    need(matching,'exact ID + name + actual pair has no manifest entry')
    for entry in matching:
        need(required<=set(entry),'manifest minimum published fields missing')
        need(entry['scope']=='corpus_variant' and entry['source_family_key']==family and entry['auxiliary_scalar_variant']==a['auxiliary_scalar_variant'],'manifest scope/family/variant mismatch')
        need(type(entry['observation_length']) is int and entry['observation_length']==length,'manifest typed length mismatch')
        need(entry['exact_candidate_sha256']==candidate_sha,'manifest exact candidate hash mismatch')
    if matched_owner:
        need(len(claimant['semantic_task']['observations'])==length,'RECOMMENDED claimant length mismatch')
        for key in ('auxiliary_scalar_variant','schedule_index','renderer'):need(equal(claimant[key],a[key]),'RECOMMENDED claimant coordinate mismatch '+key)
        need(equal(claimant['semantic_task']['worlds'],ta['worlds']) and equal(claimant['semantic_task']['queries'],ta['queries']),'RECOMMENDED claimant coordinate task mismatch')
    return {'prefixes':[da['prefixes'],db['prefixes']],'query_partitions':[da['answers'],db['answers']],
            'actual_pair':pair,'name':name,'matching_named_entries':len(matching),'length':length,
            'claimant_is_pair_member':claimant_id in pair,'input_hashes':[sha(enc(t).encode()) for t in (ta,tb)]}

def fixture():
    rows={};entries=[];candidate_hash='a'*64
    worlds=[{'world_id':'w0','facts':{'x':'marker','y':'amber','q':None,'r':True,'z':'a'}},
            {'world_id':'w1','facts':{'x':'marker','y':'amber','q':None,'r':1,'z':'b'}},
            {'world_id':'w2','facts':{'x':'other','y':'linen','q':'value','r':False,'z':'a'}}]
    queries=[{'query_id':'one','fact':'q'},{'query_id':'two','fact':'r'},{'query_id':'three','fact':'x'}]
    for variant in ('identity','transposed'):
        for style in ('forward','reverse'):
            for context in ('A','D','E','G','F','H'):
                obs=[{'fact':'x','equals':'marker'},{'fact':'y','equals':'linen' if context in ('E','F') else 'amber'}]
                if context in ('D','F','H'):obs.append(deepcopy(obs[0]))
                table=deepcopy(worlds)
                if variant=='transposed':
                    toggle={'linen':'amber','amber':'linen'}
                    for w in table:w['facts']['y']=toggle[w['facts']['y']]
                    for o in obs:
                        if o['fact']=='y':o['equals']=toggle[o['equals']]
                task={'kind':'belief','worlds':table,'observations':obs,'queries':deepcopy(queries)}
                sid='arbitrary '+context+' λ '+variant+' '+style
                rows[sid]={'sample_id':sid,'source_family_key':'toy family','controlled_context':context,
                    'auxiliary_scalar_variant':variant,'schedule_index':0,'renderer':style,'semantic_task':task}
            for left,right in (('E','G'),('F','H')):
                pair=['arbitrary '+c+' λ '+variant+' '+style for c in (left,right)]
                for name in NAMES:entries.append({'name':name,'scope':'corpus_variant','source_family_key':'toy family',
                    'auxiliary_scalar_variant':variant,'observation_length':2 if left=='E' else 3,
                    'control_id':'same descriptive key Ω','pair_sample_ids':pair,'exact_candidate_sha256':candidate_hash,
                    'optional_provenance_note':'Not a mandatory private evidence shape.'})
    return rows,entries,candidate_hash

def make_claim(rows,sid,name=NAMES[0]):
    row=rows[sid];left,right=('E','G') if len(row['semantic_task']['observations'])==2 else ('F','H')
    return {'name':name,'scope':'corpus_variant','evidence_manifest':'hand_manifest.json',
        'control_id':'same descriptive key Ω','pair_sample_ids':['arbitrary '+c+' λ '+row['auxiliary_scalar_variant']+' '+row['renderer'] for c in (left,right)]}

class HandTests(unittest.TestCase):
    def negative(self,label,rows,entries,candidate,sid,claim,matched=True):
        try:join(rows,sid,claim,entries,candidate,matched)
        except ValueError as exc:NEGATIVE_EVENTS.append({'label':label,'error':str(exc)});COUNTS['negative_cases']+=1
        else:self.fail('negative mutation silently accepted: '+label)

    def test_all_toy_names_lengths_variants_styles_members_nonmembers(self):
        rows,entries,h=fixture()
        for sid in rows:
            for name in NAMES:
                out=join(rows,sid,make_claim(rows,sid,name),entries,h,True)
                self.assertEqual(out['length'],len(rows[sid]['semantic_task']['observations']))
                COUNTS['positive_claim_joins']+=1
        self.assertEqual(len(rows),24);COUNTS['toy_rows']+=len(rows)

    def test_shared_ids_property_alias_and_identical_duplicate_entries(self):
        rows,entries,h=fixture();sid=next(iter(rows));c=make_claim(rows,sid)
        c['property']=c.pop('name');c['source_family_key']='toy family'
        join(rows,sid,c,entries,h,True);entries.append(deepcopy(entries[0]))
        self.assertEqual(join(rows,sid,c,entries,h,True)['matching_named_entries'],2)
        COUNTS['positive_claim_joins']+=2

    def test_closed_claim_shape_and_required_inline_fields(self):
        rows,entries,h=fixture();sid=next(iter(rows));base=make_claim(rows,sid)
        for key in ('name','scope','pair_sample_ids','control_id','evidence_manifest'):
            bad=deepcopy(base);bad.pop(key);self.negative('missing '+key,rows,entries,h,sid,bad)
        for key,value in (('property',NAMES[0]),('expected_status','empty'),('probe_id','unexecuted'),('scope','audit_probe'),('name','unknown'),('control_id',True),('evidence_manifest','/absolute'),('source_family_key','foreign')):
            bad=deepcopy(base);bad[key]=value;self.negative('closed/type '+key,rows,entries,h,sid,bad)

    def test_manifest_resolution_required_fields_and_hash_not_private_format(self):
        rows,entries,h=fixture();sid=next(iter(rows));c=make_claim(rows,sid)
        for key,value in (('control_id','missing explicit mapping'),('name',NAMES[1]),('scope','audit_probe'),('source_family_key','foreign'),('observation_length',True),('observation_length',3),('auxiliary_scalar_variant','transposed'),('exact_candidate_sha256','b'*64)):
            bad=deepcopy(entries);bad[0][key]=value
            if key=='name':bad=[e for i,e in enumerate(bad) if i!=1]
            self.negative('manifest '+key,rows,bad,h,sid,c)
        bad=deepcopy(entries);bad[0].pop('exact_candidate_sha256');self.negative('manifest absent candidate hash',rows,bad,h,sid,c)

    def test_actual_pair_ids_contexts_family_coordinate_and_typed_indices(self):
        rows,entries,h=fixture();sid=next(iter(rows));base=make_claim(rows,sid)
        for pair in ([base['pair_sample_ids'][0]]*2,list(reversed(base['pair_sample_ids'])),['absent',base['pair_sample_ids'][1]]):
            c=deepcopy(base);c['pair_sample_ids']=pair;self.negative('bad actual pair',rows,entries,h,sid,c)
        for key,value in (('source_family_key','foreign'),('controlled_context','A'),('schedule_index',1),('schedule_index',True),('renderer','reverse'),('auxiliary_scalar_variant','transposed')):
            bad=deepcopy(rows);bad[base['pair_sample_ids'][1]][key]=value;self.negative('pair coordinate '+key,bad,entries,h,sid,base)

    def test_false_pair_facts_queries_first_prefix_and_repeat(self):
        rows,entries,h=fixture();sid=next(iter(rows));base=make_claim(rows,sid);g=base['pair_sample_ids'][1]
        for mode in ('world','query','first','same_second','wrong_fact'):
            bad=deepcopy(rows);t=bad[g]['semantic_task']
            if mode=='world':t['worlds'][0]['facts']['y']='other'
            elif mode=='query':t['queries'].reverse()
            elif mode=='first':t['observations'][0]['equals']='other'
            elif mode=='same_second':t['observations'][1]['equals']='linen'
            else:t['observations'][1]['fact']='z'
            self.negative('false facts '+mode,bad,entries,h,sid,base)
        d=next(s for s,r in rows.items() if r['controlled_context']=='D');c=make_claim(rows,d);bad=deepcopy(rows)
        bad[c['pair_sample_ids'][1]]['semantic_task']['observations'][2]['equals']='other'
        self.negative('F/H false repeat',bad,entries,h,d,c)

    def test_nonmember_scope_and_recommendation_not_unpublished_gate(self):
        rows,entries,h=fixture();a=next(iter(rows));d=next(s for s,r in rows.items() if r['controlled_context']=='D')
        out=join(rows,a,make_claim(rows,a),entries,h,True);self.assertFalse(out['claimant_is_pair_member'])
        c=make_claim(rows,a);self.assertFalse(join(rows,d,c,entries,h,False)['claimant_is_pair_member'])
        self.negative('recommended claimant length',rows,entries,h,d,c,True)
        for key,value in (('schedule_index',1),('renderer','reverse'),('auxiliary_scalar_variant','transposed')):
            bad=deepcopy(rows);bad[a][key]=value
            self.assertTrue(join(bad,a,c,entries,h,False));self.negative('recommended owner '+key,bad,entries,h,a,c,True)

    def test_typed_bins_exact_first_prefix_and_scalar_involution(self):
        rows,_,_=fixture();g=next(r for r in rows.values() if r['controlled_context']=='G' and r['auxiliary_scalar_variant']=='identity')
        out=derive(g['semantic_task']);self.assertEqual(out['answers'][1]['bins'],[[1,1],[True,1]])
        altered=deepcopy(g['semantic_task']);altered['observations']=[{'fact':'r','equals':True}]
        self.assertEqual(derive(altered)['final'],['w0']);altered['observations'][0]['equals']=1
        self.assertEqual(derive(altered)['final'],['w1']);COUNTS['typed_probes']+=2
        for context in ('A','D','E','G','F','H'):
            a=rows['arbitrary '+context+' λ identity forward']['semantic_task'];b=rows['arbitrary '+context+' λ transposed forward']['semantic_task']
            self.assertEqual(derive(a),derive(b));self.assertNotEqual(enc(a['worlds']),enc(b['worlds']));COUNTS['scalar_invariance_pairs']+=1

    def test_queried_auxiliary_breaks_exact_output_invariance(self):
        rows,_,_=fixture();a=deepcopy(rows['arbitrary G λ identity forward']['semantic_task']);b=deepcopy(rows['arbitrary G λ transposed forward']['semantic_task'])
        a['queries'].append({'query_id':'aux','fact':'y'});b['queries'].append({'query_id':'aux','fact':'y'})
        self.assertNotEqual(derive(a)['answers'],derive(b)['answers']);COUNTS['negative_semantic_controls']+=1

    def test_aggregate_names_remain_separate_unapproved_group_proof(self):
        rows,entries,h=fixture();sid=next(iter(rows))
        for name in AGGREGATES:
            out=join(rows,sid,{'name':name,'scope':'corpus_variant','control_id':'aggregate shared Ω'},entries,h)
            self.assertEqual(out['scope'],'DEFERRED_FULL_GROUP_PROOF');COUNTS['aggregate_interface_examples']+=1

def run():
    bindings={}
    for path,expected in SOURCE_HASHES.items():
        actual=sha((BASE/path).read_bytes());need(actual==expected,'authorized public source changed '+path);bindings[path]=actual
    for path in (Path(__file__),HERE/'proposal.md'):
        digest=sha(path.read_bytes());blob=BASE/'reproducibility/source_blobs'/(digest+'.source')
        need(blob.exists() and blob.read_bytes()==path.read_bytes(),'new source/proposal not archived before execution')
        bindings[str(path.relative_to(BASE))]=digest
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(HandTests))
    rows,entries,candidate=fixture();example=next(iter(rows));c=make_claim(rows,example)
    evidence=join(rows,example,c,entries,candidate,True)
    hand={'scope':'Three-world illustrative join; placeholder candidate hash belongs to a toy, never a claimed real artifact hash.',
        'rows':rows,'manifest_entries':entries,'claim_example':c,'independently_computed_pair_evidence':evidence}
    hand_path=HERE/'hand_examples.json';need(not hand_path.exists(),'hand output immutable');hand_path.write_text(json.dumps(hand,indent=2,ensure_ascii=False)+'\n')
    output={'actual_execution_id':'/root/formal_oracle','role':'Public-only nonbinding preparation; no production/data approval',
        'at_utc':datetime.now(timezone.utc).isoformat(),'python_executable':sys.executable,'source_sha256':bindings,
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'counts':dict(COUNTS),
        'test_output':stream.getvalue(),'preserved_expected_negative_results':NEGATIVE_EVENTS,
        'hand_examples_sha256':sha(hand_path.read_bytes()),'hand_examples_bytes':hand_path.stat().st_size,
        'limitations':'No production candidate, full four-family A-H/profile/27-schedule/3456-cohort validation or old checker read. Candidate hash is an explicitly marked fixture token. Prototype does not resolve a real file; actual resolver/candidate/record-hash custody needs future separately frozen implementation. Aggregate group proof deliberately deferred rather than approved. Recommended-owner mode is prospective recommendation, not asserted existing V4 hard requirement.',
        'guided_scope':'Frozen V1/V2/V3 public docs, own interpretation and public ticket, plus later explicitly authorized frozen V4/public author receipt. No linked private evidence read.',
        'prior_authorship':'Author wrote prior belief checkers; same-model/root-guided preparation is not a new-model/blind-review result.',
        'old_files_unchanged_after_run':all(sha((BASE/p).read_bytes())==h for p,h in bindings.items())}
    out=HERE/'execution.json';need(not out.exists(),'execution output immutable');out.write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(enc({'path':str(out.relative_to(BASE)),'sha256':sha(out.read_bytes()),'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'counts':dict(COUNTS)}))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':run()
