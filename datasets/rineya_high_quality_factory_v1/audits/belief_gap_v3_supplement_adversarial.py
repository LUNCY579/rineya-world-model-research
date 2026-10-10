#!/usr/bin/env python3
"""Own executed adversarial mutants, residual marginal witnesses and immutable custody.
Only reviewer-authored solver and actual frozen source inputs are used.
"""
import collections,copy,gzip,hashlib,io,json,pathlib
import belief_gap_v3_adversarial_audit as a

def badsolve(t,mode):
    worlds=t['worlds'];remaining=worlds[:];sizes=[]
    for o in t['observations']:
        if mode=='forget_prior_observations':remaining=worlds[:]
        if mode=='coercing_filter':remaining=[w for w in remaining if w['facts'][o['fact']]==o['equals']]
        else:remaining=[w for w in remaining if a.typed(w['facts'][o['fact']])==a.typed(o['equals'])]
        sizes.append(len(remaining))
    qs=[]
    for q in t['queries']:
        vals=[w['facts'][q['fact']] for w in remaining]
        if mode=='coercing_partition':vals=[int(v) if type(v) is bool else v for v in vals]
        bins=collections.Counter(a.typed(v) for v in vals)
        if mode=='drop_multiplicity':bins={k:1 for k in bins}
        entries=[{'value':json.loads(k),'count':bins[k]} for k in sorted(bins)];ans=a.answer(q,entries,len(remaining))
        if mode=='null_cannot_be_entailed' and ans['guaranteed'] is None:ans['entailed']=False
        qs.append(ans)
    return {'surviving_worlds':sorted(w['world_id'] for w in remaining),'belief_sizes':sizes,'queries':qs}

def counts(t):return [sum(a.typed(w['facts'][o['fact']])==a.typed(o['equals']) for w in t['worlds']) for o in t['observations']]

def main():
    source=a.load(a.INPUT);pred={p['sample_id']:p for p in a.load(a.OUT/f'{a.BATCH}.input_predictions.jsonl')};cand=a.load(a.CAND);frozen={p:a.sha(a.OUT/p) for p in [f'{a.BATCH}.input_predictions.jsonl',f'{a.BATCH}.first_judgments.jsonl',f'{a.BATCH}.jsonl']};mutants={}
    for mode in ('coercing_filter','coercing_partition','forget_prior_observations','drop_multiplicity','null_cannot_be_entailed'):
        failed=[];hist=collections.Counter()
        for r in source:
            actual=badsolve(r['semantic_task'],mode);expect=pred[r['sample_id']]['expected_target']
            if a.canon(actual)!=a.canon(expect):
                failed.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'bad_target':actual,'exact_target':expect});hist[r['sample_id'].split('-p')[1].split('-s')[0]]+=1
        a.require(failed,'Mutant was not discriminated '+mode);mutants[mode]={'failed_tasks':len(failed),'total_tasks':len(source),'context_variant_failure_histogram':dict(hist),'first_actual_failure':failed[0]}
    # Both actual full tasks have N12 and ordered standalone supports [2,6], but opposite final status.
    positive=next(r for r in cand if r['source_family_index']==12 and r['controlled_context']=='G' and r['auxiliary_scalar_variant']=='identity' and r['schedule_index']==0 and r['renderer']=='forward')
    negative=next(r for r in cand if r['source_family_index']==18 and r['controlled_context']=='E' and r['auxiliary_scalar_variant']=='identity' and r['schedule_index']==0 and r['renderer']=='forward')
    tp,tn=positive['semantic_task'],negative['semantic_task'];a.require(counts(tp)==counts(tn)==[2,6],'Actual equal coarse marginal-support example');a.require(len(tp['worlds'])==len(tn['worlds'])==12,'Same actual world count');ap,dp=a.solve(tp);an,dn=a.solve(tn);a.require(ap['surviving_worlds'] and not an['surviving_worlds'],'Opposite actual status')
    def small(t,empty):
        x,y=t['observations'];qx=t['queries'][0];first=next(w for w in t['worlds'] if a.typed(w['facts'][x['fact']])==a.typed(x['equals']))
        if empty:second=next(w for w in t['worlds'] if a.typed(w['facts'][y['fact']])==a.typed(y['equals']))
        else:second=next(w for w in t['worlds'] if a.typed(w['facts'][x['fact']])!=a.typed(x['equals']) and a.typed(w['facts'][y['fact']])!=a.typed(y['equals']))
        fields={x['fact'],y['fact'],qx['fact']}
        return {'kind':'belief','worlds':[{'world_id':w['world_id'],'facts':{k:v for k,v in w['facts'].items() if k in fields}} for w in (first,second)],'observations':copy.deepcopy(t['observations']),'queries':[copy.deepcopy(qx)]}
    sp,sn=small(tp,False),small(tn,True);sap,sdp=a.solve(sp);san,sdn=a.solve(sn);a.require(counts(sp)==counts(sn)==[1,1] and len(sp['worlds'])==len(sn['worlds'])==2 and sap['surviving_worlds'] and not san['surviving_worlds'],'Actual reduced same-marginal counterexample')
    marginal={'positive_source_id':positive['sample_id'],'positive_hash':positive['record_hash'],'negative_source_id':negative['sample_id'],'negative_hash':negative['record_hash'],'actual_full_tasks_same_world_count':12,'actual_full_tasks_same_ordered_standalone_supports':[2,6],'actual_full_positive_target':ap,'actual_full_negative_target':an,'minimal_positive_task':sp,'minimal_positive_exact_target':sap,'minimal_negative_task':sn,'minimal_negative_exact_target':san,'minimal_same_world_count':2,'minimal_same_ordered_standalone_supports':[1,1],'scope':'Actual-source reduced two-world/one-query audit probes; no corpus additions. Marginal supports cannot establish the joint intersection.'}
    # All typed first-stage calibration remains frozen. Actual storage custody records actual executing author.
    raw=a.CAND.read_bytes();a.require(len(raw)==40541020 and a.sha(a.CAND)==a.CAND_SHA,'Frozen source bytes')
    def compress(raw):
        stream=io.BytesIO()
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,compresslevel=9,mtime=0) as f:f.write(raw)
        return stream.getvalue()
    compressed=compress(raw);again=compress(raw);a.require(compressed==again and gzip.decompress(compressed)==raw,'Deterministic lossless gzip roundtrip')
    destination=a.ROOT/'audits/custody/belief-gap-v3-adversarial-source.jsonl.gz';destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():a.require(destination.read_bytes()==compressed,'Refuse archive replacement')
    else:destination.write_bytes(compressed)
    custody={'actual_archive_author_execution_id':a.EXECUTION,'original_raw_path':str(a.CAND.relative_to(a.ROOT)),'raw_bytes':len(raw),'raw_sha256':hashlib.sha256(raw).hexdigest(),'gzip_path':str(destination.relative_to(a.ROOT)),'gzip_bytes':len(compressed),'gzip_sha256':a.sha(destination),'gzip_mtime':0,'gzip_original_filename':'','compression_level':9,'two_compressions_exact_byte_equal':True,'full_decoded_raw_byte_equal':True,'decoded_sha256':hashlib.sha256(gzip.decompress(compressed)).hexdigest(),'raw_source_preserved_in_place':True,'scope':'Own deterministic lossless custody copy only; no immutable source/release/other reviewer edits.'}
    a.freeze(a.ROOT/'audits/custody/belief-gap-v3-adversarial-source.manifest.jsonl',[custody])
    a.require(all(a.sha(a.OUT/p)==h for p,h in frozen.items()),'Frozen decisions/predictions unchanged')
    result={'reviewer_execution_id':a.EXECUTION,'batch':a.BATCH,'status':'completed','candidate_sha256':a.sha(a.CAND),'input_only_sha256':a.sha(a.INPUT),'script_sha256':a.sha(pathlib.Path(__file__)),'independent_reviewer_helper_sha256':a.sha(pathlib.Path(a.__file__)),'frozen_evidence_hashes':frozen,'executed_bad_semantic_mutants':mutants,'residual_marginal_count_minimal_counterexample':marginal,'oversized_source_custody':custody,'scope':'Fresh executed supplemental checks after first/main final judgments frozen; no producer/reference/oracle/checker/test/other verdict/diagnostic manifest imports. Same-model/common-public-design guided correction; no approval/publication/training/efficacy claim.'}
    output=a.ROOT/'audits/execution/belief-gap-v3-supplement-adversarial.json';a.save(output,result)
    print(a.canon({'supplement_sha256':a.sha(output),'script_sha256':a.sha(pathlib.Path(__file__)),'mutant_failures':{k:v['failed_tasks'] for k,v in mutants.items()},'custody':custody,'frozen_evidence_unchanged':True}),flush=True)

if __name__=='__main__':main()
