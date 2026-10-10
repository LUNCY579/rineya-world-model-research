#!/usr/bin/env python3
"""Executed incomplete-resolver mutants and minimal public-contract typed probes.
Supplement only: preserve frozen data predictions and every semantic receipt.
"""
import collections,copy,json,pathlib,hashlib
from belief_gap_adversarial_audit import ROOT,EXECUTION,BATCH,INPUT,OUT,solve,canon,sha,load,require,save,answer,typed

def wrong(t,mode):
    worlds=t['worlds'];remaining=worlds[:];sizes=[]
    for o in t['observations']:
        if mode=='coercing_filter':remaining=[w for w in remaining if w['facts'][o['fact']]==o['equals']]
        elif mode=='reset_filter_each_observation':remaining=[w for w in worlds if typed(w['facts'][o['fact']])==typed(o['equals'])]
        else:remaining=[w for w in remaining if typed(w['facts'][o['fact']])==typed(o['equals'])]
        sizes.append(len(remaining))
    qs=[]
    for q in t['queries']:
        if mode=='coercing_partitions':
            counts=collections.Counter(w['facts'][q['fact']] for w in remaining);bins=sorted([{'value':v,'count':n} for v,n in counts.items()],key=lambda b:typed(b['value']))
        else:
            counts=collections.Counter(typed(w['facts'][q['fact']]) for w in remaining);bins=[{'value':json.loads(v),'count':counts[v]} for v in sorted(counts)]
        if mode=='drop_multiplicity':
            for b in bins:b['count']=1
        a=answer(q,bins,len(remaining))
        if mode=='null_means_not_entailed' and a['guaranteed'] is None:a['entailed']=False
        qs.append(a)
    return {'surviving_worlds':sorted(w['world_id'] for w in remaining),'belief_sizes':sizes,'queries':qs}

def main():
    frozen=[OUT/f'{BATCH}.{s}' for s in ('input_predictions.jsonl','first_judgments.jsonl','jsonl')];bound={str(p.relative_to(ROOT)):sha(p) for p in frozen};rows=load(INPUT);pred={r['sample_id']:r for r in load(frozen[0])};modes=['coercing_filter','coercing_partitions','reset_filter_each_observation','drop_multiplicity','null_means_not_entailed'];metrics={m:collections.Counter() for m in modes};examples={}
    train_empty=set();train_nonempty=set()
    for r in rows:
        a=pred[r['sample_id']]['expected_target'];t=r['semantic_task']
        if r['split']=='train':
            tokens={o['equals'] for o in t['observations'] if type(o['equals']) is str};(train_empty if not a['surviving_worlds'] else train_nonempty).update(tokens)
        for m in modes:
            observed=wrong(t,m);bad=canon(observed)!=canon(a);metrics[m]['tasks']+=1;metrics[m]['complete_target_failures']+=bad
            if bad and m not in examples:examples[m]={'sample_id':r['sample_id'],'record_hash':r['record_hash'],'correct':a,'incorrect_mutant':observed,'input':t}
    learned=train_empty-train_nonempty;require(learned=={'stone','unknown'},'Marker counterexample must be derived from TRAIN only')
    train_marker_evidence={'train_empty_observation_string_values':sorted(train_empty),'train_nonempty_observation_string_values':sorted(train_nonempty),'empty_only_marker_values':sorted(learned),'classification_results':{s:{'records':sum(r['split']==s for r in rows),'correct':sum(any(type(o['equals']) is str and o['equals'] in learned for o in r['semantic_task']['observations'])==(not pred[r['sample_id']]['expected_target']['surviving_worlds']) for r in rows if r['split']==s)} for s in ('train','dev_challenge')}}
    probes=[]
    for exact,alias in [(True,1),(False,0)]:
        t={'kind':'belief','worlds':[{'world_id':'a','facts':{'f':exact}},{'world_id':'b','facts':{'f':alias}}],'observations':[],'queries':[{'query_id':'q','fact':'f'}]};a,d=solve(t);require(a['queries'][0]['distinct_values']==2 and not a['queries'][0]['entailed'],'typed bins must stay distinct');probes.append({'name':'typed_query_partition_'+typed(exact),'scope':'audit_probe','task':t,'correct':a,'coercing_wrong':wrong(t,'coercing_partitions')})
        z=copy.deepcopy(t);z['observations']=[{'fact':'f','equals':exact}];b,bd=solve(z);require(b['surviving_worlds']==['a'] and b['belief_sizes']==[1],'typed filter active separator');probes.append({'name':'typed_observation_filter_'+typed(exact),'scope':'audit_probe','task':z,'correct':b,'coercing_wrong':wrong(z,'coercing_filter')})
    nullt={'kind':'belief','worlds':[{'world_id':'a','facts':{'f':None,'g':0}},{'world_id':'b','facts':{'f':None,'g':1}}],'observations':[],'queries':[{'query_id':'q','fact':'f'}]};a,d=solve(nullt);require(a['queries'][0]['entailed'] and a['queries'][0]['guaranteed'] is None and a['queries'][0]['counts']==[{'value':None,'count':2}],'multiworld entailed null');probes.append({'name':'multiworld_entailed_null','scope':'audit_probe','task':nullt,'correct':a})
    ambiguous=copy.deepcopy(nullt);ambiguous['worlds'][1]['facts']['f']=0;a,d=solve(ambiguous);require(not a['queries'][0]['entailed'] and a['queries'][0]['guaranteed'] is None and a['queries'][0]['distinct_values']==2,'ambiguous null');probes.append({'name':'ambiguous_null','scope':'audit_probe','task':ambiguous,'correct':a})
    empty=copy.deepcopy(ambiguous);empty['observations']=[{'fact':'f','equals':'unsupported'}];a,d=solve(empty);require(not a['surviving_worlds'] and not a['queries'][0]['entailed'] and a['queries'][0]['guaranteed'] is None and a['queries'][0]['counts']==[],'empty belief null');probes.append({'name':'empty_null_guarantee','scope':'audit_probe','task':empty,'correct':a})
    literal=copy.deepcopy(nullt)
    for w in literal['worlds']:w['facts']['f']='unknown'
    a,d=solve(literal);require(a['queries'][0]['entailed'] and a['queries'][0]['guaranteed']=='unknown','literal unknown is ordinary scalar');probes.append({'name':'literal_unknown_ordinary_scalar','scope':'audit_probe','task':literal,'correct':a})
    contradiction={'kind':'belief','worlds':[{'world_id':'a','facts':{'x':0,'y':0}},{'world_id':'b','facts':{'x':1,'y':1}}],'observations':[{'fact':'x','equals':0},{'fact':'y','equals':1},{'fact':'x','equals':1}],'queries':[{'query_id':'q','fact':'x'}]};a,d=solve(contradiction);require(a['belief_sizes']==[1,0,0] and all(d['individual_observation_supports']),'individually supported cross-fact contradiction and absorption');probes.append({'name':'cross_fact_contradiction_absorption','scope':'audit_probe','task':contradiction,'correct':a,'reset_wrong':wrong(contradiction,'reset_filter_each_observation')})
    repeat=copy.deepcopy(contradiction);repeat['observations']=[repeat['observations'][0],repeat['observations'][0]];a,d=solve(repeat);require(a['belief_sizes']==[1,1] and a['surviving_worlds']==['a'],'nonempty repeated observation');probes.append({'name':'nonempty_repetition','scope':'audit_probe','task':repeat,'correct':a})
    # Valid positive lexical counterexample, complementing actual empty scalar-bijection witnesses.
    stone=copy.deepcopy(nullt)
    for w in stone['worlds']:w['facts'].update(x='stone',y='linen')
    stone['observations']=[{'fact':'x','equals':'stone'},{'fact':'y','equals':'linen'}];a,d=solve(stone);require(a['surviving_worlds']==['a','b'],'token marker cannot define absence');probes.append({'name':'stone_observation_jointly_supported','scope':'audit_probe','task':stone,'correct':a,'marker_wrong':'empty belief'})
    malformed=[]
    for name,modify in [('float_fact',lambda t:t['worlds'][0]['facts'].update(f=1.0)),('float_observation',lambda t:t['observations'].append({'fact':'f','equals':1.0})),('duplicate_world_id',lambda t:t['worlds'][1].update(world_id='a')),('unknown_query_fact',lambda t:t['queries'][0].update(fact='missing'))]:
        t=copy.deepcopy(nullt);modify(t)
        try:solve(t)
        except ValueError:malformed.append(name)
        else:raise ValueError('malformed probe accepted '+name)
    require({str(p.relative_to(ROOT)):sha(p) for p in frozen}==bound,'Frozen actual judgments changed')
    report={'reviewer_execution_id':EXECUTION,'batch':BATCH,'status':'completed','script_sha256':sha(pathlib.Path(__file__)),'frozen_judgments_sha256':bound,'stripped_input_sha256':sha(INPUT),'own_solver_script_sha256':sha(ROOT/'audits/belief_gap_adversarial_audit.py'),'incomplete_resolver_mutants':{m:dict(c) for m,c in metrics.items()},'mutant_actual_failure_examples':examples,'train_only_lexical_marker_derivation':train_marker_evidence,'minimal_public_contract_diagnostic_probes':probes,'malformed_rejections':malformed,'diagnostic_scope':'Executed audit probes add no supplied records, contexts or source families. No producer/reference/oracle/checker/tests or reviewer verdicts read. Frozen input expectations/first/final judgments unchanged.','same_model_common_public_design':True}
    p=ROOT/'audits/execution'/f'{BATCH}-mutants-adversarial.json';save(p,report);print(canon({'report_sha256':sha(p),'mutants':report['incomplete_resolver_mutants'],'minimal_typed_probes':len(probes),'malformed_rejections':len(malformed),'marker':train_marker_evidence}))

if __name__=='__main__':main()
