#!/usr/bin/env python3
"""Execute exporter logic with independent JSON/hash helpers and public-contract solvers.

No generator/reference/reviewer/oracle imports; no release writes or dataset approval.
"""
import ast
import copy
import hashlib
import json
import pathlib
from collections import Counter,defaultdict
import jsonschema
from adversarial_audit import ROOT,EXECUTION,canon,digest,critic,solve,require


def load_exporter():
    path=ROOT/'factory'/'exports.py';source=path.read_text();tree=ast.parse(source)
    # The imported canonical/digest/write_jsonl helpers are replaced by independent,
    # deterministic equivalents; the write helper is disabled. All export logic is unchanged.
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='factory.common')]
    namespace={'canonical':canon,'digest':digest,'write_jsonl':lambda *args:None}
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace['specialized'],hashlib.sha256(source.encode()).hexdigest()


def semantic_pair(pair):
    return canon([{k:v for k,v in json.loads(pair[field]).items() if k!='candidate_id'} for field in ('chosen','rejected')])


def prompt_payload(prompt):
    return json.loads(prompt.split('World and response options JSON:\n',1)[1])


def main(batch='expansion-v1'):
    exporter,source_hash=load_exporter()
    preference_schema_path=ROOT/'schemas/preference/v1.schema.json';process_schema_path=ROOT/'schemas/process_reward/v1.schema.json'
    preference_validator=jsonschema.Draft202012Validator(json.loads(preference_schema_path.read_text()))
    process_validator=jsonschema.Draft202012Validator(json.loads(process_schema_path.read_text()))
    paths=sorted((ROOT/'data').glob('*/candidate_renderings/'+batch+'.jsonl'))
    rows=[json.loads(line) for p in paths for line in p.read_text().splitlines() if line.strip()]
    rv=[r for r in rows if r['track_id']=='RVM-001'];rg=[r for r in rows if r['track_id']=='RGM-001']
    checked=Counter();pair_stats=[];errors=[];counterexamples=[]
    source_answers={r['sample_id']:solve(r['semantic_task']) for r in rv+rg}
    by_input={r['rendered_input']:r for r in rv+rg}
    for split in ('train','dev_challenge'):
        data=exporter(rv,split)
        for pair in data.get(split+'-preference.jsonl',[]):
            preference_validator.validate(pair)
            payload=prompt_payload(pair['prompt']);chosen=json.loads(pair['chosen']);rejected=json.loads(pair['rejected'])
            require(set(chosen)==set(rejected)=={'candidate_id','plan','claimed_after'},'preference response shape')
            require(chosen in payload['options'] and rejected in payload['options'] and len(payload['options'])==2,'options mismatch')
            answer=critic({'kind':'critic','world':payload['world'],'candidates':payload['options']})
            require(answer['ranking']==[[chosen['candidate_id']],[rejected['candidate_id']]],'pair not a strict independent preference')
            require(all(token not in pair['prompt'] for token in ('semantic_parent_id','semantic_family_id','review_receipt_ids','source_provenance','record_hash','generation_seed','review_status')),'privileged metadata in preference prompt')
            d=len(chosen['plan'])-len(rejected['plan']);bucket=0 if d==0 else 1 if d>0 else -1
            pair_stats.append({'split':split,'length_bucket':bucket,'lower_ID_chosen':chosen['candidate_id']<rejected['candidate_id'],
                               'chosen_position':payload['options'].index(chosen),'string_length_sign':(len(pair['chosen'])>len(pair['rejected']))-(len(pair['chosen'])<len(pair['rejected']))})
            checked['strict_preference_pairs']+=1
        for row in data.get(split+'-process_reward.jsonl',[]):
            process_validator.validate(row)
            source=by_input[row['input']];answer=source_answers[source['sample_id']]
            expected=next(e for e in answer['evaluations'] if e['candidate_id']==row['candidate']['candidate_id'])
            require(canon(row['labels'])==canon(expected),'candidate-level labels disagree with independent solver')
            require(row['candidate'] in source['semantic_task']['candidates'],'process candidate not submitted')
            require(row['candidate']['candidate_id']==row['labels']['candidate_id'],'candidate/label ID mismatch')
            checked['candidate_process_labels']+=1
        data=exporter(rg,split)
        for row in data.get(split+'-trajectories.jsonl',[]):
            source=by_input[row['input']];answer=source_answers[source['sample_id']]
            require(answer['valid'] and answer['reaches_goal'],'exported invalid/non-goal trajectory')
            require(canon(row['actions'])==canon(source['semantic_task']['plan']) and canon(row['steps'])==canon(answer['steps']) and canon(row['after'])==canon(answer['after']),'trajectory replay mismatch')
            checked['valid_goal_trajectories']+=1
    # Renaming may change which identifier represents semantically identical tied copies,
    # so invariance is asserted on selected plans and claims, with identifiers stripped.
    for r in rv:
        original=exporter([r],r['split'])[r['split']+'-preference.jsonl']
        renamed=copy.deepcopy(r);candidates=renamed['semantic_task']['candidates']
        old_ids=[c['candidate_id'] for c in candidates]
        mapping={old:'opaque_%04d'%(len(old_ids)-i) for i,old in enumerate(old_ids)}
        for c in candidates:c['candidate_id']=mapping[c['candidate_id']]
        renamed['verified_target']=critic(renamed['semantic_task'])
        output=exporter([renamed],r['split'])[r['split']+'-preference.jsonl']
        if [semantic_pair(p) for p in original]!=[semantic_pair(p) for p in output]:
            errors.append({'code':'candidate_ID_changes_semantic_pair_selection','sample_id':r['sample_id']})
        checked['candidate_ID_renaming_semantic_invariance']+=1
        reordered=copy.deepcopy(r);reordered['semantic_task']['candidates'].reverse();reordered['verified_target']=critic(reordered['semantic_task'])
        output=exporter([reordered],r['split'])[r['split']+'-preference.jsonl']
        require([semantic_pair(p) for p in original]==[semantic_pair(p) for p in output],'candidate input order changes pair semantics')
        checked['candidate_order_semantic_invariance']+=1
    # Mutation witnesses ensure the strengthened schema rejects bad types/missing fields.
    if rv:
        sample=exporter(rv,rv[0]['split'])[rv[0]['split']+'-process_reward.jsonl'][0]
        mutations=[]
        for field,value in [('valid','yes'),('claim_correct',42),('progress','huge'),('usefulness',False),('first_error',-1),('repair_plan',3)]:
            bad=copy.deepcopy(sample);bad['labels'][field]=value;mutations.append((field,bad))
        bad=copy.deepcopy(sample);bad['candidate']={};mutations.append(('empty_candidate',bad))
        bad=copy.deepcopy(sample);bad['candidate']['claimed_after']['fuel']=-1;mutations.append(('negative_claimed_fuel',bad))
        for name,bad in mutations:
            rejected=bool(list(process_validator.iter_errors(bad)))
            counterexamples.append({'mutation':name,'invalid_payload':bad,'schema_rejects':rejected})
            if not rejected:errors.append({'code':'schema_accepts_invalid_types','mutation':name})
            checked['schema_mutation_witnesses']+=1
    stats={'pairs':len(pair_stats),'lower_ID_chosen':sum(p['lower_ID_chosen'] for p in pair_stats),
           'chosen_positions':dict(Counter(p['chosen_position'] for p in pair_stats)),
           'plan_length_buckets':dict(Counter(p['length_bucket'] for p in pair_stats)),
           'response_string_length_signs':dict(Counter(p['string_length_sign'] for p in pair_stats)),
           'by_split':{},'by_length_bucket':{}}
    for field,label in [('split','by_split'),('length_bucket','by_length_bucket')]:
        for key in sorted({p[field] for p in pair_stats}):
            ps=[p for p in pair_stats if p[field]==key]
            stats[label][str(key)]={'pairs':len(ps),'lower_ID_chosen':sum(p['lower_ID_chosen'] for p in ps),'lower_ID_choice_rate':sum(p['lower_ID_chosen'] for p in ps)/len(ps)}
    evidence={'reviewer_role':'Adversarial Red-Team code review','reviewer_execution_id':EXECUTION,'decision':'REVISE' if errors else 'ACCEPT_CODE',
              'dataset_acceptance_granted':False,'batch_used_for_export_verification':batch,'source_sha256':source_hash,
              'schema_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (preference_schema_path,process_schema_path)},
              'checks':dict(checked),'preference_shortcut_counts':stats,'issues':errors,
              'evidence_source':'Own public-contract critic/graph/block solvers and metamorphic tests; original exporter executed with independent canonical JSON/SHA256 callbacks and disabled writes.',
              'scope':['Pairs are relative strict preferences. A chosen candidate may still have an incorrect claim or illegal trajectory if it ranks above the rejected one.',
                       'Process labels supervise whole candidates, not individual steps. first_error and repair_plan retain the public candidate-level definitions.',
                       'Trajectories are submitted valid goal-reaching plans; they need not minimize moves.',
                       'ID-renaming invariance is measured on semantic pair content because duplicated equal-quality candidates may change representative IDs.',
                       'Statistical pair counts share source families and renderings; they are not independent trials or evidence of efficacy.',
                       'No expansion dataset ACCEPT is created by this code review.'],
              'private_reference_or_reviewer_verdicts_read':False}
    out=ROOT/'reviews'/'adversarial';out.mkdir(parents=True,exist_ok=True)
    (out/'exports-code-review.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (out/('exports-code-review-'+source_hash[:12]+'.json')).write_text(json.dumps(evidence,indent=2)+'\n')
    counterpath=ROOT/'audits'/'shortcut_detection'/('export-counterexamples-'+source_hash[:12]+'.json')
    counterpath.write_text(json.dumps(counterexamples,indent=2)+'\n')
    print(canon({'source_hash':source_hash,'decision':evidence['decision'],'checks':dict(checked),'preference_counts':stats,'errors':errors}))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--batch',default='expansion-v1');main(parser.parse_args().batch)
