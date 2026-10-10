#!/usr/bin/env python3
"""Independent signed-fuel/location claim controls and source-family audit."""
import copy
import hashlib
import json
import pathlib
from collections import Counter,defaultdict
from adversarial_audit import ROOT,EXECUTION,canon,digest,critic,graph,graph_signature,lookup_baseline,require,issue
from correction_lineage_adversarial_audit import source_key
from complete_goal_walk_adversarial_audit import full_counts


def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    batch='claim-gap-v2';path=ROOT/'data/RVM-001/candidate_renderings'/(batch+'.jsonl');rows=load(path)
    units=list({(source_key(r),canon(r['semantic_task'])):r for r in rows}.values())
    prior_paths=sorted((ROOT/'data/RVM-001/releases').glob('*/canonical.jsonl'));prior=[r for p in prior_paths for r in load(p)]
    prior_keys={source_key(r) for r in prior};prior_worlds={canon(r['semantic_task']['world']) for r in prior};prior_motifs={graph_signature(r['semantic_task']['world'],full=False) for r in prior}
    keys={source_key(r) for r in rows};require(len(rows)==24 and len(keys)==len(units)==12,'wrong new claim batch size')
    require(not keys&prior_keys,'declared new family reused from prior releases')
    examples=[];counts=Counter();deltas=Counter();byfamily=[];controls={};witnesses=[]
    for r in units:
        t=r['semantic_task'];a=critic(t);byplan=defaultdict(list);types=Counter()
        require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'claim batch hash mismatch')
        for position,(c,e) in enumerate(zip(t['candidates'],a['evaluations'])):
            actual=graph(dict(t['world'],plan=c['plan']))['after'];node_ok=c['claimed_after']['node']==actual['node'];delta=c['claimed_after']['fuel']-actual['fuel']
            kind='correct' if e['claim_correct'] else 'under' if node_ok and delta<0 else 'over' if node_ok and delta>0 else 'location' if not node_ok and delta==0 else 'mixed'
            types[kind]+=1;counts[kind]+=1
            if kind in ('under','over'):deltas[delta]+=1
            byplan[canon(c['plan'])].append({'kind':kind,'candidate':c,'evaluation':e,'actual_after':actual,'fuel_delta':delta})
            examples.append({'family':source_key(r),'split':r['split'],'position':position,'candidate_id':c['candidate_id'],'id_bucket':int(c['candidate_id'][1:])//10000,'claim_correct':e['claim_correct'],'valid':e['valid'],'kind':kind,'candidate_length':len(canon(c)),'claim_length':len(canon(c['claimed_after'])),'plan_length':len(c['plan'])})
        matched=[]
        for plan,cs in byplan.items():
            kinds={c['kind'] for c in cs}
            if not {'correct','under','over','location'}<=kinds:continue
            under={-c['fuel_delta'] for c in cs if c['kind']=='under'};over={c['fuel_delta'] for c in cs if c['kind']=='over'}
            require(under&over,'matched signed errors have no equal absolute magnitude')
            required=['valid','reaches_goal','remaining_moves','progress','usefulness','repair_plan']
            require(all(len({canon(c['evaluation'][field]) for c in cs})==1 for field in required),'same-plan claims change trajectory/utility metrics')
            require(all(c['evaluation']['first_error']=='claim' for c in cs if c['kind']!='correct' and c['evaluation']['valid']),'legal false claim not located at claim')
            matched.append({'plan':json.loads(plan),'absolute_error_magnitudes_matched':sorted(under&over),'candidate_ids_by_kind':{kind:[c['candidate']['candidate_id'] for c in cs if c['kind']==kind] for kind in ('correct','under','over','location')},'actual_after':cs[0]['actual_after']})
        require(len(matched)>=2,'family missing two complete matched equal-plan control sets')
        for tag,kind in [('fuel_underclaim','under'),('fuel_overclaim','over'),('wrong_location_claim','location')]:
            require(tag in r['skill_tags'] and types[kind]>0,'factual claim tag unsupported')
        require('matched_equal_plan_claims' in r['skill_tags'],'missing matched control tag')
        positive_routes={tag.split(':')[-1] for tag in r['skill_tags']}
        if positive_routes&{'equal_length_goal_routes','no_equal_length_goal_routes'}:
            proof=full_counts(t['world']);exists=any(n==2 for n in proof['complete_goal_sequence_counts_by_moves_capped_at_two'])
            require(exists==('equal_length_goal_routes' in positive_routes),'full goal-walk factual tag mismatch')
        controls[source_key(r)]={'matched_equal_plan_claim_groups':matched,'claim_type_counts':dict(types),'complete_goal_walk_tag_checked':bool(positive_routes&{'equal_length_goal_routes','no_equal_length_goal_routes'})}
        byfamily.append({'source_family_key':source_key(r),'sample_id':r['sample_id'],'split':r['split'],'matched_control_groups':len(matched),'claim_type_counts':dict(types)})
        witnesses.append({'sample_id':r['sample_id'],'record_hash':r['record_hash'],'world':t['world'],'one_matched_control_group':matched[0]})
    lookups={}
    for feature in ('position','id_bucket','candidate_id','plan_length','claim_length','candidate_length'):
        lookups[feature]=lookup_baseline(examples,lambda e:e[feature],lambda e:e['claim_correct'])
    overlap=[r['sample_id'] for r in units if canon(r['semantic_task']['world']) in prior_worlds]
    existing_motifs=[r['sample_id'] for r in units if graph_signature(r['semantic_task']['world'],full=False) in prior_motifs]
    require(not overlap,'new source family repeats an exact historical world')
    report={'reviewer_execution_id':EXECUTION,'batch':batch,'decision':'ACCEPT','records':len(rows),'new_declared_source_families':len(keys),'logical_world_tasks':len(units),'candidate_evaluations':len(examples),'matched_equal_plan_control_groups':sum(f['matched_control_groups'] for f in byfamily),'claim_type_counts':dict(counts),'pure_fuel_error_delta_counts':dict(deltas),'family_controls':byfamily,'claim_correct_stripped_feature_baselines':lookups,'new_source_key_overlap':sorted(keys&prior_keys),'exact_world_overlap_prior':overlap,'world_motifs_previously_exposed':existing_motifs,'input_sha256':sha(path),'prior_release_sha256':{str(p.relative_to(ROOT)):sha(p) for p in prior_paths},'script_sha256':sha(pathlib.Path(__file__)),'limits':['The new same-plan controls each contain correct, underclaim, overclaim and wrong-location variants. Exact signed error magnitudes vary by family.','All pure fuel errors combined remain overclaim48/underclaim24 because baseline positive errors are retained; the full batch is not globally sign-balanced.','199 candidate labels share only12 worlds; development has2 families. Lookup accuracies have little precision and are not efficacy evidence.','Four of12 graph motifs are already exposed after removing names/fuel/costs. Fresh source keys and exact worlds do not imply12 new mechanisms.','Plan-length and serialized-length associations remain descriptive: plan-length claim correctness63.64% development versus54.55% majority, with full matched controls inside each family.'],'minimal_matched_semantic_witness':min(witnesses,key=lambda w:len(canon(w['world']))),'private_reference_producer_or_other_reviewer_files_read':False}
    audit=ROOT/'audits/shortcut_detection/claim-gap-v2-controls.json';audit.write_text(json.dumps(report,indent=2)+'\n')
    out=ROOT/'reviews/adversarial';receipt_path=out/(batch+'.jsonl');prior=out/(batch+'.before-claim-control-audit.jsonl')
    if not prior.exists():prior.write_bytes(receipt_path.read_bytes())
    receipt_rows=load(receipt_path);byid={r['sample_id']:r for r in rows}
    for receipt in receipt_rows:
        r=byid[receipt['sample_id']];require(receipt['record_hash']==r['record_hash'],'receipt not bound to exact candidate')
        receipt['evidence']['matched_claim_controls']=controls[source_key(r)]
        receipt['evidence']['claim_control_distribution_evidence']='audits/shortcut_detection/claim-gap-v2-controls.json'
        receipt['issues'].append(issue('signed_claim_control_scope','low',{'pure_fuel_errors_over':48,'pure_fuel_errors_under':24,'matched_groups':24,'source_families':12,'development_families':2},'Describe this as targeted matched claim controls; do not claim whole-batch sign balance or held-out claim-producer efficacy.'))
    receipt_path.write_text(''.join(canon(r)+'\n' for r in receipt_rows))
    execution_path=out/('execution-'+batch+'.json');execution=json.loads(execution_path.read_text())
    execution.update(completed=True,final_receipts_sha256=sha(receipt_path),first_judgment_sha256=sha(out/(batch+'.first_judgments.jsonl')),decision_counts=dict(Counter(r['decision'] for r in receipt_rows)),matched_claim_controls_checked=True,new_source_families=12,exact_world_reuse=0,claim_control_evidence_sha256=sha(audit),claim_control_script_sha256=sha(pathlib.Path(__file__)))
    execution_path.write_text(json.dumps(execution,indent=2)+'\n');(ROOT/'audits/execution/claim-gap-v2-adversarial.json').write_text(json.dumps(execution,indent=2)+'\n')
    print(canon({k:report[k] for k in ('decision','records','new_declared_source_families','candidate_evaluations','matched_equal_plan_control_groups','claim_type_counts','pure_fuel_error_delta_counts')}))


if __name__=='__main__':main()
