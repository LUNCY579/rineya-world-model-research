#!/usr/bin/env python3
"""Supplement immutable current receipts with exhaustive full goal-walk counts.

Goal is not absorbing: every layer records goal-ending sequences and continues
all legal walks. No producer/oracle/private expectation/reviewer files are read.
"""
import hashlib
import json
import pathlib
from collections import Counter,defaultdict
from adversarial_audit import ROOT,EXECUTION,canon,digest,graph,graph_validate,require


def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def full_counts(w):
    graph_validate(w);adj=defaultdict(list)
    for a,b,c in w['edges']:adj[a].append((b,c))
    minimum=min((c for a,b,c in w['edges']),default=1);bound=w['fuel']//minimum
    layer={(w['start'],w['fuel']):[()]};counts=[];witnesses=[]
    for hops in range(bound+1):
        ending=[p for (node,fuel),paths in layer.items() if node==w['goal'] for p in paths]
        counts.append(min(2,len(ending)))
        if len(ending)>1:
            pair=[list(p) for p in ending[:2]]
            require(pair[0]!=pair[1],'duplicate complete sequence')
            for path in pair:
                result=graph(dict(w,plan=path));require(result['valid'] and result['reaches_goal'],'walk witness not legal goal-ending')
            witnesses.append({'moves':hops,'distinct_complete_goal_sequences':pair})
        later=defaultdict(list)
        # Keep two distinct paths per equal (node,remaining fuel,number of moves).
        # All future suffixes are identical choices for this state. Capping at two
        # cannot hide a >=2 terminal count; it is sufficient for negative proofs.
        for (node,fuel),paths in layer.items():
            for dest,cost in adj[node]:
                if cost<=fuel:
                    state=(dest,fuel-cost)
                    for path in paths:
                        if len(later[state])<2:later[state].append(path+(dest,))
        layer=later
    return {'minimum_edge_cost':minimum,'complete_move_bound':bound,'complete_goal_sequence_counts_by_moves_capped_at_two':counts,'equal_length_witnesses':witnesses}


def probes():
    revisit={'kind':'graph','nodes':['s','g','a','b'],'edges':[['s','g',1],['g','a',1],['a','g',1],['g','b',1],['b','g',1]],'start':'s','goal':'g','fuel':3,'plan':[]}
    unique={'kind':'graph','nodes':['s','b','g'],'edges':[['s','b',1],['b','g',1],['g','b',1]],'start':'s','goal':'g','fuel':6,'plan':[]}
    initial=dict(revisit,start='g',fuel=2)
    r=full_counts(revisit);u=full_counts(unique);i=full_counts(initial)
    require(r['complete_goal_sequence_counts_by_moves_capped_at_two']==[0,1,0,2],'goal revisit tie must be detected')
    require(u['complete_goal_sequence_counts_by_moves_capped_at_two']==[0,0,1,0,1,0,1],'unique repeated goal cycle must remain unique')
    require(i['complete_goal_sequence_counts_by_moves_capped_at_two']==[1,0,2],'empty success and continuing goal departures')
    return [{'task':revisit,'counts':r,'failure_of_first_arrival_only':'First arrival has one length1 sequence, but two complete length3 sequences depart and revisit goal.'},{'task':unique,'counts':u},{'task':initial,'counts':i}]


def main():
    sources=[('expansion-v2','RVM-001'),('pilot-v4','RVM-001')];checked=[];cache={};filehashes={};receipt_hashes={};errors=[]
    out=ROOT/'reviews/adversarial';supplements=[]
    for batch,track in sources:
        path=ROOT/'data'/track/'candidate_renderings'/(batch+'.jsonl');receipt_path=out/(batch+'.jsonl')
        rows=load(path);receipts={r['sample_id']:r for r in load(receipt_path)};filehashes[str(path.relative_to(ROOT))]=sha(path);receipt_hashes[str(receipt_path.relative_to(ROOT))]=sha(receipt_path)
        archive=out/(batch+'.before-complete-goal-walk-audit.jsonl')
        if not archive.exists():archive.write_bytes(receipt_path.read_bytes())
        for r in rows:
            labels={tag.split(':',1)[-1] for tag in r['skill_tags']}
            if not labels & {'no_equal_length_goal_routes','equal_length_goal_routes'}:continue
            require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'current candidate hash mismatch')
            receipt=receipts[r['sample_id']];require(receipt['record_hash']==r['record_hash'],'receipt exact hash mismatch')
            w=r['semantic_task']['world'];wk=canon(w)
            if wk not in cache:cache[wk]=full_counts(w)
            proof=cache[wk];equal=any(v==2 for v in proof['complete_goal_sequence_counts_by_moves_capped_at_two'])
            expected='equal_length_goal_routes' in labels;ok=equal==expected
            item={'batch':batch,'sample_id':r['sample_id'],'record_hash':r['record_hash'],'existing_receipt_id':receipt['receipt_id'],'world_sha256':digest(w),'tag_has_equal_routes':expected,'contract_holds':ok,'complete_walk_proof':proof}
            checked.append(item)
            if not ok:errors.append(item)
            supplements.append({'receipt_id':'adv-full-walk-'+digest([r['sample_id'],r['record_hash'],EXECUTION])[:24],'sample_id':r['sample_id'],'record_hash':r['record_hash'],'reviewer_execution_id':EXECUTION,'reviewer_role':'Adversarial Red-Team supplementary evidence','supplements_receipt_id':receipt['receipt_id'],'existing_decision':receipt['decision'],'supplemental_result':'PASS' if ok else 'REVISE','public_contract_sha256':sha(ROOT/'schemas/COMPETENCY_TAGS.md'),'evidence':item})
        require(sha(receipt_path)==receipt_hashes[str(receipt_path.relative_to(ROOT))],'existing final receipts modified')
    report={'reviewer_execution_id':EXECUTION,'completed':True,'semantics':'All complete feasible goal-ending destination sequences, including departures and revisits. Goal is not absorbing.','method':'Exhaustive bounded layered dynamic path count, capped at two per node/fuel/move state. Counts 0,1,2 mean zero, exactly one, at least two distinct sequences.','records_checked':len(checked),'negative_records_checked':sum(not r['tag_has_equal_routes'] for r in checked),'distinct_worlds_checked':len(cache),'input_file_sha256':filehashes,'preserved_final_receipt_sha256':receipt_hashes,'public_contract_sha256':sha(ROOT/'schemas/COMPETENCY_TAGS.md'),'script_sha256':sha(pathlib.Path(__file__)),'all_tags_hold':not errors,'revision_required':bool(errors),'checks':checked,'errors':errors,'minimal_metaprobe_results':probes(),'private_reference_producer_or_other_reviewer_files_read':False}
    (ROOT/'audits/shortcut_detection/complete-goal-walks.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'complete-goal-walk-supplement.jsonl').write_text(''.join(canon(r)+'\n' for r in supplements))
    print(canon({k:report[k] for k in ('completed','records_checked','negative_records_checked','distinct_worlds_checked','all_tags_hold','revision_required','preserved_final_receipt_sha256')}))
    require(not errors,'full goal-walk factual tag correction required; existing decisions preserved pending a fresh revision')


if __name__=='__main__':main()
