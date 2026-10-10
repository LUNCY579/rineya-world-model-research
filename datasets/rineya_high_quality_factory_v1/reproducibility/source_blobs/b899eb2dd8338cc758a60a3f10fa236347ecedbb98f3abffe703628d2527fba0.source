#!/usr/bin/env python3
"""Bind fresh correction receipts to unchanged predecessor tasks and inputs.

Reads candidates, immutable semantic canonical records, and own receipts only.
No producer, reference, independent-oracle or other reviewer imports/files.
"""
import hashlib
import json
import pathlib
import re
from collections import Counter
from adversarial_audit import ROOT,EXECUTION,SCORE_KEYS,canon,digest,require


def load(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_key(r):
    if r.get('source_family_key'):return r['source_family_key']
    i=int(re.search(r'-p(\d+)$',r['semantic_family_id']).group(1))
    return f"{r['track_id']}-seed{r['generation_seed']}-p{i:04d}"
def key(r):return (source_key(r),r['renderer'],r['target_format'],canon(r['semantic_task']))


def main():
    cases=[('expansion-v2','RVM-001','data/RVM-001/candidate_renderings/expansion-v1.jsonl',300),
           ('expansion-v3','RGM-001','data/RGM-001/candidate_renderings/expansion-v1.jsonl',360),
           ('pilot-v4','RVM-001','data/RVM-001/releases/v0001/canonical.jsonl',60)]
    completed=[];out=ROOT/'reviews/adversarial';audit=ROOT/'audits/execution';audit.mkdir(exist_ok=True)
    for batch,track,old_relative,n in cases:
        candidate=ROOT/'data'/track/'candidate_renderings'/(batch+'.jsonl');old_path=ROOT/old_relative
        rows=load(candidate);old_rows=load(old_path);old_map={key(r):r for r in old_rows}
        receipts=load(out/(batch+'.jsonl'));first=out/(batch+'.first_judgments.jsonl')
        require(len(rows)==len(receipts)==n,'wrong review coverage')
        require(len({r['sample_id'] for r in rows})==n,'duplicate sample identity')
        require(len({r['receipt_id'] for r in receipts})==n,'duplicate receipt identity')
        receipt_index={r['sample_id']:r for r in receipts};parents=[]
        require(set(receipt_index)=={r['sample_id'] for r in rows},'missing or extra final receipt')
        for r in rows:
            require(digest({k:v for k,v in r.items() if k!='record_hash'})==r['record_hash'],'candidate hash mismatch')
            require(key(r) in old_map,'correction introduces new semantic variant')
            old=old_map[key(r)];receipt=receipt_index[r['sample_id']]
            require(old['rendered_input']==r['rendered_input'],'correction changes prompt input')
            require(canon(old['verified_target'])==canon(r['verified_target']),'correction changes target')
            require(old['split']==r['split'],'correction changes family split')
            require(old['record_hash']!=r['record_hash'],'correction is not a fresh complete-record hash')
            require(receipt['record_hash']==r['record_hash'] and receipt['reviewer_execution_id']==EXECUTION,'receipt binding mismatch')
            require(set(receipt['scores'])==set(SCORE_KEYS),'receipt score schema mismatch')
            require(receipt['evidence']['independent_public_contract_recomputation'],'no independent recomputation evidence')
            require(receipt['evidence']['metamorphic_tests'],'no metamorphic checks')
            require(receipt['evidence']['mechanism_tag_audit'],'no factual tag check')
            link={'predecessor_sample_id':old['sample_id'],'predecessor_record_hash':old['record_hash'],'source_family_key':source_key(r),'input_and_target_unchanged':True,'new_independent_family':False}
            receipt['evidence']['correction_lineage']=link
            parents.append(dict(sample_id=r['sample_id'],record_hash=r['record_hash'],**link))
        (out/(batch+'.jsonl')).write_text(''.join(canon(r)+'\n' for r in receipts))
        # The same finalized receipts are available under the supplemental name.
        (out/(batch+'.mechanism-tags.jsonl')).write_text(''.join(canon(r)+'\n' for r in receipts))
        execution_path=out/('execution-'+batch+'.json');execution=json.loads(execution_path.read_text())
        execution.update(decision_counts=dict(Counter(r['decision'] for r in receipts)),completed=True,release_selected_track=track,record_count=n,family_count=len({source_key(r) for r in rows}),new_independent_families=0,first_judgment_sha256=sha(first),final_receipts_sha256=sha(out/(batch+'.jsonl')),source_files_sha256={str(candidate.relative_to(ROOT)):sha(candidate),str(old_path.relative_to(ROOT)):sha(old_path)},correction_lineage=parents,mechanism_tags_independently_checked=True,correction_lineage_script_sha256=sha(pathlib.Path(__file__)),mechanism_tag_script_sha256=sha(ROOT/'audits/mechanism_tag_adversarial_audit.py'),private_reference_producer_or_other_reviewer_files_read=False,scope=['These fresh hashes correct factual metadata only. Inputs, targets, family custody, and splits are unchanged.','The first judgments for original and correction batches remain immutable. Prior tag verdicts and errata are retained.','Prior tag-defect challenge topics were disclosed by root; factual corrections were checked independently.','No new-family, new-mechanism, sealed-evaluation or efficacy claim is created.'])
        execution_path.write_text(json.dumps(execution,indent=2)+'\n')
        (audit/(batch+'-adversarial.json')).write_text(json.dumps(execution,indent=2)+'\n')
        completed.append({k:execution[k] for k in ('release_selected_track','record_count','family_count','new_independent_families','decision_counts','first_judgment_sha256','final_receipts_sha256')})
    print(canon({'completed_correction_reviews':completed}))


if __name__=='__main__':main()
