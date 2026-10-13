import json
C=json.load(open('c_verify_result.json'))
A=json.load(open('audit_results.json'))
B=json.load(open('audit_b_result3.json'))
amap={'v2_actions':'v2_proposer_actions_train','distill_rows':'distill_rows','mix_v3':'mix_v3','mix_v2c':'mix_v2c','v1_actions':'v1_proposer_actions_train','superseded_v1_actions':'superseded_v1_proposer_actions_train'}
bmap={'v2_actions':'runs/BW_data_v2/proposer_actions_train.jsonl','v2_states':'runs/BW_data_v2/proposer_states_train.jsonl','distill_rows':'runs/BW_DISTILL_r1/distill_rows.jsonl','mix_v3':'runs/BW_DISTILL_r1/mix_v3.jsonl','mix_v2c':'runs/BW_DISTILL_r1/mix_v2c.jsonl','v1_actions':'runs/BW_data/proposer_actions_train.jsonl','v1_states':'runs/BW_data/proposer_states_train.jsonl','superseded_v1_actions':'runs/_superseded/BW_data_v1_suffix_leak/proposer_actions_train.jsonl'}
aset={'PB602':'planbench_PB602','hard':'hard','mirror':'mirror','b3fresh':'b3fresh','val_large':'val_large','seeds_fresh':'seeds_fresh','val_v2':'val_v2','val_v1':'val_v1','distill240':'distill240'}
bset={'PB602':'PB-602','hard':'hard','mirror':'mirror','b3fresh':'b3fresh','val_large':'val_large','seeds_fresh':'seeds_fresh','val_v2':'val_v2(select)','val_v1':'val_v1','distill240':'distill-240','unsolvable101_extra':'unsolvable-101(extra)'}
print('nonforest', C['eval_nonforest_instances'], 'rcx', C['rcx_e07_trainval']['n'], C['rcx_e07_trainval']['splits'], 'hits', C['rcx_e07_trainval']['full_hits'])
print({k:v for k,v in C.items() if k.startswith('user_json')})
for f,rec in C['files'].items():
    print('==',f, rec['rows'], rec['row_kinds'], rec['parse_errors'])
    for s in bset:
        c=rec['full'].get(s)
        cids=set(c['instance_ids']) if c else set()
        # A
        a=None
        if f in amap and s in aset:
            k=f"{aset[s]} x {amap[f]}"
            if k in A['pairs']:
                ap=A['pairs'][k]; a=(set(ap.get('full_ids',[])), ap.get('full_matching_rows'), ap.get('full_contaminated_instances_problem_level'))
        b=None
        bm=B['files'].get(bmap.get(f),{}).get('matches',{}).get(bset[s],{}).get('full')
        if bm is not None: b=(set(bm['all_matched_instances']), bm['row_instance_pairs'], bm['instances_by_rowkind'].get('start(t=0)|outcome=success',0) if 'instances_by_rowkind' in bm else None)
        elif bmap.get(f) in B['files']: b=(set(),0,0)
        line=f"  {s:20s} C inst={len(cids):4d} pairs={c['pairs'] if c else 0:6d} rows={c['rows'] if c else 0:5d} t0inst={c['instances_with_t0_row'] if c else 0:4d}"
        if a is not None: line+=f" | A ids_equal={a[0]==cids} pairs={a[1]} t0={a[2]}"
        if b is not None: line+=f" | B ids_equal={b[0]==cids} pairs={b[1]} t0={b[2]}"
        if c or (a and a[0]) or (b and b[0]): print(line)
