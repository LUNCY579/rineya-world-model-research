import json, sys
sys.dont_write_bytecode = True
RUN = 'C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/runs/retention/confirm_1708484032065101819'
S = 'C:/Users/clclc/AppData/Local/Temp/claude/bwm_5c_scratch/'
R = json.load(open(RUN + '/analysis/RET_RESULTS.json', encoding='utf-8'))
M = json.load(open(S + 'my_results.json', encoding='utf-8'))
mism, ncmp = [], 0


def cmp(path, a, b, exact=True):
    global ncmp
    ncmp += 1
    if isinstance(a, float) or isinstance(b, float):
        ok = (a == b) if exact else abs(a - b) <= 1e-12 * max(1.0, abs(a))
        if not ok:
            mism.append((path, a, b))
            return
    elif a != b:
        mism.append((path, a, b))


cmp('categories', R['categories'], M['categories'])
for s, blk in R['accuracy'].items():
    for c, v in blk.items():
        if c.startswith('_'):
            if isinstance(v, dict):
                for k, x in v.items():
                    cmp(f'accuracy.{s}.{c}.{k}', x, M['accuracy'][s][c][k])
            else:
                cmp(f'accuracy.{s}.{c}', v, M['accuracy'][s][c])
        else:
            for k, x in v.items():
                cmp(f'accuracy.{s}.{c}.{k}', x, M['accuracy'][s][c][k])
for name, rc in R['comparisons'].items():
    mc = M['comparisons'][name]
    for k in ('arm', 'reference', 'gating', 'verdict', 'go', 'go_label'):
        cmp(f'{name}.{k}', rc[k], mc[k])
    cmp(f'{name}.binding.k', rc['binding']['k'], mc['binding_k'])
    cmp(f'{name}.binding.m', rc['binding']['m'], mc['binding_m'])
    cmp(f'{name}.binding.min_lost_to_fail', rc['binding']['min_lost_to_fail'], M['min_lost_fail_alone'])
    cmp(f'{name}.binding.categories', rc['binding']['categories'],
        [c for c in M['categories'] if mc['graders']['primary']['categories'][c]['binding']])
    for g in ('primary', 'lenient'):
        cmp(f'{name}.flags.{g}', rc['flags_net_below_minus5'][g], mc['flags'][g])
        rg, mg = rc['graders'][g], mc['graders'][g]
        for k in ('detected', 'category_margin_exceeded', 'margin_exceeded'):
            cmp(f'{name}.{g}.{k}', rg[k], mg[k])
        for k, x in rg['pooled'].items():
            cmp(f'{name}.{g}.pooled.{k}', x, mg['pooled'][k])
        for c, rr in rg['categories'].items():
            mm = mg['categories'][c]
            for k, x in rr.items():
                if k == 'binding':
                    cmp(f'{name}.{g}.{c}.binding.binding', x['binding'], mm['binding'])
                    cmp(f'{name}.{g}.{c}.binding.min_lost_to_fail', x['min_lost_to_fail'], M['min_lost_fail_alone'])
                elif k == 'fails':
                    cmp(f'{name}.{g}.{c}.fails', x, mm['detected_loss'] or mm['margin_exceeded'])
                else:
                    cmp(f'{name}.{g}.{c}.{k}', x, mm[k])
            extra = set(rr) - {'n', 'ref_correct', 'arm_correct', 'lost', 'gained', 'net', 'discordant', 'p', 'p_holm',
                               'detected_loss', 'margin_exceeded', 'fails', 'flag_net_below_minus5', 'binding'}
            if extra:
                print('UNCOMPARED keys', name, g, c, extra)
    # KL block: only the counts are checkable without the lp vectors
    kl = rc['kl']
    pp = mc['graders']['primary']
    cmp(f'{name}.kl.n_items', kl['n_items'], sum(pp['categories'][c]['n'] for c in M['categories']))
    cmp(f'{name}.kl.auroc_flip.n_pos', kl['auroc_flip']['n_pos'], pp['pooled']['lost'] + pp['pooled']['gained'])
    cmp(f'{name}.kl.auroc_loss.n', kl['auroc_loss_among_ref_correct']['n'],
        sum(pp['categories'][c]['ref_correct'] for c in M['categories']))
    cmp(f'{name}.kl.auroc_loss.n_pos', kl['auroc_loss_among_ref_correct']['n_pos'], pp['pooled']['lost'])
for k in R['go']:
    cmp(f'go.{k}', R['go'][k], M['comparisons'][k]['go'])
    cmp(f'go_label.{k}', R['go_label'][k], M['comparisons'][k]['go_label'])
print('fields compared:', ncmp, 'mismatches:', len(mism))
for x in mism:
    print('MISMATCH', x)
