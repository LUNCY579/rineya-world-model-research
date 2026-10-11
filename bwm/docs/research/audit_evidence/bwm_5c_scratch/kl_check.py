"""Report-only cross-check of RET_RESULTS kl blocks (no torch). numpy only, chunked, mmap."""
import sys, json
sys.dont_write_bytecode = True
import numpy as np
assert 'torch' not in sys.modules
RUN = 'C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/runs/retention/confirm_1708484032065101819'
DRAW = 'C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/docs/retention/draws/confirm_1708484032065101819.jsonl'
S = 'C:/Users/clclc/AppData/Local/Temp/claude/bwm_5c_scratch/'
R = json.load(open(RUN + '/analysis/RET_RESULTS.json', encoding='utf-8'))
rows = [json.loads(l) for l in open(DRAW, encoding='utf-8')]
P = {}
for l in open(S + 'my_prompt_grades.jsonl', encoding='utf-8'):
    d = json.loads(l); P[d['pid']] = d
paths = {'off': 'sparse/lp_off.npy', 't2s0': 'sparse/lp_t2s0.npy', 'x_bwf': 'sparse/lp_x_bwf.npy',
         'x_bwpa': 'sparse/lp_x_bwpa.npy', 'supa': 'supa/lp_supa.npy'}
gens = {s: {json.loads(l)['pid']: json.loads(l)['lp_row'] for l in open(RUN + '/' + p.replace('lp_', 'gen_').replace('.npy', '.jsonl'), encoding='utf-8')} for s, p in paths.items()}


def lsm(x):
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def kl(ref, arm):
    A = np.load(RUN + '/' + paths[ref], mmap_mode='r'); B = np.load(RUN + '/' + paths[arm], mmap_mode='r')
    out = np.empty(len(rows))
    for i0 in range(0, len(rows), 16):
        idx = list(range(i0, min(i0 + 16, len(rows))))
        ra = [gens[ref][rows[i]['pid']] for i in idx]; rb = [gens[arm][rows[i]['pid']] for i in idx]
        la, lb = lsm(np.asarray(A[ra])), lsm(np.asarray(B[rb]))
        out[i0:i0 + len(idx)] = (np.exp(la) * (la - lb)).sum(1)
    return out, A.shape, A.dtype


def auroc(score, lab):
    score = np.asarray(score, float); lab = np.asarray(lab, bool)
    order = np.argsort(score, kind='mergesort'); s = score[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    npos, nneg = lab.sum(), (~lab).sum()
    return float((ranks[lab].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def item_ok(arm):
    d = {}
    for r in rows:
        d[r['item']] = d.get(r['item'], True) and P[r['pid']][arm]['P']
    return d


for name in ['x_bwf_vs_off', 'x_bwpa_vs_off', 't2s0_vs_off']:
    c = R['comparisons'][name]; arm, ref = c['arm'], c['reference']
    k, shape, dt = kl(ref, arm)
    per = {}
    for i, r in enumerate(rows):
        per.setdefault(r['item'], []).append(k[i])
    items = list(per)
    score = np.array([np.mean(per[it]) for it in items])
    ro, ao = item_ok(ref), item_ok(arm)
    flip = np.array([ro[it] != ao[it] for it in items]); refok = np.array([ro[it] for it in items])
    loss = np.array([ro[it] and not ao[it] for it in items])
    mine = {'kl_mean': float(score.mean()), 'kl_median': float(np.median(score)),
            'kl_mean_flipped': float(score[flip].mean()), 'kl_mean_unflipped': float(score[~flip].mean()),
            'auroc_flip': auroc(score, flip), 'auroc_loss': auroc(score[refok], loss[refok])}
    theirs = {'kl_mean': c['kl']['kl_mean'], 'kl_median': c['kl']['kl_median'], 'kl_mean_flipped': c['kl']['kl_mean_flipped'],
              'kl_mean_unflipped': c['kl']['kl_mean_unflipped'], 'auroc_flip': c['kl']['auroc_flip']['auroc'],
              'auroc_loss': c['kl']['auroc_loss_among_ref_correct']['auroc']}
    print(name, shape, dt)
    for kk in mine:
        print(f'   {kk:18s} mine {mine[kk]:.10f} theirs {theirs[kk]:.10f} absdiff {abs(mine[kk]-theirs[kk]):.2e}')
