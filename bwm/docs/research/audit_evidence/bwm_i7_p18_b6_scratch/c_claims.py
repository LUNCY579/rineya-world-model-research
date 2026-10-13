"""Post-hoc sensitivity: recompute registered statistics excluding confirmed contaminated instances (read-only)."""
import json, os, math
from math import comb

ROOT = 'C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs'
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, 'c_verify_result.json')))


def contaminated(evset, files):
    s = set()
    for f in files:
        c = C['files'][f]['full'].get(evset)
        if c:
            s |= set(c['instance_ids'])
    return s


DISAGREE = []


def load(rel, field=None):
    out = {}
    for line in open(os.path.join(ROOT, rel), encoding='utf-8'):
        r = json.loads(line)
        if 'val' in r and 'success' in r and r['val'] != r['success']:
            DISAGREE.append((rel, r['id']))
        out[r['id']] = bool(r['val']) if ('val' in r and field != 'success') else bool(r['success'])
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cp(k, n, a=0.05):
    def cdf(x, p):
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(100):
            mid = (lo + hi) / 2
            if f(mid): hi = mid
            else: lo = mid
        return (lo + hi) / 2
    lo = 0.0 if k == 0 else solve(lambda p: 1 - cdf(k - 1, p) >= a / 2)
    hi = 1.0 if k == n else solve(lambda p: cdf(k, p) <= a / 2)
    return round(lo, 3), round(hi, 3)


def paired(A, B, excl=frozenset(), ids=None):
    ids = [i for i in (ids or A) if i in A and i not in excl]
    assert all(i in B for i in ids), 'id mismatch'
    a = sum(A[i] for i in ids); b_ = sum(B[i] for i in ids)
    onlyA = sum(1 for i in ids if A[i] and not B[i]); onlyB = sum(1 for i in ids if B[i] and not A[i])
    return {'n': len(ids), 'A': a, 'B': b_, 'A_only': onlyA, 'B_only': onlyB, 'p_exact_mcnemar': float('%.3g' % mcnemar(onlyA, onlyB)),
            'A_CI95': cp(a, len(ids))}


res = {}
V2 = ['v2_actions']                      # Pa_v2_s1/s2/s3 training rows (+ init of Pa_v3 / Pa_v2c)
V3 = ['v2_actions', 'mix_v3', 'distill_rows']
V2C = ['v2_actions', 'mix_v2c']
ex = {
    'hard': contaminated('hard', V3 + V2C), 'mirror': contaminated('mirror', V3 + V2C), 'PB602': contaminated('PB602', V3 + V2C),
    'b3fresh': contaminated('b3fresh', V3 + V2C), 'seeds_fresh': contaminated('seeds_fresh', V2),
    'val_v2': contaminated('val_v2', V2), 'val_large': contaminated('val_large', V2),
}
res['excluded_sets'] = {k: sorted(v) if len(v) < 10 else len(v) for k, v in ex.items()}

# P-08 / P-10 (B2)
for s, pre in (('hard', 'hard'), ('mirror', 'mirror'), ('PB602', 'planbench')):
    beam = load(f'BW_TEST_B2/{pre}_beam/plans.jsonl'); direct = load(f'BW_TEST_B2/{pre}_direct/plans.jsonl')
    res[f'P-10 {s} B2 beam vs direct (original)'] = paired(beam, direct)
    res[f'P-10 {s} B2 beam vs direct (excl. contaminated)'] = paired(beam, direct, ex[s])

# P-11 (B3)
P = {f'{a}_{s}': load(f'BW_B3_EVAL/plans_{a}_{s}.jsonl', 'success') for a in ('v3', 'v2c', 'v2') for s in ('b3fresh', 'mirror')}
for s in ('b3fresh', 'mirror'):
    for a, b in (('v3', 'v2c'), ('v3', 'v2'), ('v2c', 'v2')):
        res[f'P-11 {s} {a} vs {b} (original)'] = paired(P[f'{a}_{s}'], P[f'{b}_{s}'])
        res[f'P-11 {s} {a} vs {b} (excl.)'] = paired(P[f'{a}_{s}'], P[f'{b}_{s}'], ex[s])

# P-13 (H1) and P-14 (H2)
hb = load('BW_TEST_B2/hard_beam/plans.jsonl'); hd = load('BW_TEST_B2/hard_direct/plans.jsonl')
h1 = load('BW_B3_HARD/h1_direct/plans.jsonl'); h2 = load('BW_B3_HARD/h2_beam/plans.jsonl')
res['P-13 H1 vs B2 direct (original)'] = paired(h1, hd); res['P-13 H1 vs B2 direct (excl.)'] = paired(h1, hd, ex['hard'])
res['P-14 H2 vs B2 beam (original)'] = paired(h2, hb); res['P-14 H2 vs B2 beam (excl.)'] = paired(h2, hb, ex['hard'])

# P-16 (published per-instance, hard)
pub = {}
for line in open(os.path.join(ROOT, 'HARD_PUBLISHED_PAIRED/per_instance.jsonl'), encoding='utf-8'):
    r = json.loads(line); iid = 'hard/' + r['instance']
    for k, v in r['bwm'].items():
        pub.setdefault(k, {})[iid] = bool(v['val'])
    for k, v in r['published'].items():
        pub.setdefault(k, {})[iid] = bool(v['pb_pddl_val'])
res['P-16 solved counts'] = {k: sum(v.values()) for k, v in pub.items()}
res['P-16 consistency: per_instance bwm == plans files'] = {
    'B2 S-beam': pub['B2 S-beam'] == hb, 'B2 S-direct': pub['B2 S-direct'] == hd, 'H1': pub['H1 Pa_v3 S-direct'] == h1, 'H2': pub['H2 Pa_v3 S-beam'] == h2}
for a in ('H2 Pa_v3 S-beam', 'B2 S-beam', 'H1 Pa_v3 S-direct', 'B2 S-direct'):
    for b in ('DeepSeek-R1', 'o1-preview', 'o1-mini'):
        res[f'P-16 {a} vs {b} (excl.)'] = paired(pub[a], pub[b], ex['hard'])

# P-17 item 7
for R in ('R0', 'R1', 'R2'):
    beam = load(f'BW_SEEDS/{R}_beam/plans.jsonl'); direct = load(f'BW_SEEDS/{R}_direct/plans.jsonl')
    res[f'P-17 {R} beam vs direct (original)'] = paired(beam, direct)
    res[f'P-17 {R} beam vs direct (excl. seeds_fresh/0)'] = paired(beam, direct, ex['seeds_fresh'])
    res[f'P-17 {R} seeds_fresh/0 outcome'] = {'beam': beam['seeds_fresh/0'], 'direct': direct['seeds_fresh/0']}
r0b = load('BW_SEEDS/R0_beam/plans.jsonl')
for R in ('R1', 'R2'):
    rb = load(f'BW_SEEDS/{R}_beam/plans.jsonl')
    res[f'P-17 spread {R} vs R0 beam (original)'] = paired(rb, r0b)
    res[f'P-17 spread {R} vs R0 beam (excl.)'] = paired(rb, r0b, ex['seeds_fresh'])

# BW_SELECT_v1 selection (val_large-30 + val_v2-50), rule: max mean success over both sets among beam configs
sel = [json.loads(l) for l in open(os.path.join(ROOT, 'BW_SELECT_v1/select.jsonl'))]
rows = []
for i, s in enumerate(sel):
    f = f'BW_SELECT_v1/plans_{i}_{s["set"]}.jsonl'
    recs = [json.loads(l) for l in open(os.path.join(ROOT, f), encoding='utf-8')]
    assert all(r.get('config_id', s['config_id']) == s['config_id'] for r in recs), f
    succ = {r['id']: bool(r['success']) for r in recs}
    exs = ex.get(s['set'], set())
    kept = [k for k in succ if k not in exs]
    rows.append({'i': i, 'config': s['config_id'], 'set': s['set'], 'n': len(succ), 'success_reported': s['success'],
                 'success_recomputed': round(sum(succ.values()) / len(succ), 4), 'n_clean': len(kept),
                 'success_clean': round(sum(succ[k] for k in kept) / len(kept), 4) if kept else None,
                 'failed_ids': [k for k, v in succ.items() if not v][:10],
                 'f_queries_per_instance': s.get('f_queries_per_instance'), 'tokens': s.get('proposer_tokens_per_instance')})
res['BW_SELECT_v1 rows'] = rows
beams = {}
for r in rows:
    if '"beam"' in r['config']:
        beams.setdefault(r['config'], {})[r['set']] = r
rank = []
for cfg, d in beams.items():
    if 'val_large' in d and 'val_v2' in d:
        rank.append((cfg, round((d['val_large']['success_recomputed'] + d['val_v2']['success_recomputed']) / 2, 4),
                     round((d['val_large']['success_clean'] + d['val_v2']['success_clean']) / 2, 4),
                     max(d['val_large']['f_queries_per_instance'], d['val_v2']['f_queries_per_instance'])))
res['BW_SELECT_v1 rule (cfg, mean original, mean clean, max F/inst)'] = sorted(rank, key=lambda t: -t[2])
res['val_vs_success_disagreements'] = DISAGREE
res['val_v2_first50_contaminated'] = len([i for i in range(50) if f'val_v2/{i}' in ex['val_v2']])
res['val_v2_first100_contaminated'] = len([i for i in range(100) if f'val_v2/{i}' in ex['val_v2']])
json.dump(res, open(os.path.join(HERE, 'c_claims_result.json'), 'w'), indent=1, default=list)
for k, v in res.items():
    print(k, ':', json.dumps(v, default=list)[:600])
