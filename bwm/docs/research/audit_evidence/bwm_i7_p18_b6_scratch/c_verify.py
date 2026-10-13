"""METHOD C (reconciliation): independent exact canonical form + explicit bijection check.

Read-only on the repo. Stdlib only. One process, streams jsonl.

Encoding: problem -> (global nullary facts tagged I/G, per-vertex unary labels tagged I/G,
per-vertex 'ports' Id/Iu/Gd/Gu for binary 'on' facts).  Requires each vertex to have at most one
edge per port (true for valid Blocksworld states and for goals that are linear forests); any
violation raises (counted, never silently accepted).  Rooted code = deterministic BFS from a root
following ports in fixed order; component code = min over roots; problem code = (globals, sorted
component codes).  Equal codes <=> isomorphic (see report).  Every hit is re-confirmed by building
the bijection from the BFS orders and checking exact fact-set equality after renaming.
"""
import json, re, sys, os, glob, hashlib, time
from collections import defaultdict, Counter

ROOT = 'C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'c_verify_result.json')
PORTS = ('Id', 'Iu', 'Gd', 'Gu')
UNARY = {'clear', 'ontable', 'holding'}
NULLARY = {'handempty'}


def build(objs, init, goal):
    lab = {o: [] for o in objs}
    glb = []
    ports = {o: {} for o in objs}
    for tag, facts in (('I', init), ('G', goal)):
        for f in set(map(tuple, facts)):
            if len(f) == 1:
                assert f[0] in NULLARY, f
                glb.append(tag + f[0])
            elif len(f) == 2:
                assert f[0] in UNARY, f
                lab[f[1]].append(tag + f[0])          # KeyError if block not in objs
            elif len(f) == 3:
                assert f[0] == 'on', f
                x, y = f[1], f[2]
                kd, ku = tag + 'd', tag + 'u'
                if kd in ports[x] or ku in ports[y]:
                    raise ValueError('port not unique')
                ports[x][kd] = y
                ports[y][ku] = x
            else:
                raise ValueError(f)
    return tuple(sorted(glb)), {o: tuple(sorted(v)) for o, v in lab.items()}, ports


def rooted(r, lab, ports):
    order = [r]; idx = {r: 0}; i = 0
    while i < len(order):
        v = order[i]; i += 1
        pv = ports[v]
        for k in PORTS:
            w = pv.get(k)
            if w is not None and w not in idx:
                idx[w] = len(order); order.append(w)
    code = tuple((lab[v], tuple(idx[ports[v][k]] if k in ports[v] else -1 for k in PORTS)) for v in order)
    return code, order


def canon(objs, init, goal, want_orders=False):
    glb, lab, ports = build(objs, init, goal)
    seen = set(); comps = []
    for o in sorted(objs):
        if o in seen:
            continue
        _, comp = rooted(o, lab, ports)
        seen.update(comp)
        best = None
        for r in comp:
            c, order = rooted(r, lab, ports)
            if best is None or c < best[0]:
                best = (c, order)
        comps.append(best)
    comps.sort(key=lambda t: t[0])
    key = (glb, tuple(c for c, _ in comps))
    if want_orders:
        return key, [o for _, o in comps]
    return key


def explicit_map(e, t):
    """e, t = (objs, init, goal). Returns bijection e-block -> t-block, verified, or None."""
    ke, oe = canon(*e, want_orders=True)
    kt, ot = canon(*t, want_orders=True)
    if ke != kt:
        return None
    m = {}
    for a, b in zip(oe, ot):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            m[x] = y
    # verification: bijection on full object sets and exact fact equality
    if set(m) != set(e[0]) or set(m.values()) != set(t[0]) or len(set(m.values())) != len(m):
        return None
    ren = lambda facts: {(f[0],) + tuple(m[x] for x in f[1:]) for f in map(tuple, facts)}
    if ren(e[1]) != set(map(tuple, t[1])) or ren(e[2]) != set(map(tuple, t[2])):
        return None
    return m


def simulate(init, plan, goal):
    on = {}; table = set(); clear = set(); hold = None
    for f in init:
        if f[0] == 'on': on[f[1]] = f[2]
        elif f[0] == 'ontable': table.add(f[1])
        elif f[0] == 'clear': clear.add(f[1])
        elif f[0] == 'holding': hold = f[1]
    for a in plan:
        op = a[0]
        if op == 'pick-up':
            x = a[1]
            if not (hold is None and x in clear and x in table): return False
            table.discard(x); clear.discard(x); hold = x
        elif op == 'put-down':
            x = a[1]
            if hold != x: return False
            table.add(x); clear.add(x); hold = None
        elif op == 'stack':
            x, y = a[1], a[2]
            if not (hold == x and y in clear): return False
            on[x] = y; clear.discard(y); clear.add(x); hold = None
        elif op == 'unstack':
            x, y = a[1], a[2]
            if not (hold is None and on.get(x) == y and x in clear): return False
            del on[x]; clear.add(y); clear.discard(x); hold = x
        else:
            return False
    return all(on.get(f[1]) == f[2] for f in goal)


# ---------------- parsing ----------------
FACT_RE = re.compile(r'\(([^()]*)\)')


def parse_pddl(path):
    txt = re.sub(r';[^\n]*', '', open(path, encoding='utf-8').read().lower())
    objs = re.search(r'\(:objects([^)]*)\)', txt).group(1).split()
    assert all(re.fullmatch(r'[a-z0-9_]+', o) for o in objs), objs
    ii, gi = txt.index('(:init'), txt.index('(:goal')
    init = [tuple(s.split()) for s in FACT_RE.findall(txt[ii:gi]) if s.strip()]
    goal = [tuple(s.split()) for s in FACT_RE.findall(txt[gi:]) if s.strip() and s.split()[0] != 'and']
    return objs, init, goal


def pb_sort(p):
    return int(os.path.basename(p).split('.')[0].split('-')[1])


def load_evals():
    ev = {}
    def js(name, rel, prefix):
        L = json.load(open(os.path.join(ROOT, rel), encoding='utf-8'))
        ev[name] = [(f'{prefix}/{i}', (list(r['objs']), [tuple(f) for f in r['init']], [tuple(f) for f in r['goal']])) for i, r in enumerate(L)]
    out = []
    for sub in ('generated_basic_3', 'generated_basic'):
        for p in sorted(glob.glob(f'{ROOT}/bench/planbench/blocksworld/{sub}/*.pddl'), key=pb_sort):
            out.append((f'{sub}/' + os.path.basename(p)[:-5], parse_pddl(p)))
    ev['PB602'] = out
    ev['hard'] = [('hard/' + os.path.basename(p)[:-5], parse_pddl(p)) for p in
                  sorted(glob.glob(f'{ROOT}/bench/planbench/blocksworld_hard/generated/*.pddl'), key=pb_sort)]
    js('mirror', 'runs/BW_val_large/instances_mirror_hard.json', 'mirror')
    js('b3fresh', 'runs/BW_val_large/instances_b3_fresh.json', 'b3fresh')
    js('val_large', 'runs/BW_val_large/instances_val_large.json', 'val_large')
    js('seeds_fresh', 'runs/BW_val_large/instances_seeds_fresh.json', 'seeds_fresh')
    js('val_v2', 'runs/BW_data_v2/instances_val.json', 'val_v2')
    js('val_v1', 'runs/BW_data/instances_val.json', 'val')
    d = []
    with open(f'{ROOT}/runs/BW_DISTILL_r1/solutions.jsonl', encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            d.append((r['id'], (list(r['objs']), [tuple(f) for f in r['init']], [tuple(f) for f in r['goal']])))
    ev['distill240'] = d
    ev['unsolvable101_extra'] = [('unsolv/' + os.path.basename(p)[:-5], parse_pddl(p)) for p in
                                 sorted(glob.glob(f'{ROOT}/bench/planbench/unsolvable_blocksworld/generated_basic/*.pddl'), key=pb_sort)]
    return ev


UTAG = '<|im_start|>user\n'


def parse_row(line):
    d = json.loads(line)
    p = d['prompt']
    i = p.index(UTAG) + len(UTAG); j = p.index('<|im_end|>', i)
    u = json.loads(p[i:j])
    assert set(u) <= {'goal', 'outcome', 'state'}, u.keys()
    st = u['state']
    assert set(st) <= {'clear', 'hand', 'on', 'table'}, st.keys()
    init = [('clear', x) for x in st['clear']] + [('ontable', x) for x in st['table']] + [('on', x, y) for x, y in st['on']]
    init.append(('handempty',) if st['hand'] == 'empty' else ('holding', st['hand']))
    goal = [('on', x, y) for x, y in u['goal']]
    objs = set(st['clear']) | set(st['table']) | {b for xy in st['on'] for b in xy}
    if st['hand'] != 'empty':
        objs.add(st['hand'])
    gb = {b for xy in u['goal'] for b in xy}
    assert gb <= objs, ('goal block missing from state', gb - objs)
    tgt = json.loads(d['target'])
    plan = [tuple(a.split()) for a in tgt['actions']] if 'actions' in tgt else None
    return (sorted(objs), init, goal), u.get('outcome', d.get('outcome')), plan, hashlib.sha1((d['prompt'] + '\x00' + d['target']).encode()).hexdigest(), p[i:j]


def main():
    t0 = time.time()
    ev = load_evals()
    index = defaultdict(list)       # full canon -> [(set, id)]
    iindex = defaultdict(list)      # init-only canon
    gindex = defaultdict(list)      # goal-only canon
    probs = {}
    nblocks = {}
    eval_nonforest = {}; gindex_skip = set()
    for s, L in ev.items():
        for eid, pr in L:
            probs[eid] = pr
            nblocks[eid] = len(pr[0])
            try:
                index[canon(*pr)].append((s, eid))
            except ValueError:
                # non-unique port (e.g. two blocks on one block in an unsolvable goal): such a problem cannot be
                # isomorphic to any training row (all of which pass build()), so it is provably not contaminated.
                eval_nonforest[s] = eval_nonforest.get(s, 0) + 1
                gindex_skip.add(eid)
                iindex[canon(pr[0], pr[1], [])].append((s, eid))
                continue
            iindex[canon(pr[0], pr[1], [])].append((s, eid))
            gindex[canon(pr[0], [], pr[2])].append((s, eid))
    res = {'eval_sizes': {s: len(L) for s, L in ev.items()},
           'eval_blocks': {s: sorted(Counter(len(pr[0]) for _, pr in L).items()) for s, L in ev.items()},
           'files': {}, 'eval_nonforest_instances': eval_nonforest}
    # eval-eval duplicates (descriptive)
    dup = Counter()
    for k, members in index.items():
        sets = [m[0] for m in members]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                dup['|'.join(sorted((members[a][0], members[b][0])))] += 1
    res['eval_eval_full_isomorphic_pairs'] = dict(dup)
    print('evals loaded', {s: len(L) for s, L in ev.items()}, round(time.time() - t0, 1), flush=True)

    files = [
        ('v2_actions', 'runs/BW_data_v2/proposer_actions_train.jsonl'),
        ('v2_states', 'runs/BW_data_v2/proposer_states_train.jsonl'),
        ('distill_rows', 'runs/BW_DISTILL_r1/distill_rows.jsonl'),
        ('mix_v3', 'runs/BW_DISTILL_r1/mix_v3.jsonl'),
        ('mix_v2c', 'runs/BW_DISTILL_r1/mix_v2c.jsonl'),
        ('v1_actions', 'runs/BW_data/proposer_actions_train.jsonl'),
        ('v1_states', 'runs/BW_data/proposer_states_train.jsonl'),
        ('superseded_v1_actions', 'runs/_superseded/BW_data_v1_suffix_leak/proposer_actions_train.jsonl'),
    ]
    kind_of = {}
    src = {}  # sha1(prompt,target) -> (file, line) for v2_actions and distill_rows
    prompt_sha = {}  # for identity check of *_states twins
    for fname, rel in files:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            res['files'][fname] = {'missing': rel}; continue
        hits = defaultdict(list)    # eval set -> [(eid, line, kind, outcome, map, plan_solves, src)]
        ihits = defaultdict(set); ghits = defaultdict(set)
        blocks = Counter(); kinds = Counter(); errs = Counter()
        prev_goal = None; grp_init = None; grp_first = 0; end_group = False
        psh = []
        n = 0
        with open(path, encoding='utf-8') as fh:
            for ln, line in enumerate(fh, 1):
                n += 1
                try:
                    pr, outcome, plan, sh, user = parse_row(line)
                except Exception as ex:
                    errs[type(ex).__name__ + ':' + str(ex)[:60]] += 1; continue
                psh.append(hashlib.sha1(user.encode()).hexdigest()[:16])
                if fname in ('v2_actions', 'distill_rows'):
                    src[sh] = (fname, ln)
                origin = src.get(sh, (None, None)) if fname.startswith('mix') else (fname, ln)
                blocks[len(pr[0])] += 1
                gkey = tuple(sorted(map(tuple, pr[2])))
                skey = tuple(sorted(map(tuple, pr[1])))
                if gkey != prev_goal or end_group:
                    grp_init = skey; grp_first = ln; kind = 't0'; end_group = False
                elif skey == grp_init:
                    kind = 'from-init-again(noisy)' if outcome != 'success' else 'from-init-again(success)'
                    end_group = True
                else:
                    kind = 'suffix'
                prev_goal = gkey
                if fname in ('v2_actions', 'distill_rows'):
                    kind_of[(fname, ln)] = kind
                if fname.startswith('mix'):
                    kind = 'src:' + str(origin[0]) + ':' + kind_of.get(origin, 'unknown')
                kinds[kind] += 1
                try:
                    k = canon(*pr)
                    ki = canon(pr[0], pr[1], [])
                    kg = canon(pr[0], [], pr[2])
                except Exception as ex:
                    errs['canon:' + str(ex)[:60]] += 1; continue
                for s, eid in iindex.get(ki, ()):
                    ihits[s].add(eid)
                for s, eid in gindex.get(kg, ()):
                    ghits[s].add(eid)
                for s, eid in index.get(k, ()):
                    m = explicit_map(probs[eid], pr)
                    assert m is not None, ('UNCONFIRMED', fname, ln, eid)
                    inv = {v: kk for kk, v in m.items()}
                    solves = None
                    if plan is not None and outcome == 'success':
                        mp = [(a[0],) + tuple(inv[x] for x in a[1:]) for a in plan]
                        solves = simulate(probs[eid][1], mp, probs[eid][2])
                    hits[s].append((eid, ln, kind, outcome, m, solves, origin))
        rec = {'path': rel, 'rows': n, 'blocks': sorted(blocks.items()), 'row_kinds': dict(kinds), 'parse_errors': dict(errs),
               'prompt_sha_of_user_json_first': psh[:1], 'full': {}, 'init_only_instances': {s: len(v) for s, v in ihits.items()},
               'goal_only_instances': {s: len(v) for s, v in ghits.items()}}
        prompt_sha[fname] = psh
        for s, L in hits.items():
            inst = sorted({h[0] for h in L}, key=lambda e: (e.split('/')[0], int(re.sub(r'\D', '', e.split('/')[-1]) or 0)))
            rec['full'][s] = {
                'instances': len(inst), 'instance_ids': inst, 'pairs': len(L), 'rows': len({h[1] for h in L}),
                'instances_with_t0_row': len({h[0] for h in L if h[2].endswith('t0')}),
                'pairs_by_kind': dict(Counter(h[2] for h in L)),
                'pairs_target_solves': sum(1 for h in L if h[5]), 'pairs_success_rows': sum(1 for h in L if h[5] is not None),
                'origin_counts': dict(Counter(str(h[6][0]) for h in L)),
                'first20': [{'eval': h[0], 'line': h[1], 'kind': h[2], 'outcome': h[3], 'map_eval_to_train': h[4],
                             'target_solves_eval': h[5], 'origin': h[6]} for h in L[:20]],
                'all_pairs': [(h[0], h[1], h[2], h[6]) for h in L],
            }
        res['files'][fname] = rec
        print(fname, n, 'rows', dict(blocks), {s: (v['instances'], v['pairs']) for s, v in rec['full'].items()}, dict(errs), round(time.time() - t0, 1), flush=True)
    for a, b in (('v2_actions', 'v2_states'), ('v1_actions', 'v1_states')):
        if a in prompt_sha and b in prompt_sha:
            res[f'user_json_identical_{a}_vs_{b}'] = (prompt_sha[a] == prompt_sha[b], len(prompt_sha[a]), len(prompt_sha[b]))
    del prompt_sha
    # RCX-E07 GNN train/val instances (descriptive extra)
    rc = json.load(open(f'{ROOT}/runs/BW_B4_SCREEN/rcx_snapshot/runs/RCX-E07/main/data/instances.json', encoding='utf-8'))
    rch = []
    for r in rc:
        if r.get('split') not in ('train', 'val'):
            continue
        pr = (list(r['objs']), [tuple(f) for f in r['init']], [tuple(f) for f in r['goal']])
        for s, eid in index.get(canon(*pr), ()):
            rch.append((r['id'], eid, explicit_map(probs[eid], pr)))
    res['rcx_e07_trainval'] = {'n': sum(1 for r in rc if r.get('split') in ('train', 'val')), 'splits': dict(Counter(r.get('split') for r in rc)), 'full_hits': rch}
    res['wall_s'] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, 'w'), default=list)
    print('done', res['wall_s'])


if __name__ == '__main__':
    main()
