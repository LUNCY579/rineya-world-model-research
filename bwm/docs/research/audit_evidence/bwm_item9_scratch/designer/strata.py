import json, sys, collections
CL = sys.argv[1]
S = json.load(open(sys.argv[2]))['generated_basic']
G = json.load(open(sys.argv[3]))
def pk(i): return S[str(i)]['packages']
ids = range(2, 287)
cnt = collections.Counter(pk(i) for i in ids)
print('t1 ids 2-286 by packages', sorted(cnt.items()))
for lo, hi in [(2, 201), (1, 200), (202, 286)]:
    c = collections.Counter(pk(i) for i in range(lo, hi + 1))
    print(lo, hi, sorted(c.items()))
# GT length by package
L = collections.defaultdict(list)
for i in ids: L[pk(i)].append(len(G[str(i)]))
print({k: (min(v), max(v), round(sum(v)/len(v),1)) for k, v in sorted(L.items())})
# cities/airplanes/locs for <=3 and >=4
for grp, sel in [('p<=3', lambda p: p <= 3), ('p>=4', lambda p: p >= 4)]:
    c = collections.Counter((S[str(i)]['cities'], S[str(i)]['max_locs_per_city'], S[str(i)]['airplanes']) for i in ids if sel(pk(i)))
    print(grp, sum(c.values()), sorted(c.items()))
files = {
 'gpt4_1shot_nl': 'logistics/gpt-4_chat/task_1_plan_generation.json',
 'gpt4_1shot_pddl': 'logistics/gpt-4_chat/task_1_plan_generation_pddl.json',
 'gpt4_0shot_nl': 'logistics/gpt-4_chat/task_1_zero_shot_plan_generation.json',
 'gpt4_0shot_pddl': 'logistics/gpt-4_chat/task_1_zero_shot_plan_generation_pddl.json',
 'llama405_0shot_pddl': 'logistics/llama-3.1-405b_aws/task_1_plan_generation_zero_shot_pddl.json',
 'o1p_0shot_pddl': 'logistics/o1-preview_chat/task_1_plan_generation_zero_shot_pddl.json',
 'gpt35i_1shot_nl': 'logistics/gpt-3.5-turbo-instruct/task_1_plan_generation.json',
}
for name, f in files.items():
    d = json.load(open(CL + '/' + f))
    insts = d['instances']
    by = collections.defaultdict(lambda: [0, 0])
    idl = []
    for r in insts:
        i = r['instance_id']; idl.append(i)
        ok = r.get('llm_correct', r.get('correct'))
        p = pk(i); b = '1-3' if p <= 3 else ('4' if p == 4 else '5-7')
        by[b][0] += bool(ok); by[b][1] += 1
        by['all'][0] += bool(ok); by['all'][1] += 1
    print(name, min(idl), max(idl), len(idl), dict(by))
