import json, sys, collections
CL = sys.argv[1]
S = json.load(open('inst_stats.json'))['generated_basic']
pk = lambda i: S[str(i)]['packages']
A = [i for i in range(2, 201) if pk(i) >= 4]
B = [i for i in range(202, 287) if pk(i) >= 4]
IN = [i for i in range(2, 287) if pk(i) <= 3]
print('A', len(A), collections.Counter(pk(i) for i in A), 'B', len(B), 'IN', len(IN), 'id201 pk', pk(201))
shape = lambda i: (S[str(i)]['cities'], S[str(i)]['max_locs_per_city'], S[str(i)]['airplanes'], pk(i))
print('A shapes distinct', len(set(shape(i) for i in A)))
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
    m = {r['instance_id']: bool(r.get('llm_correct', r.get('correct'))) for r in d['instances']}
    a = [m[i] for i in A if i in m]
    inn = [m[i] for i in IN if i in m]
    print(f'{name}: PB-OUT-A {sum(a)}/{len(a)}; PB-IN covered {sum(inn)}/{len(inn)}')
