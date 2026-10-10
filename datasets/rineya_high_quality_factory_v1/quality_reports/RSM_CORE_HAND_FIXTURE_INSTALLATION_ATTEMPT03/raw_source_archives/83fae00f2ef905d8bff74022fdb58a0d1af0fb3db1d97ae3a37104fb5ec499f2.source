"""Targeted critic gaps: signed fuel errors and wrong-location claims."""
from factory.reference import graph_run


def augment(world,candidates,rng):
    candidates=list(candidates)
    used={c['candidate_id'] for c in candidates}
    def append(plan,claim):
        while True:
            cid=f'c{rng.randrange(10000,99999)}'
            if cid not in used:break
        used.add(cid);candidates.append({'candidate_id':cid,'plan':list(plan),'claimed_after':claim})
    # Matched equal-plan claims isolate arithmetic sign and location, rather
    # than relying on validity or trajectory length to identify the error.
    for plan in [[],[world['nodes'][1]]]:
        actual=graph_run(world,plan)['after']
        assert actual['fuel']>0
        delta=rng.randint(1,min(3,actual['fuel']))
        append(plan,{**actual,'fuel':actual['fuel']-delta})
        append(plan,{**actual,'fuel':actual['fuel']+delta})
        other=rng.choice([n for n in world['nodes'] if n!=actual['node']])
        append(plan,{**actual,'node':other})
    rng.shuffle(candidates)
    return candidates
