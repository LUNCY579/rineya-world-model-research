"""Factual competency annotations after graph augmentation."""
from functools import lru_cache


def equal_length_goal_routes(world):
    outgoing={n:[] for n in world['nodes']}
    for a,b,cost in world['edges']:outgoing[a].append((b,cost))
    @lru_cache(None)
    def ways(node,fuel,depth):
        if depth==0:return int(node==world['goal'])
        return min(2,sum(ways(n,fuel-cost,depth-1) for n,cost in outgoing[node] if cost<=fuel))
    maximum=world['fuel']//min(cost for a,b,cost in world['edges'])
    return any(ways(world['start'],world['fuel'],depth)>=2 for depth in range(maximum+1))
