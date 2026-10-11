"""Minimal h* labelling example (HSTAR_LABEL_RECIPE.md section 4). Pure numpy/stdlib; no BWM import needed."""
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rcx.bw import space as SP  # noqa: E402


def main():
    objs = ["a", "b", "c", "d", "e"]
    goal = frozenset({("on", "a", "b"), ("on", "b", "c"), ("on", "d", "e")})
    labels = SP.GoalLabels(objs, goal)
    rng = random.Random(20261289)
    for _ in range(5):
        s = labels.state(rng.randrange(len(labels.space)))
        h_bfs = labels.h(s)
        h_astar, exp = SP.astar_hstar(objs, s, goal)
        assert h_bfs == h_astar, (h_bfs, h_astar)
        print(f"h*={h_bfs:.0f} (A* agrees, {exp} expansions); optimal actions {sorted(labels.opt_actions(s))[:3]}...")
    print(f"states in space: {len(labels.space)}; all h* finite: {bool((labels.hstar_array() < float('inf')).all())}")


if __name__ == "__main__":
    main()
