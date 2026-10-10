"""One bounded pilot runner for the remaining seven tracks; producer only."""
import argparse
import json
import random

from factory.common import ROOT, load_jsonl, record, record_hash, split_for, write_jsonl
from generators.skills.generate import make_task as make_skills
from generators.memory.generate import make_modules, make_episodic
from generators.other_tracks.reference import contracts, solve


def reorder(value):
    if isinstance(value, dict):
        return {key: reorder(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reorder(item) for item in value]
    return value


def make_belief(index, split, rng):
    dev = split == "dev_challenge"
    facts = ["hue", "portal", "sigil"] if dev else ["color", "door", "token"]
    size = 6 if dev else 4
    cycle = index // 6
    worlds = [dict(world_id=f"{'scenario' if dev else 'w'}{j}",
                   facts={facts[0]: [1, True, "blue", None][j % 4], facts[1]: "open" if (j + cycle) % 2 == 0 else "closed",
                          facts[2]: "known" if j % 3 == cycle % 3 else "unknown"}) for j in range(size)]
    mode = index % 6
    observations = [[], [dict(fact=facts[0], equals=1)],
                    [dict(fact=facts[0], equals=1), dict(fact=facts[0], equals=True)],
                    [dict(fact=facts[1], equals="open")], [dict(fact=facts[0], equals=None)],
                    [dict(fact=facts[1], equals="open"), dict(fact=facts[2], equals="known")]][mode]
    queries = [dict(query_id=f"q{j}", fact=fact) for j, fact in enumerate(facts)]
    return dict(kind="belief", worlds=worlds, observations=observations, queries=queries), \
        ["unfiltered_query_counts", "type_sensitive_observation", "inconsistent_empty_belief", "distinguishing_query",
         "entailed_null", "conjoined_observations"][mode], 3 if mode in (2, 5) else 2, \
        "Explicit finite worlds, exact filtering and value counts; no learned Bayesian inference or entropy optimization."


def make_predictive(index, split, rng):
    dev = split == "dev_challenge"
    states = rng.sample(["u", "v", "w", "x", "y", "z"] if dev else ["a", "b", "c", "d"], 6 if dev else 4)
    actions = ["pulse", "turn", "rest"] if dev else ["step", "flip"]
    mode, size, cycle = index % 6, len(states), index // 6
    observable, transition = {}, {}
    for j, state in enumerate(states):
        if mode == 0:
            value, step, flip = cycle, states[0] if cycle else states[(j + 1) % size], state
        elif mode == 1:
            value = int(j >= size // 2)
            begin = 0 if j < size // 2 else size // 2
            step, flip = states[begin + (j - begin + 1) % (size // 2)], state
            if cycle:
                step = states[(j + size // 2) % size]
        elif mode == 2:
            value, step, flip = int(j == size - 1), states[min(j + 1, size - 1)], state
            if cycle:
                flip = states[(j + 2) % size]
        elif mode == 3:
            value = (j // 2) % 2
            step, flip = states[((j // 2 + 1) % (size // 2)) * 2 + j % 2], states[j ^ 1]
            if cycle:
                flip = state
        elif mode == 4:
            value, step, flip = (True if j < size // 2 else 1), state, state
            if cycle:
                step = states[(j + size // 2) % size]
        else:
            value, step, flip = ("alert" if j == size - 1 else "quiet"), states[(j + 1) % size], states[0]
            if cycle:
                flip = states[-1]
        observable[state] = value
        transition[state] = {actions[0]: step, actions[1]: flip}
        if dev:
            transition[state][actions[2]] = state
    return dict(kind="predictive_abstraction", states=states, actions=actions, observable=observable, transition=transition), \
        ["one_equivalence_class", "observable_lumps", "iterated_refinement", "equivalent_state_pairs",
         "typed_observable_classes", "cyclic_distinguishability"][mode], 3 if mode in (2, 3, 5) else 2, \
        "Exact finite deterministic transition equivalence; no learned representation, latent training or generalization efficacy claim."


def make_repair(index, split, rng):
    dev = split == "dev_challenge"
    alphabet, slots = (["cedar", "elm", "fir"], 4) if dev else (["ash", "birch"], 3)
    a, b = alphabet[:2]
    mode = index % 6
    constraints = [[], [dict(kind="equal", positions=[0, 1]), dict(kind="count", value=a, min=1, max=slots)],
                   [dict(kind="different", positions=[0, 1]), dict(kind="allowed", position=0, values=[a])],
                   [dict(kind="allowed", position=2, values=[])],
                   [dict(kind="equal", positions=[0, 1]), dict(kind="different", positions=[1, 2]),
                    dict(kind="allowed", position=2, values=[b])],
                   [dict(kind="count", value=a, min=2, max=2), dict(kind="different", positions=[0, 2])]][mode]
    assignments = [[a] * slots, [b] * slots, [alphabet[j % len(alphabet)] for j in range(slots)],
                   [rng.choice(alphabet) for _ in range(slots)]]
    if dev:
        assignments.append([alphabet[-1]] * slots)
    ids = rng.sample(range(1000, 9999), len(assignments))
    candidates = [dict(candidate_id=f"c{cid}", assignment=assignment) for cid, assignment in zip(ids, assignments)]
    rng.shuffle(candidates)
    return dict(kind="constraint_repair", alphabet=alphabet, slots=slots, constraints=constraints, candidates=candidates,
                min_distance=slots + 1 if mode == 5 else index % 3), \
        ["unconstrained_greedy_diversity", "equal_count_repair", "different_allowed_repair", "unsatisfiable_repair",
         "conjoined_repair", "distance_exceeds_assignment_size"][mode], 2 if mode == 0 else 3, \
        "Finite exhaustive minimum Hamming repair and declared greedy diversity; no diffusion model or globally optimal packing claim."


def make_agents(index, split, rng):
    dev = split == "dev_challenge"
    objects = ["coin", "map", "vase"] if dev else ["book", "key"]
    places = ["atrium", "closet", "garden", "loft"] if dev else ["desk", "drawer", "shelf"]
    ids = ["Cora", "Dax", "Eli"] if dev else ["Ada", "Bex"]
    initial = {obj: places[j] for j, obj in enumerate(objects)}
    agents = []
    mode = index % 6
    for j, agent_id in enumerate(ids):
        beliefs = dict(initial)
        if j == 1:
            beliefs[objects[0]] = None
            beliefs[objects[-1]] = places[-1]
        if j == 2 or (mode == 4 and j == 0):
            beliefs = {obj: None for obj in objects}
        priority = rng.sample(objects, len(objects))
        if j == 1 and index % 4 == 0:
            priority = []
        agents.append(dict(agent_id=agent_id, initial_beliefs=beliefs, object_priority=priority))
    events = []
    for step in range(5 if dev else 3):
        observers = (ids if mode == 0 else ids[:1] if mode == 1 else [] if mode in (2, 4)
                     else [ids[step % len(ids)]] if mode == 3 else ids if step < 2 else ids[1:2])
        events.append(dict(object=objects[step % len(objects)], to=places[(step + 1 + index // 6) % len(places)], observed_by=observers))
    return dict(kind="agent_history", objects=objects, places=places, initial_locations=initial, agents=agents, events=events), \
        ["shared_observation", "private_observation", "hidden_moves", "different_observation_histories",
         "unknown_belief_abstention", "late_private_move"][mode], 3, \
        "Supplied first-order observation histories and declared search policies; no inferred psychology or second-order beliefs."


BUILDERS = {"RSM-001": make_skills, "RMM-001": make_modules, "REM-001": make_episodic,
            "RBM-001": make_belief, "RJP-001": make_predictive, "RDP-001": make_repair, "ROM-001": make_agents}

CHALLENGE_AXES = {"RSM-001": "larger_procedure_library", "RMM-001": "more_declared_modules",
                  "REM-001": "longer_event_history", "RBM-001": "more_possible_worlds",
                  "RJP-001": "more_states_and_actions", "RDP-001": "more_slots_and_symbols", "ROM-001": "more_agents_objects_and_events"}


def generate(count=12, batch="extended-pilot-v1", offset=0):
    if count < 0 or offset < 0 or not batch or "/" in batch or "\\" in batch:
        raise ValueError("Use nonnegative count/offset and a filename-safe batch")
    student_contracts = contracts()
    all_rows, output = [], []
    for track_number, (track, builder) in enumerate(BUILDERS.items()):
        rows, parents = [], []
        for index in range(offset, offset + count):
            split = split_for(index)
            rng = random.Random(20261009 + 10007 * track_number + 53 * index)
            task, mechanism, structural_difficulty, limitation = builder(index, split, rng)
            target = solve(task)
            challenge_axes = ["heldout_names", CHALLENGE_AXES[track]] if split == "dev_challenge" else []
            parent_id = f"{track}-{batch}-p{index:04d}"
            parents.append(dict(semantic_parent_id=parent_id, semantic_family_id=parent_id, split=split, tasks=[task],
                                mechanism=mechanism, structural_difficulty=structural_difficulty,
                                challenge_axes=challenge_axes, scope_limitation=limitation))
            for renderer in ("forward", "reverse"):
                payload = task if renderer == "forward" else reorder(task)
                prompt = "Executable task contract:\n" + student_contracts[task["kind"]] + "Task JSON:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                row = record(track, batch, index, "base", task, prompt, target, ["finite_synthetic_proxy", task["kind"], mechanism],
                             structural_difficulty, "structured", renderer)
                row.update(structural_difficulty=structural_difficulty, challenge_axes=challenge_axes,
                           scope_limitation=limitation, producer_role="producer", producer_agent_id="/root/relational_critic_producer")
                row["record_hash"] = record_hash(row)
                rows.append(row)
        output.extend([(ROOT / f"data/{track}/semantic_parents/{batch}.jsonl", parents),
                       (ROOT / f"data/{track}/candidate_renderings/{batch}.jsonl", rows)])
        all_rows.extend(rows)
    # Check all destinations before writing any file; older batches stay immutable.
    for path, contents in output:
        if path.exists() and load_jsonl(path) != contents:
            raise ValueError(f"Batch {path.name} already exists with different content; use a new batch version")
    for path, contents in output:
        write_jsonl(path, contents)
    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--batch", default="extended-pilot-v1")
    parser.add_argument("--offset", type=int, default=0)
    arguments = parser.parse_args()
    generate(arguments.count, arguments.batch, arguments.offset)
