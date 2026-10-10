"""Factual competency metadata checks authored from public contracts only.

Only ``semantic_task`` and the tag list are read. In particular, target labels,
producer code, and reviewer verdicts are never evidence for factual tags.
Goal-route multiplicities use resource-indexed dynamic frontier expansion;
reaching the goal does not terminate expansion. Counts saturate at two because
the public property distinguishes zero, one, and at least two sequences.
"""

from .independent import OracleError, _blocks_world, _graph_world


def goal_route_counts(world):
    """Return per-length goal-route counts, each saturated at two.

    A route is a complete legal destination sequence, including an empty route
    when initially at the goal. Edge costs are positive, making the bound finite.
    Distinct prefixes reaching the same (node, fuel) keep their multiplicity.
    """
    initial, goal, _, edges, adjacent = _graph_world(world)
    bound = initial[1] // min(edges.values()) if edges else 0
    frontier = {initial: 1}
    counts = []
    for depth in range(bound + 1):
        counts.append(min(2, sum(multiplicity for (node, _), multiplicity in frontier.items() if node == goal)))
        if depth == bound:
            break
        following = {}
        for (node, fuel), multiplicity in frontier.items():
            for destination, cost in adjacent[node]:
                if cost <= fuel:
                    state = (destination, fuel - cost)
                    following[state] = min(2, following.get(state, 0) + multiplicity)
        frontier = following
        if not frontier:
            counts.extend([0] * (bound - depth))
            break
    return counts


_GRAPH_TAGS = {
    "branched_resource_graph", "single_successor_resource_graph",
    "initial_goal_reachable", "initial_goal_unreachable", "initially_unreachable",
    "equal_length_goal_routes", "no_equal_length_goal_routes", "equal_length_routes",
}
_BLOCK_TAGS = {
    "multiple_initial_table_stacks", "multiple_explicit_goal_table_roots",
    "block_goal_composition", "multi_stack_goal_composition",
}


def check(row):
    """Return factual metadata failure strings for one canonical row.

    Unknown tags do not assert properties in COMPETENCY_TAGS.md and are ignored.
    A relevant tag on an incompatible or malformed semantic world is a failure.
    The metadata field is ``skill_tags``; colon-composed tags check each
    component independently. Legacy names remain diagnostic
    aliases and are checked against their factual property, never their labels.
    """
    if type(row) is not dict:
        return ["metadata_row: expected an object"]
    task = row.get("semantic_task")
    if type(task) is not dict:
        return ["semantic_task: expected an object"]
    tags = row.get("skill_tags")
    if type(tags) is not list or not all(type(tag) is str for tag in tags):
        return ["skill_tags: expected a list of strings"]
    relevant = {component for tag in tags for component in tag.split(":")} & (_GRAPH_TAGS | _BLOCK_TAGS)
    if not relevant:
        return []
    failures = []
    kind = task.get("kind")
    graph_tags = relevant & _GRAPH_TAGS
    if graph_tags:
        world = task.get("world") if kind == "critic" else task if kind == "graph" else None
        if type(world) is not dict:
            failures.extend(tag + ": requires a graph or critic graph world" for tag in sorted(graph_tags))
        else:
            try:
                _, _, nodes, edges, _ = _graph_world(world)
                outdegrees = {node: 0 for node in nodes}
                for origin, _ in edges:
                    outdegrees[origin] += 1
                maximum = max(outdegrees.values(), default=0)
                counts = goal_route_counts(world)
                reachable = any(count > 0 for count in counts)
                equal_routes = any(count >= 2 for count in counts)
                facts = {
                    "branched_resource_graph": maximum > 1,
                    "single_successor_resource_graph": maximum <= 1,
                    "initial_goal_reachable": reachable,
                    "initial_goal_unreachable": not reachable,
                    "initially_unreachable": not reachable,
                    "equal_length_goal_routes": equal_routes,
                    "no_equal_length_goal_routes": not equal_routes,
                    "equal_length_routes": equal_routes,
                }
                for tag in sorted(graph_tags):
                    if not facts[tag]:
                        failures.append(tag + ": false for semantic_task (maximum_outdegree=" + str(maximum)
                                        + ", initial_goal_reachable=" + str(reachable).lower()
                                        + ", goal_route_counts_capped_at_two=" + repr(counts) + ")")
            except (OracleError, TypeError) as error:
                failures.extend(tag + ": malformed graph world: " + str(error) for tag in sorted(graph_tags))
    block_tags = relevant & _BLOCK_TAGS
    if block_tags:
        if kind != "blocks":
            failures.extend(tag + ": requires a blocks world" for tag in sorted(block_tags))
        else:
            try:
                _blocks_world(task)
                initial_roots = sum(support == "TABLE" for support in task["on"].values())
                explicit_goal_roots = sum(support == "TABLE" for support in task["goal_on"].values())
                facts = {"multiple_initial_table_stacks": initial_roots >= 2,
                         "multiple_explicit_goal_table_roots": explicit_goal_roots >= 2,
                         "block_goal_composition": True,
                         "multi_stack_goal_composition": initial_roots >= 2 or explicit_goal_roots >= 2}
                for tag in sorted(block_tags):
                    if not facts[tag]:
                        failures.append(tag + ": false for semantic_task (initial_TABLE_roots=" + str(initial_roots)
                                        + ", explicit_goal_TABLE_roots=" + str(explicit_goal_roots) + ")")
            except (OracleError, TypeError) as error:
                failures.extend(tag + ": malformed blocks world: " + str(error) for tag in sorted(block_tags))
    return failures
