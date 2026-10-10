# Factual competency tags

Tags are metadata, separately checked from student-visible tasks and exact target correctness. Tags that name a property must hold in the generated world. A generator family is not proof that every sampled world has that property.

- branched_resource_graph: at least one world node has more than one outgoing edge. This is topology, not a claim that a search must visit the branch.
- single_successor_resource_graph: every node has at most one outgoing edge. Cycles can still exist.
- initial_goal_reachable / initial_goal_unreachable: there is / is not a feasible fresh goal-reaching plan from the initial node and fuel, allowing empty success.
- equal_length_goal_routes: at least two distinct feasible complete destination sequences terminate at the goal with the same number of moves. Repeated nodes and revisiting the goal are permitted. Positive edge costs bound moves by initial fuel divided by minimum edge cost. This does not imply two shortest routes.
- no_equal_length_goal_routes: no move length in that finite bound admits two such sequences.
- multiple_initial_table_stacks: at least two initial on-map values equal TABLE.
- multiple_explicit_goal_table_roots: at least two explicitly specified goal supports equal TABLE. Do not infer unspecified supports in partial goals.
- block_goal_composition: generic finite block-goal task; no multiple-stack claim.

Older initially_unreachable, equal_length_routes and multi_stack_goal_composition tags are retained in diagnostic/history bytes. Newly discovered false tags require an erratum or fresh batch, even when target labels are correct. Current metadata does not identify independent mechanism classes merely by generating new instances.

Legacy aliases for diagnostics: initially_unreachable means initial fresh-plan infeasibility; equal_length_routes means the same full feasible sequence property as equal_length_goal_routes; multi_stack_goal_composition requires at least two initial TABLE roots or at least two explicitly specified goal TABLE roots. These definitions do not infer missing partial-goal supports.
