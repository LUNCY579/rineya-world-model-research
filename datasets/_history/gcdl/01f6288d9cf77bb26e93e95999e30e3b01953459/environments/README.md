# Executable environment contract

All states, rules, goals and constraints are validated before execution. Keys inside these structures are closed: unknown keys fail with `ValueError`. Integers exclude Python/JSON booleans. Costs are nonnegative integers. Goals are nonempty conjunctions of exact equality atoms; every referenced entity and status must be declared. A syntactically valid conjunction may be unattainable. Constraints may conflict or already fail initially, producing an infeasible outcome rather than altering or repairing a task.

The primary implementation is `core.py`. The differential checker has its own validation, transitions, scoring and aggregation in `oracles/independent.py`; it does not import this implementation. Both were authored by the same engineer, so correlated mistakes remain possible. Separately authored hand-calculated adversarial tests provide external review evidence.

## Ledger

State: `accounts` maps account names to integer balances; `entries` maps entry names to `{account, amount, status}`. Signed entry amounts must be nonzero. Rules declare three distinct arbitrary status strings in `statuses.pending`, `.settled`, `.void`, a `void_effect` of `reverse` or `retain`, a Boolean `allow_overdraft`, inclusive `balance_bounds`, and costs for `settle`, `void`, `transfer`, `noop`. Current balances are authoritative; no undocumented historical transaction is inferred from a settled status.

- `settle(entry)` is legal only for pending entries. Add the signed amount to the owning account and change status to settled.
- `void(entry)` is legal for pending or settled entries. Pending voids change no balance. Settled voids subtract the signed amount under `reverse`, or preserve balances under `retain`. Change status to void.
- `transfer(source,destination,amount)` requires distinct declared accounts and a positive integer amount. Subtract from source and add to destination atomically.
- `noop` preserves state.

Any resulting balance outside the inclusive bounds is illegal. If overdrafts are disabled, a negative resulting balance is illegal. Goals optionally specify `balances` and/or `statuses`. Constraints optionally specify `minimum_balances`, `forbidden_statuses`, `max_cost`.

## Blocksworld

State: `on` maps every block to either one other block or the declared table string. The table cannot be a block. Valid states are acyclic stacks with at most one block directly above each block; any number of stacks may stand on the table. Inclusive `max_height` counts blocks in a stack. Rules declare the table, maximum height and costs for `move` and `noop`.

`move(block,destination)` requires a declared source block and destination block/table, different source/destination, a change of support, a clear source and a clear destination unless the destination is the table. Resulting height must obey `max_height`. Error precedence is unknown block, self support, unchanged support, blocked source, blocked destination, height exceeded. `noop` preserves state. Goals specify a nonempty `on` mapping. Constraints optionally specify `forbidden_on` pairs, `protected_blocks` whose initial support must persist at every successful step, and `max_cost`.

## Register

State: `registers` maps names to bounded integers. Rules declare inclusive `value_bounds`, overflow mode `reject`, `clamp`, or `wrap`, a nonempty subset of enabled operations, and exact costs for those operations. `add(register,value)`, `multiply(register,value)`, and `set(register,value)` use integer operands; `copy(source,destination)` copies the source; `swap(left,right)` swaps simultaneous values; `noop` preserves state.

Out-of-range results reject atomically, clamp to the closest bound, or wrap over the inclusive integer interval. Unknown registers and disabled operations are illegal. Goals specify nonempty `values`. Constraints optionally specify `minimum_values`, `maximum_values`, `protected_registers` whose initial values must persist at every successful step, and `max_cost`.

## Execution and probability

`oracles.primary.evaluate(task)` and `oracles.independent.evaluate(task)` return equal dictionaries. A task supplies `environment`, `initial_state`, `rules`, `goal`, `constraints`, and `candidates` as `{id,plan}` entries. Empty plans are valid. Action shape/type errors reject the task; declared-shape semantic errors (unknown entity, blocked move, insufficient funds) are labeled illegal. An unknown operation with only an `op` field is labeled illegal; an unknown operation with unrecognized arguments is malformed.

Plans stop on the first illegal action. Failed actions preserve state and cost zero; costs from the successful prefix remain. `steps` includes attempted actions, while `trajectory` includes the initial state and successful states. Constraints are checked initially and after every successful step. A later repair does not erase an earlier violation.

Per-world goal satisfaction is a Boolean fact; goal score counts exactly satisfied atoms. Utility is lexicographic `[legal and zero constraint violations, goal score, -successful cost]`. Candidate aggregates use exact reduced rational dictionaries. `success_probability` includes legality and constraints; `legal_probability` excludes constraints; `feasible_probability` includes constraints. `executed_steps` counts attempted transitions over all declared worlds, including failed actions.

Fully observed tasks have a single deterministic initial state. Partial-information tasks declare `information` as `missing` or `ambiguous`, put null in unknown observation leaves, provide at least two distinct complete `possible_worlds` with identical field/entity structure, and state `probability_assumption="uniform_over_possible_worlds"`. Every known observation leaf must agree with each world. Probabilities are analytical conditional probabilities over that explicit finite support, not inferred priors or claims that a support set captures all real-world possibilities. This implementation has no stochastic transitions.
