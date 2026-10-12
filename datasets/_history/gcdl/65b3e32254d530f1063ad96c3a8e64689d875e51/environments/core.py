"""New stdlib simulators; no dependency on RCWM/BWM experiment code.

Rules are task data. Failed actions preserve state, incur zero cost, and stop plans.
Constraints are trajectory properties; they never change the transition function.
"""
from copy import deepcopy

ENVIRONMENT_VERSION = "environments-1.0"


def integer(value, name, minimum=None):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name}: expected integer" + (f" >= {minimum}" if minimum is not None else ""))


def obj(value, required, optional=(), name="object"):
    if type(value) is not dict or set(value) - set(required) - set(optional) or not set(required) <= set(value):
        raise ValueError(f"{name}: unsupported or missing keys")


def names(value, name, nonempty=False):
    if type(value) is not dict or (nonempty and not value) or any(type(k) is not str or not k for k in value):
        raise ValueError(f"{name}: expected named mapping")


def sequence(value, name):
    if type(value) is not list:
        raise ValueError(f"{name}: expected list")


def costs(rules, operations):
    obj(rules["action_costs"], operations, name="action_costs")
    for value in rules["action_costs"].values():
        integer(value, "action cost", 0)


def bounds(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError("bounds: expected [lower, upper]")
    for x in value:
        integer(x, "bound")
    if value[0] > value[1]:
        raise ValueError("bounds: reversed")


def refs(mapping, universe, name):
    names(mapping, name)
    if not set(mapping) <= set(universe):
        raise ValueError(f"{name}: unknown entity")
    for value in mapping.values():
        integer(value, name)


def unique_strings(value, universe, name):
    sequence(value, name)
    if any(type(v) is not str for v in value) or len(set(value)) != len(value) or not set(value) <= set(universe):
        raise ValueError(f"{name}: unknown or duplicate names")


class Ledger:
    operations = ("settle", "void", "transfer", "noop")

    def validate(self, state, rules, goal, constraints):
        obj(rules, ("statuses", "void_effect", "allow_overdraft", "balance_bounds", "action_costs"), name="ledger rules")
        obj(rules["statuses"], ("pending", "settled", "void"), name="statuses")
        vals = list(rules["statuses"].values())
        if any(type(v) is not str or not v for v in vals) or len(set(vals)) != 3:
            raise ValueError("statuses: three distinct nonempty strings required")
        if rules["void_effect"] not in ("reverse", "retain") or type(rules["allow_overdraft"]) is not bool:
            raise ValueError("ledger rule value")
        bounds(rules["balance_bounds"])
        costs(rules, self.operations)
        obj(state, ("accounts", "entries"), name="ledger state")
        names(state["accounts"], "accounts", True)
        names(state["entries"], "entries")
        lo, hi = rules["balance_bounds"]
        for value in state["accounts"].values():
            integer(value, "balance")
            if not lo <= value <= hi or (not rules["allow_overdraft"] and value < 0):
                raise ValueError("initial balance violates rules")
        for entry in state["entries"].values():
            obj(entry, ("account", "amount", "status"), name="entry")
            if type(entry["account"]) is not str or entry["account"] not in state["accounts"] or type(entry["status"]) is not str or entry["status"] not in vals:
                raise ValueError("entry account/status unknown")
            integer(entry["amount"], "entry amount")
            if entry["amount"] == 0:
                raise ValueError("entry amount must be nonzero")
        obj(goal, (), ("balances", "statuses"), name="ledger goal")
        if not goal or not any(goal.values()):
            raise ValueError("empty goal")
        refs(goal.get("balances", {}), state["accounts"], "balance goal")
        refs_status = goal.get("statuses", {})
        names(refs_status, "status goal")
        if not set(refs_status) <= set(state["entries"]) or any(type(v) is not str or v not in vals for v in refs_status.values()):
            raise ValueError("invalid status goal")
        obj(constraints, (), ("minimum_balances", "forbidden_statuses", "max_cost"), name="ledger constraints")
        refs(constraints.get("minimum_balances", {}), state["accounts"], "minimum balances")
        unique_strings(constraints.get("forbidden_statuses", []), vals, "forbidden statuses")
        if "max_cost" in constraints:
            integer(constraints["max_cost"], "max_cost", 0)

    def validate_action(self, action):
        if type(action) is not dict or type(action.get("op")) is not str or not action["op"]:
            raise ValueError("action requires string op")
        op = action["op"]
        shapes = {"settle": ("op", "entry"), "void": ("op", "entry"), "transfer": ("op", "source", "destination", "amount"), "noop": ("op",)}
        if op not in shapes:
            obj(action, ("op",), name="unsupported action")
            return
        obj(action, shapes[op], name="ledger action")
        for key in ("entry", "source", "destination"):
            if key in action and (type(action[key]) is not str or not action[key]):
                raise ValueError("action entity requires nonempty string")
        if "amount" in action:
            integer(action["amount"], "transfer amount")

    def transition(self, state, action, rules):
        result = deepcopy(state)
        op = action["op"]
        if op not in self.operations:
            return state, "unsupported_operation"
        if op == "noop":
            return result, None
        if op == "transfer":
            src, dst, amount = action["source"], action["destination"], action["amount"]
            if src not in state["accounts"] or dst not in state["accounts"]:
                return state, "unknown_account"
            if src == dst or amount <= 0:
                return state, "invalid_transfer"
            result["accounts"][src] -= amount
            result["accounts"][dst] += amount
        else:
            key = action["entry"]
            if key not in state["entries"]:
                return state, "unknown_entry"
            entry = result["entries"][key]
            status = entry["status"]
            if op == "settle":
                if status != rules["statuses"]["pending"]:
                    return state, "entry_not_pending"
                result["accounts"][entry["account"]] += entry["amount"]
                entry["status"] = rules["statuses"]["settled"]
            else:
                if status == rules["statuses"]["void"]:
                    return state, "entry_already_void"
                if status == rules["statuses"]["settled"] and rules["void_effect"] == "reverse":
                    result["accounts"][entry["account"]] -= entry["amount"]
                entry["status"] = rules["statuses"]["void"]
        low, high = rules["balance_bounds"]
        if any(not low <= v <= high for v in result["accounts"].values()):
            return state, "balance_out_of_bounds"
        if not rules["allow_overdraft"] and any(v < 0 for v in result["accounts"].values()):
            return state, "overdraft_forbidden"
        return result, None

    def score(self, state, goal):
        tests = [state["accounts"][k] == v for k, v in goal.get("balances", {}).items()]
        tests += [state["entries"][k]["status"] == v for k, v in goal.get("statuses", {}).items()]
        return sum(tests), all(tests)

    def violations(self, state, initial, constraints, cost):
        violations = [f"minimum_balance:{k}" for k, v in constraints.get("minimum_balances", {}).items() if state["accounts"][k] < v]
        violations += [f"forbidden_status:{k}" for k, e in state["entries"].items() if e["status"] in constraints.get("forbidden_statuses", [])]
        if "max_cost" in constraints and cost > constraints["max_cost"]:
            violations.append("max_cost")
        return sorted(violations)


class Blocksworld:
    operations = ("move", "noop")

    def validate(self, state, rules, goal, constraints):
        obj(rules, ("table", "max_height", "action_costs"), name="blocksworld rules")
        if type(rules["table"]) is not str or not rules["table"]:
            raise ValueError("table name must be nonempty string")
        integer(rules["max_height"], "max_height", 1)
        costs(rules, self.operations)
        obj(state, ("on",), name="blocksworld state")
        names(state["on"], "on", True)
        table = rules["table"]
        if table in state["on"]:
            raise ValueError("table collides with block name")
        supports = [v for v in state["on"].values() if v != table]
        if any(type(v) is not str or v not in state["on"] for v in supports) or len(set(supports)) != len(supports):
            raise ValueError("unknown support or branching stack")
        for block in state["on"]:
            path = set()
            cursor = block
            while cursor != table:
                if cursor in path:
                    raise ValueError("block cycle")
                path.add(cursor)
                cursor = state["on"][cursor]
            if len(path) > rules["max_height"]:
                raise ValueError("stack exceeds max_height")
        obj(goal, ("on",), name="blocksworld goal")
        names(goal["on"], "on goal", True)
        if not set(goal["on"]) <= set(state["on"]):
            raise ValueError("unknown goal block")
        for k, v in goal["on"].items():
            if type(v) is not str or v not in {*state["on"], table} or v == k:
                raise ValueError("invalid goal support")
        obj(constraints, (), ("forbidden_on", "protected_blocks", "max_cost"), name="block constraints")
        sequence(constraints.get("forbidden_on", []), "forbidden_on")
        pairs = []
        for pair in constraints.get("forbidden_on", []):
            if type(pair) is not list or len(pair) != 2 or type(pair[0]) is not str or type(pair[1]) is not str or pair[0] not in state["on"] or pair[1] not in {*state["on"], table}:
                raise ValueError("invalid forbidden_on")
            pairs.append(tuple(pair))
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate forbidden_on")
        unique_strings(constraints.get("protected_blocks", []), state["on"], "protected_blocks")
        if "max_cost" in constraints:
            integer(constraints["max_cost"], "max_cost", 0)

    def validate_action(self, action):
        if type(action) is not dict or type(action.get("op")) is not str or not action["op"]:
            raise ValueError("action requires string op")
        if action["op"] == "move":
            obj(action, ("op", "block", "destination"), name="move action")
            if any(type(action[k]) is not str or not action[k] for k in ("block", "destination")):
                raise ValueError("move names must be strings")
        else:
            obj(action, ("op",), name="noop/unsupported action")

    def transition(self, state, action, rules):
        if action["op"] == "noop":
            return deepcopy(state), None
        if action["op"] != "move":
            return state, "unsupported_operation"
        block, destination = action["block"], action["destination"]
        on, table = state["on"], rules["table"]
        if block not in on or destination not in {*on, table}:
            return state, "unknown_block"
        if block == destination:
            return state, "self_support"
        if on[block] == destination:
            return state, "already_at_destination"
        if block in on.values():
            return state, "source_not_clear"
        if destination != table and destination in on.values():
            return state, "destination_not_clear"
        result = deepcopy(state)
        result["on"][block] = destination
        height = 1
        cursor = destination
        while cursor != table:
            height += 1
            cursor = on[cursor]
        if height > rules["max_height"]:
            return state, "height_exceeded"
        return result, None

    def score(self, state, goal):
        tests = [state["on"][k] == v for k, v in goal["on"].items()]
        return sum(tests), all(tests)

    def violations(self, state, initial, constraints, cost):
        result = [f"forbidden_on:{a}:{b}" for a, b in constraints.get("forbidden_on", []) if state["on"][a] == b]
        result += [f"protected_block:{b}" for b in constraints.get("protected_blocks", []) if state["on"][b] != initial["on"][b]]
        if "max_cost" in constraints and cost > constraints["max_cost"]:
            result.append("max_cost")
        return sorted(result)


class Register:
    operations = ("add", "multiply", "set", "copy", "swap", "noop")

    def validate(self, state, rules, goal, constraints):
        obj(rules, ("value_bounds", "overflow", "operations", "action_costs"), name="register rules")
        bounds(rules["value_bounds"])
        if rules["overflow"] not in ("reject", "clamp", "wrap"):
            raise ValueError("invalid overflow rule")
        unique_strings(rules["operations"], self.operations, "operations")
        if not rules["operations"]:
            raise ValueError("empty operations")
        costs(rules, rules["operations"])
        obj(state, ("registers",), name="register state")
        names(state["registers"], "registers", True)
        lo, hi = rules["value_bounds"]
        for value in state["registers"].values():
            integer(value, "register value")
            if not lo <= value <= hi:
                raise ValueError("initial register outside bounds")
        obj(goal, ("values",), name="register goal")
        refs(goal["values"], state["registers"], "register goal")
        if not goal["values"]:
            raise ValueError("empty goal")
        obj(constraints, (), ("minimum_values", "maximum_values", "protected_registers", "max_cost"), name="register constraints")
        for key in ("minimum_values", "maximum_values"):
            refs(constraints.get(key, {}), state["registers"], key)
        unique_strings(constraints.get("protected_registers", []), state["registers"], "protected_registers")
        if "max_cost" in constraints:
            integer(constraints["max_cost"], "max_cost", 0)

    def validate_action(self, action):
        if type(action) is not dict or type(action.get("op")) is not str or not action["op"]:
            raise ValueError("action requires string op")
        op = action["op"]
        if op in ("add", "multiply", "set"):
            obj(action, ("op", "register", "value"), name="register action")
            integer(action["value"], "operand")
        elif op == "copy":
            obj(action, ("op", "source", "destination"), name="copy action")
        elif op == "swap":
            obj(action, ("op", "left", "right"), name="swap action")
        else:
            obj(action, ("op",), name="noop/unsupported action")
        if any(type(v) is not str or not v for k, v in action.items() if k not in ("value",)):
            raise ValueError("register action names must be strings")

    def transition(self, state, action, rules):
        op = action["op"]
        if op not in rules["operations"]:
            return state, "unsupported_operation"
        result = deepcopy(state)
        vals = result["registers"]
        if op == "noop":
            return result, None
        refs = [v for k, v in action.items() if k not in ("op", "value")]
        if any(key not in vals for key in refs):
            return state, "unknown_register"
        if op in ("add", "multiply", "set"):
            reg, operand = action["register"], action["value"]
            vals[reg] = {"add": lambda: vals[reg] + operand, "multiply": lambda: vals[reg] * operand, "set": lambda: operand}[op]()
        elif op == "copy":
            vals[action["destination"]] = vals[action["source"]]
        elif op == "swap":
            a, b = action["left"], action["right"]
            vals[a], vals[b] = vals[b], vals[a]
        lo, hi = rules["value_bounds"]
        for key, value in vals.items():
            if not lo <= value <= hi:
                if rules["overflow"] == "reject":
                    return state, "value_out_of_bounds"
                vals[key] = max(lo, min(hi, value)) if rules["overflow"] == "clamp" else lo + (value - lo) % (hi - lo + 1)
        return result, None

    def score(self, state, goal):
        tests = [state["registers"][k] == v for k, v in goal["values"].items()]
        return sum(tests), all(tests)

    def violations(self, state, initial, constraints, cost):
        vals = state["registers"]
        result = [f"minimum_value:{k}" for k, v in constraints.get("minimum_values", {}).items() if vals[k] < v]
        result += [f"maximum_value:{k}" for k, v in constraints.get("maximum_values", {}).items() if vals[k] > v]
        result += [f"protected_register:{k}" for k in constraints.get("protected_registers", []) if vals[k] != initial["registers"][k]]
        if "max_cost" in constraints and cost > constraints["max_cost"]:
            result.append("max_cost")
        return sorted(result)


def get_environment(name):
    try:
        return {"ledger": Ledger, "blocksworld": Blocksworld, "register": Register}[name]()
    except (KeyError, TypeError):
        raise ValueError("unsupported environment") from None
