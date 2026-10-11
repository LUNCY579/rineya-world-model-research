def step(S, a):
    S = set(S); op = a[0]
    if op == "pick-up":
        x = a[1]
        if not ({("clear", x), ("ontable", x), ("handempty",)} <= S): return None
        S -= {("clear", x), ("ontable", x), ("handempty",)}; S.add(("holding", x))
    elif op == "put-down":
        x = a[1]
        if ("holding", x) not in S: return None
        S.discard(("holding", x)); S |= {("clear", x), ("ontable", x), ("handempty",)}
    elif op == "stack":
        x, y = a[1], a[2]
        if not ({("holding", x), ("clear", y)} <= S) or x == y: return None
        S -= {("holding", x), ("clear", y)}; S |= {("on", x, y), ("clear", x), ("handempty",)}
    elif op == "unstack":
        x, y = a[1], a[2]
        if not ({("on", x, y), ("clear", x), ("handempty",)} <= S) or x == y: return None
        S -= {("on", x, y), ("clear", x), ("handempty",)}; S |= {("holding", x), ("clear", y)}
    else: return None
    return frozenset(S)
