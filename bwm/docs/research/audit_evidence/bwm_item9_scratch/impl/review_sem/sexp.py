"""Independent s-expression PDDL problem parser (reviewer)."""
import re
from pathlib import Path

def tokenize(txt):
    txt = re.sub(r";[^\n]*", "", txt)
    return re.findall(r"\(|\)|[^\s()]+", txt)

def parse(tokens):
    stack = [[]]
    for t in tokens:
        if t == "(":
            stack.append([])
        elif t == ")":
            x = stack.pop()
            stack[-1].append(x)
        else:
            stack[-1].append(t.lower())
    assert len(stack) == 1
    return stack[0]

def problem(path):
    tree = parse(tokenize(Path(path).read_text()))
    assert len(tree) == 1
    d = tree[0]
    assert d[0] == "define"
    sec = {}
    for x in d[1:]:
        sec[x[0]] = x[1:]
    objs = sec[":objects"]
    init = [tuple(f) for f in sec[":init"]]
    g = sec[":goal"]
    assert len(g) == 1
    g = g[0]
    if g[0] == "and":
        goal = [tuple(f) for f in g[1:]]
    else:
        goal = [tuple(g)]
    name = sec["problem"][0]
    dom = sec[":domain"][0]
    return name, dom, objs, init, goal
