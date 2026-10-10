"""Tests for the independent RCL oracle (rcx/rcl/oracle_indep.py), built from SPEC_RCL.md.

The oracle module is loaded by file path so that no other rcx code is executed.
"""
import ast as pyast
import importlib.util
import itertools
import json
import os
import random
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RCL_DIR = os.path.join(ROOT, "rcx", "rcl")
_spec = importlib.util.spec_from_file_location("oracle_indep_ut", os.path.join(RCL_DIR, "oracle_indep.py"))
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)


def S(op, *words):
    return {"op": op, "set": list(words)}


def K(*kinds):
    return {"op": "K+", "kinds": list(kinds)}


def T(op, t):
    return {"op": op, "t": t}


def C(op, a, b):
    return {"op": op, "args": [a, b]}


# (template_id, rule_ast, input text, ok, listed, ans, trace) -- all values computed by hand.
GOLD = [
    ("SPOS:t0", S("S+", "posted"),
     'RULE: Only events with status posted take effect; other events change nothing.\n'
     'STATE: {"A":10,"B":20,"C":30,"D":40}\n'
     'EVENT: {"account":"B","amount":5,"kind":"add","other":"C","status":"posted"}',
     True, True, '{"after":{"A":10,"B":25,"C":30,"D":40}}',
     '{"cmp":{"status":"posted","listed":true,"ok":true},"calc":{"B":25},"after":{"A":10,"B":25,"C":30,"D":40}}'),
    ("SPOS:t1", S("S+", "posted"),
     'RULE: An event is applied only if its status is posted.\n'
     'STATE: {"A":7,"B":3,"C":12,"D":0}\n'
     'EVENT: {"account":"A","amount":4,"kind":"transfer","other":"D","status":"void"}',
     False, False, '{"after":{"A":7,"B":3,"C":12,"D":0}}',
     '{"cmp":{"status":"void","listed":false,"ok":false},"calc":{"A":7,"D":0},"after":{"A":7,"B":3,"C":12,"D":0}}'),
    ("SPOS:t2", S("S+", "cleared", "settled"),
     'RULE: Apply the event when its status is cleared or settled; otherwise leave the balances unchanged.\n'
     'STATE: {"A":50,"B":20,"C":5,"D":9}\n'
     'EVENT: {"status":"settled","kind":"transfer","account":"A","other":"C","amount":15}',
     True, True, '{"after":{"A":35,"B":20,"C":20,"D":9}}',
     '{"cmp":{"status":"settled","listed":true,"ok":true},"calc":{"A":35,"C":20},"after":{"A":35,"B":20,"C":20,"D":9}}'),
    ("SPOS:t3", S("S+", "booked", "queued", "flagged"),
     'RULE: Events whose status is booked, queued or flagged update the balances. All other events are ignored.\n'
     'STATE: {"C":1,"A":2,"D":3,"B":4}\n'
     'EVENT: {"amount":6,"account":"D","status":"final","other":"A","kind":"add"}',
     False, False, '{"after":{"C":1,"A":2,"D":3,"B":4}}',
     '{"cmp":{"status":"final","listed":false,"ok":false},"calc":{"D":3},"after":{"C":1,"A":2,"D":3,"B":4}}'),
    ("SPOS:h0", S("S+", "lapsed", "pruned", "masked"),
     'RULE: If the status is lapsed, pruned or masked, carry out the event; for any other status, do nothing.\n'
     'STATE: {"P":100,"Q":200,"R":300,"S":400,"T":500}\n'
     'EVENT: {"account":"R","amount":-50,"kind":"add","other":"P","status":"masked"}',
     True, True, '{"after":{"P":100,"Q":200,"R":250,"S":400,"T":500}}',
     '{"cmp":{"status":"masked","listed":true,"ok":true},"calc":{"R":250},"after":{"P":100,"Q":200,"R":250,"S":400,"T":500}}'),
    ("SPOS:h1", S("S+", "parked", "routed"),
     'RULE: Balances change only for parked or routed events.\n'
     'STATE: {"west":5,"south":6,"pine":7,"oak":8,"north":9,"east":10}\n'
     'EVENT: {"status":"parked","other":"oak","kind":"transfer","amount":3,"account":"north"}',
     True, True, '{"after":{"west":5,"south":6,"pine":7,"oak":11,"north":6,"east":10}}',
     '{"cmp":{"status":"parked","listed":true,"ok":true},"calc":{"north":6,"oak":11},'
     '"after":{"west":5,"south":6,"pine":7,"oak":11,"north":6,"east":10}}'),
    ("SNEG:t0", S("S-", "void"),
     'RULE: Events with status void are ignored; all other events take effect.\n'
     'STATE: {"A":1,"B":1,"C":1,"D":1}\n'
     'EVENT: {"account":"C","amount":2,"kind":"add","other":"A","status":"void"}',
     False, True, '{"after":{"A":1,"B":1,"C":1,"D":1}}',
     '{"cmp":{"status":"void","listed":true,"ok":false},"calc":{"C":1},"after":{"A":1,"B":1,"C":1,"D":1}}'),
    ("SNEG:t0", S("S-", "void"),
     'RULE: Events with status void are ignored; all other events take effect.\n'
     'STATE: {"A":1,"B":1,"C":1,"D":1}\n'
     'EVENT: {"account":"C","amount":2,"kind":"add","other":"A","status":"Void"}',
     True, False, '{"after":{"A":1,"B":1,"C":3,"D":1}}',
     '{"cmp":{"status":"Void","listed":false,"ok":true},"calc":{"C":3},"after":{"A":1,"B":1,"C":3,"D":1}}'),
    ("SNEG:t1", S("S-", "rejected", "disputed"),
     'RULE: An event is applied unless its status is rejected or disputed.\n'
     'STATE: {"A":0,"B":0,"C":0,"D":0}\n'
     'EVENT: {"kind":"transfer","status":"paid","account":"B","other":"A","amount":7}',
     True, False, '{"after":{"A":7,"B":-7,"C":0,"D":0}}',
     '{"cmp":{"status":"paid","listed":false,"ok":true},"calc":{"B":-7,"A":7},"after":{"A":7,"B":-7,"C":0,"D":0}}'),
    ("SNEG:t2", S("S-", "frozen", "paid", "billed"),
     'RULE: Skip the event when its status is frozen, paid or billed; otherwise apply it.\n'
     'STATE: {"D":40,"C":30,"B":20,"A":10}\n'
     'EVENT: {"other":"B","amount":5,"status":"billed","account":"C","kind":"transfer"}',
     False, True, '{"after":{"D":40,"C":30,"B":20,"A":10}}',
     '{"cmp":{"status":"billed","listed":true,"ok":false},"calc":{"C":30,"B":20},"after":{"D":40,"C":30,"B":20,"A":10}}'),
    ("SNEG:t3", S("S-", "void"),
     'RULE: Events whose status is void change nothing. All other events update the balances.\n'
     'STATE: {"A":5,"B":6,"C":7,"D":8}\n'
     'EVENT: {"account":"A","amount":2,"kind":"add","other":"B","status":"voided"}',
     True, False, '{"after":{"A":7,"B":6,"C":7,"D":8}}',
     '{"cmp":{"status":"voided","listed":false,"ok":true},"calc":{"A":7},"after":{"A":7,"B":6,"C":7,"D":8}}'),
    ("SNEG:h0", S("S-", "tagged", "minted"),
     'RULE: If the status is tagged or minted, do nothing; for any other status, carry out the event.\n'
     'STATE: {"P":1,"Q":2,"R":3,"S":4,"T":5,"U":6}\n'
     'EVENT: {"account":"U","amount":10,"kind":"add","other":"P","status":"minted"}',
     False, True, '{"after":{"P":1,"Q":2,"R":3,"S":4,"T":5,"U":6}}',
     '{"cmp":{"status":"minted","listed":true,"ok":false},"calc":{"U":6},"after":{"P":1,"Q":2,"R":3,"S":4,"T":5,"U":6}}'),
    ("SNEG:h1", S("S-", "void"),
     'RULE: Every event takes effect except void events.\n'
     'STATE: {"north":10,"south":20,"east":30,"west":40}\n'
     'EVENT: {"account":"east","amount":5,"kind":"transfer","other":"west","status":"posted"}',
     True, False, '{"after":{"north":10,"south":20,"east":25,"west":45}}',
     '{"cmp":{"status":"posted","listed":false,"ok":true},"calc":{"east":25,"west":45},'
     '"after":{"north":10,"south":20,"east":25,"west":45}}'),
    ("AND:t0", C("AND", S("S+", "posted"), K("add")),
     'RULE: An event takes effect only if its status is in [posted] and its kind is add; otherwise nothing changes.\n'
     'STATE: {"A":10,"B":20,"C":30,"D":40}\n'
     'EVENT: {"account":"A","amount":5,"kind":"transfer","other":"B","status":"posted"}',
     False, True, '{"after":{"A":10,"B":20,"C":30,"D":40}}',
     '{"cmp":{"status":"posted","listed":true,"ok":false},"calc":{"A":10,"B":20},"after":{"A":10,"B":20,"C":30,"D":40}}'),
    ("AND:t1", C("AND", S("S-", "void", "closed"), T("AGE", 10)),
     'RULE: Apply the event when its status is not in [void, closed] and its amount is at least 10. '
     'Otherwise leave the balances unchanged.\n'
     'STATE: {"A":100,"B":0,"C":0,"D":0}\n'
     'EVENT: {"account":"A","amount":10,"kind":"transfer","other":"D","status":"open"}',
     True, False, '{"after":{"A":90,"B":0,"C":0,"D":10}}',
     '{"cmp":{"status":"open","listed":false,"ok":true},"calc":{"A":90,"D":10},"after":{"A":90,"B":0,"C":0,"D":10}}'),
    ("AND:h0", C("AND", S("S+", "cleared", "settled", "booked"), T("ALT", 5)),
     'RULE: Both conditions must hold for an event to take effect: its status is in [cleared, settled, booked]; '
     'its amount is below 5.\n'
     'STATE: {"A":3,"B":4,"C":5,"D":6}\n'
     'EVENT: {"account":"B","amount":5,"kind":"add","other":"C","status":"booked"}',
     False, True, '{"after":{"A":3,"B":4,"C":5,"D":6}}',
     '{"cmp":{"status":"booked","listed":true,"ok":false},"calc":{"B":4},"after":{"A":3,"B":4,"C":5,"D":6}}'),
    ("OR:t0", C("OR", S("S+", "posted"), T("BGE", 50)),
     'RULE: An event takes effect if its status is in [posted] or if the account balance is at least 50; '
     'otherwise nothing changes.\n'
     'STATE: {"A":50,"B":49,"C":0,"D":0}\n'
     'EVENT: {"account":"A","amount":20,"kind":"transfer","other":"B","status":"void"}',
     True, False, '{"after":{"A":30,"B":69,"C":0,"D":0}}',
     '{"cmp":{"status":"void","listed":false,"ok":true},"calc":{"A":30,"B":69},"after":{"A":30,"B":69,"C":0,"D":0}}'),
    ("OR:t0", C("OR", S("S+", "posted"), T("BGE", 50)),
     'RULE: An event takes effect if its status is in [posted] or if the account balance is at least 50; '
     'otherwise nothing changes.\n'
     'STATE: {"A":50,"B":49,"C":0,"D":0}\n'
     'EVENT: {"account":"B","amount":20,"kind":"transfer","other":"A","status":"void"}',
     False, False, '{"after":{"A":50,"B":49,"C":0,"D":0}}',
     '{"cmp":{"status":"void","listed":false,"ok":false},"calc":{"B":49,"A":50},"after":{"A":50,"B":49,"C":0,"D":0}}'),
    ("OR:t1", C("OR", S("S-", "locked"), K("transfer")),
     'RULE: Apply the event when its status is not in [locked] or when its kind is transfer. '
     'Otherwise leave the balances unchanged.\n'
     'STATE: {"A":1,"B":2,"C":3,"D":4}\n'
     'EVENT: {"account":"D","amount":1,"kind":"add","other":"C","status":"locked"}',
     False, True, '{"after":{"A":1,"B":2,"C":3,"D":4}}',
     '{"cmp":{"status":"locked","listed":true,"ok":false},"calc":{"D":4},"after":{"A":1,"B":2,"C":3,"D":4}}'),
    ("OR:h0", C("OR", K("add", "transfer"), S("S+", "waived")),
     'RULE: An event takes effect when either its kind is add or transfer or its status is in [waived].\n'
     'STATE: {"A":0,"B":0,"C":0,"D":0}\n'
     'EVENT: {"account":"A","amount":3,"kind":"add","other":"B","status":"Waived"}',
     True, False, '{"after":{"A":3,"B":0,"C":0,"D":0}}',
     '{"cmp":{"status":"Waived","listed":false,"ok":true},"calc":{"A":3},"after":{"A":3,"B":0,"C":0,"D":0}}'),
    ("ATOM:t0", K("transfer"),
     'RULE: An event takes effect only if its kind is transfer; otherwise nothing changes.\n'
     'STATE: {"A":10,"B":10,"C":10,"D":10}\n'
     'EVENT: {"account":"B","amount":4,"kind":"transfer","other":"C","status":"posted"}',
     True, None, '{"after":{"A":10,"B":6,"C":14,"D":10}}',
     '{"cmp":{"status":"posted","listed":null,"ok":true},"calc":{"B":6,"C":14},"after":{"A":10,"B":6,"C":14,"D":10}}'),
    ("ATOM:t0", T("AGE", -3),
     'RULE: An event takes effect only if its amount is at least -3; otherwise nothing changes.\n'
     'STATE: {"A":5,"B":5,"C":5,"D":5}\n'
     'EVENT: {"account":"C","amount":-4,"kind":"add","other":"D","status":"x"}',
     False, None, '{"after":{"A":5,"B":5,"C":5,"D":5}}',
     '{"cmp":{"status":"x","listed":null,"ok":false},"calc":{"C":5},"after":{"A":5,"B":5,"C":5,"D":5}}'),
    ("ATOM:t0", T("ALT", 0),
     'RULE: An event takes effect only if its amount is below 0; otherwise nothing changes.\n'
     'STATE: {"A":5,"B":5,"C":5,"D":5}\n'
     'EVENT: {"account":"D","amount":-1,"kind":"add","other":"A","status":"posted"}',
     True, None, '{"after":{"A":5,"B":5,"C":5,"D":4}}',
     '{"cmp":{"status":"posted","listed":null,"ok":true},"calc":{"D":4},"after":{"A":5,"B":5,"C":5,"D":4}}'),
    ("ATOM:t0", T("BGE", 100),
     'RULE: An event takes effect only if the account balance is at least 100; otherwise nothing changes.\n'
     'STATE: {"oak":100,"pine":99,"east":0,"west":1,"north":2}\n'
     'EVENT: {"account":"pine","amount":1,"kind":"add","other":"oak","status":"posted"}',
     False, None, '{"after":{"oak":100,"pine":99,"east":0,"west":1,"north":2}}',
     '{"cmp":{"status":"posted","listed":null,"ok":false},"calc":{"pine":99},'
     '"after":{"oak":100,"pine":99,"east":0,"west":1,"north":2}}'),
    ("AND:t0", C("AND", T("AGE", 2), S("S-", "ab_1", "c-d")),
     'RULE: An event takes effect only if its amount is at least 2 and its status is not in [ab_1, c-d]; '
     'otherwise nothing changes.\n'
     'STATE: {"A":0,"B":0,"C":0,"D":0}\n'
     'EVENT: {"account":"C","amount":2,"kind":"add","other":"A","status":"c-d"}',
     False, True, '{"after":{"A":0,"B":0,"C":0,"D":0}}',
     '{"cmp":{"status":"c-d","listed":true,"ok":false},"calc":{"C":0},"after":{"A":0,"B":0,"C":0,"D":0}}'),
    ("AND:h0", C("AND", T("BGE", 0), K("add", "transfer")),
     'RULE: Both conditions must hold for an event to take effect: the account balance is at least 0; '
     'its kind is add or transfer.\n'
     'STATE: {"A":0,"B":5,"C":0,"D":0}\n'
     'EVENT: {"account":"A","amount":2,"kind":"transfer","other":"B","status":"posted"}',
     True, None, '{"after":{"A":-2,"B":7,"C":0,"D":0}}',
     '{"cmp":{"status":"posted","listed":null,"ok":true},"calc":{"A":-2,"B":7},"after":{"A":-2,"B":7,"C":0,"D":0}}'),
    ("OR:h0", C("OR", S("S+", "open"), S("S-", "closed")),
     'RULE: An event takes effect when either its status is in [open] or its status is not in [closed].\n'
     'STATE: {"A":1,"B":2,"C":3,"D":4}\n'
     'EVENT: {"account":"A","amount":1,"kind":"add","other":"B","status":"closed"}',
     False, False, '{"after":{"A":1,"B":2,"C":3,"D":4}}',
     '{"cmp":{"status":"closed","listed":false,"ok":false},"calc":{"A":1},"after":{"A":1,"B":2,"C":3,"D":4}}'),
    ("SPOS:t1", S("S+", "or", "and", "in"),
     'RULE: An event is applied only if its status is or, and or in.\n'
     'STATE: {"A":1,"B":2,"C":3,"D":4}\n'
     'EVENT: {"account":"B","amount":1,"kind":"add","other":"A","status":"and"}',
     True, True, '{"after":{"A":1,"B":3,"C":3,"D":4}}',
     '{"cmp":{"status":"and","listed":true,"ok":true},"calc":{"B":3},"after":{"A":1,"B":3,"C":3,"D":4}}'),
    ("AND:t1", C("AND", S("S+", "Posted", "POSTED"), T("ALT", -2)),
     'RULE: Apply the event when its status is in [Posted, POSTED] and its amount is below -2. '
     'Otherwise leave the balances unchanged.\n'
     'STATE: {"B":-5,"A":8,"D":0,"C":1}\n'
     'EVENT: {"account":"A","other":"B","kind":"transfer","status":"POSTED","amount":-3}',
     True, True, '{"after":{"B":-8,"A":11,"D":0,"C":1}}',
     '{"cmp":{"status":"POSTED","listed":true,"ok":true},"calc":{"A":11,"B":-8},"after":{"B":-8,"A":11,"D":0,"C":1}}'),
]

# Copy-world variant (SPEC section 1.1): set / swap. Same tuple layout, computed by hand.
COPY_GOLD = [
    ("SPOS:t0", S("S+", "posted"),
     'RULE: Only events with status posted take effect; other events change nothing.\n'
     'STATE: {"A":10,"B":20,"C":30,"D":40}\n'
     'EVENT: {"account":"B","amount":7,"kind":"set","other":"C","status":"posted"}',
     True, True, '{"after":{"A":10,"B":7,"C":30,"D":40}}',
     '{"cmp":{"status":"posted","listed":true,"ok":true},"calc":{"B":7},"after":{"A":10,"B":7,"C":30,"D":40}}'),
    ("SPOS:t0", S("S+", "posted"),
     'RULE: Only events with status posted take effect; other events change nothing.\n'
     'STATE: {"A":10,"B":20,"C":30,"D":40}\n'
     'EVENT: {"account":"C","amount":30,"kind":"set","other":"A","status":"posted"}',
     True, True, '{"after":{"A":10,"B":20,"C":30,"D":40}}',
     '{"cmp":{"status":"posted","listed":true,"ok":true},"calc":{"C":30},"after":{"A":10,"B":20,"C":30,"D":40}}'),
    ("SPOS:t1", S("S+", "posted"),
     'RULE: An event is applied only if its status is posted.\n'
     'STATE: {"A":5,"B":6,"C":7,"D":8}\n'
     'EVENT: {"account":"C","amount":0,"kind":"set","other":"A","status":"void"}',
     False, False, '{"after":{"A":5,"B":6,"C":7,"D":8}}',
     '{"cmp":{"status":"void","listed":false,"ok":false},"calc":{"C":7},"after":{"A":5,"B":6,"C":7,"D":8}}'),
    ("SPOS:t2", S("S+", "cleared", "settled"),
     'RULE: Apply the event when its status is cleared or settled; otherwise leave the balances unchanged.\n'
     'STATE: {"A":11,"B":22,"C":33,"D":44}\n'
     'EVENT: {"status":"settled","kind":"swap","account":"A","other":"C","amount":3}',
     True, True, '{"after":{"A":33,"B":22,"C":11,"D":44}}',
     '{"cmp":{"status":"settled","listed":true,"ok":true},"calc":{"A":33,"C":11},"after":{"A":33,"B":22,"C":11,"D":44}}'),
    ("SPOS:t3", S("S+", "booked", "queued", "flagged"),
     'RULE: Events whose status is booked, queued or flagged update the balances. All other events are ignored.\n'
     'STATE: {"C":1,"A":2,"D":3,"B":4}\n'
     'EVENT: {"amount":6,"account":"D","status":"final","other":"A","kind":"swap"}',
     False, False, '{"after":{"C":1,"A":2,"D":3,"B":4}}',
     '{"cmp":{"status":"final","listed":false,"ok":false},"calc":{"D":3,"A":2},"after":{"C":1,"A":2,"D":3,"B":4}}'),
    ("SPOS:h0", S("S+", "lapsed", "pruned"),
     'RULE: If the status is lapsed or pruned, carry out the event; for any other status, do nothing.\n'
     'STATE: {"U":6,"R":3,"P":1,"T":5,"Q":2,"S":4}\n'
     'EVENT: {"account":"R","amount":12,"kind":"swap","other":"S","status":"pruned"}',
     True, True, '{"after":{"U":6,"R":4,"P":1,"T":5,"Q":2,"S":3}}',
     '{"cmp":{"status":"pruned","listed":true,"ok":true},"calc":{"R":4,"S":3},'
     '"after":{"U":6,"R":4,"P":1,"T":5,"Q":2,"S":3}}'),
    ("SPOS:h1", S("S+", "parked", "routed"),
     'RULE: Balances change only for parked or routed events.\n'
     'STATE: {"P":-5,"Q":17,"R":0,"S":42}\n'
     'EVENT: {"account":"Q","amount":512,"kind":"set","other":"S","status":"routed"}',
     True, True, '{"after":{"P":-5,"Q":512,"R":0,"S":42}}',
     '{"cmp":{"status":"routed","listed":true,"ok":true},"calc":{"Q":512},"after":{"P":-5,"Q":512,"R":0,"S":42}}'),
    ("SNEG:t0", S("S-", "void"),
     'RULE: Events with status void are ignored; all other events take effect.\n'
     'STATE: {"A":1,"B":2,"C":3,"D":4}\n'
     'EVENT: {"account":"D","amount":98,"kind":"set","other":"B","status":"posted"}',
     True, False, '{"after":{"A":1,"B":2,"C":3,"D":98}}',
     '{"cmp":{"status":"posted","listed":false,"ok":true},"calc":{"D":98},"after":{"A":1,"B":2,"C":3,"D":98}}'),
    ("SNEG:t1", S("S-", "rejected", "disputed"),
     'RULE: An event is applied unless its status is rejected or disputed.\n'
     'STATE: {"A":50,"B":60,"C":70,"D":80}\n'
     'EVENT: {"kind":"set","status":"disputed","account":"A","other":"B","amount":12}',
     False, True, '{"after":{"A":50,"B":60,"C":70,"D":80}}',
     '{"cmp":{"status":"disputed","listed":true,"ok":false},"calc":{"A":50},"after":{"A":50,"B":60,"C":70,"D":80}}'),
    ("SNEG:t2", S("S-", "frozen", "paid", "billed"),
     'RULE: Skip the event when its status is frozen, paid or billed; otherwise apply it.\n'
     'STATE: {"D":40,"B":20,"A":10,"C":30}\n'
     'EVENT: {"other":"D","amount":5,"status":"open","account":"C","kind":"swap"}',
     True, False, '{"after":{"D":30,"B":20,"A":10,"C":40}}',
     '{"cmp":{"status":"open","listed":false,"ok":true},"calc":{"C":40,"D":30},"after":{"D":30,"B":20,"A":10,"C":40}}'),
    ("SNEG:t3", S("S-", "void"),
     'RULE: Events whose status is void change nothing. All other events update the balances.\n'
     'STATE: {"A":9,"B":8,"C":7,"D":6}\n'
     'EVENT: {"account":"B","amount":1,"kind":"swap","other":"D","status":"void"}',
     False, True, '{"after":{"A":9,"B":8,"C":7,"D":6}}',
     '{"cmp":{"status":"void","listed":true,"ok":false},"calc":{"B":8,"D":6},"after":{"A":9,"B":8,"C":7,"D":6}}'),
    ("SNEG:h0", S("S-", "tagged", "minted"),
     'RULE: If the status is tagged or minted, do nothing; for any other status, carry out the event.\n'
     'STATE: {"oak":3,"pine":-8,"east":5,"west":0,"north":2,"south":9}\n'
     'EVENT: {"kind":"swap","account":"south","amount":7,"status":"boxed","other":"pine"}',
     True, False, '{"after":{"oak":3,"pine":9,"east":5,"west":0,"north":2,"south":-8}}',
     '{"cmp":{"status":"boxed","listed":false,"ok":true},"calc":{"south":-8,"pine":9},'
     '"after":{"oak":3,"pine":9,"east":5,"west":0,"north":2,"south":-8}}'),
    ("SNEG:h1", S("S-", "void"),
     'RULE: Every event takes effect except void events.\n'
     'STATE: {"west":5,"south":6,"pine":7,"oak":8,"north":9}\n'
     'EVENT: {"status":"Void","other":"oak","kind":"set","amount":0,"account":"north"}',
     True, False, '{"after":{"west":5,"south":6,"pine":7,"oak":8,"north":0}}',
     '{"cmp":{"status":"Void","listed":false,"ok":true},"calc":{"north":0},'
     '"after":{"west":5,"south":6,"pine":7,"oak":8,"north":0}}'),
    ("AND:t0", C("AND", S("S+", "posted"), K("set")),
     'RULE: An event takes effect only if its status is in [posted] and its kind is set; otherwise nothing changes.\n'
     'STATE: {"A":10,"B":20,"C":30,"D":40}\n'
     'EVENT: {"account":"A","amount":2,"kind":"swap","other":"D","status":"posted"}',
     False, True, '{"after":{"A":10,"B":20,"C":30,"D":40}}',
     '{"cmp":{"status":"posted","listed":true,"ok":false},"calc":{"A":10,"D":40},"after":{"A":10,"B":20,"C":30,"D":40}}'),
    ("AND:t1", C("AND", S("S-", "void"), K("set", "swap")),
     'RULE: Apply the event when its status is not in [void] and its kind is set or swap. '
     'Otherwise leave the balances unchanged.\n'
     'STATE: {"A":0,"B":0,"C":0,"D":0}\n'
     'EVENT: {"account":"C","amount":45,"kind":"set","other":"B","status":"open"}',
     True, False, '{"after":{"A":0,"B":0,"C":45,"D":0}}',
     '{"cmp":{"status":"open","listed":false,"ok":true},"calc":{"C":45},"after":{"A":0,"B":0,"C":45,"D":0}}'),
    ("AND:h0", C("AND", K("swap"), S("S+", "cleared")),
     'RULE: Both conditions must hold for an event to take effect: its kind is swap; its status is in [cleared].\n'
     'STATE: {"B":2,"D":4,"A":1,"C":3}\n'
     'EVENT: {"account":"B","amount":4,"kind":"swap","other":"C","status":"cleared"}',
     True, True, '{"after":{"B":3,"D":4,"A":1,"C":2}}',
     '{"cmp":{"status":"cleared","listed":true,"ok":true},"calc":{"B":3,"C":2},"after":{"B":3,"D":4,"A":1,"C":2}}'),
    ("OR:t0", C("OR", S("S+", "posted"), K("swap")),
     'RULE: An event takes effect if its status is in [posted] or if its kind is swap; otherwise nothing changes.\n'
     'STATE: {"A":15,"B":25,"C":35,"D":45}\n'
     'EVENT: {"account":"D","amount":9,"kind":"swap","other":"A","status":"void"}',
     True, False, '{"after":{"A":45,"B":25,"C":35,"D":15}}',
     '{"cmp":{"status":"void","listed":false,"ok":true},"calc":{"D":15,"A":45},"after":{"A":45,"B":25,"C":35,"D":15}}'),
    ("OR:t1", C("OR", S("S-", "locked"), K("add", "set")),
     'RULE: Apply the event when its status is not in [locked] or when its kind is add or set. '
     'Otherwise leave the balances unchanged.\n'
     'STATE: {"A":1,"B":2,"C":3,"D":4}\n'
     'EVENT: {"account":"A","amount":5,"kind":"swap","other":"C","status":"locked"}',
     False, True, '{"after":{"A":1,"B":2,"C":3,"D":4}}',
     '{"cmp":{"status":"locked","listed":true,"ok":false},"calc":{"A":1,"C":3},"after":{"A":1,"B":2,"C":3,"D":4}}'),
    ("OR:h0", C("OR", K("set", "swap"), S("S+", "waived")),
     'RULE: An event takes effect when either its kind is set or swap or its status is in [waived].\n'
     'STATE: {"A":3,"B":4,"C":5,"D":6}\n'
     'EVENT: {"account":"B","amount":60,"kind":"set","other":"A","status":"Waived"}',
     True, False, '{"after":{"A":3,"B":60,"C":5,"D":6}}',
     '{"cmp":{"status":"Waived","listed":false,"ok":true},"calc":{"B":60},"after":{"A":3,"B":60,"C":5,"D":6}}'),
    ("OR:h0", C("OR", S("S+", "set"), K("swap")),
     'RULE: An event takes effect when either its status is in [set] or its kind is swap.\n'
     'STATE: {"A":7,"B":8,"C":9,"D":10}\n'
     'EVENT: {"account":"C","amount":1,"kind":"set","other":"D","status":"swap"}',
     False, False, '{"after":{"A":7,"B":8,"C":9,"D":10}}',
     '{"cmp":{"status":"swap","listed":false,"ok":false},"calc":{"C":9},"after":{"A":7,"B":8,"C":9,"D":10}}'),
    ("ATOM:t0", K("set"),
     'RULE: An event takes effect only if its kind is set; otherwise nothing changes.\n'
     'STATE: {"A":10,"B":10,"C":10,"D":10}\n'
     'EVENT: {"account":"A","amount":33,"kind":"set","other":"B","status":"posted"}',
     True, None, '{"after":{"A":33,"B":10,"C":10,"D":10}}',
     '{"cmp":{"status":"posted","listed":null,"ok":true},"calc":{"A":33},"after":{"A":33,"B":10,"C":10,"D":10}}'),
    ("ATOM:t0", K("swap"),
     'RULE: An event takes effect only if its kind is swap; otherwise nothing changes.\n'
     'STATE: {"A":1,"B":2,"C":3,"D":4}\n'
     'EVENT: {"account":"A","amount":2,"kind":"set","other":"B","status":"posted"}',
     False, None, '{"after":{"A":1,"B":2,"C":3,"D":4}}',
     '{"cmp":{"status":"posted","listed":null,"ok":false},"calc":{"A":1},"after":{"A":1,"B":2,"C":3,"D":4}}'),
    ("ATOM:t0", T("AGE", 10),
     'RULE: An event takes effect only if its amount is at least 10; otherwise nothing changes.\n'
     'STATE: {"A":1,"B":2,"C":3,"D":4}\n'
     'EVENT: {"account":"D","amount":12,"kind":"swap","other":"B","status":"x"}',
     True, None, '{"after":{"A":1,"B":4,"C":3,"D":2}}',
     '{"cmp":{"status":"x","listed":null,"ok":true},"calc":{"D":2,"B":4},"after":{"A":1,"B":4,"C":3,"D":2}}'),
    ("ATOM:t0", T("BGE", 50),
     'RULE: An event takes effect only if the account balance is at least 50; otherwise nothing changes.\n'
     'STATE: {"A":50,"B":0,"C":0,"D":0}\n'
     'EVENT: {"account":"A","amount":5,"kind":"set","other":"B","status":"posted"}',
     True, None, '{"after":{"A":5,"B":0,"C":0,"D":0}}',
     '{"cmp":{"status":"posted","listed":null,"ok":true},"calc":{"A":5},"after":{"A":5,"B":0,"C":0,"D":0}}'),
]

ALL_GOLD = GOLD + COPY_GOLD

BASE = GOLD[0][2]
BASE_RULE, BASE_STATE, BASE_EVENT = BASE.split("\n")


class GoldCases(unittest.TestCase):
    def test_count_and_coverage(self):
        self.assertGreaterEqual(len(GOLD), 25)
        self.assertEqual(set(O.TEMPLATE_IDS), {g[0] for g in GOLD})
        ops = {g[1]["op"] for g in GOLD} | {a["op"] for g in GOLD for a in g[1].get("args", [])}
        self.assertEqual(ops, {"S+", "S-", "K+", "AGE", "ALT", "BGE", "AND", "OR"})

    def test_parse_input(self):
        for tid, ast, text, *_ in ALL_GOLD:
            with self.subTest(text=text):
                p = O.parse_input(text)
                self.assertEqual(p["template_id"], tid)
                self.assertEqual(p["rule_ast"], ast)
                state_s, event_s = text.split("\n")[1][7:], text.split("\n")[2][7:]
                self.assertEqual(list(p["state"].items()), list(json.loads(state_s).items()))
                self.assertEqual(list(p["event"].items()), list(json.loads(event_s).items()))
                self.assertEqual(p["state_order"], list(json.loads(state_s)))
                self.assertEqual(O.render_input(ast, p["state"], p["event"], tid), text)

    def test_gold_targets(self):
        for tid, ast, text, ok, listed, ans, trace in ALL_GOLD:
            with self.subTest(text=text):
                g = O.gold_targets(text)
                self.assertIs(g["ok"], ok)
                self.assertIs(g["listed"], listed)
                self.assertEqual(g["ans"], ans)
                self.assertEqual(g["trace"], trace)
                self.assertEqual(json.dumps({"after": g["after"]}, separators=(",", ":")), ans)

    def test_targets_parse_back(self):
        for tid, ast, text, ok, listed, ans, trace in ALL_GOLD:
            with self.subTest(text=text):
                pa = O.parse_target(ans, "ans", source=text)
                pt = O.parse_target(trace, "trace", source=text)
                self.assertEqual(pa, json.loads(ans))
                self.assertEqual(pt, json.loads(trace))
                self.assertEqual(pt["cmp"], {"status": O.parse_input(text)["event"]["status"],
                                             "listed": listed, "ok": ok})


class MalformedInputs(unittest.TestCase):
    RULES = [
        "Only events with status posted take effect; other events change nothing",
        "Only events with status posted takes effect; other events change nothing.",
        "only events with status posted take effect; other events change nothing.",
        "Only events with status posted or posted take effect; other events change nothing.",
        "Only events with status a, b, c or d take effect; other events change nothing.",
        "Only events with status a, b take effect; other events change nothing.",
        "Only events with status a and b take effect; other events change nothing.",
        "Only events with status a or b or c take effect; other events change nothing.",
        "Only events with status a,b or c take effect; other events change nothing.",
        'Only events with status "posted" take effect; other events change nothing.',
        "Only events with status post.ed take effect; other events change nothing.",
        "Only events with status  take effect; other events change nothing.",
        "Only events with status posted  take effect; other events change nothing.",
        "Only events with status [posted] take effect; other events change nothing.",
        "An event takes effect only if its status is in [a, b, c, d] and its kind is add; otherwise nothing changes.",
        "An event takes effect only if its status is in [a, a] and its kind is add; otherwise nothing changes.",
        "An event takes effect only if its status is in [a, b ] and its kind is add; otherwise nothing changes.",
        "An event takes effect only if its status is in [] and its kind is add; otherwise nothing changes.",
        "An event takes effect only if its status is in [a] and its kind is add or add; otherwise nothing changes.",
        "An event takes effect only if its status is in [a] and its kind is scale; otherwise nothing changes.",
        "An event takes effect only if its status is in [a] and its kind is add, transfer; otherwise nothing changes.",
        "An event takes effect only if its amount is at least 05; otherwise nothing changes.",
        "An event takes effect only if its amount is at least -0; otherwise nothing changes.",
        "An event takes effect only if its amount is at least +5; otherwise nothing changes.",
        "An event takes effect only if its amount is at least 5.0; otherwise nothing changes.",
        "An event takes effect only if its amount is at least ５; otherwise nothing changes.",
        "An event takes effect only if its status is in [a]; otherwise nothing changes.",
        "An event takes effect only if its status is not in [a]; otherwise nothing changes.",
        "An event takes effect only if its amount is at least 5 and its kind is add and its kind is add; "
        "otherwise nothing changes.",
        "An event takes effect only if its amount is at least 5 or its kind is add; otherwise nothing changes.",
        "Both conditions must hold for an event to take effect: its amount is below 5.",
        "An event takes effect when either its amount is below 5 or its amount is below 5",
        # K+ clauses naming the copy-world kinds (SPEC 1.1): only distinct names from {add, transfer, set, swap}.
        "An event takes effect only if its kind is set or set; otherwise nothing changes.",
        "An event takes effect only if its kind is Set; otherwise nothing changes.",
        "An event takes effect only if its kind is SWAP; otherwise nothing changes.",
        "An event takes effect only if its kind is set, swap; otherwise nothing changes.",
        "An event takes effect only if its kind is set and swap; otherwise nothing changes.",
        "An event takes effect only if its kind is set or  swap; otherwise nothing changes.",
        "An event takes effect only if its kind is set or swap or; otherwise nothing changes.",
        "An event takes effect only if its kind is [set]; otherwise nothing changes.",
        "An event takes effect only if its kind is copy; otherwise nothing changes.",
        "An event takes effect only if its kind is swapped; otherwise nothing changes.",
        "An event takes effect only if its kind is swap or scale; otherwise nothing changes.",
        "An event takes effect only if its kind is add or transfer or set or swap or add; otherwise nothing changes.",
        "An event takes effect only if its status is in [a] and its kind is set or set; otherwise nothing changes.",
        "An event takes effect if its status is in [a] or if its kind is swap or swap; otherwise nothing changes.",
        "An event takes effect when either its kind is set or swap or its kind is scale.",
        "Both conditions must hold for an event to take effect: its kind is set; its kind is sets.",
    ]

    def test_bad_rules(self):
        for rule in self.RULES:
            with self.subTest(rule=rule):
                with self.assertRaises(ValueError):
                    O.parse_input(f"RULE: {rule}\n{BASE_STATE}\n{BASE_EVENT}")

    def test_bad_layout(self):
        texts = [
            BASE + "\n", BASE.replace("\n", "\r\n"), "\n".join([BASE_STATE, BASE_RULE, BASE_EVENT]),
            BASE.replace("RULE: ", "RULE:  "), BASE.replace("RULE: ", "Rule: "), BASE.replace("STATE: ", "STATE:"),
            "\n".join([BASE_RULE, BASE_STATE]), BASE + "\nEXTRA: 1", " " + BASE, "",
        ]
        for text in texts:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    O.parse_input(text)

    def test_bad_state_event(self):
        ev = 'EVENT: {"account":"B","amount":5,"kind":"add","other":"C","status":"posted"}'
        bad = [
            ('STATE: {"A": 10,"B":20,"C":30,"D":40}', ev),
            ('STATE: {"A":10.0,"B":20,"C":30,"D":40}', ev),
            ('STATE: {"A":1e1,"B":20,"C":30,"D":40}', ev),
            ('STATE: {"A":"10","B":20,"C":30,"D":40}', ev),
            ('STATE: {"A":true,"B":20,"C":30,"D":40}', ev),
            ('STATE: {"A":NaN,"B":20,"C":30,"D":40}', ev),
            ('STATE: {"A":10,"B":20,"B":30,"C":40}', ev),
            ('STATE: {"A":-0,"B":20,"C":30,"D":40}', ev),
            ('STATE: {"B":20}', ev),
            ('STATE: [10,20]', ev),
            ('STATE: {"A":10,"B":20,"C":30,"D":40} ', ev),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"add","other":"C"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"add","other":"C","status":"posted","tick":1}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"add","other":"B","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"E","amount":5,"kind":"add","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"add","other":"Z","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"refund","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"Set","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"SWAP","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"swapped","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":null,"kind":"swap","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":"7","kind":"set","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"swap","other":"B","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"swap","other":"E","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"set","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"add","other":"C","status":"not posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"add","other":"C","status":""}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"add","other":"C","status":"po.sted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":true,"kind":"add","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":"5","kind":"add","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5.5,"kind":"add","other":"C","status":"posted"}'),
            (BASE_STATE, 'EVENT: {"account":"B","amount":5,"kind":"add","other":"C","status":7}'),
            (BASE_STATE, 'EVENT: {"account":"B", "amount":5,"kind":"add","other":"C","status":"posted"}'),
            ("STATE: " + "[" * 100000, ev),
            (BASE_STATE, "EVENT: " + '{"a":' * 100000),
        ]
        for st, evl in bad:
            with self.subTest(state=st, event=evl):
                with self.assertRaises(ValueError):
                    O.parse_input(f"{BASE_RULE}\n{st}\n{evl}")


class TargetParsing(unittest.TestCase):
    ANS = '{"after":{"A":10,"B":25,"C":30,"D":40}}'
    TRACE = GOLD[8][6]
    SRC_TRACE = GOLD[8][2]

    def test_valid(self):
        self.assertEqual(O.parse_target(self.ANS, "ans"), json.loads(self.ANS))
        self.assertEqual(O.parse_target(" \n" + self.ANS + " \n\t", "ans", source=BASE), json.loads(self.ANS))
        self.assertEqual(O.parse_target('{"after":{"B":25,"A":10,"C":30,"D":40}}', "ans")["after"]["B"], 25)
        self.assertIsNotNone(O.parse_target(self.TRACE, "trace", source=O.parse_input(self.SRC_TRACE)))
        self.assertIsNone(O.parse_target(GOLD[20][6], "trace")["cmp"]["listed"])

    def test_malformed_ans(self):
        bad = [
            '{"after":{"A":10,"B":25,"C":30,"D":40}} done', '{"after":{"A":10,"B":25,"C":30,"D":40}}}',
            '{"after":{"A":10.0,"B":25,"C":30,"D":40}}', '{"after":{"A":1e1,"B":25,"C":30,"D":40}}',
            '{"after":{"A":true,"B":25,"C":30,"D":40}}', '{"after":{"A":"10","B":25,"C":30,"D":40}}',
            '{"after":{"A":NaN,"B":25,"C":30,"D":40}}', '{"after":{"A":null,"B":25,"C":30,"D":40}}',
            '{"after":{"A":10,"B":25,"C":30,"D":40},"ok":true}', '{"ok":true,"after":{"A":10,"B":25,"C":30,"D":40}}',
            '{"after":{"A":10,"A":25,"C":30,"D":40}}', '{"after":{}}', '{"after":[10,25,30,40]}', '{}', '[]',
            'null', '', '   ', 'after: {"A":10}', '```{"after":{"A":10,"B":25,"C":30,"D":40}}```',
            '{"after":{"A":10,"B":25,"C":30,"D":40}', self.TRACE,
            self.ANS + "\x1f", self.ANS + "\x0c", self.ANS + "\xa0", self.ANS + " 　", "\x1c" + self.ANS,
            " " + self.ANS, "[" * 100000,
        ]
        for text in bad:
            with self.subTest(text=text):
                self.assertIsNone(O.parse_target(text, "ans"))
        self.assertIsNone(O.parse_target(None, "ans"))

    def test_ans_key_order_with_source(self):
        for text in ('{"after":{"B":25,"A":10,"C":30,"D":40}}', '{"after":{"A":10,"B":25,"C":30}}',
                     '{"after":{"A":10,"B":25,"C":30,"D":40,"E":0}}'):
            with self.subTest(text=text):
                self.assertIsNone(O.parse_target(text, "ans", source=BASE))

    def test_malformed_trace(self):
        after = '"after":{"A":7,"B":-7,"C":0,"D":0}'
        cmp_ = '"cmp":{"status":"paid","listed":false,"ok":true}'
        bad = [
            "{" + '"calc":{"B":-7,"A":7},' + cmp_ + "," + after + "}",
            "{" + cmp_ + "," + after + ',"calc":{"B":-7,"A":7}}',
            '{"cmp":{"listed":false,"status":"paid","ok":true},"calc":{"B":-7,"A":7},' + after + "}",
            '{"cmp":{"status":"paid","listed":"false","ok":true},"calc":{"B":-7,"A":7},' + after + "}",
            '{"cmp":{"status":"paid","listed":false,"ok":1},"calc":{"B":-7,"A":7},' + after + "}",
            '{"cmp":{"status":5,"listed":false,"ok":true},"calc":{"B":-7,"A":7},' + after + "}",
            '{"cmp":{"status":"paid","listed":false,"ok":true,"x":1},"calc":{"B":-7,"A":7},' + after + "}",
            '{"cmp":{"status":"paid","ok":true},"calc":{"B":-7,"A":7},' + after + "}",
            "{" + cmp_ + ',"calc":{"B":-7,"A":7,"C":0},' + after + "}",
            "{" + cmp_ + ',"calc":{},' + after + "}",
            "{" + cmp_ + ',"calc":{"B":-7.0,"A":7},' + after + "}",
            "{" + cmp_ + ',"calc":{"B":-7,"A":7},' + after + ',"fill":" ."}',
            "{" + cmp_ + ',"calc":{"B":-7,"A":7},' + after + "} x",
            "{" + cmp_ + "," + after + "}",
            self.ANS,
        ]
        for text in bad:
            with self.subTest(text=text):
                self.assertIsNone(O.parse_target(text, "trace"))

    def test_trace_key_order_with_source(self):
        swapped_calc = self.TRACE.replace('"calc":{"B":-7,"A":7}', '"calc":{"A":7,"B":-7}')
        self.assertIsNotNone(O.parse_target(swapped_calc, "trace"))
        self.assertIsNone(O.parse_target(swapped_calc, "trace", source=self.SRC_TRACE))
        one_key = self.TRACE.replace('"calc":{"B":-7,"A":7}', '"calc":{"B":-7}')
        self.assertIsNone(O.parse_target(one_key, "trace", source=self.SRC_TRACE))
        swapped_after = self.TRACE.replace('"after":{"A":7,"B":-7,', '"after":{"B":-7,"A":7,')
        self.assertIsNone(O.parse_target(swapped_after, "trace", source=self.SRC_TRACE))

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            O.parse_target(self.ANS, "fill")


class Semantics(unittest.TestCase):
    ST = {"A": 10, "B": 20, "C": 30, "D": 40}

    def ev(self, **kw):
        e = {"account": "A", "amount": 5, "kind": "add", "other": "B", "status": "posted"}
        e.update(kw)
        return e

    def test_atoms(self):
        cases = [
            (S("S+", "posted"), self.ev(), True), (S("S+", "posted"), self.ev(status="Posted"), False),
            (S("S-", "void"), self.ev(status="void"), False), (S("S-", "void"), self.ev(status="VOID"), True),
            (K("add"), self.ev(kind="transfer"), False), (K("transfer", "add"), self.ev(kind="transfer"), True),
            (K("add", "transfer"), self.ev(kind="scale"), False),
            (T("AGE", 5), self.ev(amount=5), True), (T("AGE", 5), self.ev(amount=4), False),
            (T("ALT", 5), self.ev(amount=5), False), (T("ALT", 5), self.ev(amount=4), True),
            (T("BGE", 10), self.ev(account="A", other="D"), True), (T("BGE", 11), self.ev(account="A", other="D"), False),
            (T("BGE", 40), self.ev(account="D", other="A"), True),
            (C("AND", S("S+", "a"), T("AGE", 0)), self.ev(status="a", amount=0), True),
            (C("AND", S("S+", "a"), T("AGE", 0)), self.ev(status="a", amount=-1), False),
            (C("OR", S("S+", "a"), T("AGE", 0)), self.ev(status="b", amount=-1), False),
            (C("OR", S("S+", "a"), T("AGE", 0)), self.ev(status="b", amount=0), True),
        ]
        for rule, ev, want in cases:
            with self.subTest(rule=rule, ev=ev):
                self.assertIs(O.evaluate(rule, self.ST, ev), want)

    def test_listed(self):
        ev = self.ev(status="x")
        self.assertIs(O.listed_of(S("S-", "x"), ev), True)
        self.assertIsNone(O.listed_of(C("OR", K("add"), T("BGE", 1)), ev))
        self.assertIs(O.listed_of(C("AND", T("AGE", 1), S("S+", "y")), ev), False)
        self.assertIs(O.listed_of(C("OR", S("S+", "y"), S("S-", "x")), ev), False)

    def test_transition(self):
        st = {"D": 4, "B": 2, "A": 3, "C": 1}
        after = O.transition(st, self.ev(kind="scale", amount=5, account="A"), True)
        self.assertEqual(list(after.items()), [("D", 4), ("B", 2), ("A", 15), ("C", 1)])
        after = O.transition(st, self.ev(kind="transfer", amount=-2, account="C", other="D"), True)
        self.assertEqual(list(after.items()), [("D", 2), ("B", 2), ("A", 3), ("C", 3)])
        noop = O.transition(st, self.ev(kind="transfer"), False)
        self.assertEqual(list(noop.items()), list(st.items()))
        self.assertIsNot(noop, st)
        self.assertEqual(st, {"D": 4, "B": 2, "A": 3, "C": 1})
        with self.assertRaises(ValueError):
            O.transition(st, self.ev(), 1)

    def test_invalid_ast(self):
        bad = [
            {"op": "XOR", "args": [K("add"), K("add")]}, {"op": "S+", "set": []},
            {"op": "S+", "set": ["a", "b", "c", "d"]}, {"op": "S+", "set": ["a", "a"]}, {"op": "S+", "set": "a"},
            {"op": "S+", "set": ["a b"]}, {"op": "K+", "kinds": ["scale"]}, {"op": "K+", "kinds": []},
            {"op": "AGE", "t": True}, {"op": "AGE", "t": 1.0}, {"op": "AGE", "t": 1, "x": 0},
            {"op": "AND", "args": [C("OR", K("add"), K("add")), K("add")]},
            {"op": "AND", "args": [K("add")]}, {"op": "OR", "args": [K("add"), K("add"), K("add")]},
            {"op": ["S+"], "set": ["a"]}, "S+", None,
        ]
        for rule in bad:
            with self.subTest(rule=rule):
                with self.assertRaises(ValueError):
                    O.evaluate(rule, self.ST, self.ev())


ALL_K = ("add", "transfer", "set", "swap")             # nameable in K+ (SPEC 2 + 1.1)
ALL_EVENT_KINDS = ("add", "transfer", "scale", "set", "swap")


def _copy_case(tid, kind):
    return next(g for g in COPY_GOLD if g[0] == tid and O.parse_input(g[2])["event"]["kind"] == kind)


class CopyWorld(unittest.TestCase):
    """SPEC section 1.1: the arithmetic-free `set` / `swap` kinds and K+ clauses naming them."""
    ST = {"D": 4, "B": 2, "A": 3, "C": 1}

    def ev(self, **kw):
        e = {"account": "A", "amount": 5, "kind": "set", "other": "B", "status": "posted"}
        e.update(kw)
        return e

    def test_kind_constants(self):
        self.assertEqual(set(O.KINDS), set(ALL_K))
        self.assertEqual(set(O.EVENT_KINDS), set(ALL_EVENT_KINDS))

    def test_gold_coverage(self):
        self.assertEqual({g[0] for g in COPY_GOLD}, set(O.TEMPLATE_IDS))
        parsed = [(g, O.parse_input(g[2])) for g in COPY_GOLD]
        self.assertEqual({(p["event"]["kind"], g[3]) for g, p in parsed},
                         {(k, ok) for k in ("set", "swap") for ok in (True, False)})
        fam_kind_ok = {(g[0].split(":")[0], p["event"]["kind"], g[3]) for g, p in parsed}
        for fam in ("SPOS", "SNEG"):
            for kind in ("set", "swap"):
                for ok in (True, False):
                    self.assertIn((fam, kind, ok), fam_kind_ok)
        named = {(g[1]["op"], k) for g in COPY_GOLD if g[1]["op"] in ("AND", "OR")
                 for a in g[1]["args"] if a["op"] == "K+" for k in a["kinds"]}
        self.assertLessEqual({(op, k) for op in ("AND", "OR") for k in ("set", "swap")}, named)
        nonadjacent = [g for g, p in parsed if p["event"]["kind"] == "swap" and g[3]
                       and p["state_order"] != sorted(p["state_order"])
                       and abs(p["state_order"].index(p["event"]["account"])
                               - p["state_order"].index(p["event"]["other"])) > 1]
        self.assertGreaterEqual(len(nonadjacent), 3)

    def test_gold_calc_and_after_order(self):
        for tid, ast, text, ok, listed, ans, trace in COPY_GOLD:
            with self.subTest(text=text):
                p = O.parse_input(text)
                ev, before = p["event"], p["state"]
                acc, oth = ev["account"], ev["other"]
                tr = json.loads(trace)
                self.assertEqual(list(tr["after"]), p["state_order"])
                self.assertEqual(list(json.loads(ans)["after"]), p["state_order"])
                want_keys = [acc, oth] if ev["kind"] == "swap" else [acc]
                self.assertEqual(list(tr["calc"]), want_keys)
                self.assertEqual(tr["calc"], {k: tr["after"][k] for k in want_keys})
                untouched = [k for k in before if k not in want_keys]
                self.assertEqual({k: tr["after"][k] for k in untouched}, {k: before[k] for k in untouched})
                if not ok:
                    self.assertEqual(tr["after"], before)
                elif ev["kind"] == "set":
                    self.assertEqual(tr["after"][acc], ev["amount"])
                else:
                    self.assertEqual((tr["after"][acc], tr["after"][oth]), (before[oth], before[acc]))

    def test_transition_set(self):
        after = O.transition(self.ST, self.ev(account="A", amount=0), True)
        self.assertEqual(list(after.items()), [("D", 4), ("B", 2), ("A", 0), ("C", 1)])
        after = O.transition(self.ST, self.ev(account="C", other="D", amount=-7), True)
        self.assertEqual(list(after.items()), [("D", 4), ("B", 2), ("A", 3), ("C", -7)])
        after = O.transition(self.ST, self.ev(account="D", amount=4), True)
        self.assertEqual(list(after.items()), list(self.ST.items()))
        noop = O.transition(self.ST, self.ev(account="B", other="C", amount=99), False)
        self.assertEqual(list(noop.items()), list(self.ST.items()))
        self.assertIsNot(noop, self.ST)
        self.assertEqual(self.ST, {"D": 4, "B": 2, "A": 3, "C": 1})

    def test_transition_swap(self):
        # account after other in the state order, and the two are non-adjacent (positions 3 and 0).
        after = O.transition(self.ST, self.ev(kind="swap", account="C", other="D", amount=12), True)
        self.assertEqual(list(after.items()), [("D", 1), ("B", 2), ("A", 3), ("C", 4)])
        after = O.transition(self.ST, self.ev(kind="swap", account="D", other="A", amount=1), True)
        self.assertEqual(list(after.items()), [("D", 3), ("B", 2), ("A", 4), ("C", 1)])
        for amt in (-5, 0, 1, 12, 999):  # amount is ignored by swap
            after = O.transition(self.ST, self.ev(kind="swap", account="B", other="C", amount=amt), True)
            self.assertEqual(list(after.items()), [("D", 4), ("B", 1), ("A", 3), ("C", 2)])
        twice = O.transition(after, self.ev(kind="swap", account="C", other="B"), True)
        self.assertEqual(list(twice.items()), list(self.ST.items()))
        eq = {"A": 5, "B": 5, "C": 0, "D": 1}
        self.assertEqual(list(O.transition(eq, self.ev(kind="swap", account="A", other="B"), True).items()),
                         list(eq.items()))
        noop = O.transition(self.ST, self.ev(kind="swap", account="C", other="D"), False)
        self.assertEqual(list(noop.items()), list(self.ST.items()))
        self.assertEqual(self.ST, {"D": 4, "B": 2, "A": 3, "C": 1})

    def test_atoms(self):
        cases = [
            (K("set"), self.ev(kind="set"), True), (K("set"), self.ev(kind="swap"), False),
            (K("swap"), self.ev(kind="swap"), True), (K("swap"), self.ev(kind="transfer"), False),
            (K("swap", "add"), self.ev(kind="swap"), True), (K("add", "transfer"), self.ev(kind="set"), False),
            (K("set"), self.ev(kind="add"), False), (K("add", "transfer", "set", "swap"), self.ev(kind="scale"), False),
            (K("transfer", "swap", "set", "add"), self.ev(kind="add"), True),
            (T("AGE", 10), self.ev(kind="swap", amount=10), True), (T("ALT", 10), self.ev(kind="swap", amount=10), False),
            (T("BGE", 3), self.ev(kind="set", account="A", amount=99), True),
            (T("BGE", 4), self.ev(kind="set", account="A", amount=99), False),
            (C("AND", S("S+", "posted"), K("set")), self.ev(kind="set"), True),
            (C("AND", S("S+", "posted"), K("set")), self.ev(kind="swap"), False),
            (C("OR", S("S-", "posted"), K("swap")), self.ev(kind="set"), False),
            (C("OR", S("S-", "posted"), K("swap")), self.ev(kind="swap"), True),
        ]
        for rule, ev, want in cases:
            with self.subTest(rule=rule, ev=ev):
                self.assertIs(O.evaluate(rule, self.ST, ev), want)

    def test_invalid(self):
        for rule in (K("set", "set"), K("Set"), K("scale", "set"), K("add", "transfer", "set", "swap", "add"),
                     {"op": "K+", "kinds": "set"}, {"op": "K+", "kinds": ["set or swap"]}):
            with self.subTest(rule=rule):
                with self.assertRaises(ValueError):
                    O.evaluate(rule, self.ST, self.ev())
                with self.assertRaises(ValueError):
                    O.render_rule(rule, "ATOM:t0")
        for ev in (self.ev(kind="Set"), self.ev(kind="SWAP"), self.ev(kind="copy"),
                   self.ev(kind="swap", amount=None), self.ev(kind="swap", other="A")):
            with self.subTest(ev=ev):
                with self.assertRaises(ValueError):
                    O.evaluate(K("set"), self.ST, ev)
                with self.assertRaises(ValueError):
                    O.transition(self.ST, ev, True)

    def test_trace_calc_keys_with_source(self):
        _, _, src, _, _, ans, trace = _copy_case("SNEG:t2", "swap")  # calc {"C":40,"D":30}
        self.assertIsNotNone(O.parse_target(trace, "trace", source=src))
        swapped = trace.replace('"calc":{"C":40,"D":30}', '"calc":{"D":30,"C":40}')
        self.assertNotEqual(swapped, trace)
        self.assertIsNotNone(O.parse_target(swapped, "trace"))
        self.assertIsNone(O.parse_target(swapped, "trace", source=src))
        for calc in ('"calc":{"C":40}', '"calc":{"D":30}', '"calc":{"C":40,"A":10}'):
            bad = trace.replace('"calc":{"C":40,"D":30}', calc)
            self.assertIsNotNone(O.parse_target(bad, "trace"))
            self.assertIsNone(O.parse_target(bad, "trace", source=src))
        sorted_after = '"after":{"A":10,"B":20,"C":40,"D":30}'
        bad = trace.replace('"after":{"D":30,"B":20,"A":10,"C":40}', sorted_after)
        self.assertNotEqual(bad, trace)
        self.assertIsNone(O.parse_target(bad, "trace", source=src))
        self.assertIsNone(O.parse_target("{" + sorted_after + "}", "ans", source=src))
        self.assertEqual(O.parse_target(ans, "ans", source=src), json.loads(ans))

        _, _, src, _, _, _, trace = _copy_case("SPOS:t1", "set")  # calc {"C":7}, other A
        self.assertIsNotNone(O.parse_target(trace, "trace", source=src))
        for calc in ('"calc":{"C":7,"A":5}', '"calc":{"A":5,"C":7}', '"calc":{"A":5}'):
            bad = trace.replace('"calc":{"C":7}', calc)
            self.assertNotEqual(bad, trace)
            self.assertIsNotNone(O.parse_target(bad, "trace"))
            self.assertIsNone(O.parse_target(bad, "trace", source=src))

    def test_k_plus_every_kind_list_round_trips(self):
        """Every ordered non-empty K+ kind list renders and parses back in every frame and slot."""
        lists = [list(p) for r in range(1, len(ALL_K) + 1) for p in itertools.permutations(ALL_K, r)]
        self.assertEqual(len(lists), 64)
        st = {"A": 1, "B": 2, "C": 3, "D": 4}
        status_atom = S("S+", "set", "swap")
        for kinds in lists:
            k = K(*kinds)
            rules = [("ATOM:t0", k)]
            for op, pool in (("AND", O.AND_FRAMES), ("OR", O.OR_FRAMES)):
                for fid in pool:
                    rules += [(f"{op}:{fid}", C(op, k, status_atom)), (f"{op}:{fid}", C(op, status_atom, k)),
                              (f"{op}:{fid}", C(op, k, K("swap"))), (f"{op}:{fid}", C(op, K("set", "add"), k))]
            for tid, rule in rules:
                with self.subTest(tid=tid, rule=rule):
                    for kind in ALL_EVENT_KINDS:
                        ev = {"kind": kind, "status": "add", "account": "D", "other": "A", "amount": 3}
                        text = O.render_input(rule, st, ev, tid)
                        p = O.parse_input(text)
                        self.assertEqual((p["template_id"], p["rule_ast"]), (tid, rule))
                        self.assertIs(O.evaluate(p["rule_ast"], p["state"], p["event"]),
                                      RoundTripProperty.ref_ok(rule, st, ev))


class TemplateDrift(unittest.TestCase):
    def test_strings_match_templates_py(self):
        """The hard-coded copies must equal the literals in templates.py (read as text, not imported)."""
        path = os.path.join(RCL_DIR, "templates.py")
        if not os.path.exists(path):
            self.skipTest("templates.py not present")
        with open(path, encoding="utf-8") as f:
            tree = pyast.parse(f.read())
        lits = {}
        for node in pyast.walk(tree):
            if isinstance(node, pyast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], pyast.Name):
                try:
                    lits[node.targets[0].id] = pyast.literal_eval(node.value)
                except ValueError:
                    pass
        for name in ("SPOS", "SNEG", "AND_FRAMES", "OR_FRAMES", "CLAUSE"):
            self.assertEqual(lits[name], getattr(O, name), name)
        self.assertEqual(lits["frame"], O.ATOM_FRAME)


class RoundTripProperty(unittest.TestCase):
    TRICKY = ["or", "and", "in", "not", "is", "its", "status", "events", "event", "if", "when", "either", "do",
              "Void", "VOID", "void_", "un-posted", "posted", "void", "-", "_", "0", "-5", "add", "transfer",
              "t0", "nothing", "carry", "out", "only", "except", "balances", "set", "swap", "scale", "kind"]
    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    NAMES = (list("ABCD"), list("PQRSTU"), ["north", "south", "east", "west", "pine", "oak"])

    def word(self, rng):
        if rng.random() < 0.35:
            return rng.choice(self.TRICKY)
        return "".join(rng.choice(self.CHARS) for _ in range(rng.randint(1, 8)))

    def words(self, rng, n):
        out = []
        while len(out) < n:
            w = self.word(rng)
            if w not in out:
                out.append(w)
        return out

    def atom(self, rng, ops=("S+", "S-", "K+", "AGE", "ALT", "BGE")):
        op = rng.choice(ops)
        if op in ("S+", "S-"):
            return S(op, *self.words(rng, rng.randint(1, 3)))
        if op == "K+":
            return K(*rng.sample(ALL_K, rng.choice((1, 1, 2, 2, 3, 4))))
        return T(op, rng.randint(-150, 150))

    def case(self, rng):
        tid = rng.choice(O.TEMPLATE_IDS)
        fam = tid.split(":")[0]
        if fam == "SPOS":
            rule = S("S+", *self.words(rng, rng.randint(1, 3)))
        elif fam == "SNEG":
            rule = S("S-", *self.words(rng, rng.randint(1, 3)))
        elif fam == "ATOM":
            rule = self.atom(rng, ("K+", "AGE", "ALT", "BGE"))
        else:
            rule = C(fam, self.atom(rng), self.atom(rng))
        pool = rng.choice(self.NAMES)
        names = pool[:4] if pool[0] == "A" else rng.sample(pool, rng.randint(4, 6))
        order = rng.choice(["sorted", "reversed", "shuffled"])
        names = sorted(names, reverse=order == "reversed") if order != "shuffled" else rng.sample(names, len(names))
        state = {n: rng.randint(-999, 999) for n in names}
        acc, oth = rng.sample(names, 2)
        sets = [a["set"] for a in rule.get("args", [rule]) if "set" in a]
        status = rng.choice(sets[0]) if sets and rng.random() < 0.5 else self.word(rng)
        kind = rng.choice(ALL_EVENT_KINDS)
        fields = {"account": acc, "amount": rng.randint(-120, 120), "kind": kind, "other": oth, "status": status}
        event = {k: fields[k] for k in rng.sample(sorted(fields), 5)}
        return tid, rule, state, event

    @staticmethod
    def ref_ok(rule, state, event):
        tests = {
            "S+": lambda a: any(event["status"] == w for w in a["set"]),
            "S-": lambda a: all(event["status"] != w for w in a["set"]),
            "K+": lambda a: event["kind"] in a["kinds"],
            "AGE": lambda a: not event["amount"] < a["t"],
            "ALT": lambda a: event["amount"] - a["t"] < 0,
            "BGE": lambda a: state[event["account"]] - a["t"] >= 0,
        }
        if rule["op"] == "AND":
            return all(tests[a["op"]](a) for a in rule["args"])
        if rule["op"] == "OR":
            return any(tests[a["op"]](a) for a in rule["args"])
        return tests[rule["op"]](rule)

    def test_round_trip_2000(self):
        rng = random.Random(20261009)
        seen_ids, seen_kind_ok, seen_k_named, n_ok = set(), set(), set(), 0
        for i in range(2000):
            tid, rule, state, event = self.case(rng)
            text = O.render_input(rule, state, event, tid)
            with self.subTest(i=i, text=text):
                p = O.parse_input(text)
                self.assertEqual(p["template_id"], tid)
                self.assertEqual(p["rule_ast"], rule)
                self.assertEqual(list(p["state"].items()), list(state.items()))
                self.assertEqual(list(p["event"].items()), list(event.items()))
                self.assertEqual(p["state_order"], list(state))
                self.assertEqual(O.render_input(p["rule_ast"], p["state"], p["event"], p["template_id"]), text)
                g = O.gold_targets(text)
                want_ok = self.ref_ok(rule, state, event)
                self.assertIs(g["ok"], want_ok)
                acc, oth, amt, kind = event["account"], event["other"], event["amount"], event["kind"]
                want = dict(state)
                if want_ok:
                    if kind == "add":
                        want[acc] += amt
                    elif kind == "transfer":
                        want[acc], want[oth] = want[acc] - amt, want[oth] + amt
                    elif kind == "scale":
                        want[acc] *= amt
                    elif kind == "set":
                        want[acc] = amt
                    else:
                        self.assertEqual(kind, "swap")
                        want[acc], want[oth] = state[oth], state[acc]
                self.assertEqual(list(g["after"].items()), list(want.items()))
                self.assertEqual(O.parse_target(g["ans"], "ans", source=text), {"after": want})
                pt = O.parse_target(g["trace"], "trace", source=p)
                self.assertEqual(pt["after"], want)
                self.assertEqual(pt["cmp"], {"status": event["status"], "listed": g["listed"], "ok": want_ok})
                calc_keys = [acc, oth] if kind in ("transfer", "swap") else [acc]
                self.assertEqual(list(pt["calc"].items()), [(k, want[k]) for k in calc_keys])
                seen_ids.add(tid)
                seen_kind_ok.add((kind, want_ok))
                seen_k_named |= {(rule["op"], k) for a in rule.get("args", [rule]) if a["op"] == "K+"
                                 for k in a["kinds"]}
                n_ok += want_ok
        self.assertEqual(seen_ids, set(O.TEMPLATE_IDS))
        self.assertEqual(seen_kind_ok, {(k, ok) for k in ALL_EVENT_KINDS for ok in (True, False)})
        self.assertLessEqual({(op, k) for op in ("K+", "AND", "OR") for k in ALL_K}, seen_k_named)
        self.assertTrue(300 < n_ok < 1700, n_ok)

    def test_mutations_never_misparse(self):
        """A mutated rule line is either rejected or re-renders to exactly the mutated text."""
        rng = random.Random(7)
        pool = self.CHARS + " ,.;[]" + "or"
        accepted = 0
        for i in range(2000):
            tid, rule, state, event = self.case(rng)
            text = O.render_input(rule, state, event, tid)
            head, rest = text.split("\n", 1)
            j = rng.randrange(6, len(head) + 1)
            op = rng.choice(["ins", "del", "sub"])
            if op == "ins":
                head = head[:j] + rng.choice(pool) + head[j:]
            elif op == "del" and j < len(head):
                head = head[:j] + head[j + 1:]
            elif j < len(head):
                head = head[:j] + rng.choice(pool) + head[j + 1:]
            mutated = head + "\n" + rest
            try:
                p = O.parse_input(mutated)
            except ValueError:
                continue
            accepted += 1
            with self.subTest(i=i, text=mutated):
                self.assertEqual(O.render_input(p["rule_ast"], p["state"], p["event"], p["template_id"]), mutated)
        self.assertGreater(accepted, 0)


if __name__ == "__main__":
    unittest.main()
