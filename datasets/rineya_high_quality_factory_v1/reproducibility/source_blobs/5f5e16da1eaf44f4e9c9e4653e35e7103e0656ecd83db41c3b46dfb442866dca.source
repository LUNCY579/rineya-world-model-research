"""Rule-conditioned semantic parents and matched counterfactual twins."""
import copy
import json
import random
from factory.common import ROOT, record, record_hash, load_jsonl, write_jsonl, write_json
from factory.reference import solve


def const(x):return {"const":x}
def var(x):return {"var":x}
def expr(op,*args):return {"op":op,"args":list(args)}


CONTRACT = """Interpret the AST exactly: const is a literal; var reads event.status, event.amount, state.x or state.y. eq/ne/ge/gt/le/lt compare two arguments without coercion; all/any combine boolean arguments; not negates one. For each event, evaluate rule and precondition on the current state; execute only when both are true. Otherwise keep state unchanged. add increases source by amount; scale multiplies source by amount; transfer simultaneously subtracts from source and adds to destination; swap exchanges source and destination simultaneously. Signed balances are allowed unless the stated precondition forbids them. Ignore annotation. Execute events in given sequence.\n"""


def reorder(obj):
    if isinstance(obj,dict):return {k:reorder(obj[k]) for k in reversed(list(obj))}
    if isinstance(obj,list):return [reorder(v) for v in obj]
    return obj


def generate(count=30,batch="pilot-v2",offset=0):
    rng=random.Random(20261009+offset)
    rows,parents=[],[]
    statuses=["posted","void","pending","cancelled","draft","approved","unposted","voided","zunik","vosted"]
    for local in range(count):
        i=local+offset
        split="dev_challenge" if i%10>=8 else "train"
        status=("xelth","quom")[(i//10)%2] if split=="dev_challenge" else statuses[(i+i//10)%len(statuses)]
        scale=100 if split=="dev_challenge" else 1
        state={"x":rng.randint(3,40)*scale,"y":rng.randint(1,30)*scale}
        amount=rng.randint(1,9)*scale
        mechanism=i%6
        if mechanism==0: rule=expr("eq",var("event.status"),const(status))
        elif mechanism==1: rule=expr("ne",var("event.status"),const(status))
        elif mechanism==2: rule=expr("ge",var("event.amount"),const(amount))
        elif mechanism==3: rule=expr("all",expr("eq",var("event.status"),const(status)),expr("ge",var("event.amount"),const(amount)))
        elif mechanism==4: rule=expr("any",expr("ne",var("event.status"),const(status)),expr("lt",var("state.y"),const(state["y"]+1)))
        else: rule=expr("not",expr("all",expr("ne",var("event.status"),const(status)),expr("gt",var("state.x"),const(state["x"]))))
        # Preconditions have true/false boundary cases independently of rule truth.
        pre=expr("ge",var("state.x"),var("event.amount")) if i%3==0 else const(True)
        if i%9==6: pre=expr("gt",var("state.x"),const(state["x"]))
        length=1 if i%5<2 else 2+i%3
        events=[]
        for step in range(length):
            events.append(dict(status=status if step%2==0 else statuses[(i+step)%len(statuses)],
                               amount=amount+step,op=["add","transfer","scale","swap"][(i+step)%4],source="x" if step%2==0 else "y",destination="y" if step%2==0 else "x"))
        first={"kind":"causal","state":state,"rule":rule,"precondition":pre,"events":events,"annotation":"weather record: cloudy"}
        twin=copy.deepcopy(first);twin["rule"]=expr("not",rule)
        tasks=[first,twin]
        parents.append(dict(semantic_parent_id=f"RCM-001-{batch}-p{i:04d}",split=split,mechanism=f"predicate-{mechanism}",tasks=tasks))
        for v,t in enumerate(tasks):
            gold=solve(t)
            # Cross both output formats with both input key orders. Format and
            # field order cannot be mistaken for one another in future A/Bs.
            for rendering in range(4):
                answer_only=rendering<2
                reverse=rendering%2==1
                fmt="answer_only" if answer_only else "explicit_trace"
                request=("Return JSON {after: final_state} only." if answer_only else "Return JSON {after: final_state, steps: [{cmp: {rule: boolean, precondition: boolean}, calc: {operation: executed_op_or_identity}, after: state_after_event}]}.")
                body=json.dumps(t if not reverse else reorder(t),ensure_ascii=False,separators=(",",":"))
                prompt=CONTRACT+request+"\nTask JSON:\n"+body
                target={"after":gold["after"]} if answer_only else gold
                difficulty=4 if length>1 else 3 if mechanism>=3 or pre!=const(True) else 1
                r=record("RCM-001",batch,i,f"twin{v}-{fmt}",t,prompt,target,["rule_conditioning","counterfactual","conditional_execution",f"predicate_{mechanism}"],difficulty,fmt,"reverse" if reverse else "forward")
                r.update(generator_version="1.1.0",structural_difficulty=difficulty,
                         challenge_axes=["heldout_first_status","larger_magnitudes"] if split=="dev_challenge" else [],
                         source_family_key=f"RCM-001-seed20261009-p{i:04d}")
                r["record_hash"]=record_hash(r);rows.append(r)
    outputs=[(ROOT/f"data/RCM-001/semantic_parents/{batch}.jsonl",parents),(ROOT/f"data/RCM-001/candidate_renderings/{batch}.jsonl",rows)]
    for p,rs in outputs:
        if p.exists() and load_jsonl(p)!=rs:raise ValueError("Immutable candidate batch differs; use a new batch name")
        write_jsonl(p,rs)
    return rows


if __name__=="__main__":generate()
