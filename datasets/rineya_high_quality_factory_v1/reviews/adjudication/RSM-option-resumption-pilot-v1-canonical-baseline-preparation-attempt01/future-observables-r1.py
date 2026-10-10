"""Future dependency diagnostics exclude inert surviving checkpoint bindings."""
import copy,json

def equal(a,b):return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)
def value_at(v,path):
    for key in path:v=v[key]
    return v
def scalar_change(v,path,change=1):
    z=copy.deepcopy(v);p=z
    for k in path[:-1]:p=p[k]
    if type(p[path[-1]]) is not int:raise ValueError('activity probe needs strict integer leaf')
    p[path[-1]]+=change;return z

def observe(task,snapshot,solver,bound=48):
    """Trusted diagnostic only: never validates/approves a changed checkpoint.
    Returned checkpoint is separately diagnostic, never part of activity equality.
    """
    machine=solver.Machine(task,snapshot=snapshot);events=[]
    for _ in range(bound):
        event=machine.step()
        if event is None:break
        events.append(event)
    cp=machine.snapshot()
    evaluated=[]
    for e in events:
        kind=e['event']
        if kind=='effect':evaluated.append(copy.deepcopy(e))
        elif kind=='branch':evaluated.append(copy.deepcopy(e))
        elif kind=='return':evaluated.append(copy.deepcopy(e))
        elif kind=='enter':
            # Raw entered bindings alone can carry an unused argument. Entry
            # applicability, call selection and subsequent evaluation remain.
            evaluated.append({k:copy.deepcopy(v) for k,v in e.items() if k!='bindings'})
    projection={'evaluated_future_events':evaluated,'effect_after_states':[e['state'] for e in events if e['event']=='effect'],'terminal_phase':cp['phase'],'terminal_state':cp['state'],'bound_exhausted':cp['phase']=='running'}
    return {'observable_projection':projection,'full_future_events':events,'diagnostic_checkpoint':cp}

def activity(task,snapshot,locator,solver,delta=1):
    before=observe(task,snapshot,solver);changed=scalar_change(snapshot,locator,delta);after=observe(task,changed,solver)
    if before['observable_projection']['bound_exhausted'] or after['observable_projection']['bound_exhausted']:
        return {'status':'UNKNOWN','reason':'bounded diagnostic did not reach terminal','locator':locator,'before':before,'after':after}
    return {'status':'ACTIVE' if not equal(before['observable_projection'],after['observable_projection']) else 'INACTIVE','locator':locator,'delta':delta,'future_observable_changed':not equal(before['observable_projection'],after['observable_projection']),'full_checkpoint_only_changed':not equal(before['diagnostic_checkpoint'],after['diagnostic_checkpoint']) and equal(before['observable_projection'],after['observable_projection']),'before':before,'after':after,'changed_snapshot_is_not_validated_or_approved':True}
