"""Separate complete recursive execution, then prefix/suffix slicing (not frame dispatch).
Shares only public grammar validation with frame solver; evaluates effects/predicates independently.
"""
import copy
from frame_solver_r1 import validate, invalid


def typed_key(value):
    if type(value) is dict: return ('object',tuple(sorted((k,typed_key(v)) for k,v in value.items())))
    if type(value) is list: return ('array',tuple(map(typed_key,value)))
    return (type(value).__name__,value)


class Halt(Exception): pass


def full_execution(task, starting_state=None):
    state = copy.deepcopy(task['initial_state'] if starting_state is None else starting_state)
    stack = []; events = []; snapshots = []; counter = 0; phase = 'running'; pending = True
    def expression(node, local):
        key,value = next(iter(node.items()))
        return value if key=='const' else state[value] if key=='reg' else local[value]
    def guard(node, local):
        values=[]
        for atom in node['all']:
            left,right=expression(atom['left'],local),expression(atom['right'],local)
            values.append(left<right if atom['cmp']=='lt' else left==right if atom['cmp']=='eq' else left>=right)
        return all(values)
    def snapshot():
        frames = copy.deepcopy(stack)
        while frames and frames[-1]['frame_kind']=='branch':
            frame=frames[-1]; body=task['options'][frame['option']]['body']; path=frame['body_path']
            for j in range(0,len(path),2):body=body[path[j]][path[j+1]]
            if frame['next_index']<len(body):break
            frames.pop()
        return {'state':copy.deepcopy(state),'frames':frames,'next_instance_id':counter,'root_pending':pending,'phase':phase}
    snapshots.append(snapshot())
    def emit(event):
        events.append(copy.deepcopy(event));snapshots.append(snapshot())
    def invocation(option, arguments, site):
        nonlocal counter,phase
        instance=counter;counter+=1
        admitted=guard(task['options'][option]['initiation'],arguments)
        if admitted:
            frame={'frame_kind':'option','instance_id':instance,'option':option,'bindings':copy.deepcopy(arguments),'body_path':[],'next_index':0}
            stack.append(frame)
        else:phase='initiation_failed'
        emit({'event':'enter','instance_id':instance,'option':option,'bindings':arguments,'call_site':site,'initiation_ok':admitted})
        if not admitted:raise Halt()
        sequence(task['options'][option]['body'],frame)
        succeeded=guard(task['options'][option]['success'],arguments)
        stack.pop()
        if not succeeded:phase='success_failed'
        elif not stack:phase='completed'
        emit({'event':'return','instance_id':instance,'option':option,'success_ok':succeeded})
        if not succeeded:raise Halt()
    def sequence(body,frame):
        for index,instruction in enumerate(body):
            op=instruction['op'];frame['next_index']=index+1
            owner=frame['option'];instance=frame['instance_id'];path=frame['body_path'];local=frame['bindings']
            if op=='call':
                # All expressions evaluate before a callee can mutate global registers.
                captured={key:expression(value,local) for key,value in instruction['args'].items()}
                invocation(instruction['option'],captured,{'caller_instance':instance,'body_path':copy.deepcopy(path),'index':index})
            elif op=='if':
                chosen=guard(instruction['test'],local);branch='then' if chosen else 'else'
                child={'frame_kind':'branch','instance_id':instance,'option':owner,'bindings':copy.deepcopy(local),'body_path':path+[index,branch],'next_index':0}
                stack.append(child)
                emit({'event':'branch','instance_id':instance,'option':owner,'body_path':path,'index':index,'condition':chosen})
                sequence(instruction[branch],child);stack.pop()
            else:
                event={'event':'effect','instance_id':instance,'option':owner,'body_path':path,'index':index,'op':op}
                if op=='swap':
                    old=state.copy();state['x']=old['y'];state['y']=old['x']
                else:
                    operand=expression(instruction['value'],local);register=instruction['register']
                    state[register]=state[register]+operand if op=='add' else operand*state[register]
                    event.update(register=register,value=operand)
                event['state']=state.copy();emit(event)
    pending=False
    captured={key:expression(value,{}) for key,value in task['entry']['args'].items()}
    try:invocation(task['entry']['option'],captured,None)
    except Halt:pass
    return events,snapshots


def solve(task):
    try:validate(task)
    except (ValueError,TypeError,KeyError):return invalid('invalid_task')
    all_events,all_snapshots=full_execution(task)
    history=task['history']
    if type(history) is not list or len(history)>len(all_events):return invalid('invalid_history')
    if typed_key(history)!=typed_key(all_events[:len(history)]):return invalid('invalid_history')
    checkpoint=all_snapshots[len(history)]
    if task['mode']=='fresh':
        if history or task['checkpoint'] is not None:return invalid('invalid_checkpoint')
    elif typed_key(task['checkpoint'])!=typed_key(checkpoint):return invalid('invalid_checkpoint')
    previous=sum(event['event']=='effect' for event in history)
    if task['mode']=='restart':
        all_events,all_snapshots=full_execution(task,checkpoint['state']);start=0
    else:start=len(history)
    stop=min(len(all_events),start+task['event_budget'])
    machine=all_snapshots[stop];new_events=all_events[start:stop]
    status='paused' if machine['phase']=='running' else machine['phase']
    return {'status':status,'error':None,'checkpoint':machine,'new_events':new_events,'previous_effects':previous,
            'new_effects':sum(event['event']=='effect' for event in new_events),
            'next_event':all_events[stop] if status=='paused' else None}
