"""Separate recursive fresh-trace simulator; no frame-machine import or production read.
It assumes parser-validated input and returns full trace/state/phase, not checkpoints.
"""
import copy

def simulate(t):
    state=copy.deepcopy(t['initial_state']);events=[];counter=0;phase='running'
    def value(e,b):
        if set(e)=={'const'}:return e['const']
        if set(e)=={'reg'}:return state[e['reg']]
        return b[e['param']]
    def predicate(p,b):
        outcomes=[]
        for a in p['all']:
            l,r=value(a['left'],b),value(a['right'],b)
            outcomes.append({'lt':lambda:l<r,'eq':lambda:l==r,'ge':lambda:l>=r}[a['cmp']]())
        return all(outcomes)
    def body(items,opt,instance,b,path):
        for i,ins in enumerate(items):
            op=ins['op']
            if op=='call':
                copied={k:value(e,b) for k,e in ins['args'].items()}
                if not option(ins['option'],copied,{'caller_instance':instance,'body_path':copy.deepcopy(path),'index':i}):return False
            elif op=='if':
                decision=predicate(ins['test'],b);events.append({'event':'branch','instance_id':instance,'option':opt,'body_path':copy.deepcopy(path),'index':i,'condition':decision})
                if not body(ins['then' if decision else 'else'],opt,instance,dict(b),path+[i,'then' if decision else 'else']):return False
            else:
                event={'event':'effect','instance_id':instance,'option':opt,'body_path':copy.deepcopy(path),'index':i,'op':op}
                if op=='swap':state.update(x=state['y'],y=state['x'])
                else:
                    operand=value(ins['value'],b);r=ins['register'];event.update(register=r,value=operand)
                    if op=='add':state[r]+=operand
                    else:state[r]*=operand
                event['state']=copy.deepcopy(state);events.append(event)
        return True
    def option(opt,b,site):
        nonlocal counter,phase
        instance=counter;counter+=1;o=t['options'][opt];passed=predicate(o['initiation'],b)
        events.append({'event':'enter','instance_id':instance,'option':opt,'bindings':dict(b),'call_site':site,'initiation_ok':passed})
        if not passed:phase='initiation_failed';return False
        if not body(o['body'],opt,instance,b,[]):return False
        passed=predicate(o['success'],b);events.append({'event':'return','instance_id':instance,'option':opt,'success_ok':passed})
        if not passed:phase='success_failed';return False
        return True
    args={k:value(e,{}) for k,e in t['entry']['args'].items()}
    if option(t['entry']['option'],args,None):phase='completed'
    return {'events':events,'state':state,'phase':phase,'next_instance_id':counter}
