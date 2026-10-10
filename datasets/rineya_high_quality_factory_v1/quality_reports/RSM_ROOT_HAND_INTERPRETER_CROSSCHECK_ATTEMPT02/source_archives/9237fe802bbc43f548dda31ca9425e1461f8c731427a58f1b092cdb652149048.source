"""Own public option_resume v1 parser and event/frame solver. No production reads."""
import copy,json

class InvalidTask(ValueError):pass

def need(condition,reason):
    if not condition:raise InvalidTask(reason)
def exact(d,keys):return type(d) is dict and set(d)==set(keys)
def integer(v):return type(v) is int
def name(v):return type(v) is str and bool(v)
def typed_equal(a,b):
    try:return json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(b,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (ValueError,TypeError):return False

def parse_task(t):
    need(exact(t,['kind','version','initial_state','options','entry','mode','history','checkpoint','event_budget']),'envelope')
    need(t['kind']=='option_resume' and integer(t['version']) and t['version']==1,'kind/version')
    need(exact(t['initial_state'],['x','y']) and all(integer(v) for v in t['initial_state'].values()),'state')
    need(type(t['options']) is dict and bool(t['options']) and all(name(k) for k in t['options']),'options')
    need(type(t['mode']) is str and t['mode'] in ('fresh','resume','restart'),'mode')
    need(integer(t['event_budget']) and t['event_budget']>=0,'budget')
    options=t['options'];edges={k:set() for k in options}
    for key,o in options.items():
        need(exact(o,['params','initiation','success','body']),'option fields')
        need(type(o['params']) is list and all(name(p) for p in o['params']) and len(set(o['params']))==len(o['params']),'params')
    def expr(e,params):
        need(type(e) is dict and len(e)==1,'expression shape')
        if 'const' in e:need(integer(e['const']),'const integer')
        elif 'reg' in e:need(type(e['reg']) is str and e['reg'] in ('x','y'),'register')
        elif 'param' in e:need(name(e['param']) and e['param'] in params,'parameter reference')
        else:need(False,'expression key')
    def predicate(p,params):
        need(exact(p,['all']) and type(p['all']) is list,'predicate')
        for a in p['all']:
            need(exact(a,['left','cmp','right']) and type(a['cmp']) is str and a['cmp'] in ('lt','eq','ge'),'atom')
            expr(a['left'],params);expr(a['right'],params)
    def arguments(target,args,params):
        need(name(target) and target in options,'call target')
        need(type(args) is dict and set(args)==set(options[target]['params']),'argument key set')
        for e in args.values():expr(e,params)
    def body(items,params,owner):
        need(type(items) is list,'instruction list')
        for i in items:
            need(type(i) is dict and type(i.get('op')) is str,'instruction')
            op=i['op']
            if op in ('add','scale'):
                need(exact(i,['op','register','value']) and type(i['register']) is str and i['register'] in ('x','y'),'effect shape');expr(i['value'],params)
            elif op=='swap':need(exact(i,['op']),'swap shape')
            elif op=='call':
                need(exact(i,['op','option','args']),'call shape');arguments(i['option'],i['args'],params);edges[owner].add(i['option'])
            elif op=='if':
                need(exact(i,['op','test','then','else']),'if shape');predicate(i['test'],params);body(i['then'],params,owner);body(i['else'],params,owner)
            else:need(False,'unknown instruction')
    for key,o in options.items():
        predicate(o['initiation'],o['params']);predicate(o['success'],o['params']);body(o['body'],o['params'],key)
    need(exact(t['entry'],['option','args']),'entry');arguments(t['entry']['option'],t['entry']['args'],[])
    visiting=set();done=set()
    def visit(k):
        need(k not in visiting,'recursive call graph including unused branches/options')
        if k in done:return
        visiting.add(k)
        for v in edges[k]:visit(v)
        visiting.remove(k);done.add(k)
    for key in options:visit(key)
    return t

def evaluate(e,state,bindings):
    if 'const' in e:return e['const']
    if 'reg' in e:return state[e['reg']]
    return bindings[e['param']]
def test(p,state,bindings):
    def atom(a):
        x,y=evaluate(a['left'],state,bindings),evaluate(a['right'],state,bindings)
        return x<y if a['cmp']=='lt' else x==y if a['cmp']=='eq' else x>=y
    return all(atom(a) for a in p['all'])

class Machine:
    def __init__(self,t,state=None,snapshot=None):
        self.t=t
        if snapshot is not None:
            s=copy.deepcopy(snapshot);self.state=s['state'];self.frames=s['frames'];self.counter=s['next_instance_id'];self.pending=s['root_pending'];self.phase=s['phase']
        else:
            self.state=copy.deepcopy(t['initial_state'] if state is None else state);self.frames=[];self.counter=0;self.pending=True;self.phase='running'
    def instructions(self,frame):
        items=self.t['options'][frame['option']]['body'];path=frame['body_path']
        for n in range(0,len(path),2):items=items[path[n]][path[n+1]]
        return items
    def normalize(self):
        while self.frames and self.frames[-1]['frame_kind']=='branch' and self.frames[-1]['next_index']==len(self.instructions(self.frames[-1])):self.frames.pop()
    def snapshot(self):
        self.normalize()
        return copy.deepcopy({'state':self.state,'frames':self.frames,'next_instance_id':self.counter,'root_pending':self.pending,'phase':self.phase})
    def peek(self):
        if self.phase!='running':return None
        duplicate=Machine(self.t,snapshot=self.snapshot());return duplicate.step()
    def step(self):
        self.normalize()
        if self.phase!='running':return None
        if self.pending:
            target=self.t['entry']['option'];bindings={k:evaluate(e,self.state,{}) for k,e in self.t['entry']['args'].items()};site=None;self.pending=False;event=self.enter(target,bindings,site)
        else:
            frame=self.frames[-1];items=self.instructions(frame);index=frame['next_index'];opt=frame['option'];instance=frame['instance_id'];path=frame['body_path']
            if index==len(items):
                assert frame['frame_kind']=='option'
                success=test(self.t['options'][opt]['success'],self.state,frame['bindings']);self.frames.pop();self.normalize()
                event={'event':'return','instance_id':instance,'option':opt,'success_ok':success}
                if not success:self.phase='success_failed'
                elif not self.frames:self.phase='completed'
            else:
                i=items[index];op=i['op']
                if op=='call':
                    bindings={k:evaluate(e,self.state,frame['bindings']) for k,e in i['args'].items()};site={'caller_instance':instance,'body_path':copy.deepcopy(path),'index':index};frame['next_index']+=1;event=self.enter(i['option'],bindings,site)
                elif op=='if':
                    decision=test(i['test'],self.state,frame['bindings']);frame['next_index']+=1
                    event={'event':'branch','instance_id':instance,'option':opt,'body_path':copy.deepcopy(path),'index':index,'condition':decision}
                    self.frames.append({'frame_kind':'branch','instance_id':instance,'option':opt,'bindings':copy.deepcopy(frame['bindings']),'body_path':path+[index,'then' if decision else 'else'],'next_index':0})
                else:
                    event={'event':'effect','instance_id':instance,'option':opt,'body_path':copy.deepcopy(path),'index':index,'op':op}
                    if op=='swap':self.state['x'],self.state['y']=self.state['y'],self.state['x']
                    else:
                        value=evaluate(i['value'],self.state,frame['bindings']);reg=i['register'];event.update(register=reg,value=value)
                        self.state[reg]=self.state[reg]+value if op=='add' else self.state[reg]*value
                    event['state']=copy.deepcopy(self.state);frame['next_index']+=1
        self.normalize();return event
    def enter(self,opt,bindings,site):
        instance=self.counter;self.counter+=1;passed=test(self.t['options'][opt]['initiation'],self.state,bindings)
        event={'event':'enter','instance_id':instance,'option':opt,'bindings':copy.deepcopy(bindings),'call_site':site,'initiation_ok':passed}
        if passed:self.frames.append({'frame_kind':'option','instance_id':instance,'option':opt,'bindings':copy.deepcopy(bindings),'body_path':[],'next_index':0})
        else:self.phase='initiation_failed'
        return event

def invalid(code):return {'status':'invalid','error':code,'checkpoint':None,'new_events':[],'previous_effects':None,'new_effects':0,'next_event':None}
def solve(t):
    try:parse_task(t)
    except (InvalidTask,TypeError,KeyError,RecursionError):return invalid('invalid_task')
    m=Machine(t);history=t['history']
    if type(history) is not list:return invalid('invalid_history')
    for supplied in history:
        expected=m.step()
        if expected is None or not typed_equal(supplied,expected):return invalid('invalid_history')
    snap=m.snapshot()
    if t['mode']=='fresh':
        if history or t['checkpoint'] is not None:return invalid('invalid_checkpoint')
    elif not typed_equal(t['checkpoint'],snap):return invalid('invalid_checkpoint')
    previous=sum(e['event']=='effect' for e in history)
    if t['mode']=='restart':m=Machine(t,state=snap['state'])
    events=[]
    for _ in range(t['event_budget']):
        event=m.step()
        if event is None:break
        events.append(event)
    status='paused' if m.phase=='running' else m.phase
    return {'status':status,'error':None,'checkpoint':m.snapshot(),'new_events':events,'previous_effects':previous,'new_effects':sum(e['event']=='effect' for e in events),'next_event':m.peek() if status=='paused' else None}

def parse_raw(text):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise InvalidTask('duplicate JSON key')
            result[k]=v
        return result
    try:return solve(json.loads(text,object_pairs_hook=unique,parse_constant=lambda v:(_ for _ in ()).throw(InvalidTask('nonfinite JSON'))))
    except (ValueError,TypeError):return invalid('invalid_task')
