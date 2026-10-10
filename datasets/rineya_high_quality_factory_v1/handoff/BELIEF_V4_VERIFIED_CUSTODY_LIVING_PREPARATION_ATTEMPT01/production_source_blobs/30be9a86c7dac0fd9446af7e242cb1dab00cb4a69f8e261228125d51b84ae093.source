"""Hand and separately structured recursive checks; future production bytes unread."""
import copy,importlib.util,json,pathlib,unittest
HERE=pathlib.Path(__file__).resolve().parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=module('own_option_frame_solver_r1',HERE/'solver-r1.py')
r=module('own_option_recursive_simulator_r1',HERE/'recursive-simulator-r1.py')
FIX=json.loads((HERE/'hand-calibration-r1.json').read_text())

def clone(v):return copy.deepcopy(v)
def reverse_objects(v):
    if type(v) is dict:return {k:reverse_objects(v[k]) for k in reversed(v)}
    if type(v) is list:return [reverse_objects(x) for x in v]
    return v

def resume_rich(cut=5,budget=48,mode='resume'):
    t=clone(FIX['rich']['task']);t.update(mode=mode,history=clone(FIX['rich']['events'][:cut]),checkpoint=clone(FIX['rich'][f'checkpoint_after_{cut}']),event_budget=budget);return t

class Calibration(unittest.TestCase):
    def eq(self,a,b):self.assertTrue(s.typed_equal(a,b),f'actual={a!r}; expected={b!r}')
    def code(self,t,code):self.eq(s.solve(t),s.invalid(code))
    def test_manual_full_trace(self):
        rich=FIX['rich'];o=s.solve(rich['task']);self.eq(o['new_events'],rich['events']);self.eq(o['checkpoint'],rich['terminal'])
        self.eq(o,{'status':'completed','error':None,'checkpoint':rich['terminal'],'new_events':rich['events'],'previous_effects':0,'new_effects':4,'next_event':None})
        rec=r.simulate(rich['task']);self.eq(rec['events'],rich['events']);self.eq(rec['state'],rich['terminal']['state']);self.eq(rec['phase'],'completed')
    def test_manual_checkpoint_copies(self):
        for cut in (5,6):
            t=clone(FIX['rich']['task']);t['event_budget']=cut;o=s.solve(t);self.eq(o['checkpoint'],FIX['rich'][f'checkpoint_after_{cut}']);self.eq(o['new_events'],FIX['rich']['events'][:cut]);self.eq(o['next_event'],FIX['rich']['events'][cut]);self.eq(o['status'],'paused')
    def test_resume_retains_guard_branch_and_saved_argument(self):
        t=resume_rich();o=s.solve(t);self.eq(o['new_events'],FIX['rich']['events'][5:]);self.eq(o['checkpoint'],FIX['rich']['terminal']);self.eq(o['previous_effects'],2);self.eq(o['new_effects'],2)
        active=t['checkpoint']['frames'][-1];self.assertFalse(s.test(t['options']['child']['initiation'],t['checkpoint']['state'],active['bindings']));self.assertFalse(s.test(t['options']['child']['body'][0]['test'],t['checkpoint']['state'],active['bindings']))
        self.eq(o['new_events'][0]['value'],1);self.eq(t['checkpoint']['state']['x'],4)
    def test_restart_replays_effect_and_rebinds_at_entry(self):
        o=s.solve(resume_rich(mode='restart'));self.eq(o['new_events'],FIX['rich']['restart_events']);self.eq(o['checkpoint'],FIX['rich']['restart_terminal']);self.eq(o['status'],'initiation_failed');self.eq(o['previous_effects'],2);self.eq(o['new_effects'],1)
    def test_second_checkpoint_and_pending_return(self):
        p=s.solve(resume_rich(budget=1));self.eq(p['checkpoint'],FIX['rich']['checkpoint_after_6']);self.eq(p['new_effects'],1);self.eq(p['status'],'paused');self.eq(p['next_event'],FIX['rich']['events'][6])
        u=s.solve(resume_rich(cut=6));self.eq(u['new_events'],FIX['rich']['events'][6:]);self.eq(u['checkpoint'],FIX['rich']['terminal']);self.eq(u['previous_effects'],3);self.eq(u['new_effects'],1)
    def test_argument_counterfactual(self):
        t=clone(FIX['rich']['task']);t['entry']['args']['k']['const']=3;o=s.solve(t);self.eq(o['status'],'completed');self.eq(o['checkpoint']['state'],{'x':4,'y':4});self.eq(o['new_events'][7]['value'],3)
    def test_entry_failure_has_no_body_effect(self):
        t=clone(FIX['rich']['task']);t['entry']['args']['k']['const']=-1;o=s.solve(t);self.eq(o['status'],'initiation_failed');self.eq(o['new_effects'],0);self.eq(o['new_events'],[{'event':'enter','instance_id':0,'option':'root','bindings':{'k':-1},'call_site':None,'initiation_ok':False}]);self.eq(o['checkpoint'],{'state':{'x':0,'y':0},'frames':[],'next_instance_id':1,'root_pending':False,'phase':'initiation_failed'})
    def test_success_failure_preserves_effects(self):
        t=clone(FIX['rich']['task']);t['options']['root']['success']['all'][0]['right']['const']=10;o=s.solve(t);self.eq(o['status'],'success_failed');self.eq(o['new_events'][:-1],FIX['rich']['events'][:-1]);self.eq(o['new_events'][-1],{'event':'return','instance_id':0,'option':'root','success_ok':False});self.eq(o['checkpoint']['state'],{'x':4,'y':3});self.eq(o['new_effects'],4)
    def test_active_leaf_binding_tamper_is_invalid(self):
        t=resume_rich();t['checkpoint']['frames'][-1]['bindings']['n']=2;self.code(t,'invalid_checkpoint')
    def test_other_hand_programs_and_recursive_trace(self):
        for f in FIX['extra']:
            with self.subTest(fixture=f['name']):
                t=f['task'];s.parse_task(t);o=s.solve(t);rec=r.simulate(t)
                self.eq(o['status'],f['expected_phase']);self.eq(o['checkpoint']['state'],f['expected_state']);self.eq(len(o['new_events']),f['expected_events']);self.eq(o['new_effects'],f['expected_effects']);self.eq(o['checkpoint']['next_instance_id'],f['expected_instances']);self.eq(o['new_events'],rec['events']);self.eq(o['checkpoint']['state'],rec['state']);self.eq(o['status'],rec['phase'])
    def test_empty_branch_stack_normalization(self):
        t=clone(FIX['extra'][1]['task']);t['event_budget']=2;o=s.solve(t);self.eq(o['checkpoint']['frames'],[{'frame_kind':'option','instance_id':0,'option':'root','bindings':{},'body_path':[],'next_index':1}]);self.eq(o['next_event'],{'event':'return','instance_id':0,'option':'root','success_ok':True})
    def test_failed_child_advanced_caller_cannot_continue(self):
        t=clone(FIX['extra'][2]['task']);o=s.solve(t);self.eq(o['checkpoint']['frames'],[{'frame_kind':'option','instance_id':0,'option':'root','bindings':{},'body_path':[],'next_index':1}]);t.update(mode='resume',history=o['new_events'],checkpoint=o['checkpoint'],event_budget=48);res=s.solve(t);self.eq(res['new_events'],[]);self.eq(res['status'],'success_failed');self.eq(res['checkpoint'],o['checkpoint'])
    def test_every_boundary_and_budget_against_recursive_full_trace(self):
        for f in [FIX['rich']]+FIX['extra']:
            t=clone(f['task']);rec=r.simulate(t);events=rec['events'];machine=s.Machine(t)
            for cut in range(len(events)+1):
                snapshot=machine.snapshot();remaining=events[cut:]
                for budget in range(len(events)+3):
                    u=clone(t);u.update(mode='resume',history=clone(events[:cut]),checkpoint=clone(snapshot),event_budget=budget);o=s.solve(u)
                    with self.subTest(fixture=f.get('name','rich'),cut=cut,budget=budget):
                        self.eq(o['new_events'],remaining[:budget]);self.eq(o['previous_effects'],sum(e['event']=='effect' for e in events[:cut]));self.eq(o['new_effects'],sum(e['event']=='effect' for e in remaining[:budget]));self.eq(o['status'],rec['phase'] if budget>=len(remaining) else 'paused');self.eq(o['next_event'],None if budget>=len(remaining) else remaining[budget])
                if cut<len(events):self.eq(machine.step(),events[cut])
    def test_prefix_composition_for_two_interruptions(self):
        t=clone(FIX['rich']['task']);t['event_budget']=5;a=s.solve(t);t.update(mode='resume',history=a['new_events'],checkpoint=a['checkpoint'],event_budget=1);b=s.solve(t);t.update(history=a['new_events']+b['new_events'],checkpoint=b['checkpoint'],event_budget=48);c=s.solve(t);self.eq(a['new_events']+b['new_events']+c['new_events'],FIX['rich']['events']);self.eq(c['checkpoint'],FIX['rich']['terminal'])
    def test_next_event_does_not_commit_or_count(self):
        t=clone(FIX['rich']['task']);t['event_budget']=0;o=s.solve(t);self.eq(o['new_events'],[]);self.eq(o['new_effects'],0);self.eq(o['next_event'],FIX['rich']['events'][0]);self.eq(o['checkpoint'],{'state':{'x':0,'y':0},'frames':[],'next_instance_id':0,'root_pending':True,'phase':'running'})
        m=s.Machine(t);before=m.snapshot();first=m.peek();self.eq(m.snapshot(),before);self.eq(m.peek(),first);self.eq(m.step(),first)
    def test_terminal_exact_budget_and_zero_terminal_resume(self):
        for budget in (9,10,48):
            t=clone(FIX['rich']['task']);t['event_budget']=budget;self.eq(s.solve(t)['status'],'completed')
        t=clone(FIX['rich']['task']);t.update(mode='resume',history=FIX['rich']['events'],checkpoint=FIX['rich']['terminal'],event_budget=0);o=s.solve(t);self.eq(o['status'],'completed');self.eq(o['new_events'],[]);self.eq(o['previous_effects'],4);self.eq(o['next_event'],None)
    def test_object_order_invariance_and_list_order_significance(self):
        for t in [FIX['rich']['task'],resume_rich()]+[f['task'] for f in FIX['extra']]:self.eq(s.solve(t),s.solve(reverse_objects(t)))
        t=resume_rich();t['history'][1],t['history'][2]=t['history'][2],t['history'][1];self.code(t,'invalid_history')
    def test_exact_task_and_integer_typing(self):
        cases=[]
        for field,value in [('version',True),('event_budget',True),('event_budget',-1),('mode','retry'),('kind','skills')]:
            t=clone(FIX['rich']['task']);t[field]=value;cases.append(t)
        for value in (True,1.0,'1'):
            t=clone(FIX['rich']['task']);t['initial_state']['x']=value;cases.append(t)
            t=clone(FIX['rich']['task']);t['entry']['args']['k']={'const':value};cases.append(t)
        t=clone(FIX['rich']['task']);t['extra']=0;cases.append(t)
        t=clone(FIX['rich']['task']);del t['entry'];cases.append(t)
        for t in cases:
            with self.subTest(task=t):self.code(t,'invalid_task')
    def test_program_references_shapes_and_unused_validation(self):
        cases=[]
        t=clone(FIX['rich']['task']);t['options']['root']['params']=['k','k'];cases.append(t)
        t=clone(FIX['rich']['task']);t['options']['root']['params']=[''];cases.append(t)
        t=clone(FIX['rich']['task']);t['options']['root']['body'][0]['value']={'param':'missing'};cases.append(t)
        t=clone(FIX['rich']['task']);t['options']['child']['body'][0]['else'][0]['value']={'reg':'z'};cases.append(t)
        t=clone(FIX['rich']['task']);t['options']['child']['body'][0]['else'][0]['value']={'const':2,'reg':'x'};cases.append(t)
        t=clone(FIX['rich']['task']);t['entry']['args']={};cases.append(t)
        t=clone(FIX['rich']['task']);t['entry']['args']['extra']={'const':1};cases.append(t)
        t=clone(FIX['rich']['task']);t['entry']['args']['k']={'param':'k'};cases.append(t)
        t=clone(FIX['rich']['task']);t['options']['child']['body'][0]['test']['all'][0]['cmp']='le';cases.append(t)
        t=clone(FIX['rich']['task']);t['options']['child']['body'][0]['else']=[{'op':'unknown'}];cases.append(t)
        t=clone(FIX['rich']['task']);t['options']['unused']={'params':[],'initiation':{'all':[]},'success':{'all':[]},'body':[{'op':'call','option':'unused','args':{}}]};cases.append(t)
        t=clone(FIX['rich']['task']);t['options']['child']['body'][0]['else']=[{'op':'call','option':'root','args':{'k':{'const':2}}}];cases.append(t)
        for t in cases:
            with self.subTest(task=t):self.code(t,'invalid_task')
    def test_raw_json_duplicate_keys_and_nonfinite_numbers(self):
        self.eq(s.parse_raw(json.dumps(FIX['rich']['task'])),s.solve(FIX['rich']['task']))
        for text in ('{"kind":1,"kind":2}','NaN','Infinity','{"x":NaN}','[1,2]','{'):
            with self.subTest(raw=text):self.eq(s.parse_raw(text),s.invalid('invalid_task'))
    def test_history_typed_fields_and_terminal_policy(self):
        cases=[]
        t=resume_rich();t['history']={};cases.append(t)
        t=resume_rich();t['history'][1]['value']=True;cases.append(t)
        t=resume_rich();t['history'][1]['state']['x']=1.0;cases.append(t)
        t=resume_rich();t['history'][0]['extra']=0;cases.append(t)
        t=resume_rich();t['history'][0]['initiation_ok']=1;cases.append(t)
        t=resume_rich();t['history'][3]['condition']=1;cases.append(t)
        t=clone(FIX['rich']['task']);t.update(mode='resume',history=clone(FIX['rich']['events'])+[clone(FIX['rich']['events'][-1])],checkpoint=clone(FIX['rich']['terminal']));cases.append(t)
        for t in cases:
            with self.subTest(task=t):self.code(t,'invalid_history')
    def test_checkpoint_all_leaf_categories_and_normalization(self):
        for path,value in [(['state','x'],5),(['frames',0,'next_index'],1),(['frames',1,'bindings','n'],2),(['frames',2,'bindings','n'],2),(['next_instance_id'],3),(['root_pending'],True),(['phase'],'completed'),(['frames',0,'instance_id'],True)]:
            t=resume_rich();node=t['checkpoint']
            for key in path[:-1]:node=node[key]
            node[path[-1]]=value
            with self.subTest(path=path):self.code(t,'invalid_checkpoint')
        t=resume_rich(cut=6);branch=clone(FIX['rich']['checkpoint_after_5']['frames'][-1]);branch['next_index']=2;t['checkpoint']['frames'].append(branch);self.code(t,'invalid_checkpoint')
        t=resume_rich();t['checkpoint']['extra']=0;self.code(t,'invalid_checkpoint')
        t=resume_rich();t['checkpoint']=None;self.code(t,'invalid_checkpoint')
        t=clone(FIX['rich']['task']);t['checkpoint']={};self.code(t,'invalid_checkpoint')
        t=resume_rich();t['mode']='fresh';self.code(t,'invalid_checkpoint')
    def test_error_precedence(self):
        t=resume_rich();t['version']=True;t['history']={};t['checkpoint']=None;self.code(t,'invalid_task')
        t=resume_rich();t['history']={};t['checkpoint']=None;self.code(t,'invalid_history')
        t=resume_rich();t['checkpoint']=None;self.code(t,'invalid_checkpoint')
    def test_restart_from_failed_and_completed_checkpoint_is_new_run(self):
        for f in [FIX['rich']]+FIX['extra']:
            t=clone(f['task']);first=s.solve(t);t.update(mode='restart',history=first['new_events'],checkpoint=first['checkpoint']);o=s.solve(t);fresh=clone(f['task']);fresh['initial_state']=clone(first['checkpoint']['state']);expected=s.solve(fresh);self.eq(o['new_events'],expected['new_events']);self.eq(o['checkpoint'],expected['checkpoint']);self.eq(o['previous_effects'],first['new_effects'])

if __name__=='__main__':unittest.main(verbosity=2)
