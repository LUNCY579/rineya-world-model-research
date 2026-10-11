import json, collections
S='C:/Users/clclc/AppData/Local/Temp/claude/bwm_5c_scratch/'
R='C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1/runs/retention/confirm_1708484032065101819/'
arms=['off','t2s0','x_bwf','x_bwpa','supa']
c1={}
for line in open(S+'my_prompt_grades.jsonl'):
    r=json.loads(line); c1[r['pid']]=r
c2=json.load(open(S+'graded.json'))['graded']
gen={}
for a in arms:
    p=R+('supa/' if a=='supa' else 'sparse/')+'gen_%s.jsonl'%a
    gen[a]={json.loads(l)['pid']:json.loads(l) for l in open(p)}
dis=collections.Counter(); n=0; textmis=0; ex=[]
for a in arms:
    for pid,r in c1.items():
        n+=1
        if r[a]['text']!=gen[a][pid]['text']: textmis+=1
        g=c2[a][pid]
        for k1,k2 in (('P','primary'),('L','lenient'),('P','frozen_primary'),('L','frozen_lenient')):
            if r[a][k1]!=g[k2]:
                dis[(a,k2)]+=1
                if len(ex)<10: ex.append((a,pid,k2,r[a]['text'],r['gold']))
print('prompts x arms',n,'c1 text vs gen mismatches',textmis,'pids c1',len(c1),'c2',{a:len(c2[a]) for a in arms})
print('disagreements check1 vs check2 (own / frozen ret_grade):',dict(dis) or 0)
print(ex)
# unanswerable-twin answers per arm
for a in arms:
    bad=[]
    for pid,r in c1.items():
        if r['category']!='unknown_twin': continue
        if (r['gold'] is None or str(r['gold']).upper()=='UNKNOWN') and gen[a][pid]['text'].strip().upper()!='UNKNOWN':
            bad.append((pid.split('/',1)[1],gen[a][pid]['text']))
    print(a,'unanswerable non-UNKNOWN',len(bad),bad)
# x_bwf unit_conv losses/gains
for a in ['x_bwf','x_bwpa']:
    lost=[(pid.split('/',1)[1],r['gold'],r['off']['text'],r[a]['text']) for pid,r in c1.items() if r['category']=='unit_conv' and r['off']['P'] and not r[a]['P']]
    gain=[(pid.split('/',1)[1],r['gold'],r['off']['text'],r[a]['text']) for pid,r in c1.items() if r['category']=='unit_conv' and not r['off']['P'] and r[a]['P']]
    print(a,'unit_conv lost',len(lost),lost); print(a,'unit_conv gained',len(gain),gain)
