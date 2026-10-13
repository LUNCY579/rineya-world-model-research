import sys, random, json, statistics as st
ROOT=r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze")
n_tok=lambda x: len(tok(x, add_special_tokens=False).input_ids)
rng=random.Random(424242)
mx=0; vals=[]
for i in range(400):
    g=None
    while g is None: g=B.generate(rng,15,full_goal_frac=1.0)
    objs,init,goal=g
    p=n_tok(B.proposer_prompt(init,goal)); vals.append(p); mx=max(mx,p)
    # also a holding state variant not needed
print("15-block prompt tokens mean",round(st.mean(vals),1),"max",mx)
# tokens per action on random action sequences of 92 actions
acts=[]
names=B.NAMES
for k in range(200):
    plan=[]
    for j in range(92):
        t=rng.choice(["pick-up","put-down","stack","unstack"])
        a,b=rng.sample(names,2)
        plan.append((t,a,b) if t in("stack","unstack") else (t,a))
    try:
        s=B.target_actions(plan)
    except Exception as e:
        print("err",e); break
    acts.append(n_tok(s)/92)
print("tok/action random 92-action", round(st.mean(acts),3), round(max(acts),3))
# realistic: unstack/put-down/pick-up/stack balanced from solved plans in h2
rows=[json.loads(l) for l in open(ROOT+"/runs/BW_B3_HARD/h2_beam/plans.jsonl")]
ratios=[]
for r in rows:
    if r["success"] and r["len"]>0:
        plan=[tuple(x.split()) for x in r["plan"]]
        ratios.append(n_tok(B.target_actions(plan))/len(plan))
print("H2 solved tok/action", round(st.mean(ratios),3), round(max(ratios),3), "max len", max(r["len"] for r in rows if r["success"]))
