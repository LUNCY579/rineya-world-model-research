import json, glob, os
R = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs"
rows = []
for d in ["BW_B3_HARD/h2_beam", "BW_TEST_B2/hard_beam", "BW_TEST_B2/mirror_beam", "BW_SEEDS/R0_beam", "BW_SEEDS/R1_beam", "BW_SEEDS/R2_beam"]:
    for sh in sorted(glob.glob(f"{R}/{d}/shard_*")):
        sp = f"{sh}/summary.json"
        if not os.path.exists(sp): continue
        s = json.load(open(sp))["summary"]
        pl = [json.loads(l) for l in open(f"{sh}/plans.jsonl")]
        fc = sum(1 for _ in open(f"{sh}/fcache.jsonl")) if os.path.exists(f"{sh}/fcache.jsonl") else None
        tok = sum(p["gen_tokens"] for p in pl)
        big = sum(p["n_blocks"] >= 11 for p in pl)
        tokbig = sum(p["gen_tokens"] for p in pl if p["n_blocks"] >= 11)
        prog = open(f"{sh}/progress.txt").read().splitlines() if os.path.exists(f"{sh}/progress.txt") else []
        rows.append((d.split('/')[-1] if 'SEEDS' not in d else d.split('/')[-1], os.path.basename(sh), len(pl), big, s["wall_s"], fc, s.get("f_model_queries"), tok, tokbig, sum(p['success'] for p in pl)))
print("set shard n n>=11 wall_s fcache_lines logged_fq gen_tokens gen_tokens_big succ s/inst s/ktok")
for r in rows:
    print(*r, round(r[4]/r[2],1), round(r[4]/(r[7]/1000),2))
