import hashlib, json, os, sys
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1"
B = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs"
D = R + "/runs/retention/confirm_1708484032065101819"
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1<<20), b''):
            h.update(ch)
    return h.hexdigest()
exp = {
 "BW_Fdelta_s0": ("2ed0bb0064bc6b11ec9c661840a38094e468462cea2ff833cda3bf234ead65ae","3038c77961f33461e1bef736959765434c0be473d991b0ef92c4813d269b3b8c"),
 "BW_Pa_v2_s1": ("ee1c5aec5e4a55e30f6df0e30a3cd68242c928cd158f9e5cf06184f409108110","f6a2c4efb381678ac44bebef1165d84d587a81e72c8a8ea1c27062f2bc7494bf"),
}
print("== adapters on disk")
for run,(m,c) in exp.items():
    d = B + "/" + run + "/adapter"
    print(run, sorted(os.listdir(d)))
    sm = sha(d + "/adapter_model.safetensors"); sc = sha(d + "/adapter_config.json")
    st = os.stat(d + "/adapter_model.safetensors")
    print("  model", sm, sm==m, "mtime", st.st_mtime, "size", st.st_size)
    print("  config", sc, sc==c, "mtime", os.stat(d+"/adapter_config.json").st_mtime)
    print("  config content:", open(d+"/adapter_config.json").read())
print("== freeze files")
fr = json.load(open(R + "/docs/retention/RET_FREEZE.json"))
for k,v in fr["files"].items():
    s = sha(R + "/" + k); print(k, s==v, s[:16], os.stat(R+"/"+k).st_mtime)
print("freeze file sha", sha(R + "/docs/retention/RET_FREEZE.json"))
print("== draw")
print("draw", sha(R + "/docs/retention/draws/confirm_1708484032065101819.jsonl"))
print("seal", sha(R + "/docs/retention/draws/confirm_1708484032065101819.SEAL.json"))
print("== run files")
done = json.load(open(D + "/DONE.json"))
for st_, files in done["files"].items():
    for k,v in files.items():
        s = sha(D + "/" + k); print(k, s==v)
print("provenance", sha(D+"/provenance.json"), done["provenance_sha256"])
for f in ["analysis/RET_RESULTS.json","analysis/RET_RESULTS.md","DONE.json"]:
    p = D + "/" + f; print(f, sha(p), os.stat(p).st_mtime, os.stat(p).st_ctime)
