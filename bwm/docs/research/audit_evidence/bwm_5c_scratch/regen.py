import sys, json, hashlib
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_Consequence_World_Model_Bundle_2026-10-05/experiments/RCWM-T1"
sys.path.insert(0, R + "/scripts")
import ret_common as RC, ret_gen as RGEN
assert "torch" not in sys.modules
seal = json.load(open(R + "/docs/retention/draws/confirm_1708484032065101819.SEAL.json"))
rows, _ = RGEN.generate(int(seal["seed"]), seal["purpose"], {c: list(v) for c, v in seal["plan"].items()}, bool(seal["surface_diversity"]), int(seal["n_per_cell"]))
print("regen sha", hashlib.sha256(RGEN.serialize(rows).encode("utf-8")).hexdigest(), "seal", seal["file_sha256"])
sel = R + "/docs/retention/DEV_SELECTION.json"
print("selection sha", RC.sha256_file(sel), "seal", seal["selection_sha256"])
d = json.load(open(sel))
print(json.dumps({k: v for k, v in d.items() if k not in ("accuracy",)}, indent=0)[:2500])
print("torch imported:", "torch" in sys.modules)
