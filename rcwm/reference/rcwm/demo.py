"""Run a small deterministic CPU training demonstration, not a Rineya benchmark."""
from __future__ import annotations
import argparse,json,platform,time
from pathlib import Path
import numpy as np
import torch
from .model import FeatureTrajectoryTransformer
from .synthetic import make_dataset,rollout,FEATURES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("demo_output"))
    parser.add_argument("--steps",type=int,default=250)
    parser.add_argument("--seed",type=int,default=17)
    args=parser.parse_args()
    if args.steps < 1: parser.error("steps must be positive")
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    train=tuple(torch.from_numpy(x) for x in make_dataset(1024,4,args.seed))
    check=tuple(torch.from_numpy(x) for x in make_dataset(256,4,args.seed+1))
    model=FeatureTrajectoryTransformer(max_horizon=8)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    model.eval()
    with torch.no_grad():
        initial=float(torch.nn.functional.mse_loss(model(check[0],check[1])["features"],check[2]))
    losses=[];start=time.perf_counter()
    model.train()
    for step in range(args.steps):
        idx=torch.randint(len(train[0]),(64,))
        pred=model(train[0][idx],train[1][idx])["features"]
        loss=torch.nn.functional.mse_loss(pred,train[2][idx])
        if not torch.isfinite(loss): raise RuntimeError("nonfinite loss")
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step%25==0 or step==args.steps-1:
            losses.append({"step":step+1,"train_batch_mse":float(loss.detach())})
    elapsed=time.perf_counter()-start
    model.eval()
    plans=torch.tensor([[1,2,3],[4,0,0]],dtype=torch.long)
    state=torch.zeros(2,4)
    with torch.no_grad():
        final=float(torch.nn.functional.mse_loss(model(check[0],check[1])["features"],check[2]))
        predicted=model(state,plans)["features"].sum(1).numpy()
    actual=np.stack([rollout(np.zeros(4),p.tolist())[0].sum(0) for p in plans])
    # Equal 3-step windows; wait has cost 0.1 and other actions cost 1.0.
    # Physical consequences stay fixed while damage/cost preferences change.
    goals={"avoid_damage":[0,0,10,-20,-1],"prefer_speed":[0,0,10,-1,-2]}
    results={g:{"weights":w,"predicted_scores":(predicted@np.array(w)).tolist(),
                 "actual_scores":(actual@np.array(w)).tolist()}
             for g,w in goals.items()}
    report={"status":"SYNTHETIC_MECHANICS_ONLY","seed":args.seed,
            "steps":args.steps,"parameters":sum(p.numel() for p in model.parameters()),
            "initial_check_mse":initial,"final_check_mse":final,
            "training_seconds":elapsed,"features":list(FEATURES),
            "predicted_cumulative_features":predicted.tolist(),
            "actual_cumulative_features":actual.tolist(),"goal_revaluation":results,
            "environment":{"python":platform.python_version(),"torch":torch.__version__,
                           "numpy":np.__version__,"device":"cpu"},
            "limitations":["Tiny repeated-support simulator, not held-out family transfer",
                           "Supervised measured features, not unsupervised semantic discovery",
                           "No native Rineya checkpoint, calibrated uncertainty or RL policy training"],
            "loss_trace":losses}
    (args.out/"demo_result.json").write_text(json.dumps(report,indent=2)+"\n")
    torch.save({"state_dict":model.state_dict(),"seed":args.seed,
                "scope":"synthetic_mechanics"},args.out/"synthetic_demo_weights.pt")
    print(json.dumps({k:report[k] for k in ("status","parameters","initial_check_mse",
                                           "final_check_mse","training_seconds")},indent=2))

if __name__=="__main__": main()
