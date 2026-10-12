"""Freeze fresh RegisterWorld recovery TRAIN/DEV data; never generate confirmation."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsv3wm import REPO
from dsv3wm.recovery_data import write_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=202610112)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-public", type=Path, action="append",
                        help="public prior train/dev/confirmation.jsonl; repeat all three, never label vault files")
    args = parser.parse_args()
    prior = args.prior_public or [REPO / "dsv3wm/datasets/pilot-v1" / (name + ".jsonl")
                                  for name in ("train", "dev", "confirmation")]
    manifest = write_recovery(args.output, seed=args.seed, prior_public=prior)
    print(json.dumps({"manifest": str(args.output.resolve() / "manifest.json"), "counts": manifest["counts"],
                      "parent_counts": manifest["primary_parent_counts"], "train_legality": manifest["train_legality"],
                      "confirmation_rows": 0}, indent=2))


if __name__ == "__main__":
    main()
