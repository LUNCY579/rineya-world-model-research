"""Build fresh tick-compatible T2 Ledger TRAIN/DEV inputs; no models or sealed labels."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsv3wm.ledger_recovery import T2_TRAIN, write_ledger_recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--seed", type=int, default=202610113)
    parser.add_argument("--prior-train", type=Path, default=T2_TRAIN, help="SHA-bound original t2_train.jsonl, if available")
    parser.add_argument("--prior-public", action="append", default=[], type=Path,
                        help="known new public train/dev/confirmation.jsonl for parent exclusions; never a label vault")
    args = parser.parse_args()
    result = write_ledger_recovery(args.output, seed=args.seed, prior_train=args.prior_train, prior_public=args.prior_public)
    print(json.dumps({"manifest": str(args.output.resolve() / "manifest.json"), "counts": result["counts"],
                      "training_support_available": result["training_support"]["available"],
                      "confirmation_rows": 0}, indent=2))


if __name__ == "__main__":
    main()
