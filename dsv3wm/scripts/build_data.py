"""Generate fresh causal data and seal confirmation without logging labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsv3wm.data import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--rows", type=int, default=120, help="rows per train/dev/confirmation split")
    parser.add_argument("--output", type=Path, required=True, help="new dataset directory")
    parser.add_argument("--vault", type=Path, required=True, help="new external private confirmation directory")
    args = parser.parse_args()
    manifest = write_dataset(args.output, args.vault, seed=args.seed, rows_per_split=args.rows)
    print(json.dumps({"manifest": str(args.output.resolve() / "manifest.json"),
                      "rows": manifest["integrity"]["counts"],
                      "confirmation_sha256": manifest["sealed_confirmation"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
