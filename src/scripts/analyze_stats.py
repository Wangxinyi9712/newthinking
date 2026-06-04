from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from analysis.stats import run_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--metric", default="dice")
    parser.add_argument("--out", default="stats_report.json")
    args = parser.parse_args()

    report = run_stats(args.csv, metric=args.metric)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
