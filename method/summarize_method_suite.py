from __future__ import annotations
import argparse
import csv
from pathlib import Path
import statistics
import yaml


def read_last_row(csv_path: Path):
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="src/configs/method_suite.yaml")
    ap.add_argument("--runs-root", default="runs/method_suite")
    ap.add_argument("--methods", nargs="+", default=None, help="Optional override method list")
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    ap.add_argument("--out", default="method/method_comparison.csv")
    args = ap.parse_args()

    if args.methods is None:
        with open(args.suite, "r", encoding="utf-8") as f:
            suite = yaml.safe_load(f)
        methods = list(suite["methods"].keys())
    else:
        methods = args.methods

    out_rows = []
    for m in methods:
        vals = {"dice": [], "iou": [], "precision": [], "recall": [], "hd95": []}
        for s in args.seeds:
            h = Path(args.runs_root) / m / f"seed_{s}" / "history.csv"
            if not h.exists():
                continue
            last = read_last_row(h)
            if not last:
                continue
            for k in vals:
                vals[k].append(float(last[f"val_{k}"]))

        if len(vals["dice"]) == 0:
            continue

        out_rows.append({
            "method": m,
            "dice_mean": statistics.mean(vals["dice"]),
            "dice_std": statistics.pstdev(vals["dice"]) if len(vals["dice"]) > 1 else 0.0,
            "iou_mean": statistics.mean(vals["iou"]),
            "precision_mean": statistics.mean(vals["precision"]),
            "recall_mean": statistics.mean(vals["recall"]),
            "hd95_mean": statistics.mean(vals["hd95"]),
            "n_seed": len(vals["dice"]),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method", "dice_mean", "dice_std", "iou_mean",
                "precision_mean", "recall_mean", "hd95_mean", "n_seed"
            ],
        )
        w.writeheader()
        w.writerows(out_rows)

    print(f"[DONE] {out_path}")


if __name__ == "__main__":
    main()