from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="runs/experiments")
    parser.add_argument("--out", type=str, default="runs/experiments_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)
    out = Path(args.out)

    rows = []

    for result_file in sorted(root.rglob("eval_results_*.json")):
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        group = result_file.parents[1].name if len(result_file.parents) >= 2 else result_file.parent.name
        seed_dir = result_file.parent.name

        row = {
            "group": group,
            "seed_dir": seed_dir,
            "result_file": str(result_file),
            "dice": data.get("dice", ""),
            "iou": data.get("iou", ""),
            "precision": data.get("precision", ""),
            "recall": data.get("recall", ""),
            "f1": data.get("f1", ""),
            "minority_f1": data.get("minority_f1", ""),
            "hd95": data.get("hd95", ""),
            "source": data.get("source", ""),
            "ckpt": data.get("ckpt", ""),
        }
        rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "group",
        "seed_dir",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
        "minority_f1",
        "hd95",
        "source",
        "ckpt",
        "result_file",
    ]

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"[Saved] {out}")
    print(f"[Found] {len(rows)} result files")


if __name__ == "__main__":
    main()