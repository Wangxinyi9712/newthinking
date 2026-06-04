from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_last_row(csv_path: Path) -> dict | None:
    if not csv_path.exists():
        return None
    last = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, help="e.g. runs/brats_group_a runs/brats_group_b")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows: list[dict] = []

    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.exists():
            continue

        # 兼容两种结构：
        # 1) runs/group/seed_0/history.csv
        # 2) runs/group/history.csv
        seed_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]) if run_dir.is_dir() else []

        if seed_dirs:
            group_name = run_dir.name
            for sd in seed_dirs:
                hist = sd / "history.csv"
                last = _read_last_row(hist)
                if last is None:
                    continue
                last["group"] = group_name
                last["seed_dir"] = sd.name
                rows.append(last)
        else:
            hist = run_dir / "history.csv"
            last = _read_last_row(hist)
            if last is None:
                continue
            last["group"] = run_dir.name
            last["seed_dir"] = "single"
            rows.append(last)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        # 统一字段顺序：group/seed_dir 放前面
        base_fields = list(rows[0].keys())
        for k in ["group", "seed_dir"]:
            if k in base_fields:
                base_fields.remove(k)
        fieldnames = ["group", "seed_dir"] + base_fields

        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[collect] wrote {len(rows)} rows to {out}")
    else:
        print("[collect] no rows found; check run dirs and history.csv files")


if __name__ == "__main__":
    main()