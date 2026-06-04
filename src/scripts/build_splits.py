from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.examples import (
    build_brats_split,
    build_isic_split,
    build_msd_liver_split,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["brats", "isic", "msd_liver"], required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)

    # 新增：可控划分参数
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--labeled-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()

    if args.dataset == "brats":
        build_brats_split(
            brats_root=args.root,
            out_json=args.out,
            val_ratio=args.val_ratio,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
        )
    elif args.dataset == "isic":
        build_isic_split(
            isic_root=args.root,
            out_json=args.out,
            val_ratio=args.val_ratio,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
        )
    else:
        build_msd_liver_split(
            msd_liver_root=args.root,
            out_json=args.out,
            val_ratio=args.val_ratio,
            labeled_ratio=args.labeled_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()