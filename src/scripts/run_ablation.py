from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.datasets import build_dataloaders


DEFAULT_EXPERIMENTS = ["a", "b", "c", "d", "e", "r1", "r2", "r3", "r4"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="src/configs")
    parser.add_argument("--prefix", default="brats_group_")
    parser.add_argument("--metric-csv", default="runs/ablation_summary.csv")
    parser.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    args = parser.parse_args()

    # 训练各组
    for g in args.experiments:
        cfg = Path(args.config_dir) / f"{args.prefix}{g}.yaml"
        if not cfg.exists():
            raise FileNotFoundError(f"Config not found: {cfg}")
        print(f"[run_ablation] training group={g} config={cfg}")
        subprocess.run(["python", "src/scripts/train.py", "--config", str(cfg)], check=True)

    # 收集最终指标（按 seed）
    run_dirs = [f"runs/{args.prefix}{g}" for g in args.experiments]
    print(f"[run_ablation] collecting metrics from: {run_dirs}")
    subprocess.run(
        ["python", "src/scripts/collect_epoch_metrics.py", "--run-dirs", *run_dirs, "--out", args.metric_csv],
        check=True,
    )

    # 统计分析（默认 dice）
    print(f"[run_ablation] running stats on: {args.metric_csv}")
    subprocess.run(["python", "src/scripts/analyze_stats.py", "--csv", args.metric_csv], check=True)
    print("[run_ablation] done.")


if __name__ == "__main__":
    main()