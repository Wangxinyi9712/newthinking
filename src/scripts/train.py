from __future__ import annotations

import os

if not os.environ.get("OMP_NUM_THREADS", "").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.datasets import build_dataloaders
from engine.trainer import MeanTeacherTrainer
from models.seg_model import HybridUNet
from utils.config import load_config
from utils.seed import set_seed


def build_model(cfg: dict) -> HybridUNet:
    return HybridUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=tuple(cfg["model"].get("channels", [32, 64, 128, 256])),
        dim=3 if cfg["data"].get("dim", "3d") == "3d" else 2,
        use_transformer=cfg["model"].get("use_transformer", True),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg_obj = load_config(args.config)
    cfg = {
        "data": cfg_obj.data,
        "model": cfg_obj.model,
        "train": cfg_obj.train,
        "loss": cfg_obj.loss,
        "inference": cfg_obj.inference,
        "log": cfg_obj.log,
    }

    # 关键调试输出：确认到底加载了哪份配置
    print(f"[DEBUG] config_path={args.config}")
    print(f"[DEBUG] data_cfg={cfg['data']}")

    seeds = cfg["train"].get("seed", [0])
    if not isinstance(seeds, list):
        seeds = [seeds]

    out_dir = Path(cfg["log"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    repeats = []

    for seed in seeds:
        set_seed(int(seed))
        run_dir = out_dir / f"seed_{seed}"
        run_cfg = {**cfg, "log": {**cfg["log"], "out_dir": str(run_dir)}}

        loaders = build_dataloaders(run_cfg["data"])
        model = build_model(run_cfg)
        trainer = MeanTeacherTrainer(model, run_cfg)
        trainer.fit(loaders, str(run_dir))
        m = trainer.evaluate(
            loaders["val"],
            use_teacher_ema=bool(cfg["inference"].get("use_teacher_ema", False)),
        )
        repeats.append({"seed": seed, **m.__dict__})

        with open(run_dir / "config_used.json", "w", encoding="utf-8") as f:
            json.dump(run_cfg, f, indent=2)

    rep_file = out_dir / "repeats_summary.csv"
    with open(rep_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "dice", "iou", "precision", "recall", "f1", "minority_f1", "hd95"],
        )
        writer.writeheader()
        writer.writerows(repeats)

    agg = {}
    for k in ["dice", "iou", "precision", "recall", "f1", "minority_f1", "hd95"]:
        vals = [r[k] for r in repeats]
        agg[f"{k}_mean"] = statistics.mean(vals)
        agg[f"{k}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0

    with open(out_dir / "repeats_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)


if __name__ == "__main__":
    main()