from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.datasets import build_dataloaders
from engine.trainer import MeanTeacherTrainer
from models.seg_model import HybridUNet
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--source", choices=["teacher", "student", "ensemble"], default="teacher")
    args = parser.parse_args()

    cfg_obj = load_config(args.config)
    cfg = {k: getattr(cfg_obj, k) for k in ["data", "model", "train", "loss", "inference", "log"]}
    loaders = build_dataloaders(cfg["data"])

    model = HybridUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        channels=tuple(cfg["model"].get("channels", [32, 64, 128, 256])),
        dim=3 if cfg["data"].get("dim", "3d") == "3d" else 2,
        use_transformer=cfg["model"].get("use_transformer", True),
    )
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["student"], strict=True)

    trainer = MeanTeacherTrainer(model, cfg)
    trainer.teacher.load_state_dict(state["teacher"], strict=True)

    metrics = trainer.evaluate(
        loaders["val"],
        use_teacher_ema=(args.source == "teacher"),
        use_ensemble=(args.source == "ensemble"),
    )

    out = Path(cfg["log"]["out_dir"]) / "eval_metrics.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics.__dict__, f, indent=2)


if __name__ == "__main__":
    main()